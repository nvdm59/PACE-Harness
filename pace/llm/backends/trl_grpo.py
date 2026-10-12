"""TRL arm of the phase-1b reproduction, plus the shared evaluator for every stack.

  DynamicSamplingGRPOTrainer   TRL GRPOTrainer with DAPO-style dynamic sampling added by hand:
                               after each generation, groups whose rewards are all equal (zero GRPO
                               advantage) are dropped and replaced by groups of freshly drawn prompts,
                               up to `max_gen_batches` generations per step. If the batch is still
                               short, it is topped up with dropped groups (and the shortfall is logged).
  train (Modal)                phase-1b settings with the price FIXED in prompt and reward:
                               no SFT, LoRA r16/a32, 4 prompts x 16 completions, T 1.0, beta 0.05,
                               entropy 0.01, lr 1e-5, max 16 tokens; `--dynamic` + `--clip-high 0.28`
                               turn on the recipe. Same stack as the TRL phase-1 runs (modal_common.image).
  evaluate (Modal)             one evaluator for all stacks: base + PEFT adapter, HF generate (the
                               phase-1 eval engine), greedy and sampled T=1, on the 2,000 seed-777 tasks
                               with the fixed-price prompt; re-scored with common.eval_stats.

    modal run pace/llm/backends/trl_grpo.py --run trl_dapo_1b --dynamic --clip-high 0.28 --steps 200
    modal run pace/llm/backends/trl_grpo.py::eval_main --adapters trl_dapo_1b=/vol/...,verl_dapo_1b=/vol/...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if "/root/src" not in sys.path and os.path.isdir("/root/src"):
    sys.path.insert(0, "/root/src")
elif len(HERE.parents) > 2 and str(HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[2]))

PHASE1B_LAM = 0.44
VOL_PATH = "/vol"
RUN_ROOT = f"{VOL_PATH}/backends/trl"


# ------------------------------------------------------------------ dynamic sampling helpers (no torch at import)
def group_is_constant(rewards: list[float], tol: float = 1e-9) -> bool:
    return max(rewards) - min(rewards) <= tol


def split_groups(rewards: list[float], g: int) -> list[bool]:
    """Per group of g consecutive completions: True when all rewards are equal (zero advantage)."""
    return [group_is_constant(rewards[i:i + g]) for i in range(0, len(rewards), g)]


def concat_batches(parts: list[dict], keep: list[list[int]], g: int) -> dict:
    """Concatenate the kept groups (row indices = group * g ... ) of several TRL generation outputs.
    Prompt tensors are left-padded, completion-aligned tensors right-padded to a common width."""
    import torch
    out: dict = {}
    rows = [[r for grp in k for r in range(grp * g, grp * g + g)] for k in keep]
    keys = [k for k, v in parts[0].items() if isinstance(v, torch.Tensor) and k != "num_items_in_batch"]
    for key in keys:
        ts = [p[key][idx] for p, idx in zip(parts, rows) if idx]
        if ts[0].dim() == 1:
            out[key] = torch.cat(ts)
            continue
        width = max(t.shape[1] for t in ts)
        padded = []
        for t in ts:
            pad = width - t.shape[1]
            if pad:
                fill = torch.zeros((t.shape[0], pad), dtype=t.dtype, device=t.device)
                t = torch.cat([fill, t], 1) if key.startswith("prompt") else torch.cat([t, fill], 1)
            padded.append(t)
        out[key] = torch.cat(padded)
    out["num_items_in_batch"] = out["completion_mask"].sum()
    return out


# ------------------------------------------------------------------ Modal (lazy)
try:
    import modal  # noqa: F401
except ImportError:
    modal = None

if modal is not None:
    from pace.llm.modal_common import image as _trl_image

    volume = modal.Volume.from_name("pace-llm", create_if_missing=True)
    app = modal.App("pace-llm-trl-backend", image=_trl_image)

    @app.function(gpu="A100-40GB", timeout=2400, volumes={VOL_PATH: volume})
    def train(run: str = "trl_dapo_1b", steps: int = 200, dynamic: bool = True, clip_high: float = 0.28,
              num_generations: int = 16, batch: int = 64, micro: int = 16, beta: float = 0.05,
              entropy_coef: float = 0.01, temperature: float = 1.0, lam: float = PHASE1B_LAM,
              max_gen_batches: int = 8, seed: int = 0, no_step_kill_s: int = 900) -> dict:
        import random

        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer

        from pace.llm.backends.common import score, step_stats, task_rows
        from pace.llm.modal_common import LORA_TARGETS, load_model, render

        t0 = time.time()
        out = Path(RUN_ROOT) / run
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "train_log.jsonl"
        log_path.write_text("")
        model, tok, name, _ = load_model("Qwen/Qwen3-1.7B")
        rows = task_rows(lam=lam)
        ds = Dataset.from_list([{"prompt": render(tok, r["user"]), "task": json.dumps(r["task"], sort_keys=True)}
                                for r in rows])
        g = num_generations

        class Reward:
            __name__ = "priced_reward_fixed"

            def __init__(self):
                self.last: list[dict] = []

            def __call__(self, prompts=None, completions=None, task=None, **kw):
                self.last = []
                for c, t in zip(completions, task):
                    o = score(t, c)
                    self.last.append({"task": t, "completion": c, "reward": o.reward})
                return [r["reward"] for r in self.last]

        reward = Reward()
        stats = {"pre": [], "post": [], "gens": [], "short": [], "kept": []}
        rng = random.Random(seed + 1)

        class DSTrainer(GRPOTrainer):
            def _generate_and_score_completions(self, inputs):
                first = super()._generate_and_score_completions(inputs)
                recs = list(reward.last)
                const = split_groups([r["reward"] for r in recs], g)
                stats["pre"].append(sum(const) / len(const))
                if not dynamic:
                    stats["post"].append(stats["pre"][-1]); stats["gens"].append(1); stats["short"].append(0)
                    stats["kept"].append(recs)
                    return first
                n_groups = len(const)
                parts, keep, dropped, kept_recs = [first], [[i for i, c in enumerate(const) if not c]], [], []
                dropped.append([i for i, c in enumerate(const) if c])
                kept_recs += [r for i in keep[0] for r in recs[i * g:(i + 1) * g]]
                have, gens = len(keep[0]), 1
                while have < n_groups and gens < max_gen_batches:
                    idx = [rng.randrange(len(ds)) for _ in range(n_groups)]
                    fresh = [dict(ds[i]) for i in idx for _ in range(g)]
                    part = super()._generate_and_score_completions(fresh)
                    gens += 1
                    recs2 = list(reward.last)
                    c2 = split_groups([r["reward"] for r in recs2], g)
                    k2 = [i for i, c in enumerate(c2) if not c][: n_groups - have]
                    parts.append(part); keep.append(k2); dropped.append([i for i, c in enumerate(c2) if c])
                    kept_recs += [r for i in k2 for r in recs2[i * g:(i + 1) * g]]
                    have += len(k2)
                short = n_groups - have
                if short:   # top up with dropped (zero-advantage) groups of the first draw
                    fill = dropped[0][:short]
                    keep[0] = keep[0] + fill
                    kept_recs += [r for i in fill for r in recs[i * g:(i + 1) * g]]
                stats["post"].append(short / n_groups); stats["gens"].append(gens); stats["short"].append(short)
                stats["kept"].append(kept_recs)
                return concat_batches(parts, keep, g)

        trainer_ref = {}

        class Log(TrainerCallback):
            def __init__(self):
                self.last_t = time.time()
                self.entropy = None

            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "entropy" in logs:
                    self.entropy = logs["entropy"]

            def on_step_end(self, args, state, control, **kw):
                now = time.time()
                kept = stats["kept"][-1] if stats["kept"] else []
                st = step_stats({"task": r["task"], "completion": r["completion"]} for r in kept)
                rec = {"step": state.global_step, **st,
                       "identical_reward_group_rate_pre": round(stats["pre"][-1], 4) if stats["pre"] else None,
                       "identical_reward_group_rate_post": round(stats["post"][-1], 4) if stats["post"] else None,
                       "gen_batches": stats["gens"][-1] if stats["gens"] else None,
                       "entropy": self.entropy, "step_s": round(now - self.last_t, 2),
                       "elapsed_s": round(now - t0, 1), "lambda": lam}
                self.last_t = now
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if state.global_step % 10 == 0 or state.global_step == 1:
                    print(f"[trl step {rec['step']}] r={rec.get('mean_reward')} q={rec.get('mean_quality')} "
                          f"c={rec.get('mean_cost')} pre={rec['identical_reward_group_rate_pre']} "
                          f"post={rec['identical_reward_group_rate_post']} gens={rec['gen_batches']} "
                          f"ent={self.entropy} {rec.get('action_histogram')}", flush=True)
                if state.global_step % 50 == 0:
                    trainer_ref["t"].model.save_pretrained(str(out / "adapter_last"))   # survives a time-out
                    (out / "adapter_last" / "step.txt").write_text(str(state.global_step))
                    volume.commit()

        cfg = GRPOConfig(
            output_dir=str(out / "trainer"), max_steps=steps, per_device_train_batch_size=micro,
            gradient_accumulation_steps=batch // micro, num_generations=g, max_completion_length=16,
            learning_rate=1e-5, beta=beta, bf16=True, gradient_checkpointing=False, temperature=temperature,
            entropy_coef=entropy_coef, epsilon=0.2, epsilon_high=clip_high, logging_steps=1, save_strategy="no",
            report_to=[], seed=seed)
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                              task_type="CAUSAL_LM")
        trainer = DSTrainer(model=model, reward_funcs=[reward], args=cfg, train_dataset=ds, processing_class=tok,
                            peft_config=peft_cfg, callbacks=[Log()])
        trainer_ref["t"] = trainer
        t_train = time.time()
        trainer.train()
        train_s = time.time() - t_train
        trainer.model.save_pretrained(str(out / "adapter_final"))
        meta = {"run": run, "steps": steps, "dynamic": dynamic, "clip": [0.2, clip_high], "num_generations": g,
                "batch_completions": batch, "beta": beta, "entropy_coef": entropy_coef, "temperature": temperature,
                "lambda": lam, "max_gen_batches": max_gen_batches, "loss_type": cfg.loss_type,
                "train_wall_s": round(train_s, 1), "container_s": round(time.time() - t0, 1),
                "gpu": torch.cuda.get_device_name(0), "adapter": str(out / "adapter_final")}
        (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
        volume.commit()
        return {"meta": meta, "lines": [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]}

    @app.function(gpu="A100-40GB", timeout=1800, volumes={VOL_PATH: volume})
    def evaluate(adapters: dict, lam: float = PHASE1B_LAM, n: int = 2000, seed: int = 777,
                 include_base: bool = True) -> dict:
        import torch
        from peft import PeftModel

        from pace.llm.backends.common import eval_stats, task_rows
        from pace.llm.modal_common import generate, load_model, render

        t0 = time.time()
        volume.reload()
        rows = task_rows(n, seed, lam)
        tasks = [r["task"] for r in rows]
        base, tok, _, _ = load_model("Qwen/Qwen3-1.7B")
        base.to("cuda").eval()
        prompts = [render(tok, r["user"]) for r in rows]
        res = {}
        todo = ([("base zero-shot", None)] if include_base else []) + list(adapters.items())
        for name, path in todo:
            if path is None:
                m = base
            else:
                m = PeftModel.from_pretrained(base, path).to("cuda").eval()
                n_lora = sum(1 for k, _ in m.named_parameters() if "lora_" in k)
            greedy = generate(m, tok, prompts, greedy=True)
            sampled = generate(m, tok, prompts, greedy=False, seed=seed, temperature=1.0)
            res[name] = {"adapter": path, "greedy": eval_stats(tasks, greedy), "sampled": eval_stats(tasks, sampled),
                         "lora_tensors": None if path is None else n_lora,
                         "examples": sorted(set(greedy))[:6]}
            print(f"[eval] {name}: greedy {res[name]['greedy']['quality']}/{res[name]['greedy']['cost']} "
                  f"sampled {res[name]['sampled']['quality']}/{res[name]['sampled']['cost']}", flush=True)
            if path is not None:
                base = m.unload()          # back to the plain base model for the next adapter
        return {"results": res, "eval_s": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
                "n": n, "seed": seed, "lambda": lam}

    @app.local_entrypoint()
    def main(run: str = "trl_dapo_1b", steps: int = 200, dynamic: bool = True, clip_high: float = 0.28):
        out = Path(__file__).resolve().parents[3] / "results" / "llm" / "backends"
        out.mkdir(parents=True, exist_ok=True)
        res = train.remote(run=run, steps=steps, dynamic=dynamic, clip_high=clip_high)
        (out / f"{run}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res["lines"]))
        (out / f"{run}_meta.json").write_text(json.dumps(res["meta"], indent=1) + "\n")
        print(json.dumps(res["meta"], indent=1))

    @app.local_entrypoint()
    def eval_main(adapters: str, out_name: str = "eval_phase1b_repro", include_base: bool = True):
        out = Path(__file__).resolve().parents[3] / "results" / "llm" / "backends"
        ad = dict(kv.split("=", 1) for kv in adapters.split(",") if kv)
        res = evaluate.remote(ad, include_base=include_base)
        (out / f"{out_name}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: {m: (v[m]["quality"], v[m]["cost"], v[m]["invalid_rate"]) for m in ("greedy", "sampled")}
                          for k, v in res["results"].items()}, indent=1))
