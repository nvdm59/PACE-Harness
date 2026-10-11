"""GRPO-train the PACE controller as a small open LLM (LoRA) on Modal.

The LLM picks the first (tier, op) of an episode; the frozen harness and a
fixed continuation finish it; the reward is quality - lambda * cost with
lambda learned by dual ascent toward the budget (pace.llm.rewards).

  modal run pace/llm/train_grpo_modal.py --steps 300 --budget 0.30
  # phase 2: lookup rewards from a CRMArena counterfactual log
  modal run pace/llm/train_grpo_modal.py --env lookup \
      --log-file traces/crmarena_b2b_counterfactual.jsonl --budget 0.10 --run crm_b2b
  # phase 2 on CRMArena, fold 0 (SFT + GRPO + eval in one A100-40GB container; needs
  # results/crmarena/fold0_for_llm.json from scripts/crmarena_offline.py)
  modal run pace/llm/train_grpo_modal.py --env crm_phase2

Outputs on the Modal Volume `pace-llm` under <run>/:
  train_log.jsonl   one line per step: step, mean_reward, mean_quality, mean_cost,
                    lambda, action_histogram, invalid_rate (+ cost_ema, lambda_used)
  run_meta.json     model actually loaded, library versions, GPU, config, wall clock
  adapter_step100/, adapter_step200/, ..., adapter_final/   LoRA adapters
This file only imports `modal` at top level; everything else runs in the container.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

if modal.is_local():                    # the container gets `pace` on PYTHONPATH from the image
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pace.llm.modal_common import (DEFAULT_MODEL, GPUS, LORA_TARGETS, VOL_PATH, generate, image,  # noqa: E402
                                   load_model, render, run_sft, volume)

app = modal.App("pace-llm-grpo", image=image)


def sft_stage(model, tok, out: Path, budget: float, steps: int, lr: float, n: int) -> dict:
    """SFT warm start on per-difficulty oracle labels at a shown price; verify agreement on
    held-out prompts and report the greedy action split by difficulty at fixed prices."""
    from collections import Counter

    from pace.llm.env_text import parse_action, sim_rows, with_price
    from pace.llm.sft import LAM_GRID, oracle_action, sft_rows

    t0 = time.time()
    train_rows = sft_rows(n, seed=101, budget=budget)
    pm, hist = run_sft(model.to("cuda"), tok, train_rows, steps=steps, lr=lr,
                       log=lambda m: print(m, flush=True))
    t_train = time.time() - t0
    # held-out agreement (fresh tasks, fresh shuffles, prices from the same grid)
    ev = sft_rows(1000, seed=202, budget=budget)
    outs = generate(pm, tok, [render(tok, r["user"]) for r in ev], greedy=True)
    agree = [parse_action(o) == parse_action(r["completion"]) for o, r in zip(outs, ev)]
    invalid = sum(parse_action(o) is None for o in outs) / len(outs)
    by_lam = {}
    for r, a in zip(ev, agree):
        by_lam.setdefault(r["lam"], []).append(a)
    # greedy action split by difficulty at fixed shown prices
    probe = sim_rows(200, seed=303, budget=budget, price=True)
    split = {}
    for lam in (0.3, 0.5, 1.2):
        o = generate(pm, tok, [render(tok, with_price(r["user"], lam)) for r in probe], greedy=True)
        for proxy in ("easy", "hard"):
            acts = [(" ".join(parse_action(x)) if parse_action(x) else "INVALID")
                    for x, r in zip(o, probe) if json.loads(r["task"])["proxy"] == proxy]
            split[f"lambda={lam} {proxy}"] = {"n": len(acts), "oracle": " ".join(oracle_action(proxy, lam)),
                                             "greedy": dict(Counter(acts).most_common())}
    rep = {"steps": steps, "lr": lr, "n_train": len(train_rows), "lam_grid": LAM_GRID,
           "loss_first": hist[0], "loss_last25": sum(hist[-25:]) / len(hist[-25:]),
           "heldout_n": len(ev), "heldout_agreement": sum(agree) / len(agree), "heldout_invalid": invalid,
           "agreement_by_lambda": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(by_lam.items())},
           "greedy_split_by_difficulty": split, "sft_train_s": round(t_train, 1),
           "sft_total_s": round(time.time() - t0, 1)}
    pm.save_pretrained(str(out / "adapter_sft"))
    tok.save_pretrained(str(out / "adapter_sft"))
    (out / "sft_report.json").write_text(json.dumps(rep, indent=1))
    print("[sft] " + json.dumps({k: rep[k] for k in ("heldout_agreement", "heldout_invalid", "sft_total_s")}),
          flush=True)
    print("[sft] split " + json.dumps(split), flush=True)
    if rep["heldout_agreement"] < 0.95:
        print(f"[sft] WARNING: held-out agreement {rep['heldout_agreement']:.3f} < 0.95", flush=True)
    rep["_model"] = pm.merge_and_unload()
    return rep


def run_grpo(model, tok, ds, reward_fn, lam, out: Path, meta: dict, log_path: Path, t_start: float, *, steps: int,
             batch: int, micro: int, num_generations: int, beta: float, entropy_coef: float, temperature: float,
             temp_escalate: float, degen_threshold: float, degen_patience: int, seed: int):
    """The GRPO stage shared by `train` and `crm_phase2` (runs inside the container): TRL GRPOTrainer
    with a fresh LoRA, the priced reward, a per-step callback that drains the reward buffer, updates
    lambda, logs to `log_path`, raises the temperature once on persistent group degeneracy, and saves
    adapters every 100 steps plus adapter_final. Returns the trainer."""
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    def save_adapter(tag: str):
        trainer.model.save_pretrained(str(out / tag))
        tok.save_pretrained(str(out / tag))

    esc = {"run": 0, "done": False}

    def raise_temperature(t_new: float, step: int):
        """Raise the rollout temperature mid-run, consistently: TRL applies it both in the
        GenerationConfig and inside the patched (fused) LM head that scores log-probs."""
        from trl.trainer.utils import add_fused_lm_head
        from peft import PeftModel
        t_old = trainer.temperature
        trainer.temperature = t_new
        trainer.generation_config.temperature = t_new
        m = trainer.model
        add_fused_lm_head(m.get_base_model() if isinstance(m, PeftModel) else m, temperature=t_new,
                          cast_lm_head_to_fp32=cfg.cast_lm_head_to_fp32, outputs=("log_probs", "entropy"))
        ev = {"step": step, "event": "temperature_raised", "from": t_old, "to": t_new,
              "reason": f"degenerate_group_rate > {degen_threshold} for {degen_patience} consecutive steps"}
        meta["events"].append(ev)
        print(f"[event] {ev}", flush=True)
        return ev

    class LambdaLog(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            st = reward_fn.drain()
            event = None
            if st is not None and temp_escalate > 0 and not esc["done"]:
                dg = st["degenerate_group_rate"]
                esc["run"] = esc["run"] + 1 if (dg is not None and dg > degen_threshold) else 0
                if esc["run"] >= degen_patience:
                    esc["done"] = True
                    event = raise_temperature(temp_escalate, state.global_step)
            if st is not None:
                rec = {"step": state.global_step, "mean_reward": round(st["mean_reward"], 5),
                       "mean_quality": round(st["mean_quality"], 5), "mean_cost": round(st["mean_cost"], 5),
                       "lambda": round(st["lambda"], 5), "lambda_used": round(st["lambda_used"], 5),
                       "cost_ema": round(st["cost_ema"], 5), "action_histogram": st["action_histogram"],
                       "invalid_rate": round(st["invalid_rate"], 4), "n": st["n"],
                       "degenerate_group_rate": (None if st["degenerate_group_rate"] is None
                                                 else round(st["degenerate_group_rate"], 4)),
                       "lambda_shown": st.get("lambda_shown"), "temperature": trainer.temperature,
                       "elapsed_s": round(time.time() - t_start, 1)}
                if event:
                    rec["event"] = event
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if state.global_step % 10 == 0 or state.global_step == 1:
                    print(f"[step {rec['step']}] r={rec['mean_reward']:.3f} q={rec['mean_quality']:.3f} "
                          f"c={rec['mean_cost']:.3f} lam={rec['lambda']:.3f} inv={rec['invalid_rate']:.2f} "
                          f"degen={rec['degenerate_group_rate']} "
                          f"{rec['action_histogram']}", flush=True)
            if state.global_step % 100 == 0:
                save_adapter(f"adapter_step{state.global_step}")
                meta["lambda"] = lam.state_dict()
                meta["last_saved_step"] = state.global_step
                (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
                volume.commit()
            elif state.global_step % 20 == 0:
                volume.commit()

    cfg = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=steps, per_device_train_batch_size=micro,
        # one 64-completion generation (8 prompts x 8) per optimizer step, in micro-batches of 16
        gradient_accumulation_steps=batch // micro, num_generations=num_generations, max_completion_length=16,
        learning_rate=1e-5, beta=beta, bf16=True, gradient_checkpointing=False,
        temperature=temperature, entropy_coef=entropy_coef, logging_steps=10, save_strategy="no", report_to=[], seed=seed,
    )
    assert (cfg.max_completion_length, cfg.num_generations, cfg.beta) == (16, num_generations, beta)
    meta["trl_generation_batch_size"] = cfg.generation_batch_size
    meta["trl_steps_per_generation"] = cfg.steps_per_generation
    meta["trl_loss_type"] = cfg.loss_type
    meta["trl_scale_rewards"] = cfg.scale_rewards
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                          task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn], args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg, callbacks=[LambdaLog()])
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    t_train = time.time()
    trainer.train()
    save_adapter("adapter_final")
    meta["train_wall_s"] = round(time.time() - t_train, 1)
    return trainer


@app.function(gpu=GPUS, timeout=3600, volumes={VOL_PATH: volume})
def train(steps: int = 300, budget: float = 0.30, model_name: str = DEFAULT_MODEL, env: str = "sim",
          log_lines: list | None = None, run: str = "phase1", n_tasks: int = 2000, seed: int = 0,
          batch: int = 64, micro: int = 16, lam_lr: float = 0.10, lam_kp: float = 2.0, lam_ema: float = 0.9,
          beta: float = 0.02, entropy_coef: float = 0.0, temperature: float = 1.0, num_generations: int = 8,
          price_in_prompt: bool = False, lam0: float = 0.0, sft_steps: int = 0, sft_lr: float = 2e-4,
          sft_n: int = 2000, temp_escalate: float = 0.0, degen_threshold: float = 0.6,
          degen_patience: int = 20) -> dict:
    import torch
    import transformers, trl, peft, accelerate, datasets  # noqa: E401
    from datasets import Dataset

    from pace.llm.env_text import round_lam, sim_rows, with_price
    from pace.llm.rewards import DualAscentLambda, LookupReward, PricedRewardFn, lookup_rows

    t_start = time.time()
    out = Path(VOL_PATH) / run
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_path.write_text("")

    model, tok, loaded, load_note = load_model(model_name)
    sft_report = None
    if sft_steps > 0:
        if env != "sim" or not price_in_prompt:
            raise ValueError("the SFT warm start is built for --env sim with --price-in-prompt")
        sft_report = sft_stage(model, tok, out, budget, sft_steps, sft_lr, sft_n)
        model = sft_report.pop("_model")          # SFT LoRA merged: GRPO's KL reference is the SFT policy
        volume.commit()
    lookup = None
    if env == "sim":
        rows = sim_rows(n_tasks, seed=seed, budget=budget, price=price_in_prompt)
    elif env == "lookup":
        if not log_lines:
            raise ValueError("--env lookup needs --log-file")
        lookup = LookupReward(log_lines)
        rows = lookup_rows(lookup, budget)
        print(f"[data] lookup log: {len(lookup.table)} (task, tier) cells, {len(rows)} complete tasks", flush=True)
    else:
        raise ValueError(env)
    lam = DualAscentLambda(budget, lr=lam_lr, kp=lam_kp, ema=lam_ema, lam0=lam0)
    if price_in_prompt:
        # the prompt shows the CURRENT learned lambda (rounded to 0.1), filled in when the batch is drawn;
        # the shown value travels in the task column and is the price the reward charges
        ds = Dataset.from_list([{"user": r["user"], "task": r["task"]} for r in rows])

        def fill(batch):
            shown = round_lam(lam.lam)
            return {"prompt": [render(tok, with_price(u, shown)) for u in batch["user"]],
                    "task": [json.dumps({**json.loads(t), "lam": shown}, sort_keys=True) for t in batch["task"]]}
        ds.set_transform(fill)
    else:
        ds = Dataset.from_list([{"prompt": render(tok, r["user"]), "task": r["task"]} for r in rows])
    print(f"[data] {len(ds)} prompts; example:\n{ds[0]['prompt']}", flush=True)

    reward_fn = PricedRewardFn(lam, lookup=lookup)

    meta = {"run": run, "env": env, "model_requested": model_name, "model_loaded": loaded,
            "model_load_note": load_note, "gpu": torch.cuda.get_device_name(0),
            "versions": {m.__name__: m.__version__ for m in (torch, transformers, trl, peft, accelerate, datasets)},
            "steps": steps, "budget": budget, "n_tasks": len(ds), "seed": seed, "batch_completions": batch, "micro_batch": micro,
            "num_generations": num_generations, "max_completion_length": 16,
            "price_in_prompt": price_in_prompt, "lam0": lam0, "sft": sft_report, "temp_escalate": temp_escalate,
            "degen_threshold": degen_threshold, "degen_patience": degen_patience, "events": [], "lr": 1e-5, "beta": beta, "entropy_coef": entropy_coef, "temperature": temperature,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS},
            "lambda": lam.state_dict()}

    trainer = run_grpo(model, tok, ds, reward_fn, lam, out, meta, log_path, t_start, steps=steps, batch=batch,
                       micro=micro, num_generations=num_generations, beta=beta, entropy_coef=entropy_coef,
                       temperature=temperature, temp_escalate=temp_escalate, degen_threshold=degen_threshold,
                       degen_patience=degen_patience, seed=seed)
    meta.update({"lambda": lam.state_dict(), "lambda_final": lam.lam, "last_saved_step": steps,
                 "total_wall_s": round(time.time() - t_start, 1)})
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    print(f"[done] {json.dumps({k: meta[k] for k in ('model_loaded', 'gpu', 'train_wall_s', 'total_wall_s', 'lambda_final')})}",
          flush=True)
    return meta


@app.function(gpu="A100-40GB", timeout=1740, volumes={VOL_PATH: volume})
def crm_phase2(log_lines: list, train_ids: list, test_ids: list, linear_probs: dict, run: str = "crm_phase2",
               model_name: str = DEFAULT_MODEL, budget: float = 0.10, tol: float = 0.005, sft_steps: int = 150,
               sft_lr: float = 2e-4, steps: int = 100, num_generations: int = 16, temperature: float = 1.2,
               temp_escalate: float = 1.5, lam0: float = 1.66, lam_lr: float = 2.0, lam_kp: float = 5.0,
               lam_ema: float = 0.95, beta: float = 0.05, entropy_coef: float = 0.01, n_samples: int = 16,
               lam_window: int = 50, seed: int = 0) -> dict:
    """Phase 2 in ONE container (pinned to one A100-40GB, hard timeout 29 min):
    zero-shot decode -> SFT on per-category oracle labels (fold-0 training tasks, lam grid 0..20/USD)
    -> GRPO lookup mode (shown lam = learned lam rounded to 1/USD, slow dual step toward the budget)
    -> decode SFT-only and GRPO-final, greedy (every grid price) and sampled (n_samples per prompt at
    the run's lam_bar) -> exact-lookup eval on the held-out and the training tasks."""
    import torch
    import transformers, trl, peft, accelerate, datasets  # noqa: E401
    from datasets import Dataset
    from peft import PeftModel

    from pace.llm.crm import (LAM_GRID_USD, LAM_STEP_USD, eval_tables, oracle_tier, parsed_tier, price_rows,
                              sft_rows_crm)
    from pace.llm.env_text import round_lam, with_price
    from pace.llm.rewards import DualAscentLambda, LookupReward, PricedRewardFn

    t_start = time.time()
    out = Path(VOL_PATH) / run
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_path.write_text("")
    train_ids, test_ids = list(train_ids), list(test_ids)
    assert not set(train_ids) & set(test_ids)
    lk_all = LookupReward(log_lines)
    keep = set(train_ids)
    lk_train = LookupReward([l for l in log_lines if l.strip() and json.loads(l)["task_id"] in keep])
    assert lk_train.complete_tasks() == sorted(train_ids), "training log must hold all three arms of every task"
    all_ids = sorted(train_ids) + sorted(test_ids)
    eval_rows = {r_id: r for r_id, r in zip(all_ids, price_rows(lk_all, all_ids, budget, train_ids, shuffle_seed=7))}
    timing = {}
    comps: dict = {}

    def decode(m, tok, tag: str, lams, greedy: bool, n: int = 1):
        t = time.time()
        keys = [(i, lam) for lam in lams for i in all_ids]
        prompts = [render(tok, with_price(eval_rows[i]["user"], lam, LAM_STEP_USD)) for i, lam in keys]
        if greedy:
            uniq = sorted(set(prompts))
            dec = dict(zip(uniq, generate(m, tok, uniq, greedy=True)))
            outs = [[dec[p]] for p in prompts]
        else:
            flat = generate(m, tok, [p for p in prompts for _ in range(n)], greedy=False, seed=seed)
            outs = [flat[k * n:(k + 1) * n] for k in range(len(prompts))]
        for (i, lam), o in zip(keys, outs):
            comps.setdefault(tag, {}).setdefault(str(lam), {})[i] = o
        timing[f"decode {tag}"] = round(time.time() - t, 1)
        print(f"[decode] {tag}: {len(prompts)} prompts x {n} in {timing[f'decode {tag}']}s", flush=True)

    # ------------------------------------------------------------ zero-shot
    model, tok, loaded, load_note = load_model(model_name)
    model = model.to("cuda").eval()
    decode(model, tok, "zero-shot (greedy)", LAM_GRID_USD, True)

    # ------------------------------------------------------------------ SFT
    t = time.time()
    rows = sft_rows_crm(lk_all, train_ids, budget)
    pm, hist = run_sft(model, tok, rows, steps=sft_steps, lr=sft_lr, log=lambda m_: print(m_, flush=True))
    timing["sft_train"] = round(time.time() - t, 1)
    decode(pm, tok, "SFT (greedy)", LAM_GRID_USD, True)
    agree = {}
    for split, ids in (("train", train_ids), ("held-out", test_ids)):
        hits = [parsed_tier(comps["SFT (greedy)"][str(lam)][i][0])
                == oracle_tier(lk_all, train_ids, lk_all.features[i]["category"], lam)
                for lam in LAM_GRID_USD for i in ids]
        agree[split] = sum(hits) / len(hits)
    sft_report = {"steps": sft_steps, "lr": sft_lr, "n_train_pairs": len(rows), "lam_grid": LAM_GRID_USD,
                  "loss_first": hist[0], "loss_last25": sum(hist[-25:]) / len(hist[-25:]),
                  "greedy_agreement_with_category_oracle": agree, "sft_train_s": timing["sft_train"]}
    print(f"[sft] {json.dumps(sft_report)}", flush=True)
    pm.save_pretrained(str(out / "adapter_sft"))
    tok.save_pretrained(str(out / "adapter_sft"))
    (out / "sft_report.json").write_text(json.dumps(sft_report, indent=1))
    model = pm.merge_and_unload()
    volume.commit()

    # ----------------------------------------------------------------- GRPO
    lam = DualAscentLambda(budget, lr=lam_lr, kp=lam_kp, ema=lam_ema, lam0=lam0)
    grows = price_rows(lk_train, sorted(train_ids), budget, train_ids, shuffle_seed=seed)
    ds = Dataset.from_list([{"user": r["user"], "task": r["task"]} for r in grows])

    def fill(batch):
        shown = round_lam(lam.lam, LAM_STEP_USD)
        return {"prompt": [render(tok, with_price(u, shown, LAM_STEP_USD)) for u in batch["user"]],
                "task": [json.dumps({**json.loads(x), "lam": shown}, sort_keys=True) for x in batch["task"]]}
    ds.set_transform(fill)
    print(f"[data] {len(ds)} GRPO prompts; example:\n{ds[0]['prompt']}", flush=True)
    reward_fn = PricedRewardFn(lam, lookup=lk_train)
    meta = {"run": run, "env": "lookup", "model_requested": model_name, "model_loaded": loaded,
            "model_load_note": load_note, "gpu": torch.cuda.get_device_name(0),
            "versions": {m_.__name__: m_.__version__ for m_ in (torch, transformers, trl, peft, accelerate, datasets)},
            "steps": steps, "budget": budget, "n_tasks": len(ds), "train_ids": sorted(train_ids),
            "test_ids": sorted(test_ids), "seed": seed, "batch_completions": 64, "micro_batch": 16,
            "num_generations": num_generations, "max_completion_length": 16, "price_in_prompt": True,
            "lam_step": LAM_STEP_USD, "lam0": lam0, "sft": sft_report, "temp_escalate": temp_escalate,
            "degen_threshold": 0.6, "degen_patience": 20, "events": [], "lr": 1e-5, "beta": beta,
            "entropy_coef": entropy_coef, "temperature": temperature,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS}, "lambda": lam.state_dict()}
    trainer = run_grpo(model, tok, ds, reward_fn, lam, out, meta, log_path, t_start, steps=steps, batch=64,
                       micro=16, num_generations=num_generations, beta=beta, entropy_coef=entropy_coef,
                       temperature=temperature, temp_escalate=temp_escalate, degen_threshold=0.6,
                       degen_patience=20, seed=seed)
    del trainer, model
    torch.cuda.empty_cache()
    log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    tail = log[-lam_window:]
    lam_bar = sum(r["lambda"] for r in tail) / len(tail)
    shown = round_lam(lam_bar, LAM_STEP_USD)
    meta.update({"lambda": lam.state_dict(), "lambda_final": lam.lam, "lambda_bar": lam_bar, "lambda_window": len(tail),
                 "eval_shown_lambda": shown, "last_saved_step": steps})
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()

    # --------------------------------------------------------------- decode
    base, tok, _, _ = load_model(model_name)
    base = base.to("cuda").eval()
    m_sft = PeftModel.from_pretrained(base, str(out / "adapter_sft")).eval()
    decode(m_sft, tok, f"SFT (sampled T=1, {n_samples}/prompt)", [shown], False, n_samples)
    merged = m_sft.merge_and_unload()
    m_grpo = PeftModel.from_pretrained(merged, str(out / "adapter_final")).eval()
    decode(m_grpo, tok, f"GRPO step {steps} (greedy)", LAM_GRID_USD, True)
    decode(m_grpo, tok, f"GRPO step {steps} (sampled T=1, {n_samples}/prompt)", [shown], False, n_samples)
    (out / "eval_completions.json").write_text(json.dumps(comps))
    volume.commit()

    # ---------------------------------------------------------------- score
    at = {k: v[str(shown)] for k, v in comps.items() if str(shown) in v}     # rows decoded at the shown price
    hdr = (f"Run `{run}` (model {loaded}; SFT {sft_steps} steps on per-category oracle labels at λ ∈ 0..20/USD from "
           f"the 40 training tasks of fold 0, then GRPO lookup mode {steps} steps, {num_generations} generations, "
           f"temperature {temperature}" + (f" (events {meta['events']})" if meta["events"] else "")
           + f"; dual step lr {lam_lr}, kp {lam_kp}, ema {lam_ema}, λ0 {lam0}; β {beta}; entropy {entropy_coef}). "
           f"Budget B = ${budget:.2f}; 'within budget' = cost <= B + ${tol:.3f}. "
           f"λ̄ = {lam_bar:.3f}/USD (mean learned λ over the last {len(tail)} GRPO steps; final {lam.lam:.3f}); "
           f"the LLM prompts show λ={shown:g} (λ̄ rounded to the 1/USD grid) and J = quality − {shown:g}·cost. "
           "Quality and cost are exact lookups of the logged arm the first action picks. Sampled rows average "
           f"{n_samples} samples per task.")
    md, scored = eval_tables(lk_all, at, {"held-out (fold 0 test)": sorted(test_ids),
                                          "training (fold 0 train)": sorted(train_ids)},
                             train_ids, lam_bar, budget, tol, linear_probs, hdr)
    # greedy first action by shown price, for every decoded model
    md += "\nGreedy tier mix on the 10 held-out tasks by shown price (λ per USD):\n\n| model | " + " | ".join(
        f"λ={x:g}" for x in (0, 1, 2, 5, 10, 20)) + " |\n|---|" + "---|" * 6 + "\n"
    for tag in [k for k in comps if "greedy" in k]:
        cells = []
        for x in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
            from collections import Counter
            c = Counter(parsed_tier(comps[tag][str(x)][i][0]) for i in sorted(test_ids))
            cells.append(", ".join(f"{k} {v}" for k, v in c.most_common()))
        md += f"| {tag} | " + " | ".join(cells) + " |\n"
    meta["total_wall_s"] = round(time.time() - t_start, 1)
    meta["timing"] = timing
    md += f"\nWall clock in the container: {meta['total_wall_s']:.0f}s on {meta['gpu']} (GRPO {meta.get('train_wall_s')}s).\n"
    (out / "eval_table.md").write_text(md)
    (out / "eval_results.json").write_text(json.dumps(scored, indent=1))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    print(md, flush=True)
    return {"table": md, "results": scored, "meta": meta, "sft_report": sft_report,
            "train_log": log_path.read_text(), "completions": comps}


def _crm_main(run: str, log_file: str, fold0_json: str, steps: int, sft_steps: int):
    """Local side of crm_phase2. fold0_json is written by scripts/crmarena_offline.py: fold 0's
    train/test task ids, the fold-0 linear constrained policy's tier probabilities per task, and its
    learned lambda (used as lam0)."""
    root = Path(__file__).resolve().parents[2]
    lines = (root / log_file).read_text().splitlines()
    f0 = json.loads((root / fold0_json).read_text())
    train_ids, test_ids, linear_probs, lam0 = f0["train"], f0["test"], f0["linear_probs"], float(f0["lambda_bar"])
    res = crm_phase2.remote(lines, train_ids, test_ids, linear_probs, run=run, steps=steps, sft_steps=sft_steps,
                            lam0=lam0)
    dst = root / "results" / "llm"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"eval_table_{run}.md").write_text(res["table"])
    (dst / f"train_log_{run}.jsonl").write_text(res["train_log"])
    (dst / f"run_meta_{run}.json").write_text(json.dumps(res["meta"], indent=1))
    (dst / f"sft_report_{run}.json").write_text(json.dumps(res["sft_report"], indent=1))
    (dst / f"eval_results_{run}.json").write_text(json.dumps({"results": res["results"],
                                                              "completions": res["completions"]}, indent=1))
    print(f"wrote results/llm/*_{run}.*")


@app.local_entrypoint()
def main(steps: int = 300, budget: float = 0.30, model: str = DEFAULT_MODEL, env: str = "sim",
         log_file: str = "", run: str = "phase1", n_tasks: int = 2000, seed: int = 0, batch: int = 64,
         lam_lr: float = 0.10, lam_kp: float = 2.0, lam_ema: float = 0.9, beta: float = 0.02,
         entropy_coef: float = 0.0, temperature: float = 1.0, num_generations: int = 8,
         price_in_prompt: bool = False, lam0: float = 0.0, sft_steps: int = 0, sft_lr: float = 2e-4,
         temp_escalate: float = 0.0, fold0_json: str = "results/crmarena/fold0_for_llm.json"):
    if env == "crm_phase2":     # phase 2: SFT + GRPO + eval on the CRMArena log in one A100-40GB container
        _crm_main(run if run != "phase1" else "crm_phase2", log_file or "traces/crmarena_b2b_counterfactual.jsonl",
                  fold0_json, steps if steps != 300 else 100, sft_steps or 150)
        return
    lines = Path(log_file).read_text().splitlines() if log_file else None
    meta = train.remote(steps=steps, budget=budget, model_name=model, env=env, log_lines=lines, run=run,
                        n_tasks=n_tasks, seed=seed, batch=batch, lam_lr=lam_lr, lam_kp=lam_kp, lam_ema=lam_ema,
                        beta=beta, entropy_coef=entropy_coef, temperature=temperature,
                        num_generations=num_generations, price_in_prompt=price_in_prompt, lam0=lam0,
                        sft_steps=sft_steps, sft_lr=sft_lr, temp_escalate=temp_escalate)
    print(json.dumps(meta, indent=1))
