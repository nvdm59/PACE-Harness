"""verl (volcengine/verl, HybridFlow) entry point for the price-conditioned controller task.

Three pieces:
  build_rows / write_parquet   the phase-1c task pool as verl's RL dataset (one row per prompt:
                               chat `prompt`, `data_source`, `reward_model.ground_truth` = task JSON)
  compute_score                verl's custom reward function (data_source, solution_str,
                               ground_truth, extra_info) -> {"score": r, quality, cost, valid}
                               via pace.llm.rewards.SimEpisodeReward at the fixed price
  Modal launcher               verl 0.9.1 in verl's vLLM 0.24 image, one A100-40GB,
                               `python -m verl.trainer.main_ppo` with the overrides of
                               verl_config.yaml, a no-step watchdog, the rollouts re-scored with
                               our reward and written to results/llm/backends/verl_smoke.jsonl

    modal run pace/llm/backends/verl_grpo.py                 # 20-step smoke
    modal run pace/llm/backends/verl_grpo.py --steps 200     # longer run, same config

Nothing heavy is imported at module level: tests import this file without modal / verl / pandas.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "verl_config.yaml"
VOL_PATH = "/vol"
RUN_ROOT = f"{VOL_PATH}/backends/verl"

if "/root/src" not in sys.path and os.path.isdir("/root/src"):
    sys.path.insert(0, "/root/src")          # verl loads this file by path inside Ray workers
elif len(HERE.parents) > 2 and str(HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[2]))  # `modal run` from a checkout: the repo root


# ------------------------------------------------------------------ dataset
def build_rows(n: int | None = None, seed: int | None = None, split: str = "train",
               lam: float | None = None) -> list[dict]:
    """verl RL-dataset rows (plain dicts) for the fixed-price task pool."""
    from pace.llm.backends.common import (DATA_SOURCE, FIXED_LAM, N_TRAIN, N_VAL, TRAIN_SEED, VAL_SEED, messages,
                                          task_rows)
    if n is None:
        n = N_TRAIN if split == "train" else N_VAL
    if seed is None:
        seed = TRAIN_SEED if split == "train" else VAL_SEED
    rows = []
    for r in task_rows(n, seed, FIXED_LAM if lam is None else lam):
        gt = json.dumps(r["task"], sort_keys=True)
        rows.append({
            "data_source": DATA_SOURCE,
            "prompt": messages(r["user"]),
            "ability": "routing",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {"split": split, "index": r["index"], "task": gt},
        })
    return rows


def write_parquet(out_dir: str | Path, lam: float | None = None) -> dict:
    """train.parquet (2,000 phase-1c tasks) + val.parquet (64 tasks of the eval seed)."""
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split in ("train", "val"):
        rows = build_rows(split=split, lam=lam)
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet")
        sizes[split] = len(rows)
    return sizes


# ------------------------------------------------------------------ reward
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """verl custom reward function. `ground_truth` carries the task JSON (with its price)."""
    from pace.llm.backends.common import score
    task = ground_truth
    if not task and isinstance(extra_info, dict):
        task = extra_info.get("task")
    o = score(task, solution_str)
    return {"score": float(o.reward), "quality": float(o.quality), "cost": float(o.cost),
            "valid": float(o.valid)}


# ------------------------------------------------------------------ config -> CLI
def load_config(path: str | Path = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def hydra_overrides(cfg: dict, recipe: str = "", **fill) -> list[str]:
    """`key=value` overrides for verl.trainer.main_ppo, with {placeholders} filled; a recipe's
    overrides replace the base ones key by key."""
    ov = dict(cfg["overrides"])
    if recipe:
        ov.update(cfg["recipes"][recipe]["overrides"])
    out = []
    for k, v in ov.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, str):
            v = v.format(**fill) if "{" in v and fill else v
        out.append(f"{k}={v}")
    return out


# ------------------------------------------------------------------ rollouts -> smoke log
def rescore_rollout_dir(rollout_dir: str | Path, metrics: dict[int, dict] | None = None) -> list[dict]:
    """verl's per-step rollout dumps ({step}.jsonl: output, gts, score) -> one line per step in the
    TRL train-log schema, re-scored with our reward. `score_agrees` checks verl's own score."""
    from pace.llm.backends.common import load_jsonl, score, step_stats
    lines = []
    files = sorted(Path(rollout_dir).glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    for f in files:
        if not f.stem.isdigit():
            continue
        recs = load_jsonl(f)
        st = step_stats({"task": r["gts"], "completion": r["output"]} for r in recs)
        agree = sum(abs(score(r["gts"], r["output"]).reward - float(r["score"])) < 1e-6 for r in recs)
        line = {"step": int(f.stem), **st, "score_agrees": f"{agree}/{len(recs)}"}
        m = (metrics or {}).get(int(f.stem)) or {}
        ev = next((v for k, v in m.items() if k.endswith("filter_groups/evicted_samples")), None)
        if ev is not None and st.get("n"):
            g = max(1, len(recs) // max(1, len({r["gts"] for r in recs})))
            kept, evicted = st["n"] / g, float(ev) / g
            line["identical_reward_group_rate_pre"] = round(evicted / (evicted + kept), 4)
        line["identical_reward_group_rate_post"] = st.get("identical_reward_group_rate")
        if m:
            line["step_s"] = m.get("timing_s/step")
            line["entropy"] = next((v for k, v in m.items() if k.endswith("actor/entropy")), None)
            line["platform"] = {k: v for k, v in m.items() if k.split("/")[0] in
                                ("actor", "critic", "timing_s", "response_length", "perf", "train", "training")
                                and isinstance(v, (int, float)) and "traject" not in k}
        lines.append(line)
    return lines


def read_file_logger(path: str | Path) -> dict[int, dict]:
    from pace.llm.backends.common import load_jsonl
    return {int(x["step"]): x.get("data", {}) for x in load_jsonl(path)} if Path(path).exists() else {}


# ------------------------------------------------------------------ Modal (lazy)
try:
    import modal  # noqa: F401
except ImportError:          # tests and plain imports: no launcher
    modal = None

if modal is not None:
    from pace.llm.backends.common import MODEL as _MODEL

    # in a checkout: the repo root; inside the container Modal mounts this file at /root
    _REPO = HERE.parents[2] if (len(HERE.parents) > 2 and (HERE.parents[2] / "pace").is_dir()) else Path("/root/src")
    _CFG_PINS = {"verl": "0.9.1", "image": "verlai/verl:vllm024.dev2"}   # mirrored in verl_config.yaml
    verl_image = (
        modal.Image.from_registry(_CFG_PINS["image"])
        .pip_install(f"verl[verl-core]=={_CFG_PINS['verl']}", "pyyaml")
        .run_commands(f"python -c \"from huggingface_hub import snapshot_download; "
                      f"snapshot_download('{_MODEL}', local_dir='/models/qwen3-1.7b')\"")
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false", "RAY_DEDUP_LOGS": "0",
              "VLLM_LOGGING_LEVEL": "WARNING", "HF_HUB_OFFLINE": "1"})
        .add_local_dir(str(_REPO / "pace"), "/root/src/pace", ignore=["**/__pycache__/**"])
    )
    volume = modal.Volume.from_name("pace-llm", create_if_missing=True)
    app = modal.App("pace-llm-verl", image=verl_image)

    @app.function(gpu="A100-40GB", timeout=2700, volumes={VOL_PATH: volume}, cpu=8, memory=65536)
    def train(steps: int = 20, run: str = "verl_smoke", no_step_kill_s: int = 900, recipe: str = "",
              time_cap_s: int = 1320) -> dict:
        import signal
        import subprocess
        t0 = time.time()
        cfg = load_config("/root/src/pace/llm/backends/verl_config.yaml")
        root = Path(RUN_ROOT) / run
        data_dir, rollout_dir, ckpt_dir = root / "data", root / "rollouts", root / "ckpt"
        for d in (data_dir, rollout_dir, ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)
        rc_cfg = cfg["recipes"][recipe] if recipe else {}
        sizes = write_parquet(data_dir, lam=rc_cfg.get("lam"))
        metrics_path = root / "metrics.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()
        args = hydra_overrides(cfg, data_dir=data_dir, model="/models/qwen3-1.7b", rollout_dir=rollout_dir,
                               ckpt_dir=ckpt_dir, reward_path="/root/src/pace/llm/backends/verl_grpo.py",
                               recipe=recipe)
        args = [a for a in args if not a.startswith("trainer.total_training_steps=")]
        args.append(f"trainer.total_training_steps={steps}")
        cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *args]
        print("[verl] " + " ".join(cmd), flush=True)
        env = {**os.environ, "VERL_FILE_LOGGER_PATH": str(metrics_path)}
        log_path = root / "stdout.log"
        status = "ok"
        with open(log_path, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            first_step_s = None
            while p.poll() is None:
                time.sleep(10)
                n_logged = len(read_file_logger(metrics_path))
                if n_logged and first_step_s is None:
                    first_step_s = time.time() - t0
                    print(f"[verl] first step logged at {first_step_s:.0f}s after container start", flush=True)
                if first_step_s is None and time.time() - t0 > no_step_kill_s:
                    status = f"killed: no step within {no_step_kill_s}s of container start"
                elif time.time() - t0 > time_cap_s - 120:
                    status = "killed: container time limit"
                else:
                    continue
                print(f"[verl] {status}", flush=True)
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(15)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                break
        rc = p.wait()
        tail = log_path.read_text()[-6000:]
        if rc != 0 and status == "ok":
            status = f"exit code {rc}"
        metrics = read_file_logger(metrics_path)
        lines = rescore_rollout_dir(rollout_dir, metrics)
        adapter = None
        saved = sorted(ckpt_dir.glob("global_step_*"), key=lambda d: int(d.name.split("_")[-1]))
        actor = saved[-1] / "actor" if saved else ckpt_dir / "none"
        if recipe and actor.exists():   # verl's own merger writes the LoRA in PEFT format (lora_adapter/)
            mres = subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                                   "--local_dir", str(actor), "--target_dir", str(root / "merged")],
                                  capture_output=True, text=True)
            print(mres.stdout[-2000:], mres.stderr[-3000:], flush=True)
            if (root / "merged" / "lora_adapter" / "adapter_config.json").exists():
                adapter = str(root / "merged" / "lora_adapter")
            volume.commit()
        import torch
        volume.commit()
        return {"status": status, "returncode": rc, "lines": lines, "container_s": round(time.time() - t0, 1),
                "first_step_s": first_step_s, "dataset": sizes, "cmd": cmd, "log_tail": tail, "adapter": adapter,
                "adapter_step": int(actor.parent.name.split("_")[-1]) if adapter else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "versions": _versions(["verl", "vllm", "torch", "transformers", "ray", "peft"])}

    def _versions(names):
        from importlib.metadata import PackageNotFoundError, version
        out = {}
        for n in names:
            try:
                out[n] = version(n)
            except PackageNotFoundError:
                out[n] = None
        return out

    @app.local_entrypoint()
    def main(steps: int = 20, run: str = "verl_smoke", recipe: str = "", time_cap_s: int = 1320):
        from pace.llm.backends.summarize import write_results
        res = train.remote(steps=steps, run=run, recipe=recipe, time_cap_s=time_cap_s)
        write_results("verl", run, res)
