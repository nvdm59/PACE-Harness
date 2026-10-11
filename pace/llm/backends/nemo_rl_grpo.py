"""NeMo-RL (NVIDIA-NeMo/RL) entry point for the price-conditioned controller task.

  PaceSimEnvLogic / make_env_actor   the task as a NeMo-RL environment (EnvironmentInterface:
                                     step(message_log_batch, metadata) -> EnvironmentReturn).
                                     One turn: the assistant's action is scored with
                                     pace.llm.rewards.SimEpisodeReward at the fixed price and the
                                     episode terminates. Every scored completion is appended to a
                                     JSONL record file, from which the smoke log is built.
  build_datums                       the phase-1c task pool as NeMo-RL DatumSpecs (prompt rendered
                                     with the Qwen3 chat template, thinking off; task in extra_env_info)
  run_training                       in-container driver, modeled on examples/run_grpo_sliding_puzzle.py
                                     (setup -> grpo_train), with the logger's metrics mirrored to JSONL
  Modal launcher                     NVIDIA's NGC container nvcr.io/nvidia/nemo-rl:v0.7.0, one A100-40GB

    modal run pace/llm/backends/nemo_rl_grpo.py                # 20-step smoke

Nothing heavy is imported at module level: tests import this file without modal / ray / nemo_rl / torch.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "nemo_rl_config.yaml"
VOL_PATH = "/vol"
RUN_ROOT = f"{VOL_PATH}/backends/nemo_rl"
NRL_HOME = "/opt/nemo-rl"
TASK_NAME = "pace_sim"

if "/root/src" not in sys.path and os.path.isdir("/root/src"):
    sys.path.insert(0, "/root/src")
elif len(HERE.parents) > 2 and str(HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[2]))


# ------------------------------------------------------------------ environment
class PaceSimEnvLogic:
    """Plain-Python core of the environment (no torch / ray), so it can be tested on CPU."""

    def __init__(self, cfg: dict | None = None):
        if "/root/src" not in sys.path and os.path.isdir("/root/src"):
            sys.path.insert(0, "/root/src")           # inside a Ray worker
        cfg = dict(cfg or {})
        self.record_path = cfg.get("record_path")
        self.calls = 0

    @staticmethod
    def completion_of(message_log: list[dict]) -> str:
        return "".join(str(m.get("content", "")) for m in message_log if m.get("role") == "assistant")

    def score_batch(self, message_log_batch: list[list[dict]], metadata: list[dict]) -> list[dict]:
        from pace.llm.backends.common import clean_completion, score
        out = []
        for log, task in zip(message_log_batch, metadata):
            c = self.completion_of(log)
            o = score(task, c)
            out.append({"call": self.calls, "task": task, "completion": clean_completion(c), "reward": o.reward,
                        "quality": o.quality, "cost": o.cost, "valid": o.valid, "action": o.action})
        if self.record_path:
            Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.record_path, "a") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
        self.calls += 1
        return out

    def step(self, message_log_batch, metadata):
        recs = self.score_batch(message_log_batch, metadata)
        n = len(recs)
        rewards = [r["reward"] for r in recs]
        fields = dict(observations=[{"role": "environment", "content": "episode finished"}] * n,
                      metadata=list(metadata), next_stop_strings=[None] * n, rewards=rewards,
                      terminateds=[True] * n, answers=[r["completion"] for r in recs])
        try:
            import torch
            from nemo_rl.environments.interfaces import EnvironmentReturn
        except ImportError:
            return fields
        fields["rewards"] = torch.tensor(rewards, dtype=torch.float32)
        fields["terminateds"] = torch.tensor(fields["terminateds"], dtype=torch.bool)
        return EnvironmentReturn(**fields)

    def global_post_process_and_metrics(self, batch):
        return batch, {}

    def shutdown(self):
        pass


def make_env_actor():
    """The Ray actor class NeMo-RL expects (EnvironmentInterface subclass)."""
    import ray
    from nemo_rl.environments.interfaces import EnvironmentInterface

    class PaceSimEnv(PaceSimEnvLogic, EnvironmentInterface):
        pass

    return ray.remote(PaceSimEnv)


# ------------------------------------------------------------------ dataset
def render_prompt(tokenizer, user: str) -> str:
    from pace.llm.backends.common import messages
    kw = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages(user), enable_thinking=False, **kw)
    except TypeError:
        return tokenizer.apply_chat_template(messages(user), **kw)


def build_datums(tokenizer, rows: list[dict], to_tensor=None) -> list[dict]:
    """task_rows(...) -> NeMo-RL DatumSpec dicts (single user message holding the rendered prompt)."""
    if to_tensor is None:
        import torch
        to_tensor = torch.tensor
    out = []
    for r in rows:
        text = render_prompt(tokenizer, r["user"])
        ids = to_tensor(tokenizer(text, add_special_tokens=False)["input_ids"])
        out.append({"message_log": [{"role": "user", "content": text, "token_ids": ids}],
                    "length": len(ids), "extra_env_info": dict(r["task"]), "loss_multiplier": 1.0,
                    "idx": int(r["index"]), "task_name": TASK_NAME})
    return out


def load_config(path: str | Path = CONFIG) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text())


def records_to_lines(records: list[dict], metrics: dict[int, dict] | None = None, group: int | None = None,
                     prompts_per_step: int | None = None) -> list[dict]:
    """Env records -> one line per GRPO step in the TRL train-log schema.

    Without dynamic sampling every env call is one step. With it, NeMo-RL logs
    `train/dynamic_sampling_num_gen_batches` per step: that many consecutive calls belong to the step;
    the pre-filter identical-reward rate is taken over all of them, and the step's stats over the groups
    it trained on (the first `prompts_per_step` groups with non-constant rewards)."""
    from pace.llm.backends.common import step_stats
    by_call: dict[int, list] = {}
    for r in records:
        by_call.setdefault(int(r["call"]), []).append(r)
    calls = sorted(by_call)
    metrics = metrics or {}
    gens = {s: int(m["train/dynamic_sampling_num_gen_batches"]) for s, m in metrics.items()
            if m.get("train/dynamic_sampling_num_gen_batches") is not None}
    if gens and group:
        plan, i = [], 0
        for step in sorted(gens):
            plan.append((step, calls[i:i + gens[step]]))
            i += gens[step]
    else:
        plan = [(c + 1, [c]) for c in calls]
    lines = []
    for step, cs in plan:
        recs = [r for c in cs for r in by_call.get(c, [])]
        if not recs:
            continue
        line = {"step": step}
        if gens and group:
            groups = [recs[i:i + group] for i in range(0, len(recs), group)]
            const = [max(x["reward"] for x in gr) - min(x["reward"] for x in gr) <= 1e-9 for gr in groups]
            kept = [gr for gr, c in zip(groups, const) if not c][: prompts_per_step or len(groups)]
            line.update(step_stats(r for gr in kept for r in gr))
            line["identical_reward_group_rate_pre"] = round(sum(const) / len(const), 4)
            line["gen_batches"] = len(cs)
        else:
            line.update(step_stats(recs))
            line["identical_reward_group_rate_pre"] = line.get("identical_reward_group_rate")
        line["identical_reward_group_rate_post"] = line.get("identical_reward_group_rate")
        m = metrics.get(step) or {}
        if m:
            line["step_s"] = m.get("timing/train/total_step_time")
            line["entropy"] = m.get("train/approx_entropy")
            line["platform"] = {k: v for k, v in m.items() if isinstance(v, (int, float)) and "traject" not in k}
        lines.append(line)
    return lines


# ------------------------------------------------------------------ in-container driver
def run_training(config_path: str, metrics_path: str, overrides: list[str] | None = None) -> None:
    import pprint

    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config as nrl_load, parse_hydra_overrides, register_omegaconf_resolvers
    from omegaconf import OmegaConf
    from torch.utils.data import Dataset

    from pace.llm.backends.common import task_rows

    register_omegaconf_resolvers()
    config = nrl_load(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    env_cfg = config.pop("env")
    for k in ("pins", "modal", "recipes"):
        config.pop(k, None)
    config["env"] = env_cfg
    config = MasterConfig(**config)
    pprint.pprint(config)
    init_ray()
    set_seed(config.grpo["seed"])
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    config.policy["generation"] = configure_generation_config(config.policy["generation"], tokenizer)

    class ListDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    lam = float(env_cfg[TASK_NAME].get("lam", 0.4))
    dataset = ListDataset(build_datums(tokenizer, task_rows(lam=lam)))
    env = make_env_actor().options(num_gpus=0).remote(cfg=dict(env_cfg[TASK_NAME]))
    task_to_env = {TASK_NAME: env}
    (policy, policy_generation, _gym, _cluster, dataloader, val_dataloader, loss_fn, logger, checkpointer,
     grpo_state, master_config, *_rest) = setup(config, tokenizer, dataset, None)

    orig = logger.log_metrics

    def log_metrics(metrics, step, prefix="", *a, **kw):
        try:
            flat = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, bool))}
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"step": int(step), **flat}) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[pace] metrics mirror failed: {e!r}", flush=True)
        return orig(metrics, step, prefix, *a, **kw)

    logger.log_metrics = log_metrics
    grpo_train(policy, policy_generation, dataloader, val_dataloader, tokenizer, loss_fn, task_to_env,
               task_to_env, logger, checkpointer, grpo_state, master_config)


def read_metrics(path: str | Path) -> dict[int, dict]:
    from pace.llm.backends.common import load_jsonl
    out: dict[int, dict] = {}
    if Path(path).exists():
        for x in load_jsonl(path):
            out.setdefault(int(x.pop("step")), {}).update(x)
    return out


# ------------------------------------------------------------------ Modal (lazy)
try:
    import modal  # noqa: F401
except ImportError:
    modal = None

if modal is not None and __name__ != "__main__":
    from pace.llm.backends.common import MODEL as _MODEL

    # in a checkout: the repo root; inside the container Modal mounts this file at /root
    _REPO = HERE.parents[2] if (len(HERE.parents) > 2 and (HERE.parents[2] / "pace").is_dir()) else Path("/root/src")
    nrl_image = (
        modal.Image.from_registry("nvcr.io/nvidia/nemo-rl:v0.7.0")
        .run_commands(f"python -c \"from huggingface_hub import snapshot_download; "
                      f"snapshot_download('{_MODEL}', local_dir='/models/qwen3-1.7b')\"")
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"})
        .add_local_dir(str(_REPO / "pace"), "/root/src/pace", ignore=["**/__pycache__/**"])
    )
    volume = modal.Volume.from_name("pace-llm", create_if_missing=True)
    app = modal.App("pace-llm-nemo-rl", image=nrl_image)

    @app.function(gpu="A100-40GB", timeout=2700, volumes={VOL_PATH: volume}, cpu=8, memory=65536)
    def train(steps: int = 20, run: str = "nemo_rl_smoke", no_step_kill_s: int = 900, time_cap_s: int = 1320,
              recipe: str = "") -> dict:
        import shutil
        import signal
        import subprocess
        t0 = time.time()
        root = Path(RUN_ROOT) / run
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        cfg_dst = Path(NRL_HOME) / "examples" / "configs" / "pace_grpo_smoke.yaml"
        shutil.copy("/root/src/pace/llm/backends/nemo_rl_config.yaml", cfg_dst)
        records, metrics_path, log_path = root / "records.jsonl", root / "metrics.jsonl", root / "stdout.log"
        overrides = [f"grpo.max_num_steps={steps}", f"++env.pace_sim.record_path={records}",
                     f"logger.log_dir={root / 'logs'}"]
        if recipe:
            rc = load_config("/root/src/pace/llm/backends/nemo_rl_config.yaml")["recipes"][recipe]
            overrides += list(rc["overrides"]) + [f"++env.pace_sim.lam={rc['lam']}",
                                                   f"checkpointing.checkpoint_dir={root / 'ckpt'}"]
        cmd = [sys.executable, "/root/src/pace/llm/backends/nemo_rl_grpo.py", "--config", str(cfg_dst),
               "--metrics", str(metrics_path), *overrides]
        print("[nemo-rl] " + " ".join(cmd), flush=True)
        status, first_step_s = "ok", None
        with open(log_path, "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=NRL_HOME, start_new_session=True)
            while p.poll() is None:
                time.sleep(10)
                if first_step_s is None and records.exists() and records.stat().st_size > 0:
                    first_step_s = time.time() - t0
                    print(f"[nemo-rl] first scored batch at {first_step_s:.0f}s after container start", flush=True)
                if first_step_s is None and time.time() - t0 > no_step_kill_s:
                    status = f"killed: no step within {no_step_kill_s}s of container start"
                elif time.time() - t0 > time_cap_s - 60:
                    status = "killed: container time limit"
                else:
                    continue
                print(f"[nemo-rl] {status}", flush=True)
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(15)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                break
        rc = p.wait()
        if rc != 0 and status == "ok":
            status = f"exit code {rc}"
        from pace.llm.backends.common import load_jsonl
        recs = load_jsonl(records) if records.exists() else []
        lines = records_to_lines(recs, read_metrics(metrics_path), group=16 if recipe else None,
                                 prompts_per_step=4 if recipe else None)
        import torch
        volume.commit()
        from importlib.metadata import PackageNotFoundError, version
        vers = {}
        for n in ("nemo_rl", "vllm", "torch", "transformers", "ray"):
            try:
                vers[n] = version(n)
            except PackageNotFoundError:
                vers[n] = None
        return {"status": status, "returncode": rc, "lines": lines, "container_s": round(time.time() - t0, 1),
                "first_step_s": first_step_s, "dataset": {"train": 2000}, "cmd": cmd,
                "log_tail": log_path.read_text()[-6000:],
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "versions": vers}

    @app.function(timeout=600, cpu=2)
    def versions() -> dict:
        """CPU only: builds/caches the image and reports the installed stack (no GPU minutes)."""
        import subprocess
        out = subprocess.run([sys.executable, "-c", "import nemo_rl, vllm, torch, transformers, ray; "
                              "print(nemo_rl.__version__, vllm.__version__, torch.__version__, "
                              "transformers.__version__, ray.__version__)"], capture_output=True, text=True)
        print(out.stdout, out.stderr[-2000:])
        return {"stdout": out.stdout, "python": sys.version}

    @app.local_entrypoint()
    def main(steps: int = 20, run: str = "nemo_rl_smoke", time_cap_s: int = 1320, recipe: str = ""):
        from pace.llm.backends.summarize import write_results
        res = train.remote(steps=steps, run=run, time_cap_s=time_cap_s, recipe=recipe)
        write_results("NeMo-RL", run, res)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--metrics", required=True)
    a, rest = ap.parse_known_args()
    run_training(a.config, a.metrics, rest)
