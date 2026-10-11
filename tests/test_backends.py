"""verl / NeMo-RL backend entry points on CPU (no GPU, no modal / ray / verl / nemo_rl / torch):
dataset builders, the reward adapters vs SimEpisodeReward, config files, re-scoring of logged
rollouts, and lazy imports of the launchers. Needs pyyaml for the config tests."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pace.llm.backends import common  # noqa: E402
from pace.llm.env_text import INVALID_PENALTY, SYSTEM, sim_rows  # noqa: E402
from pace.llm.rewards import SimEpisodeReward  # noqa: E402

BLOCKED = ("modal", "ray", "verl", "nemo_rl", "torch", "pandas", "vllm")


def _import_blocked(name: str):
    """Import a launcher module with the heavy stacks made unimportable."""
    saved = {k: sys.modules.get(k) for k in BLOCKED}
    try:
        for k in BLOCKED:
            sys.modules[k] = None              # `import k` now raises ImportError
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_launchers_import_without_heavy_stacks():
    v = _import_blocked("pace.llm.backends.verl_grpo")
    n = _import_blocked("pace.llm.backends.nemo_rl_grpo")
    t = _import_blocked("pace.llm.backends.trl_grpo")
    assert t.modal is None and not hasattr(t, "app")
    assert v.modal is None and not hasattr(v, "app")
    assert n.modal is None and not hasattr(n, "app")
    assert callable(v.compute_score) and callable(n.make_env_actor)


def test_task_pool_is_phase1c_with_fixed_price():
    rows = common.task_rows(30)
    ref = sim_rows(30, seed=0, budget=0.30, price=True)
    for r, x in zip(rows, ref):
        assert r["task"]["task_id"] == json.loads(x["task"])["task_id"]
        assert r["task"]["lam"] == common.FIXED_LAM
        assert "λ=0.4" in r["user"] and "<<PRICE_OF_COST>>" not in r["user"]
        assert "latent" not in r["user"]


def test_verl_dataset_columns():
    v = _import_blocked("pace.llm.backends.verl_grpo")
    rows = v.build_rows(n=12, seed=0)
    assert len(rows) == 12
    for r in rows:
        assert set(r) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert r["data_source"] == common.DATA_SOURCE
        assert [m["role"] for m in r["prompt"]] == ["system", "user"] and r["prompt"][0]["content"] == SYSTEM
        gt = json.loads(r["reward_model"]["ground_truth"])
        assert gt["env"] == "sim" and gt["lam"] == common.FIXED_LAM
        assert r["extra_info"]["task"] == r["reward_model"]["ground_truth"]
    assert len(v.build_rows(split="val")) == common.N_VAL
    try:
        import pandas as pd
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as d:
        assert v.write_parquet(d) == {"train": common.N_TRAIN, "val": common.N_VAL}
        df = pd.read_parquet(Path(d) / "train.parquet")
        assert list(df.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def test_verl_reward_matches_simulator():
    v = _import_blocked("pace.llm.backends.verl_grpo")
    sim = SimEpisodeReward()
    for r in v.build_rows(n=10, seed=5):
        gt = r["reward_model"]["ground_truth"]
        task = json.loads(gt)
        for comp in ["<action>mid ESCALATE</action>", "<action>frontier ANSWER</action><|im_end|>"]:
            out = v.compute_score(common.DATA_SOURCE, comp, gt, r["extra_info"])
            ref = sim(task, comp.replace("<|im_end|>", ""), common.FIXED_LAM)
            assert abs(out["score"] - ref.reward) < 1e-12 and out["valid"] == 1.0
            assert abs(out["score"] - (out["quality"] - common.FIXED_LAM * out["cost"])) < 1e-12
        bad = v.compute_score(common.DATA_SOURCE, "<think></think><action>mid ANSWER</action>", gt)
        assert bad["score"] == INVALID_PENALTY and bad["valid"] == 0.0
        # ground truth missing -> the task is read from extra_info
        assert v.compute_score(common.DATA_SOURCE, "<action>cheap ANSWER</action>", None,
                               {"task": gt})["valid"] == 1.0


def test_nemo_env_scores_and_records():
    n = _import_blocked("pace.llm.backends.nemo_rl_grpo")
    rows = common.task_rows(4)
    with tempfile.TemporaryDirectory() as d:
        rec = Path(d) / "records.jsonl"
        env = n.PaceSimEnvLogic({"record_path": str(rec)})
        logs = [[{"role": "user", "content": "x"}, {"role": "assistant", "content": c}]
                for c in ["<action>mid ESCALATE</action><|im_end|>", "nonsense", "<action>cheap ANSWER</action>",
                          "<action>frontier VERIFY</action>"]]
        meta = [r["task"] for r in rows]
        out = env.step(logs, meta)        # torch absent here -> plain fields
        assert out["terminateds"] == [True] * 4 and out["next_stop_strings"] == [None] * 4
        assert out["rewards"][1] == INVALID_PENALTY
        sim = SimEpisodeReward()
        assert abs(out["rewards"][0] - sim(meta[0], "<action>mid ESCALATE</action>", common.FIXED_LAM).reward) < 1e-12
        env.step(logs, meta)
        recs = common.load_jsonl(rec)
        assert len(recs) == 8 and {r["call"] for r in recs} == {0, 1}
        lines = n.records_to_lines(recs, {1: {"timing/train/total_step_time": 2.5}})
        assert [x["step"] for x in lines] == [1, 2] and lines[0]["n"] == 4 and lines[0]["invalid_rate"] == 0.25
        assert lines[0]["step_s"] == 2.5


def test_fixed_price_line_and_eval_stats():
    from pace.llm.env_text import price_line
    assert common.price_text(0.4) == price_line(0.4)           # smoke prompts == the TRL phase-1c prompts
    rows = common.task_rows(40, 777, 0.44)
    assert all("λ=0.44" in r["user"] and r["task"]["lam"] == 0.44 for r in rows)
    tasks = [r["task"] for r in rows]
    comp = ["<action>mid ESCALATE</action>" if t["proxy"] == "easy" else "<action>frontier ANSWER</action>"
            for t in tasks]
    e = common.eval_stats(tasks, comp)
    assert e["acts_by_difficulty"] and e["invalid_rate"] == 0.0
    assert e["by_proxy"]["easy"]["mix"] == {"mid ESCALATE": 1.0}
    assert abs(e["J"] - (e["quality"] - 0.44 * e["cost"])) < 1e-3
    u = common.eval_stats(tasks, ["<action>mid ESCALATE</action>"] * 40)
    assert not u["acts_by_difficulty"]


def test_dynamic_sampling_helpers_and_recipes():
    t = _import_blocked("pace.llm.backends.trl_grpo")
    assert t.split_groups([0.1] * 16 + [0.1] * 15 + [0.2], 16) == [True, False]
    v = _import_blocked("pace.llm.backends.verl_grpo")
    a = v.hydra_overrides(v.load_config(), recipe="dapo_1b", data_dir="/d", model="/m", rollout_dir="/r",
                          ckpt_dir="/c", reward_path="/p.py")
    for want in ("algorithm.filter_groups.enable=true", "actor_rollout_ref.actor.clip_ratio_high=0.28",
                 "actor_rollout_ref.rollout.n=16", "actor_rollout_ref.actor.kl_loss_coef=0.05",
                 "data.train_batch_size=4"):
        assert want in a, want
    assert len([x for x in a if x.startswith("data.train_batch_size=")]) == 1
    rows = v.build_rows(n=3, lam=0.44)
    assert json.loads(rows[0]["reward_model"]["ground_truth"])["lam"] == 0.44
    n = _import_blocked("pace.llm.backends.nemo_rl_grpo")
    rc = n.load_config()["recipes"]["dapo_1b"]
    assert rc["lam"] == 0.44 and "grpo.use_dynamic_sampling=true" in rc["overrides"]
    assert "loss_fn.ratio_clip_max=0.28" in rc["overrides"]


class _FakeTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        assert not enable_thinking
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 50 for ch in text]}


def test_nemo_datums():
    n = _import_blocked("pace.llm.backends.nemo_rl_grpo")
    rows = common.task_rows(3)
    ds = n.build_datums(_FakeTok(), rows, to_tensor=list)
    for dm, r in zip(ds, rows):
        assert set(dm) == {"message_log", "length", "extra_env_info", "loss_multiplier", "idx", "task_name"}
        assert dm["task_name"] == n.TASK_NAME and dm["extra_env_info"] == r["task"]
        assert dm["length"] == len(dm["message_log"][0]["token_ids"]) > 0
        assert SYSTEM in dm["message_log"][0]["content"]


def test_configs_load():
    import yaml  # noqa: F401
    v = _import_blocked("pace.llm.backends.verl_grpo")
    n = _import_blocked("pace.llm.backends.nemo_rl_grpo")
    vc = v.load_config()
    o = vc["overrides"]
    assert o["algorithm.adv_estimator"] == "grpo" and o["actor_rollout_ref.rollout.n"] in (8, 16)
    assert o["actor_rollout_ref.actor.use_kl_loss"] is True and o["algorithm.use_kl_in_reward"] is False
    assert o["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    for k in ("actor_rollout_ref.actor.clip_ratio_low", "actor_rollout_ref.actor.clip_ratio_high",
              "algorithm.filter_groups.enable"):
        assert k in o
    args = v.hydra_overrides(vc, data_dir="/d", model="/m", rollout_dir="/r", ckpt_dir="/c", reward_path="/p.py")
    assert "data.train_files=/d/train.parquet" in args and "reward.custom_reward_function.path=/p.py" in args
    assert "actor_rollout_ref.actor.use_kl_loss=true" in args and all("{" not in a for a in args)
    nc = n.load_config()
    assert nc["defaults"] == "grpo_math_1B.yaml"
    assert nc["grpo"]["num_generations_per_prompt"] in (8, 16) and nc["grpo"]["max_rollout_turns"] == 1
    assert nc["policy"]["generation"]["max_new_tokens"] == 16
    assert nc["loss_fn"]["token_level_loss"] is True and "use_dynamic_sampling" in nc["grpo"]
    assert nc["policy"]["tokenizer"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_verl_rescore_and_summary():
    v = _import_blocked("pace.llm.backends.verl_grpo")
    from pace.llm.backends.summarize import write_results
    rows = v.build_rows(n=2, seed=0)
    with tempfile.TemporaryDirectory() as d:
        rd = Path(d) / "rollouts"
        rd.mkdir()
        for step in (1, 2):
            with open(rd / f"{step}.jsonl", "w") as f:
                for r in rows:
                    gt = r["reward_model"]["ground_truth"]
                    for c in ["<action>mid ESCALATE</action>", "<action>mid ANSWER</action>"]:
                        s = v.compute_score(common.DATA_SOURCE, c, gt)["score"]
                        f.write(json.dumps({"input": "", "output": c, "gts": gt, "score": s, "step": step}) + "\n")
        lines = v.rescore_rollout_dir(rd, {1: {"timing_s/step": 3.0, "actor/pg_loss": 0.1}})
        assert [x["step"] for x in lines] == [1, 2] and lines[0]["n"] == 4
        assert lines[0]["score_agrees"] == "4/4" and lines[0]["degenerate_group_rate"] == 0.0
        assert lines[0]["step_s"] == 3.0 and lines[0]["platform"] == {"timing_s/step": 3.0, "actor/pg_loss": 0.1}
        meta = write_results("verl", "t", {"status": "ok", "lines": lines}, out_dir=Path(d) / "out")
        assert meta["platform"] == "verl" and (Path(d) / "out" / "t.md").read_text().count("| 1 |") == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print(f"{'all passed' if not fails else f'{fails} failed'}")
    sys.exit(1 if fails else 0)
