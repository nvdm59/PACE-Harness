"""τ-bench harness, collector and analysis -- all offline.

τ-bench's env is replaced by a scripted fake (tools, database flag, user simulator through the
installed completion shim, calculate_reward); litellm by a scripted completion function. No
network, no API keys, no spend.

    python -m pytest tests/test_taubench.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pace.collect import EXIT_STOPPED_AT_CAP, FixedArmPolicy, run_episode  # noqa: E402
from pace.harness.taubench import (ACTIONS, FEATURE_NAMES, FORCE_END, N_FEATURES, TURN0_IDX,  # noqa: E402
                                   TauBenchHarness)
from pace.prices import MODELS, call_cost  # noqa: E402
from pace.spend import SpendLedger  # noqa: E402
from pace.trace import read_jsonl  # noqa: E402


# ------------------------------------------------------------------ fakes
class FakeAction(SimpleNamespace):
    pass


class FakeEnv:
    """Task: call `fix` once, then tell the user "done 42"; the user stops after hearing "done"."""
    tools_info = [{"type": "function", "function": {"name": "fix", "parameters": {}}}]
    wiki = "You are a retail agent."
    terminate_tools = ["transfer_to_human_agents"]

    def __init__(self, user_fn, idx):
        self.user_fn, self.idx = user_fn, idx
        self.actions, self.fixed = [], False
        self.user_fn(messages=[{"role": "system", "content": "You are a user."}])   # constructor reset, no task

    def _user(self, text):
        r = self.user_fn(messages=[{"role": "system", "content": "You are a user.\n\nInstruction: fix it"},
                                   {"role": "user", "content": text}])
        return r.choices[0].message.content

    def reset(self, task_index=None):
        self.actions, self.fixed = [], False
        return SimpleNamespace(observation=self._user("Hi! How can I help you today?"))

    def calculate_reward(self):
        ok = self.fixed and any(a.name == "respond" and "42" in a.kwargs["content"] for a in self.actions)
        return SimpleNamespace(reward=1.0 if ok else 0.0, info=SimpleNamespace(r_actions=self.fixed, r_outputs=None))

    def step(self, action):
        self.actions.append(action)
        done, info = False, SimpleNamespace(reward_info=None)
        if action.name == "respond":
            obs = self._user(action.kwargs["content"])
            done = "###STOP###" in obs
        elif action.name == "fix":
            self.fixed, obs = True, "ok: fixed"
        else:
            obs = f"Unknown action {action.name}"
        if done:
            info.reward_info = self.calculate_reward()
        return SimpleNamespace(observation=obs, reward=0.0, done=done, info=info)


class FakeBackend:
    def __init__(self, n=6):
        self.n, self.user_fn = n, None

    def install_user_sim(self, fn):
        self.user_fn = fn

    def n_tasks(self):
        return self.n

    def make_env(self, idx):
        return FakeEnv(self.user_fn, idx)

    @staticmethod
    def action(name, kwargs):
        return FakeAction(name=name, kwargs=kwargs)

    @staticmethod
    def message_to_action(m):
        if m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]
            return FakeAction(name=f["name"], kwargs=json.loads(f["arguments"]))
        return FakeAction(name="respond", kwargs={"content": m["content"]})


def _resp(content=None, tool=None, args="{}", pin=1000, pout=50):
    tc = [{"id": "c1", "type": "function", "function": {"name": tool, "arguments": args}}] if tool else None
    msg = {"role": "assistant", "content": content, "tool_calls": tc}
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout))


def scripted_completion(calls):
    """Agent: tool `fix` first, then "done 42". User sim: "please fix" / ###STOP### after "done"."""
    def fn(**kw):
        calls.append(kw)
        msgs = kw["messages"]
        if "tools" not in kw:            # user simulator
            last = msgs[-1]["content"]
            return _resp("###STOP###" if "done" in last else "please fix my order", pin=200, pout=10)
        if kw.get("tool_choice") == "none":
            return _resp("done 42 (forced)")
        if not any(m.get("role") == "tool" for m in msgs):
            return _resp(tool="fix")
        return _resp("done 42")
    return fn


def make(tmp, cap=None, n=6, completion=None, **kw):
    calls = []
    ledger = SpendLedger(tmp, cap)
    h = TauBenchHarness("retail", ledger=ledger, backend=FakeBackend(n),
                        completion_fn=completion or scripted_completion(calls), **kw)
    return h, calls


# ------------------------------------------------------------------ tests
def test_features_and_actions():
    assert len(ACTIONS) == 6 and FEATURE_NAMES[TURN0_IDX] == "turn=0"
    with tempfile.TemporaryDirectory() as d:
        h, _ = make(d)
        s = h.reset(h.sample_tasks(1)[0])
        x = h.featurize(s)
        assert len(x) == N_FEATURES and x[FEATURE_NAMES.index("dom=retail")] == 1.0
        assert x[TURN0_IDX] == 1.0 and x[FEATURE_NAMES.index("last=none")] == 1.0


def test_episode_continue_and_costs_split():
    with tempfile.TemporaryDirectory() as d:
        h, calls = make(d)
        task = h.sample_tasks(1)[0]
        s = h.reset(task)
        s, done, c1, _ = h.step(s, "CONTINUE@mid")
        assert not done and s.extra["last_obs"] == "tool_ok" and s.extra["tool_calls"] == 1
        assert s.extra["messages"][-1]["role"] == "tool"
        s, done, c2, _ = h.step(s, "CONTINUE@mid")
        assert done and s.extra["end_reason"] == "user_stop"
        assert h.quality(s) == 1.0 and h.evaluate(s)["reward_strict"] == 1.0
        # agent cost = agent calls only, on the mid model; user sim on the cheap model, own purpose
        assert abs(s.cost_so_far - 2 * call_cost(MODELS["mid"], 1000, 50)) < 1e-12
        meta = h.episode_meta(s)
        assert meta["user_calls"] == 2 and abs(meta["user_cost_usd"] - 2 * call_cost(MODELS["cheap"], 200, 10)) < 1e-9
        rows = [json.loads(l) for l in open(os.path.join(d, "spend.jsonl"))]
        assert {r["purpose"] for r in rows} == {"user_sim", "agent:mid"}
        assert all(r["model"] == MODELS["cheap"] for r in rows if r["purpose"] == "user_sim")
        # tools sent to the agent only, never to the user simulator
        assert all(("tools" in k) == (k["model"].endswith(MODELS["mid"])) for k in calls)


def test_answer_forces_end_and_scores():
    with tempfile.TemporaryDirectory() as d:
        h, calls = make(d)
        s = h.reset(h.sample_tasks(1)[0])
        s, _, _, _ = h.step(s, "CONTINUE@cheap")
        s, done, _, _ = h.step(s, "ANSWER@frontier")
        assert done and s.extra["end_reason"] == "forced_end"
        assert calls[-1]["tool_choice"] == "none" and calls[-1]["messages"][-1]["content"] == FORCE_END
        assert h.quality(s) == 1.0 and h.evaluate(s)["reward_strict"] == 0.0


def test_max_turns_only_answer_legal():
    with tempfile.TemporaryDirectory() as d:
        h, _ = make(d, max_turns=2)
        s = h.reset(h.sample_tasks(1)[0])
        assert set(h.legal_actions(s)) == set(ACTIONS)
        s, _, _, _ = h.step(s, "CONTINUE@cheap")
        assert all(a.startswith("ANSWER@") for a in h.legal_actions(s))
        ep = run_episode(h, FixedArmPolicy("cheap", h.all_actions, 3, TURN0_IDX), h.sample_tasks(1)[0],
                         __import__("random").Random(0), log_counterfactual=False, log_state=False)
        assert [st.action_name for st in ep.steps] == ["CONTINUE@cheap", "ANSWER@cheap"]
        assert ep.steps[0].propensity == 1 / 3 and ep.steps[1].propensity == 1.0
        assert ep.meta["end_reason"] == "max_turns"


def test_malformed_tool_args_do_not_crash():
    def fn(**kw):
        if "tools" not in kw:
            return _resp("hello", pin=10, pout=1)
        return _resp(tool="fix", args="{not json")
    with tempfile.TemporaryDirectory() as d:
        h, _ = make(d, completion=fn)
        s = h.reset(h.sample_tasks(1)[0])
        s, done, _, _ = h.step(s, "CONTINUE@cheap")
        assert not done and s.extra["last_obs"] == "invalid" and s.extra["messages"][-1]["role"] == "tool"


def test_sample_is_seeded_prefix_stable():
    with tempfile.TemporaryDirectory() as d:
        h, _ = make(d, n=20)
        a = [t.task_id for t in h.sample_tasks(5, seed=0)]
        b = [t.task_id for t in h.sample_tasks(12, seed=0)]
        assert a == b[:5] and len(set(b)) == 12 and a != [t.task_id for t in h.sample_tasks(5, seed=1)]


def test_collect_resume_and_cap_then_analyze():
    from pace.experiments.taubench.run import _analyze, collect_taubench
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "log.jsonl")
        h, _ = make(d, cap=0.06)             # enough for a few episodes only
        rc = collect_taubench(h, out, 6, log=lambda *_: None)
        assert rc == EXIT_STOPPED_AT_CAP and os.path.exists(os.path.join(d, "STOPPED_AT_CAP"))
        n1 = len(read_jsonl(out))
        assert 0 < n1 < 18
        h2, _ = make(d, cap=10.0)
        assert collect_taubench(h2, out, 6, log=lambda *_: None) == 0
        eps = read_jsonl(out)
        assert len(eps) == 18 and len({(e.task_id, e.meta["arm"]) for e in eps}) == 18
        assert all(e.steps[0].propensity == 1 / 3 for e in eps)
        md = os.path.join(d, "RESULTS.md")
        ns = SimpleNamespace(log=[out], out=md, budget=None, redraws=5, boot=20)
        try:
            _analyze(ns)
        except SystemExit:
            pass
        text = open(md).read()
        assert "6 tasks × 3 tiers" in text and "Per-task oracle" in text and "pass^1" in text
        assert "trajectory" not in text.lower()
        # crm_context is restored after the analysis
        import pace.features as F
        from pace.harness.taubench import crm_layout
        assert F._crm_layout is not crm_layout


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
