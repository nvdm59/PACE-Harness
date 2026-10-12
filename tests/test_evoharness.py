"""EvoHarnessBench adapter, collector and analysis on the offline MockBackend (no network, no API)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from pace.experiments.evoharness import run as evo_run  # noqa: E402
from pace.harness.crmarena import PricedLLM  # noqa: E402
from pace.harness.evoharness import (ACTIONS, N_FEATURES, EvoHarness, MockBackend, SDKSession,  # noqa: E402
                                     parse_action)
from pace.spend import SpendLedger  # noqa: E402


def _resp(text, i=500, o=40):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=i, completion_tokens=o))


class Script:
    """Replies from a queue; records the messages each call saw."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, **kw):
        self.seen.append(kw)
        return _resp(self.replies.pop(0))


def _harness(tmp, axis="skills", stage=2, replies=(), cap=None, **kw):
    ledger = SpendLedger(tmp, cap)
    fn = Script(replies)
    llm = PricedLLM(ledger, completion_fn=fn)
    return EvoHarness(MockBackend(), axis, stage, ledger=ledger, llm=llm, **kw), fn


def test_parse_action():
    assert parse_action('x <action>{"type": "tool", "name": "a", "arguments": {"k": 1}}</action>')["name"] == "a"
    assert parse_action('{"type": "answer", "content": "ok"} trailing')["type"] == "answer"
    assert parse_action("<action>{bad json}</action>") is None
    assert parse_action('<action>{"type": "dance"}</action>') is None


def test_episode_tool_skill_answer_graded_by_service():
    with tempfile.TemporaryDirectory() as d:
        h, fn = _harness(d, replies=[
            '<action>{"type": "skill", "name": "skill_2"}</action>',
            '<action>{"type": "tool", "name": "tool_1", "arguments": {}}</action>',
            '<action>{"type": "answer", "content": "done"}</action>'])
        task = [t for t in h.tasks() if t.task_id == "skills-s2-0"][0]   # needs tool_1 + skill_2
        s = h.reset(task)
        legal = h.legal_actions(s)
        assert "SKILL@cheap" in legal and "DELEGATE@cheap" not in legal and len(h.featurize(s)) == N_FEATURES
        s1, done, cost, _ = h.step(s, "ACT@cheap")
        assert not done and cost > 0 and s.step == 0 and s1.extra["skills"] == 1
        s2, _, _, _ = h.step(s1, "TOOL@mid")
        assert "must call exactly one tool" in fn.seen[1]["messages"][-1]["content"]
        assert fn.seen[1]["model"] == "anthropic/claude-sonnet-5-5"
        s3, done, _, _ = h.step(s2, "ACT@frontier")
        assert done and h.quality(s3) == 1.0
        m = h.episode_meta(s3)
        assert m["pass"] and m["types"] == ["skill", "tool", "answer"] and m["tiers"] == ["cheap", "mid", "frontier"]
        assert h.backend.opened[-1].closed
        assert h.ledger.n_calls == 3 and abs(h.ledger.total - s3.cost_so_far) < 1e-12


def test_invalid_reply_and_forced_answer_on_last_turn():
    with tempfile.TemporaryDirectory() as d:
        h, _ = _harness(d, axis="tools", stage=1, max_turns=2, replies=["no action here", "plain text answer"])
        s = h.reset(h.tasks()[0])
        s, done, _, _ = h.step(s, "ACT@cheap")
        assert not done and s.extra["invalid"] == 1 and s.extra["last_obs"] == "invalid"
        assert h.legal_actions(s) == [a for a in ACTIONS if a.startswith("ANSWER@")]
        with pytest.raises(ValueError):
            h.step(s, "TOOL@cheap")
        s, done, _, _ = h.step(s, "ANSWER@cheap")
        assert done and s.extra["end_reason"] == "forced_answer" and s.extra["answer"] == "plain text answer"
        assert h.quality(s) == 0.0


def test_delegate_only_on_agents_axis():
    with tempfile.TemporaryDirectory() as d:
        h, _ = _harness(d, axis="agents", stage=2, replies=[
            '<action>{"type": "delegate", "agent": "agent_2", "instruction": "fix it"}</action>'])
        s = h.reset(h.tasks()[0])
        assert "DELEGATE@mid" in h.legal_actions(s) and "SKILL@mid" not in h.legal_actions(s)
        s, _, _, _ = h.step(s, "DELEGATE@mid")
        assert s.extra["delegations"] == 1 and s.extra["last_obs"] == "ok"


def test_sdk_session_maps_documented_mcp_calls():
    class MCP:
        def __init__(self):
            self.closed = False

        def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="create_ticket", description="d", inputSchema={})])

        def call_tool(self, name, arguments):
            return SimpleNamespace(content=[SimpleNamespace(text=f"{name} ok")])

        def close(self):
            self.closed = True

    mcp = MCP()
    task = SimpleNamespace(mcp_session=lambda srv: mcp)
    sess = SDKSession(task, ["itsm"], grade=lambda ans: {"pass": True, "score": 1.0})
    assert sess.catalog()["tools"][0]["name"] == "create_ticket"
    assert sess.call_tool("create_ticket", {}) == ("create_ticket ok", True)
    assert sess.call_tool("nope", {})[1] is False
    assert sess.submit("x")["pass"]
    sess.close()
    assert mcp.closed


def test_collect_counterfactual_resume_and_analyze():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "evoharness_tools_s2.jsonl"
        h = EvoHarness(MockBackend(), "tools", 2, ledger=SpendLedger(d, 10.0),
                       llm=PricedLLM(SpendLedger(d, 10.0), completion_fn=evo_run.scripted_completion))
        h.ledger = h.llm.ledger
        tiers = ["cheap", "mid", "frontier"]
        assert evo_run.collect(h, out, 3, tiers, log=lambda *_: None) == 0
        n1 = len(out.read_text().splitlines())
        assert n1 == 9
        assert evo_run.collect(h, out, 5, tiers, log=lambda *_: None) == 0      # resume: only new tasks run
        assert len(out.read_text().splitlines()) == 15
        rep = evo_run.analyze([out], tiers)["tools/H2"]
        assert rep["n_tasks_complete"] == 5
        assert set(rep["protocol"]["heldout"]).isdisjoint(rep["protocol"]["adaptation"])
        assert rep["oracle"]["cost"] <= min(rep["per_tier"][t]["cost"] for t in tiers) + 1e-12
        assert all(v == "cheap" for v in rep["oracle_labels"].values())   # scripted model: tiers tie on score


def test_projection_stop_and_cap_stop():
    with tempfile.TemporaryDirectory() as d:
        led = SpendLedger(d, 0.5)
        h = EvoHarness(MockBackend(), "tools", 2, ledger=led,
                       llm=PricedLLM(led, completion_fn=evo_run.scripted_completion))
        rc = evo_run.collect(h, Path(d) / "evoharness_tools_s2.jsonl", 8, ["cheap", "mid", "frontier"],
                             project_n=60, project_after=1, log=lambda *_: None)
        assert rc == evo_run.EXIT_PROJECTED_OVER_CAP
    with tempfile.TemporaryDirectory() as d:
        led = SpendLedger(d, 0.001)
        h = EvoHarness(MockBackend(), "tools", 1, ledger=led,
                       llm=PricedLLM(led, completion_fn=evo_run.scripted_completion))
        rc = evo_run.collect(h, Path(d) / "evoharness_tools_s1.jsonl", 2, ["frontier"], log=lambda *_: None)
        assert rc == evo_run.EXIT_STOPPED_AT_CAP and (Path(d) / "STOPPED_AT_CAP").exists()


def test_service_requires_key(monkeypatch):
    monkeypatch.delenv("EVAL_SERVICE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="MyAuthtoken"):
        evo_run._service_backend()


def test_sdk_backend_lists_stages_splits_and_enforces_allowlist():
    from pace.harness.evoharness import SDKBackend

    class MCP:
        def list_tools(self):
            return [{"name": n, "description": "", "inputSchema": {}} for n in ("a", "b", "distractor_off_stage")]

        def call_tool(self, name, arguments):
            return {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "ok"}], "isError": False}}

        def close(self):
            pass

    class T:
        def __init__(self):
            self.closed, self.started = False, False
            self.system_prompt, self.user_prompt = "POLICY", "do the thing"
            self.resources = {"kind": "tools", "names": ["a", "b"]}
            self.oracle_tools = ["a"]
            self.mcp_servers = [SimpleNamespace(name="gym")]

        def start(self):
            self.started = True

        def mcp_session(self, srv):
            return MCP()

        def grade(self, keep_alive=False):
            return SimpleNamespace(overall_success=False, pass_rate=0.5, n_passed=1, n_total=2)

        def close(self):
            self.closed = True

    class Client:
        def __init__(self):
            self.made = []

        def benchmarks(self):
            return {"datasets": [{"dataset": "evovling_tools", "benchmarks": [{"benchmark": "eog", "domains": [
                {"domain": "hr", "versions": [{"version": 1}, {"version": 2}]},
                {"domain": "enterprise_tri_hybrid", "versions": [{"version": 1}]}]}]}]}

        def task_ids(self, dataset, benchmark, version, split, domain):
            return [f"{split}{i}" for i in range(2 if split == "test" else 1)]

        def task(self, *a, **kw):
            self.made.append(T())
            return self.made[-1]

    be = SDKBackend(client=Client())
    assert be.stages("tools") == {"hr": [1, 2]}                     # non-paper domain excluded
    ts = be.list_tasks("tools", 2)
    assert [t.split for t in ts] == ["heldout", "heldout", "adapt"]
    with tempfile.TemporaryDirectory() as d:
        led = SpendLedger(d, None)
        h = EvoHarness(be, "tools", 2, ledger=led, llm=PricedLLM(led, completion_fn=Script([
            '<action>{"type": "tool", "name": "distractor_off_stage", "arguments": {}}</action>',
            '<action>{"type": "tool", "name": "a", "arguments": {}}</action>',
            '<action>{"type": "answer", "content": "done"}</action>'])))
        s = h.reset(ts[0])
        assert s.extra["catalog"]["tools"] == 2 and s.extra["messages"][0]["content"].startswith("POLICY")
        s, _, _, _ = h.step(s, "ACT@cheap")
        assert s.extra["last_obs"] == "error"                         # off-stage tool refused
        s, _, _, _ = h.step(s, "ACT@cheap")
        assert s.extra["last_obs"] == "ok"
        s, done, _, _ = h.step(s, "ACT@cheap")
        assert done and h.quality(s) == 0.5 and be.client.made[-1].closed


def test_allocate_proportional():
    assert evo_run.allocate({1: 175, 2: 88, 3: 90}, 100) == {1: 50, 2: 25, 3: 25}
    assert sum(evo_run.allocate({1: 7, 2: 3, 3: 1}, 5).values()) == 5
    assert evo_run.allocate({1: 2}, 10) == {1: 2}


# ---------------------------------------------------------------- prompt caching in PricedLLM
def _cached_resp(prompt_tokens, out, cw, cr):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=out,
                            cache_creation_input_tokens=cw, cache_read_input_tokens=cr)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=usage)


def test_cache_breakpoints_mark_system_and_last_message_only():
    from pace.harness.crmarena import with_cache_breakpoints
    msgs = [{"role": "system", "content": "POLICY"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "obs"}]
    out = with_cache_breakpoints(msgs)
    assert out[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert out[3]["content"][-1] == {"type": "text", "text": "obs", "cache_control": {"type": "ephemeral"}}
    assert out[1]["content"] == "q" and out[2]["content"] == "a"
    assert msgs[0]["content"] == "POLICY"                                   # input untouched


def test_cached_call_pricing_and_ledger_fields(monkeypatch):
    import json as _json
    from pace.harness.crmarena import PricedLLM
    from pace.prices import PRICES_PER_MTOK
    pin, pout = PRICES_PER_MTOK["claude-sonnet-5-5"]
    seen = {}

    def fn(**kw):
        seen.update(kw)
        # litellm 1.67 semantics: prompt_tokens = uncached input + cache reads
        return _cached_resp(prompt_tokens=1000 + 20000, out=100, cw=3000, cr=20000)

    with tempfile.TemporaryDirectory() as d:
        led = SpendLedger(d, None)
        llm = PricedLLM(led, completion_fn=fn, prompt_cache=True)
        r = llm.complete("mid", [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], 64, "t", "p")
        want = (1000 * pin + 100 * pout + 3000 * pin * 1.25 + 20000 * pin * 0.10) / 1e6
        assert abs(r.usd - want) < 1e-12 and r.in_tokens == 1000
        assert seen["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        row = _json.loads((Path(d) / "spend.jsonl").read_text().splitlines()[-1])
        assert row["cache_write_tokens"] == 3000 and row["cache_read_tokens"] == 20000 and row["in_tokens"] == 1000


def test_prompt_cache_off_by_default_for_crm_and_env_opt_in(monkeypatch):
    from pace.harness.crmarena import PricedLLM
    monkeypatch.delenv("PACE_PROMPT_CACHE", raising=False)
    seen = {}

    def fn(**kw):
        seen.update(kw)
        return _resp("ok", 500, 10)

    with tempfile.TemporaryDirectory() as d:
        led = SpendLedger(d, None)
        llm = PricedLLM(led, completion_fn=fn)
        assert not llm.prompt_cache
        r = llm.complete("cheap", [{"role": "user", "content": "U"}], 16, "t", "p")
        assert seen["messages"][0]["content"] == "U"                        # unchanged, uncached pricing
        assert abs(r.usd - (500 * 1.0 + 10 * 5.0) / 1e6) < 1e-12
        monkeypatch.setenv("PACE_PROMPT_CACHE", "1")
        assert PricedLLM(led, completion_fn=fn).prompt_cache
        monkeypatch.delenv("PACE_PROMPT_CACHE")
        assert EvoHarness(MockBackend(), "tools", 1, ledger=led).llm.prompt_cache   # on for EvoHarness
