"""CRMArena-Pro harness, spend ledger and resumable collection -- all offline.

litellm is replaced by a scripted fake completion function; Salesforce by a
scripted backend. The evaluate() test loads CRMArena's real
crm_sandbox/env/env.py (if `scripts/setup_crmarena.sh` has fetched it) with
its imports stubbed, so the real Evaluator.evaluate runs against our priced
extractor shim. No network, no API keys, no spend.

    python -m pytest tests -q      |      python tests/test_crmarena.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import tempfile
import types
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pace.collect import EXIT_STOPPED_AT_CAP, FixedArmPolicy, collect_crmarena, run_episode  # noqa: E402
from pace.harness.crmarena import (ACTIONS, FORCE_ANSWER, N_FEATURES, TURN0_IDX, CRMArenaHarness,  # noqa: E402
                                   CRMTask, PricedLLM, build_evaluator)
from pace.prices import call_cost  # noqa: E402
from pace.spend import STOP_MARKER, SpendCap, SpendCapReached, SpendLedger  # noqa: E402
from pace.trace import read_jsonl  # noqa: E402

ENV_PY = os.path.join(ROOT, "third_party", "CRMArena", "crm_sandbox", "env", "env.py")


# ------------------------------------------------------------------ fakes
def _resp(text: str, pin: int, pout: int):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout))


class FakeCompletion:
    """Scripted litellm.completion. `script(kwargs) -> (text, in_tokens, out_tokens)`."""

    def __init__(self, script):
        self.script = script
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        text, pin, pout = self.script(kw)
        return _resp(text, pin, pout)


def three_turn_script(kw):
    """execute -> execute -> respond, keyed off how many assistant turns exist."""
    n_asst = sum(m["role"] == "assistant" for m in kw["messages"])
    last_user = kw["messages"][-1]["content"]
    if "extracted_answers" in kw["messages"][0]["content"]:      # CRMArena extractor prompt
        return '{"extracted_answers": ["005XX"]}', 400, 12
    if FORCE_ANSWER.strip() in last_user:
        return "<respond> 005XX </respond>", 2200, 15
    return [("<thought> find cases </thought> <execute> SELECT Id FROM Case </execute>", 1000, 50),
            ("<thought> narrow </thought> <execute> SELECT OwnerId FROM Case LIMIT 1 </execute>", 1500, 60),
            ("<thought> done </thought> <respond> 005XX </respond>", 2000, 20)][min(n_asst, 2)]


class FakeBackend:
    rule_string = "Invalid output format! Use <execute> or <respond>."

    def __init__(self, evaluator=None):
        self.queries: list[str] = []
        self._evaluator = evaluator

    def system_prompt(self, task, eval_mode="aided", privacy_aware=False):
        return f"SYS for {task.category}"

    def run_query(self, q):
        self.queries.append(q)
        if "LIMIT 1" in q:
            return [{"OwnerId": "005XX"}], 1
        return [{"Id": "500A"}, {"Id": "500B"}], 1

    def make_evaluator(self, shim):
        if self._evaluator is not None:
            return self._evaluator(shim)
        return _ExactEval()


class _ExactEval:
    """Minimal stand-in used where the real Evaluator is not under test."""

    def evaluate(self, proposed, gold, metric, task_name, traj):
        return {"reward": 1 if proposed.strip() == gold[0] else 0, "parsed_answer": [proposed]}


def make_tasks(n=4, metric="exact_match"):
    cats = ["handle_time", "case_routing", "lead_routing", "private_customer_information"]
    return [CRMTask(task_id=f"b2b-{i}", idx=i, category=cats[i % len(cats)], query=f"question {i}?",
                    answer=["005XX"], reward_metric=metric, metadata={}, org_type="b2b") for i in range(n)]


def make_harness(d, cap=None, script=three_turn_script, tasks=None, evaluator=None, **kw):
    ledger = SpendLedger(d, cap)
    fake = FakeCompletion(script)
    llm = PricedLLM(ledger, completion_fn=fake, timeout=30, effort="low")
    h = CRMArenaHarness(org="b2b", ledger=ledger, llm=llm, backend=FakeBackend(evaluator),
                        tasks=tasks or make_tasks(), **kw)
    return h, fake, ledger


# ------------------------------------------------------------------ tests
def test_three_turn_episode_well_formed_and_costed():
    with tempfile.TemporaryDirectory() as d:
        h, fake, ledger = make_harness(d)
        task = h.tasks()[0]
        pol = FixedArmPolicy("cheap", h.all_actions, 3, TURN0_IDX)
        ep = run_episode(h, pol, task, random.Random(0), log_counterfactual=False, log_state=False)

        assert len(ep.steps) == 3 and [s.action_name for s in ep.steps] == ["CONTINUE@cheap"] * 3
        assert [round(s.propensity, 6) for s in ep.steps] == [round(1 / 3, 6), 1.0, 1.0]
        assert all(len(s.features) == N_FEATURES for s in ep.steps)
        assert ep.steps[0].legal == ACTIONS
        expected = [call_cost("claude-haiku-4-5", 1000, 50), call_cost("claude-haiku-4-5", 1500, 60),
                    call_cost("claude-haiku-4-5", 2000, 20)]
        assert expected[0] == (1000 * 1 + 50 * 5) / 1e6
        for s, e in zip(ep.steps, expected):
            assert abs(s.cost - e) < 1e-12, (s.cost, e)
        assert abs(ep.total_cost - sum(expected)) < 1e-12
        assert ep.total_latency >= 0 and all(s.latency >= 0 for s in ep.steps)
        assert ep.quality == 1.0
        assert ep.meta["end_reason"] == "agent_respond" and ep.meta["tool_calls"] == 2
        assert ep.meta["answer"] == "005XX" and ep.meta["tiers"] == ["cheap"] * 3
        # step-1 features saw a tool call that returned rows
        names = __import__("pace.harness.crmarena", fromlist=["FEATURE_NAMES"]).FEATURE_NAMES
        assert ep.steps[1].features[names.index("last_rows")] == 1.0
        assert ep.steps[1].features[names.index("turn=1")] == 1.0
        # litellm kwargs: Anthropic provider string, bounded max_tokens, timeout, no sampling params
        k = fake.calls[0]
        assert k["model"] == "anthropic/claude-haiku-4-5" and k["max_tokens"] == 2048 and k["timeout"] == 30
        assert "temperature" not in k and "top_p" not in k and "output_config" not in k
        assert fake.calls[1]["messages"][-1]["content"].startswith("Salesforce instance output: [{'Id': '500A'}")
        # ledger: one line per call, totals agree
        lines = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl"))]
        assert len(lines) == 3 and {x["purpose"] for x in lines} == {"agent:cheap"}
        assert abs(sum(x["usd"] for x in lines) - ledger.total) < 1e-9
        tot = json.load(open(os.path.join(d, "spend_total.json")))
        assert abs(tot["total_usd"] - ledger.total) < 1e-6 and tot["n_calls"] == 3


def test_answer_op_forces_final_answer_on_chosen_tier():
    with tempfile.TemporaryDirectory() as d:
        h, fake, _ = make_harness(d)
        s = h.reset(h.tasks()[0])
        s, done, c1, _ = h.step(s, "CONTINUE@mid")
        assert not done and fake.calls[-1]["output_config"] == {"effort": "low"}
        s, done, c2, _ = h.step(s, "ANSWER@frontier")
        assert done and s.extra["answer"] == "005XX" and s.extra["end_reason"] == "forced_answer"
        k = fake.calls[-1]
        assert k["model"] == "anthropic/claude-opus-5-5" and k["max_tokens"] == 1024
        assert k["messages"][-1]["content"].endswith(FORCE_ANSWER)
        assert not s.extra["messages"][-2]["content"].endswith(FORCE_ANSWER)   # log keeps the clean turn
        assert abs(c1 - call_cost("claude-sonnet-5-5", 1000, 50)) < 1e-12
        assert abs(c2 - call_cost("claude-opus-5-5", 2200, 15)) < 1e-12


def test_max_turns_forces_answer_on_last_turn():
    def never_answers(kw):
        if FORCE_ANSWER.strip() in kw["messages"][-1]["content"]:
            return "<respond> 005XX </respond>", 100, 5
        return "<execute> SELECT Id FROM Case </execute>", 100, 5
    with tempfile.TemporaryDirectory() as d:
        h, _, _ = make_harness(d, script=never_answers, max_turns=3)
        ep = run_episode(h, FixedArmPolicy("cheap", h.all_actions, 3, TURN0_IDX), h.tasks()[0],
                         random.Random(0), log_counterfactual=False, log_state=False)
        assert [s.action_name for s in ep.steps] == ["CONTINUE@cheap", "CONTINUE@cheap", "ANSWER@cheap"]
        assert ep.steps[-1].legal == [a for a in ACTIONS if a.startswith("ANSWER@")]
        assert ep.meta["end_reason"] == "forced_answer"


def test_spend_cap_unit():
    cap = SpendCap(1.0)
    cap.check(0.5, 0.5)                      # exactly at the cap is allowed
    try:
        cap.check(0.5, 0.5000001)
        raise AssertionError("expected SpendCapReached")
    except SpendCapReached as e:
        assert e.cap == 1.0
    SpendCap(None).check(1e9, 1e9)           # uncapped


def test_spend_cap_stops_before_exceeding_and_writes_marker():
    with tempfile.TemporaryDirectory() as d:
        cap = 0.02          # worst case per Haiku agent call ~ 2048 x $5/M = $0.0102 + input
        h, fake, ledger = make_harness(d, cap=cap)
        out = os.path.join(d, "crmarena_b2b_counterfactual.jsonl")
        logs: list[str] = []
        rc = collect_crmarena(h, out, n_tasks=4, mode="counterfactual", tiers=["cheap"], log=logs.append)
        assert rc == EXIT_STOPPED_AT_CAP
        marker = os.path.join(d, STOP_MARKER)
        assert os.path.exists(marker)
        m = json.load(open(marker))
        assert m["cap_usd"] == cap and abs(m["total_usd"] - ledger.total) < 1e-6
        assert ledger.total <= cap
        # the call that would have crossed the cap was never sent
        assert len(fake.calls) == ledger.n_calls
        assert m["total_usd"] + m["next_call_worst_case_usd"] > cap
        lines = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl"))]
        assert abs(sum(x["usd"] for x in lines) - m["total_usd"]) < 1e-6
        assert any("STOPPED AT CAP" in x for x in logs)
        n_done = len(read_jsonl(out))
        assert 0 < n_done < 4                 # some episodes finished, the run was cut short


def test_spend_cap_via_cli_exit_code():
    from pace import collect as C
    with tempfile.TemporaryDirectory() as d:
        os.environ.pop("ANTHROPIC_API_KEY", None)        # the at-cap exit must not need a key
        led = SpendLedger(d, None)
        led.record("claude-haiku-4-5", 10_000, 0, "b2b-0", "agent:cheap")    # $0.01 already spent
        try:
            C.main(["--harness", "crmarena", "--n-tasks", "1", "--cap", "0.005", "--traces-dir", d,
                    "--env-file", os.path.join(d, "none.env")])
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert e.code == EXIT_STOPPED_AT_CAP
        assert os.path.exists(os.path.join(d, STOP_MARKER))


def test_llm_failure_aborts_run_and_rejected_calls_cost_nothing():
    from pace.collect import EXIT_LLM_FAILED

    class BadRequest(Exception):
        status_code = 400

    def rejects_frontier(kw):
        if kw["model"].endswith("opus-5-5"):
            raise BadRequest("output_config: unsupported")
        return three_turn_script(kw)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "t.jsonl")
        h, fake, ledger = make_harness(d, cap=10.0, script=rejects_frontier)
        rc = collect_crmarena(h, out, 3, "counterfactual", log=lambda *_: None)
        assert rc == EXIT_LLM_FAILED
        eps = read_jsonl(out)
        assert [e.meta["arm"] for e in eps] == ["cheap", "mid"]          # stopped at the first failure
        err = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl")) if ":error" in x]
        assert len(err) == 1 and err[0]["usd"] == 0.0 and err[0]["status"] == 400
        assert len([k for k in fake.calls if k["model"].endswith("opus-5-5")]) == 1


def test_collect_resumes_without_rerunning_logged_tasks():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "crmarena_b2b_counterfactual.jsonl")
        h, fake, _ = make_harness(d, cap=10.0)
        assert collect_crmarena(h, out, n_tasks=2, mode="counterfactual", log=lambda *_: None) == 0
        eps = read_jsonl(out)
        first2 = {t.task_id for t in h.sample_tasks(2)}
        assert len(eps) == 6 and {(e.task_id, e.meta["arm"]) for e in eps} == {
            (t, a) for t in first2 for a in ("cheap", "mid", "frontier")}
        n_calls = len(fake.calls)
        # same request again: nothing new is run
        h2, fake2, _ = make_harness(d, cap=10.0)
        assert collect_crmarena(h2, out, n_tasks=2, mode="counterfactual", log=lambda *_: None) == 0
        assert len(fake2.calls) == 0 and len(read_jsonl(out)) == 6
        # grow to 3 tasks: only the new task's three arms run
        h3, fake3, _ = make_harness(d, cap=10.0)
        assert collect_crmarena(h3, out, n_tasks=3, mode="counterfactual", log=lambda *_: None) == 0
        eps = read_jsonl(out)
        assert len(eps) == 9 and len({(e.task_id, e.meta["arm"]) for e in eps}) == 9
        assert len(fake3.calls) == n_calls // 2


def test_resume_after_cap_completes_without_duplicates():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "t.jsonl")
        h, _, _ = make_harness(d, cap=0.02)
        assert collect_crmarena(h, out, 3, "counterfactual", ["cheap"], log=lambda *_: None) == EXIT_STOPPED_AT_CAP
        before = len(read_jsonl(out))
        h2, _, led2 = make_harness(d, cap=1.0)       # cumulative ledger: resumes with the old spend
        assert led2.total > 0
        assert collect_crmarena(h2, out, 3, "counterfactual", ["cheap"], log=lambda *_: None) == 0
        eps = read_jsonl(out)
        assert len(eps) == 3 > before and len({e.task_id for e in eps}) == 3
        assert not os.path.exists(os.path.join(d, STOP_MARKER))


def test_policy_mode_logs_eps_propensities():
    from pace.controller import Controller
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "p.jsonl")
        h, _, _ = make_harness(d, cap=10.0)
        pol = Controller.uniform(h.all_actions, h.n_features)
        pol.eps = 0.2
        assert collect_crmarena(h, out, 2, "policy", policy=pol, eps=0.2, log=lambda *_: None) == 0
        for e in read_jsonl(out):
            assert e.meta["arm"] == "policy" and e.meta["eps"] == 0.2
            for s in e.steps:
                assert abs(s.propensity - 1 / len(s.legal)) < 1e-9   # uniform incl. eps mass


# ------------------------------------------------- real CRMArena evaluate()
def _load_crmarena_env_module():
    """Import CRMArena's real env.py with its third-party imports stubbed (no network)."""
    stub_names = ["litellm", "crm_sandbox", "crm_sandbox.env", "crm_sandbox.env.connect_sandbox",
                  "crm_sandbox.env.users", "crm_sandbox.agents", "crm_sandbox.agents.utils"]
    saved = {n: sys.modules.get(n) for n in stub_names}
    try:
        for n in stub_names:
            sys.modules[n] = types.ModuleType(n)

        def _no_network(*a, **k):
            raise AssertionError("real litellm must never be called in tests")
        sys.modules["litellm"].completion = _no_network
        sys.modules["crm_sandbox.env.connect_sandbox"].SalesforceConnector = object
        sys.modules["crm_sandbox.env.users"].LLMUserSimulationEnv = object
        sys.modules["crm_sandbox.agents.utils"].get_all_metrics = lambda p, g: {"f1": float(p == g)}
        spec = importlib.util.spec_from_file_location("crmarena_env_under_test", ENV_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def test_evaluate_integration_with_stubbed_extractor():
    if not os.path.exists(ENV_PY):
        print("SKIP test_evaluate_integration_with_stubbed_extractor: run scripts/setup_crmarena.sh first")
        return
    envmod = _load_crmarena_env_module()
    assert envmod.litellm.__name__ == "litellm"                  # still the stub until patched

    def script(kw):
        sysmsg = kw["messages"][0]["content"]
        if "properly refused" in sysmsg:                       # CRMArena privacy judge prompt
            return "YES", 300, 2
        if "extracted_answers" in sysmsg:                      # CRMArena extractor prompt
            body = kw["messages"][1]["content"]
            return ("['005XX']" if "heuristic" in body else '{"extracted_answers": ["005XX"]}'), 450, 12
        return three_turn_script(kw)

    with tempfile.TemporaryDirectory() as d:
        h, fake, ledger = make_harness(d, script=script, evaluator=lambda shim: build_evaluator(envmod, shim))
        task = h.tasks()[0]
        # (a) the exact-string fast path: no extractor call
        s = h.reset(task)
        s.extra["answer"], s.extra["actions"] = "005XX", ["005XX"]
        n = len(fake.calls)
        assert h.quality(s) == 1.0 and len(fake.calls) == n
        # (b) sentence answer -> CRMArena's LLM extractor, routed to the cheap tier and priced
        s = h.reset(task)
        s.extra["answer"], s.extra["actions"] = "The owner is 005XX.", ["The owner is 005XX."]
        assert h.quality(s) == 1.0
        k = fake.calls[-1]
        assert k["model"] == "anthropic/claude-haiku-4-5" and k["max_tokens"] == 256
        ev = s.extra["eval"]
        assert ev["eval_calls"] == 1 and abs(ev["eval_cost_usd"] - call_cost("claude-haiku-4-5", 450, 12)) < 1e-12
        last = json.loads(open(os.path.join(d, "spend.jsonl")).read().splitlines()[-1])
        assert last["purpose"] == "extractor" and last["model"] == "claude-haiku-4-5"
        assert last["task_id"] == task.task_id
        # (c) non-JSON extractor output -> env.py's regex fallback (needs the `re` monkeypatch)
        s = h.reset(task)
        s.extra["answer"], s.extra["actions"] = "heuristic: owner 005XX", ["x"]
        assert h.quality(s) == 1.0
        # (d) privacy_rejection -> CRMArena's privacy judge on the cheap tier
        ptask = make_tasks(1, metric="privacy_rejection")[0]
        s = h.reset(ptask)
        s.extra["answer"], s.extra["actions"] = "I can't share that.", ["I can't share that."]
        assert h.quality(s) == 1.0
        last = json.loads(open(os.path.join(d, "spend.jsonl")).read().splitlines()[-1])
        assert last["purpose"] == "privacy_judge"
        # full episode: extractor spend is in the ledger and meta, not in the controller's episode cost
        ep = run_episode(h, FixedArmPolicy("cheap", h.all_actions, 3, TURN0_IDX), task, random.Random(0),
                         log_counterfactual=False, log_state=False)
        assert ep.quality == 1.0 and abs(ep.total_cost - sum(x.cost for x in ep.steps)) < 1e-12


def test_litellm_request_shape_offline():
    """Real litellm request transformation with a fake HTTP client (only where litellm is installed,
    e.g. the .venv-crmarena venv): Anthropic provider, bounded max_tokens, effort passthrough for the
    thinking tiers, no sampling params, usage mapped back into the ledger."""
    try:
        import functools

        import httpx
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
    except ImportError:
        print("SKIP test_litellm_request_shape_offline: litellm not installed")
        return
    bodies = []

    class FakeHTTP(HTTPHandler):
        def post(self, url, data=None, json=None, **kw):
            import json as J
            body = J.loads(data) if data else json
            bodies.append((url, body))
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "id": "msg_t", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "thinking", "thinking": "", "signature": "s"},
                            {"type": "text", "text": "<execute> SELECT Id FROM Case </execute>"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1234, "output_tokens": 56}})
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-not-a-real-key"
    try:
        with tempfile.TemporaryDirectory() as d:
            ledger = SpendLedger(d, 1.0)
            llm = PricedLLM(ledger, completion_fn=functools.partial(litellm.completion, client=FakeHTTP()),
                            num_retries=0)
            msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
            r1 = llm.complete("cheap", msgs, 2048, "b2b-0", "agent:cheap")
            r2 = llm.complete("frontier", msgs, 2048, "b2b-0", "agent:frontier")
    finally:
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key
    (u1, b1), (u2, b2) = bodies
    assert u1 == "https://api.anthropic.com/v1/messages"
    assert b1["model"] == "claude-haiku-4-5" and b1["max_tokens"] == 2048 and "output_config" not in b1
    assert b2["model"] == "claude-opus-5-5" and b2["output_config"] == {"effort": "low"}
    assert all("temperature" not in b and "top_p" not in b for b in (b1, b2))
    assert r1.text.startswith("<execute>") and (r1.in_tokens, r1.out_tokens) == (1234, 56)
    assert abs(r2.usd - call_cost("claude-opus-5-5", 1234, 56)) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
