"""Process-verified MAS experiment (pace/experiments/mas), all offline: the fake LLM and fake
Salesforce from the package, a temp ledger, no network, no keys, no spend.

    python -m pytest tests/test_mas.py -q      |      python tests/test_mas.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pace.controller import Controller  # noqa: E402
from pace.experiments.mas import run as cli  # noqa: E402
from pace.experiments.mas.fake_backend import FakeBackend, FakeCompletion  # noqa: E402
from pace.experiments.mas.orchestrator import (ACTIONS, FEATURE_NAMES, N_FEATURES, EpisodeState,  # noqa: E402
                                                Orchestrator, make_policy, uniform_controller)
from pace.experiments.mas.orm import (DEFAULT_DROP, QualityModel, cross_validate, fit_quality_model,  # noqa: E402
                                      relabel_episodes)
from pace.experiments.mas.process_verify import (classify, process_reward_add, process_reward_mult,  # noqa: E402
                                                 silent_failure_analysis)
from pace.experiments.mas.roles import (PURPOSE_PLANNER, PURPOSE_VERIFIER, Plan, Planner, RoleCall,  # noqa: E402
                                        Verifier, WorkerLLM, worker_purpose)
from pace.gate import estimate  # noqa: E402
from pace.harness.crmarena import CRMArenaHarness, PricedLLM  # noqa: E402
from pace.prices import MODELS, call_cost  # noqa: E402
from pace.spend import SpendLedger  # noqa: E402
from pace.trace import Episode, TraceStep, read_jsonl  # noqa: E402
from pace.train_offline import train_awr  # noqa: E402


def make_harness(d, cap=None, seed=0, n_tasks=66):
    ledger = SpendLedger(d, cap)
    backend = FakeBackend("b2b", n_tasks, seed)
    fake = FakeCompletion(backend, seed)
    h = CRMArenaHarness(org="b2b", ledger=ledger, llm=WorkerLLM(PricedLLM(ledger, completion_fn=fake)), backend=backend)
    return h, ledger, fake


# ------------------------------------------------------------------ roles
def test_worker_llm_rewrites_only_agent_purposes():
    assert WorkerLLM.rewrite("agent:mid") == "mas:worker:mid"
    assert WorkerLLM.rewrite("agent:frontier:answer") == "mas:worker:frontier:answer"
    assert WorkerLLM.rewrite("agent:cheap:error") == "mas:worker:cheap:error"
    assert WorkerLLM.rewrite("extractor") == "extractor" and WorkerLLM.rewrite("privacy_judge") == "privacy_judge"
    assert worker_purpose("mid") == "mas:worker:mid" and worker_purpose("mid", "answer") == "mas:worker:mid:answer"


def test_planner_and_verifier_parsing():
    p = Planner.parse('{"subqueries": [{"goal": "a", "soql": "SELECT Id FROM Case"}, {"goal": "b"}], "rationale": "r"}', "q?")
    assert [s.goal for s in p.sub_queries] == ["a", "b"] and p.sub_queries[1].soql == "" and p.parse_ok
    p = Planner.parse("SELECT Id FROM Case\nSELECT Id FROM Lead", "q?")
    assert len(p.sub_queries) == 2 and not p.parse_ok
    p = Planner.parse("no plan here", "the question?")
    assert len(p.sub_queries) == 1 and p.sub_queries[0].goal == "the question?" and not p.parse_ok
    p = Planner.parse(json.dumps({"subqueries": [{"goal": str(i)} for i in range(6)]}), "q")
    assert len(p.sub_queries) == 3                       # capped at MAX_SUBQUERIES
    assert Verifier.parse('{"pass": true, "reason": "ok"}').passed
    assert not Verifier.parse('{"pass": "no", "reason": "x"}').passed
    assert not Verifier.parse("FAIL: rows are wrong").passed and Verifier.parse("pass").passed
    v = Verifier.parse("hmm")
    assert not v.passed and "unparsable" in v.reason
    w = RoleCall("worker", "mid", "mas:worker:mid", "g", "error: q", "error", "", 0.001, 0.0)
    assert Verifier.structural(w) is not None and not Verifier.structural(w).passed


def test_priced_role_calls_hit_the_ledger():
    with tempfile.TemporaryDirectory() as d:
        h, ledger, fake = make_harness(d)
        orc = Orchestrator(h, "cheap", "mid", "cheap", "mid")
        task = h.sample_tasks(1, 0)[0]
        ep = orc.run(task, make_policy("always_verify", "mid"), random.Random(0))
        proc = ep.meta["process"]
        roles = [c["role"] for c in proc]
        assert roles[0] == "planner" and roles[1] == "worker" and roles[-1] == "answer"
        assert proc[0]["purpose"] == PURPOSE_PLANNER and proc[0]["tier"] == "cheap"
        assert all(c["purpose"] == PURPOSE_VERIFIER and c["tier"] == "cheap" for c in proc if c["role"] == "verifier")
        assert all(c["purpose"] == "mas:worker:mid" for c in proc if c["role"] == "worker")
        assert proc[-1]["purpose"] == "mas:worker:mid:answer"
        # every priced call: cost == list price of its logged tokens, and it is one ledger line
        lines = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl"))]
        purposes = {x["purpose"] for x in lines}
        assert purposes == {PURPOSE_PLANNER, PURPOSE_VERIFIER, "mas:worker:mid", "mas:worker:mid:answer", "extractor"}
        priced = [c for c in proc if c["in_tokens"] > 0]
        for c in priced:
            model = MODELS[c["tier"]]
            assert abs(c["cost"] - call_cost(model, c["in_tokens"], c["out_tokens"])) < 1e-12
        structural = [c for c in proc if c["role"] == "verifier" and c["extra"]["checked_by"] == "structural"]
        assert all(c["cost"] == 0.0 and c["in_tokens"] == 0 for c in structural)
        assert len(lines) == len(priced) + ep.meta["eval_calls"]
        assert abs(ep.total_cost - sum(c["cost"] for c in proc)) < 1e-12
        assert abs(ledger.total - ep.total_cost - ep.meta["eval_cost_usd"]) < 1e-9
        assert ep.meta["eval_calls"] == 1 and ep.meta["eval_cost_usd"] > 0     # outcome via the priced extractor path
        assert ep.meta["harness"] == "mas-crmarena-pro-v1" and ep.meta["host_harness"] == "crmarena-pro-v1"
        assert abs(ep.meta["prefix_cost"] + sum(s.cost for s in ep.steps) - ep.total_cost) < 1e-12


# ------------------------------------------------------- decision points
def test_decision_points_legal_sets_and_call_cap():
    with tempfile.TemporaryDirectory() as d:
        h, _, _ = make_harness(d)
        orc = Orchestrator(h)
        pol = uniform_controller(eps=0.2)
        rng = random.Random(1)
        n_fail = n_replan = 0
        for task in h.sample_tasks(30, 3):
            ep = orc.run(task, pol, rng, eps=0.2)
            assert 3 <= ep.meta["n_calls"] <= 8, ep.meta["n_calls"]
            assert ep.steps and ep.steps[-1].action_name.startswith("ANSWER@") or ep.meta["answer_by"] == "worker_respond"
            for s in ep.steps:
                assert s.action_name in s.legal and 0 < s.propensity <= 1 and len(s.features) == N_FEATURES
                assert s.features[FEATURE_NAMES.index("v2*bias(reserved)")] == 0.0
                if s.state["n_calls_before"] == 7:
                    assert all(a.startswith("ANSWER@") for a in s.legal)
            n_fail += ep.meta["n_verify_fail"]; n_replan += ep.meta["n_replan"]
        assert n_fail > 0 and n_replan > 0
        # with 3 calls (planner + worker + answer) only ANSWER@* is ever legal
        orc3 = Orchestrator(h, max_calls=3)
        ep = orc3.run(h.sample_tasks(1, 0)[0], make_policy("always_verify"), random.Random(0))
        assert [s.action_name for s in ep.steps] == ["ANSWER@mid"] and ep.steps[0].legal == [f"ANSWER@{t}" for t in ("cheap", "mid", "frontier")]
        # legal set unit: a fresh state after one worker step
        st = EpisodeState(task=h.sample_tasks(1, 0)[0], hs=None, plan=Plan([Plan, Plan]))   # two sub-queries
        st.last_worker = RoleCall("worker", "mid", "mas:worker:mid", "", "", "rows", "", 0.0, 0.0, sub_query=0)
        st.calls = [st.last_worker, st.last_worker]
        legal = orc.legal_actions(st)
        assert {a.split("@")[0] for a in legal} == {"VERIFY", "SKIP_VERIFY", "REPLAN", "ANSWER"}
        st.last_verdict = "pass"
        assert "VERIFY@mid" not in orc.legal_actions(st)
        st.i = 1                                                           # plan exhausted
        assert "SKIP_VERIFY@mid" not in orc.legal_actions(st)
        st.n_replan = 1
        assert orc.legal_actions(st) == [f"ANSWER@{t}" for t in ("cheap", "mid", "frontier")]
        # an illegal choice is refused
        class Bad:
            actions = ACTIONS; meta = {"name": "bad"}
            def act(self, x, legal, rng=None, eps=None):
                return ACTIONS.index("REPLAN@mid"), 1.0
        try:
            orc3.run(h.sample_tasks(1, 0)[0], Bad(), random.Random(0)); raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "illegal" in str(e)


def test_fixed_baselines_do_what_they_say():
    with tempfile.TemporaryDirectory() as d:
        h, _, _ = make_harness(d)
        orc = Orchestrator(h)
        for task in h.sample_tasks(6, 0):
            a = orc.run(task, make_policy("always_verify"), random.Random(0))
            n = orc.run(task, make_policy("never_verify"), random.Random(0))
            assert n.meta["n_verify"] == 0 and all(s.action_name.split("@")[0] in ("SKIP_VERIFY", "ANSWER") for s in n.steps)
            assert all(s.action_name.split("@")[0] in ("VERIFY", "ANSWER") for s in a.steps)
            # every worker step is verified unless the call cap or an early worker <respond> cut it
            if a.meta["answer_by"] != "worker_respond" and a.meta["n_calls"] < 8:
                assert a.meta["n_verify"] == sum(1 for c in a.meta["process"] if c["role"] == "worker")


# --------------------------------------------------------- silent failures
def _ep(q, verdicts, retried=(), n_worker=None, tid="t"):
    """verdicts: list of (sub_query, pass|fail) in order; n_worker defaults to the distinct sub-queries."""
    subs = sorted({s for s, _ in verdicts} | set(range(n_worker or 0)))
    proc = []
    for s in subs:
        proc.append({"role": "worker", "tier": "mid", "purpose": "mas:worker:mid", "input_summary": "", "output_summary": "",
                     "verdict": "rows", "reason": "", "cost": 0.01, "latency": 0, "in_tokens": 1, "out_tokens": 1,
                     "sub_query": s, "retry": s in retried, "extra": {}})
    for s, v in verdicts:
        proc.append({"role": "verifier", "tier": "cheap", "purpose": "mas:verifier", "input_summary": "", "output_summary": "",
                     "verdict": v, "reason": "", "cost": 0.001, "latency": 0, "in_tokens": 1, "out_tokens": 1,
                     "sub_query": s, "retry": False, "extra": {}})
    return Episode(tid, [], q, 0.05, 0.0, {"process": proc, "category": "case_routing", "mode": "m"})


def test_silent_failure_logic_on_hand_built_episodes():
    e_ok_pass = _ep(1.0, [(0, "pass"), (1, "pass")])
    e_silent = _ep(0.0, [(0, "pass"), (1, "pass")])
    e_caught = _ep(0.0, [(0, "fail")])
    e_recov = _ep(1.0, [(0, "fail"), (0, "pass")], retried=(0,))
    e_open = _ep(1.0, [(0, "fail"), (1, "pass")])
    e_unver_wrong = _ep(0.0, [], n_worker=2)
    e_unver_ok = _ep(1.0, [], n_worker=1)
    assert classify(e_ok_pass) == "correct_all_pass" and classify(e_silent) == "wrong_all_pass"
    assert classify(e_caught) == "wrong_some_fail" and classify(e_recov) == "correct_some_fail_recovered"
    assert classify(e_open) == "correct_some_fail_open"
    assert classify(e_unver_wrong) == "wrong_unverified" and classify(e_unver_ok) == "correct_unverified"
    # rewards
    assert process_reward_mult(e_ok_pass) == 1.0 and process_reward_mult(e_open) == 0.5 and process_reward_mult(e_recov) == 1.0
    assert process_reward_mult(e_unver_ok) == 0.0                            # unverified steps are not covered
    assert abs(process_reward_add(e_open) - (1.0 + 0.25 * 0.5 - 0.25 * 0.5)) < 1e-12
    assert abs(process_reward_add(e_silent) - (0.0 + 0.25 * 1.0)) < 1e-12
    a = silent_failure_analysis([e_ok_pass, e_silent, e_caught, e_recov, e_open, e_unver_wrong, e_unver_ok])
    assert a["n"] == 7 and a["n_verified_episodes"] == 5
    assert a["cells"]["wrong_all_pass"] == 1 and a["cells"]["wrong_some_fail"] == 1 and a["cells"]["correct_some_fail_open"] == 1
    assert abs(a["silent_failure_rate"] - 1 / 5) < 1e-12 and abs(a["caught_rate"] - 1 / 5) < 1e-12
    assert abs(a["correct_despite_fail_rate"] - 2 / 5) < 1e-12
    assert abs(a["p_flagged_given_wrong"] - 1 / 2) < 1e-12 and abs(a["p_wrong_given_all_pass"] - 1 / 2) < 1e-12
    assert a["verifier_calls"] == 9 and abs(a["verifier_pass_rate"] - 6 / 9) < 1e-12
    assert abs(a["accuracy"] - 4 / 7) < 1e-12


# -------------------------------------------------------------------- ORM
def _fake_log(d, n_tasks=20, seed=0):
    h, _, _ = make_harness(d, seed=seed)
    orc = Orchestrator(h)
    eps = []
    for mode in ("always_verify", "never_verify", "controller"):
        pol = make_policy(mode, eps=0.2)
        rng = random.Random(seed)
        for t in h.sample_tasks(n_tasks, seed):
            ep = orc.run(t, pol, rng, 0.2 if mode == "controller" else None)
            ep.meta["mode"] = mode if mode != "controller" else "controller:uniform"
            eps.append(ep)
    return eps


def test_orm_trains_exports_and_the_trainer_and_gate_accept_it():
    with tempfile.TemporaryDirectory() as d:
        eps = _fake_log(d)
        assert len(eps) == 60
        cv = cross_validate(eps, "logreg", 5, 0, DEFAULT_DROP)
        assert cv["n"] == 60 and 0.0 <= cv["auroc"] <= 1.0 and 0.0 <= cv["brier"] <= 1.0 and 0 <= cv["ece"] <= 1
        assert sum(c["n"] for c in cv["calibration"]) == 60
        cv2 = cross_validate(eps, "mlp", 5, 0, DEFAULT_DROP)
        assert 0.0 <= cv2["auroc"] <= 1.0
        qm = fit_quality_model(eps, "logreg")
        assert not any(n.startswith("cat=") for n in qm.feature_names)
        p = qm.save(Path(d) / "quality_model.json")
        qm2 = QualityModel.load(p)
        js = json.loads(p.read_text())
        assert js["kind"] == "mas_orm" and js["model"]["type"] == "logreg"
        for e in eps[:10]:
            assert abs(qm.predict_episode(e) - qm2.predict_episode(e)) < 1e-12 and 0 <= qm.predict_episode(e) <= 1
        # prefix prediction (value estimate at a decision point) runs and differs from the terminal one
        e = next(e for e in eps if e.meta["n_calls"] >= 4)
        assert qm.predict_episode(e, upto=2) != qm.predict_episode(e)
        # consumer 1: relabel -> AWR on reward quality_hat - lam*cost (the core trainer, unchanged)
        explore = [e for e in eps if e.meta["mode"].startswith("controller")]
        rel = relabel_episodes(explore, qm)
        assert all(0 <= r.quality <= 1 and r.meta["quality_source"] == "mas_orm" and r.meta["quality_outcome"] in (0.0, 1.0) for r in rel)
        pol = train_awr(rel, ACTIONS, lam=2.0, epochs=20, n_features=N_FEATURES)
        assert isinstance(pol, Controller) and pol.W.shape == (len(ACTIONS), N_FEATURES) and np.isfinite(pol.W).all()
        # consumer 2: the core gate's estimators on the relabelled log
        est = estimate(rel, pol, boot=20)
        assert set(est["quality"]) == {"ips", "snips", "dr"} and np.isfinite(est["quality"]["dr"][0])
        # the trained policy serves on the orchestrator
        h, _, _ = make_harness(d, seed=5)
        ep = Orchestrator(h).run(h.sample_tasks(1, 0)[0], pol, random.Random(0), eps=0.05)
        assert ep.meta["n_calls"] <= 8


def test_offpolicy_per_decision_and_episode_level_with_ess():
    from pace.experiments.mas.offpolicy import STEP_CLIP, episode_level, ess, ess_warning, per_decision
    with tempfile.TemporaryDirectory() as d:
        eps = [e for e in _fake_log(d, n_tasks=10) if e.meta["mode"].startswith("controller")]
        beh = uniform_controller(eps=0.2)                      # the behavior policy itself
        nd = sum(len(e.steps) for e in eps)
        pd = per_decision(eps, beh, "quality", boot=20)
        assert pd["n"] == nd and abs(pd["ess"] - nd) < 1e-9 and abs(pd["max_ratio"] - 1.0) < 1e-9
        assert abs(pd["snips"][0] - pd["behavior"]) < 1e-9                       # SNIPS of the behavior is its mean
        assert abs(pd["dr"][0] - pd["behavior"]) < 0.05 and pd["dr"][1] <= pd["behavior"] <= pd["dr"][2]   # DR: unbiased, not exact
        el = episode_level(eps, beh, "quality", boot=20)
        assert el["n"] == len(eps) and abs(el["ess"] - len(eps)) < 1e-9 and abs(el["snips"][0] - el["behavior"]) < 1e-9
        assert "below 10%" not in ess_warning(pd)
        # a far-off candidate: ratios bounded by |legal|/eps per step, clipped per step at STEP_CLIP in the episode estimator
        far = Controller(ACTIONS, N_FEATURES, eps=0.0); far.W[ACTIONS.index("ANSWER@frontier"), 0] = 50.0
        pd2 = per_decision(eps, far, "cost_to_go", boot=20)
        assert pd2["max_ratio"] <= 12 / 0.2 + 1e-9 and pd2["ess"] < pd["ess"] and 0 <= pd2["snips"][0]
        el2 = episode_level(eps, far, "cost", boot=20)
        assert el2["max_ratio"] <= STEP_CLIP ** 8 + 1e-9 and el2["ess"] <= el["ess"]
        assert ess(np.array([1.0, 0.0, 0.0])) == 1.0 and abs(ess(np.ones(7)) - 7) < 1e-12
        assert "unreliable" in ess_warning({"ess": 0.5, "n": 100})


def test_paired_table_counts_flips():
    from pace.experiments.mas.run import paired_table
    A = [_ep(1.0, [(0, "pass")], tid="t1"), _ep(1.0, [(0, "fail")], tid="t2"), _ep(0.0, [(0, "fail")], tid="t3"), _ep(0.0, [], n_worker=1, tid="t4")]
    B = [_ep(1.0, [], n_worker=1, tid="t1"), _ep(0.0, [], n_worker=1, tid="t2"), _ep(1.0, [], n_worker=1, tid="t3"), _ep(0.0, [], n_worker=1, tid="t4"),
         _ep(1.0, [], n_worker=1, tid="t5")]                                  # t5 only in B: not paired
    for e in A: e.meta["mode"] = "always_verify"; e.meta["n_verify"] = 1
    for e in B: e.meta["mode"] = "never_verify"
    txt, st = paired_table({"always_verify": A, "never_verify": B})
    assert st["n"] == 4 and st["both_right"] == 1 and st["both_wrong"] == 1
    assert st["always_verify_only"] == 1 and st["never_verify_only"] == 1 and st["net"] == 0
    assert abs(st["dq_mean"]) < 1e-12 and "wrong → right" in txt and "`t2`" in txt and "`t3`" in txt


def _single_agent_log(task_ids, cats):
    """A counterfactual single-agent log (3 tiers per task) in pace.lookup's format."""
    from pace.harness.crmarena import CATEGORIES, FEATURE_NAMES as CRM_F, ORG_TYPES, N_FEATURES as CRM_N
    out = []
    for t, c in zip(task_ids, cats):
        x = [0.0] * CRM_N; x[0] = 1.0; x[1 + CATEGORIES.index(c)] = 1.0; x[1 + len(CATEGORIES) + ORG_TYPES.index("b2b")] = 1.0
        x[CRM_F.index("turn=0")] = 1.0
        for j, (tier, q, cost) in enumerate((("cheap", 0.0, 0.02), ("mid", 1.0, 0.05), ("frontier", 1.0, 0.10))):
            st = TraceStep(t, "crmarena-pro-v1", 0, x, j, f"CONTINUE@{tier}", 1 / 3, cost, 0.0, legal=[f"CONTINUE@{k}" for k in ("cheap", "mid", "frontier")])
            out.append(Episode(t, [st], q, cost, 0.0, {"arm": tier, "category": c}))
    return out


def test_single_agent_side_by_side_on_shared_tasks():
    from pace.experiments.mas.run import single_agent_table
    from pace.trace import write_jsonl
    with tempfile.TemporaryDirectory() as d:
        eps = _fake_log(d, n_tasks=6)
        by = {}
        for e in eps:
            by.setdefault(e.meta["mode"], []).append(e)
        ids = sorted({e.task_id for e in by["always_verify"]})
        cats = [next(e.meta["category"] for e in eps if e.task_id == t) for t in ids]
        sa = os.path.join(d, "sa.jsonl")
        write_jsonl(_single_agent_log(ids[:4] + ["b2b-999"], cats[:4] + ["handle_time"]), sa)   # 4 shared, 1 foreign
        txt = single_agent_table(by, Path(sa), [2.0, 5.0])
        assert "single agent, fixed mid | 4 | 1.000 | 0.0500 | 0.900 | 0.750 |" in txt
        assert "per-task oracle (max quality, cheapest) | 4 | 1.000 | 0.0500" in txt
        for m in ("always_verify", "never_verify", "controller:uniform"):   # MAS rows restricted to the 4 shared tasks
            assert f"MAS `{m}` | 4 |" in txt
        assert "episodes on" not in txt and "Paired `always_verify` vs single mid on the same 4 tasks" in txt
        by2 = dict(by); by2["controller:uniform"] = by["controller:uniform"][:2]        # a mode on a subset -> the note
        txt2 = single_agent_table(by2, Path(sa), [2.0])
        assert "MAS `controller:uniform`" not in txt2 and "`controller:uniform` (" in txt2 and "of the 4 tasks" in txt2
        assert single_agent_table({"always_verify": [e for e in by["always_verify"] if e.task_id == "b2b-0"]}, Path(sa), [2.0]) is None \
            or "b2b-0" in ids


# ------------------------------------------------------------ end to end
def test_dry_run_end_to_end_on_five_fake_tasks(capsys=None):
    with tempfile.TemporaryDirectory() as d:
        out, ledger_dir = os.path.join(d, "mas.jsonl"), os.path.join(d, "ledger")
        args = ["run", "--mode", "all", "--n-tasks", "5", "--dry-run", "--out", out, "--ledger-dir", ledger_dir]
        assert cli.main(args) == 0
        eps = read_jsonl(out)
        assert len(eps) == 15 and {e.meta["mode"] for e in eps} == {"always_verify", "never_verify", "controller:uniform"}
        assert all(e.meta["backend"] == "fake" for e in eps)
        assert os.path.exists(os.path.join(ledger_dir, "spend.jsonl")) and not os.path.exists(os.path.join(d, "traces"))
        assert cli.main(args) == 0 and len(read_jsonl(out)) == 15            # resumable: nothing re-run
        # a cap stops the run cleanly with the marker
        out2 = os.path.join(d, "mas2.jsonl")
        rc = cli.main(["run", "--mode", "always_verify", "--n-tasks", "5", "--dry-run", "--out", out2,
                       "--ledger-dir", os.path.join(d, "ledger2"), "--cap", "0.05"])
        assert rc == 3 and os.path.exists(os.path.join(d, "ledger2", "STOPPED_AT_CAP"))
        tot = json.load(open(os.path.join(d, "ledger2", "spend_total.json")))
        assert tot["total_usd"] <= 0.05
        # analyze renders every section and exports the quality model + a policy
        res = os.path.join(d, "RESULTS.md")
        assert cli.main(["analyze", "--log", out, "--out", res, "--quality-model-out", os.path.join(d, "qm.json"),
                         "--policy-out", os.path.join(d, "pol.json"), "--k", "3"]) == 0
        txt = open(res).read()
        for sec in ("## 1.", "## 2.", "## 2b.", "## 3.", "## 4.", "## 5.", "## 6.", "always_verify", "never_verify", "DRY RUN",
                    "5 tasks run under both"):
            assert sec in txt
        assert "## 7." not in txt                                   # no single-agent log given
        pre = os.path.join(d, "prelude.md"); open(pre, "w").write("AUTHORED PRELUDE TEXT")
        assert cli.main(["analyze", "--log", out, "--out", res, "--quality-model-out", os.path.join(d, "qm.json"),
                         "--policy-out", os.path.join(d, "pol.json"), "--k", "3", "--prelude", pre, "--single-agent-log", "/nonexistent"]) == 0
        txt = open(res).read()
        assert txt.index("## 0. Interpretation") < txt.index("AUTHORED PRELUDE TEXT") < txt.index("## 1.")
        assert os.path.exists(os.path.join(d, "qm.json"))
        # the estimate command runs and prices from pace/prices.py
        assert cli.main(["estimate", "--n-tasks", "50"]) == 0


def test_real_run_refuses_without_cap_or_key():
    for args in (["run", "--n-tasks", "1"], ["run", "--n-tasks", "1", "--cap", "1", "--env-file", "/nonexistent"]):
        env = dict(os.environ); os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            cli.main(args); raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "cap" in str(e) or "ANTHROPIC_API_KEY" in str(e)
        finally:
            os.environ.clear(); os.environ.update(env)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
