"""Priced Socratic self-refinement (pace/experiments/ssr) -- all offline, no API, no Salesforce.

    python -m pytest tests/test_ssr.py -q      |      python tests/test_ssr.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pace.controller import Controller  # noqa: E402
from pace.experiments.ssr import run as ssr_run  # noqa: E402
from pace.experiments.ssr.analyze import analyze, calibration, harm_table  # noqa: E402
from pace.experiments.ssr.fake import FakeBackend, FakeWorld, make_fake_tasks  # noqa: E402
from pace.experiments.ssr.policy import (N_FEATURES, FEATURE_NAMES, Always, ControllerPolicy, Never,  # noqa: E402
                                         UncertaintyGated, auroc, featurize, fit_controller, parse_policy, pareto)
from pace.experiments.ssr.refine import (Claim, normalize_answer, parse_decomposition, probe, refine_step,  # noqa: E402
                                         tag_purpose)
from pace.experiments.ssr.tree import STOP, SSRState, build_trees, evaluate_policy, evaluate_policy_on  # noqa: E402
from pace.harness.crmarena import CRMArenaHarness, PricedLLM  # noqa: E402
from pace.spend import STOP_MARKER, SpendLedger  # noqa: E402


# ------------------------------------------------------------------ fakes
def make_fake_harness(d, n_tasks=2, cap=None, seed=0):
    tasks = make_fake_tasks(n_tasks, seed)
    ledger = SpendLedger(d, cap)
    world = FakeWorld(tasks, seed)
    llm = PricedLLM(ledger, completion_fn=world, timeout=30, effort="low")
    h = CRMArenaHarness(org="b2b", ledger=ledger, llm=llm, backend=FakeBackend(), tasks=tasks)
    return h, tasks, ledger, world


def node(task_id, path, answer, q, probe_min=None, initial_cost=0.05, probe_usd=0.003, cat="handle_time"):
    pr = None
    if probe_min is not None:
        pr = {"claims": [{"claim": "c", "question": "q", "expected": answer, "self_conf": probe_min}],
              "answer_conf": probe_min, "min_conf": probe_min, "mean_conf": probe_min, "usd": probe_usd,
              "calls": 1, "parse_ok": True}
    return {"kind": "node", "task_id": task_id, "category": cat, "reward_metric": "exact_match", "org": "b2b",
            "initial_arm": "cheap", "path": list(path), "round": len(path), "answer": answer, "quality": q,
            "initial_cost": initial_cost, "probe": pr, "backend": "test"}


def edge(task_id, path, q_before, q_after, usd=0.10, conf_min=0.5, usd_eval=0.0005):
    return {"kind": "edge", "task_id": task_id, "initial_arm": "cheap", "path": list(path), "round": len(path),
            "tier": path[-1], "backend": "test",
            "refine": {"tier": path[-1], "claims": [], "conf_min": conf_min, "conf_mean": conf_min,
                       "answer_before": "a", "answer_after": "b" if q_before != q_after else "a",
                       "quality_before": q_before, "quality_after": q_after, "changed": q_before != q_after,
                       "usd_resolve": usd * 0.9, "usd_refine": usd * 0.1, "usd_eval": usd_eval, "calls": 5,
                       "worker_turns": 4, "usd": usd}}


def hand_built_tree(task_id="t1", q0=0.0, probe_min=0.3, cat="handle_time", q_cheap=1.0, q_mid=1.0, q_cc=1.0,
                    q_cm=1.0, q_mc=1.0, q_mm=1.0, initial_cost=0.05):
    """A complete 2-round tree over {cheap, mid}: edge costs cheap $0.10, mid $0.20."""
    recs = [node(task_id, [], "a0", q0, probe_min, initial_cost, cat=cat)]
    for t, qt in (("cheap", q_cheap), ("mid", q_mid)):
        recs.append(edge(task_id, [t], q0, qt, 0.10 if t == "cheap" else 0.20))
        recs.append(node(task_id, [t], "a1", qt, 0.9, initial_cost, cat=cat))
        for t2, q2 in (("cheap", q_cc if t == "cheap" else q_mc), ("mid", q_cm if t == "cheap" else q_mm)):
            recs.append(edge(task_id, [t, t2], qt, q2, 0.10 if t2 == "cheap" else 0.20))
            recs.append(node(task_id, [t, t2], "a2", q2, None, initial_cost, cat=cat))
    return recs


# ------------------------------------------------------------------ tests
def test_round_produces_priced_ledger_entries():
    with tempfile.TemporaryDirectory() as d:
        h, tasks, ledger, world = make_fake_harness(d)
        task = tasks[0]
        pr = probe(h, task, "005FAKE0000", ["SELECT Id FROM Case"])
        assert pr.parse_ok and 1 <= len(pr.claims) <= 3 and 0 <= pr.min_conf <= pr.mean_conf <= 1
        assert pr.usd > 0 and abs(pr.usd - ledger.by_purpose["ssr:decompose"]) < 1e-12
        res = refine_step(h, task, "005FAKE0000", 1.0, pr.claims, "cheap", ["SELECT Id FROM Case"], k=2, max_turns=3)
        lines = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl"))]
        purposes = {x["purpose"] for x in lines}
        assert purposes == {"ssr:decompose", "ssr:resolve:cheap", "ssr:refine:cheap"}, purposes
        n_resolve = sum(x["purpose"] == "ssr:resolve:cheap" for x in lines)
        assert n_resolve == res.worker_turns and n_resolve >= len(pr.claims) * 2        # k=2, >=1 turn each
        assert all(x["model"] == "claude-haiku-4-5" for x in lines)
        assert abs(res.usd_resolve - ledger.by_purpose["ssr:resolve:cheap"]) < 1e-12
        assert abs(res.usd_refine - ledger.by_purpose["ssr:refine:cheap"]) < 1e-12
        assert abs(res.usd - (res.usd_resolve + res.usd_refine)) < 1e-12
        assert res.usd_eval == 0.0                        # fake scorer makes no calls; never charged anyway
        for c in res.claims:
            assert len(c.samples) == 2 and c.agree in (0.0, 1.0) and c.support in (0.0, 1.0)
            assert abs(c.conf - (c.self_conf + c.agree + c.support) / 3) < 1e-12
        assert res.conf_min == min(c.conf for c in res.claims)
        assert res.quality_after in (0.0, 1.0)
        # the probe's claims were copied, not mutated
        assert all(c.samples == [] and c.conf is None for c in pr.claims)
        # a mid round bills the mid model under its own purposes
        res2 = refine_step(h, task, "005FAKE0000", 1.0, pr.claims, "mid", [], k=2, max_turns=3)
        lines = [json.loads(x) for x in open(os.path.join(d, "spend.jsonl"))]
        mids = [x for x in lines if x["purpose"] in ("ssr:resolve:mid", "ssr:refine:mid")]
        assert mids and all(x["model"] == "claude-sonnet-5-5" for x in mids)
        assert res2.usd > res.usd                         # same tokens, pricier tier
        assert h.llm is not None and not hasattr(h.llm, "inner")   # tag_purpose restored the original llm


def test_tag_purpose_restores_on_error():
    with tempfile.TemporaryDirectory() as d:
        h, tasks, ledger, world = make_fake_harness(d)
        orig = h.llm
        try:
            with tag_purpose(h, "ssr:resolve:cheap"):
                assert h.llm is not orig
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert h.llm is orig


def test_parse_decomposition_and_normalization():
    claims, aconf, ok = parse_decomposition("not json at all", "005X", "who?")
    assert not ok and len(claims) == 1 and claims[0].self_conf == 0.5 and aconf == 0.5
    txt = 'Sure: {"claims": [' + ",".join(f'{{"claim":"c{i}","question":"q{i}","expected":"v{i}","confidence":{c}}}'
                                          for i, c in enumerate([0.9, 1.7, -2, 0.4])) + '], "answer_confidence": 0.8}'
    claims, aconf, ok = parse_decomposition(txt, "005X", "who?")
    assert ok and len(claims) == 3 and [c.self_conf for c in claims] == [0.9, 1.0, 0.0] and aconf == 0.8
    assert normalize_answer("  <respond> 'B', A </respond>") == "a, b"
    assert normalize_answer(None) == ""


def test_uncertainty_gate_thresholds():
    legal = [STOP, "REFINE@cheap", "REFINE@mid"]
    s = SSRState("t", "handle_time", "cheap", 0, 0.05, legal, probe_min=0.55, probe_mean=0.7, answer_conf=0.6)
    assert UncertaintyGated(0.5, "cheap").decide(s) == STOP
    assert UncertaintyGated(0.55, "cheap").decide(s) == STOP            # strict: refine iff conf < tau
    assert UncertaintyGated(0.6, "cheap").decide(s) == "REFINE@cheap"
    assert UncertaintyGated(0.6, "mid").decide(s) == "REFINE@mid"
    s2 = SSRState("t", "handle_time", "cheap", 2, 0.3, [STOP], probe_min=0.1)
    assert UncertaintyGated(0.9, "cheap", max_rounds=2).decide(s2) == STOP
    s3 = SSRState("t", "handle_time", "cheap", 1, 0.3, [STOP, "REFINE@cheap"], probe_min=0.1)
    assert UncertaintyGated(0.9, "mid").decide(s3) == STOP               # mid branch not available
    assert UncertaintyGated(0.9, "cheap", max_rounds=1).decide(s3) == STOP
    assert Never().decide(s) == STOP and Always(1, "mid").decide(s) == "REFINE@mid"
    assert Always(1, "mid").decide(s3) == STOP
    p = parse_policy("uncertainty_gated:0.7:mid")
    assert p.tau == 0.7 and p.tier == "mid" and p.needs_probe and not Always(1).needs_probe


def test_exact_policy_evaluation_charges_probes_correctly():
    trees = build_trees(hand_built_tree("t1", q0=0.0, probe_min=0.3, q_cheap=1.0, q_cc=0.0))
    assert len(trees) == 1 and trees[0].complete(["cheap", "mid"], 2)
    T, tiers = trees[0], ["cheap", "mid"]
    o = evaluate_policy(T, Never(), tiers, 2)
    assert o.quality == 0.0 and abs(o.cost - 0.05) < 1e-12                 # initial only, no probe
    o = evaluate_policy(T, Always(1, "cheap"), tiers, 2)
    assert o.quality == 1.0 and abs(o.cost - (0.05 + 0.003 + 0.10)) < 1e-12 and o.rounds == 1
    o = evaluate_policy(T, Always(2, "cheap"), tiers, 2)
    assert o.quality == 0.0 and abs(o.cost - (0.05 + 0.003 + 0.10 + 0.003 + 0.10)) < 1e-12
    o = evaluate_policy(T, UncertaintyGated(0.2, "cheap"), tiers, 2)       # looks, then stops: pays the probe
    assert o.quality == 0.0 and abs(o.cost - 0.053) < 1e-12 and o.rounds == 0
    o = evaluate_policy(T, UncertaintyGated(0.5, "mid"), tiers, 2)         # refines once; child conf 0.9 -> stop
    assert o.quality == 1.0 and abs(o.cost - (0.05 + 0.003 + 0.20 + 0.003)) < 1e-12 and o.rounds == 1
    assert abs(o.eval_usd - 0.0005) < 1e-12
    # a branch that was never logged makes the policy not evaluable
    partial = build_trees([r for r in hand_built_tree("t2") if not (r["path"] and r["path"][0] == "mid")])
    assert evaluate_policy(partial[0], Always(1, "mid"), tiers, 2) is None
    assert evaluate_policy(partial[0], Always(1, "cheap"), tiers, 2) is not None
    r = evaluate_policy_on(trees + partial, Always(1, "mid"), tiers, 2)
    assert r["n"] == 1 and r["n_total"] == 2


def test_refinement_harm_and_calibration_accounting():
    recs = (hand_built_tree("w1", q0=0.0, probe_min=0.2, q_cheap=1.0, q_mid=1.0, q_cc=0.0, q_cm=1.0)     # fixed, then harmed on cheap
            + hand_built_tree("r1", q0=1.0, probe_min=0.9, q_cheap=0.0, q_mid=1.0, q_cc=0.0, q_cm=0.0)   # harmed on cheap at round 1
            + hand_built_tree("r2", q0=1.0, probe_min=0.8, cat="private_customer_information", q_cheap=1.0, q_mid=0.0))
    trees = build_trees(recs)
    rows = {(h["tier"], h["round"], h["group"]): h for h in harm_table(trees)}
    c1 = rows[("cheap", 1, "all")]
    assert c1["n"] == 3 and c1["fix"] == 1 and c1["n_wrong"] == 1 and c1["harm"] == 1 and c1["n_right"] == 2
    assert abs(c1["dq"] - (1.0 - 1.0 + 0.0) / 3) < 1e-12
    m1 = rows[("mid", 1, "all")]
    assert m1["fix"] == 1 and m1["harm"] == 1
    assert rows[("mid", 1, "confidentiality")]["harm"] == 1 and rows[("mid", 1, "confidentiality")]["n"] == 1
    c2 = rows[("cheap", 2, "all")]
    assert c2["n"] == 6 and c2["harm"] >= 1
    cal = calibration(trees, ["cheap", "mid"])
    assert cal["n"] == 3 and abs(cal["base_wrong"] - 1 / 3) < 1e-9
    assert cal["1 − min claim self-confidence"] == 1.0            # the wrong one had the lowest confidence
    assert auroc([0.9, 0.2], [1, 0]) == 1.0 and auroc([0.2, 0.9], [1, 0]) == 0.0 and auroc([0.5, 0.5], [1, 0]) == 0.5
    assert pareto([("a", 0.5, 0.1), ("b", 0.6, 0.2), ("c", 0.4, 0.3), ("d", 0.5, 0.2)]) == {"a", "b"}


def test_controller_features_and_fit():
    legal = [STOP, "REFINE@cheap", "REFINE@mid"]
    s = SSRState("t", "case_routing", "mid", 1, 0.12, legal, probe_min=0.3, probe_mean=0.5, answer_conf=0.4,
                 prev_conf_min=0.6, changed_last=True)
    x = featurize(s)
    assert len(x) == N_FEATURES == len(FEATURE_NAMES)
    assert x[FEATURE_NAMES.index("cat=case_routing")] == 1.0 and sum(x[1:23]) == 1.0
    assert x[FEATURE_NAMES.index("round=1")] == 1.0 and x[FEATURE_NAMES.index("round=0")] == 0.0
    assert x[FEATURE_NAMES.index("probe_min")] == 0.3 and x[FEATURE_NAMES.index("prev_conf_min")] == 0.6
    assert x[FEATURE_NAMES.index("has_prev")] == 1.0 and x[FEATURE_NAMES.index("changed_last")] == 1.0
    assert abs(x[FEATURE_NAMES.index("cost_so_far*10")] - 1.2) < 1e-12 and x[FEATURE_NAMES.index("arm=mid")] == 1.0
    x0 = featurize(SSRState("t", "unknown", "cheap", 0, 0.0, legal))
    assert sum(x0[1:23]) == 0.0 and x0[FEATURE_NAMES.index("probe_min")] == 0.5 and x0[FEATURE_NAMES.index("has_prev")] == 0.0

    # a world where refining pays only when confidence is low: the fitted controller should learn it
    recs = []
    for i in range(12):
        low = i % 2 == 0
        recs += hand_built_tree(f"t{i}", q0=0.0 if low else 1.0, probe_min=0.2 if low else 0.9,
                                cat=["handle_time", "case_routing", "lead_routing"][i % 3],
                                q_cheap=1.0 if low else 0.0, q_mid=1.0 if low else 0.0,
                                q_cc=1.0 if low else 0.0, q_cm=1.0 if low else 0.0, q_mc=1.0 if low else 0.0, q_mm=1.0 if low else 0.0)
    trees = build_trees(recs)
    ctrl = fit_controller(trees, ["cheap", "mid"], 2, budget=None, lam=2.0, steps=150, seed=0)
    assert ctrl.actions == legal and ctrl.n_features == N_FEATURES and ctrl.mix and len(ctrl.mix) == 75
    pol = ControllerPolicy(ctrl)
    lo = pol.probs(SSRState("t", "handle_time", "cheap", 0, 0.05, legal, probe_min=0.2, probe_mean=0.2, answer_conf=0.2))
    hi = pol.probs(SSRState("t", "handle_time", "cheap", 0, 0.05, legal, probe_min=0.9, probe_mean=0.9, answer_conf=0.9))
    assert abs(sum(lo.values()) - 1) < 1e-9 and abs(sum(hi.values()) - 1) < 1e-9
    assert lo[STOP] < 0.5 < hi[STOP], (lo, hi)
    assert lo["REFINE@cheap"] > lo["REFINE@mid"]         # same quality, cheaper
    res = evaluate_policy_on(trees, pol, ["cheap", "mid"], 2)
    never = evaluate_policy_on(trees, Never(), ["cheap", "mid"], 2)
    assert res["quality"] > never["quality"] + 0.3 and res["cost"] < 0.05 + 0.003 + 0.10 + 0.01
    # budgeted training moves lambda and still round-trips through JSON
    ctrl_b = fit_controller(trees, ["cheap", "mid"], 2, budget=0.08, lam=0.0, steps=100, seed=0)
    assert ctrl_b.meta["budget"] == 0.08 and ctrl_b.meta["lambda_bar"] >= 0.0
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.json")
        ctrl_b.save(p)
        again = ControllerPolicy.load(p)
        assert again.ctrl.actions == legal and again.needs_probe
        bad = Controller(legal, 5)
        bad.save(os.path.join(d, "bad.json"))
        try:
            ControllerPolicy.load(os.path.join(d, "bad.json"))
            raise AssertionError("expected a feature-count error")
        except ValueError:
            pass


def test_dry_run_end_to_end_with_cap_and_resume():
    with tempfile.TemporaryDirectory() as d:
        out, ledger_dir = os.path.join(d, "tree.jsonl"), os.path.join(d, "ledger")
        base = ["run", "--dry-run", "--n-tasks", "5", "--out", out, "--traces-dir", ledger_dir, "--tiers", "cheap,mid"]
        # 1) a tight cap stops the run before exceeding it and writes the marker
        rc = ssr_run.main(base + ["--cap", "0.5"])
        assert rc == 3 and os.path.exists(os.path.join(ledger_dir, STOP_MARKER))
        tot = json.load(open(os.path.join(ledger_dir, "spend_total.json")))
        assert tot["total_usd"] <= 0.5
        n_before = len(open(out).read().splitlines())
        assert n_before > 0
        # 2) resume with a higher cap: finishes, skips logged branches, no duplicate keys
        rc = ssr_run.main(base + ["--cap", "100"])
        assert rc == 0 and not os.path.exists(os.path.join(ledger_dir, STOP_MARKER))
        recs = [json.loads(x) for x in open(out)]
        keys = [(r["kind"], r["task_id"], r["initial_arm"], tuple(r["path"])) for r in recs]
        assert len(keys) == len(set(keys)) and len(recs) > n_before
        trees = build_trees(recs)
        assert len(trees) == 5 and all(t.complete(["cheap", "mid"], 2) for t in trees)
        assert all(t.root.record.get("queries") is not None for t in trees)
        purposes = {json.loads(x)["purpose"] for x in open(os.path.join(ledger_dir, "spend.jsonl"))}
        assert {"agent:cheap", "ssr:decompose", "ssr:resolve:cheap", "ssr:resolve:mid",
                "ssr:refine:cheap", "ssr:refine:mid"} <= purposes
        # 3) a served policy logs only its branch and is evaluable; others are not
        out2 = os.path.join(d, "gated.jsonl")
        rc = ssr_run.main(base[:4] + ["--out", out2, "--traces-dir", ledger_dir, "--policy", "uncertainty_gated:0.6:cheap",
                                      "--cap", "100"])
        assert rc == 0
        t2 = build_trees([json.loads(x) for x in open(out2)])
        assert len(t2) == 5 and all(n.record.get("decision") in (STOP, "REFINE@cheap") for t in t2 for n in t.nodes()
                                   if n.record.get("decision") is not None)
        g = evaluate_policy_on(t2, UncertaintyGated(0.6, "cheap"), ["cheap"], 2)
        assert g["n"] == 5
        a = evaluate_policy_on(t2, Always(2, "cheap"), ["cheap"], 2)
        assert a["n"] < 5 or all(t.root.probe["min_conf"] < 0.6 for t in t2)
        # 4) analyze renders the report and the frontier CSV
        md, csv_path, pol = os.path.join(d, "R.md"), os.path.join(d, "F.csv"), os.path.join(d, "ctrl.json")
        rows, meta = analyze(out, md, csv_path, budget=0.10, folds=2, seed=0, policy_out=pol)
        text = open(md).read()
        for s in ("DRY RUN", "## 1. Is the confidence signal", "## 2. Refinement harm", "## 3. Quality vs cost",
                  "## 4. Priced comparison", "## 5. Automatic checks", "| never |", "always_1:cheap", "uncertainty_gated:0.7:mid",
                  "controller (B=$0.10, learned λ), 2-fold held-out", "oracle (max quality, cheapest)", "H1", "H2", "H3"):
            assert s in text, s
        assert os.path.exists(pol) and len(open(csv_path).read().splitlines()) == len(rows) + 1
        never = next(r for r in rows if r["policy"] == "never")
        for r in rows:
            assert r["cost"] >= never["cost"] - 1e-9 and 0 <= r["quality"] <= 1
        assert any(r["pareto"] for r in rows)
        assert 0 <= never["q_se"] <= 0.5 and len(never["per_task"]) == never["n"]
        assert "quality ± SE" in text and not any(r["family"] == "tier_arm" for r in rows)   # fake log: no arms


def test_analyze_prelude_and_tier_arm_reference_rows():
    """A hand-built real-backend tree + a tiny counterfactual log: the authored prelude becomes §0,
    the single-agent arms become reference rows and §6 pairs every policy against them."""
    from pace.trace import Episode, TraceStep, write_jsonl
    recs = []
    for i, (q0, qc, qm) in enumerate([(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]):
        recs += [r for r in hand_built_tree(f"b2b-{i}", q0=q0, probe_min=0.3, q_cheap=qc, q_mid=qm)
                 if len(r["path"]) <= 1 and not (r["kind"] == "node" and len(r["path"]) == 1 and False)]
    # depth-1 tree: keep only root, round-1 edges and their child nodes (no probe on the children)
    recs = [r for r in recs if len(r["path"]) <= 1]
    for r in recs:
        r["backend"] = "crmarena"
        if r["kind"] == "node" and r["path"]:
            r["probe"] = None
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "tree.jsonl")
        with open(log, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        eps = []
        for i in range(4):
            for arm, q, c in (("cheap", float(i % 2), 0.05), ("mid", 1.0, 0.12), ("frontier", 1.0, 0.16)):
                st = TraceStep(f"b2b-{i}", "crmarena-pro-v1", 0, [1.0], 0, f"CONTINUE@{arm}", 1 / 3, c, 0.0)
                eps.append(Episode(f"b2b-{i}", [st], q, c, 0.0, {"arm": arm}))
        cf = os.path.join(d, "cf.jsonl")
        write_jsonl(eps, cf)
        pre = os.path.join(d, "INTERPRETATION.md")
        open(pre, "w").write("Authored verdict: **kept across regenerations**.")
        md, csv_path = os.path.join(d, "R.md"), os.path.join(d, "F.csv")
        rows, _ = analyze(log, md, csv_path, folds=2, prelude=pre, counterfactual=cf)
        text = open(md).read()
        assert text.index("## 0. Interpretation") < text.index("## 1.") and "kept across regenerations" in text
        mid = next(r for r in rows if r["family"] == "tier_arm" and r["tier"] == "mid")
        assert mid["quality"] == 1.0 and abs(mid["cost"] - 0.12) < 1e-12 and mid["n"] == 4
        assert "## 5. Refine the cheap answer" in text and "## 6. Automatic checks" in text
        line = next(l for l in text.splitlines() if l.startswith("| always_1:cheap |") and "vs" not in l and l.count("|") == 8)
        # always_1:cheap quality 0.5 vs mid arm 1.0 on the same 4 tasks: paired Δq = -0.5
        assert "-0.500 ±" in line, line
        # the arm rows compete on the frontier and are not paired against themselves
        assert mid["pareto"] and not any(l.startswith("| buy mid") for l in text.split("## 5.")[1].split("## 6.")[0].splitlines())


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
