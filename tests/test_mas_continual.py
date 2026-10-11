"""Continual MAS orchestrator (pace/experiments/mas_continual): CPU only, no network, no spend.

    python -m pytest tests/test_mas_continual.py -q      |      python tests/test_mas_continual.py
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
os.chdir(ROOT)

import numpy as np  # noqa: E402

from pace.controller import Controller  # noqa: E402
from pace.experiments.mas.orchestrator import ACTIONS as MAS_ACTIONS  # noqa: E402
from pace.experiments.mas.orchestrator import FEATURE_NAMES as MAS_FEATURE_NAMES  # noqa: E402
from pace.experiments.mas.offpolicy import episode_level, per_decision  # noqa: E402
from pace.experiments.mas_continual import analyze, drift, simenv  # noqa: E402
from pace.experiments.mas_continual.router import Handbook, HandbookRouter, category_of, tune_lambda_c  # noqa: E402
from pace.experiments.mas_continual.run_continual import Config, run  # noqa: E402
from pace.experiments.mas_continual.simenv import (ACTIONS, FEATURE_NAMES, MAS_ACTION_MAP, N_FEATURES, FixedPolicy,  # noqa: E402
                                                   MASSimHarness, UniformPolicy, serve)
from pace.experiments.mas_continual.train import train_cgrpo_replay, true_eval  # noqa: E402
from pace.gate import gate  # noqa: E402


def _h(stage: int = 0, seed: int = 0, schedule: str = "default") -> MASSimHarness:
    gens = drift.schedule(schedule)
    return MASSimHarness(gens[stage].params, seed=seed, version=gens[stage].version)


# ------------------------------------------------------------ calibration
def test_calibration_reproduces_from_the_committed_logs():
    c = simenv.calibrate()
    for k in ("plan_len_counts", "replan_len_counts", "worker_kind_counts", "retry_kind_counts", "tokens", "worker_in_by_pos",
              "verifier_counts", "flag_given_wrong", "flag_given_correct", "verifier_calls_per_episode", "modes",
              "mas_cat_counts", "single_tier", "single_cat_counts"):
        assert getattr(c, k) == getattr(simenv.M, k), k
    p = simenv.measured_params()
    assert abs(sum(p.plan_len_p) - 1.0) < 1e-9 and abs(p.plan_len_p[2] - 0.8) < 1e-9
    assert abs(p.verifier_fail_rows["verify-1"]["good"] - 72 / 124) < 1e-9
    assert p.verifier_fail_rows["verify-1"]["good"] == p.verifier_fail_rows["verify-1"]["bad"]   # the measured verifier is uninformative
    assert abs(p.worker_in_slope - (10330 - 6833) / 2) < 1e-9
    assert set(p.available_agents) == {"plan-1", "plan-2", "plan-3", "data-1", "data-2", "data-3", "verify-1", "answer-1", "answer-2", "answer-3"}
    assert p.available_tools == () and p.available_skills == ()


def test_simulated_fixed_modes_reproduce_the_measured_process_numbers():
    h = _h()
    tasks = h.sample_tasks(600, seed=5)
    for mode, pol in (("always_verify", FixedPolicy(True)), ("never_verify", FixedPolicy(False))):
        s = simenv.mode_stats(serve(h, tasks, pol, seed=3, with_state=False))
        m = simenv.M.modes[mode]
        assert abs(s["cost"] - m["cost"]) / m["cost"] < 0.10, (mode, s["cost"], m["cost"])
        assert abs(s["calls"] - m["calls"]) < 0.5, (mode, s["calls"])
        assert abs(s["verifier_calls"] - m["verifier_calls"]) < 0.4, (mode, s["verifier_calls"])
        assert abs(s["accuracy"] - m["accuracy"]) < 0.08, (mode, s["accuracy"])
        if mode == "always_verify":
            assert s["cap_hits"] > 0.85 and 0.85 < s["p_flag_given_wrong"] < 0.99 and 0.85 < s["p_flag_given_correct"] < 0.99
        else:
            assert s["cap_hits"] == 0 and s["verifier_calls"] == 0


# ------------------------------------------------------------ interface reuse
def test_harness_reuses_the_real_orchestrators_feature_layout_and_rules():
    assert FEATURE_NAMES[:len(MAS_FEATURE_NAMES)] == list(MAS_FEATURE_NAMES) and N_FEATURES == len(FEATURE_NAMES)
    assert set(MAS_ACTION_MAP) == set(MAS_ACTIONS)                      # every real macro action has a mapping
    h = _h()
    s = h.reset(h.sample_tasks(1, 0)[0])
    x = h.featurize(s)
    assert len(x) == N_FEATURES and x[0] == 1.0 and x[7] == 1.0        # bias, step0 (the shared base block)
    legal = h.legal_actions(s)
    assert {f"answer@answer-{i}" for i in (1, 2, 3)} <= set(legal)
    assert "verify@verify-1" in legal and "verify@verify-2" not in legal   # stage 0: only the real verifier
    assert not any(a.startswith(("call@", "engage@")) for a in legal)
    assert "delegate@data-4" not in legal and "answer@answer-4" not in legal
    # a pace Controller acts on it; step does not mutate the input state; snapshot/restore round-trips
    pol = Controller(ACTIONS, N_FEATURES, eps=0.1)
    rng = random.Random(0)
    a, prop = pol.act(x, legal, rng)
    assert 0 < prop <= 1 and ACTIONS[a] in legal
    before = json.dumps(h.snapshot(s), sort_keys=True)
    s2, done, cost, _ = h.step(s, "verify@verify-1")
    assert json.dumps(h.snapshot(s), sort_keys=True) == before
    assert s2.step == 1 and cost >= 0.0 and s2.extra["st"].last_verdict in ("pass", "fail")
    r = h.restore(h.snapshot(s2))
    assert h.featurize(r) == h.featurize(s2) and h.legal_actions(r) == h.legal_actions(s2)
    # after a verifier fail, delegate retries the current sub-query once; after a pass it advances
    st = s2.extra["st"]
    i0 = st.i
    s3, _, _, _ = h.step(s2, "delegate@data-2")
    st3 = s3.extra["st"]
    if st.last_verdict == "fail":
        assert st3.n_retry == 1 and st3.i == i0
    else:
        assert st3.n_retry == 0 and (st3.i == i0 + 1 or st3.done)
    # an episode always ends with an answer within the call cap
    ep = serve(h, h.sample_tasks(50, 1), UniformPolicy(), seed=2)
    assert all(e.meta["n_calls"] <= simenv.MAX_CALLS and e.meta["answer_by"] for e in ep)
    assert all(0 < s.propensity <= 1 for e in ep for s in e.steps) and all(e.steps for e in ep)


# ------------------------------------------------------------ drift stages
def test_drift_stages_change_the_right_parameters_and_only_those():
    gens = drift.schedule("default")
    assert [g.axis for g in gens] == ["none", "price", "agents", "tools", "skills"] and [g.day for g in gens] == [1, *drift.EVENT_DAYS]
    p = [g.params for g in gens]
    assert set(drift.changed_fields(p[0], p[1])) == {"stage", "price_factors"}
    assert p[1].price_factors == {"cheap": 1.0, "mid": 0.5, "frontier": 1.5}
    assert set(drift.changed_fields(p[1], p[2])) == {"stage", "available_agents", "verifier_fail_rows", "worker_kind_p",
                                                      "good_given_rows", "answer_logit"}
    assert set(p[2].available_agents) - set(p[1].available_agents) == {"verify-2", "data-4", "answer-4"}
    v = p[2].verifier_fail_rows
    assert v["verify-2"]["good"] < v["verify-2"]["bad"] == v["verify-1"]["bad"]      # precision improves, recall unchanged
    assert set(drift.changed_fields(p[2], p[3])) == {"stage", "available_tools"} and set(p[3].available_tools) == {"schema_lookup", "aggregate_calc"}
    assert set(drift.changed_fields(p[3], p[4])) == {"stage", "available_skills", "plan_len_p"}
    assert len(p[4].plan_len_p) == 4 and abs(sum(p[4].plan_len_p) - 1) < 1e-9
    for a, b in zip(p, p[1:]):      # the harness only grows
        assert set(a.available_agents) <= set(b.available_agents) and set(a.available_tools) <= set(b.available_tools)
        assert set(a.available_skills) <= set(b.available_skills)
    # the legal set follows the stage
    h3 = _h(3)
    s = h3.reset(h3.sample_tasks(1, 0)[0])
    legal = h3.legal_actions(s)
    assert "call@schema_lookup" in legal and "delegate@data-4" in legal and "verify@verify-2" in legal
    assert not any(a.startswith("engage@") for a in legal)
    h4 = _h(4)
    assert any(a.startswith("engage@") for a in h4.legal_actions(h4.reset(h4.sample_tasks(1, 0)[0])))
    # the mild schedule is milder on every axis
    mild = [g.params for g in drift.schedule("mild")]
    assert mild[1].price_factors["mid"] > p[1].price_factors["mid"] and mild[1].price_factors["frontier"] < p[1].price_factors["frontier"]
    assert mild[2].verifier_fail_rows["verify-2"]["good"] > p[2].verifier_fail_rows["verify-2"]["good"]
    assert mild[2].worker_kind_p["data-4"]["rows"] < p[2].worker_kind_p["data-4"]["rows"]
    # the stage-0 harness versions never look like a CRMArena bandit log to pace.gate
    assert not gens[0].version.startswith("crmarena")


def test_repricing_changes_cost_but_not_the_process():
    h0, h1 = _h(0, seed=4), _h(1, seed=4)
    tasks = h0.sample_tasks(200, seed=9)
    e0 = serve(h0, tasks, FixedPolicy(False), seed=1, with_state=False)
    e1 = serve(h1, tasks, FixedPolicy(False), seed=1, with_state=False)
    assert [e.meta["n_calls"] for e in e0] == [e.meta["n_calls"] for e in e1]            # same draws, same process
    assert [e.meta["expected_quality"] for e in e0] == [e.meta["expected_quality"] for e in e1]
    c0, c1 = np.mean([e.total_cost for e in e0]), np.mean([e.total_cost for e in e1])
    assert c1 < c0 and c1 / c0 < 0.7            # never_verify is all-mid: planner on cheap stays, the rest halves


def test_verifier_ablation_knobs():
    gens = drift.schedule("default", verifier2_from_day1=True, max_calls=12, verifier_exempt_from_cap=True)
    p0 = gens[0].params
    assert "verify-2" in p0.available_agents and p0.max_calls == 12 and p0.verifier_exempt_from_cap
    assert p0.verifier_fail_rows["verify-2"]["good"] < p0.verifier_fail_rows["verify-1"]["good"]
    assert all(g.params.max_calls == 12 and g.params.verifier_exempt_from_cap and "verify-2" in g.params.available_agents for g in gens)
    h = MASSimHarness(p0, seed=3, version=gens[0].version)
    s = h.reset(h.sample_tasks(1, 0)[0])
    assert "verify@verify-2" in h.legal_actions(s)
    ep = serve(h, h.sample_tasks(60, 1), FixedPolicy(True, verifier="verify-2"), seed=2, with_state=False)
    st = simenv.mode_stats(ep)
    assert st["cap_hits"] == 0.0                              # 3 sub-queries x (worker + verifier + retry) + planner + answer fit
    assert all(c["extra"]["agent"] == "verify-2" for e in ep for c in e.meta["process"] if c["role"] == "verifier")
    assert max(sum(1 for c in e.meta["process"] if c["role"] != "verifier") for e in ep) <= 12
    st1 = simenv.mode_stats(serve(h, h.sample_tasks(60, 1), FixedPolicy(True, verifier="verify-1"), seed=2, with_state=False))
    assert st["pass_rate"] > st1["pass_rate"] + 0.1          # the informative verifier passes good steps; the measured one does not
    assert st["p_flag_given_correct"] < st1["p_flag_given_correct"]
    # the committed configuration is unchanged by the defaults
    g0 = drift.schedule("default")[0].params
    assert g0.max_calls == 8 and not g0.verifier_exempt_from_cap and "verify-2" not in g0.available_agents
    # the real-log retry table reproduces the counts in the write-up
    from pace.experiments.mas_continual.ablation import retry_outcome_table
    _, r = retry_outcome_table()
    assert r["verifier_fails"] == 107 and r["fails_followed_by_retry"] == 53 and r["bad_to_rows"] == (10, 20) and r["rows_to_rows"] == (26, 33)
    assert r["paired_flips"]["wrong→right"] == 6 and r["paired_flips"]["right→wrong"] == 5


# ------------------------------------------------------------ gate on hand-built cases
def _candidate(pref: list[str]) -> Controller:
    pol = Controller(ACTIONS, N_FEATURES, eps=0.1)
    pol.W[:, 0] = -3.0
    for k, a in enumerate(pref):
        pol.W[ACTIONS.index(a), 0] = 3.0 - 0.5 * k
    return pol


def test_gate_decisions_on_hand_built_candidates():
    h = _h(0, seed=2)
    beh = Controller(ACTIONS, N_FEATURES, eps=0.1)
    log = serve(h, h.sample_tasks(400, seed=3), beh, seed=4, eps=0.1, mode="explore")
    prefix = np.mean([e.total_cost - sum(s.cost for s in e.steps) for e in log])
    held = h.sample_tasks(200, seed=8)
    # (a) a frontier-everywhere candidate is over budget: the cost criterion fails, REVERT
    spend = _candidate(["delegate@data-3", "answer@answer-3"])
    g = gate(log, spend, beh, 0.09 - prefix, 0.05, 0.005, boot=50)
    assert g["decision"] == "REVERT" and not g["cost_ok"]
    assert true_eval(h, spend, held)["cost"] > 0.09 + 0.005
    # (b) the incumbent itself passes both criteria at eps_q = 0.05 (and not at the core default 0.03:
    #     with 400 binary outcomes the DR quality CI half-width is about 0.04, the sample-size finding
    #     the run's configuration states)
    g = gate(log, beh.copy(), beh, 0.09 - prefix, 0.05, 0.005, boot=50)
    assert g["decision"] == "PROMOTE" and g["quality_ok"] and g["cost_ok"]
    assert g["candidate"]["quality"]["dr"][0] - g["candidate"]["quality"]["dr"][1] > 0.03
    # (c) answering at once on the cheap tier is cheaper and worse: the DR point estimates say so
    cheap = _candidate(["answer@answer-1"])
    g = gate(log, cheap, beh, 0.09 - prefix, 0.05, 0.005, boot=50)
    assert g["cost_ok"] and g["candidate"]["cost"]["dr"][0] < g["incumbent"]["cost"][0]
    assert g["candidate"]["quality"]["dr"][0] < g["incumbent"]["quality"][0]
    t_cheap, t_beh = true_eval(h, cheap, held), true_eval(h, beh, held)
    assert t_cheap["cost"] < t_beh["cost"] and t_cheap["quality"] < t_beh["quality"]
    # the per-decision estimators run on these logs and report an ESS
    pd = per_decision(log, cheap, boot=20)
    assert 0 < pd["ess"] <= pd["n"] and len(pd["snips"]) == 3


# ------------------------------------------------------------ trainer
def test_cgrpo_replay_learns_a_budgeted_policy_and_reuses_the_core_pieces():
    h = _h(0, seed=5)
    beh = Controller(ACTIONS, N_FEATURES, eps=0.1)
    log = serve(h, h.sample_tasks(96, seed=1), beh, seed=2, eps=0.1)
    snaps = [s.state for e in log for s in e.steps]
    trail: list = []
    pol, lam = train_cgrpo_replay(snaps, h, budget=0.09, iters=30, batch=8, group=4, init=beh, serve_eps=0.1, seed=0, log=trail)
    assert pol.mix and len(trail) == 30 and lam >= 0 and pol.meta["method"] == "cgrpo_replay"
    assert pol.n_features == N_FEATURES and pol.actions == ACTIONS
    held = h.sample_tasks(150, seed=6)
    before, after = true_eval(h, beh, held), true_eval(h, pol, held)
    assert after["quality"] + 0.03 >= before["quality"] or after["cost"] < before["cost"]   # it moved in a useful direction
    # the evaluation is reproducible whatever the harness drew before
    assert true_eval(h, pol, held) == after


# ------------------------------------------------------------ router
def test_handbook_router_scores_competence_minus_price_with_cost_tie_break():
    h = _h(0, seed=6)
    beh = Controller(ACTIONS, N_FEATURES, eps=0.1)
    log = serve(h, h.sample_tasks(120, seed=2), beh, seed=3, eps=0.1)
    hb = Handbook().learn(log)
    assert hb.n_episodes == 120 and hb.avg_cost["answer@answer-3"] > hb.avg_cost["answer@answer-1"]
    cat = category_of(log[0].steps[0].features)
    assert cat == log[0].meta["category"]
    # with λ_c = 0 and equal competence the cheaper action wins; a large λ_c picks the cheapest
    hb.competence = {}; hb.action_mean = {a: 0.5 for a in ACTIONS}
    r0 = HandbookRouter(hb, 0.0)
    legal = ["answer@answer-1", "answer@answer-3"]
    assert r0.choose(log[0].steps[0].features, legal) == "answer@answer-1"
    hb.action_mean["answer@answer-3"] = 0.9
    assert HandbookRouter(hb, 0.0).choose(log[0].steps[0].features, legal) == "answer@answer-3"
    assert HandbookRouter(hb, 100.0).choose(log[0].steps[0].features, legal) == "answer@answer-1"
    # the router has the surface the off-policy estimators need, and λ_c tuning picks from the grid
    hb2 = Handbook().learn(log[:80])
    tl = tune_lambda_c(hb2, log[80:], 0.09, 0.005, boot=10)
    assert tl["lambda_c"] in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0) and len(tl["rows"]) == 6
    est = episode_level(log[80:], HandbookRouter(hb2, tl["lambda_c"]), boot=10)
    assert 0 < est["ess"] <= est["n"]
    ep = serve(h, h.sample_tasks(10, 4), HandbookRouter(hb2, 1.0, eps=0.3), seed=1, eps=0.3)
    assert all(0 < s.propensity <= 1 for e in ep for s in e.steps)


# ------------------------------------------------------------ BWT arithmetic
def test_bwt_fwt_recovery_and_regret_arithmetic():
    gens = [{"index": 0, "name": "g0", "axis": "none", "day": 1, "date": "", "description": "", "changed": []},
            {"index": 1, "name": "g1", "axis": "price", "day": 3, "date": "", "description": "", "changed": []}]
    meta = {"generations": gens, "oracle": {"0": {"J": 0.50}, "1": {"J": 0.60}}, "uniform": {"0": {"J": 0.1}, "1": {"J": 0.2}},
            "config": {"recover_tol": 0.02, "warmup_days": 1}}
    def row(day, hold):
        return {"day": day, "arm": "continual", "served": {"cost": 0.08, "expected_quality": 0.4},
                "heldout": {str(k): {"J": v, "quality": v, "cost": 0.0} for k, v in hold.items()}}
    days = [row(1, {0: 0.30}), row(2, {0: 0.40}), row(3, {0: 0.42, 1: 0.35}), row(4, {0: 0.36, 1: 0.55}), row(5, {0: 0.39, 1: 0.59})]
    mt = analyze.metrics({"days": days, "meta": meta}, 0.02, 1)
    s0, s1 = mt["arms"]["continual"]["per_stage"]
    assert abs(s0["R_jj"] - 0.40) < 1e-9 and abs(s0["bwt"] - (0.39 - 0.40)) < 1e-9        # R[T, 0] − R[0, 0]
    assert abs(s1["fwt"] - (0.35 - 0.2)) < 1e-9 and abs(s1["hit"] - (0.35 - 0.42)) < 1e-9   # R[0, 1] − b_1 ; the event-day policy on g1 minus on g0
    assert abs(s1["recovery"] - (0.59 - 0.35)) < 1e-9 and s1["days_to_recover"] == 3 and s1["bwt"] is None
    assert abs(s1["stationary_J"] - np.mean([0.35, 0.55, 0.59])) < 1e-9
    assert abs(mt["arms"]["continual"]["regret"] - ((0.5 - 0.40) + (0.6 - 0.35) + (0.6 - 0.55) + (0.6 - 0.59))) < 1e-9
    assert abs(mt["arms"]["continual"]["mean_bwt"] - s0["bwt"]) < 1e-9


# ------------------------------------------------------------ the day loop
def test_a_few_days_of_the_loop_with_a_tiny_policy():
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(n_days=4, n_tasks=12, cgrpo_iters=4, cgrpo_batch=4, cgrpo_group=3, oracle_iters=4, oracle_batch=4,
                     oracle_collect=20, n_heldout=20, warmup_days=2, boot=20, awr_epochs=10, refit_days=1, seed=1)
        meta = run(cfg, Path(d), log=lambda *a, **k: None)
        rows = [json.loads(l) for l in (Path(d) / "days.jsonl").read_text().splitlines()]
        gates = [json.loads(l) for l in (Path(d) / "gates.jsonl").read_text().splitlines()]
        assert {r["day"] for r in rows} == {1, 2, 3, 4} and len(gates) == 8
        arms_by_day = {day: {r["arm"] for r in rows if r["day"] == day} for day in (1, 4)}
        assert arms_by_day[1] == {"continual", "continual_awr", "always_verify", "never_verify"}
        assert arms_by_day[4] == set(analyze.ARMS)                     # frozen and the routers exist after the warm-up
        assert all(set(r["heldout"]) == {"0"} for r in rows)           # one stage in 4 days
        assert all(g["decision"] in ("PROMOTE", "REVERT") and 0 < g["per_decision_ess"] <= g["per_decision_n"] for g in gates)
        assert all(g["gate_budget"] < cfg.budget and g["prefix_cost"] > 0 for g in gates)
        assert "0" in meta["oracle"] and "static" in meta["handbooks"] and meta["calibration_check"]["simulated"]["always_verify"]["n"] == 1000
        text = analyze.report(Path(d), None, None)
        assert "## 2. Per-stage transfer" in text and "## 6. Gate history" in text and "## 10. Simulator calibration" in text


# ------------------------------------------------------------ RAO-style delegation credit, meta-init
def test_rao_reward_shaping_arithmetic():
    from pace.experiments.mas_continual.rao_reward import group_advantages, loo_baseline, node_rewards
    assert loo_baseline([1.0, 2.0, 3.0]) == [2.5, 2.0, 1.5]
    ops = ["delegate", "verify", "delegate", "delegate", "answer"]
    sub = [[1.0], [], [1.0, 0.0], [], []]
    r = node_rewards(0.4, ops, sub, 0.5)
    # +0.5·1, nothing for a verify, +0.5·mean(1, 0) (averaging, not counting), nothing for a delegate without a sub-call
    assert np.allclose(r, [0.9, 0.4, 0.65, 0.4, 0.4])
    assert node_rewards(0.4, ops, sub, 0.5, credit="none") == [0.4] * 5
    roots = [0.2, 0.5, 0.8]
    g_ops = [["delegate", "answer"], ["verify", "answer"], ["delegate", "answer"]]
    g_sub = [[[0.0], []], [[], []], [[1.0], []]]
    adv = group_advantages(roots, g_ops, g_sub, 0.5, credit="rao")
    b = loo_baseline(roots)                      # root-anchored: the bonus is not in the baseline
    assert np.allclose(adv[0], [0.2 + 0.0 - b[0], 0.2 - b[0]])
    assert np.allclose(adv[1], [0.5 - b[1], 0.5 - b[1]])
    assert np.allclose(adv[2], [0.8 + 0.5 - b[2], 0.8 - b[2]])
    none = group_advantages(roots, g_ops, g_sub, 0.5, credit="none")
    assert np.allclose([v for row in none for v in row], [-0.3, -0.3, 0.0, 0.0, 0.3, 0.3])


def test_rao_trainer_reduces_to_the_committed_one_and_credits_only_spawned_sub_calls():
    from pace.experiments.mas_continual.rao_reward import rollout_with_subcalls, train_cgrpo_replay_shaped
    h0 = _h(2, seed=5)
    snaps = [s.state for e in serve(h0, h0.sample_tasks(12, seed=3), UniformPolicy(), seed=4) for s in e.steps]
    kw = dict(iters=3, batch=4, group=3, serve_eps=0.1, seed=9)
    p1, l1 = train_cgrpo_replay(snaps, _h(2, seed=6), 0.09, **kw)
    p2, l2 = train_cgrpo_replay_shaped(snaps, _h(2, seed=6), 0.09, credit="none", **kw)
    assert np.allclose(p1.W, p2.W) and l1 == l2
    p3, _ = train_cgrpo_replay_shaped(snaps, _h(2, seed=6), 0.09, credit="rao", lambda_d=0.5, **kw)
    assert not np.allclose(p1.W, p3.W) and p3.meta["n_delegate_bonus"] > 0
    h, rng = _h(2, seed=7), random.Random(1)
    pol = Controller(ACTIONS, N_FEATURES, eps=1.0)
    for t in h.sample_tasks(30, seed=2):
        s0 = h.reset(t)
        steps, ops, sub, s = rollout_with_subcalls(h, s0, pol, rng)
        assert len(steps) == len(ops) == len(sub)
        n_workers = sum(1 for c in s.extra["st"].calls if c.role == "worker")
        assert sum(len(x) for x in sub) == n_workers - 1           # all but the policy-independent prefix call
        for o, x in zip(ops, sub):
            assert len(x) <= 1 and (o in ("delegate", "replan", "engage") or not x)


def test_delegation_share_computation():
    from types import SimpleNamespace as NS
    from pace.experiments.mas_continual.rao_reward import delegation_share, delegation_share_from_mix
    ep = lambda *names: NS(steps=[NS(action_name=n) for n in names])
    eps_ = [ep("delegate@data-2", "verify@verify-1", "delegate@data-4", "answer@answer-2"), ep("delegate@data-4", "answer@answer-4")]
    d = delegation_share(eps_)
    assert d["n_decisions"] == 6 and abs(d["share"] - 3 / 6) < 1e-12
    assert d["by_agent"] == {"data-2": 1 / 6, "data-4": 2 / 6}
    assert delegation_share([]) ["share"] == 0.0
    assert abs(delegation_share_from_mix({"delegate@data-2": 0.26, "answer@answer-2": 0.15, "delegate@data-4": 0.24}) - 0.50) < 1e-12
    h = _h(0)
    eps2 = serve(h, h.sample_tasks(40, seed=1), FixedPolicy(False), seed=2, with_state=False)
    n_del = sum(1 for e in eps2 for s in e.steps if s.action_name.startswith("delegate@"))
    assert abs(delegation_share(eps2)["share"] - n_del / sum(len(e.steps) for e in eps2)) < 1e-12
    days = [{"arm": "continual", "gen_index": 0, "day": k, "served": {"action_mix": {"delegate@data-2": 0.1 * k}, "delegate_share": 0.2 * k,
             "delegate_by_agent": {"data-2": 0.2 * k}}} for k in (1, 2)]
    full, mix = analyze.delegation_by_stage(days), analyze.delegation_by_stage(days, full=False)
    assert abs(full[0]["share"] - 0.3) < 1e-12 and abs(mix[0]["share"] - 0.15) < 1e-12 and abs(full[0]["by_agent"]["data-2"] - 0.3) < 1e-12


def test_meta_init_never_reads_heldout_sets_or_the_last_stage():
    from pace.experiments.mas_continual import meta_init
    gens = drift.schedule("default")
    seen = []
    orig = MASSimHarness.sample_tasks

    def spy(self, n, seed=0):
        seen.append((self.version, seed))
        return orig(self, n, seed)

    MASSimHarness.sample_tasks = spy
    try:
        for seed in (0, 3):
            seen.clear()
            pol, rep = meta_init.meta_learn(outer=2, inner_days=1, n_tasks=6, iters=2, batch=4, group=2, seed=seed, gens=gens,
                                            log=lambda *a, **k: None)
            assert seen and {s for _, s in seen}.isdisjoint(meta_init.heldout_seeds(seed))
            assert all(sd >= meta_init.META_SEED_BASE for _, sd in seen)                  # disjoint from daily traffic and oracle seeds too
            assert {v for v, _ in seen} == {gens[g].version + "-meta" for g in meta_init.META_STAGES}
            assert not any(v.startswith(gens[4].version) for v, _ in seen)               # the last stage is never instantiated
            assert meta_init.audit(rep, seed)["ok"] and pol.W.shape == (len(ACTIONS), N_FEATURES)
            assert np.abs(pol.W).sum() > 0 and pol.mix is None
    finally:
        MASSimHarness.sample_tasks = orig
    bad = dict(rep, seen={"stages": [0, 4], "task_seeds": [500 + 3]})
    assert meta_init.audit(bad, 0) == {"heldout_seed_overlap": [503], "last_stage_seen": [4], "ok": False}
    try:
        meta_init.meta_learn(stages=(0, 4), outer=1, gens=gens, log=lambda *a, **k: None)
        raise AssertionError("the last stage must be refused")
    except ValueError:
        pass


def test_the_loop_runs_with_rao_credit_and_meta_init():
    for kw in ({"delegation_credit": "rao", "lambda_d": 0.5}, {"init": "meta", "meta_outer": 1, "meta_inner_days": 1}):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(n_days=3, n_tasks=12, cgrpo_iters=3, cgrpo_batch=4, cgrpo_group=3, oracle_iters=3, oracle_batch=4,
                         oracle_collect=20, n_heldout=20, warmup_days=2, boot=20, refit_days=1, seed=1, arms="ablation", **kw)
            meta = run(cfg, Path(d), log=lambda *a, **k: None)
            rows = [json.loads(l) for l in (Path(d) / "days.jsonl").read_text().splitlines()]
            assert all(0.0 <= r["served"]["delegate_share"] <= 1.0 for r in rows)
            assert all(abs(sum(r["served"]["delegate_by_agent"].values()) - r["served"]["delegate_share"]) < 1e-9 for r in rows)
            if "init" in kw:
                assert meta["meta_init"]["audit"]["ok"] and (Path(d) / "policies" / "meta_init.json").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
