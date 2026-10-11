"""Offline CRMArena pieces on a fake 6-task counterfactual log (no network, no API, no torch):
the DR bucketing with its fallback, the lookup table and K-fold split, the constrained lookup
trainer, first-step AWR, the bandit gate estimators, repricing from the spend ledger, and the
phase-2 SFT labels."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pace.features import CRMBuckets, crm_context  # noqa: E402
from pace.gate import estimate_bandit, gate  # noqa: E402
from pace.harness.crmarena import CATEGORIES, N_FEATURES, ORG_TYPES, TURN0_IDX  # noqa: E402
from pace.lookup import (DECISION_ACTIONS, LookupTable, fixed_arm_log, kfold_tasks,  # noqa: E402
                         reprice_episodes)
from pace.prices import MODELS, PRICES_PER_MTOK, TIERS, call_cost  # noqa: E402
from pace.trace import Episode, TraceStep  # noqa: E402
from pace.train_offline import train_awr, train_cgrpo_lookup  # noqa: E402

# category -> tier -> (quality, cost); 3 tasks of A, 2 of B, 1 of C
ARMS = {"lead_routing": {"cheap": (1.0, 0.02), "mid": (1.0, 0.05), "frontier": (1.0, 0.10)},
        "case_routing": {"cheap": (0.0, 0.03), "mid": (1.0, 0.06), "frontier": (1.0, 0.12)},
        "knowledge_qa": {"cheap": (0.0, 0.03), "mid": (0.0, 0.07), "frontier": (1.0, 0.15)}}
TASKS = [("b2b-1", "lead_routing"), ("b2b-2", "lead_routing"), ("b2b-3", "lead_routing"),
         ("b2b-4", "case_routing"), ("b2b-5", "case_routing"), ("b2b-6", "knowledge_qa")]


def _x(cat: str) -> list[float]:
    x = [0.0] * N_FEATURES
    x[0] = 1.0
    x[1 + CATEGORIES.index(cat)] = 1.0
    x[1 + len(CATEGORIES) + ORG_TYPES.index("b2b")] = 1.0
    x[TURN0_IDX] = 1.0
    return x


def fake_log() -> tuple[list[Episode], list[dict]]:
    """Two-step episodes (CONTINUE then ANSWER on the arm's tier) and their spend records,
    with token counts that price exactly to the step costs."""
    eps, spend = [], []
    for k, (tid, cat) in enumerate(TASKS):
        for tier in TIERS:
            q, c = ARMS[cat][tier]
            pin, pout = 1000 * (k + 1), 100 * (k + 1)
            c0 = call_cost(MODELS[tier], pin, pout)
            c1 = c - c0
            out1 = 50
            pin_usd, pout_usd = PRICES_PER_MTOK[MODELS[tier]]
            in1 = int(round((c1 * 1e6 - out1 * pout_usd) / pin_usd))
            c1 = call_cost(MODELS[tier], in1, out1)
            j = TIERS.index(tier)
            steps = [TraceStep(tid, "crmarena-pro-v1", 0, _x(cat), j, f"CONTINUE@{tier}", 1 / 3, c0, 1.0,
                               legal=list(DECISION_ACTIONS) + [f"ANSWER@{t}" for t in TIERS]),
                     TraceStep(tid, "crmarena-pro-v1", 1, _x(cat), 3 + j, f"ANSWER@{tier}", 1.0, c1, 1.0)]
            eps.append(Episode(tid, steps, q, c0 + c1, 2.0, {"arm": tier, "category": cat, "org": "b2b"}))
            spend += [{"model": MODELS[tier], "in_tokens": pin, "out_tokens": pout, "usd": c0, "task_id": tid,
                       "purpose": f"agent:{tier}"},
                      {"model": MODELS[tier], "in_tokens": in1, "out_tokens": out1, "usd": c1, "task_id": tid,
                       "purpose": f"agent:{tier}:answer"}]
    return eps, spend


def test_crm_buckets_fallback():
    eps, _ = fake_log()
    ctx = [crm_context(e.steps[0].features) for e in eps]
    assert ctx[0] == ("lead_routing", "b2b") and ctx[-1] == ("knowledge_qa", "b2b")
    tiers = [e.meta["arm"] for e in eps]
    q = [e.quality for e in eps]
    b = CRMBuckets(3).fit([c for c, _ in ctx], [o for _, o in ctx], tiers, q)
    # 3 lead_routing tasks per tier -> the category bucket is used
    assert b.key("lead_routing", "b2b", "cheap") == ("cat", "lead_routing", "cheap")
    assert b.predict("lead_routing", "b2b", "cheap") == 1.0
    # 2 case_routing tasks (< 3) -> falls back to (org, tier): mean over all 6 tasks of that tier
    assert b.key("case_routing", "b2b", "cheap") == ("org", "b2b", "cheap")
    assert abs(b.predict("case_routing", "b2b", "cheap") - 3 / 6) < 1e-12
    assert abs(b.predict("knowledge_qa", "b2b", "mid") - 5 / 6) < 1e-12
    # unseen org -> tier mean; unseen tier -> global mean; tier=None -> state baseline
    assert b.key("lead_routing", "b2c", "frontier") == ("cat", "lead_routing", "frontier")
    assert b.key("case_routing", "b2c", "frontier") == ("tier", "frontier")
    assert b.predict("x", "y", "nope") == np.mean(q)
    s = CRMBuckets(3).fit([c for c, _ in ctx], [o for _, o in ctx], [None] * len(q), q)
    assert s.key("lead_routing", "b2b", None) == ("cat", "lead_routing", None)
    assert s.key("knowledge_qa", "b2b", None) == ("cat", "knowledge_qa", None)   # 3 arms = 3 samples
    assert CRMBuckets(4).fit([c for c, _ in ctx], [o for _, o in ctx], [None] * len(q), q).key(
        "knowledge_qa", "b2b", None) == ("org", "b2b", None)


def test_lookup_table_and_folds():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    assert T.task_ids == [t for t, _ in TASKS] and T.Q.shape == (6, 3)
    assert T.cats[3] == "case_routing" and np.allclose(T.C[5], [0.03, 0.07, 0.15], atol=1e-5)
    q, c = T.value(T.oracle())
    assert q == 1.0 and abs(c - (T.C[:3, 0].sum() + T.C[3:5, 1].sum() + T.C[5, 2]) / 6) < 1e-12
    q, c = T.value(T.oracle(lam=100.0))          # at a high price the oracle takes cheap everywhere
    assert q == 0.5 and abs(c - T.C[:, 0].mean()) < 1e-12
    folds = kfold_tasks(T, 3, seed=1)
    assert sorted(t for _, te in folds for t in te) == sorted(T.task_ids)
    for tr, te in folds:
        assert not set(tr) & set(te) and len(tr) + len(te) == 6
        assert sum(T.cats[T.task_ids.index(t)] == "lead_routing" for t in te) == 1     # stratified
    one = fixed_arm_log(eps, seed=3)
    assert len(one) == 6 and len({e.task_id for e in one}) == 6


def test_cgrpo_lookup_meets_budget_and_learns_lambda():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    loose, lam_loose = train_cgrpo_lookup(T, budget=0.50, iters=300, seed=0)
    q, c = T.value(T.probs(loose))
    assert lam_loose < 0.5 and q > 0.9, (lam_loose, q, c)
    tight, lam_tight = train_cgrpo_lookup(T, budget=0.045, iters=400, seed=0)
    q2, c2 = T.value(T.probs(tight))
    assert lam_tight > lam_loose and c2 <= 0.045 + 0.01, (lam_tight, q2, c2)
    P = T.probs(tight)
    assert P[:3, 0].mean() > 0.5, P[:3]           # lead_routing: cheap already has quality 1
    assert tight.mix and tight.meta["method"] == "cgrpo_lookup"


def test_awr_first_step_prefers_cheap_where_it_is_enough():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    pol = train_awr(eps, DECISION_ACTIONS, lam=5.0, first_step_only=True, legal=DECISION_ACTIONS,
                    n_features=N_FEATURES, serve_eps=0.0)
    P = T.probs(pol)
    assert P[0].argmax() == 0 and P[5].argmax() == 2, P      # lead_routing -> cheap, knowledge_qa -> frontier
    assert pol.meta["n_records"] == 18


def test_bandit_gate_estimators():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    pol, _ = train_cgrpo_lookup(T, budget=0.06, iters=200, seed=1)
    q, c = T.value(T.probs(pol))
    est = estimate_bandit(eps, pol, boot=30)
    for e in ("ips", "snips", "dr"):         # every arm logged: all three estimators are exact
        assert abs(est["quality"][e][0] - q) < 1e-9 and abs(est["cost"][e][0] - c) < 1e-9
    assert est["n"] == 18 and est["n_tasks"] == 6
    one = fixed_arm_log(eps, seed=0)
    e1 = estimate_bandit(one, pol, boot=30)
    lo, hi = e1["quality"]["dr"][1], e1["quality"]["dr"][2]
    assert lo <= e1["quality"]["dr"][0] <= hi
    g = gate(one, pol, "behavior", 0.06, tol=0.005, boot=30)
    assert g["decision"] in ("PROMOTE", "REVERT") and "n_tasks" in g["candidate"]


def test_reprice_from_spend_ledger():
    eps, spend = fake_log()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "spend.jsonl")
        extra = dict(spend[0], usd=0.5)            # a sunk-cost record from an interrupted episode
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in [extra] + spend))
        same = reprice_episodes(eps, p, {})
        assert all(abs(a.total_cost - b.total_cost) < 1e-12 for a, b in zip(eps, same))
        up = reprice_episodes(eps, p, {"frontier": 1.5})
        for a, b in zip(eps, up):
            f = 1.5 if a.meta["arm"] == "frontier" else 1.0
            assert abs(b.total_cost - f * a.total_cost) < 1e-12 and b.quality == a.quality
            assert abs(sum(s.cost for s in b.steps) - b.total_cost) < 1e-12
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in spend[1:]))
        try:
            reprice_episodes(eps, p, {"mid": 0.5})
            raise AssertionError("a step without a spend record must raise")
        except ValueError:
            pass


def test_phase2_category_oracle_labels():
    from pace.llm.crm import category_oracle, oracle_tier, sft_rows_crm
    from pace.llm.env_text import parse_action
    from pace.llm.rewards import LookupReward
    eps, _ = fake_log()
    lines = [json.dumps(e.to_dict()) for e in eps]
    L = LookupReward(lines)
    train = ["b2b-1", "b2b-2", "b2b-4", "b2b-6"]
    assert oracle_tier(L, train, "lead_routing", 0.0) == "cheap"           # tie at q = 1 -> the cheaper tier
    assert oracle_tier(L, train, "case_routing", 1.0) == "mid"
    assert oracle_tier(L, train, "knowledge_qa", 1.0) == "frontier"
    assert oracle_tier(L, train, "knowledge_qa", 20.0) == "cheap"         # 1 - 20 * 0.15 < 0 - 20 * 0.03
    o = category_oracle(L, train, 0.0)
    assert set(o) == {"lead_routing", "case_routing", "knowledge_qa", None}
    rows = sft_rows_crm(L, train, 0.10, lam_grid=[0.0, 1.0, 20.0])
    assert len(rows) == 12
    for r in rows:
        t = json.loads(r["task"])
        assert t["task_id"] in train and f"λ={r['lam']:g} per USD" in r["user"]
        assert parse_action(r["completion"], [(k, "CONTINUE") for k in TIERS])[0] == oracle_tier(
            L, train, t["category"], r["lam"])
        assert "cheap $0.0250" in r["user"]          # tier costs shown = the TRAINING tasks' means


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
