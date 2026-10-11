"""Regression tests for the estimator and trainer audit (CPU, no network, no torch).

Each test here pins down one property that was wrong or unguarded before the review pass:
  * the policy gradient of the eps-mixed SERVED policy (train_cgrpo sampled rollouts with eps
    but differentiated the bare softmax);
  * the constrained lookup trainer meets the budget for the policy as served, eps included;
  * repricing from the spend ledger keeps cache tokens and tolerates the ledger's 8-decimal
    rounding of usd;
  * the CRM context decoder follows the harness's feature layout;
  * K-fold splits never share a task between train and test;
  * the multi-step IPS / SNIPS estimators reproduce the on-policy mean when the candidate IS
    the behavior policy, and the bandit estimators are unbiased over redraws of a realized log.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pace.collect import collect, load_policy  # noqa: E402
from pace.controller import Controller  # noqa: E402
from pace.features import crm_context  # noqa: E402
from pace.gate import estimate, estimate_bandit, on_policy  # noqa: E402
from pace.harness import make_sim  # noqa: E402
from pace.harness.crmarena import CATEGORIES, FEATURE_NAMES, N_FEATURES, ORG_TYPES  # noqa: E402
from pace.lookup import LookupTable, fixed_arm_log, join_spend, kfold_tasks, reprice_episodes  # noqa: E402
from pace.prices import MODELS, TIERS, call_cost  # noqa: E402
from pace.train_offline import train_cgrpo_lookup  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_crm_offline import TASKS, fake_log  # noqa: E402


# ------------------------------------------------------------ policy gradient under eps
def test_eps_score_weights_make_the_policy_gradient_exact():
    """J(W) = sum_a pi_eps(a|x; W) r(a). The REINFORCE estimator under samples from pi_eps is
    E_{a ~ pi_eps}[r(a) grad log pi_eps(a)] = sum_a pi_eps(a) r(a) * factor(a) * grad log pi(a)
    with factor = (1 - eps) pi / pi_eps. Without the factor (what the trainer did before) the
    expectation is not the gradient of the served objective."""
    rng = np.random.default_rng(0)
    eps = 0.2
    W = rng.normal(size=(3, 2))
    x = np.array([1.0, -0.5])
    r = np.array([1.0, 0.2, -0.4])
    legal = ["a", "b", "c"]

    def J(Wm):
        return float(Controller(legal, 2, eps=eps, W=Wm).probs(x, legal) @ r)

    exact = np.zeros_like(W)
    h = 1e-6
    for i in range(3):
        for j in range(2):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += h; Wm[i, j] -= h
            exact[i, j] = (J(Wp) - J(Wm)) / (2 * h)
    pol = Controller(legal, 2, eps=eps, W=W)
    X = np.tile(x, (3, 1)); M = np.ones((3, 3), bool); A = np.arange(3)
    p_eps = pol.probs(x, legal)
    corrected = pol.grad_logits_batch(X, M, A, p_eps * r * pol.eps_score_weights(X, M, A, eps))
    naive = pol.grad_logits_batch(X, M, A, p_eps * r)
    assert np.allclose(corrected, exact, atol=1e-5), (corrected, exact)
    assert not np.allclose(naive, exact, atol=1e-3), "the uncorrected score is biased for the served policy"
    # eps = 0: the factor is 1 and the plain score is exact
    assert np.allclose(pol.eps_score_weights(X, M, A, 0.0), 1.0)


def test_cgrpo_lookup_meets_the_budget_as_served_with_eps():
    """With serve_eps > 0 the served policy carries a uniform component whose cost the budget
    must cover; the trainer samples from the served policy and tracks its cost."""
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    budget = 0.05
    pol, lam = train_cgrpo_lookup(T, budget=budget, iters=400, seed=0, serve_eps=0.2)
    assert pol.eps == 0.2 and pol.meta["serve_eps"] == 0.2
    q, c = T.value(T.probs(pol))                     # T.probs = the policy as served (eps included)
    assert c <= budget + 0.005, (q, c, lam)
    assert lam > 0.0


# ------------------------------------------------------------------- repricing join
def test_reprice_keeps_cache_tokens_and_tolerates_ledger_rounding():
    eps, spend = fake_log()
    # (a) a call that used prompt caching: its usd includes cache write / read tokens
    tgt = eps[0]                                     # b2b-1, cheap arm, step 0 = agent:cheap
    rec = next(r for r in spend if r["task_id"] == tgt.task_id and r["purpose"] == "agent:cheap")
    rec.update({"cache_write_tokens": 700, "cache_read_tokens": 300})
    rec["usd"] = call_cost(rec["model"], rec["in_tokens"], rec["out_tokens"], 700, 300)
    tgt.steps[0].cost = rec["usd"]
    tgt.total_cost = sum(s.cost for s in tgt.steps)
    # (b) the ledger stores usd rounded to 8 decimals while the trace keeps the full float
    other = eps[4]                                   # b2b-2, mid arm
    rec2 = next(r for r in spend if r["task_id"] == other.task_id and r["purpose"] == "agent:mid")
    true_cost = 0.0123456789                         # 10 decimals: rounds to 0.01234568 in the ledger
    other.steps[0].cost = true_cost
    other.total_cost = sum(s.cost for s in other.steps)
    rec2["usd"] = round(true_cost, 8)
    assert abs(rec2["usd"] - true_cost) > 1e-9       # the old 1e-9 tolerance would not have matched
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "spend.jsonl")
        open(p, "w").write("".join(json.dumps(r) + "\n" for r in spend))
        joined = join_spend(eps, p)
        assert joined[(0, 0)] is not None and joined[(4, 0)]["usd"] == rec2["usd"]
        up = reprice_episodes(eps, p, {"cheap": 2.0})
    assert abs(up[0].steps[0].cost - 2.0 * tgt.steps[0].cost) < 1e-12, "cache tokens must be repriced too"
    assert abs(up[0].total_cost - 2.0 * tgt.total_cost) < 1e-12
    # the rounded record reprices from its tokens (mid unchanged): back to call_cost of the tokens
    assert abs(up[4].steps[0].cost - call_cost(rec2["model"], rec2["in_tokens"], rec2["out_tokens"])) < 1e-12


# ------------------------------------------------------------------ feature layout
def test_crm_context_follows_the_harness_feature_layout():
    for cat in CATEGORIES[:3] + CATEGORIES[-1:]:
        for org in ORG_TYPES:
            x = [0.0] * N_FEATURES
            x[FEATURE_NAMES.index(f"cat={cat}")] = 1.0
            x[FEATURE_NAMES.index(f"org={org}")] = 1.0
            assert crm_context(x) == (cat, org)
    assert crm_context([0.0] * N_FEATURES) == ("unknown", "unknown")


# ---------------------------------------------------------------------- K-fold split
def test_kfold_never_shares_a_task_between_train_and_test():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    for k in (2, 3, 5):
        folds = kfold_tasks(T, k, seed=0)
        assert len(folds) == k
        tests = [t for _, te in folds for t in te]
        assert sorted(tests) == sorted(T.task_ids), "every task is held out exactly once"
        for tr, te in folds:
            assert not set(tr) & set(te) and sorted(tr + te) == sorted(T.task_ids)
            # the training table is built from training ids only
            Ttr = T.subset(tr)
            assert not set(Ttr.task_ids) & set(te)
    assert len(TASKS) == len(T)


# ------------------------------------------------------------- estimator sanity checks
def test_multistep_ips_and_snips_reproduce_the_on_policy_mean_for_the_behavior_policy():
    h = make_sim("v1", seed=21)
    beh = load_policy("uniform", h)
    log = collect(h, beh, 300, seed=21)
    est = estimate(log, beh, boot=20)
    truth = on_policy(log, boot=20)
    for m in ("quality", "cost"):
        assert abs(est[m]["ips"][0] - truth[m][0]) < 1e-9, (m, est[m]["ips"], truth[m])
        assert abs(est[m]["snips"][0] - truth[m][0]) < 1e-9, (m, est[m]["snips"], truth[m])
        assert est[m]["dr"][1] <= est[m]["dr"][0] <= est[m]["dr"][2]
    assert est["max_rho"] == 1.0


def test_bandit_estimators_are_unbiased_over_redraws_of_a_realized_log():
    eps, _ = fake_log()
    T = LookupTable.from_episodes(eps)
    pol, _ = train_cgrpo_lookup(T, budget=0.06, iters=150, seed=3)
    q_ex, c_ex = T.value(T.probs(pol))
    draws = {e: [] for e in ("ips", "snips", "dr")}
    for k in range(300):
        est = estimate_bandit(fixed_arm_log(eps, seed=k), pol, boot=1, seed=k)
        for e in draws:
            draws[e].append(est["quality"][e][0])
    assert abs(np.mean(draws["ips"]) - q_ex) < 0.04, (np.mean(draws["ips"]), q_ex)   # unbiased
    assert abs(np.mean(draws["dr"]) - q_ex) < 0.04, (np.mean(draws["dr"]), q_ex)
    assert np.std(draws["dr"]) <= np.std(draws["ips"]) + 1e-9, "DR should not be noisier than IPS here"
    assert set(MODELS) == set(TIERS)


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
