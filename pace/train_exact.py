"""Exact-gradient baselines on a fully observed first-step lookup table (no sampling).

On a counterfactual CRMArena log every tier of every task is logged (pace.lookup.LookupTable),
so the Lagrangian objective of a first-step tier policy on the training tasks is a finite sum,

    L(theta, lam) = mean_i sum_a pi_theta(a | x_i) [Q(i, a) - lam * C(i, a)],

and its gradient can be computed exactly instead of estimated from sampled re-rolls:

    dL/dtheta = mean_i sum_a pi(a | x_i) (r_ia - rbar_i) grad log pi(a | x_i),
    r_ia = Q(i, a) - lam * C(i, a),  rbar_i = sum_a pi(a | x_i) r_ia,

i.e. the GRPO estimator of train_offline.train_cgrpo_lookup with an infinite group (the
group-mean baseline becomes the exact on-policy value, which leaves the expectation unchanged
because sum_a pi grad log pi = 0). Two baselines live here:

`train_exact_lookup`  the constrained trainer of train_cgrpo_lookup (same Controller class, same
                      budget, same PI-damped dual ascent on an EMA of cost, same iterate
                      averaging over the second half) with that exact gradient. With
                      batch=None every iteration uses every training task (deterministic; the
                      seed is unused); with batch=k it samples k tasks per iteration (seeded)
                      but still sums exactly over the three arms of each.
`category_lp`         the category-conditioned budgeted LP on training-fold means, solved
                      exactly by the greedy hull method (see its docstring).
"""
from __future__ import annotations

import numpy as np

from .controller import Controller
from .train_offline import _dual_step, _new_policy


def exact_objective(pol: Controller, X: np.ndarray, Q: np.ndarray, C: np.ndarray, lam: float,
                    eps: float = 0.0) -> float:
    """mean_i sum_a pi_eps(a | x_i) (Q - lam C) at the current W (no iterate mix)."""
    M = np.ones(Q.shape, bool)
    P = pol.probs_batch(X, M, eps=eps, use_mix=False)
    return float((P * (Q - lam * C)).sum(1).mean())


def exact_grad(pol: Controller, X: np.ndarray, Q: np.ndarray, C: np.ndarray, lam: float,
               eps: float = 0.0) -> np.ndarray:
    """Exact dL/dW of `exact_objective` (same shape as pol.W), through the soft logit clip.
    The served policy is (1 - eps) softmax + eps/3; the uniform part carries no gradient, so
    the gradient of the served objective is (1 - eps) times that of the softmax one."""
    n, k = Q.shape
    M = np.ones((n, k), bool)
    P = pol.probs_batch(X, M, eps=0.0, use_mix=False)                  # bare softmax
    R = Q - lam * C
    coef = P * (R - (P * R).sum(1, keepdims=True))                      # pi_a (r_a - rbar)
    Xr, Mr = np.repeat(X, k, axis=0), np.repeat(M, k, axis=0)
    Ar = np.tile(np.arange(k), n)
    return (1.0 - eps) * pol.grad_logits_batch(Xr, Mr, Ar, coef.ravel()) / n


def train_exact_lookup(table, budget: float, iters: int = 300, batch: int | None = None,
                       lr: float = 1.5, lr_lam: float = 20.0, kp: float = 40.0, lam0: float = 0.0,
                       ema: float = 0.9, snap_every: int = 5, serve_eps: float = 0.0, seed: int = 0,
                       init: Controller | None = None, log: list | None = None) -> tuple[Controller, float]:
    """train_cgrpo_lookup with the exact policy gradient (no sampled actions). Same defaults for
    lr, lr_lam, kp, ema, snap_every and iterate averaging, so the only change is the gradient
    estimator (and, with batch=None, the task minibatch). The dual variable tracks an EMA of the
    EXACT expected cost of the current iterate on the batch. Returns (policy, lam_bar).
    pol.meta['n_lookups'] counts the (task, arm) table reads used by the policy gradient."""
    from .lookup import DECISION_ACTIONS
    rng = np.random.default_rng(seed)
    if init is not None:
        pol = _new_policy(DECISION_ACTIONS, init, serve_eps)
    else:
        pol = Controller(DECISION_ACTIONS, table.X.shape[1], eps=serve_eps)
    pol.mix = None
    n = len(table)
    k = len(DECISION_ACTIONS)
    M = np.ones(k, bool)
    lam_i, lam, cost_ema = lam0, lam0, None
    lam_sum, n_avg, n_lookups = 0.0, 0, 0
    snaps: list[np.ndarray] = []
    for it in range(iters):
        ix = np.arange(n) if batch is None else rng.choice(n, size=min(batch, n), replace=False)
        X, Q, C = table.X[ix], table.Q[ix], table.C[ix]
        g = exact_grad(pol, X, Q, C, lam, serve_eps)
        P = pol.probs_batch(X, np.tile(M, (len(ix), 1)), eps=serve_eps, use_mix=False)   # served, pre-step
        c_batch = float((P * C).sum(1).mean())
        q_batch = float((P * Q).sum(1).mean())
        n_lookups += Q.size
        pol.W += lr * g
        cost_ema = c_batch if cost_ema is None else ema * cost_ema + (1 - ema) * c_batch
        lam_i, lam = _dual_step(lam_i, cost_ema, budget, lr_lam, kp)
        if it >= iters // 2:
            lam_sum += lam; n_avg += 1
            if (it - iters // 2) % snap_every == 0:
                snaps.append(pol.W.copy())
        if log is not None:
            log.append({"iter": it, "lambda": round(lam, 4), "cost_ema": round(cost_ema, 5),
                        "cost_batch": round(c_batch, 5), "quality_batch": round(q_batch, 4)})
    pol.mix = snaps
    lam_bar = lam_sum / max(n_avg, 1)
    pol.meta.update({"method": "exact_lookup", "budget": budget, "lambda": lam_bar, "lambda_final": lam,
                     "iters": iters, "batch": batch, "n_snapshots": len(snaps), "n_tasks": n,
                     "lr_lam": lr_lam, "kp": kp, "serve_eps": serve_eps, "n_lookups": n_lookups})
    return pol, lam_bar


def _hull(c: np.ndarray, q: np.ndarray) -> list[int]:
    """Indices of the upper-left convex hull of points (c, q) from the cheapest point upward:
    the LP-efficient tiers of one category (every other tier is dominated by a mix of two
    hull tiers). Ties in cost keep the higher quality."""
    order = sorted(range(len(c)), key=lambda j: (c[j], -q[j]))
    pts: list[int] = []
    for j in order:
        if pts and c[j] == c[pts[-1]]:
            continue                                    # same cost, lower quality: dominated
        if pts and q[j] <= q[pts[-1]]:
            continue                                    # costlier and no better: dominated
        while len(pts) >= 2:
            a, b = pts[-2], pts[-1]
            # drop b if it lies on or below the chord a -> j (not concave)
            if (q[b] - q[a]) * (c[j] - c[a]) <= (q[j] - q[a]) * (c[b] - c[a]):
                pts.pop()
            else:
                break
        pts.append(j)
    return pts


def category_lp(cats: list[str], Q: np.ndarray, C: np.ndarray, budget: float) -> dict:
    """Category-conditioned budgeted LP on training means:

        max_p   sum_c n_c sum_t p[c, t] Qbar(c, t)
        s.t.    sum_c n_c sum_t p[c, t] Cbar(c, t) <= budget * N,   p[c, .] in the simplex,

    with Qbar, Cbar the per-(category, tier) means of the rows given and n_c the category counts.

    Solved exactly without an LP solver (scipy is not a dependency). This is the LP relaxation
    of a multiple-choice knapsack, whose structure makes a greedy exact (Sinha & Zemel 1979;
    Zemel 1980): (1) in each category only the upper convex hull of its (Cbar, Qbar) points can
    carry mass at an optimum (a point under the hull is dominated by a mix of two hull points
    of no greater cost); (2) start every category at its cheapest tier; moving mass along the
    hull from vertex h to h+1 in category c buys quality at the constant rate
    s = (Qbar_h+1 - Qbar_h) / (Cbar_h+1 - Cbar_h), weight n_c, and by concavity these rates
    decrease along each hull; (3) the problem is then a fractional knapsack over the hull
    segments (value n_c dQ, weight n_c dC), solved exactly by taking segments in decreasing
    rate order while they have positive rate and fit, splitting the last one. By LP duality
    the multiplier mu of the constraint equals the rate of the split segment when the budget
    binds and 0 when every positive-rate segment fits; at most one category ends up randomized.

    If the cheapest tiers alone exceed the budget the LP is infeasible; the cheapest tier per
    category (the minimum-cost solution) is returned with feasible=False.

    Returns {'cats': [...], 'p': (n_cats, 3), 'mu': float, 'value_q', 'value_c', 'feasible'}."""
    cv = sorted(set(cats))
    ci = np.array([cv.index(c) for c in cats])
    k = Q.shape[1]
    n_c = np.bincount(ci, minlength=len(cv)).astype(float)
    Qb = np.array([Q[ci == j].mean(0) for j in range(len(cv))])
    Cb = np.array([C[ci == j].mean(0) for j in range(len(cv))])
    N = float(n_c.sum())
    cap = budget * N
    hulls = [_hull(Cb[j], Qb[j]) for j in range(len(cv))]
    p = np.zeros((len(cv), k))
    for j, h in enumerate(hulls):
        p[j, h[0]] = 1.0
    spend = float(sum(n_c[j] * Cb[j, h[0]] for j, h in enumerate(hulls)))
    if spend > cap + 1e-15:
        return {"cats": cv, "p": p, "mu": float("inf"), "value_q": float((n_c[:, None] * p * Qb).sum() / N),
                "value_c": spend / N, "feasible": False}
    segs = []                                           # (rate, category, from, to)
    for j, h in enumerate(hulls):
        for a, b in zip(h[:-1], h[1:]):
            segs.append(((Qb[j, b] - Qb[j, a]) / (Cb[j, b] - Cb[j, a]), j, a, b))
    segs.sort(key=lambda s: (-s[0], cv[s[1]], s[2]))   # deterministic tie order
    mu = 0.0                                            # budget slack at the optimum -> mu = 0
    for rate, j, a, b in segs:
        if rate <= 0:
            break
        w = n_c[j] * (Cb[j, b] - Cb[j, a])
        if spend + w <= cap:
            p[j, a] -= 1.0; p[j, b] += 1.0              # the whole segment: a -> b
            spend += w
            continue
        f = (cap - spend) / w                           # split segment: category j randomizes
        p[j, a] -= f; p[j, b] += f
        spend = cap
        mu = rate                                       # binding budget: mu = rate of the split segment
        break
    p = np.clip(p, 0.0, 1.0)
    return {"cats": cv, "p": p, "mu": float(mu), "value_q": float((n_c[:, None] * p * Qb).sum() / N),
            "value_c": float((n_c[:, None] * p * Cb).sum() / N), "feasible": True}


def lp_probs(lp: dict, cats: list[str], fallback: np.ndarray) -> np.ndarray:
    """Tier probabilities (n, 3) of the LP policy for tasks of categories `cats`; a category
    absent from the training rows gets `fallback` (a length-3 probability vector)."""
    pos = {c: i for i, c in enumerate(lp["cats"])}
    return np.array([lp["p"][pos[c]] if c in pos else fallback for c in cats])
