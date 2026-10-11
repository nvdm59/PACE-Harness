"""Off-policy estimates for the MAS decision log, with effective sample sizes.

The orchestration log has 2-4 decisions per episode over 12 actions, logged by a uniform
exploration policy. The cumulative importance ratio of an episode-level estimator is then a
product of up to 4 ratios each as large as |legal| ~ 12, so on a 100-episode log a plain IPS
or clip-at-50 DR (pace.gate) is dominated by a handful of episodes and returns values outside
[0, 1]. Two estimators are used instead, and every estimate is reported with its effective
sample size ESS = (sum w)^2 / sum w^2 (Kish), which says how many logged units actually carry
the estimate; the report flags any estimate whose ESS is below 10% of n.

per_decision   each logged decision point is one contextual-bandit sample (context x_t, action
               a_t, behavior propensity mu_t). The single-step ratio w_t = pi(a_t | x_t) / mu_t
               is self-normalised (SNIPS) or combined with a bucketed direct model (DR):
                   SNIPS  sum_t w_t r_t / sum_t w_t
                   DR     mean_t [ sum_a pi(a|x_t) Qhat(x_t, a) + w_t (r_t - Qhat(x_t, a_t)) ]
               with r_t the episode's quality (outcome or quality_hat) or the cost-to-go from
               the decision, and Qhat(bucket, a) the mean of r over logged decisions in the
               same (pace.features.bucket_key, action) cell, falling back to the bucket mean
               and the global mean. This is the value of changing ONE decision while the rest
               of the episode stays as logged (a one-step policy-improvement quantity), not
               the value of the full policy: it is biased by the downstream decisions the
               candidate would also change, in exchange for ratios bounded by |legal| / eps.
episode_level  self-normalised IPS with the per-step ratio clipped at `clip` (default 5) before
               the product, i.e. a bounded-variance, biased estimate of the full-policy value.

90% CIs by bootstrap over episodes (each episode's decisions resampled together).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from ...controller import Controller
from ...features import bucket_key
from ...trace import Episode

STEP_CLIP = 5.0
ESS_WARN_FRAC = 0.10


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def _ci(samples) -> tuple[float, float]:
    return float(np.percentile(samples, 5)), float(np.percentile(samples, 95))


def cost_to_go(ep: Episode) -> list[float]:
    c = [s.cost for s in ep.steps]
    return list(np.cumsum(c[::-1])[::-1]) if c else []


def decision_table(episodes: list[Episode], policy: Controller, reward: str | Callable[[Episode], float]):
    """Flattened decisions: (episode index, w, r, pi row over actions, Q-bucket key, action index)."""
    pol = policy.copy()
    pol.ensure_actions(sorted({a for e in episodes for s in e.steps for a in (s.legal or [s.action_name])}))
    ei, W, R, P, K, A = [], [], [], [], [], []
    for i, e in enumerate(episodes):
        ctg = cost_to_go(e)
        for t, s in enumerate(e.steps):
            p = pol.probs(s.features, s.legal)
            a = pol.index(s.action_name)
            ei.append(i); W.append(p[a] / s.propensity); P.append(p); A.append(a); K.append(bucket_key(s.features))
            if reward == "quality":
                R.append(float(e.quality))
            elif reward == "cost_to_go":
                R.append(float(ctg[t]))
            else:
                R.append(float(reward(e)))
    return np.array(ei), np.array(W), np.array(R), np.array(P), K, np.array(A), pol


def _dm(K, A, R, nA: int):
    """(bucket, action) mean of R with bucket -> global fallback. Returns Qhat matrix (n, nA)."""
    sums, cnts = defaultdict(float), defaultdict(int)
    bs, bc = defaultdict(float), defaultdict(int)
    for k, a, r in zip(K, A, R):
        sums[(k, a)] += r; cnts[(k, a)] += 1; bs[k] += r; bc[k] += 1
    g = float(np.mean(R)) if len(R) else 0.0
    Q = np.empty((len(K), nA))
    for i, k in enumerate(K):
        for a in range(nA):
            Q[i, a] = sums[(k, a)] / cnts[(k, a)] if cnts[(k, a)] else (bs[k] / bc[k] if bc[k] else g)
    return Q


def _point_per_decision(ix, W, R, P, K, A, nA):
    w, r = W[ix], R[ix]
    snips = float((w * r).sum() / max(w.sum(), 1e-12))
    Q = _dm([K[i] for i in ix], A[ix], r, nA)
    dr = float((( P[ix] * Q).sum(1) + w * (r - Q[np.arange(len(ix)), A[ix]])).mean())
    return snips, dr


def per_decision(episodes: list[Episode], policy: Controller, reward="quality", boot: int = 200, seed: int = 0) -> dict:
    ei, W, R, P, K, A, pol = decision_table(episodes, policy, reward)
    nA = len(pol.actions)
    n = len(W)
    ix_all = np.arange(n)
    pt = _point_per_decision(ix_all, W, R, P, K, A, nA)
    rows_of = defaultdict(list)
    for j, i in enumerate(ei):
        rows_of[i].append(j)
    rng = np.random.default_rng(seed)
    n_ep = len(episodes)
    bs = []
    for _ in range(boot):
        ix = np.concatenate([rows_of[i] for i in rng.integers(0, n_ep, n_ep)])
        bs.append(_point_per_decision(ix, W, R, P, K, A, nA))
    bs = np.array(bs)
    return {"n": n, "unit": "decision", "ess": ess(W), "max_ratio": float(W.max()),
            "behavior": float(R.mean()),
            "snips": (pt[0],) + _ci(bs[:, 0]), "dr": (pt[1],) + _ci(bs[:, 1])}


def episode_level(episodes: list[Episode], policy: Controller, reward="quality", clip: float = STEP_CLIP,
                  boot: int = 200, seed: int = 0) -> dict:
    """Self-normalised IPS of the full policy with each per-step ratio clipped at `clip`."""
    pol = policy.copy()
    pol.ensure_actions(sorted({a for e in episodes for s in e.steps for a in (s.legal or [s.action_name])}))
    W, R = [], []
    for e in episodes:
        rho = 1.0
        for s in e.steps:
            p = pol.probs(s.features, s.legal)
            rho *= min(p[pol.index(s.action_name)] / s.propensity, clip)
        W.append(rho)
        R.append(float(e.quality) if reward == "quality" else float(e.total_cost) if reward == "cost" else float(reward(e)))
    W, R = np.array(W), np.array(R)
    pt = float((W * R).sum() / max(W.sum(), 1e-12))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        ix = rng.integers(0, len(W), len(W))
        bs.append(float((W[ix] * R[ix]).sum() / max(W[ix].sum(), 1e-12)))
    return {"n": len(W), "unit": "episode", "ess": ess(W), "max_ratio": float(W.max()), "clip": clip,
            "behavior": float(R.mean()), "snips": (pt,) + _ci(bs)}


def ess_warning(est: dict) -> str:
    frac = est["ess"] / max(est["n"], 1)
    return f"ESS {est['ess']:.1f} of {est['n']} ({100 * frac:.0f}%) — below 10%: unreliable" if frac < ESS_WARN_FRAC \
        else f"ESS {est['ess']:.1f} of {est['n']} ({100 * frac:.0f}%)"
