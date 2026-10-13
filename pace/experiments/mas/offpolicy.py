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
               of the trajectory stays as logged (a one-step policy-improvement quantity), not
               the value of the full policy: it is biased by the downstream decisions the
               candidate would also change, in exchange for ratios bounded by |legal| / eps.
               Unit of the per-decision estimators (`unit=`):
               "decision"  (default, the historical behaviour) every logged decision row has weight 1,
                           so an episode with n decisions counts n times and its terminal quality is
                           repeated n times. Estimands:
                             quality:    E_{decisions}[ Q(x_t, pi) ], a decision-weighted mean over
                                         the logged decision distribution, which over-weights long
                                         episodes (a 10-decision episode weighs 10x a 1-decision one);
                             cost_to_go: the mean, over decision rows, of the cost-to-go from the
                                         intervened decision. Cost-to-go values of one episode
                                         overlap (step k's cost is in every row t <= k), so this is
                                         NOT a cost per task.
               "task"      each episode's rows are weighted 1 / n_decisions(episode), so every task
                           counts once (rows of an episode jointly carry unit mass). Estimands:
                             quality:    E_{tasks}[ mean_{t in episode} Q(x_t, pi) ], the task-average
                                         of the one-decision-intervention value; lengthening an
                                         episode without changing its outcome leaves it unchanged;
                             cost_to_go: the TOTAL episode cost per task under the intervention,
                                         prefix_t + cost-to-go(x_t, pi), where prefix_t is everything
                                         the episode spent before decision t (including any cost not
                                         attributed to a logged step, e.g. the policy-independent
                                         planner prefix: total_cost - sum of step costs). The direct
                                         model is fitted on the cost-to-go; the logged prefix is a
                                         known offset. Averaged task-weighted as for quality.
               Task-unit forms: SNIPS sum_t u_t w_t r_t / sum_t u_t w_t, DR (1/N_tasks) sum_t u_t
               [prefix_t + sum_a pi Qhat + w_t (r_t - Qhat_t)], u_t = 1/n_decisions; ESS is Kish over
               tasks on the per-task weight sum_t u_t w_t (n = n_tasks for this unit).
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


UNITS = ("decision", "task")


def unit_weights(episodes: list[Episode], unit: str = "decision") -> np.ndarray:
    """Row weight of every flattened decision (decision_table order): 1 per row ("decision") or
    1 / n_decisions of its episode ("task", every episode carries total mass 1)."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    return np.array([1.0 if unit == "decision" else 1.0 / len(e.steps) for e in episodes for _ in e.steps])


def cost_prefix(episodes: list[Episode]) -> np.ndarray:
    """Per decision row: everything the episode spent before that decision, i.e. the costs of the
    earlier logged steps plus any episode cost not attributed to a step (total_cost - sum of step
    costs, e.g. a policy-independent planner prefix). prefix_t + cost_to_go_t = total_cost."""
    out = []
    for e in episodes:
        c = [s.cost for s in e.steps]
        extra = float(e.total_cost) - float(sum(c))
        out += list(extra + np.concatenate([[0.0], np.cumsum(c)[:-1]])) if c else []
    return np.array(out, float)


def _point_per_decision(ix, W, R, P, K, A, nA, U=None, OFF=None, n_units=None):
    w, r = W[ix], R[ix]
    Q = _dm([K[i] for i in ix], A[ix], r, nA)
    terms = (P[ix] * Q).sum(1) + w * (r - Q[np.arange(len(ix)), A[ix]])
    if U is None:                                   # decision unit: every row weight 1
        snips = float((w * r).sum() / max(w.sum(), 1e-12))
        dr = float(terms.mean())
        return snips, dr
    u, off = U[ix], OFF[ix]
    snips = float((u * w * (r + off)).sum() / max((u * w).sum(), 1e-12))
    dr = float((u * (off + terms)).sum() / n_units)
    return snips, dr


def per_decision(episodes: list[Episode], policy: Controller, reward="quality", boot: int = 200, seed: int = 0,
                 unit: str = "decision") -> dict:
    """Per-decision SNIPS / DR with episode-bootstrap 90% CIs. `unit` = "decision" (rows weighted
    equally; historical default) or "task" (each episode's rows weighted 1/n_decisions; with
    reward="cost_to_go" the estimand becomes total episode cost = prefix + cost-to-go). See the
    module docstring for the estimand of each."""
    U = unit_weights(episodes, unit)
    ei, W, R, P, K, A, pol = decision_table(episodes, policy, reward)
    nA = len(pol.actions)
    n = len(W)
    n_ep = len(episodes)
    if unit == "task":
        OFF = cost_prefix(episodes) if reward == "cost_to_go" else np.zeros(n)
        kw = lambda: {"U": U, "OFF": OFF, "n_units": n_ep}
    else:
        OFF = np.zeros(n)
        kw = lambda: {}
    ix_all = np.arange(n)
    pt = _point_per_decision(ix_all, W, R, P, K, A, nA, **kw())
    rows_of = defaultdict(list)
    for j, i in enumerate(ei):
        rows_of[i].append(j)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        ix = np.concatenate([rows_of[i] for i in rng.integers(0, n_ep, n_ep)])
        bs.append(_point_per_decision(ix, W, R, P, K, A, nA, **kw()))
    bs = np.array(bs)
    if unit == "task":
        behavior = float((U * (R + OFF)).sum() / n_ep)
        estimand = ("task-weighted one-decision-intervention quality" if reward == "quality" else
                    "total episode cost per task (prefix + cost-to-go)" if reward == "cost_to_go" else
                    "task-weighted mean of the custom reward")
        w_eff = np.bincount(ei, weights=U * W, minlength=n_ep)      # per-task weight sum_t u_t w_t
    else:
        behavior = float(R.mean())
        estimand = ("decision-weighted one-decision-intervention quality" if reward == "quality" else
                    "mean cost-to-go over decision rows (not a per-task cost)" if reward == "cost_to_go" else
                    "decision-weighted mean of the custom reward")
        w_eff = W
    return {"n": n if unit == "decision" else n_ep, "n_decisions": n, "n_tasks": n_ep, "unit": unit, "estimand": estimand, "ess": ess(w_eff),
            "max_ratio": float(W.max()), "behavior": behavior,
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
