"""RAO-style delegation credit for the continual orchestrator's constrained GRPO.

Recursive Agent Optimization (arXiv 2605.06639) trains a recursive agent system by giving every
node of the call tree its own reward and passing credit from children to the parent that
delegated to them. Here the call tree is one level deep: the orchestrator's `delegate@<agent>`
decisions spawn data-worker sub-calls. The committed trainer (`train.train_cgrpo_replay`) gives
every decision of a rollout the same priced episode outcome R = quality − λ·cost-to-go and a
group-mean baseline. With `credit = "rao"` the node reward of a delegate decision becomes

    r_t = R  +  λ_d · mean(success of the sub-call(s) that decision spawned)

(averaging, not counting: a decision that spawned two successful sub-calls earns the same bonus
as one that spawned one), every other decision keeps r_t = R, and every decision of rollout k
uses the same root-anchored leave-one-out baseline b_k = mean_{j≠k} R_j over the group (the
shaping term is NOT in the baseline: it is credit, not a re-centred return). A delegate decision
that spawned no sub-call (the plan was finished or the call cap reached) gets no bonus.

A sub-call's success is the simulator's latent flag of the worker step (`good`: it returned rows
and they were the right rows). On a real harness the same slot would be filled by a per-call
verifier verdict or a judge on the worker's output; the simulator's flag is the noiseless version
of that signal, so this is an upper bound on what the shaping could carry.

Everything else (rollouts from replayed snapshots under the current stage, the ε-score
correction, the PI-damped dual step on full-episode cost, iterate averaging) is the committed
trainer's, line for line.
"""
from __future__ import annotations

import random
from collections import Counter

import numpy as np

from ...controller import Controller
from ...harness.base import Harness
from ...train_offline import _dual_step
from .simenv import MASSimHarness, split_action
from .train import new_policy

CREDITS = ("none", "rao")


# ------------------------------------------------------------------ pure arithmetic
def loo_baseline(rs: list[float]) -> list[float]:
    """Leave-one-out mean of the group: b_k = (Σ_j r_j − r_k) / (G − 1)."""
    n = len(rs)
    if n < 2:
        return [0.0] * n
    s = float(sum(rs))
    return [(s - r) / (n - 1) for r in rs]


def node_rewards(root: float, ops: list[str], sub_success: list[list[float]], lambda_d: float, credit: str = "rao") -> list[float]:
    """Per-decision rewards of one rollout. `ops[t]` is the operation of decision t (delegate,
    verify, ...) and `sub_success[t]` the success flags of the sub-calls decision t spawned."""
    if credit not in CREDITS:
        raise ValueError(credit)
    out = []
    for op, ss in zip(ops, sub_success):
        bonus = lambda_d * float(np.mean(ss)) if (credit == "rao" and op == "delegate" and len(ss) > 0) else 0.0
        out.append(root + bonus)
    return out


def group_advantages(roots: list[float], ops: list[list[str]], sub_success: list[list[list[float]]], lambda_d: float,
                     credit: str = "rao") -> list[list[float]]:
    """Advantages for a group of rollouts from one start. credit="none" is the committed trainer
    (group-mean baseline, one advantage per rollout); credit="rao" is the node reward above
    minus the root-anchored leave-one-out baseline."""
    if credit == "none":
        b = sum(roots) / len(roots)
        return [[r - b] * len(o) for r, o in zip(roots, ops)]
    base = loo_baseline(roots)
    return [[v - bk for v in node_rewards(r, o, ss, lambda_d, credit)] for r, o, ss, bk in zip(roots, ops, sub_success, base)]


# ------------------------------------------------------------------ delegation share
def delegation_counts(episodes) -> tuple[int, int, Counter]:
    """(delegate decisions, all decisions, delegate decisions by agent id) over served episodes."""
    n_dec, n_del, by = 0, 0, Counter()
    for e in episodes:
        for s in e.steps:
            n_dec += 1
            op, target = split_action(s.action_name)
            if op == "delegate":
                n_del += 1
                by[target] += 1
    return n_del, n_dec, by


def delegation_share(episodes) -> dict:
    """Delegation share of served decisions, and each data agent's share of ALL decisions."""
    n_del, n_dec, by = delegation_counts(episodes)
    d = max(n_dec, 1)
    return {"share": n_del / d, "by_agent": {a: by[a] / d for a in sorted(by)}, "n_decisions": n_dec}


def delegation_share_from_mix(action_mix: dict) -> float:
    """The same number from a days.jsonl `action_mix` (top-6 actions only, rounded to 0.001: a
    lower bound when a delegate action falls outside the top six)."""
    return float(sum(v for a, v in action_mix.items() if a.startswith("delegate@")))


# ------------------------------------------------------------------ trainer
def rollout_with_subcalls(h: Harness, s, pol: Controller, rng: random.Random):
    """Finish an episode from `s` with the served policy. Returns (steps [(x, mask, a)], ops,
    sub-call successes per decision, terminal state)."""
    steps, ops, n_before = [], [], []
    while not s.done:
        legal = h.legal_actions(s)
        x = h.featurize(s)
        a, _ = pol.act(x, legal, rng, use_mix=False)
        steps.append((x, pol.mask(legal), a))
        ops.append(split_action(pol.actions[a])[0])
        n_before.append(s.extra["st"].n_calls)
        s, _, _, _ = h.step(s, pol.actions[a])
    calls = s.extra["st"].calls
    bounds = n_before + [len(calls)]
    sub = [[1.0 if c.extra.get("good") else 0.0 for c in calls[bounds[t]:bounds[t + 1]] if c.role == "worker"]
           for t in range(len(ops))]
    return steps, ops, sub, s


def train_cgrpo_replay_shaped(snapshots: list[dict], harness: MASSimHarness, budget: float, iters: int = 60, batch: int = 16,
                              group: int = 6, lr: float = 1.5, lr_lam: float = 20.0, kp: float = 40.0, lam0: float = 0.0,
                              ema: float = 0.9, snap_every: int = 3, init: Controller | None = None, serve_eps: float = 0.1,
                              seed: int = 0, credit: str = "rao", lambda_d: float = 0.5,
                              log: list | None = None) -> tuple[Controller, float]:
    """`train.train_cgrpo_replay` with the per-decision advantages of `group_advantages`. With
    credit="none" it reproduces the committed trainer exactly (same random draws, same update)."""
    rng = random.Random(seed)
    pol = new_policy(init, serve_eps)
    starts0 = [s for s in snapshots if s["step"] == 0]
    if not starts0:
        raise ValueError("replay needs episode-start snapshots")
    lam_i, lam, cost_ema = lam0, lam0, None
    lam_sum, n_avg = 0.0, 0
    snaps: list[np.ndarray] = []
    n_del_bonus = 0
    for it in range(iters):
        picks = [rng.choice(starts0) for _ in range(batch // 2)] + [rng.choice(snapshots) for _ in range(batch - batch // 2)]
        X, M, A, ADV, full_costs, quals = [], [], [], [], [], []
        for snap in picks:
            base = harness.restore(snap)
            rows, roots, ops, subs = [], [], [], []
            for _ in range(group):
                st, op, sub, s = rollout_with_subcalls(harness, base, pol, rng)
                q = harness.expected_quality(s)
                roots.append(q - lam * (s.cost_so_far - base.cost_so_far))
                rows.append(st); ops.append(op); subs.append(sub)
                quals.append(q)
                if base.step == 0:
                    full_costs.append(s.cost_so_far)
            advs = group_advantages(roots, ops, subs, lambda_d, credit)
            for st, adv, op, sub in zip(rows, advs, ops, subs):
                for (x, m, a), v, o, ss in zip(st, adv, op, sub):
                    X.append(x); M.append(m); A.append(a); ADV.append(v)
                    n_del_bonus += int(credit == "rao" and o == "delegate" and len(ss) > 0)
        X, M, A, ADV = np.asarray(X), np.asarray(M), np.asarray(A), np.asarray(ADV)
        g = pol.grad_logits_batch(X, M, A, ADV * pol.eps_score_weights(X, M, A, serve_eps))
        pol.W += lr * g / (batch * group)
        c_batch = float(np.mean(full_costs))
        cost_ema = c_batch if cost_ema is None else ema * cost_ema + (1 - ema) * c_batch
        lam_i, lam = _dual_step(lam_i, cost_ema, budget, lr_lam, kp)
        if it >= iters // 2:
            lam_sum += lam; n_avg += 1
            if (it - iters // 2) % snap_every == 0:
                snaps.append(pol.W.copy())
        if log is not None:
            log.append({"iter": it, "lambda": round(lam, 4), "cost_ema": round(cost_ema, 5), "cost_batch": round(c_batch, 5),
                        "quality_batch": round(float(np.mean(quals)), 4)})
    pol.mix = snaps
    lam_bar = lam_sum / max(n_avg, 1)
    pol.meta.update({"method": "cgrpo_replay" if credit == "none" else "cgrpo_replay_rao", "budget": budget, "lambda": lam_bar,
                     "lambda_final": lam, "iters": iters, "group": group, "batch": batch, "n_snapshots": len(snaps),
                     "serve_eps": serve_eps, "harness_version": harness.version, "n_starts": len(snapshots),
                     "credit": credit, "lambda_d": lambda_d, "n_delegate_bonus": n_del_bonus})
    return pol, lam_bar
