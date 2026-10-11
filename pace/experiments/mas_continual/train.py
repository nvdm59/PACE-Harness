"""Trainers and evaluation for the continual orchestrator.

train_cgrpo_replay   constrained GRPO on replay from logged state snapshots: the same algorithm as
                     `pace.train_offline.train_cgrpo` (group-relative advantages, learned λ by the
                     PI-damped dual ascent `_dual_step` on an EMA of full-episode cost against the
                     budget, rollouts sampled from the ε-mixed served policy with the ε-score
                     correction, iterate averaging over the second half) with one difference: that
                     function hard-codes the simulator feature map of pace.features, whereas the MAS
                     harness owns its own (`MASSimHarness.featurize`). Nothing else is re-implemented:
                     the policy, its gradient, the dual step and the snapshot mix are the core's.
                     "Replay" means re-rolling from a logged state under the CURRENT harness stage,
                     not replaying old outcomes: a state logged under stage g is a valid start under
                     stage g+1 if its calls are still available there (states that used a withdrawn
                     capability cannot occur, because the harness only grows).
train_awr_day        `pace.train_offline.train_awr` on the day's log (AWR + IPS, fixed price λ).
true_eval            Monte Carlo value of a policy on a fixed task set under one stage, with the
                     simulator's expected quality (the sim-only ground truth the gate is judged by).
"""
from __future__ import annotations

import random

import numpy as np

from ...controller import Controller
from ...harness.base import Harness
from ...trace import Episode
from ...train_offline import _dual_step, train_awr
from .simenv import ACTIONS, N_FEATURES, MASSimHarness, SimTask


def new_policy(init: Controller | None, serve_eps: float, actions: list[str] = ACTIONS) -> Controller:
    if init is None:
        pol = Controller(actions, N_FEATURES)
    else:
        pol = init.copy()
        pol.ensure_actions(actions)
    pol.mix = None
    pol.eps = serve_eps
    return pol


def rollout(h: Harness, s, pol: Controller, rng: random.Random, use_mix: bool = False):
    """Finish an episode from state `s` with the policy as served (eps included). Returns
    (episode [(x, mask, a)], terminal state)."""
    traj = []
    while not s.done:
        legal = h.legal_actions(s)
        x = h.featurize(s)
        a, _ = pol.act(x, legal, rng, use_mix=use_mix)
        traj.append((x, pol.mask(legal), a))
        s, _, _, _ = h.step(s, pol.actions[a])
    return traj, s


def train_cgrpo_replay(snapshots: list[dict], harness: MASSimHarness, budget: float, iters: int = 60, batch: int = 16,
                       group: int = 6, lr: float = 1.5, lr_lam: float = 20.0, kp: float = 40.0, lam0: float = 0.0,
                       ema: float = 0.9, snap_every: int = 3, init: Controller | None = None, serve_eps: float = 0.1,
                       seed: int = 0, log: list | None = None) -> tuple[Controller, float]:
    """Returns (policy with iterate mix, λ̄ over the averaged half). Half of every batch is drawn from
    episode starts (step 0), whose full episode cost feeds the dual variable; lr_lam / kp are per
    dollar of cost gap (CRMArena-scale costs, as in train_cgrpo_lookup)."""
    rng = random.Random(seed)
    pol = new_policy(init, serve_eps)
    starts0 = [s for s in snapshots if s["step"] == 0]
    if not starts0:
        raise ValueError("replay needs episode-start snapshots")
    lam_i, lam, cost_ema = lam0, lam0, None
    lam_sum, n_avg = 0.0, 0
    snaps: list[np.ndarray] = []
    for it in range(iters):
        picks = [rng.choice(starts0) for _ in range(batch // 2)] + [rng.choice(snapshots) for _ in range(batch - batch // 2)]
        X, M, A, ADV, full_costs, quals = [], [], [], [], [], []
        for snap in picks:
            base = harness.restore(snap)
            rows, rs = [], []
            for _ in range(group):
                traj, s = rollout(harness, base, pol, rng)
                q = harness.expected_quality(s)
                rs.append(q - lam * (s.cost_so_far - base.cost_so_far))
                rows.append(traj)
                quals.append(q)
                if base.step == 0:
                    full_costs.append(s.cost_so_far)
            b = sum(rs) / group
            for traj, r in zip(rows, rs):
                for x, m, a in traj:
                    X.append(x); M.append(m); A.append(a); ADV.append(r - b)
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
    pol.meta.update({"method": "cgrpo_replay", "budget": budget, "lambda": lam_bar, "lambda_final": lam, "iters": iters,
                     "group": group, "batch": batch, "n_snapshots": len(snaps), "serve_eps": serve_eps,
                     "harness_version": harness.version, "n_starts": len(snapshots)})
    return pol, lam_bar


def train_awr_day(episodes: list[Episode], lam: float, init: Controller | None, serve_eps: float = 0.1,
                  epochs: int = 200, beta: float = 0.25) -> Controller:
    """AWR + IPS on logged decisions (reward = quality − λ·cost-to-go), warm-started from the
    incumbent. The core trainer's per-bucket baseline uses pace.features.bucket_key, which reads the
    base block the MAS layout shares, so it applies unchanged."""
    pol = train_awr(episodes, ACTIONS, lam=lam, beta=beta, epochs=epochs, init=new_policy(init, serve_eps),
                    serve_eps=serve_eps, n_features=N_FEATURES)
    pol.meta.update({"method": "awr_ips", "lambda": lam})
    return pol


def true_eval(h: MASSimHarness, policy, tasks: list[SimTask], seed: int = 0, eps: float | None = None,
              legal_filter: list[str] | None = None, lam_j: float = 2.0) -> dict:
    """Expected quality / cost / J of `policy` on `tasks` under harness `h`. `legal_filter`
    restricts the legal set to the actions the policy knows (a frozen policy ignores capabilities
    shipped after it was trained). J = quality − lam_j · cost (USD)."""
    rng = random.Random(seed)
    saved, h.rng = h.rng, random.Random(seed + 1)      # the harness draws from a dedicated stream: the
    try:                                                # evaluation is reproducible whatever ran before
        q = c = 0.0
        n_calls = 0
        for t in tasks:
            s = h.reset(t)
            while not s.done:
                legal = h.legal_actions(s)
                if legal_filter is not None:
                    legal = [a for a in legal if a in legal_filter] or legal
                a, _ = policy.act(h.featurize(s), legal, rng, eps)
                s, _, _, _ = h.step(s, policy.actions[a])
            q += h.expected_quality(s)
            c += s.cost_so_far
            n_calls += s.extra["st"].n_calls
    finally:
        h.rng = saved
    n = len(tasks)
    return {"quality": q / n, "cost": c / n, "J": q / n - lam_j * c / n, "calls": n_calls / n, "n": n}
