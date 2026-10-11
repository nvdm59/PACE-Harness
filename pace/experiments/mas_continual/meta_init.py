"""A meta-learned initial controller for the continual orchestrator (first-order MAML / Reptile).

The continual learner's deficit against the fixed never_verify mode was its cold start: it served
the uniform exploration policy until the gate's first promotion (day 13 in the committed seed).
"End-to-End Test-Time Training for Long Context" (arXiv 2512.23675) meta-learns the
initialization its test-time learner starts from, so that a few inner steps at test time are
enough. The same idea here: learn the controller weights W_0 (and the price λ_0) the continual
learner starts from, such that a few days of its own cGRPO on a harness stage lands somewhere good.

    φ ← 0 (the uniform policy), λ ← 0
    repeat `outer` times:
        for each meta-training stage g in META_STAGES (= g0..g3):
            W_g ← φ;  for `inner_days` days: serve `n_tasks` fresh tasks of stage g with W_g (ε-mixed),
                      train cGRPO on replay of those days warm-started from W_g (the committed trainer)
            W_g ← the iterate average of the last inner day's run
        φ ← φ + β · (mean_g W_g − φ);   λ ← mean_g λ̄_g          (β = 1: average the adapted weights)

What the meta-init sees, exactly (enforced by `audit` and the tests):
  * the simulator's harness DEFINITIONS (Params) of stages g0..g3 of the schedule; never g4;
  * tasks it samples itself with seeds `meta_task_seed(...)` >= META_SEED_BASE, disjoint from the
    run's held-out seeds (seed + 500 + g), daily-traffic seeds (seed·100000 + day) and oracle seeds;
  * its own rollouts' simulator expected quality and cost (the same signal the learner trains on).
It never reads a held-out task set of any stage, never the run's daily traffic, never an oracle.
It DOES see the definitions of stages that arrive later in the run (g1..g3): a deployed system has
no simulator of its future harness, so on real data the meta-training stages would have to be past
generations of this or another harness (earlier price sheets, agent pools, tool sets), with the
schedule's future stages excluded. That is the main caveat of the result.
"""
from __future__ import annotations

import random
import time

import numpy as np

from ...controller import Controller
from . import drift
from .simenv import ACTIONS, N_FEATURES, MASSimHarness, serve
from .train import train_cgrpo_replay

META_STAGES = (0, 1, 2, 3)
META_SEED_BASE = 7_000_000


def meta_task_seed(seed: int, outer: int, stage: int, day: int) -> int:
    return META_SEED_BASE + 100_000 * seed + 1_000 * outer + 10 * stage + day


def heldout_seeds(seed: int, n_stages: int = 5) -> set[int]:
    """The run's held-out task seeds (run_continual: `sample_tasks(n_heldout, seed=cfg.seed + 500 + g)`)."""
    return {seed + 500 + g for g in range(n_stages)}


def _snapshots(eps_) -> list[dict]:
    return [s.state for e in eps_ for s in e.steps if s.state is not None]


def meta_learn(budget: float = 0.090, eps: float = 0.10, schedule: str = "default", stages: tuple = META_STAGES,
               outer: int = 3, inner_days: int = 2, n_tasks: int = 96, iters: int = 60, batch: int = 16, group: int = 6,
               beta: float = 1.0, seed: int = 0, gens: list | None = None, log=print) -> tuple[Controller, dict]:
    """Returns (initial controller, report). `gens` defaults to the schedule's generations; only
    the indices in `stages` are ever instantiated."""
    t0 = time.time()
    gens = gens or drift.schedule(schedule)
    if any(g >= len(gens) - 1 for g in stages):
        raise ValueError("the last stage of the schedule is held out of meta-training")
    phi = np.zeros((len(ACTIONS), N_FEATURES))
    lam_phi = 0.0
    seen = {"stages": sorted(stages), "task_seeds": [], "n_episodes": 0}
    rounds = []
    for k in range(outer):
        adapted, lams = [], []
        for g in stages:
            gen = gens[g]
            h = MASSimHarness(gen.params, seed=META_SEED_BASE + 1_000 * seed + 10 * k + g, version=gen.version + "-meta")
            pol = Controller(ACTIONS, N_FEATURES, eps=eps, W=phi.copy(), meta={"name": "meta-inner", "lambda": lam_phi})
            lam = lam_phi
            for day in range(1, inner_days + 1):
                ts = meta_task_seed(seed, k, g, day)
                seen["task_seeds"].append(ts)
                logs = serve(h, h.sample_tasks(n_tasks, seed=ts), pol, seed=ts + 1, eps=eps, mode="meta")
                seen["n_episodes"] += len(logs)
                pol, lam = train_cgrpo_replay(_snapshots(logs), h, budget, iters=iters, batch=batch, group=group, init=pol,
                                              serve_eps=eps, lam0=lam, seed=ts + 2)
                pol.meta["lambda"] = lam
            W_g = np.mean(pol.mix, axis=0) if pol.mix else pol.W
            adapted.append(W_g); lams.append(lam)
        phi = phi + beta * (np.mean(adapted, axis=0) - phi)
        lam_phi = float(np.mean(lams))
        rounds.append({"outer": k, "lambda_by_stage": [round(x, 3) for x in lams], "lambda_phi": round(lam_phi, 3),
                       "w_norm": round(float(np.linalg.norm(phi)), 3)})
        log(f"[meta] outer {k}: λ by stage {[round(x, 2) for x in lams]}  |φ| {np.linalg.norm(phi):.2f}")
    init = Controller(ACTIONS, N_FEATURES, eps=eps, W=phi, meta={"name": "meta-init", "lambda": lam_phi, "method": "reptile_cgrpo"})
    rep = {"outer": outer, "inner_days": inner_days, "n_tasks": n_tasks, "iters": iters, "beta": beta, "stages": list(stages),
           "rounds": rounds, "seen": seen, "lambda0": lam_phi, "runtime_s": round(time.time() - t0, 1)}
    return init, rep


def audit(rep: dict, seed: int, n_stages: int = 5) -> dict:
    """What the meta-init touched against what it must not: held-out seeds, the last stage."""
    bad_seeds = sorted(set(rep["seen"]["task_seeds"]) & heldout_seeds(seed, n_stages))
    bad_stage = [g for g in rep["seen"]["stages"] if g >= n_stages - 1]
    return {"heldout_seed_overlap": bad_seeds, "last_stage_seen": bad_stage, "ok": not bad_seeds and not bad_stage}
