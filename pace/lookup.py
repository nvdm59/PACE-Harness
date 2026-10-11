"""Counterfactual lookup table over a CRMArena-Pro log (offline, no API calls).

`collect.py --mode counterfactual` runs every task on every tier with a fixed continuation
(CONTINUE on that tier until the agent answers), so for each task the log holds the exact
(quality, cost) of every first-step tier choice. The first-step decision is then a contextual
bandit with a FULLY OBSERVED reward table (the supervised-to-bandit setting of Dudik, Langford
& Li 2011, arXiv:1103.4601, with every arm revealed): the value of ANY tier policy on a logged
task is an exact lookup,
    V(pi) = mean_tasks sum_tier pi(tier | x0) * r(task, tier),
which is what makes held-out evaluation on 50 tasks exact rather than estimated.

`LookupTable`      tasks x tiers arrays of quality and cost + step-0 features
`reprice_episodes` recompute every step's cost from the token counts in the spend ledger
                   (traces/spend.jsonl, joined on task_id + purpose and matched on the logged
                   usd) under a repriced price table
`fixed_arm_log`    the realized log of the uniform behavior: one arm per task
`kfold_tasks`      category-stratified K-fold split over task ids (train/test disjoint by task)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .controller import Controller
from .features import crm_context
from .prices import TIERS, call_cost, repriced
from .trace import Episode, TraceStep

DECISION_ACTIONS = [f"CONTINUE@{t}" for t in TIERS]     # what the first step chooses between
USD_MATCH_TOL = 1e-8     # the ledger rounds usd to 8 decimals; the trace keeps the full float


def arm_of(ep: Episode) -> str:
    arm = (ep.meta or {}).get("arm")
    if arm in TIERS:
        return arm
    return ep.steps[0].action_name.split("@")[1]


@dataclass
class LookupTable:
    task_ids: list[str]
    X: np.ndarray            # (n, d) step-0 features
    cats: list[str]
    orgs: list[str]
    Q: np.ndarray            # (n, 3) quality per tier (TIERS order)
    C: np.ndarray            # (n, 3) episode cost per tier, USD

    @classmethod
    def from_episodes(cls, episodes: list[Episode]) -> "LookupTable":
        """Only tasks with all three arms logged become rows."""
        by: dict[str, dict[str, Episode]] = defaultdict(dict)
        for e in episodes:
            by[e.task_id][arm_of(e)] = e
        ids = sorted(t for t, d in by.items() if all(k in d for k in TIERS))
        X = np.array([by[t][TIERS[0]].steps[0].features for t in ids], float)
        ctx = [crm_context(list(x)) for x in X]
        Q = np.array([[by[t][k].quality for k in TIERS] for t in ids], float)
        C = np.array([[by[t][k].total_cost for k in TIERS] for t in ids], float)
        return cls(ids, X, [c for c, _ in ctx], [o for _, o in ctx], Q, C)

    def __len__(self) -> int:
        return len(self.task_ids)

    def subset(self, ids: list[str]) -> "LookupTable":
        pos = {t: i for i, t in enumerate(self.task_ids)}
        ix = [pos[t] for t in ids]
        return LookupTable([self.task_ids[i] for i in ix], self.X[ix], [self.cats[i] for i in ix],
                           [self.orgs[i] for i in ix], self.Q[ix], self.C[ix])

    def probs(self, policy: Controller) -> np.ndarray:
        """(n, 3) first-step tier probabilities of a Controller as served (iterate mix, eps)."""
        pol = policy.copy()
        pol.ensure_actions(DECISION_ACTIONS)
        M = np.tile(pol.mask(DECISION_ACTIONS), (len(self), 1))
        P = pol.probs_batch(self.X, M)
        return P[:, [pol.index(a) for a in DECISION_ACTIONS]]

    def value(self, P: np.ndarray) -> tuple[float, float]:
        """Exact (mean quality, mean cost) per task of tier probabilities P (n, 3)."""
        return float((P * self.Q).sum(1).mean()), float((P * self.C).sum(1).mean())

    def oracle(self, lam: float | None = None) -> np.ndarray:
        """Per-task oracle as one-hot P. lam=None: the highest-quality tier, cheapest among ties;
        else argmax quality - lam * cost (cheapest among ties)."""
        score = self.Q * 1e6 - self.C if lam is None else (self.Q - lam * self.C) * 1e6 - self.C
        P = np.zeros_like(self.Q)
        P[np.arange(len(self)), score.argmax(1)] = 1.0
        return P

    def episodes(self, tiers_logged: list[str] | None = None) -> list[Episode]:
        """Synthetic one-step episodes (step-0 decision only) for every (task, tier) cell,
        propensity 1/3: what the gate's bandit estimators consume."""
        out = []
        for i, t in enumerate(self.task_ids):
            for j, k in enumerate(TIERS):
                if tiers_logged is not None and tiers_logged[i] != k:
                    continue
                st = TraceStep(t, "crmarena-pro-v1", 0, list(self.X[i]), j, DECISION_ACTIONS[j], 1.0 / len(TIERS),
                               float(self.C[i, j]), 0.0, legal=list(DECISION_ACTIONS))
                out.append(Episode(t, [st], float(self.Q[i, j]), float(self.C[i, j]), 0.0, {"arm": k}))
        return out


def fixed(tier: str, n: int) -> np.ndarray:
    P = np.zeros((n, len(TIERS)))
    P[:, TIERS.index(tier)] = 1.0
    return P


def uniform(n: int) -> np.ndarray:
    return np.full((n, len(TIERS)), 1.0 / len(TIERS))


def fixed_arm_log(episodes: list[Episode], seed: int = 0) -> list[Episode]:
    """The realized log of the uniform-tier behavior: one seeded arm per task (each with
    propensity 1/3 at step 0), i.e. what serving the behavior policy would have logged."""
    by: dict[str, dict[str, Episode]] = defaultdict(dict)
    for e in episodes:
        by[e.task_id][arm_of(e)] = e
    rng = random.Random(seed)
    return [by[t][rng.choice(sorted(by[t]))] for t in sorted(by)]


def kfold_tasks(table: LookupTable, k: int = 5, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """Category-stratified K folds over TASK ids: shuffle (seeded) within each category, then
    deal the tasks round-robin across folds, so a fold holds at most ceil(n_cat / k) tasks of a
    category and, for categories with <= k tasks, every category keeps a training task in every
    fold. Every task lands in exactly one test fold; train and test never share a task.
    Returns [(train_ids, test_ids)] per fold."""
    rng = random.Random(seed)
    by: dict[str, list[str]] = defaultdict(list)
    for t, c in zip(table.task_ids, table.cats):
        by[c].append(t)
    folds: list[list[str]] = [[] for _ in range(k)]
    j = 0
    for c in sorted(by):
        ts = sorted(by[c])
        rng.shuffle(ts)
        for t in ts:
            folds[j % k].append(t)
            j += 1
    return [(sorted(t for i, f in enumerate(folds) if i != fi for t in f), sorted(folds[fi])) for fi in range(k)]


# --------------------------------------------------------------- repricing
def _purpose(step: TraceStep, tier: str) -> str:
    op = step.action_name.split("@")[0]
    return f"agent:{tier}" + (":answer" if op == "ANSWER" else "")


def join_spend(episodes: list[Episode], spend_path: str | Path) -> dict[tuple[int, int], dict]:
    """(episode index, step index) -> its spend-ledger record. Each step is matched to an unused
    record with the same task_id and purpose and the same usd (to USD_MATCH_TOL: the ledger
    stores usd rounded to 8 decimals, the trace the unrounded float). Records left unmatched
    are calls of episodes interrupted by the cap and re-run (sunk cost, not in the trace)."""
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in Path(spend_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            pool[(str(r.get("task_id")), r["purpose"])].append(r)
    used: set[int] = set()
    out = {}
    for i, e in enumerate(episodes):
        tier = arm_of(e)
        for t, s in enumerate(e.steps):
            cands = [r for r in pool[(e.task_id, _purpose(s, tier))]
                     if id(r) not in used and abs(float(r["usd"]) - s.cost) <= USD_MATCH_TOL]
            if not cands:
                raise ValueError(f"no spend record for {e.task_id} {tier} step {t} (usd {s.cost})")
            used.add(id(cands[0]))
            out[(i, t)] = cands[0]
    return out


def reprice_episodes(episodes: list[Episode], spend_path: str | Path, factors: dict[str, float]) -> list[Episode]:
    """Episodes with every step's cost recomputed from its logged token counts (input, output
    and, when present, cache write / read) under the price table scaled by `factors`
    ({tier: factor}). Quality and actions are unchanged: a repricing changes what the same
    episode costs, not what it does."""
    prices = repriced(factors)
    rec = join_spend(episodes, spend_path)
    out = []
    for i, e in enumerate(episodes):
        steps = []
        for t, s in enumerate(e.steps):
            r = rec[(i, t)]
            c = call_cost(r["model"], int(r["in_tokens"]), int(r["out_tokens"]),
                          int(r.get("cache_write_tokens", 0)), int(r.get("cache_read_tokens", 0)), prices=prices)
            steps.append(replace(s, cost=c))
        out.append(replace(e, steps=steps, total_cost=float(sum(s.cost for s in steps)),
                           meta={**(e.meta or {}), "reprice": dict(factors)}))
    return out
