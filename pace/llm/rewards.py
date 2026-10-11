"""Priced rewards for the LLM controller, and the learned price of cost.

r = quality - lambda * cost, with lambda from DualAscentLambda (the same
PI-damped dual ascent on an EMA of cost that pace.train_offline.train_cgrpo
uses). Invalid completions get the fixed INVALID_PENALTY as reward, and are
executed as the cheapest safe action so their quality and cost still count
in the batch statistics and in the lambda update.

SimEpisodeReward   run the simulator from the task's initial state: the parsed
                   first action, then the fixed continuation (ANSWER on the same
                   tier). Quality is the harness's expected success probability
                   (the Monte Carlo ground truth pace.collect.true_eval uses);
                   the realized 0/1 outcome is returned too, from an RNG seeded
                   by (task_id, action) so a fixed task+action is reproducible.
LookupReward       CRMArena counterfactual log: (task_id, tier) -> (quality, cost)
                   of the logged CONTINUE-until-answer arm on that tier.
"""
from __future__ import annotations

import json
import random
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..harness.sim import action_name, make_sim
from .env_text import (INVALID_PENALTY, LOOKUP_FALLBACK, LOOKUP_OPS, SIM_FALLBACK, crm_task_features,
                       legal_for, parse_action, sim_task)


@dataclass
class Outcome:
    reward: float
    quality: float          # expected quality (sim) / logged quality (lookup)
    cost: float
    valid: bool
    action: str             # "tier OP" actually executed (the fallback when invalid)
    realized: float | None = None   # sim only: realized 0/1 outcome


class DualAscentLambda:
    """lambda = max(0, lam_i + kp * (cost_ema - B)),  lam_i += lr * (cost_ema - B).

    Same constants and damping as train_cgrpo: lr_lam 0.10, kp 2.0, ema 0.9, lam0 0.
    `update(mean_cost)` is called once per GRPO step with that step's batch mean cost.
    """

    def __init__(self, budget: float, lr: float = 0.10, kp: float = 2.0, ema: float = 0.9, lam0: float = 0.0):
        self.budget, self.lr, self.kp, self.ema = float(budget), float(lr), float(kp), float(ema)
        self.lam_i = self.lam = float(lam0)
        self.cost_ema: float | None = None
        self.history: list[dict] = []

    def update(self, mean_cost: float) -> float:
        c = float(mean_cost)
        self.cost_ema = c if self.cost_ema is None else self.ema * self.cost_ema + (1 - self.ema) * c
        gap = self.cost_ema - self.budget
        self.lam_i = max(0.0, self.lam_i + self.lr * gap)
        self.lam = max(0.0, self.lam_i + self.kp * gap)
        self.history.append({"cost": c, "cost_ema": self.cost_ema, "lambda": self.lam})
        return self.lam

    def state_dict(self) -> dict:
        return {"budget": self.budget, "lr": self.lr, "kp": self.kp, "ema": self.ema,
                "lam_i": self.lam_i, "lam": self.lam, "cost_ema": self.cost_ema}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


class SimEpisodeReward:
    """Execute the first action + fixed continuation on the simulator."""

    def __init__(self, invalid_penalty: float = INVALID_PENALTY):
        self.invalid_penalty = invalid_penalty
        self._h = {}

    def _harness(self, task: dict):
        key = (task.get("version", "v1"), json.dumps(task.get("reprice") or {}, sort_keys=True))
        if key not in self._h:
            self._h[key] = make_sim(key[0], seed=0, reprice=task.get("reprice") or None)
        return self._h[key]

    def run(self, task: dict, tier: str, op: str) -> tuple[float, float, float]:
        """-> (expected quality, cost, realized outcome) of first action (tier, op)
        followed by ANSWER on the same tier."""
        h = self._harness(task)
        h.rng = random.Random(_seed(task["task_id"], tier, op))
        s = h.reset(sim_task(task))
        a = action_name(op, tier)
        while not s.done:
            s, _, _, _ = h.step(s, a)
            a = action_name("ANSWER", tier)
        return h.expected_quality(s), s.cost_so_far, h.quality(s)

    def __call__(self, task: dict, completion: str, lam: float) -> Outcome:
        act = parse_action(completion, legal_for(task))
        valid = act is not None
        tier, op = act if valid else SIM_FALLBACK
        q, c, real = self.run(task, tier, op)
        r = q - lam * c if valid else self.invalid_penalty
        return Outcome(r, q, c, valid, f"{tier} {op}", real)


class LookupReward:
    """CRMArena counterfactual trace -> (task_id, tier) -> (quality, cost)."""

    def __init__(self, path_or_lines, invalid_penalty: float = INVALID_PENALTY):
        self.invalid_penalty = invalid_penalty
        self.table: dict[tuple[str, str], tuple[float, float]] = {}
        self.features: dict[str, dict] = {}
        if isinstance(path_or_lines, (str, Path)):
            lines = Path(path_or_lines).read_text().splitlines()
        else:
            lines = list(path_or_lines)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                              # torn final line of a live log
            meta = d.get("meta") or {}
            tier = meta.get("arm")
            if tier not in ("cheap", "mid", "frontier"):
                continue                              # policy-mode episodes are not arms
            tid = str(d["task_id"])
            self.table[(tid, tier)] = (float(d["quality"]), float(d["total_cost"]))
            self.features.setdefault(tid, crm_task_features(d))

    def complete_tasks(self, tiers=("cheap", "mid", "frontier")) -> list[str]:
        """Task ids that have every tier logged (the only ones usable as bandit rows)."""
        return sorted(t for t in self.features if all((t, k) in self.table for k in tiers))

    def tier_cost(self) -> dict[str, float]:
        out = {}
        for tier in ("cheap", "mid", "frontier"):
            cs = [c for (t, k), (_, c) in self.table.items() if k == tier]
            if cs:
                out[tier] = sum(cs) / len(cs)
        return out

    def __call__(self, task: dict, completion: str, lam: float) -> Outcome:
        act = parse_action(completion, legal_for(task))
        valid = act is not None
        tier, op = act if valid else LOOKUP_FALLBACK
        q, c = self.table[(str(task["task_id"]), tier)]
        r = q - lam * c if valid else self.invalid_penalty
        return Outcome(r, q, c, valid, f"{tier} {op}")


class PricedRewardFn:
    """TRL reward function adapter: reward_func(prompts, completions, task=[...], **kw) -> list[float].

    The task state travels in the dataset column `task` (JSON). Every call's
    outcomes are buffered; the training callback drains the buffer once per
    optimizer step (`drain()`), updates lambda from the batch mean cost and logs.
    """

    def __init__(self, lam: DualAscentLambda, sim: SimEpisodeReward | None = None,
                 lookup: LookupReward | None = None):
        self.lam = lam
        self.sim = sim or SimEpisodeReward()
        self.lookup = lookup
        self.buffer: list[Outcome] = []
        self.groups: list[bool] = []          # per prompt group: were all completions identical?
        self.shown: list[float] = []          # lambda shown in price-conditioned prompts
        self.__name__ = "priced_reward"

    def score(self, task: dict, completion: str) -> Outcome:
        env = self.sim if task["env"] == "sim" else self.lookup
        if env is None:
            raise ValueError("lookup task but no LookupReward configured")
        # a price-conditioned prompt carries the lambda it showed; charge exactly that price
        lam = float(task["lam"]) if task.get("lam") is not None else self.lam.lam
        return env(task, completion, lam)

    def __call__(self, prompts=None, completions=None, task=None, **kw) -> list[float]:
        outs = []
        for comp, t in zip(completions, task):
            if isinstance(comp, list):                 # conversational completion
                comp = comp[-1]["content"] if comp else ""
            td = json.loads(t) if isinstance(t, str) else t
            if td.get("lam") is not None:
                self.shown.append(float(td["lam"]))
            outs.append(self.score(td, comp))
        self.buffer.extend(outs)
        by_task: dict[str, set] = {}
        for comp, t in zip(completions, task):
            key = t if isinstance(t, str) else json.dumps(t, sort_keys=True)
            text = comp[-1]["content"] if isinstance(comp, list) and comp else comp
            by_task.setdefault(key, set()).add(str(text).strip())
        self.groups.extend(len(v) == 1 for v in by_task.values())
        return [o.reward for o in outs]

    def drain(self) -> dict | None:
        """Stats of the buffered outcomes since the last drain; updates lambda."""
        if not self.buffer:
            return None
        b, self.buffer = self.buffer, []
        g, self.groups = self.groups, []
        sh, self.shown = self.shown, []
        n = len(b)
        lam_used = self.lam.lam
        stats = {"n": n, "mean_reward": sum(o.reward for o in b) / n,
                 "mean_quality": sum(o.quality for o in b) / n,
                 "mean_cost": sum(o.cost for o in b) / n,
                 "invalid_rate": sum(not o.valid for o in b) / n,
                 "action_histogram": dict(Counter(o.action if o.valid else "INVALID" for o in b).most_common()),
                 "lambda_used": lam_used,
                 # groups whose completions are all identical: zero group advantage, no policy gradient
                 "degenerate_group_rate": (sum(g) / len(g)) if g else None,
                 "lambda_shown": (sum(sh) / len(sh)) if sh else None}
        stats["lambda"] = self.lam.update(stats["mean_cost"])
        stats["cost_ema"] = self.lam.cost_ema
        return stats


def lookup_rows(lookup: LookupReward, budget: float, price: bool = False, task_ids: list[str] | None = None,
                tier_cost: dict[str, float] | None = None, shuffle_seed: int | None = None) -> list[dict]:
    """One prompt row per complete task (or per id in `task_ids`). `tier_cost` overrides the
    typical tier costs shown (e.g. the training tasks' means, so held-out prompts carry no
    held-out information)."""
    from .env_text import lookup_prompt, lookup_task_dict
    tc = tier_cost or lookup.tier_cost()
    ids = lookup.complete_tasks() if task_ids is None else list(task_ids)
    return [{"user": lookup_prompt(lookup.features[t], budget, tc, price=price,
                                   shuffle_seed=None if shuffle_seed is None else shuffle_seed * 1_000_003 + i),
             "task": json.dumps(lookup_task_dict(lookup.features[t]), sort_keys=True)}
            for i, t in enumerate(ids)]
