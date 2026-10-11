"""Supervised warm start for the LLM controller (SFT -> GRPO).

Labels come from the simulator's own counterfactual one-step lookahead
(SimHarness.counterfactual: the first action followed by ANSWER on the same
tier, i.e. exactly the episode the reward runs). Per observed difficulty proxy
and price lambda, the label is the action maximizing the MEAN of
quality - lambda * cost over a large seeded task sample: the per-difficulty
oracle. It conditions only on what the controller observes (proxy, lambda),
never on the latent difficulty, so it is learnable to 100% agreement.

The price shown in an SFT prompt is drawn from LAM_GRID and the label is the
oracle at that price, so the warm-started policy is price-conditioned. Note: at
lambda = 0.5 the oracle is `mid ESCALATE` for BOTH proxies; the oracle differs
by difficulty for lambda <= 0.4 and for 1.1 <= lambda <= 1.9.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache

from ..harness.sim import make_sim, parse_action as sim_parse
from .env_text import format_action, round_lam, sim_rows, with_price

LAM_GRID = [round(0.1 * i, 1) for i in range(21)]          # 0.0 .. 2.0


@lru_cache(maxsize=None)
def _arm_means(version: str = "v1", n: int = 20000, seed: int = 99) -> dict:
    """(proxy, action) -> (mean cost, mean expected quality) of first action + same-tier ANSWER."""
    h = make_sim(version)
    agg: dict = {}
    for t in h.sample_tasks(n, seed):
        s = h.reset(t)
        for a, (c, q) in h.counterfactual(s).items():
            r = agg.setdefault((s.difficulty_proxy, a), [0.0, 0.0, 0])
            r[0] += c; r[1] += q; r[2] += 1
    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in agg.items()}


def oracle_action(proxy: str, lam: float, version: str = "v1") -> tuple[str, str]:
    """(tier, op) maximizing mean quality - lam * cost for this observed proxy."""
    m = _arm_means(version)
    acts = sorted(a for (p, a) in m if p == proxy and "@" in a)
    best = max(acts, key=lambda a: m[(proxy, a)][1] - round_lam(lam) * m[(proxy, a)][0])
    op, tier = sim_parse(best)
    return tier, op


def sft_rows(n: int, seed: int, budget: float, lam_grid: list[float] | None = None) -> list[dict]:
    """n (prompt -> oracle action) pairs; the shown price is drawn from lam_grid."""
    grid = list(lam_grid or LAM_GRID)
    rng = random.Random(seed + 4242)
    out = []
    for r in sim_rows(n, seed=seed, budget=budget, price=True):
        task = json.loads(r["task"])
        lam = rng.choice(grid)
        task["lam"] = lam
        out.append({"user": with_price(r["user"], lam), "task": json.dumps(task, sort_keys=True),
                    "completion": format_action(*oracle_action(task["proxy"], lam)), "proxy": task["proxy"],
                    "lam": lam})
    return out
