"""Text environment: render a decision point as a short prompt, parse one action.

The controller LLM answers with exactly one tag, `<action>TIER OP</action>`
(e.g. `<action>mid RETRIEVE</action>` on the simulator, `<action>mid CONTINUE</action>`
on CRMArena lookup mode). The parser is strict: the stripped completion must be
that single tag and nothing else, and (tier, op) must be in the legal set.
Anything else is INVALID: the reward is the fixed `INVALID_PENALTY`, and the
episode is accounted for as the cheapest safe action (what a serving system
would fall back to), so invalid outputs still cost money in the batch mean.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from ..harness.base import HarnessState
from ..harness.sim import TIERS, SimHarness, SimTask, make_sim

INVALID_PENALTY = -1.0
STRICT_RE = re.compile(r"\A\s*<action>(cheap|mid|frontier) ([A-Z_]+)</action>\s*\Z")

# The fixed continuation policy after the controller's first action (both modes):
# stay on the chosen tier and answer. On the sim that is ANSWER@tier on the next
# step (exactly the one-step lookahead SimHarness.counterfactual() prices); on
# CRMArena it is collect.py's counterfactual-mode FixedArmPolicy (CONTINUE on the
# tier until the agent answers, ANSWER forced on the last turn).
SIM_FALLBACK = ("cheap", "ANSWER")
LOOKUP_OPS = ["CONTINUE"]           # the only op a counterfactual CRMArena log covers
LOOKUP_FALLBACK = ("cheap", "CONTINUE")

SYSTEM = ("You are the routing controller of an agent harness. You pick which model tier runs "
          "the next operation. Reply with exactly one action tag and nothing else.")


def format_action(tier: str, op: str) -> str:
    return f"<action>{tier} {op}</action>"


def parse_action(text: str, legal: list[tuple[str, str]] | None = None) -> tuple[str, str] | None:
    """Strict parse. Returns (tier, op) or None when the completion is not exactly
    one well-formed tag, or the action is not legal."""
    if not isinstance(text, str):
        return None
    m = STRICT_RE.match(text)
    if not m:
        return None
    act = (m.group(1), m.group(2))
    if legal is not None and act not in legal:
        return None
    return act


# ---------------------------------------------------------------- simulator
def sim_legal(harness: SimHarness, state: HarnessState) -> list[tuple[str, str]]:
    out = []
    for a in harness.legal_actions(state):
        if "@" in a:
            op, tier = a.split("@", 1)
            out.append((tier, op))
    return out


def sim_task_dict(task, version: str = "v1", reprice: dict[str, float] | None = None) -> dict:
    """Everything needed to rebuild the episode inside the reward function
    (carried through a dataset column as JSON)."""
    return {"env": "sim", "version": version, "reprice": dict(reprice or {}), "task_id": task.task_id,
            "latent": task.latent, "proxy": task.proxy, "q_noise": float(task.q_noise)}


def sim_task(task: dict) -> SimTask:
    return SimTask(task["task_id"], task["latent"], task["proxy"], float(task["q_noise"]))


LAM_PLACEHOLDER = "<<PRICE_OF_COST>>"


def price_line(lam: float, step: float = 0.1) -> str:
    """step 0.1: the simulator's price line (cost in sim units). Any other step: CRMArena lookup
    mode, where cost is USD and the price is shown on a grid of `step` per dollar."""
    if step == 0.1:
        return f"Price of cost: λ={round_lam(lam):.1f} (your reward is quality − λ·cost)."
    return f"Price of cost: λ={round_lam(lam, step):g} per USD (your reward is quality − λ·cost in USD)."


def round_lam(lam: float, step: float = 0.1) -> float:
    """The price is shown (and charged) rounded to `step` (0.1 on the sim, 1 per USD on CRMArena)."""
    if step == 0.1:
        return round(float(lam) + 1e-9, 1)
    return round(round(float(lam) / step + 1e-9) * step, 6)


def with_price(user: str, lam: float, step: float = 0.1) -> str:
    """Fill the price line of a prompt rendered with `price=True`."""
    return user.replace(LAM_PLACEHOLDER, price_line(lam, step))


def sim_prompt(harness: SimHarness, state: HarnessState, budget: float, shuffle_seed: int | None = None,
               price: bool = False) -> str:
    """Render the observable initial state (never the latent difficulty).

    With `shuffle_seed`, the legal actions are listed in a seeded random order:
    a fixed order and a fixed example action anchor a small LLM's samples
    on one or two arms, and GRPO can only reinforce arms it samples. The example
    action is then also drawn at random (the format still needs a concrete
    example: without one, Qwen3-1.7B's zero-shot outputs were ~80% invalid)."""
    legal = sim_legal(harness, state)
    tiers = [t for t in TIERS if any(l[0] == t for l in legal)]
    listed = list(legal)
    example = ("mid", "ANSWER")
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(listed)
        example = rng.choice(legal)
    tier_prices = ", ".join(f"{t} {harness.price('ANSWER', t):.3f}" for t in tiers)
    lines = [
        "Task: answer one question. Observed difficulty: " + state.difficulty_proxy
        + " (a noisy estimate, right about 85% of the time).",
        f"State: step {state.step}, evidence {state.evidence}, verified {'yes' if state.verified else 'no'}, "
        f"cost so far {state.cost_so_far:.3f}.",
        f"Budget: average cost per task must stay at or below {budget:.2f}.",
    ]
    if price:      # price-conditioned controller: the current learned lambda, filled in per step
        lines.append(LAM_PLACEHOLDER)
    lines += [
        f"Price of one ANSWER call by tier: {tier_prices}. RETRIEVE and VERIFY cost 0.5x (RETRIEVE adds a 0.005 "
        "search fee), ESCALATE costs 1.5x.",
        "Ops: ANSWER = answer now; RETRIEVE = fetch evidence; VERIFY = check with a judge; "
        "ESCALATE = answer now with extra compute.",
        "After a non-final op the harness answers on the same tier.",
        "Legal actions: " + ", ".join(f"{t} {o}" for t, o in listed) + ".",
        ("Choose the action that maximizes expected quality − λ·cost." if price
         else "Choose the action that maximizes expected answer quality within the budget."),
        "Reply with exactly one tag, for example " + format_action(*example) + ".",
    ]
    return "\n".join(lines)


def sim_rows(n: int, seed: int, budget: float, version: str = "v1",
             reprice: dict[str, float] | None = None, shuffle: bool = True, price: bool = False) -> list[dict]:
    """n seeded sim tasks -> dataset rows {"user": prompt text, "task": json}."""
    h = make_sim(version, seed=seed, reprice=reprice)
    rows = []
    for i, t in enumerate(h.sample_tasks(n, seed)):
        s = h.reset(t)
        rows.append({"user": sim_prompt(h, s, budget, shuffle_seed=(seed * 1_000_003 + i) if shuffle else None,
                                         price=price),
                     "task": json.dumps(sim_task_dict(t, version, reprice), sort_keys=True)})
    return rows


# --------------------------------------------------- CRMArena lookup mode
def crm_task_features(episode_dict: dict) -> dict:
    """Feature dict for a CRMArena task from one logged episode (any arm)."""
    meta = episode_dict.get("meta") or {}
    f = {"task_id": str(episode_dict["task_id"]), "org": meta.get("org", "b2b"),
         "category": meta.get("category", "unknown"), "reward_metric": meta.get("reward_metric", "unknown")}
    if meta.get("query"):
        f["query"] = str(meta["query"])[:400]
    return f


def lookup_prompt(features: dict, budget: float, tier_cost: dict[str, float], price: bool = False,
                  shuffle_seed: int | None = None) -> str:
    """price=True: a price-conditioned prompt with the LAM_PLACEHOLDER line (filled by
    `with_price(..., step=1.0)`) and the matching objective. shuffle_seed: list the legal
    actions in a seeded random order (as sim_prompt does)."""
    legal = [(t, "CONTINUE") for t in TIERS if t in tier_cost]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(legal)
    lines = [
        f"Task: a CRM agent task (org {features.get('org')}, category {features.get('category')}, "
        f"scored by {features.get('reward_metric')}).",
    ]
    if features.get("query"):
        lines.append("Query: " + features["query"])
    lines += [
        f"Budget: average cost per task must stay at or below ${budget:.3f}.",
    ]
    if price:
        lines.append(LAM_PLACEHOLDER)
    lines += [
        "Typical cost of a full episode by tier: "
        + ", ".join(f"{t} ${tier_cost[t]:.4f}" for t in TIERS if t in tier_cost) + ".",
        "CONTINUE = run the agent on that tier until it answers.",
        "Legal actions: " + ", ".join(f"{t} {o}" for t, o in legal) + ".",
        ("Choose the tier that maximizes expected quality − λ·cost." if price
         else "Choose the tier that maximizes expected answer quality within the budget."),
        "Reply with exactly one tag of the form <action>TIER OP</action> naming one legal action.",
    ]
    return "\n".join(lines)


def lookup_task_dict(features: dict) -> dict:
    return {"env": "lookup", **features}


def chat_messages(user: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def legal_for(task: dict) -> list[tuple[str, str]]:
    """Legal (tier, op) set of a task's initial decision point."""
    if task["env"] == "sim":
        h = make_sim(task.get("version", "v1"))
        return sim_legal(h, h.reset(sim_task(task)))
    return [(t, o) for t in TIERS for o in LOOKUP_OPS]
