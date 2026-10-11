"""Simulated frozen harness with a scripted drift event (the only harness that can be re-rolled
from logged states, so the only one `train_cgrpo` can train on). Costs are in sim units, not USD;
the tier names come from pace.prices so the sim and the API harnesses share one vocabulary.

Tasks have a latent difficulty (easy p=0.65 / hard p=0.35) the controller
never sees; the harness exposes a noisy difficulty proxy (agrees with the
latent 85% of the time). Actions are (op, tier) pairs:

  ANSWER@tier    terminal; success prob from the tier quality table, plus
                 evidence and verification bonuses
  RETRIEVE@tier  +1 evidence (max 2) with tier-dependent success prob
  VERIFY@tier    sets verified; adds a tier-scaled bonus to the final answer
  ESCALATE@tier  terminal; ANSWER at that tier with extra compute (+bonus, 1.5x cost)

At most 4 steps; on the last step only terminal ops are legal.
Retrieval success is independent of the latent, so the observable state is
Markov for the controller in v1.

SimHarnessV2 (the drift event, "the host harness ships v2"):
  + CACHED_RETRIEVE: near-free evidence, but only works on (latent) easy tasks
  - the verifier AGED: VERIFY still sets the flag but adds nothing on hard tasks
  - frontier repriced x1.5
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..prices import TIERS
from .base import Harness, HarnessState

OPS = ["ANSWER", "RETRIEVE", "VERIFY", "ESCALATE"]
TERMINAL_OPS = {"ANSWER", "ESCALATE"}
MAX_STEPS = 4

BASE_PRICE = {"cheap": 0.02, "mid": 0.10, "frontier": 0.40}    # per answer-sized call
OP_MULT = {"ANSWER": 1.0, "RETRIEVE": 0.5, "VERIFY": 0.5, "ESCALATE": 1.5}
SEARCH_FEE = 0.005
BASE_LATENCY = {"cheap": 0.4, "mid": 1.0, "frontier": 2.5}       # seconds
BASE_Q = {"easy": {"cheap": 0.70, "mid": 0.82, "frontier": 0.90},
          "hard": {"cheap": 0.20, "mid": 0.45, "frontier": 0.68}}
EV_BONUS = {"easy": 0.05, "hard": 0.10}            # per evidence item (max 2)
VER_BONUS = {"easy": 0.04, "hard": 0.10}
VER_TIER = {"cheap": 0.5, "mid": 0.8, "frontier": 1.0}
RETRIEVE_P = {"cheap": 0.75, "mid": 0.90, "frontier": 0.97}
ESC_BONUS = 0.08
P_EASY = 0.65
PROXY_ACC = 0.85
TASK_NOISE = 0.04

CACHED = "CACHED_RETRIEVE"
CACHED_COST = 0.003
CACHED_LATENCY = 0.05


def action_name(op: str, tier: str) -> str:
    return f"{op}@{tier}"


def parse_action(name: str) -> tuple[str, str | None]:
    if "@" in name:
        op, tier = name.split("@", 1)
        return op, tier
    return name, None


V1_ACTIONS = [action_name(o, t) for o in OPS for t in TIERS]


@dataclass
class SimTask:
    task_id: str
    latent: str
    proxy: str
    q_noise: float


class SimHarness(Harness):
    version = "v1"

    def __init__(self, seed: int = 0, price_factors: dict[str, float] | None = None):
        self.rng = random.Random(seed)
        self.price_factor = {t: 1.0 for t in TIERS}
        for t, f in (price_factors or {}).items():
            self.reprice(t, f)

    # -------------------------------------------------------------- pricing
    def reprice(self, tier: str, factor: float) -> None:
        if tier not in self.price_factor:
            raise ValueError(f"unknown tier {tier!r}")
        self.price_factor[tier] *= float(factor)

    def price(self, op: str, tier: str) -> float:
        c = BASE_PRICE[tier] * self.price_factor[tier] * OP_MULT[op]
        return c + (SEARCH_FEE if op == "RETRIEVE" else 0.0)

    # ------------------------------------------------------------ interface
    @property
    def all_actions(self) -> list[str]:
        return list(V1_ACTIONS)

    def sample_tasks(self, n: int, seed: int = 0):
        rng = random.Random(seed)
        for i in range(n):
            latent = "easy" if rng.random() < P_EASY else "hard"
            proxy = latent if rng.random() < PROXY_ACC else ("hard" if latent == "easy" else "easy")
            yield SimTask(f"{self.version}-s{seed}-{i}", latent, proxy, rng.gauss(0.0, TASK_NOISE))

    def reset(self, task: SimTask) -> HarnessState:
        return HarnessState(task_id=task.task_id, difficulty_proxy=task.proxy, version=self.version,
                            extra={"latent": task.latent, "q_noise": task.q_noise,
                                   "ver_bonus": 0.0, "final_p": None, "outcome": None})

    def legal_actions(self, state: HarnessState) -> list[str]:
        if state.done:
            return []
        acts = self.all_actions
        if state.step >= MAX_STEPS - 1:
            acts = [a for a in acts if parse_action(a)[0] in TERMINAL_OPS]
        return acts

    def _answer_p(self, s: HarnessState, tier: str, escalate: bool, evidence=None, ver_bonus=None) -> float:
        lat = s.extra["latent"]
        ev = s.evidence if evidence is None else evidence
        vb = s.extra["ver_bonus"] if ver_bonus is None else ver_bonus
        p = BASE_Q[lat][tier] + s.extra["q_noise"] + EV_BONUS[lat] * ev + vb
        if escalate:
            p += ESC_BONUS
        return min(max(p, 0.01), 0.99)

    def _verify_bonus(self, latent: str, tier: str) -> float:
        return VER_BONUS[latent] * VER_TIER[tier]

    def _latency(self, op: str, tier: str | None) -> float:
        if op == CACHED:
            return CACHED_LATENCY
        return BASE_LATENCY[tier] * OP_MULT[op]

    def _cost(self, op: str, tier: str | None) -> float:
        if op == CACHED:
            return CACHED_COST
        return self.price(op, tier)

    def step(self, state: HarnessState, action: str):
        if state.done:
            raise RuntimeError("step() on a terminal state")
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action!r} at step {state.step}")
        s = state.clone()
        op, tier = parse_action(action)
        cost, lat = self._cost(op, tier), self._latency(op, tier)
        if op in TERMINAL_OPS:
            p = self._answer_p(s, tier, escalate=(op == "ESCALATE"))
            s.extra["final_p"] = p
            s.extra["outcome"] = 1.0 if self.rng.random() < p else 0.0
            s.done = True
        elif op == "RETRIEVE":
            if self.rng.random() < RETRIEVE_P[tier]:
                s.evidence = min(s.evidence + 1, 2)
        elif op == "VERIFY":
            s.verified = True
            s.extra["ver_bonus"] = max(s.extra["ver_bonus"], self._verify_bonus(s.extra["latent"], tier))
        elif op == CACHED:
            if s.extra["latent"] == "easy":
                s.evidence = min(s.evidence + 1, 2)
        else:
            raise ValueError(action)
        s.step += 1
        s.cost_so_far += cost
        s.latency_so_far += lat
        return s, s.done, cost, lat

    def quality(self, state: HarnessState) -> float:
        return float(state.extra["outcome"] or 0.0)

    def expected_quality(self, state: HarnessState) -> float:
        return float(state.extra["final_p"] or 0.0)

    def counterfactual(self, state: HarnessState) -> dict:
        """action -> (cost, expected_quality). Terminal actions: exact.
        Non-terminal actions: one-step lookahead, i.e. the op followed
        immediately by ANSWER at the same tier (CACHED: at cheap)."""
        out = {}
        lat = state.extra["latent"]
        for a in self.legal_actions(state):
            op, tier = parse_action(a)
            if op in TERMINAL_OPS:
                out[a] = (round(self._cost(op, tier), 6), round(self._answer_p(state, tier, op == "ESCALATE"), 6))
            elif op == "RETRIEVE":
                pr = RETRIEVE_P[tier]
                q = (pr * self._answer_p(state, tier, False, evidence=min(state.evidence + 1, 2))
                     + (1 - pr) * self._answer_p(state, tier, False))
                out[a] = (round(self._cost(op, tier) + self.price("ANSWER", tier), 6), round(q, 6))
            elif op == "VERIFY":
                vb = max(state.extra["ver_bonus"], self._verify_bonus(lat, tier))
                q = self._answer_p(state, tier, False, ver_bonus=vb)
                out[a] = (round(self._cost(op, tier) + self.price("ANSWER", tier), 6), round(q, 6))
            elif op == CACHED:
                ev = min(state.evidence + 1, 2) if lat == "easy" else state.evidence
                q = self._answer_p(state, "cheap", False, evidence=ev)
                out[a] = (round(CACHED_COST + self.price("ANSWER", "cheap"), 6), round(q, 6))
        return out

    def snapshot(self, state: HarnessState) -> dict:
        return self.state_to_dict(state)

    def restore(self, snap: dict) -> HarnessState:
        d = dict(snap)
        d["extra"] = dict(d.get("extra", {}))
        return HarnessState(**d)


class SimHarnessV2(SimHarness):
    """Drift: + CACHED_RETRIEVE (easy-only), aged verifier on hard, frontier x1.5."""
    version = "v2"

    def __init__(self, seed: int = 0, price_factors: dict[str, float] | None = None):
        super().__init__(seed=seed, price_factors=None)
        self.reprice("frontier", 1.5)
        for t, f in (price_factors or {}).items():
            self.reprice(t, f)

    @property
    def all_actions(self) -> list[str]:
        return list(V1_ACTIONS) + [CACHED]

    def _verify_bonus(self, latent: str, tier: str) -> float:
        if latent == "hard":
            return 0.0     # the judge aged: it no longer catches hard-task errors
        return super()._verify_bonus(latent, tier)


def make_sim(version: str = "v1", seed: int = 0, reprice: dict[str, float] | None = None) -> SimHarness:
    if version == "v1":
        return SimHarness(seed=seed, price_factors=reprice)
    if version == "v2":
        return SimHarnessV2(seed=seed, price_factors=reprice)
    raise ValueError(f"unknown sim version {version!r}")


def parse_reprice(specs: list[str] | None) -> dict[str, float]:
    """['mid:1.2', 'frontier:0.8'] -> {'mid': 1.2, 'frontier': 0.8}"""
    out: dict[str, float] = {}
    for s in specs or []:
        tier, f = s.split(":")
        out[tier] = out.get(tier, 1.0) * float(f)
    return out
