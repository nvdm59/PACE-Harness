"""The orchestration loop with explicit, priced decision points.

    PLANNER(tier_p)  ->  WORKER(sub 1, tier_w)  ->  [decision]  ->  ...  ->  ANSWER(tier_a)

After every worker step (and after a verifier call that is not followed by a worker call) the
orchestrator reaches a DECISION POINT and asks a policy for one of

    {VERIFY, SKIP_VERIFY, REPLAN, ANSWER} x {cheap, mid, frontier}

where the tier is the tier of the NEXT model call: the worker call that follows VERIFY or
SKIP_VERIFY, the planner call of REPLAN, or the answer call. The verifier itself always runs on
its own (cheap) tier.

    VERIFY@t       verifier on the last worker result. On FAIL the same sub-query is retried once
                   on tier t with the verifier's reason; else the next sub-query runs on tier t.
    SKIP_VERIFY@t  no verification; the next sub-query runs on tier t.
    REPLAN@t       planner (tier t) sees the evidence so far and emits the remaining sub-queries;
                   the first of them runs on tier t.
    ANSWER@t       the harness's ANSWER op on tier t; the episode ends.

Role calls are capped at `max_calls` (default 8, the planner and the answer included). The legal
set shrinks so that an ANSWER call always fits: with one call left only ANSWER@* is legal.

Policies implement `act(features, legal, rng) -> (action index, propensity)` over `ACTIONS`:
the two fixed baselines (`always_verify`, `never_verify`, propensity 1) and `pace.controller.
Controller` (the PACE softmax-linear policy; propensity includes its eps), so the logged
decisions are usable by `pace.gate` and `pace.train_offline` unchanged.

Feature layout (`FEATURE_NAMES`): the first 12 dims follow `pace.features`' base block (bias,
difficulty proxy, evidence, verified, step, cost) so the core's DR bucket key and AWR baseline
work on these logs; then their difficulty crosses, one reserved version-cross dim (always 0:
this log holds one harness version), then the MAS block (category, org, verification counts,
plan progress, last observation kind, last worker tier).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from ...controller import Controller
from ...harness.base import HarnessState
from ...harness.crmarena import CATEGORIES, ORG_TYPES, CRMArenaHarness, CRMTask
from ...prices import TIERS
from ...trace import Episode, TraceStep
from .roles import Answerer, Plan, Planner, RoleCall, SubQuery, Verifier, Worker

VERSION = "mas-crmarena-pro-v1"       # must not start with "crmarena": pace.gate would treat it as a bandit log
OPS = ["VERIFY", "SKIP_VERIFY", "REPLAN", "ANSWER"]
ACTIONS = [f"{o}@{t}" for o in OPS for t in TIERS]
MAX_CALLS = 8
OBS_KINDS = ["rows", "empty", "error", "invalid"]

_BASE = ["bias", "proxy_easy", "proxy_hard", "ev0", "ev1", "ev2", "verified",
         "step0", "step1", "step2", "step3", "cost_so_far"]
FEATURE_NAMES = (list(_BASE) + [f"hard*{n}" for n in _BASE] + ["v2*bias(reserved)"]
                 + [f"cat={c}" for c in CATEGORIES] + [f"org={o}" for o in ORG_TYPES]
                 + ["n_verify/4", "n_fail/4", "pass_rate", "has_verifier_calls", "n_replan", "n_retry",
                    "sub_progress", "plan_len/3", "n_calls/8", "last_unverified"]
                 + [f"last_obs={k}" for k in OBS_KINDS] + [f"last_tier={t}" for t in TIERS])
N_FEATURES = len(FEATURE_NAMES)


def split_action(name: str) -> tuple[str, str]:
    op, tier = name.split("@")
    return op, tier


# ------------------------------------------------------------------ policies
class FixedPolicy:
    """`always_verify` / `never_verify`: deterministic, propensity 1."""

    def __init__(self, verify: bool, worker_tier: str = "mid", answer_tier: str | None = None):
        self.verify = bool(verify)
        self.worker_tier, self.answer_tier = worker_tier, answer_tier or worker_tier
        self.actions = list(ACTIONS)
        self.name = "always_verify" if verify else "never_verify"
        self.meta = {"name": self.name}

    def act(self, x, legal, rng=None, eps=None) -> tuple[int, float]:
        prefs = ([f"VERIFY@{self.worker_tier}"] if self.verify else []) + [f"SKIP_VERIFY@{self.worker_tier}",
                                                                             f"ANSWER@{self.answer_tier}"]
        for a in prefs:
            if a in legal:
                return ACTIONS.index(a), 1.0
        return ACTIONS.index(legal[0]), 1.0


def uniform_controller(eps: float = 0.0) -> Controller:
    pol = Controller.uniform(ACTIONS, N_FEATURES)
    pol.eps = eps
    pol.meta["name"] = "uniform"
    return pol


# --------------------------------------------------------------- episode state
@dataclass
class EpisodeState:
    task: CRMTask
    hs: HarnessState
    plan: Plan
    i: int = 0                           # index of the current sub-query
    calls: list[RoleCall] = field(default_factory=list)
    decisions: list[TraceStep] = field(default_factory=list)
    last_worker: RoleCall | None = None
    last_verdict: str | None = None      # pass | fail for the last worker step, None if unverified
    retried_current: bool = False
    n_replan: int = 0
    n_retry: int = 0
    prefix_cost: float = 0.0
    done: bool = False
    answer_by: str | None = None
    t0: float = 0.0

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.calls)

    def verifier_calls(self) -> list[RoleCall]:
        return [c for c in self.calls if c.role == "verifier"]

    def evidence(self) -> list[str]:
        out = []
        for c in self.calls:
            if c.role == "worker":
                tag = ""
                v = [x for x in self.calls if x.role == "verifier" and x.sub_query == c.sub_query]
                if v:
                    tag = f" [verifier: {v[-1].verdict}: {v[-1].reason}]"
                out.append(f"{c.input_summary} -> {c.output_summary}{tag}")
        return out


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, harness: CRMArenaHarness, planner_tier: str = "cheap", worker_tier: str = "mid",
                 verifier_tier: str = "cheap", answer_tier: str | None = None, max_calls: int = MAX_CALLS,
                 max_replans: int = 1, max_retries: int = 1, planner_max_tokens: int = 600,
                 verifier_max_tokens: int = 200):
        self.h = harness
        self.planner = Planner(harness.llm, planner_tier, planner_max_tokens)
        self.worker = Worker(harness)
        self.verifier = Verifier(harness.llm, verifier_tier, verifier_max_tokens)
        self.answerer = Answerer(harness)
        self.planner_tier, self.worker_tier, self.verifier_tier = planner_tier, worker_tier, verifier_tier
        self.answer_tier = answer_tier or worker_tier
        self.max_calls, self.max_replans, self.max_retries = int(max_calls), int(max_replans), int(max_retries)
        for t in (planner_tier, worker_tier, verifier_tier, self.answer_tier):
            assert t in TIERS, t

    # ---------------------------------------------------------- legal set
    def legal_actions(self, st: EpisodeState) -> list[str]:
        if st.done:
            return []
        remaining = self.max_calls - st.n_calls
        legal = [f"ANSWER@{t}" for t in TIERS]
        if remaining <= 1:
            return legal
        has_next = st.i + 1 < len(st.plan.sub_queries)
        if st.last_worker is not None and st.last_verdict is None:
            legal = [f"VERIFY@{t}" for t in TIERS] + legal
        if has_next:
            legal = [f"SKIP_VERIFY@{t}" for t in TIERS] + legal
        if remaining >= 3 and st.n_replan < self.max_replans:
            legal = [f"REPLAN@{t}" for t in TIERS] + legal
        return [a for a in ACTIONS if a in legal]

    # ---------------------------------------------------------- features
    def featurize(self, st: EpisodeState) -> list[float]:
        lw = st.last_worker
        last_kind = lw.verdict if lw is not None else "rows"
        hard = 1.0 if last_kind in ("empty", "error", "invalid") else 0.0
        n_rows = sum(1 for c in st.calls if c.role == "worker" and c.verdict == "rows")
        ev = min(n_rows, 2)
        verified = 1.0 if st.last_verdict == "pass" else 0.0
        step = min(len(st.decisions), 3)
        base = [1.0, 1.0 - hard, hard, 0.0, 0.0, 0.0, verified, 0.0, 0.0, 0.0, 0.0, float(st.cost)]
        base[3 + ev] = 1.0
        base[7 + step] = 1.0
        x = base + [hard * v for v in base] + [0.0]
        cat = [0.0] * len(CATEGORIES)
        if st.task.category in CATEGORIES:
            cat[CATEGORIES.index(st.task.category)] = 1.0
        org = [0.0] * len(ORG_TYPES)
        if st.task.org_type in ORG_TYPES:
            org[ORG_TYPES.index(st.task.org_type)] = 1.0
        vc = st.verifier_calls()
        n_fail = sum(1 for c in vc if c.verdict == "fail")
        n_plan = max(len(st.plan.sub_queries), 1)
        mas = [len(vc) / 4.0, n_fail / 4.0, (len(vc) - n_fail) / len(vc) if vc else 0.0, 1.0 if vc else 0.0,
               float(st.n_replan), float(st.n_retry), min(st.i + 1, n_plan) / n_plan, n_plan / 3.0,
               st.n_calls / float(self.max_calls), 1.0 if (lw is not None and st.last_verdict is None) else 0.0]
        obs = [1.0 if last_kind == k else 0.0 for k in OBS_KINDS]
        tier = [1.0 if (lw is not None and lw.tier == t) else 0.0 for t in TIERS]
        x = x + cat + org + mas + obs + tier
        assert len(x) == N_FEATURES, (len(x), N_FEATURES)
        return x

    # ---------------------------------------------------------- role steps
    def _record(self, st: EpisodeState, call: RoleCall) -> None:
        if st.n_calls >= self.max_calls:
            raise RuntimeError(f"role-call cap {self.max_calls} exceeded")       # legal set should prevent this
        st.calls.append(call)
        st.hs.cost_so_far = st.cost

    def _worker(self, st: EpisodeState, tier: str, retry_reason: str | None = None) -> None:
        sub = st.plan.sub_queries[st.i]
        st.hs, call = self.worker.run(st.hs, sub, st.i, len(st.plan.sub_queries), tier, retry_reason)
        self._record(st, call)
        st.last_worker, st.last_verdict = call, None
        if retry_reason is not None:
            st.n_retry += 1
            st.retried_current = True
        if st.hs.done:
            st.done, st.answer_by = True, "worker_respond"

    def _verify(self, st: EpisodeState) -> bool:
        sub = st.plan.sub_queries[st.last_worker.sub_query] if st.last_worker.sub_query is not None else SubQuery("?")
        v, call = self.verifier.verify(st.task, sub, st.last_worker)
        self._record(st, call)
        st.last_verdict = "pass" if v.passed else "fail"
        return v.passed

    def _replan(self, st: EpisodeState, tier: str) -> None:
        sys_prompt = st.hs.extra["messages"][0]["content"]
        plan, call = self.planner.plan(st.task, sys_prompt, tier, evidence=st.evidence())
        self._record(st, call)
        st.plan, st.i, st.n_replan = plan, 0, st.n_replan + 1
        st.last_worker, st.last_verdict, st.retried_current = None, None, False

    def _answer(self, st: EpisodeState, tier: str) -> None:
        st.hs, call = self.answerer.run(st.hs, tier)
        self._record(st, call)
        st.done, st.answer_by = True, f"answer@{tier}"

    def _advance(self, st: EpisodeState, tier: str) -> None:
        """Move to the next sub-query and run it on `tier` if a call fits (one must stay for ANSWER)."""
        st.i += 1
        st.retried_current = False
        if st.i < len(st.plan.sub_queries) and self.max_calls - st.n_calls >= 2:
            self._worker(st, tier)
        else:
            st.last_worker, st.last_verdict = None, None     # nothing new to verify

    # ---------------------------------------------------------- the loop
    def execute(self, st: EpisodeState, action: str) -> None:
        op, tier = split_action(action)
        if op == "ANSWER":
            self._answer(st, tier)
        elif op == "SKIP_VERIFY":
            self._advance(st, tier)
        elif op == "REPLAN":
            self._replan(st, tier)
            if self.max_calls - st.n_calls >= 2:
                self._worker(st, tier)
        elif op == "VERIFY":
            ok = self._verify(st)
            if self.max_calls - st.n_calls < 2:
                return
            if not ok and not st.retried_current and st.n_retry < self.max_retries:
                self._worker(st, tier, retry_reason=st.calls[-1].reason)
            else:
                self._advance(st, tier)
        else:
            raise ValueError(action)

    def run(self, task: CRMTask, policy, rng: random.Random | None = None, eps: float | None = None) -> Episode:
        rng = rng or random.Random(0)
        hs = self.h.reset(task)
        sys_prompt = hs.extra["messages"][0]["content"]
        plan, pcall = self.planner.plan(task, sys_prompt, self.planner_tier)
        st = EpisodeState(task=task, hs=hs, plan=plan, t0=time.time())
        self._record(st, pcall)
        self._worker(st, self.worker_tier)
        st.prefix_cost = st.cost
        while not st.done:
            legal = self.legal_actions(st)
            x = self.featurize(st)
            n0, c0, t0 = st.n_calls, st.cost, time.time()
            a, prop = policy.act(x, legal, rng, eps) if isinstance(policy, Controller) else policy.act(x, legal, rng)
            name = policy.actions[a]
            if name not in legal:
                raise ValueError(f"policy chose illegal action {name}; legal {legal}")
            self.execute(st, name)
            st.decisions.append(TraceStep(task_id=task.task_id, harness_version=VERSION, step=len(st.decisions),
                                          features=[round(v, 6) for v in x], action=ACTIONS.index(name),
                                          action_name=name, propensity=float(prop), cost=st.cost - c0,
                                          latency=time.time() - t0, legal=legal,
                                          state={"n_calls_before": n0, "n_calls_after": st.n_calls}))
            if st.n_calls >= self.max_calls and not st.done:
                raise RuntimeError("call cap reached without an answer")      # legal set should prevent this
        q = self.h.quality(st.hs)
        meta = dict(self.h.episode_meta(st.hs))
        vc = st.verifier_calls()
        meta.update({"harness": VERSION, "host_harness": self.h.version, "answer_by": st.answer_by,
                     "n_calls": st.n_calls, "prefix_cost": round(st.prefix_cost, 8),
                     "process": [c.to_dict() for c in st.calls], "plan": st.plan.to_dict(),
                     "n_verify": len(vc), "n_verify_fail": sum(1 for c in vc if c.verdict == "fail"),
                     "n_replan": st.n_replan, "n_retry": st.n_retry,
                     "role_tiers": {"planner": self.planner_tier, "worker": self.worker_tier,
                                    "verifier": self.verifier_tier, "answer": self.answer_tier},
                     "policy": getattr(policy, "meta", {}).get("name", type(policy).__name__),
                     "wall_s": round(time.time() - st.t0, 3)})
        return Episode(task_id=task.task_id, steps=st.decisions, quality=q, total_cost=st.cost,
                       total_latency=sum(c.latency for c in st.calls), meta=meta)


def make_policy(mode: str, worker_tier: str = "mid", answer_tier: str | None = None,
                policy_path: str | None = None, eps: float = 0.0):
    if mode == "always_verify":
        return FixedPolicy(True, worker_tier, answer_tier)
    if mode == "never_verify":
        return FixedPolicy(False, worker_tier, answer_tier)
    if mode == "controller":
        if policy_path in (None, "uniform"):
            return uniform_controller(eps)
        pol = Controller.load(policy_path)
        pol.ensure_actions(ACTIONS)
        if pol.n_features != N_FEATURES:
            raise ValueError(f"policy {policy_path} has {pol.n_features} features; MAS needs {N_FEATURES}")
        pol.eps = eps
        pol.meta.setdefault("name", policy_path)
        return pol
    raise ValueError(f"unknown mode {mode!r}")
