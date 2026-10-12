"""A multi-agent harness simulator calibrated to the logged real MAS run, with the harness
expressed as H = (tools, skills, agent pool) and the orchestrator's decision as a typed,
priced agent/skill/tool selection.

Why a simulator. The real MAS experiment (pace/experiments/mas, results/mas/mas_b2b.jsonl: 130
episodes on 50 CRMArena-Pro b2b tasks, $10.97) logged every role call with its observation kind,
verifier verdict, tokens and cost. A continually trained orchestrator needs days of traffic under
several harness changes, which that budget does not buy. This module re-expresses the logged
process as a seeded simulator whose decision points, legal-set rules and feature layout are the
real orchestrator's (`Orchestrator.legal_actions` / `Orchestrator.featurize` are called on a
view of the simulated state, not re-implemented), so `pace.controller`, `pace.gate`,
`pace.train_offline.train_awr` and `pace.experiments.mas.offpolicy` run on its logs unchanged.

Harness schema (EvoHarnessBench, arXiv 2609.04280: H_t = {tools, skills, specialist agents},
growing monotonically by stages; SkillOrchestra, arXiv 2602.19672: an agent pool of typed
specialists (id, backbone, mode, price) routed per query by a competence profile over skills with
an explicit cost penalty):

    Agent    (agent_id, backbone tier, mode, stage)   mode in {planner, data-worker, verifier, answer}
    Tool     (tool_id, stage, effect, cost)           an executable interface the orchestrator can call
    Skill    (skill_id, stage, effect, cost)          a reusable procedure the orchestrator can engage
    Handbook competence[agent][competence skill]      here the competence skills are the 22 CRMArena
                                                      task categories (the measured axis: the real run
                                                      found the category carries the signal, AUROC 0.82)

Actions (`<op>@<id>`, the tier and price implied by the id):
    delegate@<data agent>   run the next sub-query on that agent; after a verifier FAIL with a retry
                            still available it retries the CURRENT sub-query instead (the real
                            orchestrator's VERIFY-then-retry rule, made explicit)
    verify@<verifier>       the verifier agent checks the last worker step (structural fail at $0 on
                            query errors, as in the real one)
    replan@<planner>        the planner sees the evidence and emits the remaining sub-queries
    answer@<answer agent>   the final answer; the episode ends
    call@<tool>             an executable tool whose effect applies to the next worker step
    engage@<skill>          a procedure that changes how the following steps behave
The real orchestrator's 12 macro actions map onto these: VERIFY@t = verify@verify-1 then
delegate@data-<t>; SKIP_VERIFY@t = delegate@data-<t>; REPLAN@t = replan@plan-<t>; ANSWER@t =
answer@answer-<t>. Decision points sit after every non-terminal role call; at most MAX_CALLS = 8
role calls per episode (the real cap), and the legal set shrinks so an answer always fits.

Every parameter is either MEASURED (the number and the log it came from are next to it) or
ASSUMED (marked). `calibrate()` recomputes the measured ones from the two committed logs and the
test suite checks they match the constants below.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from ...harness.base import Harness, HarnessState
from ...harness.crmarena import CATEGORIES
from ...prices import TIERS, call_cost, model_for, repriced
from ...trace import Episode, TraceStep
from ..mas.orchestrator import FEATURE_NAMES as MAS_FEATURE_NAMES
from ..mas.orchestrator import Orchestrator
from ..mas.roles import Plan, RoleCall, SubQuery

MAS_LOG = Path("results/mas/mas_b2b.jsonl")
SINGLE_AGENT_LOG = Path("traces/crmarena_b2b_counterfactual.jsonl")
MAX_CALLS, MAX_REPLANS, MAX_RETRIES = 8, 1, 1          # the real orchestrator's defaults
KINDS = ["rows", "empty", "error", "invalid", "respond"]
TIER_OF_AGENT_INDEX = {1: "cheap", 2: "mid", 3: "frontier"}

# ============================================================== measured constants
# results/mas/mas_b2b.jsonl (130 episodes: 50 always_verify, 50 never_verify, 30 controller:uniform)
# unless noted. Counts are kept so the provenance is visible; rates are derived from them.
M = SimpleNamespace(
    n_episodes=130,
    # initial planner output (Haiku), 130 plans: 1, 2, 3 sub-queries
    plan_len_counts={1: 4, 2: 22, 3: 104},
    # re-plans (15, all in controller:uniform): every one emitted exactly 1 remaining sub-query
    replan_len_counts={1: 15},
    # worker observation kind by tier (verdict field of worker role calls)
    worker_kind_counts={"mid": {"rows": 252, "empty": 50, "error": 19, "invalid": 9, "respond": 1},      # n = 331
                        "cheap": {"rows": 4, "empty": 1, "error": 1, "invalid": 0, "respond": 5},        # n = 11 (thin)
                        "frontier": {"rows": 8, "empty": 1, "error": 1, "invalid": 0, "respond": 0}},    # n = 10 (thin)
    # retried worker steps: 36 rows of 53 (0.68) against 228 of 299 first attempts (0.76): treated as the
    # same distribution (the difference is inside the noise of 53 calls)
    retry_kind_counts={"rows": 36, "empty": 10, "error": 5, "invalid": 1, "respond": 1},
    # token means per (role, tier): (in, out), with the number of calls
    tokens={("planner", "cheap"): (5029, 279, 135), ("planner", "mid"): (6996, 159, 5), ("planner", "frontier"): (6902, 311, 5),
            ("worker", "cheap"): (6647, 115, 11), ("worker", "mid"): (8369, 158, 331), ("worker", "frontier"): (9598, 196, 10),
            ("verifier", "cheap"): (711, 62, 160),
            ("answer", "cheap"): (7393, 245, 15), ("answer", "mid"): (12250, 91, 105), ("answer", "frontier"): (15227, 332, 4)},
    # mid worker input tokens by position of the worker call in the episode (context grows)
    worker_in_by_pos={0: 6833, 1: 8571, 2: 10330},
    # verifier verdicts by the kind of the step it checked (160 calls, 12 structural at $0)
    verifier_counts={"rows": {"pass": 52, "fail": 72}, "empty": {"pass": 1, "fail": 23},
                     "error": {"pass": 0, "fail": 10}, "invalid": {"pass": 0, "fail": 2}},
    # episode level, always_verify: flags (>= 1 fail) among the 26 wrong and the 24 correct episodes
    flag_given_wrong=(24, 26), flag_given_correct=(23, 24), verifier_calls_per_episode=2.94,
    # per-mode outcome and cost (the targets the simulated fixed baselines are checked against)
    modes={"always_verify": {"n": 50, "accuracy": 0.480, "cost": 0.0906, "calls": 7.88, "verifier_calls": 2.94, "cap_hits": 47},
           "never_verify": {"n": 50, "accuracy": 0.460, "cost": 0.0821, "calls": 4.74, "verifier_calls": 0.0, "cap_hits": 0},
           "controller:uniform": {"n": 30, "accuracy": 0.533, "cost": 0.0758, "calls": 5.00, "verifier_calls": 0.43, "cap_hits": 4}},
    # MAS accuracy by task category, pooled over the three modes (correct, n)
    mas_cat_counts={"activity_priority": (5, 5), "best_region_identification": (6, 8), "case_routing": (7, 8),
                    "confidential_company_knowledge": (0, 5), "conversion_rate_comprehension": (3, 5), "handle_time": (7, 8),
                    "internal_operation_data": (0, 5), "invalid_config": (0, 5), "knowledge_qa": (0, 8),
                    "lead_qualification": (0, 5), "lead_routing": (2, 5), "monthly_trend_analysis": (4, 5),
                    "named_entity_disambiguation": (4, 8), "policy_violation_identification": (5, 6),
                    "private_customer_information": (0, 5), "quote_approval": (0, 5), "sales_amount_understanding": (4, 5),
                    "sales_cycle_understanding": (1, 5), "sales_insight_mining": (1, 5), "top_issue_identification": (6, 6),
                    "transfer_count": (6, 8), "wrong_stage_rectification": (2, 5)},
    # traces/crmarena_b2b_counterfactual.jsonl (50 tasks x 3 tiers): single-agent accuracy by tier
    single_tier=({"cheap": (22, 50), "mid": (30, 50), "frontier": (33, 50)}),
    # and by (category, tier): correct of n (n = 2 or 3)
    single_cat_counts={
        "activity_priority": {"cheap": (2, 2), "mid": (2, 2), "frontier": (2, 2)},
        "best_region_identification": {"cheap": (2, 3), "mid": (2, 3), "frontier": (3, 3)},
        "case_routing": {"cheap": (3, 3), "mid": (3, 3), "frontier": (3, 3)},
        "confidential_company_knowledge": {"cheap": (0, 2), "mid": (0, 2), "frontier": (0, 2)},
        "conversion_rate_comprehension": {"cheap": (1, 2), "mid": (2, 2), "frontier": (2, 2)},
        "handle_time": {"cheap": (1, 3), "mid": (2, 3), "frontier": (3, 3)},
        "internal_operation_data": {"cheap": (0, 2), "mid": (0, 2), "frontier": (0, 2)},
        "invalid_config": {"cheap": (1, 2), "mid": (2, 2), "frontier": (2, 2)},
        "knowledge_qa": {"cheap": (0, 3), "mid": (0, 3), "frontier": (0, 3)},
        "lead_qualification": {"cheap": (1, 2), "mid": (1, 2), "frontier": (0, 2)},
        "lead_routing": {"cheap": (0, 2), "mid": (2, 2), "frontier": (2, 2)},
        "monthly_trend_analysis": {"cheap": (1, 2), "mid": (2, 2), "frontier": (2, 2)},
        "named_entity_disambiguation": {"cheap": (3, 3), "mid": (2, 3), "frontier": (3, 3)},
        "policy_violation_identification": {"cheap": (0, 2), "mid": (2, 2), "frontier": (2, 2)},
        "private_customer_information": {"cheap": (0, 2), "mid": (0, 2), "frontier": (1, 2)},
        "quote_approval": {"cheap": (1, 2), "mid": (1, 2), "frontier": (0, 2)},
        "sales_amount_understanding": {"cheap": (1, 2), "mid": (1, 2), "frontier": (0, 2)},
        "sales_cycle_understanding": {"cheap": (0, 2), "mid": (1, 2), "frontier": (2, 2)},
        "sales_insight_mining": {"cheap": (0, 2), "mid": (0, 2), "frontier": (0, 2)},
        "top_issue_identification": {"cheap": (2, 2), "mid": (2, 2), "frontier": (2, 2)},
        "transfer_count": {"cheap": (2, 3), "mid": (2, 3), "frontier": (3, 3)},
        "wrong_stage_rectification": {"cheap": (1, 2), "mid": (1, 2), "frontier": (1, 2)}},
)

# ============================================================== assumed constants
# Nothing in the logs identifies whether a worker step that returned rows was the RIGHT rows, nor
# how a step's quality propagates to the final answer (the real ORM found no out-of-fold signal in
# the process features). The simulator needs that link for an orchestrator to have anything to
# learn, so it is assumed, with the magnitudes pinned by the measured mode-level numbers.
A = SimpleNamespace(
    good_given_rows={"cheap": 0.60, "mid": 0.75, "frontier": 0.85},   # P(a rows step is the right rows | tier)
    coverage_logit=1.5,        # logit gain of the final answer from no good step to all sub-queries covered
    task_noise_sd=0.4,         # per-task latent difficulty noise on the logit scale
    shrink_pseudo=2.0,         # pseudo-count pulling thin per-category cells toward their pooled mean
    # harness additions in later stages (drift.py); none of these exists in the logs
    data4={"rows": 0.85, "empty": 0.10, "error": 0.03, "invalid": 0.02, "respond": 0.0, "good": 0.80},  # database specialist
    answer4_strong_categories=["handle_time", "transfer_count", "monthly_trend_analysis", "conversion_rate_comprehension",
                               "sales_amount_understanding", "best_region_identification", "top_issue_identification",
                               "activity_priority"],                   # the answer specialist is mid-grade there, cheap-grade elsewhere
    verify2_fail_rows_good=0.114,    # from the stated target P(flag | correct) = 0.30 at 2.94 calls: 1 - 0.70 ** (1 / 2.94)
    tool_cost_tokens=(711, 62),      # a tool call is priced like a verifier-sized cheap call
    aggregate_categories=["handle_time", "transfer_count", "monthly_trend_analysis", "conversion_rate_comprehension",
                          "sales_amount_understanding", "best_region_identification", "top_issue_identification"],
    date_categories=["monthly_trend_analysis", "handle_time", "transfer_count", "sales_cycle_understanding"],
    skill_good_bonus=0.15,           # engaging a matching skill raises P(good | rows) of later steps by this
    tool_aggregate_coverage_bonus=0.34,   # aggregate_calc adds one sub-query's worth of coverage on aggregation tasks
    planner_v2_plan_len_p=(0.0, 0.10, 0.40, 0.50),   # the stage-4 planner decomposes into 1..4 sub-queries
)


def _norm(counts: dict) -> dict:
    n = float(sum(counts.values()))
    return {k: v / n for k, v in counts.items()}


def _shrink(k: float, n: float, prior: float, pseudo: float) -> float:
    return (k + pseudo * prior) / (n + pseudo)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# ================================================================ harness schema
@dataclass(frozen=True)
class Agent:
    agent_id: str
    tier: str            # backbone: cheap = Haiku 4.5, mid = Sonnet 5.5, frontier = Opus 5.5 (pace.prices)
    mode: str            # planner | data-worker | verifier | answer
    stage: int           # harness stage that introduces it (0 = the real run's pool)
    note: str = ""


@dataclass(frozen=True)
class Tool:
    tool_id: str
    stage: int
    note: str = ""


@dataclass(frozen=True)
class Skill:
    skill_id: str
    stage: int
    note: str = ""


# the registry of everything the harness can ever contain (fixed feature slots; drift.py decides
# which stage makes each available)
AGENTS = [
    Agent("plan-1", "cheap", "planner", 0, "the real run's planner (Haiku)"),
    Agent("plan-2", "mid", "planner", 0, "re-planner on Sonnet (REPLAN@mid in the real run)"),
    Agent("plan-3", "frontier", "planner", 0, "re-planner on Opus (REPLAN@frontier)"),
    Agent("data-1", "cheap", "data-worker", 0, "SOQL worker on Haiku"),
    Agent("data-2", "mid", "data-worker", 0, "SOQL worker on Sonnet (the real run's default worker)"),
    Agent("data-3", "frontier", "data-worker", 0, "SOQL worker on Opus"),
    Agent("verify-1", "cheap", "verifier", 0, "the real run's verifier (Haiku; P(flag|correct) 0.96)"),
    Agent("answer-1", "cheap", "answer", 0, "answer on Haiku"),
    Agent("answer-2", "mid", "answer", 0, "answer on Sonnet (the real run's default)"),
    Agent("answer-3", "frontier", "answer", 0, "answer on Opus"),
    Agent("verify-2", "cheap", "verifier", 2, "ASSUMED: a better verifier, P(flag|correct) 0.30, same recall"),
    Agent("data-4", "cheap", "data-worker", 2, "ASSUMED: database specialist at cheap price, rows 0.85 / good 0.80"),
    Agent("answer-4", "cheap", "answer", 2, "ASSUMED: reporting specialist at cheap price, mid-grade on 8 categories"),
]
TOOLS = [
    Tool("schema_lookup", 3, "ASSUMED: the next worker step cannot error (error/invalid mass moves to rows)"),
    Tool("aggregate_calc", 3, "ASSUMED: +0.34 coverage on aggregation categories, else nothing"),
]
SKILLS = [
    Skill("soql_aggregate", 4, "ASSUMED: +0.15 P(good | rows) on aggregation categories for the rest of the episode"),
    Skill("date_window", 4, "ASSUMED: +0.15 P(good | rows) on date-window categories"),
    Skill("decompose_fine", 4, "ASSUMED: re-plan into 3-4 finer sub-queries (the planner change that alters sub-query counts)"),
]
AGENT = {a.agent_id: a for a in AGENTS}
TOOL = {t.tool_id: t for t in TOOLS}
SKILL = {s.skill_id: s for s in SKILLS}
OPS = ["delegate", "verify", "replan", "answer", "call", "engage"]
ACTIONS = ([f"delegate@{a.agent_id}" for a in AGENTS if a.mode == "data-worker"]
           + [f"verify@{a.agent_id}" for a in AGENTS if a.mode == "verifier"]
           + [f"replan@{a.agent_id}" for a in AGENTS if a.mode == "planner"]
           + [f"answer@{a.agent_id}" for a in AGENTS if a.mode == "answer"]
           + [f"call@{t.tool_id}" for t in TOOLS] + [f"engage@{s.skill_id}" for s in SKILLS])
MAS_ACTION_MAP = {  # the real orchestrator's macro actions -> this schema
    **{f"VERIFY@{t}": f"verify@verify-1 then delegate@data-{i}" for i, t in TIER_OF_AGENT_INDEX.items()},
    **{f"SKIP_VERIFY@{t}": f"delegate@data-{i}" for i, t in TIER_OF_AGENT_INDEX.items()},
    **{f"REPLAN@{t}": f"replan@plan-{i}" for i, t in TIER_OF_AGENT_INDEX.items()},
    **{f"ANSWER@{t}": f"answer@answer-{i}" for i, t in TIER_OF_AGENT_INDEX.items()},
}


def split_action(name: str) -> tuple[str, str]:
    op, target = name.split("@", 1)
    return op, target


# ================================================================ parameters
@dataclass(frozen=True)
class Params:
    """Everything the simulator's dynamics depend on. Measured fields carry the log they came
    from in `M`; assumed fields are in `A`. Stage changes (drift.py) produce a new Params."""
    stage: int = 0
    plan_len_p: tuple = (0.0, 0.0, 0.0)                     # P(initial plan has 1, 2, ... sub-queries)
    replan_len_p: tuple = (1.0,)
    worker_kind_p: dict = field(default_factory=dict)       # data agent -> {kind: p}
    good_given_rows: dict = field(default_factory=dict)     # data agent -> P(good | rows)
    worker_in_base: float = 6833.0                          # worker input tokens at the first worker call
    worker_in_slope: float = 1748.5                         # growth per earlier worker call (context)
    worker_out: dict = field(default_factory=dict)          # tier -> output tokens
    planner_tokens: dict = field(default_factory=dict)      # tier -> (in, out)
    verifier_tokens: tuple = (711, 62)
    answer_tokens: dict = field(default_factory=dict)       # tier -> (in, out)
    verifier_fail_empty: float = 0.0
    verifier_fail_rows: dict = field(default_factory=dict)  # verifier agent -> {"good": p, "bad": p}
    answer_logit: dict = field(default_factory=dict)        # answer agent -> {category: logit P(correct) at neutral coverage}
    coverage_logit: float = A.coverage_logit
    coverage_center: float = 0.0                            # mean coverage of never_verify at stage 0 (self-calibrated)
    task_noise_sd: float = A.task_noise_sd
    price_factors: dict = field(default_factory=lambda: {t: 1.0 for t in TIERS})
    available_agents: tuple = ()
    available_tools: tuple = ()
    available_skills: tuple = ()
    max_calls: int = MAX_CALLS
    max_replans: int = MAX_REPLANS
    max_retries: int = MAX_RETRIES
    verifier_exempt_from_cap: bool = False      # ablation D: verifier calls do not count toward max_calls

    def prices(self) -> dict:
        return repriced({t: f for t, f in self.price_factors.items() if f != 1.0})

    def with_stage_pool(self, stage: int) -> "Params":
        return replace(self, stage=stage,
                       available_agents=tuple(a.agent_id for a in AGENTS if a.stage <= stage),
                       available_tools=tuple(t.tool_id for t in TOOLS if t.stage <= stage),
                       available_skills=tuple(s.skill_id for s in SKILLS if s.stage <= stage))


def measured_params() -> Params:
    """Stage-0 parameters from the constants above (the real run's harness: plan-1, data-1..3,
    verify-1, answer-1..3; prices as in pace/prices.py)."""
    plan_p = tuple(M.plan_len_counts.get(k, 0) / M.n_episodes for k in (1, 2, 3))
    kinds = {f"data-{i}": _norm(M.worker_kind_counts[t]) for i, t in TIER_OF_AGENT_INDEX.items()}
    good = {f"data-{i}": A.good_given_rows[t] for i, t in TIER_OF_AGENT_INDEX.items()}
    pos = M.worker_in_by_pos
    slope = (pos[2] - pos[0]) / 2.0
    vr = M.verifier_counts
    fail_rows = vr["rows"]["fail"] / (vr["rows"]["pass"] + vr["rows"]["fail"])
    fail_empty = vr["empty"]["fail"] / (vr["empty"]["pass"] + vr["empty"]["fail"])
    # the measured verifier is uninformative about the step's correctness (P(flag|wrong) 0.92 vs
    # P(flag|correct) 0.96): the same per-call fail rate whether the step was good or bad
    vfail = {"verify-1": {"good": fail_rows, "bad": fail_rows}}
    # answer competence per (agent, category): the MAS per-category accuracy (answer on mid)
    # shifted by the single-agent (category, tier) offset relative to mid, every cell shrunk
    mas_pool = sum(k for k, _ in M.mas_cat_counts.values()) / sum(n for _, n in M.mas_cat_counts.values())
    tier_mean = {t: k / n for t, (k, n) in M.single_tier.items()}
    logit = {f"answer-{i}": {} for i in (1, 2, 3)}
    for c in CATEGORIES:
        k, n = M.mas_cat_counts[c]
        base = _logit(_shrink(k, n, mas_pool, A.shrink_pseudo))
        sa = {t: _logit(_shrink(*M.single_cat_counts[c][t], tier_mean[t], A.shrink_pseudo)) for t in TIERS}
        for i, t in TIER_OF_AGENT_INDEX.items():
            logit[f"answer-{i}"][c] = base + (sa[t] - sa["mid"])
    p = Params(plan_len_p=plan_p, replan_len_p=(1.0,), worker_kind_p=kinds, good_given_rows=good,
               worker_in_base=float(pos[0]), worker_in_slope=slope,
               worker_out={t: M.tokens[("worker", t)][1] for t in TIERS},
               planner_tokens={t: M.tokens[("planner", t)][:2] for t in TIERS},
               verifier_tokens=M.tokens[("verifier", "cheap")][:2],
               answer_tokens={t: M.tokens[("answer", t)][:2] for t in TIERS},
               verifier_fail_empty=fail_empty, verifier_fail_rows=vfail, answer_logit=logit).with_stage_pool(0)
    # the coverage term is centred on never_verify's mean coverage, so the category base rates are
    # reproduced by the never_verify process and other policies move around them
    return replace(p, coverage_center=_mean_coverage_never_verify(p))


def _mean_coverage_never_verify(p: Params) -> float:
    """Expected share of sub-queries with a good step under the never_verify process, in closed
    form: the worker runs each sub-query once on data-2 (no retries) and a plan of L sub-queries
    needs L + 2 <= max_calls calls, which always holds for L <= 3."""
    g = p.worker_kind_p["data-2"]["rows"] * p.good_given_rows["data-2"]
    return g


# ================================================================ calibration from the logs
def calibrate(mas_log: str | Path = MAS_LOG, single_log: str | Path = SINGLE_AGENT_LOG) -> SimpleNamespace:
    """Recompute the measured constants from the two committed logs (the test compares them to `M`)."""
    import re

    from ...trace import read_jsonl
    mas = read_jsonl(mas_log)
    cf = read_jsonl(single_log)
    out = SimpleNamespace(n_episodes=len(mas))
    plan0, replan = Counter(), Counter()
    kinds = defaultdict(Counter)
    retry = Counter()
    tok = defaultdict(list)
    pos = defaultdict(list)
    ver = defaultdict(Counter)
    for e in mas:
        proc = e.meta["process"]
        k = 0
        for i, c in enumerate(proc):
            if c["role"] == "planner":
                n = int(re.match(r"(\d+) sub-queries", c["output_summary"]).group(1))
                (plan0 if c["verdict"] == "plan" else replan)[n] += 1
            elif c["role"] == "worker":
                kinds[c["tier"]][c["verdict"]] += 1
                if c["retry"]:
                    retry[c["verdict"]] += 1
                if c["tier"] == "mid":
                    pos[k].append(c["in_tokens"]); k += 1
            elif c["role"] == "verifier":
                w = [x for x in proc[:i] if x["role"] == "worker"][-1]
                ver[w["verdict"]][c["verdict"]] += 1
            tok[(c["role"], c["tier"])].append((c["in_tokens"], c["out_tokens"]))
    out.plan_len_counts, out.replan_len_counts = dict(plan0), dict(replan)
    out.worker_kind_counts = {t: {k: kinds[t].get(k, 0) for k in KINDS} for t in TIERS}
    out.retry_kind_counts = {k: retry.get(k, 0) for k in KINDS if retry.get(k, 0)}
    out.tokens = {k: (round(sum(a for a, _ in v) / len(v)), round(sum(b for _, b in v) / len(v)), len(v)) for k, v in tok.items()}
    out.worker_in_by_pos = {k: round(sum(v) / len(v)) for k, v in pos.items() if len(v) > 10}
    out.verifier_counts = {k: {"pass": ver[k].get("pass", 0), "fail": ver[k].get("fail", 0)} for k in ("rows", "empty", "error", "invalid")}
    av = [e for e in mas if e.meta["mode"] == "always_verify"]
    flagged = lambda e: any(c["role"] == "verifier" and c["verdict"] == "fail" for c in e.meta["process"])
    wrong = [e for e in av if e.quality < 0.5]
    corr = [e for e in av if e.quality >= 0.5]
    out.flag_given_wrong = (sum(flagged(e) for e in wrong), len(wrong))
    out.flag_given_correct = (sum(flagged(e) for e in corr), len(corr))
    out.verifier_calls_per_episode = round(sum(sum(1 for c in e.meta["process"] if c["role"] == "verifier") for e in av) / len(av), 2)
    modes = {}
    for m in ("always_verify", "never_verify", "controller:uniform"):
        eps = [e for e in mas if e.meta["mode"] == m]
        modes[m] = {"n": len(eps), "accuracy": round(sum(e.quality >= 0.5 for e in eps) / len(eps), 3),
                    "cost": round(sum(e.total_cost for e in eps) / len(eps), 4),
                    "calls": round(sum(len(e.meta["process"]) for e in eps) / len(eps), 2),
                    "verifier_calls": round(sum(sum(1 for c in e.meta["process"] if c["role"] == "verifier") for e in eps) / len(eps), 2),
                    "cap_hits": sum(len(e.meta["process"]) >= MAX_CALLS for e in eps)}
    out.modes = modes
    cat = defaultdict(lambda: [0, 0])
    for e in mas:
        cat[e.meta["category"]][0] += int(e.quality >= 0.5); cat[e.meta["category"]][1] += 1
    out.mas_cat_counts = {c: tuple(v) for c, v in sorted(cat.items())}
    st = defaultdict(lambda: [0, 0])
    sc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # the single-agent constants are measured on the same tasks the MAS run used (the first 50 of the
    # seeded sample); the counterfactual log has since been extended on that sample, so restrict to them
    mas_tasks = {e.task_id for e in mas}
    for e in cf:
        if e.task_id not in mas_tasks:
            continue
        t = e.steps[0].action_name.split("@")[1]
        st[t][0] += int(e.quality >= 0.5); st[t][1] += 1
        sc[e.meta["category"]][t][0] += int(e.quality >= 0.5); sc[e.meta["category"]][t][1] += 1
    out.single_tier = {t: tuple(v) for t, v in st.items()}
    out.single_cat_counts = {c: {t: tuple(v) for t, v in d.items()} for c, d in sorted(sc.items())}
    return out


# ================================================================ simulated state
@dataclass
class SimTask:
    task_id: str
    category: str
    noise: float
    org_type: str = "b2b"


@dataclass
class SimState:
    task: SimTask
    plan_len: int
    plan_epoch: int = 0
    i: int = 0
    calls: list = field(default_factory=list)          # RoleCall objects (the process log)
    last_worker: int | None = None                     # index into calls
    last_verdict: str | None = None
    retried_current: bool = False
    n_replan: int = 0
    n_retry: int = 0
    n_decisions: int = 0
    good: dict = field(default_factory=dict)           # "epoch:i" -> bool, the latest attempt's quality
    coverage_bonus: float = 0.0
    tools_called: dict = field(default_factory=dict)   # tool -> sub-query key it was called for
    skills: list = field(default_factory=list)
    no_error_next: bool = False
    done: bool = False
    answer_by: str | None = None
    final_p: float | None = None
    outcome: float | None = None

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def key(self) -> str:
        return f"{self.plan_epoch}:{self.i}"

    def copy(self) -> "SimState":
        return replace(self, calls=list(self.calls), good=dict(self.good), tools_called=dict(self.tools_called),
                       skills=list(self.skills))


def _rc(role, tier, verdict, cost, sub_query=None, retry=False, in_tokens=0, out_tokens=0, extra=None) -> RoleCall:
    return RoleCall(role, tier, f"mas:{role}", "", verdict or "", verdict, "", float(cost), 0.0,
                    int(in_tokens), int(out_tokens), sub_query, retry, extra or {})


# ================================================================ the harness
class MASSimHarness(Harness):
    """The simulated frozen multi-agent harness at one stage. `version` names the stage so logs of
    different stages are distinguishable (and never start with "crmarena", which pace.gate
    treats as a bandit log)."""

    def __init__(self, params: Params, seed: int = 0, version: str | None = None):
        self.p = params
        self.rng = random.Random(seed)
        self.version = version or f"mas-sim-stage{params.stage}"
        self.price_table = params.prices()
        self._rules = SimpleNamespace(max_calls=params.max_calls, max_replans=params.max_replans)
        self._hs_dummy = HarnessState(task_id="")

    # ---------------------------------------------------------- interface
    @property
    def all_actions(self) -> list[str]:
        return list(ACTIONS)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def sample_tasks(self, n: int, seed: int = 0) -> list[SimTask]:
        rng = random.Random(seed)
        return [SimTask(f"{self.version}-s{seed}-{i}", rng.choice(CATEGORIES), rng.gauss(0.0, self.p.task_noise_sd))
                for i in range(n)]

    def reset(self, task: SimTask) -> HarnessState:
        st = SimState(task=task, plan_len=self._draw_len(self.p.plan_len_p))
        self._planner(st, "plan-1", replan=False)
        self._worker(st, "data-2", allow_respond=False)       # the policy-independent prefix of the real run
        return self._hs(st)

    def _remaining(self, st: SimState) -> int:
        n = st.n_calls
        if self.p.verifier_exempt_from_cap:
            n -= sum(1 for c in st.calls if c.role == "verifier")
        return self.p.max_calls - n

    def legal_actions(self, state: HarnessState) -> list[str]:
        st: SimState = state.extra["st"]
        if st.done:
            return []
        p = self.p
        remaining = self._remaining(st)
        answers = [f"answer@{a}" for a in p.available_agents if AGENT[a].mode == "answer"]
        if remaining <= 1:
            return answers
        legal = list(answers)
        can_retry = st.last_verdict == "fail" and not st.retried_current and st.n_retry < p.max_retries
        has_next = st.i + 1 < st.plan_len
        if has_next or can_retry:
            legal += [f"delegate@{a}" for a in p.available_agents if AGENT[a].mode == "data-worker"]
        if st.last_worker is not None and st.last_verdict is None:
            legal += [f"verify@{a}" for a in p.available_agents if AGENT[a].mode == "verifier"]
        if remaining >= 3 and st.n_replan < p.max_replans:
            legal += [f"replan@{a}" for a in p.available_agents if AGENT[a].mode == "planner"]
        if remaining >= 3 and (has_next or can_retry):
            legal += [f"call@{t}" for t in p.available_tools if st.tools_called.get(t) != st.key()]
        if remaining >= 3:
            legal += [f"engage@{s}" for s in p.available_skills if s not in st.skills
                      and (s != "decompose_fine" or st.n_replan < p.max_replans)]
        return [a for a in ACTIONS if a in legal]

    def step(self, state: HarnessState, action: str):
        if state.done:
            raise RuntimeError("step() on a terminal state")
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action!r}; legal {self.legal_actions(state)}")
        st = state.extra["st"].copy()
        c0 = st.cost
        self._execute(st, action)
        st.n_decisions += 1
        s2 = self._hs(st)
        return s2, s2.done, st.cost - c0, 0.0

    def quality(self, state: HarnessState) -> float:
        return float(state.extra["st"].outcome or 0.0)

    def expected_quality(self, state: HarnessState) -> float:
        return float(state.extra["st"].final_p or 0.0)

    def featurize(self, state: HarnessState) -> list[float]:
        st: SimState = state.extra["st"]
        x = Orchestrator.featurize(self._rules, self._view(st))
        p = self.p
        avail = [1.0 if a.agent_id in p.available_agents else 0.0 for a in AGENTS if a.mode != "planner"]
        tools = [1.0 if t.tool_id in p.available_tools else 0.0 for t in TOOLS]
        called = [1.0 if st.tools_called.get(t.tool_id) == st.key() else 0.0 for t in TOOLS]
        skills = [1.0 if s.skill_id in p.available_skills else 0.0 for s in SKILLS]
        engaged = [1.0 if s.skill_id in st.skills else 0.0 for s in SKILLS]
        x = x + avail + tools + called + skills + engaged + [p.stage / 4.0, 1.0 if st.last_verdict == "fail" else 0.0]
        assert len(x) == N_FEATURES, (len(x), N_FEATURES)
        return x

    def episode_meta(self, state: HarnessState) -> dict:
        st: SimState = state.extra["st"]
        vc = [c for c in st.calls if c.role == "verifier"]
        return {"category": st.task.category, "org": st.task.org_type, "harness": self.version, "stage": self.p.stage,
                "answer_by": st.answer_by, "n_calls": st.n_calls, "expected_quality": st.final_p,
                "cap_hit": self._remaining(st) <= 0,
                "n_verify": len(vc), "n_verify_fail": sum(1 for c in vc if c.verdict == "fail"),
                "n_replan": st.n_replan, "n_retry": st.n_retry, "skills": list(st.skills),
                "tools": sorted(st.tools_called), "process": [c.to_dict() for c in st.calls]}

    def snapshot(self, state: HarnessState) -> dict:
        st: SimState = state.extra["st"]
        return {"step": st.n_decisions, "task": [st.task.task_id, st.task.category, st.task.noise, st.task.org_type],
                "plan_len": st.plan_len, "plan_epoch": st.plan_epoch, "i": st.i,
                "calls": [[c.role, c.tier, c.verdict, c.cost, c.sub_query, c.retry, c.in_tokens, c.out_tokens] for c in st.calls],
                "last_worker": st.last_worker, "last_verdict": st.last_verdict, "retried_current": st.retried_current,
                "n_replan": st.n_replan, "n_retry": st.n_retry, "good": dict(st.good), "coverage_bonus": st.coverage_bonus,
                "tools_called": dict(st.tools_called), "skills": list(st.skills), "no_error_next": st.no_error_next,
                "done": st.done, "answer_by": st.answer_by, "final_p": st.final_p, "outcome": st.outcome}

    def restore(self, snap: dict) -> HarnessState:
        d = snap
        st = SimState(task=SimTask(*d["task"]), plan_len=d["plan_len"], plan_epoch=d["plan_epoch"], i=d["i"],
                      calls=[_rc(r, t, v, c, sq, rt, it, ot) for r, t, v, c, sq, rt, it, ot in d["calls"]],
                      last_worker=d["last_worker"], last_verdict=d["last_verdict"], retried_current=d["retried_current"],
                      n_replan=d["n_replan"], n_retry=d["n_retry"], n_decisions=d["step"], good=dict(d["good"]),
                      coverage_bonus=d["coverage_bonus"], tools_called=dict(d["tools_called"]), skills=list(d["skills"]),
                      no_error_next=d["no_error_next"], done=d["done"], answer_by=d["answer_by"], final_p=d["final_p"],
                      outcome=d["outcome"])
        return self._hs(st)

    # ---------------------------------------------------------- views
    def _view(self, st: SimState):
        """The real orchestrator's EpisodeState, built on the simulated state, so its feature map
        and legal-set rules apply verbatim."""
        from ..mas.orchestrator import EpisodeState
        plan = Plan([SubQuery("") for _ in range(st.plan_len)])
        return EpisodeState(task=st.task, hs=self._hs_dummy, plan=plan, i=st.i, calls=st.calls,
                            decisions=[None] * st.n_decisions,
                            last_worker=None if st.last_worker is None else st.calls[st.last_worker],
                            last_verdict=st.last_verdict, retried_current=st.retried_current,
                            n_replan=st.n_replan, n_retry=st.n_retry, done=st.done)

    def _hs(self, st: SimState) -> HarnessState:
        lw = None if st.last_worker is None else st.calls[st.last_worker]
        kind = lw.verdict if lw is not None else "rows"
        return HarnessState(task_id=st.task.task_id, difficulty_proxy="hard" if kind in ("empty", "error", "invalid") else "easy",
                            evidence=sum(1 for c in st.calls if c.role == "worker" and c.verdict == "rows"),
                            verified=st.last_verdict == "pass", step=st.n_decisions, cost_so_far=st.cost,
                            done=st.done, version=self.version, extra={"st": st})

    # ---------------------------------------------------------- pricing
    def _cost(self, tier: str, in_tokens: float, out_tokens: float) -> float:
        return call_cost(model_for(tier), int(round(in_tokens)), int(round(out_tokens)), prices=self.price_table)

    def _draw_len(self, probs: tuple) -> int:
        u, acc = self.rng.random(), 0.0
        for k, p in enumerate(probs, start=1):
            acc += p
            if u < acc:
                return k
        return len(probs)

    # ---------------------------------------------------------- role calls
    def _planner(self, st: SimState, agent: str, replan: bool) -> None:
        tier = AGENT[agent].tier
        it, ot = self.p.planner_tokens[tier]
        st.calls.append(_rc("planner", tier, "replan" if replan else "plan", self._cost(tier, it, ot), in_tokens=it, out_tokens=ot))

    def _worker(self, st: SimState, agent: str, retry: bool = False, allow_respond: bool = True) -> None:
        tier = AGENT[agent].tier
        kp = dict(self.p.worker_kind_p[agent])
        if not allow_respond:
            kp["respond"] = 0.0
        if st.no_error_next:
            kp["rows"] += kp["error"] + kp["invalid"]; kp["error"] = kp["invalid"] = 0.0
            st.no_error_next = False
        tot = sum(kp.values())
        u, acc, kind = self.rng.random() * tot, 0.0, "rows"
        for k in KINDS:
            acc += kp[k]
            if u < acc:
                kind = k
                break
        g = self.p.good_given_rows[agent]
        cat = st.task.category
        if "soql_aggregate" in st.skills and cat in A.aggregate_categories:
            g += A.skill_good_bonus
        if "date_window" in st.skills and cat in A.date_categories:
            g += A.skill_good_bonus
        good = kind == "rows" and self.rng.random() < min(g, 0.99)
        n_prev = sum(1 for c in st.calls if c.role == "worker")
        it = self.p.worker_in_base + self.p.worker_in_slope * min(n_prev, 3)
        ot = self.p.worker_out[tier]
        st.calls.append(_rc("worker", tier, kind, self._cost(tier, it, ot), sub_query=st.i, retry=retry,
                            in_tokens=it, out_tokens=ot, extra={"agent": agent, "good": good}))
        st.last_worker, st.last_verdict = len(st.calls) - 1, None
        st.good[st.key()] = good
        if retry:
            st.n_retry += 1
            st.retried_current = True
        if kind == "respond":
            self._finish(st, f"answer-{[i for i, t in TIER_OF_AGENT_INDEX.items() if t == tier][0]}", "worker_respond")

    def _verify(self, st: SimState, agent: str) -> bool:
        lw = st.calls[st.last_worker]
        tier = AGENT[agent].tier
        if lw.verdict in ("error", "invalid"):
            passed, cost, it, ot = False, 0.0, 0, 0                     # structural pre-check, no model call
        else:
            it, ot = self.p.verifier_tokens
            cost = self._cost(tier, it, ot)
            if lw.verdict == "empty":
                p_fail = self.p.verifier_fail_empty
            else:
                p_fail = self.p.verifier_fail_rows[agent]["good" if st.good.get(f"{st.plan_epoch}:{lw.sub_query}") else "bad"]
            passed = self.rng.random() >= p_fail
        st.calls.append(_rc("verifier", tier, "pass" if passed else "fail", cost, sub_query=lw.sub_query,
                            in_tokens=it, out_tokens=ot, extra={"agent": agent}))
        st.last_verdict = "pass" if passed else "fail"
        return passed

    def _replan(self, st: SimState, agent: str, len_p: tuple | None = None) -> None:
        self._planner(st, agent, replan=True)
        st.plan_len = self._draw_len(len_p or self.p.replan_len_p)
        st.plan_epoch += 1
        st.i, st.n_replan = 0, st.n_replan + 1
        st.last_worker, st.last_verdict, st.retried_current = None, None, False

    def _answer(self, st: SimState, agent: str) -> None:
        tier = AGENT[agent].tier
        it, ot = self.p.answer_tokens[tier]
        st.calls.append(_rc("answer", tier, "answered", self._cost(tier, it, ot), in_tokens=it, out_tokens=ot, extra={"agent": agent}))
        self._finish(st, agent, f"answer@{agent}")

    def coverage(self, st: SimState) -> float:
        base = max(st.plan_len if st.plan_epoch == 0 else 3, 1)
        return min(1.0, sum(1 for v in st.good.values() if v) / base + st.coverage_bonus)

    def _finish(self, st: SimState, answer_agent: str, by: str) -> None:
        z = (self.p.answer_logit[answer_agent][st.task.category] + st.task.noise
             + self.p.coverage_logit * (self.coverage(st) - self.p.coverage_center))
        p = min(max(_sigmoid(z), 0.01), 0.99)
        st.final_p, st.outcome = p, 1.0 if self.rng.random() < p else 0.0
        st.done, st.answer_by = True, by

    def _advance(self, st: SimState, agent: str) -> None:
        st.i += 1
        st.retried_current = False
        if st.i < st.plan_len and self._remaining(st) >= 2:
            self._worker(st, agent)
        else:
            st.last_worker, st.last_verdict = None, None

    # ---------------------------------------------------------- decisions
    def _execute(self, st: SimState, action: str) -> None:
        op, target = split_action(action)
        if op == "answer":
            self._answer(st, target)
        elif op == "delegate":
            if st.last_verdict == "fail" and not st.retried_current and st.n_retry < self.p.max_retries:
                self._worker(st, target, retry=True)
            else:
                self._advance(st, target)
        elif op == "verify":
            self._verify(st, target)
        elif op == "replan":
            self._replan(st, target)
            if self._remaining(st) >= 2:
                self._worker(st, "data-2" if "data-2" in self.p.available_agents else self.p.available_agents[0])
        elif op == "call":
            it, ot = A.tool_cost_tokens
            st.calls.append(_rc("tool", "cheap", target, self._cost("cheap", it, ot), sub_query=st.i, in_tokens=it, out_tokens=ot))
            st.tools_called[target] = st.key()
            if target == "schema_lookup":
                st.no_error_next = True
            elif target == "aggregate_calc" and st.task.category in A.aggregate_categories:
                st.coverage_bonus += A.tool_aggregate_coverage_bonus
        elif op == "engage":
            it, ot = A.tool_cost_tokens
            st.calls.append(_rc("skill", "cheap", target, self._cost("cheap", it, ot), sub_query=st.i, in_tokens=it, out_tokens=ot))
            st.skills.append(target)
            if target == "decompose_fine":
                self._replan(st, "plan-1", len_p=(0.0, 0.0, 0.5, 0.5))
                if self._remaining(st) >= 2:
                    self._worker(st, "data-2")
        else:
            raise ValueError(action)


# the feature layout: the real orchestrator's 66 dims, then the harness block
FEATURE_NAMES = (list(MAS_FEATURE_NAMES)
                 + [f"avail={a.agent_id}" for a in AGENTS if a.mode != "planner"]
                 + [f"tool_avail={t.tool_id}" for t in TOOLS] + [f"tool_called={t.tool_id}" for t in TOOLS]
                 + [f"skill_avail={s.skill_id}" for s in SKILLS] + [f"skill_engaged={s.skill_id}" for s in SKILLS]
                 + ["stage/4", "last_fail"])
N_FEATURES = len(FEATURE_NAMES)


# ================================================================ fixed policies
class FixedPolicy:
    """The real run's two modes in this action space. always_verify: verify every unverified
    step on verify-1, retry a failed step once on data-2, otherwise run the next sub-query on
    data-2 and answer on answer-2. never_verify: data-2 until the plan is done, then answer-2.
    Propensity 1 (deterministic)."""

    def __init__(self, verify: bool, worker: str = "data-2", answer: str = "answer-2", verifier: str = "verify-1"):
        self.verify, self.worker, self.answer, self.verifier = verify, worker, answer, verifier
        self.actions = list(ACTIONS)
        self.name = "always_verify" if verify else "never_verify"
        self.meta = {"name": self.name}

    def act(self, x, legal, rng=None, eps=None) -> tuple[int, float]:
        prefs = ([f"verify@{self.verifier}"] if self.verify else []) + [f"delegate@{self.worker}", f"answer@{self.answer}"]
        for a in prefs:
            if a in legal:
                return ACTIONS.index(a), 1.0
        return ACTIONS.index(legal[0]), 1.0


class UniformPolicy:
    """Uniform over the legal set (the exploration behavior; the untrained reference b_j of FWT)."""
    actions = list(ACTIONS)
    meta = {"name": "uniform"}

    def act(self, x, legal, rng=None, eps=None) -> tuple[int, float]:
        rng = rng or random
        return ACTIONS.index(rng.choice(legal)), 1.0 / len(legal)


# ================================================================ episode runner
def run_episode(h: MASSimHarness, task: SimTask, policy, rng: random.Random, eps: float | None = None,
                mode: str = "", with_state: bool = True, legal_filter: list[str] | None = None) -> Episode:
    """One served episode: every decision logged with features, propensity (eps included when the
    policy is a pace Controller), cost, legal set and, optionally, a state snapshot for replay.
    `legal_filter` restricts the legal set to the actions a policy knows (a frozen policy does not
    use capabilities shipped after it was trained); the logged legal set is the restricted one."""
    s = h.reset(task)
    steps: list[TraceStep] = []
    while not s.done:
        legal = h.legal_actions(s)
        if legal_filter is not None:
            legal = [a for a in legal if a in legal_filter] or legal
        x = h.featurize(s)
        snap = h.snapshot(s) if with_state else None
        a, prop = policy.act(x, legal, rng, eps)
        name = policy.actions[a]
        if name not in legal:
            raise ValueError(f"policy chose illegal action {name}; legal {legal}")
        s2, done, cost, lat = h.step(s, name)
        steps.append(TraceStep(task_id=task.task_id, harness_version=h.version, step=len(steps), features=[round(v, 6) for v in x],
                               action=ACTIONS.index(name), action_name=name, propensity=float(prop), cost=cost, latency=lat,
                               legal=legal, state=snap))
        s = s2
    meta = h.episode_meta(s)
    meta.update({"mode": mode, "policy": getattr(policy, "meta", {}).get("name", type(policy).__name__), "eps": eps})
    return Episode(task_id=task.task_id, steps=steps, quality=h.quality(s), total_cost=s.cost_so_far, total_latency=0.0, meta=meta)


def serve(h: MASSimHarness, tasks: Iterable[SimTask], policy, seed: int, eps: float | None = None, mode: str = "",
          with_state: bool = True, legal_filter: list[str] | None = None) -> list[Episode]:
    rng = random.Random(seed)
    return [run_episode(h, t, policy, rng, eps, mode, with_state, legal_filter) for t in tasks]


def mode_stats(episodes: list[Episode]) -> dict:
    """The per-mode numbers the real run reported, computed the same way on simulated episodes."""
    n = len(episodes)
    proc = lambda e: e.meta["process"]
    flagged = lambda e: any(c["role"] == "verifier" and c["verdict"] == "fail" for c in proc(e))
    wrong = [e for e in episodes if e.quality < 0.5]
    corr = [e for e in episodes if e.quality >= 0.5]
    nv = sum(sum(1 for c in proc(e) if c["role"] == "verifier") for e in episodes)
    npass = sum(sum(1 for c in proc(e) if c["role"] == "verifier" and c["verdict"] == "pass") for e in episodes)
    return {"n": n, "accuracy": sum(e.quality >= 0.5 for e in episodes) / n,
            "retries": sum(e.meta["n_retry"] for e in episodes) / n,
            "expected_quality": sum(e.meta["expected_quality"] for e in episodes) / n,
            "cost": sum(e.total_cost for e in episodes) / n,
            "calls": sum(len(proc(e)) for e in episodes) / n, "verifier_calls": nv / n,
            "pass_rate": npass / nv if nv else float("nan"), "cap_hits": sum(e.meta["cap_hit"] for e in episodes) / n,
            "p_flag_given_wrong": sum(flagged(e) for e in wrong) / len(wrong) if wrong else float("nan"),
            "p_flag_given_correct": sum(flagged(e) for e in corr) / len(corr) if corr else float("nan")}
