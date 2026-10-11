"""Scripted drift on a day schedule: the harness evolves by STAGES, each a named, dated generation.

EvoHarnessBench (arXiv 2609.04280) grows a harness H_t = {tools, skills, specialist agents}
monotonically, H_1 ⊆ H_2 ⊆ ..., and measures what a FIXED agent policy loses or gains per axis
(tools / skills / agents). Here the axes are the same, plus repricing (the drift the PACE core
already models), and the question is turned around: the structure at each stage is frozen and
the ORCHESTRATOR'S POLICY adapts online.

Stages (the pre-registered default schedule, 40 days, events on days 9 / 17 / 25 / 33):

  g0  baseline        the real run's pool: plan-1, data-1..3, verify-1, answer-1..3; list prices
  g1  reprice         mid x0.5 then frontier x1.5 (axis: price)
  g2  agents+         verify-2 (better verifier: P(flag|correct) 0.96 -> 0.30), data-4 (database
                      specialist, cheap), answer-4 (reporting specialist, cheap)   (axis: agents)
  g3  tools+          schema_lookup, aggregate_calc                                 (axis: tools)
  g4  skills+         soql_aggregate, date_window, decompose_fine; the planner now decomposes
                      more finely (plan length 1..4)                                (axis: skills)

The MILD schedule (sensitivity row) keeps the same days and axes with smaller effects: reprice
mid x0.8 / frontier x1.2, verify-2 at P(flag|correct) 0.60, data-4 at rows 0.78 / good 0.75,
answer-4 mid-grade on 3 categories, half the tool and skill bonuses, planner v2 at 1..3 sub-queries.
Both schedules were written down before the first full run and were not changed afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from . import simenv
from .simenv import AGENTS, A, Params

EVENT_DAYS = (9, 17, 25, 33)
DAY0 = "2026-09-01"       # day 1; dates are labels for the generations, not run dates


def _date(day: int) -> str:
    import datetime as dt
    return (dt.date.fromisoformat(DAY0) + dt.timedelta(days=day - 1)).isoformat()


@dataclass(frozen=True)
class Generation:
    index: int
    name: str
    axis: str              # price | agents | tools | skills | none
    day: int               # first day it is in service
    date: str
    description: str
    params: Params

    @property
    def version(self) -> str:
        return f"mas-sim-{self.name}"


# ------------------------------------------------------------------ events
def reprice(p: Params, mid: float, frontier: float) -> Params:
    f = dict(p.price_factors)
    f["mid"] *= mid
    f["frontier"] *= frontier
    return replace(p, price_factors=f)


def add_agents(p: Params, stage: int, verify2_fail_rows_good: float, data4: dict, answer4_strong: list[str]) -> Params:
    kinds = dict(p.worker_kind_p)
    kinds["data-4"] = {k: data4[k] for k in simenv.KINDS}
    good = dict(p.good_given_rows)
    good["data-4"] = data4["good"]
    vfail = dict(p.verifier_fail_rows)
    vfail["verify-2"] = {"good": verify2_fail_rows_good, "bad": p.verifier_fail_rows["verify-1"]["bad"]}
    logit = {k: dict(v) for k, v in p.answer_logit.items()}
    logit["answer-4"] = {c: (logit["answer-2"][c] if c in answer4_strong else logit["answer-1"][c]) for c in logit["answer-2"]}
    return replace(p, worker_kind_p=kinds, good_given_rows=good, verifier_fail_rows=vfail, answer_logit=logit).with_stage_pool(stage)


def add_tools(p: Params, stage: int) -> Params:
    return p.with_stage_pool(stage)


def add_skills(p: Params, stage: int, plan_len_p: tuple) -> Params:
    return replace(p, plan_len_p=plan_len_p).with_stage_pool(stage)


# ------------------------------------------------------------------ schedules
SCHEDULES = {
    "default": {"reprice": (0.5, 1.5), "verify2_fail_rows_good": A.verify2_fail_rows_good, "data4": A.data4,
                "answer4_strong": A.answer4_strong_categories, "bonus_scale": 1.0,
                "planner_v2_plan_len_p": A.planner_v2_plan_len_p},
    "mild": {"reprice": (0.8, 1.2), "verify2_fail_rows_good": 1 - 0.60 ** (1 / 2.94),
             "data4": {"rows": 0.78, "empty": 0.14, "error": 0.05, "invalid": 0.03, "respond": 0.0, "good": 0.75},
             "answer4_strong": A.answer4_strong_categories[:3], "bonus_scale": 0.5,
             "planner_v2_plan_len_p": (0.0, 0.10, 0.90, 0.0)},
}


def schedule(name: str = "default", base: Params | None = None, event_days: tuple = EVENT_DAYS,
             verifier2_from_day1: bool = False, max_calls: int = simenv.MAX_CALLS,
             verifier_exempt_from_cap: bool = False) -> list[Generation]:
    """The list of generations in service order. `bonus_scale` is applied by the harness through
    `simenv.A` at run time (see `apply_bonus_scale`). The three keyword flags are the verifier
    ablation (RESULTS_ablation_verifier.md): the informative verifier verify-2 available to every
    arm from day 1, a higher role-call cap, and verifier calls exempt from the cap."""
    s = SCHEDULES[name]
    p0 = base or simenv.measured_params()
    p0 = replace(p0, max_calls=int(max_calls), verifier_exempt_from_cap=bool(verifier_exempt_from_cap))
    if verifier2_from_day1:
        vfail = dict(p0.verifier_fail_rows)
        vfail["verify-2"] = {"good": s["verify2_fail_rows_good"], "bad": p0.verifier_fail_rows["verify-1"]["bad"]}
        p0 = replace(p0, verifier_fail_rows=vfail, available_agents=tuple(p0.available_agents) + ("verify-2",))
    gens = [Generation(0, "g0-baseline", "none", 1, _date(1), "the real run's harness (plan-1, data-1..3, verify-1, answer-1..3), list prices", p0)]
    p1 = reprice(p0, *s["reprice"])
    gens.append(Generation(1, "g1-reprice", "price", event_days[0], _date(event_days[0]),
                           f"repricing: mid x{s['reprice'][0]}, frontier x{s['reprice'][1]}", replace(p1, stage=1)))
    p2 = add_agents(p1, 2, s["verify2_fail_rows_good"], s["data4"], s["answer4_strong"])
    gens.append(Generation(2, "g2-agents", "agents", event_days[1], _date(event_days[1]),
                           f"agent pool + verify-2 (per-call P(fail | good rows) {s['verify2_fail_rows_good']:.3f}), "
                           f"data-4 (rows {s['data4']['rows']}, good {s['data4']['good']}), answer-4 ({len(s['answer4_strong'])} strong categories)", p2))
    p3 = add_tools(p2, 3)
    gens.append(Generation(3, "g3-tools", "tools", event_days[2], _date(event_days[2]),
                           f"tools + schema_lookup, aggregate_calc (bonus scale {s['bonus_scale']})", p3))
    p4 = add_skills(p3, 4, s["planner_v2_plan_len_p"])
    gens.append(Generation(4, "g4-skills", "skills", event_days[3], _date(event_days[3]),
                           f"skills + soql_aggregate, date_window, decompose_fine; planner v2 plan length p = {s['planner_v2_plan_len_p']}", p4))
    return gens


def apply_bonus_scale(name: str) -> None:
    """Scale the assumed tool / skill effects for the mild schedule (module-level constants in
    simenv.A; the harness reads them at call time)."""
    scale = SCHEDULES[name]["bonus_scale"]
    A.skill_good_bonus = 0.15 * scale
    A.tool_aggregate_coverage_bonus = 0.34 * scale


def generation_for(day: int, gens: list[Generation]) -> Generation:
    cur = gens[0]
    for g in gens:
        if g.day <= day:
            cur = g
    return cur


def changed_fields(a: Params, b: Params) -> list[str]:
    """Names of the Params fields that differ between two generations (for the report and tests)."""
    out = []
    for f in a.__dataclass_fields__:
        if getattr(a, f) != getattr(b, f):
            out.append(f)
    return out
