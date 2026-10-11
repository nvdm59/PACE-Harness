"""The three roles of the multi-agent system, each a priced call through the spend ledger.

PLANNER   decomposes the CRM request into 1-3 sub-queries (goal + candidate SOQL). Cheap or mid
          tier. Purpose `mas:planner`.
WORKER    executes one sub-query through the host harness's own step path
          (`CRMArenaHarness.step(state, "CONTINUE@<tier>")`): the steering text is appended to the
          last user turn, the worker model emits <execute>SOQL</execute>, the harness runs it and
          pastes the Salesforce observation back. Tier per call. The harness prices the call as
          `agent:<tier>`; `WorkerLLM` rewrites that purpose to `mas:worker:<tier>` on its way to
          the ledger, so the ledger attributes every role call to the MAS without touching the
          harness. The final answer is the same path with the ANSWER op (`mas:worker:<tier>:answer`).
VERIFIER  checks one sub-result against the plan and the question: well-formed, non-empty when it
          should be, consistent with the goal. A structural pre-check (query error, invalid
          format) fails without a model call; otherwise one cheap-tier call returns
          {"pass": bool, "reason": str}. Purpose `mas:verifier`.

Every call is recorded as a `RoleCall` {role, tier, input summary, output summary, verdict,
reason, cost, latency, tokens}: that list is the process log the verifier-based rewards and the
orchestration reward model consume.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ...harness.base import HarnessState
from ...harness.crmarena import CRMTask, LLMResult, PricedLLM
from ...prices import TIERS

PURPOSE_PLANNER = "mas:planner"
PURPOSE_VERIFIER = "mas:verifier"
MAX_SUBQUERIES = 3

PLANNER_HEADER = "[MAS PLANNER]"
VERIFIER_HEADER = "[MAS VERIFIER]"
WORKER_HEADER = "[MAS WORKER]"


def worker_purpose(tier: str, op: str = "continue") -> str:
    return f"mas:worker:{tier}" + (":answer" if op == "answer" else "")


@dataclass
class RoleCall:
    role: str                    # planner | worker | verifier | answer
    tier: str
    purpose: str
    input_summary: str
    output_summary: str
    verdict: str | None          # verifier: pass|fail; worker: rows|empty|error|invalid|respond; answer: answered
    reason: str
    cost: float
    latency: float
    in_tokens: int = 0
    out_tokens: int = 0
    sub_query: int | None = None
    retry: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubQuery:
    goal: str
    soql: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    sub_queries: list[SubQuery]
    rationale: str = ""
    parse_ok: bool = True

    def to_dict(self) -> dict:
        return {"sub_queries": [s.to_dict() for s in self.sub_queries], "rationale": self.rationale,
                "parse_ok": self.parse_ok}


def _short(text: str, n: int = 160) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n - 3] + "..."


def _json_object(text: str) -> dict | None:
    """The first {...} block in `text` that parses as a JSON object, else None."""
    start = text.find("{")
    while start != -1:
        end = text.rfind("}")
        while end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                end = text.rfind("}", start, end)
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------- ledger hook
class WorkerLLM:
    """Purpose-rewriting proxy in front of `PricedLLM`, so the harness's own step path (which
    prices its calls as `agent:<tier>` / `agent:<tier>:answer`) shows up in the ledger as
    `mas:worker:<tier>` / `mas:worker:<tier>:answer`. Everything else (extractor, privacy judge,
    planner, verifier) passes through unchanged."""

    def __init__(self, inner: PricedLLM):
        self.inner = inner

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    @staticmethod
    def rewrite(purpose: str) -> str:
        if purpose.startswith("agent:"):
            _, tier, *rest = purpose.split(":")          # agent:<tier>[:answer][:error]
            op = "answer" if rest and rest[0] == "answer" else "continue"
            tail = rest[1:] if op == "answer" else rest
            return worker_purpose(tier, op) + "".join(":" + p for p in tail)
        return purpose

    def complete(self, tier_or_model: str, messages: list[dict], max_tokens: int, task_id: str | None,
                 purpose: str) -> LLMResult:
        return self.inner.complete(tier_or_model, messages, max_tokens, task_id, self.rewrite(purpose))


# -------------------------------------------------------------------- planner
PLANNER_INSTRUCTIONS = (
    PLANNER_HEADER + " You are the planner of a CRM data team. Decompose the user's request into 1 to "
    f"{MAX_SUBQUERIES} sub-queries that a data worker will run one at a time against the Salesforce "
    "instance described above. Each sub-query has a `goal` (what it should establish) and a candidate "
    "`soql` (SOQL or SOSL text; leave it empty if the worker should write it). Order them so later "
    "sub-queries can use earlier results. Reply with JSON only:\n"
    '{"subqueries": [{"goal": "...", "soql": "..."}], "rationale": "one sentence"}')
REPLAN_INSTRUCTIONS = (
    PLANNER_HEADER + " You are re-planning. The evidence gathered so far is listed below, with the "
    "verifier's findings. Produce the REMAINING sub-queries (1 to "
    f"{MAX_SUBQUERIES}) needed to answer the request; do not repeat sub-queries whose results were "
    "verified. Same JSON format.")


class Planner:
    def __init__(self, llm: PricedLLM, tier: str = "cheap", max_tokens: int = 600):
        assert tier in TIERS, tier
        self.llm, self.tier, self.max_tokens = llm, tier, int(max_tokens)

    @staticmethod
    def parse(text: str, question: str) -> Plan:
        obj = _json_object(text)
        subs: list[SubQuery] = []
        if obj and isinstance(obj.get("subqueries"), list):
            for s in obj["subqueries"][:MAX_SUBQUERIES]:
                if isinstance(s, dict) and str(s.get("goal", "")).strip():
                    subs.append(SubQuery(str(s["goal"]).strip(), str(s.get("soql", "") or "").strip()))
        if subs:
            return Plan(subs, str(obj.get("rationale", "") if obj else ""), True)
        # fallback: any SELECT/FIND lines become sub-queries; else one free-form sub-query
        lines = [ln.strip() for ln in text.splitlines() if re.match(r"\s*(SELECT|FIND)\b", ln, re.I)]
        if lines:
            return Plan([SubQuery(f"run: {_short(ln, 80)}", ln) for ln in lines[:MAX_SUBQUERIES]], "", False)
        return Plan([SubQuery(question.strip())], "", False)

    def plan(self, task: CRMTask, system_prompt: str, tier: str | None = None,
             evidence: list[str] | None = None) -> tuple[Plan, RoleCall]:
        tier = tier or self.tier
        if evidence is None:
            user = f"Request: {task.query.strip()}\n\n{PLANNER_INSTRUCTIONS}"
        else:
            ev = "\n".join(f"- {_short(e, 400)}" for e in evidence) or "- (none)"
            user = f"Request: {task.query.strip()}\n\nEvidence so far:\n{ev}\n\n{REPLAN_INSTRUCTIONS}"
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        r = self.llm.complete(tier, msgs, self.max_tokens, task.task_id, PURPOSE_PLANNER)
        plan = self.parse(r.text, task.query)
        call = RoleCall("planner", tier, PURPOSE_PLANNER,
                        _short(task.query), f"{len(plan.sub_queries)} sub-queries: "
                        + _short(" | ".join(s.goal for s in plan.sub_queries)),
                        "replan" if evidence is not None else "plan",
                        "" if plan.parse_ok else "plan JSON not parsed; fallback used",
                        r.usd, r.latency, r.in_tokens, r.out_tokens, extra={"parse_ok": plan.parse_ok})
        return plan, call


# --------------------------------------------------------------------- worker
class Worker:
    """Runs one sub-query through the host harness's step path."""

    def __init__(self, harness):
        self.harness = harness

    @staticmethod
    def steering(sub: SubQuery, k: int, n: int, retry_reason: str | None) -> str:
        head = f"{WORKER_HEADER} Sub-query {k + 1} of {n}: {sub.goal}"
        if sub.soql:
            head += f"\nCandidate SOQL: {sub.soql}"
        if retry_reason:
            head += (f"\nThe verifier found a problem with your previous attempt: {retry_reason}. "
                     "Fix the query and run it again.")
        return head + "\nRun exactly one query now with <execute>...</execute> (adapt the candidate if needed)."

    def run(self, state: HarnessState, sub: SubQuery, k: int, n: int, tier: str,
            retry_reason: str | None = None) -> tuple[HarnessState, RoleCall]:
        assert tier in TIERS, tier
        s = state.clone()
        msgs = list(s.extra["messages"])
        last = dict(msgs[-1])
        if last.get("role") != "user":
            msgs.append({"role": "user", "content": self.steering(sub, k, n, retry_reason)})
        else:
            last["content"] = (last.get("content") or "") + "\n\n" + self.steering(sub, k, n, retry_reason)
            msgs[-1] = last
        s.extra["messages"] = msgs
        led = self.harness.ledger
        i0, o0 = led.in_tokens, led.out_tokens
        s2, done, cost, lat = self.harness.step(s, f"CONTINUE@{tier}")
        kind = "respond" if (done and s2.extra.get("end_reason") == "agent_respond") else s2.extra["last_obs"]
        executed = s2.extra["actions"][-1] if (kind != "invalid" and s2.extra["actions"]) else ""
        obs = ""
        if kind in ("rows", "empty", "error"):
            obs = s2.extra["messages"][-1].get("content") or ""
        out = f"{kind}: {_short(executed, 100)}" if kind != "respond" else f"respond: {_short(executed, 100)}"
        call = RoleCall("worker", tier, worker_purpose(tier), _short(sub.goal), out, kind,
                        "" if kind in ("rows", "respond") else f"observation kind {kind}", cost, lat,
                        led.in_tokens - i0, led.out_tokens - o0, sub_query=k, retry=retry_reason is not None,
                        extra={"executed": executed[:500], "observation": obs[:3000]})
        return s2, call


class Answerer:
    """The final answer: the harness's ANSWER op on the chosen tier (forced <respond>)."""

    def __init__(self, harness):
        self.harness = harness

    def run(self, state: HarnessState, tier: str) -> tuple[HarnessState, RoleCall]:
        assert tier in TIERS, tier
        led = self.harness.ledger
        i0, o0 = led.in_tokens, led.out_tokens
        s2, done, cost, lat = self.harness.step(state, f"ANSWER@{tier}")
        call = RoleCall("answer", tier, worker_purpose(tier, "answer"), "final answer",
                        _short(str(s2.extra.get("answer")), 120), "answered", "", cost, lat,
                        led.in_tokens - i0, led.out_tokens - o0)
        return s2, call


# ------------------------------------------------------------------- verifier
@dataclass
class Verdict:
    passed: bool
    reason: str
    checked_by: str          # structural | llm


VERIFIER_INSTRUCTIONS = (
    VERIFIER_HEADER + " You verify one step of a CRM data worker. Given the user's request, the "
    "sub-query goal, the query that was run and the Salesforce output, decide whether this step's "
    "result is usable: well-formed, non-empty when the goal needs rows, and consistent with the goal "
    "(right object, right filter, right field). Reply with JSON only: {\"pass\": true|false, "
    "\"reason\": \"one sentence\"}")


class Verifier:
    def __init__(self, llm: PricedLLM, tier: str = "cheap", max_tokens: int = 200, obs_max_chars: int = 3000):
        assert tier in TIERS, tier
        self.llm, self.tier, self.max_tokens, self.obs_max_chars = llm, tier, int(max_tokens), int(obs_max_chars)

    @staticmethod
    def structural(worker: RoleCall) -> Verdict | None:
        if worker.verdict == "error":
            return Verdict(False, "the query returned an error", "structural")
        if worker.verdict == "invalid":
            return Verdict(False, "the worker produced no <execute> query", "structural")
        return None

    @staticmethod
    def parse(text: str) -> Verdict:
        obj = _json_object(text)
        if obj is not None and "pass" in obj:
            p = obj["pass"]
            passed = p if isinstance(p, bool) else str(p).strip().lower() in ("true", "yes", "pass", "1")
            return Verdict(bool(passed), _short(str(obj.get("reason", "")), 200), "llm")
        low = text.lower()
        if re.search(r"\bfail", low) and not re.search(r"\bpass", low):
            return Verdict(False, _short(text, 200), "llm")
        if re.search(r"\bpass", low) and not re.search(r"\bfail", low):
            return Verdict(True, _short(text, 200), "llm")
        return Verdict(False, "unparsable verdict (treated as fail)", "llm")

    def verify(self, task: CRMTask, sub: SubQuery, worker: RoleCall, tier: str | None = None) -> tuple[Verdict, RoleCall]:
        tier = tier or self.tier
        v = self.structural(worker)
        if v is not None:
            call = RoleCall("verifier", tier, PURPOSE_VERIFIER, _short(sub.goal), v.reason, "fail", v.reason,
                            0.0, 0.0, sub_query=worker.sub_query, extra={"checked_by": "structural"})
            return v, call
        obs = worker.extra.get("observation", "")
        if len(obs) > self.obs_max_chars:
            obs = obs[:self.obs_max_chars] + f" ... [truncated {len(obs) - self.obs_max_chars} chars]"
        user = (f"Request: {task.query.strip()}\nSub-query goal: {sub.goal}\n"
                f"Query run: {worker.extra.get('executed', '')}\nOutput: {obs}\n\n{VERIFIER_INSTRUCTIONS}")
        msgs = [{"role": "user", "content": user}]
        t0 = time.time()
        r = self.llm.complete(tier, msgs, self.max_tokens, task.task_id, PURPOSE_VERIFIER)
        v = self.parse(r.text)
        call = RoleCall("verifier", tier, PURPOSE_VERIFIER, _short(sub.goal), v.reason,
                        "pass" if v.passed else "fail", v.reason, r.usd, max(r.latency, time.time() - t0),
                        r.in_tokens, r.out_tokens, sub_query=worker.sub_query, extra={"checked_by": "llm"})
        return v, call
