"""One Socratic self-refinement (SSR) round on a CRMArena-Pro answer, priced through the ledger.

SSR (arXiv 2511.10621) treats a response as a set of verifiable sub-claims: decompose, re-solve
each sub-claim independently, estimate confidence per step, and rewrite only where confidence is
low. Here the response is a CRMArena-Pro answer and the "re-solve" is a real SOQL worker.

A round is split in two so that a gate can look before it pays:

  probe(node)      one cheap-tier call (`ssr:decompose`) that turns the current answer into 1-3
                   sub-claims, each with a Socratic question, the value the claim asserts and a
                   self-reported confidence. This is the gating signal. It costs about $0.003.
  refine_step(node, tier)
                   for every claim, k independent worker mini-episodes on `tier` through
                   `CRMArenaHarness.step` (`ssr:resolve:<tier>`; the SOQL runs on the real
                   backend), then per-claim confidence = mean(self-reported, agreement across the
                   k samples, support for the asserted value), then one rewrite call on `tier`
                   (`ssr:refine:<tier>`) that sees the claims and the workers' findings. The
                   refined answer is scored by CRMArena's own Evaluator (extractor / privacy judge
                   billed as in the core harness, not charged to the round).

The harness is not modified. The ledger purposes are set by `tag_purpose`, a context manager that
wraps `harness.llm` for the duration of a worker episode.
"""
from __future__ import annotations

import contextlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ...harness.crmarena import CRMTask, parse_action
from ...prices import TIERS

MAX_CLAIMS = 3
DECOMPOSE_MAX_TOKENS = 600
REFINE_MAX_TOKENS = 512
RESOLVE_MAX_TURNS = 3           # worker mini-episode: CONTINUE, CONTINUE, then ANSWER is forced
K_SAMPLES = 2

DECOMPOSE_SYSTEM = (
    "You audit answers produced by a CRM assistant that queries a Salesforce instance with SOQL/SOSL. "
    "You do not run queries yourself. You decompose a proposed answer into verifiable sub-claims and "
    "estimate how likely each is to be true. Be calibrated: a confidence of 0.9 should be wrong about "
    "one time in ten.")

DECOMPOSE_USER = """Task: {task_id}
Question asked of the assistant:
{query}

Queries the assistant ran before answering ({n_queries}):
{queries}

Proposed answer:
{answer}

Decompose the proposed answer into 1 to {max_claims} sub-claims that together imply the answer. \
For each sub-claim write one Socratic question that a worker can settle by running SOQL/SOSL on the \
same Salesforce instance, the value the sub-claim asserts (short, literal), and your confidence in \
[0, 1] that the sub-claim is true. Then give your confidence that the proposed answer as a whole is \
correct. If the proposed answer is a refusal or "None", make the single claim "the request should be \
refused / cannot be answered" with a question that checks whether the data is accessible.
Output JSON only, no prose:
{{"claims": [{{"claim": "...", "question": "...", "expected": "...", "confidence": 0.0}}], "answer_confidence": 0.0}}"""

REFINE_SYSTEM = (
    "You finalize answers produced by a CRM assistant that queries a Salesforce instance. You are given "
    "the question, the proposed answer, and the findings of independent workers who re-checked each "
    "sub-claim with their own queries. Keep the proposed answer when the findings support it; correct "
    "it when they contradict it; keep a refusal when the request concerns confidential data. Do not "
    "hedge, do not run queries, and output exactly one <respond> ... </respond> block with the final "
    "answer in the same format as the proposed answer.")

REFINE_USER = """Task: {task_id}
Question:
{query}

Proposed answer:
{answer}

Sub-claims and worker findings:
{findings}

Give the final answer now. Output only <respond> final answer </respond>."""

WORKER_SUFFIX = ("\n\nAnswer this sub-question only. Run the queries you need, then reply with "
                 "<respond> the value </respond> and nothing else in the response.")


# ----------------------------------------------------------------- hooks
class _TaggedLLM:
    """Delegates to a PricedLLM but overrides the ledger purpose of every call."""

    def __init__(self, inner, purpose: str):
        self.inner, self.purpose = inner, purpose
        self.usd = 0.0
        self.calls = 0

    def complete(self, tier_or_model, messages, max_tokens, task_id, purpose):
        r = self.inner.complete(tier_or_model, messages, max_tokens, task_id, self.purpose)
        self.usd += r.usd
        self.calls += 1
        return r

    def __getattr__(self, name):           # effort, ledger, timeout ... fall through
        return getattr(self.inner, name)


@contextlib.contextmanager
def tag_purpose(harness, purpose: str):
    """Within the block, every call the harness makes is billed under `purpose`."""
    tagged = _TaggedLLM(harness.llm, purpose)
    orig = harness.llm
    harness.llm = tagged
    try:
        yield tagged
    finally:
        harness.llm = orig


# ------------------------------------------------------------- data model
@dataclass
class Claim:
    claim: str
    question: str
    expected: str
    self_conf: float
    samples: list[str] = field(default_factory=list)   # the k worker answers (normalized on compare)
    agree: float | None = None                         # all k samples agree
    support: float | None = None                       # majority sample matches `expected`
    conf: float | None = None                          # mean(self_conf, agree, support)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Probe:
    claims: list[Claim]
    answer_conf: float
    min_conf: float            # min self-reported claim confidence: the gate's signal
    mean_conf: float
    usd: float
    calls: int
    parse_ok: bool
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d


@dataclass
class RefineResult:
    tier: str
    claims: list[Claim]
    conf_min: float            # min per-claim conf after re-solving (self + agreement + support)
    conf_mean: float
    answer_before: str
    answer_after: str
    quality_before: float
    quality_after: float
    changed: bool
    usd_resolve: float
    usd_refine: float
    usd_eval: float            # extractor / privacy judge: billed, not charged (as in the core harness)
    calls: int
    worker_turns: int

    @property
    def usd(self) -> float:
        return self.usd_resolve + self.usd_refine

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        d["usd"] = self.usd
        return d


# ------------------------------------------------------------- utilities
def normalize_answer(s: Any) -> str:
    """Lowercase, trim, drop quotes/brackets, sort comma-separated parts (ids, lists)."""
    s = "" if s is None else str(s)
    s = re.sub(r"</?respond>", "", s, flags=re.I)
    s = s.strip().strip("<>").strip()
    s = re.sub(r"[\"'\[\]\(\)\{\}]", "", s).lower()
    s = re.sub(r"\s+", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(sorted(parts))


def matches(a: Any, b: Any) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return na == nb
    return na == nb or (len(nb) >= 4 and nb in na) or (len(na) >= 4 and na in nb)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    for cand in (m.group(0), text):
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def _clip01(x: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


def parse_decomposition(text: str, answer: str, query: str) -> tuple[list[Claim], float, bool]:
    """JSON -> claims. On a parse failure, fall back to one claim covering the whole answer at
    confidence 0.5, so the gate sees 'uncertain' rather than crashing a paid run."""
    d = _extract_json(text)
    claims: list[Claim] = []
    if d and isinstance(d.get("claims"), list):
        for c in d["claims"][:MAX_CLAIMS]:
            if not isinstance(c, dict):
                continue
            claim = str(c.get("claim") or "").strip()
            q = str(c.get("question") or "").strip()
            if not claim or not q:
                continue
            claims.append(Claim(claim, q, str(c.get("expected") or answer).strip(), _clip01(c.get("confidence"))))
    if claims:
        return claims, _clip01(d.get("answer_confidence"), min(c.self_conf for c in claims)), True
    fallback = Claim(f"The answer to the question is {answer}",
                     f"What is the answer to: {query}", str(answer), 0.5)
    return [fallback], 0.5, False


def format_queries(queries: list[str], max_chars: int = 400) -> str:
    out = []
    for i, q in enumerate(queries or [], 1):
        q = re.sub(r"\s+", " ", str(q)).strip()
        out.append(f"{i}. {q[:max_chars]}{' ...' if len(q) > max_chars else ''}")
    return "\n".join(out) if out else "(none)"


# ----------------------------------------------------------------- probe
def probe(harness, task: CRMTask, answer: str, queries: list[str], decompose_tier: str = "cheap") -> Probe:
    """Decompose `answer` into claims with self-reported confidence (one `ssr:decompose` call)."""
    user = DECOMPOSE_USER.format(task_id=task.task_id, query=task.query.strip(), n_queries=len(queries or []),
                                 queries=format_queries(queries), answer=str(answer), max_claims=MAX_CLAIMS)
    msgs = [{"role": "system", "content": DECOMPOSE_SYSTEM}, {"role": "user", "content": user}]
    r = harness.llm.complete(decompose_tier, msgs, DECOMPOSE_MAX_TOKENS, task.task_id, "ssr:decompose")
    claims, aconf, ok = parse_decomposition(r.text, str(answer), task.query)
    confs = [c.self_conf for c in claims]
    return Probe(claims, aconf, min(confs), sum(confs) / len(confs), r.usd, 1, ok, r.text[:500])


# ---------------------------------------------------------------- workers
def worker_episode(harness, task: CRMTask, question: str, tier: str, max_turns: int = RESOLVE_MAX_TURNS,
                   round_idx: int = 1, claim_idx: int = 0, sample_idx: int = 0) -> tuple[str, float, int]:
    """One sub-question solved through the harness step path (CONTINUE@tier until the agent
    answers; ANSWER@tier forced on the last allowed turn). Returns (answer, usd, turns)."""
    sub = CRMTask(task_id=task.task_id, idx=task.idx, category=task.category, query=question.strip() + WORKER_SUFFIX,
                  answer=list(task.answer), reward_metric=task.reward_metric, metadata=dict(task.metadata),
                  org_type=task.org_type)
    s = harness.reset(sub)
    s.extra["ssr"] = {"round": round_idx, "claim": claim_idx, "sample": sample_idx}
    usd, turns = 0.0, 0
    while not s.done:
        act = f"CONTINUE@{tier}" if s.step < max_turns - 1 else f"ANSWER@{tier}"
        s, _, c, _ = harness.step(s, act)
        usd += c
        turns += 1
    return str(s.extra.get("answer") or "None"), usd, turns


def resolve_claims(harness, task: CRMTask, claims: list[Claim], tier: str, k: int = K_SAMPLES,
                   max_turns: int = RESOLVE_MAX_TURNS, round_idx: int = 1) -> tuple[float, int, int]:
    """Re-solve every claim k times on `tier`; fills samples / agree / support / conf in place.
    Returns (usd, calls, worker turns)."""
    assert tier in TIERS, tier
    usd = calls = turns = 0
    with tag_purpose(harness, f"ssr:resolve:{tier}") as tagged:
        for ci, c in enumerate(claims):
            c.samples = []
            for si in range(k):
                ans, u, t = worker_episode(harness, task, c.question, tier, max_turns, round_idx, ci, si)
                c.samples.append(ans)
                usd += u
                turns += t
            norm = [normalize_answer(a) for a in c.samples]
            c.agree = 1.0 if len(set(norm)) == 1 else 0.0
            majority = max(norm, key=lambda v: (norm.count(v), -norm.index(v)))   # ties -> first sample
            c.support = 1.0 if matches(majority, c.expected) else 0.0
            c.conf = (c.self_conf + c.agree + c.support) / 3.0
        calls = tagged.calls
    return usd, calls, turns


# ----------------------------------------------------------------- refine
def format_findings(claims: list[Claim]) -> str:
    out = []
    for i, c in enumerate(claims, 1):
        samples = "; ".join(f"worker {j + 1}: {s}" for j, s in enumerate(c.samples))
        verdict = "consistent with the claim" if c.support else "does NOT match the asserted value"
        agree = "workers agree" if c.agree else "workers disagree"
        out.append(f"{i}. claim: {c.claim}\n   asserts: {c.expected}\n   question: {c.question}\n"
                   f"   findings: {samples}\n   {agree}; {verdict}; confidence {c.conf:.2f}")
    return "\n".join(out)


def rewrite(harness, task: CRMTask, answer: str, claims: list[Claim], tier: str) -> tuple[str, float]:
    user = REFINE_USER.format(task_id=task.task_id, query=task.query.strip(), answer=str(answer),
                              findings=format_findings(claims))
    msgs = [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": user}]
    r = harness.llm.complete(tier, msgs, REFINE_MAX_TOKENS, task.task_id, f"ssr:refine:{tier}")
    act = parse_action(r.text)
    refined = act["content"] if act and act["name"] == "respond" else r.text.strip()
    return (refined or "None"), r.usd


def score_answer(harness, task: CRMTask, answer: str, trajectory: list[str]) -> tuple[float, float, dict]:
    """CRMArena's Evaluator on an arbitrary answer, via a terminal state. Returns (quality, eval usd, eval)."""
    s = harness.reset(task)
    s.extra["answer"] = answer
    s.extra["actions"] = list(trajectory) + [answer]
    s.done = True
    ev = harness.evaluate(s)
    return float(ev["quality"]), float(ev.get("eval_cost_usd", 0.0)), ev


def refine_step(harness, task: CRMTask, answer: str, quality_before: float, claims: list[Claim], tier: str,
                queries: list[str], k: int = K_SAMPLES, max_turns: int = RESOLVE_MAX_TURNS,
                round_idx: int = 1) -> RefineResult:
    """Resolve + rewrite + score on `tier`. `claims` come from `probe` and are copied, so the same
    decomposition can be refined on several tiers (the counterfactual tree)."""
    cl = [Claim(c.claim, c.question, c.expected, c.self_conf) for c in claims]
    usd_res, calls, turns = resolve_claims(harness, task, cl, tier, k, max_turns, round_idx)
    refined, usd_ref = rewrite(harness, task, answer, cl, tier)
    q_after, usd_eval, _ = score_answer(harness, task, refined, queries)
    confs = [c.conf for c in cl]
    return RefineResult(tier=tier, claims=cl, conf_min=min(confs), conf_mean=sum(confs) / len(confs),
                        answer_before=str(answer), answer_after=refined, quality_before=float(quality_before),
                        quality_after=q_after, changed=normalize_answer(refined) != normalize_answer(answer),
                        usd_resolve=usd_res, usd_refine=usd_ref, usd_eval=usd_eval, calls=calls + 1,
                        worker_turns=turns)
