"""A deterministic fake of the two paid dependencies, for `--dry-run` and the tests.

`FakeBackend`     stands in for `CRMArenaBackend`: fake task records over the real CRMArena-Pro
                  category list, a fake Salesforce `run_query`, and an evaluator that runs one
                  priced extractor call through the harness's shim, like CRMArena's does.
`FakeCompletion`  stands in for `litellm.completion`: routes on the role headers in the prompt
                  (planner / verifier / worker / forced answer / extractor) and scripts a reply
                  with token counts in the range the real run measured.

Everything is a hash of (seed, task id, role, call index), so a run is reproducible and the tests
can assert exact ledger lines. The generator has a LATENT model, written down here so the dry-run
tables are interpretable as a pipeline check and nothing more:

  * each task is easy or hard (1/3 hard);
  * a worker query returns rows with p = 0.85 (easy) / 0.55 (hard), an error with p = 0.08, else
    empty; a retry after a verifier fail returns rows with p = 0.85;
  * the verifier flags empty results with p = 0.85 and passes rows with p = 0.90 (so it has both
    misses and false alarms);
  * the final answer is correct with p = base(tier) + 0.12 x (worker steps that returned rows,
    at most 3) - 0.25 x hard, base = cheap 0.40 / mid 0.55 / frontier 0.65, clipped to [0.05, 0.95].

So the outcome depends on the tiers used and on the evidence the process gathered, which is what
an orchestration reward model should be able to pick up, and the verifier is imperfect, so the
silent-failure cells are populated.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

from ...harness.crmarena import CATEGORIES, FORCE_ANSWER, CRMTask
from .roles import PLANNER_HEADER, VERIFIER_HEADER, WORKER_HEADER

EXTRACTOR_MARKER = "extracted_answers"
HARD_FRACTION = 1.0 / 3.0
P_ROWS = {"easy": 0.85, "hard": 0.55}
P_ERROR = 0.08
P_ROWS_RETRY = 0.85
P_VERIFIER_FLAGS_EMPTY = 0.85
P_VERIFIER_PASSES_ROWS = 0.90
ANSWER_BASE = {"cheap": 0.40, "mid": 0.55, "frontier": 0.65}
ANSWER_ROWS_BONUS = 0.12
ANSWER_HARD_PENALTY = 0.25
TOKENS = {"planner": (4400, 260), "verifier": (1400, 55), "extractor": (520, 24),
          "worker_in_first": 4100, "worker_in_growth": 1900,
          "worker_out": {"cheap": 160, "mid": 420, "frontier": 430}, "answer_out": 220}


def _u(seed: int, *key) -> float:
    h = hashlib.sha256(json.dumps([seed, *key], sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def fake_tasks(n: int = 66, org: str = "b2b", seed: int = 0) -> list[CRMTask]:
    out = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        metric = "privacy_rejection" if cat.startswith(("private", "internal", "confidential")) else "exact_match"
        out.append(CRMTask(task_id=f"{org}-{i}", idx=i, category=cat, query=f"Fake CRM question {i} about {cat}?",
                           answer=[f"ANS-{i}"], reward_metric=metric, metadata={}, org_type=org))
    return out


class FakeBackend:
    rule_string = "Invalid output format! Use <execute> or <respond>."

    def __init__(self, org: str = "b2b", n_tasks: int = 66, seed: int = 0):
        self.org, self.n, self.seed = org, int(n_tasks), int(seed)
        self.queries: list[str] = []

    def load_tasks(self) -> list[CRMTask]:
        return fake_tasks(self.n, self.org, self.seed)

    def difficulty(self, task_id: str) -> str:
        return "hard" if _u(self.seed, "difficulty", task_id) < HARD_FRACTION else "easy"

    def system_prompt(self, task, eval_mode="aided", privacy_aware=False) -> str:
        return ("FAKE SYSTEM PROMPT. Salesforce instance schema: Case(Id, OwnerId, Status), Lead(Id, Status), "
                f"Opportunity(Id, Amount, StageName). Task category: {task.category}. " + "x" * 400)

    def external_facing(self) -> set[str]:
        return set()

    def run_query(self, query: str):
        self.queries.append(query)
        m = re.search(r"task=(\S+)", query)
        tid = m.group(1) if m else "?"
        retry = "retry" in query
        u = _u(self.seed, "query", query)
        if "INVALID" in query:
            return "MALFORMED_QUERY: unexpected token", 0
        p_rows = P_ROWS_RETRY if retry else P_ROWS[self.difficulty(tid)]
        if u < p_rows:
            k = 1 + int(_u(self.seed, "nrows", query) * 4)
            return [{"Id": f"{tid}-{j}", "Value": 100 + j} for j in range(k)], 1
        if u < p_rows + P_ERROR:
            return "INVALID_FIELD: No such column 'Foo' on entity 'Case'", 0
        return [], 1

    def make_evaluator(self, shim):
        return FakeEvaluator(shim)


class FakeEvaluator:
    """Exact match after one priced extractor call through the harness shim (mirrors the real
    Evaluator's LLM extraction step; the real one also handles fuzzy metrics)."""

    def __init__(self, shim):
        self.shim = shim

    def evaluate(self, proposed, gold, metric, task_name, traj):
        msgs = [{"role": "user", "content": f"Extract the answer entities as JSON {{\"{EXTRACTOR_MARKER}\": [...]}}"
                                            f" from: {proposed}"}]
        res = self.shim.completion(model="claude-haiku-4-5", messages=msgs)
        text = res.choices[0].message.content
        try:
            ext = json.loads(text).get(EXTRACTOR_MARKER, [])
        except json.JSONDecodeError:
            ext = [proposed]
        ok = any(str(g) in ext for g in gold)
        return {"reward": 1 if ok else 0, "parsed_answer": ext}


class FakeCompletion:
    """Scripted litellm.completion keyed on the prompt's role header."""

    def __init__(self, backend: FakeBackend, seed: int = 0):
        self.backend, self.seed = backend, int(seed)
        self.calls: list[dict] = []
        self.n_by_task: dict[str, int] = {}

    @staticmethod
    def _resp(text: str, pin: int, pout: int):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=int(pin), completion_tokens=int(pout)))

    @staticmethod
    def _task_id(messages: list[dict]) -> str:
        for m in messages:
            mm = re.search(r"Fake CRM question (\d+)", m.get("content") or "")
            if mm:
                return mm.group(1)
        return "?"

    def __call__(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        model = kw["model"].split("/")[-1]
        tier = {"claude-haiku-4-5": "cheap", "claude-sonnet-5-5": "mid", "claude-opus-5-5": "frontier"}[model]
        last = msgs[-1]["content"] or ""
        tid_short = self._task_id(msgs)
        tid = f"{self.backend.org}-{tid_short}"
        k = self.n_by_task[tid] = self.n_by_task.get(tid, 0) + 1
        if EXTRACTOR_MARKER in last and "from:" in last:
            m = re.search(r"ANS-\d+(?:-wrong)?", last)
            return self._resp(json.dumps({EXTRACTOR_MARKER: [m.group(0) if m else "none"]}), *TOKENS["extractor"])
        if PLANNER_HEADER in last:
            n_sub = 1 + int(_u(self.seed, tid, "plan_len", k) * 3)
            subs = [{"goal": f"sub-goal {j + 1} for task {tid_short}",
                     "soql": f"SELECT Id, Value FROM Case WHERE task={tid} sub={j} plan={k}"} for j in range(n_sub)]
            text = json.dumps({"subqueries": subs, "rationale": "fake decomposition"})
            if _u(self.seed, tid, "plan_parse", k) < 0.05:
                text = "I would first look at cases, then owners."       # unparsable plan -> fallback path
            return self._resp(text, TOKENS["planner"][0] + len(last) // 20, TOKENS["planner"][1])
        if VERIFIER_HEADER in last:
            if "Output: Salesforce instance output: []" in last:
                flag = _u(self.seed, tid, "verifier", k) < P_VERIFIER_FLAGS_EMPTY
                text = json.dumps({"pass": not flag, "reason": "empty result; the goal needs rows" if flag
                                   else "empty result is plausible for this filter"})
            else:
                ok = _u(self.seed, tid, "verifier", k) < P_VERIFIER_PASSES_ROWS
                text = json.dumps({"pass": ok, "reason": "rows match the goal" if ok
                                   else "rows are from the wrong object for this goal"})
            return self._resp(text, *TOKENS["verifier"])
        n_asst = sum(m["role"] == "assistant" for m in msgs)
        pin = TOKENS["worker_in_first"] + TOKENS["worker_in_growth"] * n_asst
        if FORCE_ANSWER.strip() in last:
            return self._resp(self._answer(msgs, tid, tier, k), pin, TOKENS["answer_out"])
        # worker turn
        m = re.search(r"Candidate SOQL: (.+)", last)
        soql = m.group(1).strip() if m else f"SELECT Id FROM Case WHERE task={tid} freeform={k}"
        if "verifier found a problem" in last:
            soql += " -- retry"
        if _u(self.seed, tid, "invalid", k) < 0.03:
            return self._resp("I think the answer involves cases.", pin, 40)           # no <execute>: invalid turn
        if n_asst >= 2 and tier == "frontier" and _u(self.seed, tid, "early", k) < 0.15:
            return self._resp(self._answer(msgs, tid, tier, k), pin, TOKENS["worker_out"][tier])
        return self._resp(f"<thought> run sub-query </thought> <execute> {soql} </execute>", pin,
                          TOKENS["worker_out"][tier])

    def _answer(self, msgs: list[dict], tid: str, tier: str, k: int) -> str:
        n_rows = sum(1 for m in msgs if m["role"] == "user" and "Salesforce instance output: [{" in (m["content"] or ""))
        hard = self.backend.difficulty(tid) == "hard"
        p = ANSWER_BASE[tier] + ANSWER_ROWS_BONUS * min(n_rows, 3) - (ANSWER_HARD_PENALTY if hard else 0.0)
        p = min(0.95, max(0.05, p))
        correct = _u(self.seed, tid, "answer", n_rows, tier) < p
        idx = tid.split("-")[-1]
        return f"<respond> ANS-{idx}{'' if correct else '-wrong'} </respond>"
