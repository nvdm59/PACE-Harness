"""A fake CRMArena world for tests and --dry-run: no key, no network, no Salesforce, no spend.

The fake agent answers correctly only when a procedural-hints block is in its system prompt, so a
dry run exercises the whole path (distill -> serve -> analyze) and shows a visible lift."""
from __future__ import annotations

import json
import re

from ...harness.crmarena import CRMTask
from .distill import WRITER_SYSTEM
from .serve import HINT_HEADER

FAKE_SPEC = [("b2b-1", "handle_time"), ("b2b-2", "handle_time"), ("b2b-3", "case_routing")]


def make_fake_tasks() -> list[CRMTask]:
    return [CRMTask(task_id=tid, idx=int(tid.split("-")[1]), category=cat, query=f"[{tid}] fake {cat} question?",
                    answer=[f"A{tid.split('-')[1]}"], reward_metric="exact_match", org_type="b2b")
            for tid, cat in FAKE_SPEC]


def fake_log_rows(tasks: list[CRMTask]) -> list[dict]:
    """cheap fails everywhere; mid solves both handle_time tasks; frontier solves b2b-1 only."""
    wins = {("b2b-1", "mid"), ("b2b-2", "mid"), ("b2b-1", "frontier")}
    rows = []
    for t in tasks:
        for arm, usd in (("cheap", 0.05), ("mid", 0.10), ("frontier", 0.15)):
            q = 1.0 if (t.task_id, arm) in wins else 0.0
            ans = t.answer[0] if q else "wrong"
            rows.append({"task_id": t.task_id, "steps": [], "quality": q, "total_cost": usd, "total_latency": 1.0,
                         "meta": {"category": t.category, "arm": arm, "answer": ans, "end_reason": "agent_respond",
                                  "queries": [f"SELECT Id, OwnerId FROM Case WHERE Subject = '{t.task_id}'", ans]}})
    return rows


def write_fake_log(tasks, path) -> None:
    with open(path, "w") as f:
        for r in fake_log_rows(tasks):
            f.write(json.dumps(r) + "\n")


def _resp(text: str, pin: int = 400, pout: int = 60) -> dict:
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": pin, "completion_tokens": pout}}


class FakeCompletion:
    def __init__(self, tasks: list[CRMTask]):
        self.gold = {t.task_id: t.answer[0] for t in tasks}
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        sys_ = msgs[0].get("content", "") if msgs and msgs[0]["role"] == "system" else ""
        if sys_.startswith(WRITER_SYSTEM):
            cat = re.search(r"Task category: (\S+)", msgs[1]["content"]).group(1)
            return _resp(json.dumps({"notes": [f"For {cat} questions: query Case first, group by OwnerId.",
                                               "Filter on ClosedDate within the window; ignore Id 005Wt000003NFKoIAO."]}))
        tid = re.search(r"\[(b2b-\d+)\]", msgs[1]["content"]).group(1)
        if not any(m["role"] == "assistant" for m in msgs):
            return _resp("<execute>SELECT Id FROM Case</execute>")
        return _resp(f"<respond>{self.gold[tid] if HINT_HEADER in sys_ else 'wrong'}</respond>")


class _ExactEval:
    def evaluate(self, proposed, gold, metric, task_name, actions):
        r = int(str(proposed).strip() == str(gold[0]))
        return {"reward": r, "parsed_answer": proposed}


class FakeBackend:
    rule_string = "Use <execute> or <respond>."

    def __init__(self, tasks):
        self.tasks = tasks

    def load_tasks(self):
        return list(self.tasks)

    def system_prompt(self, task, eval_mode="aided", privacy_aware=False):
        return "You are a CRM agent. Schema: Case(Id, OwnerId, ClosedDate)."

    def run_query(self, q):
        return [{"Id": "x"}], 1

    def external_facing(self):
        return set()

    def make_evaluator(self, shim):
        return _ExactEval()
