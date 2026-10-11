"""Minimal ReAct QA harness over Anthropic models used as cost tiers.

Same action space as the sim: (op, tier) with ops ANSWER / RETRIEVE / VERIFY /
ESCALATE and tiers cheap / mid / frontier mapped to real models. Cost is
computed from the API's reported token usage x list price. Quality is
normalized exact / contains match against a gold answer, with an optional
LLM-judge fallback (judge spend is tracked separately, not charged to the
controller).

Search is pluggable via env var: EXA_API_KEY (Exa) or TAVILY_API_KEY
(Tavily); with neither set, RETRIEVE uses a stub that returns no evidence.

The `anthropic` SDK is imported lazily inside the client accessor, so this
module imports cleanly without the SDK or an API key. It has never been run against live
APIs in this repository; the tests only check that it imports and mirrors the sim interface.

Usage sketch:
    from pace.harness.react_api import ReactAPIHarness, load_tasks
    h = ReactAPIHarness(load_tasks("data/qa.jsonl"))
    episodes = pace.collect.collect(h, policy, n=500)   # sample_tasks yields the loaded tasks
"""
from __future__ import annotations

import json
import os
import re
import string
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .base import Harness, HarnessState

from ..prices import MODELS, PRICES_PER_MTOK, TIERS, call_cost  # single price table, shared with crmarena
OPS = ["ANSWER", "RETRIEVE", "VERIFY", "ESCALATE"]
TERMINAL_OPS = {"ANSWER", "ESCALATE"}
MAX_STEPS = 4
ACTIONS = [f"{o}@{t}" for o in OPS for t in TIERS]
# Tiers whose models accept output_config.effort and server-side refusal fallbacks.
_EFFORT_TIERS = {"mid", "frontier"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class QATask:
    task_id: str
    question: str
    gold: str
    difficulty: str | None = None      # optional label; else a heuristic proxy is used


def load_tasks(path: str | Path) -> list[QATask]:
    """JSONL with fields id, question, answer[, difficulty]."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(QATask(str(d["id"]), d["question"], str(d["answer"]), d.get("difficulty")))
    return out


# --------------------------------------------------------------- search tool
class SearchTool:
    def __init__(self, k: int = 3, max_chars: int = 800):
        self.k, self.max_chars = k, max_chars
        if os.environ.get("EXA_API_KEY"):
            self.provider = "exa"
        elif os.environ.get("TAVILY_API_KEY"):
            self.provider = "tavily"
        else:
            self.provider = "stub"

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def search(self, query: str) -> list[str]:
        if self.provider == "exa":
            d = self._post("https://api.exa.ai/search",
                           {"query": query, "numResults": self.k,
                            "contents": {"text": {"maxCharacters": self.max_chars}}},
                           {"x-api-key": os.environ["EXA_API_KEY"]})
            return [f"{r.get('title', '')}: {(r.get('text') or '')[:self.max_chars]}" for r in d.get("results", [])]
        if self.provider == "tavily":
            d = self._post("https://api.tavily.com/search",
                           {"api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": self.k}, {})
            return [f"{r.get('title', '')}: {(r.get('content') or '')[:self.max_chars]}" for r in d.get("results", [])]
        return []


# ------------------------------------------------------------------ scoring
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def match(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return 0.0
    return 1.0 if (p == g or g in p) else 0.0


# ------------------------------------------------------------------ harness
class ReactAPIHarness(Harness):
    version = "react-v1"

    def __init__(self, tasks: list[QATask], search: SearchTool | None = None, effort: str = "low",
                 use_judge: bool = False, judge_tier: str = "cheap", max_tokens: int = 2048, client=None):
        self.tasks = list(tasks)
        self.search_tool = search or SearchTool()
        self.effort = effort
        self.use_judge = use_judge
        self.judge_tier = judge_tier
        self.max_tokens = max_tokens
        self._client = client
        self.judge_cost = 0.0

    # -------------------------------------------------------------- client
    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when the harness actually runs
            self._client = anthropic.Anthropic()
        return self._client

    def _call(self, tier: str, system: str, user: str, effort: str | None = None,
              max_tokens: int | None = None) -> tuple[str, float, float]:
        """One model call. Returns (text, cost_usd, latency_s)."""
        model = MODELS[tier]
        kwargs = dict(model=model, max_tokens=max_tokens or self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        t0 = time.time()
        if tier in _EFFORT_TIERS:
            kwargs["output_config"] = {"effort": effort or self.effort}
            resp = self.client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        latency = time.time() - t0
        served = getattr(resp, "model", model)
        u = resp.usage
        cost = call_cost(served if served in PRICES_PER_MTOK else model, u.input_tokens, u.output_tokens,
                         getattr(u, "cache_creation_input_tokens", 0) or 0,
                         getattr(u, "cache_read_input_tokens", 0) or 0)
        if resp.stop_reason == "refusal":
            return "", cost, latency
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        return text.strip(), cost, latency

    # ----------------------------------------------------------- interface
    @property
    def all_actions(self) -> list[str]:
        return list(ACTIONS)

    def sample_tasks(self, n: int, seed: int = 0):
        return iter(self.tasks[:n])

    @staticmethod
    def difficulty_proxy(task: QATask) -> str:
        if task.difficulty in ("easy", "hard"):
            return task.difficulty
        q = task.question
        multi_hop = len(re.findall(r"\b(and|of the|whose|which|before|after)\b", q.lower()))
        return "hard" if (len(q.split()) > 25 or multi_hop >= 2) else "easy"

    def reset(self, task: QATask) -> HarnessState:
        return HarnessState(task_id=task.task_id, difficulty_proxy=self.difficulty_proxy(task),
                            version=self.version,
                            extra={"question": task.question, "gold": task.gold, "evidence": [],
                                   "draft": None, "answer": None})

    def legal_actions(self, state: HarnessState) -> list[str]:
        if state.done:
            return []
        if state.step >= MAX_STEPS - 1:
            return [a for a in ACTIONS if a.split("@")[0] in TERMINAL_OPS]
        return list(ACTIONS)

    def _evidence_block(self, s: HarnessState) -> str:
        ev = s.extra["evidence"]
        return "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(ev)) if ev else "(no evidence retrieved)"

    def step(self, state: HarnessState, action: str):
        s = state.clone()
        s.extra["evidence"] = list(s.extra["evidence"])
        op, tier = action.split("@")
        q = s.extra["question"]
        if op == "RETRIEVE":
            query, cost, lat = self._call(tier, "Write one concise web search query. Output only the query.",
                                          f"Question: {q}\nEvidence so far:\n{self._evidence_block(s)}",
                                          max_tokens=256)
            t0 = time.time()
            hits = self.search_tool.search(query or q)
            lat += time.time() - t0
            if hits:
                s.extra["evidence"].extend(hits)
                s.evidence = min(s.evidence + 1, 2)
        elif op == "VERIFY":
            out, cost, lat = self._call(
                tier, "Draft a short answer from the evidence, then check it. Reply in two lines:\n"
                      "ANSWER: <answer>\nSUPPORTED: yes|no",
                f"Question: {q}\nEvidence:\n{self._evidence_block(s)}")
            m = re.search(r"ANSWER:\s*(.+)", out)
            s.extra["draft"] = m.group(1).strip() if m else out
            s.verified = bool(re.search(r"SUPPORTED:\s*yes", out, re.I))
        elif op in TERMINAL_OPS:
            hint = f"\nA checked draft answer: {s.extra['draft']}" if s.verified and s.extra["draft"] else ""
            system = "Answer the question with a short final answer only (an entity, number, or phrase)."
            if op == "ESCALATE":
                system += " Reason carefully before answering; output only the final answer."
            out, cost, lat = self._call(tier, system,
                                        f"Question: {q}\nEvidence:\n{self._evidence_block(s)}{hint}",
                                        effort=("high" if op == "ESCALATE" else None))
            s.extra["answer"] = out.splitlines()[-1].strip() if out else ""
            s.done = True
        else:
            raise ValueError(action)
        s.step += 1
        s.cost_so_far += cost
        s.latency_so_far += lat
        return s, s.done, cost, lat

    def quality(self, state: HarnessState) -> float:
        pred, gold = state.extra.get("answer") or "", state.extra["gold"]
        m = match(pred, gold)
        if m or not self.use_judge or not pred:
            return m
        out, cost, _ = self._call(self.judge_tier, "You grade answers. Reply with exactly 'yes' or 'no'.",
                                  f"Question: {state.extra['question']}\nGold answer: {gold}\n"
                                  f"Candidate answer: {pred}\nIs the candidate correct (equivalent to gold)?",
                                  max_tokens=16)
        self.judge_cost += cost
        return 1.0 if out.lower().startswith("yes") else 0.0

    def snapshot(self, state: HarnessState) -> dict:
        return self.state_to_dict(state)     # logged for audit; API calls cannot be re-rolled deterministically
