"""Deterministic fake backend for `--dry-run`: no API, no Salesforce, no spend.

The fake is a small world with hidden truth, built so that the pipeline exercises every real
code path (harness.step for the workers, the ledger and its cap, CRMArena-style scoring) and so
that the analysis tables render with plausible structure. The structure is chosen, not measured:

  * an initial answer is correct with a per-(category, tier) probability
  * the decomposer's self-reported confidence is informative but noisy: claims of a correct answer
    draw confidence from U(0.40, 0.95), the false claim of a wrong answer from U(0.15, 0.75), so the
    min-over-claims signal separates wrong from correct initial answers imperfectly (AUROC well
    below 1)
  * a worker re-solves a sub-claim correctly with p = 0.65 on cheap and 0.85 on mid
  * the rewrite fixes a wrong answer only when the workers exposed the false claim, and it can
    flip a correct answer when the workers were wrong (more often on cheap, and much more often
    on the three confidentiality categories, where a "refinement" tends to talk the agent out of
    a correct refusal)
  * token counts per call are the CRMArena means measured in the real counterfactual run
    (cheap 9.1k in / 250 out per agent turn, mid 11.2k / 150), so dry-run dollars are in the
    right range for the cost estimate.

Nothing here is evidence about SSR on CRMArena. The real run replaces this file with
`CRMArenaBackend` and litellm; see README.md.
"""
from __future__ import annotations

import random
import re
from types import SimpleNamespace

from ...harness.crmarena import CATEGORIES, CONFIDENTIALITY_CATEGORIES, CRMTask, FORCE_ANSWER
from ...prices import MODELS

TIER_OF_MODEL = {v: k for k, v in MODELS.items()}
TOKENS = {  # (in, out) per call kind and tier
    "agent": {"cheap": (9100, 250), "mid": (11200, 150), "frontier": (10700, 175)},
    "worker": {"cheap": (6500, 120), "mid": (7800, 90), "frontier": (7800, 90)},
    "decompose": {"cheap": (1200, 220), "mid": (1200, 200), "frontier": (1200, 200)},
    "refine": {"cheap": (1800, 120), "mid": (1800, 90), "frontier": (1800, 90)},
}
P_INITIAL = {"cheap": 0.45, "mid": 0.60, "frontier": 0.66}
P_WORKER = {"cheap": 0.65, "mid": 0.85, "frontier": 0.92}
P_FIX = {"cheap": 0.70, "mid": 0.90, "frontier": 0.95}
P_FLIP = {"cheap": 0.35, "mid": 0.20, "frontier": 0.15}      # correct answer, contradicted by wrong workers


def make_fake_tasks(n: int, seed: int = 0) -> list[CRMTask]:
    out = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        gold = "REFUSE" if cat in CONFIDENTIALITY_CATEGORIES else f"005FAKE{i:04d}"
        out.append(CRMTask(task_id=f"fake-{i}", idx=i, category=cat, query=f"Fake question {i} ({cat})?",
                           answer=[gold], reward_metric="exact_match", metadata={}, org_type="b2b"))
    return out


def _resp(text: str, pin: int, pout: int):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout))


class FakeWorld:
    """Scripted litellm.completion with hidden truth. `__call__(**kw)` is the completion function."""

    def __init__(self, tasks: list[CRMTask], seed: int = 0):
        self.tasks = {t.task_id: t for t in tasks}
        self.seed = seed
        self.calls: list[dict] = []

    # ---------------------------------------------------------- helpers
    def rng(self, *key) -> random.Random:
        return random.Random("|".join(str(k) for k in (self.seed,) + key))

    def gold(self, task_id: str) -> str:
        return self.tasks[task_id].answer[0]

    def true_value(self, task_id: str, i: int) -> str:
        return f"T{task_id}-{i}"

    def p_initial(self, task: CRMTask, tier: str) -> float:
        off = (self.rng("cat", task.category).random() - 0.5) * 0.3
        return max(0.1, min(0.9, P_INITIAL[tier] + off))

    def _tokens(self, kind: str, tier: str, key) -> tuple[int, int]:
        pin, pout = TOKENS[kind][tier]
        r = self.rng("tok", kind, tier, key)
        return int(pin * (0.8 + 0.4 * r.random())), int(pout * (0.8 + 0.4 * r.random()))

    @staticmethod
    def _task_id(messages: list[dict]) -> str:
        for m in messages:
            mm = re.search(r"task=(\S+)", m.get("content") or "") or re.search(r"^Task: (\S+)", m.get("content") or "", re.M)
            if mm:
                return mm.group(1)
        raise ValueError("fake: no task id in prompt")

    # ------------------------------------------------------------- call
    def __call__(self, **kw):
        self.calls.append(kw)
        msgs = kw["messages"]
        tier = TIER_OF_MODEL[kw["model"].split("/", 1)[-1]]
        sys_text = msgs[0]["content"]
        if "DECOMPOSE" in sys_text or "decompose a proposed answer" in sys_text:
            return self.decompose(msgs, tier)
        if "finalize answers" in sys_text:
            return self.refine(msgs, tier)
        user0 = msgs[1]["content"]
        if "[ssr " in user0:
            return self.worker(msgs, tier)
        return self.agent(msgs, tier)

    # ----------------------------------------------------- main episode
    def agent(self, msgs, tier):
        tid = self._task_id(msgs)
        task = self.tasks[tid]
        n_asst = sum(m["role"] == "assistant" for m in msgs)
        pin, pout = self._tokens("agent", tier, (tid, n_asst))
        correct = self.rng("init", tid, tier).random() < self.p_initial(task, tier)
        answer = self.gold(tid) if correct else f"005WRONG{self.rng('wrong', tid, tier).randrange(10000):04d}"
        if FORCE_ANSWER.strip() in msgs[-1]["content"] or n_asst >= 2:
            return _resp(f"<thought> done </thought> <respond> {answer} </respond>", pin, pout)
        return _resp(f"<thought> look </thought> <execute> SELECT Id FROM Case LIMIT {n_asst + 1} </execute>", pin, pout)

    # ------------------------------------------------------------ probe
    def decompose(self, msgs, tier):
        user = msgs[1]["content"]
        tid = self._task_id(msgs)
        answer = re.search(r"Proposed answer:\n(.*?)\n\nDecompose", user, re.S).group(1).strip()
        correct = answer == self.gold(tid)
        r = self.rng("decomp", tid, answer)
        n = 2 + (r.random() < 0.4)
        false_i = r.randrange(n)
        claims = []
        for i in range(n):
            is_true = correct or i != false_i
            if correct:
                conf = 0.40 + 0.55 * r.random()               # true claim of a correct answer
            elif is_true:
                conf = 0.30 + 0.60 * r.random()               # true claim of a wrong answer: shakier
            else:
                conf = 0.15 + 0.60 * r.random()               # the false claim: lowest, but overlapping
            expected = self.true_value(tid, i) if is_true else f"X{tid}-{i}"
            claims.append({"claim": f"Field F{i} of the answer equals {expected}",
                           "question": f"[ssr {tid} c{i} a={answer}] What is the value of field F{i}?",
                           "expected": expected, "confidence": round(conf, 3)})
        aconf = round(min(c["confidence"] for c in claims) * (0.9 + 0.2 * r.random()), 3)
        import json
        pin, pout = self._tokens("decompose", tier, (tid, answer))
        return _resp(json.dumps({"claims": claims, "answer_confidence": min(1.0, aconf)}), pin, pout)

    # ----------------------------------------------------------- worker
    def worker(self, msgs, tier):
        user0 = msgs[1]["content"]
        m = re.search(r"\[ssr (\S+) c(\d+) a=(\S+)\]", user0)
        tid, i, answer = m.group(1), int(m.group(2)), m.group(3)
        n_asst = sum(mm["role"] == "assistant" for mm in msgs)
        # sample index: how many identical worker prompts have been seen so far for this (tid, i, answer, tier)
        key = (tid, i, answer, tier)
        seen = sum(1 for c in self.calls[:-1] if c["messages"][1]["content"] == user0
                   and TIER_OF_MODEL[c["model"].split("/", 1)[-1]] == tier
                   and sum(mm["role"] == "assistant" for mm in c["messages"]) == 0)
        pin, pout = self._tokens("worker", tier, key + (seen, n_asst))
        if n_asst == 0 and FORCE_ANSWER.strip() not in msgs[-1]["content"]:
            return _resp(f"<execute> SELECT F{i} FROM Record WHERE Id = '{answer}' </execute>", pin, pout)
        r = self.rng("worker", key, seen)
        val = self.true_value(tid, i) if r.random() < P_WORKER[tier] else f"W{tid}-{i}-{r.randrange(3)}"
        return _resp(f"<respond> {val} </respond>", pin, pout)

    # ----------------------------------------------------------- refine
    def refine(self, msgs, tier):
        user = msgs[1]["content"]
        tid = self._task_id(msgs)
        task = self.tasks[tid]
        answer = re.search(r"Proposed answer:\n(.*?)\n\nSub-claims", user, re.S).group(1).strip()
        correct = answer == self.gold(tid)
        blocks = re.findall(r"asserts: (.*?)\n.*?findings: (.*?)\n", user, re.S)
        exposed_false = any(a.startswith("X") and f"T{tid}" in f for a, f in blocks)
        contradicted_true = any(a.startswith("T") and a not in f for a, f in blocks)
        r = self.rng("refine", tid, answer, tier)
        pin, pout = self._tokens("refine", tier, (tid, answer))
        conf_cat = task.category in CONFIDENTIALITY_CATEGORIES
        if correct:
            flip = P_FLIP[tier] * (1.8 if conf_cat else 1.0)
            if contradicted_true and r.random() < flip:
                return _resp(f"<respond> 005FLIP{self.rng('flip', tid, tier, answer).randrange(10000):04d} </respond>", pin, pout)
            return _resp(f"<respond> {answer} </respond>", pin, pout)
        if exposed_false and r.random() < P_FIX[tier] * (0.5 if conf_cat else 1.0):
            return _resp(f"<respond> {self.gold(tid)} </respond>", pin, pout)
        return _resp(f"<respond> {answer} </respond>", pin, pout)


class FakeBackend:
    rule_string = "Invalid output format! Use <execute> or <respond>."

    def __init__(self):
        self.queries: list[str] = []

    def system_prompt(self, task, eval_mode="aided", privacy_aware=False):
        return f"SYS fake task={task.task_id} cat={task.category}"

    def run_query(self, q):
        self.queries.append(q)
        return [{"Id": "500FAKE"}], 1

    def make_evaluator(self, shim):
        return _ExactEval()

    def load_tasks(self):
        raise RuntimeError("fake backend: pass tasks explicitly")


class _ExactEval:
    def evaluate(self, proposed, gold, metric, task_name, traj):
        ok = str(proposed).strip().strip("<>").strip() == str(gold[0])
        return {"reward": 1 if ok else 0, "parsed_answer": [proposed]}
