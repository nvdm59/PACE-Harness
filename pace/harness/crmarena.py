"""CRMArena-Pro as a PACE host harness (version "crmarena-pro-v1").

CRMArena-Pro (Huang et al. 2025, arXiv:2505.18878; CRMArena, Huang et al. 2024,
arXiv:2411.02305; Salesforce AI Research, CC BY-NC 4.0, research use) is vendored
as an external dependency by `scripts/setup_crmarena.sh` into
`third_party/CRMArena`; nothing from it is copied into this repository.

What CRMArena's own loop looks like
-----------------------------------
For the CRMArena-Pro orgs (b2b / b2c) only the ReAct `ChatAgent` is supported
(`tool_call` asserts org_type == "original"). `ChatAgent.act(env, idx)` is a
single `while turn < max_turns` loop that calls `litellm.completion` with one
fixed model, parses `<execute>SOQL/SOSL</execute>` or `<respond>answer</respond>`
from the reply, runs the query via `ChatEnv.step -> SalesforceConnector.run_query`,
pastes `"Salesforce instance output: {rows}"` back as a user turn, and on
`<respond>` scores the answer with `Evaluator.evaluate`. There is no per-turn
hook: the model is fixed for the whole episode and the agent alone decides
when to answer.

What this module does instead
-----------------------------
It re-runs that exact loop turn by turn, so PACE can choose (op, tier) at
every turn. It REUSES CRMArena's pieces unchanged: the task records and
schemas (`crm_sandbox.data.assets`), the ReAct system prompts and rule string
(`crm_sandbox.agents.prompts`), the schema rendering (`ChatAgent._build_schema`),
the internal/external agent split (`EXTERNAL_FACING_TASKS`), the Salesforce
connector (`SalesforceConnector.run_query`) and the scorer (`Evaluator.evaluate`).
It does NOT fork their agent; only the ~30-line turn loop is re-expressed as
`step()`. The action parser uses the same two regexes as
`ChatAgent.message_action_parser`.

Actions: {CONTINUE, ANSWER} x {cheap, mid, frontier}.
  CONTINUE@t  one normal ReAct turn on tier t. If the model emits <respond>,
              the episode ends with that answer (the agent chose to answer).
  ANSWER@t    force the final answer now on tier t (an instruction is appended
              to the last user turn; the reply's <respond> body, or the whole
              reply if untagged, is the answer).
On the last allowed turn (max_turns - 1) only ANSWER@* is legal.

Deviations from CRMArena's run_tasks.py, all deliberate:
  * no `temperature` / `top_p`: Sonnet 5.5 rejects non-default sampling values
    and Opus 5.5 rejects them outright (HTTP 400); ChatAgent always sends
    temperature=0.0, top_p=1.0.
  * `max_tokens` is bounded per call (default 2048 agent turn, 1024 forced
    answer, 256 extractor) and every call has a timeout; ChatAgent uses 2000
    and the Evaluator sends none.
  * Salesforce observations are truncated to `obs_max_chars` (default 20,000;
    0 disables). CRMArena pastes raw query results, which in their published
    GPT-4o b2b logs reach ~500k prompt tokens in one episode.
  * the answer extractor / privacy judge runs on the cheap tier through the
    same priced, capped call path (CRMArena hard-codes gpt-4o via OpenAI).
  * the missing `import re` in crm_sandbox/env/env.py (used by the extractor's
    fallback heuristics) is supplied by monkeypatch.
  * non-interactive mode only (the interactive mode needs an LLM user simulator).

All LLM calls go through `PricedLLM`, which checks the `SpendLedger` cap before
the call and records real token usage after it.
"""
from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from ..prices import MODELS, TIERS, bare_model, model_for
from ..spend import SpendLedger, estimate_input_tokens
from .base import Harness, HarnessState

VERSION = "crmarena-pro-v1"
OPS = ["CONTINUE", "ANSWER"]
ACTIONS = [f"{o}@{t}" for o in OPS for t in TIERS]
MAX_TURNS = 12

# CRMArena-Pro task categories: 19 business skills + 3 confidentiality-awareness.
BUSINESS_CATEGORIES = [
    "handle_time", "transfer_count", "knowledge_qa", "best_region_identification", "case_routing",
    "named_entity_disambiguation", "policy_violation_identification", "top_issue_identification",
    "monthly_trend_analysis", "sales_amount_understanding", "lead_routing", "sales_cycle_understanding",
    "conversion_rate_comprehension", "wrong_stage_rectification", "sales_insight_mining",
    "quote_approval", "lead_qualification", "activity_priority", "invalid_config",
]
CONFIDENTIALITY_CATEGORIES = ["private_customer_information", "internal_operation_data",
                              "confidential_company_knowledge"]
CATEGORIES = BUSINESS_CATEGORIES + CONFIDENTIALITY_CATEGORIES
ORG_TYPES = ["b2b", "b2c"]
# token buckets of the approximate context size (chars / 4)
CONTEXT_BUCKETS = [4000, 8000, 16000, 32000]

FORCE_ANSWER = ("\n\nYou must give your final answer now, without running more queries. "
                "Output only <respond> your final answer </respond>.")

FEATURE_NAMES = (["bias"] + [f"cat={c}" for c in CATEGORIES] + [f"org={o}" for o in ORG_TYPES]
                 + [f"turn={i}" for i in range(6)] + ["turn>=6", "turn_frac", "tool_calls/10", "cost_so_far",
                                                      "last_rows", "last_empty", "last_error"]
                 + [f"ctx<{b // 1000}k" for b in CONTEXT_BUCKETS] + [f"ctx>={CONTEXT_BUCKETS[-1] // 1000}k"])
N_FEATURES = len(FEATURE_NAMES)
TURN0_IDX = FEATURE_NAMES.index("turn=0")


@dataclass
class CRMTask:
    task_id: str
    idx: int
    category: str
    query: str
    answer: list
    reward_metric: str
    metadata: dict = field(default_factory=dict)
    org_type: str = "b2b"

    @classmethod
    def from_record(cls, r: dict, org_type: str) -> "CRMTask":
        return cls(task_id=f"{org_type}-{r['idx']}", idx=int(r["idx"]), category=r["task"], query=r["query"],
                   answer=list(r["answer"]) if r.get("answer") is not None else [None],
                   reward_metric=r["reward_metric"], metadata=dict(r.get("metadata") or {}), org_type=org_type)


def parse_action(content: str) -> dict | None:
    """Same regexes and precedence as CRMArena's ChatAgent.message_action_parser."""
    m = re.search(r"<execute>(.*?)</execute>", content, re.DOTALL)
    if m and m.group(1).strip():
        return {"name": "execute", "content": m.group(1).strip()}
    m = re.search(r"<respond>(.*?)</respond>", content, re.DOTALL)
    if m and m.group(1).strip():
        return {"name": "respond", "content": m.group(1).strip()}
    return None


# ---------------------------------------------------------------- LLM calls
class LLMCallFailed(RuntimeError):
    """A model call still failed after litellm's retries. Collection aborts (exit 4) rather than
    letting a persistent error (bad key, rejected parameter, outage) burn turns and budget."""


@dataclass
class LLMResult:
    text: str
    usd: float
    latency: float
    in_tokens: int
    out_tokens: int
    model: str


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class PricedLLM:
    """litellm (Anthropic provider) behind the spend ledger.

    Before each call: worst case = estimated input tokens + max_tokens, priced;
    `SpendCapReached` if it would cross the cap. After: real usage is recorded.
    A call rejected with HTTP 4xx is recorded at $0 (rejected requests are not
    billed); any other failure (timeout, 5xx, network) is recorded at its worst
    case because it may have been billed. Either way `LLMCallFailed` is raised.
    """

    def __init__(self, ledger: SpendLedger, completion_fn: Callable | None = None, timeout: float = 120.0,
                 num_retries: int = 2, effort: str | None = "low", provider_prefix: str = "anthropic/",
                 prompt_cache: bool | None = None):
        self.ledger = ledger
        self._completion = completion_fn
        self.timeout, self.num_retries = timeout, num_retries
        self.effort = effort
        self.prefix = provider_prefix
        # Anthropic prompt caching: opt-in. None -> $PACE_PROMPT_CACHE ("1" on); the CRM collectors
        # leave it off so their accounting stays comparable with earlier tasks.
        if prompt_cache is None:
            prompt_cache = os.environ.get("PACE_PROMPT_CACHE", "0") == "1"
        self.prompt_cache = bool(prompt_cache)

    @property
    def completion(self) -> Callable:
        if self._completion is None:
            import litellm  # lazy: only for real runs
            litellm.drop_params = True
            self._completion = litellm.completion
        return self._completion

    def complete(self, tier_or_model: str, messages: list[dict], max_tokens: int, task_id: str | None,
                 purpose: str) -> LLMResult:
        model = bare_model(model_for(tier_or_model))
        self.ledger.guard(model, estimate_input_tokens(messages), max_tokens, purpose)
        sent = with_cache_breakpoints(messages) if self.prompt_cache else list(messages)
        kwargs: dict[str, Any] = dict(model=self.prefix + model, messages=sent, max_tokens=int(max_tokens),
                                      timeout=self.timeout, num_retries=self.num_retries)
        # effort applies to the thinking models (Sonnet 5.5 / Opus 5.5); Haiku 4.5 rejects it.
        if self.effort and model != MODELS["cheap"]:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.time()
        try:
            res = self.completion(**kwargs)
        except Exception as e:  # noqa: BLE001 -- any provider/network failure
            status = getattr(e, "status_code", None)
            rejected = isinstance(status, int) and 400 <= status < 500
            wc = 0.0 if rejected else self.ledger.worst_case(model, estimate_input_tokens(messages), max_tokens)
            self.ledger.record(model, 0, 0, task_id, purpose + ":error", usd=wc,
                               extra={"error": f"{type(e).__name__}: {str(e)[:300]}", "status": status,
                                      "charged": "zero:rejected_4xx" if rejected else "worst_case"})
            raise LLMCallFailed(f"{model} {purpose} failed (status {status}): {type(e).__name__}: {e}") from e
        lat = time.time() - t0
        usage = _get(res, "usage")
        in_tok, out_tok = _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")
        if in_tok is None or out_tok is None:
            wc = self.ledger.worst_case(model, estimate_input_tokens(messages), max_tokens)
            usd = self.ledger.record(model, 0, 0, task_id, purpose, usd=wc, extra={"charged": "worst_case:no_usage"})
            in_tok = out_tok = 0
        else:
            cw, cr = cache_tokens(usage)
            if cw or cr:
                # litellm (1.67, Anthropic) reports prompt_tokens = input_tokens + cache_read; Anthropic's
                # input_tokens already excludes both cache classes, which call_cost prices separately.
                in_tok = max(int(in_tok) - cr, 0)
                usd = self.ledger.record(model, in_tok, int(out_tok), task_id, purpose,
                                         cache_write_tokens=cw, cache_read_tokens=cr)
            else:
                usd = self.ledger.record(model, int(in_tok), int(out_tok), task_id, purpose)
        choices = _get(res, "choices") or []
        msg = _get(choices[0], "message") if choices else None
        text = _get(msg, "content") or ""
        if isinstance(text, list):
            text = "".join(_get(b, "text", "") or "" for b in text)
        return LLMResult(str(text), usd, lat, int(in_tok), int(out_tok), model)


EPHEMERAL = {"type": "ephemeral"}


def _as_blocks(content: Any) -> list[dict]:
    if isinstance(content, list):
        return [dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]
    return [{"type": "text", "text": str(content or "")}]


def with_cache_breakpoints(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with Anthropic `cache_control` breakpoints (litellm content-block syntax)
    on the system prompt (the fixed prefix: policy, protocol, tool schemas) and on the last
    message, so each turn reads the previous turns' prefix from the cache. Empty texts are
    left unmarked (Anthropic rejects cache_control on empty blocks)."""
    out = [dict(m) for m in messages]
    idx = [i for i, m in enumerate(out) if m.get("role") == "system"][:1]
    if out and out[-1].get("role") != "system":
        idx.append(len(out) - 1)
    for i in idx:
        blocks = _as_blocks(out[i].get("content"))
        if blocks and blocks[-1].get("text"):
            blocks[-1] = dict(blocks[-1], cache_control=dict(EPHEMERAL))
            out[i]["content"] = blocks
    return out


def cache_tokens(usage: Any) -> tuple[int, int]:
    """(cache_write, cache_read) input tokens from a litellm usage object (0 when absent)."""
    cw = _get(usage, "cache_creation_input_tokens") or 0
    cr = _get(usage, "cache_read_input_tokens")
    if cr is None:
        cr = _get(_get(usage, "prompt_tokens_details"), "cached_tokens") or 0
    try:
        return int(cw), int(cr)
    except (TypeError, ValueError):
        return 0, 0


class _ExtractorLiteLLM:
    """Stands in for the `litellm` module inside crm_sandbox/env/env.py so the
    Evaluator's extractor / privacy-judge calls run on the cheap tier, bounded,
    priced and capped. Exposes only `completion`, which is all env.py uses."""

    def __init__(self, llm: PricedLLM, max_tokens: int = 256):
        self.llm, self.max_tokens = llm, max_tokens
        self.task_id: str | None = None
        self.purpose = "extractor"
        self.spent = 0.0
        self.calls = 0

    def completion(self, model=None, messages=None, custom_llm_provider=None, **_ignored):
        r = self.llm.complete("cheap", list(messages or []), self.max_tokens, self.task_id, self.purpose)
        self.spent += r.usd
        self.calls += 1
        msg = SimpleNamespace(content=r.text, model_dump=lambda: {"role": "assistant", "content": r.text})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None,
                               _hidden_params={"response_cost": r.usd})


# ------------------------------------------------------------------ backend
class CRMArenaBackend:
    """The real CRMArena pieces, imported lazily from the vendored package."""

    def __init__(self, org_type: str = "b2b"):
        assert org_type in ORG_TYPES, org_type
        self.org_type = org_type
        self._connector = None
        self._prompts = None
        self._schema_str = None

    # task data + schema (Hugging Face: Salesforce/CRMArenaPro, public)
    def load_tasks(self) -> list[CRMTask]:
        from crm_sandbox.data import assets
        recs = assets.TASKS_B2B if self.org_type == "b2b" else assets.TASKS_B2C
        return [CRMTask.from_record(r, self.org_type) for r in recs]

    def _schema(self) -> str:
        if self._schema_str is None:
            from crm_sandbox.agents.chat_agent import ChatAgent
            from crm_sandbox.data import assets
            schema = assets.B2B_SCHEMA if self.org_type == "b2b" else assets.B2C_SCHEMA
            self._schema_str = ChatAgent._build_schema(None, schema)   # uses only its argument
        return self._schema_str

    @property
    def prompts(self):
        if self._prompts is None:
            from crm_sandbox.agents import prompts
            self._prompts = prompts
        return self._prompts

    @property
    def rule_string(self) -> str:
        return self.prompts.REACT_RULE_STRING

    def external_facing(self) -> set[str]:
        from crm_sandbox.data.assets import EXTERNAL_FACING_TASKS
        return set(EXTERNAL_FACING_TASKS)

    def system_prompt(self, task: CRMTask, eval_mode: str = "aided", privacy_aware: bool = False) -> str:
        """Mirrors ChatAgent.__init__ + ChatAgent.reset for non-interactive react."""
        P = self.prompts
        if task.category in self.external_facing():
            tmpl = P.REACT_PRIVACY_AWARE_EXTERNAL_PROMPT if privacy_aware else P.REACT_EXTERNAL_PROMPT
        else:
            tmpl = P.REACT_INTERNAL_PROMPT
        sys_prompt = tmpl.format(system_description=self._schema(), system="Salesforce instance")
        md = task.metadata or {}
        if md.get("required"):
            sys_prompt += P.SYSTEM_METADATA.format(system_metadata=md["required"], system="Salesforce instance")
        if eval_mode == "aided" and md.get("optional"):
            sys_prompt += "\n" + md["optional"]
        return sys_prompt.strip()

    def run_query(self, query: str) -> tuple[Any, int]:
        if self._connector is None:
            from crm_sandbox.env.connect_sandbox import SalesforceConnector
            self._connector = SalesforceConnector(org_type=self.org_type)
        try:
            return self._connector.run_query(query)
        except Exception as e:  # noqa: BLE001 -- run_query's own error parsing can raise on non-SF errors
            return f"{type(e).__name__}: {e}", 0

    def make_evaluator(self, shim: _ExtractorLiteLLM):
        import crm_sandbox.env.env as envmod
        return build_evaluator(envmod, shim)


def build_evaluator(envmod, shim: _ExtractorLiteLLM):
    """Patch CRMArena's env module (crm_sandbox/env/env.py) and build its Evaluator."""
    envmod.litellm = shim          # route extractor / privacy judge through the priced cheap tier
    envmod.re = re                 # env.py uses `re` in its fallback parser without importing it
    return envmod.Evaluator(model=MODELS["cheap"], provider="anthropic")


# ------------------------------------------------------------------ harness
class CRMArenaHarness(Harness):
    version = VERSION

    def __init__(self, org: str = "b2b", categories: list[str] | None = None, ledger: SpendLedger | None = None,
                 llm: PricedLLM | None = None, backend: Any = None, tasks: list[CRMTask] | None = None,
                 max_turns: int = MAX_TURNS, max_tokens: int = 2048, answer_max_tokens: int = 1024,
                 extractor_max_tokens: int = 256, obs_max_chars: int = 20000, eval_mode: str = "aided",
                 privacy_aware: bool = False):
        assert org in ORG_TYPES, org
        self.org = org
        self.categories = None if not categories else list(categories)
        bad = [c for c in (self.categories or []) if c not in CATEGORIES]
        if bad:
            raise ValueError(f"unknown CRMArena-Pro categories: {bad}")
        self.ledger = ledger or SpendLedger()
        self.llm = llm or PricedLLM(self.ledger)
        self.backend = backend or CRMArenaBackend(org)
        self._tasks = tasks
        self.max_turns = int(max_turns)
        self.max_tokens, self.answer_max_tokens = int(max_tokens), int(answer_max_tokens)
        self.obs_max_chars = int(obs_max_chars)
        self.eval_mode, self.privacy_aware = eval_mode, privacy_aware
        self._shim = _ExtractorLiteLLM(self.llm, extractor_max_tokens)
        self._evaluator = None

    # ------------------------------------------------------------ tasks
    @property
    def all_actions(self) -> list[str]:
        return list(ACTIONS)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def tasks(self) -> list[CRMTask]:
        if self._tasks is None:
            self._tasks = self.backend.load_tasks()
        ts = self._tasks
        if self.categories:
            ts = [t for t in ts if t.category in self.categories]
        return ts

    def sample_tasks(self, n: int, seed: int = 0) -> list[CRMTask]:
        """Deterministic, category-stratified sample: shuffle within each category
        (seeded), then round-robin across categories. The same (n, seed) always
        yields the same prefix, so resumed runs see the same task order."""
        by_cat: dict[str, list[CRMTask]] = {}
        for t in sorted(self.tasks(), key=lambda t: t.idx):
            by_cat.setdefault(t.category, []).append(t)
        rng = random.Random(seed)
        cats = [c for c in CATEGORIES if c in by_cat] + sorted(c for c in by_cat if c not in CATEGORIES)
        for c in cats:
            rng.shuffle(by_cat[c])
        out: list[CRMTask] = []
        i = 0
        while len(out) < n and any(i < len(by_cat[c]) for c in cats):
            for c in cats:
                if i < len(by_cat[c]) and len(out) < n:
                    out.append(by_cat[c][i])
            i += 1
        return out

    # ------------------------------------------------------------ state
    def reset(self, task: CRMTask) -> HarnessState:
        sys_prompt = self.backend.system_prompt(task, self.eval_mode, self.privacy_aware)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": task.query.strip()}]
        return HarnessState(task_id=task.task_id, version=self.version, extra={
            "task": task, "messages": messages, "actions": [], "tool_calls": 0, "invalid": 0,
            "last_obs": "none", "answer": None, "end_reason": None, "tiers": [], "ctx_chars": self._chars(messages),
            "obs_truncated": 0})

    @staticmethod
    def _chars(messages: list[dict]) -> int:
        return sum(len(m.get("content") or "") for m in messages)

    def legal_actions(self, state: HarnessState) -> list[str]:
        if state.done:
            return []
        if state.step >= self.max_turns - 1:
            return [a for a in ACTIONS if a.startswith("ANSWER@")]
        return list(ACTIONS)

    def featurize(self, state: HarnessState) -> list[float]:
        x = [0.0] * N_FEATURES
        x[0] = 1.0
        task: CRMTask = state.extra["task"]
        if task.category in CATEGORIES:
            x[1 + CATEGORIES.index(task.category)] = 1.0
        o = 1 + len(CATEGORIES)
        if task.org_type in ORG_TYPES:
            x[o + ORG_TYPES.index(task.org_type)] = 1.0
        o += len(ORG_TYPES)
        x[o + min(state.step, 6)] = 1.0
        o += 7
        x[o] = state.step / max(self.max_turns - 1, 1)
        x[o + 1] = state.extra["tool_calls"] / 10.0
        x[o + 2] = float(state.cost_so_far)
        o += 3
        last = state.extra["last_obs"]
        for j, k in enumerate(["rows", "empty", "error"]):
            x[o + j] = 1.0 if last == k else 0.0
        o += 3
        approx_tokens = state.extra["ctx_chars"] / 4.0
        b = sum(approx_tokens >= t for t in CONTEXT_BUCKETS)
        x[o + b] = 1.0
        return x

    # ------------------------------------------------------------- step
    def _observation(self, query: str) -> tuple[str, str]:
        result, status = self.backend.run_query(query)
        if status == 0:
            kind = "error"
        elif isinstance(result, (list, tuple)) and len(result) == 0:
            kind = "empty"
        else:
            kind = "rows"
        obs = str(result)
        if self.obs_max_chars and len(obs) > self.obs_max_chars:
            obs = obs[:self.obs_max_chars] + f" ... [truncated {len(obs) - self.obs_max_chars} chars]"
            kind += ":truncated"
        return f"Salesforce instance output: {obs}", kind

    def step(self, state: HarnessState, action: str):
        op, tier = action.split("@")
        if action not in ACTIONS:
            raise ValueError(action)
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action} at turn {state.step}")
        t0 = time.time()
        s = state.clone()
        msgs = list(s.extra["messages"])
        s.extra["actions"] = list(s.extra["actions"])
        s.extra["tiers"] = list(s.extra["tiers"]) + [tier]
        if op == "ANSWER":
            last = dict(msgs[-1])
            last["content"] = (last.get("content") or "") + FORCE_ANSWER
            r = self.llm.complete(tier, msgs[:-1] + [last], self.answer_max_tokens, s.task_id, f"agent:{tier}:answer")
            content = r.text.strip()
            msgs.append({"role": "assistant", "content": content or "(empty)"})
            act = parse_action(content)
            answer = act["content"] if act and act["name"] == "respond" else content
            s.extra["answer"] = answer or "None"
            s.extra["actions"].append(s.extra["answer"])
            s.extra["end_reason"] = "forced_answer"
            s.done = True
        else:
            r = self.llm.complete(tier, msgs, self.max_tokens, s.task_id, f"agent:{tier}")
            content = r.text.strip()
            msgs.append({"role": "assistant", "content": content or "(empty)"})
            act = parse_action(content)
            if act is None:
                msgs.append({"role": "user", "content": self.backend.rule_string})
                s.extra["invalid"] += 1
                s.extra["last_obs"] = "invalid"
            elif act["name"] == "execute":
                s.extra["actions"].append(act["content"])
                obs, kind = self._observation(act["content"])
                msgs.append({"role": "user", "content": obs})
                s.extra["tool_calls"] += 1
                s.evidence = s.extra["tool_calls"]
                if kind.endswith(":truncated"):
                    s.extra["obs_truncated"] += 1
                    kind = kind.split(":")[0]
                s.extra["last_obs"] = kind
            else:   # respond
                s.extra["actions"].append(act["content"])
                s.extra["answer"] = act["content"]
                s.extra["end_reason"] = "agent_respond"
                s.done = True
        s.extra["messages"] = msgs
        s.extra["ctx_chars"] = self._chars(msgs)
        s.step += 1
        lat = time.time() - t0
        s.cost_so_far += r.usd
        s.latency_so_far += lat
        return s, s.done, r.usd, lat

    # ---------------------------------------------------------- quality
    @property
    def evaluator(self):
        if self._evaluator is None:
            self._evaluator = self.backend.make_evaluator(self._shim)
        return self._evaluator

    def evaluate(self, state: HarnessState) -> dict:
        """CRMArena's own Evaluator.evaluate on the final answer; cached on the state."""
        if "eval" in state.extra:
            return state.extra["eval"]
        task: CRMTask = state.extra["task"]
        answer = state.extra.get("answer") or "None"
        self._shim.task_id = state.task_id
        self._shim.purpose = "privacy_judge" if task.reward_metric == "privacy_rejection" else "extractor"
        before_usd, before_calls = self._shim.spent, self._shim.calls
        res = self.evaluator.evaluate(answer, list(task.answer), task.reward_metric, task.category,
                                      list(state.extra["actions"]))
        reward = res.get("reward", 0)
        if isinstance(reward, dict):          # fuzzy_match -> {em, f1, bleu, rouge}; F1 is the score
            q = float(reward.get("f1", 0.0))
        else:
            q = float(reward or 0)
        out = {"quality": max(0.0, min(1.0, q)), "reward": reward, "parsed_answer": res.get("parsed_answer"),
               "eval_cost_usd": self._shim.spent - before_usd, "eval_calls": self._shim.calls - before_calls}
        state.extra["eval"] = out
        return out

    def quality(self, state: HarnessState) -> float:
        return self.evaluate(state)["quality"]

    def episode_meta(self, state: HarnessState) -> dict:
        task: CRMTask = state.extra["task"]
        ev = state.extra.get("eval") or {}
        return {"harness": self.version, "org": task.org_type, "category": task.category, "idx": task.idx,
                "reward_metric": task.reward_metric, "gold": task.answer, "answer": state.extra.get("answer"),
                "parsed_answer": ev.get("parsed_answer"), "reward": ev.get("reward"),
                "eval_cost_usd": round(ev.get("eval_cost_usd", 0.0), 8), "eval_calls": ev.get("eval_calls", 0),
                "end_reason": state.extra.get("end_reason"), "turns": state.step,
                "tool_calls": state.extra["tool_calls"], "invalid_turns": state.extra["invalid"],
                "obs_truncated": state.extra["obs_truncated"],
                "tiers": state.extra["tiers"], "queries": state.extra["actions"],
                "ctx_chars_final": state.extra["ctx_chars"]}
