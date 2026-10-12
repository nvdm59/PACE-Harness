"""τ-bench (retail / airline) as a PACE host harness (version "taubench-v1").

τ-bench (Yao et al. 2024, arXiv:2406.12045; Sierra, MIT license) is fetched as
an external dependency by `scripts/setup_taubench.sh` into
`third_party/tau-bench`; nothing from it is copied into this repository.

What τ-bench's own loop looks like
----------------------------------
`ToolCallingAgent.solve(env, task_index, max_num_steps=30)` resets the domain
env (which resets the LLM user simulator with the task's instruction and returns
the simulated user's first message), then loops up to 30 agent steps: one
`litellm.completion(messages, tools=env.tools_info)` call, `message_to_action`
turns the reply into either a tool call (first tool call only) or a `respond`
action, `env.step(action)` runs the tool against the in-memory database or sends
the text to the user simulator, and the episode ends when the user emits
`###STOP###` or a terminate tool (`transfer_to_human_agents`) is called. The
reward (`Env.calculate_reward`) is 1 iff the final database hash equals the hash
after replaying the gold actions AND every required output string appears in
some agent reply; otherwise 0. The model is fixed for the whole episode.

What this module does instead
-----------------------------
It re-runs that loop one agent step at a time so PACE can choose (op, tier) at
every step. It REUSES τ-bench's pieces unchanged: the domain envs (`get_env`:
tools, database, wiki system prompt, tasks of the `test` split), the user
simulator (`LLMUserSimulationEnv`), `message_to_action`, `Env.step` and the
scorer `Env.calculate_reward`.

Actions: {CONTINUE, ANSWER} x {cheap, mid, frontier}.
  CONTINUE@t  one normal agent step on tier t (tool call or a reply to the user).
  ANSWER@t    end the conversation now: tier t writes its final reply to the user
              (tools disabled for this call), the reply is recorded as a `respond`
              action, and the episode is scored by `Env.calculate_reward`.
On the last allowed step (max_turns - 1; τ-bench's default max_num_steps = 30)
only ANSWER@* is legal.

Deviations from τ-bench's run.py, all deliberate:
  * the user simulator is fixed to the CHEAP tier (Haiku 4.5) for every arm and
    every policy (τ-bench's default user model is gpt-4o). Its calls go through
    the same priced, capped ledger under purpose "user_sim" and are NOT part of
    the episode cost (logged separately in meta, like CRMArena's extractor).
  * the user simulator's constructor-time reset (no instruction; immediately
    overwritten by `env.reset(task_index)`) is answered locally at $0.
  * no `temperature`: Sonnet 5.5 / Opus 5.5 reject non-default sampling values
    (τ-bench sends temperature=0.0 to the agent).
  * `max_tokens` bounded per call (default 2048 agent step, 1024 forced end,
    512 user turn); every call has a timeout.
  * an episode that reaches the step limit without the user stopping is ended by
    ANSWER and scored by `calculate_reward` on its final state; τ-bench's runner
    would report 0 there. Both are logged: `meta.reward` (quality) and
    `meta.reward_strict` (runner semantics: 0 unless the env reported done).
  * a tool call whose arguments are not valid JSON gets an error tool result
    instead of crashing the episode (τ-bench's runner records the exception as
    reward 0).

Reported metric: pass^1 only (one trial per task per arm). τ-bench's pass^k
(probability that all k i.i.d. trials succeed) needs k trials per task and is
not estimated here.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from ..prices import TIERS
from ..spend import SpendLedger
from .base import Harness, HarnessState
from .crmarena import PricedLLM, _get

VERSION = "taubench-v1"
OPS = ["CONTINUE", "ANSWER"]
ACTIONS = [f"{o}@{t}" for o in OPS for t in TIERS]
MAX_TURNS = 30                       # τ-bench ToolCallingAgent.solve(max_num_steps=30)
DOMAINS = ["retail", "airline"]
LAST_KINDS = ["none", "user", "tool_ok", "tool_error", "invalid"]
RESPOND = "respond"                  # tau_bench.types.RESPOND_ACTION_NAME

FORCE_END = ("[Note to the agent, not from the user: the conversation must end now. Do not call any tools. "
             "Write your final message to the user.]")

FEATURE_NAMES = (["bias"] + [f"dom={d}" for d in DOMAINS] + [f"turn={i}" for i in range(6)]
                 + ["turn>=6", "turn_frac", "tool_calls/10", "cost_so_far"] + [f"last={k}" for k in LAST_KINDS])
N_FEATURES = len(FEATURE_NAMES)
TURN0_IDX = FEATURE_NAMES.index("turn=0")


@dataclass
class TauTask:
    task_id: str
    idx: int
    domain: str

    @property
    def category(self) -> str:       # what pace.collect-style logs print
        return self.domain


def crm_layout():
    """The (category, indices, org, indices) layout pace.features.crm_context decodes, for this
    harness's features: category = org = domain. See pace.experiments.taubench.run.tau_layout."""
    ix = [FEATURE_NAMES.index(f"dom={d}") for d in DOMAINS]
    return list(DOMAINS), ix, list(DOMAINS), ix


# ---------------------------------------------------------------- LLM plumbing
class ToolCompletion:
    """completion_fn for PricedLLM that adds `tools` / `tool_choice` to the agent's calls and keeps
    the raw response, so tool calls go through the exact guarded, priced, recorded call path of
    the CRM harness (PricedLLM.complete) without a second copy of it."""

    def __init__(self, base: Callable | None = None):
        self._base = base
        self.tools: list | None = None
        self.tool_choice: str | None = None
        self.last: Any = None

    @property
    def base(self) -> Callable:
        if self._base is None:
            import litellm   # lazy: only for real runs
            litellm.drop_params = True
            self._base = litellm.completion
        return self._base

    def __call__(self, **kw):
        if self.tools:
            kw["tools"] = self.tools
            if self.tool_choice:
                kw["tool_choice"] = self.tool_choice
        self.last = None
        res = self.base(**kw)
        self.last = res
        return res


def message_dict(res: Any, text: str) -> dict:
    """The assistant message of a completion as a plain dict (what τ-bench appends to history)."""
    choices = _get(res, "choices") or []
    msg = _get(choices[0], "message") if choices else None
    if msg is not None and hasattr(msg, "model_dump"):
        d = dict(msg.model_dump())
    elif isinstance(msg, dict):
        d = dict(msg)
    else:
        d = {"role": "assistant", "content": text}
    d["role"] = "assistant"
    calls = d.get("tool_calls") or []
    d["tool_calls"] = [c if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else dict(vars(c)))
                       for c in calls] or None
    return d


class UserSimCompletion:
    """Stands in for `litellm.completion` inside tau_bench/envs/user.py: the user simulator runs on
    the cheap tier through the priced, capped ledger (purpose "user_sim")."""

    def __init__(self, llm: PricedLLM, tools_ctl: ToolCompletion | None, max_tokens: int = 512):
        self.llm, self.tools_ctl, self.max_tokens = llm, tools_ctl, max_tokens
        self.task_id: str | None = None
        self.spent = 0.0
        self.calls = 0

    def __call__(self, model=None, messages=None, custom_llm_provider=None, **_ignored):
        messages = list(messages or [])
        sys_prompt = str((messages[0].get("content") if messages else "") or "")
        if "Instruction:" not in sys_prompt:
            text = "Hi."     # constructor-time reset without a task; overwritten by env.reset(task)
            usd = 0.0
        else:
            saved = None
            if self.tools_ctl is not None:
                saved, self.tools_ctl.tools = self.tools_ctl.tools, None
            try:
                r = self.llm.complete("cheap", messages, self.max_tokens, self.task_id, "user_sim")
            finally:
                if self.tools_ctl is not None:
                    self.tools_ctl.tools = saved
            text, usd = r.text, r.usd
            self.spent += usd
            self.calls += 1
        msg = SimpleNamespace(content=text, model_dump=lambda: {"role": "assistant", "content": text})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], _hidden_params={"response_cost": usd})


# ------------------------------------------------------------------- backend
class TauBenchBackend:
    """The real τ-bench pieces, imported lazily from the vendored package."""

    def __init__(self, domain: str = "retail", task_split: str = "test"):
        assert domain in DOMAINS, domain
        self.domain, self.task_split = domain, task_split
        self._n = None

    def install_user_sim(self, fn: Callable) -> None:
        import tau_bench.envs.user as usermod
        usermod.completion = fn        # LLMUserSimulationEnv calls this module-level name

    def n_tasks(self) -> int:
        if self._n is None:
            if self.domain == "retail":
                from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
            else:
                from tau_bench.envs.airline.tasks_test import TASKS as tasks
            self._n = len(tasks)
        return self._n

    def make_env(self, task_index: int):
        from tau_bench.envs import get_env
        return get_env(self.domain, user_strategy="llm", user_model="cheap-tier", task_split=self.task_split,
                       user_provider="pace", task_index=task_index)

    @staticmethod
    def action(name: str, kwargs: dict):
        from tau_bench.types import Action
        return Action(name=name, kwargs=kwargs)

    @staticmethod
    def message_to_action(message: dict):
        from tau_bench.agents.tool_calling_agent import message_to_action
        return message_to_action(message)


# ------------------------------------------------------------------- harness
class TauBenchHarness(Harness):
    version = VERSION

    def __init__(self, domain: str = "retail", ledger: SpendLedger | None = None, llm: PricedLLM | None = None,
                 backend: Any = None, completion_fn: Callable | None = None, max_turns: int = MAX_TURNS,
                 max_tokens: int = 2048, answer_max_tokens: int = 1024, user_max_tokens: int = 512,
                 timeout: float = 120.0, effort: str | None = "low"):
        assert domain in DOMAINS, domain
        self.domain = domain
        self.ledger = ledger or SpendLedger()
        self.tools_ctl = ToolCompletion(completion_fn)
        self.llm = llm or PricedLLM(self.ledger, completion_fn=self.tools_ctl, timeout=timeout, effort=effort)
        self.backend = backend or TauBenchBackend(domain)
        self.max_turns = int(max_turns)
        self.max_tokens, self.answer_max_tokens = int(max_tokens), int(answer_max_tokens)
        self.user = UserSimCompletion(self.llm, self.tools_ctl, user_max_tokens)
        self.backend.install_user_sim(self.user)

    @property
    def all_actions(self) -> list[str]:
        return list(ACTIONS)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def tasks(self) -> list[TauTask]:
        return [TauTask(f"{self.domain}-{i}", i, self.domain) for i in range(self.backend.n_tasks())]

    def sample_tasks(self, n: int, seed: int = 0) -> list[TauTask]:
        """Seeded shuffle of the task ids; the same seed always yields the same order, so a larger
        n extends a smaller run (resume per (task, tier) then skips what is logged)."""
        ts = self.tasks()
        random.Random(seed).shuffle(ts)
        return ts[:n]

    # ------------------------------------------------------------ state
    def reset(self, task: TauTask) -> HarnessState:
        self.user.task_id = task.task_id
        u0_usd, u0_calls = self.user.spent, self.user.calls
        env = self.backend.make_env(task.idx)
        first = env.reset(task_index=task.idx).observation
        messages = [{"role": "system", "content": env.wiki}, {"role": "user", "content": first}]
        return HarnessState(task_id=task.task_id, version=self.version, extra={
            "task": task, "env": env, "messages": messages, "tool_calls": 0, "invalid": 0, "last_obs": "none",
            "tiers": [], "end_reason": None, "env_done": False, "u0": (u0_usd, u0_calls), "actions": []})

    def legal_actions(self, state: HarnessState) -> list[str]:
        if state.done:
            return []
        if state.step >= self.max_turns - 1:
            return [a for a in ACTIONS if a.startswith("ANSWER@")]
        return list(ACTIONS)

    def featurize(self, state: HarnessState) -> list[float]:
        x = [0.0] * N_FEATURES
        x[0] = 1.0
        x[1 + DOMAINS.index(state.extra["task"].domain)] = 1.0
        o = 1 + len(DOMAINS)
        x[o + min(state.step, 6)] = 1.0
        o += 7
        x[o] = state.step / max(self.max_turns - 1, 1)
        x[o + 1] = state.extra["tool_calls"] / 10.0
        x[o + 2] = float(state.cost_so_far)
        o += 3
        x[o + LAST_KINDS.index(state.extra["last_obs"])] = 1.0
        return x

    # ------------------------------------------------------------- step
    def _call(self, tier: str, msgs: list[dict], tools: list | None, tool_choice: str | None, max_tokens: int,
              task_id: str, purpose: str):
        self.tools_ctl.tools, self.tools_ctl.tool_choice = tools, tool_choice
        try:
            r = self.llm.complete(tier, msgs, max_tokens, task_id, purpose)
            raw = self.tools_ctl.last
        finally:
            self.tools_ctl.tools = self.tools_ctl.tool_choice = None
        return r, message_dict(raw, r.text)

    def step(self, state: HarnessState, action: str):
        if action not in ACTIONS:
            raise ValueError(action)
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action} at turn {state.step}")
        op, tier = action.split("@")
        t0 = time.time()
        s = state.clone()
        env = s.extra["env"]
        msgs = list(s.extra["messages"])
        s.extra["tiers"] = list(s.extra["tiers"]) + [tier]
        s.extra["actions"] = list(s.extra["actions"])
        if op == "ANSWER":
            r, m = self._call(tier, msgs + [{"role": "user", "content": FORCE_END}], env.tools_info, "none",
                              self.answer_max_tokens, s.task_id, f"agent:{tier}:answer")
            content = (m.get("content") or r.text or "").strip() or "(empty)"
            msgs.append({"role": "assistant", "content": content})
            env.actions.append(self.backend.action(RESPOND, {"content": content}))   # what Env.step records
            s.extra["actions"].append(RESPOND)
            s.extra["end_reason"] = "forced_end" if state.step < self.max_turns - 1 else "max_turns"
            s.done = True
        else:
            r, m = self._call(tier, msgs, env.tools_info, None, self.max_tokens, s.task_id, f"agent:{tier}")
            if m.get("tool_calls"):
                m["tool_calls"] = m["tool_calls"][:1]       # τ-bench executes the first tool call only
            else:
                m["content"] = (m.get("content") or "").strip() or "(empty)"
            try:
                act = self.backend.message_to_action(m)
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                act, err = None, e
            if act is None:     # malformed tool-call arguments
                call = m["tool_calls"][0]
                msgs += [m, {"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                             "content": f"Error: could not parse tool arguments ({type(err).__name__})"}]
                s.extra["invalid"] += 1
                s.extra["last_obs"] = "invalid"
            else:
                resp = env.step(act)
                s.extra["actions"].append(act.name)
                if act.name != RESPOND:
                    call = m["tool_calls"][0]
                    msgs += [m, {"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                                 "content": resp.observation}]
                    s.extra["tool_calls"] += 1
                    s.evidence = s.extra["tool_calls"]
                    obs = str(resp.observation)
                    if obs.startswith("Unknown action"):
                        s.extra["last_obs"] = "invalid"
                        s.extra["invalid"] += 1
                    else:
                        s.extra["last_obs"] = "tool_error" if obs.startswith("Error") else "tool_ok"
                else:
                    msgs += [m, {"role": "user", "content": resp.observation}]
                    s.extra["last_obs"] = "user"
                if resp.done:
                    s.extra["env_done"] = True
                    s.extra["end_reason"] = "terminate_tool" if act.name != RESPOND else "user_stop"
                    s.extra["reward_info"] = resp.info.reward_info
                    s.done = True
        s.extra["messages"] = msgs
        s.step += 1
        lat = time.time() - t0
        s.cost_so_far += r.usd
        s.latency_so_far += lat
        return s, s.done, r.usd, lat

    # ---------------------------------------------------------- quality
    def evaluate(self, state: HarnessState) -> dict:
        """τ-bench's own Env.calculate_reward (0/1); cached on the state. When the env ended the
        episode itself, its reward_info is reused (calculate_reward replays gold actions on
        a reloaded database, so it must not run twice)."""
        if "eval" in state.extra:
            return state.extra["eval"]
        env = state.extra["env"]
        rr = state.extra.get("reward_info") or env.calculate_reward()
        reward = float(_get(rr, "reward", 0.0) or 0.0)
        info = _get(rr, "info")
        out = {"quality": max(0.0, min(1.0, reward)), "reward": reward,
               "reward_strict": reward if state.extra.get("env_done") else 0.0,
               "r_actions": _get(info, "r_actions"), "r_outputs": _get(info, "r_outputs")}
        state.extra["eval"] = out
        return out

    def quality(self, state: HarnessState) -> float:
        return self.evaluate(state)["quality"]

    def episode_meta(self, state: HarnessState) -> dict:
        task: TauTask = state.extra["task"]
        ev = state.extra.get("eval") or {}
        u0_usd, u0_calls = state.extra["u0"]
        return {"harness": self.version, "domain": task.domain, "category": task.domain, "idx": task.idx,
                "reward": ev.get("reward"), "reward_strict": ev.get("reward_strict"),
                "r_actions": ev.get("r_actions"), "r_outputs": ev.get("r_outputs"),
                "user_model_tier": "cheap", "user_cost_usd": round(self.user.spent - u0_usd, 8),
                "user_calls": self.user.calls - u0_calls, "end_reason": state.extra.get("end_reason"),
                "turns": state.step, "tool_calls": state.extra["tool_calls"], "invalid_turns": state.extra["invalid"],
                "tiers": state.extra["tiers"], "actions": state.extra["actions"]}
