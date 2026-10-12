"""EvoHarnessBench (Salesforce AI Research) hosted evaluation service as a PACE harness
(version "evoharness-v1").

EvoHarnessBench grows an agent's harness in stages H_1 <= H_2 <= ... <= H_T along three
axes (tools, skills, specialist agents). The service holds the gyms, sandboxes and hidden
verifiers; the agent acts over HTTP through the client SDK it serves
(`simple_agentic_evals`), and grading reads the environment state after the agent finishes.
Project page: https://mas-orchestra.salesforceresearch.ai/evoharness/ . Every `/v1/*`
route and the SDK wheel itself need `EVAL_SERVICE_API_KEY`, obtained by a browser login
(see pace/experiments/evoharness/README.md).

What this module does
---------------------
It runs the agent loop turn by turn so PACE chooses (op, tier) at every turn, exactly as
the CRMArena harness does: the tier models are called through `PricedLLM` (litellm behind
the spend ledger), the model emits one JSON action per turn, the action is executed on the
service, the observation is pasted back, and the service grades the final environment.

Actions: {ACT, TOOL, SKILL, DELEGATE, ANSWER} x {cheap, mid, frontier}.
  ACT@t       one free turn on tier t: the model picks the action type itself.
  TOOL@t      the model must call one tool of the stage's catalog.
  SKILL@t     the model must engage one skill (its procedure is loaded into context).
  DELEGATE@t  the model must hand a sub-task to one specialist agent.
  ANSWER@t    the model must finish now; the service grades the environment.
SKILL / DELEGATE are legal only when the stage exposes skills / specialists; on the last
allowed turn only ANSWER@* is legal. Whatever the op, the action type the model actually
emitted is logged per turn, so the decision points tier x {tool, skill, delegate, answer}
are recorded even under ACT.

The service is reached through a `ServiceBackend` (task listing + per-task `Session`).
`SDKBackend` / `SDKSession` drive the hosted service through the SDK it serves
(`client.task(...)`, `task.start()`, the gym's MCP servers filtered to the stage allowlist,
`task.grade()`, `task.close()`); `MockBackend` is a deterministic offline stand-in for tests.

A session holds live remote state: `HarnessState.clone` shares it, so episodes are
linear (no mid-episode counterfactual branching); counterfactual tiers are separate
episodes on fresh sessions.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..prices import TIERS
from ..spend import SpendLedger
from .base import Harness, HarnessState
from .crmarena import PricedLLM

VERSION = "evoharness-v1"
AXES = ["tools", "skills", "agents"]
BENCHMARKS = ["eog", "ale"]
OPS = ["ACT", "TOOL", "SKILL", "DELEGATE", "ANSWER"]
ACTIONS = [f"{o}@{t}" for o in OPS for t in TIERS]
ACTION_TYPES = ["tool", "skill", "delegate", "answer"]
MAX_TURNS = 12
MAX_STAGE = 12
CONTEXT_BUCKETS = [4000, 8000, 16000, 32000]
DEFAULT_BASE_URL = "https://educator-marrow-cultural.ngrok-free.dev"

FEATURE_NAMES = (["bias"] + [f"axis={a}" for a in AXES] + [f"bench={b}" for b in BENCHMARKS]
                 + ["stage_frac", "new_capability"] + [f"turn={i}" for i in range(6)]
                 + ["turn>=6", "turn_frac", "tool_calls/10", "skills/5", "delegations/5", "cost_so_far",
                    "last_ok", "last_error", "last_invalid", "catalog/50"]
                 + [f"ctx<{b // 1000}k" for b in CONTEXT_BUCKETS] + [f"ctx>={CONTEXT_BUCKETS[-1] // 1000}k"])
N_FEATURES = len(FEATURE_NAMES)
TURN0_IDX = FEATURE_NAMES.index("turn=0")

SYSTEM = """You are an agent solving a task inside a live enterprise environment. The environment is graded
by hidden verifiers that read its final state, so make the required changes with the tools; a final
answer alone does not change the environment.

Each turn, reply with exactly one action as a JSON object inside <action></action>:
  {{"type": "tool", "name": "<tool>", "arguments": {{...}}}}
  {{"type": "skill", "name": "<skill>"}}                       (loads that procedure into context)
  {{"type": "delegate", "agent": "<specialist>", "instruction": "<sub-task>"}}
  {{"type": "answer", "content": "<final answer / summary of what was done>"}}

Tools available at this harness stage:
{tools}
{skills}{agents}"""

FORCE = {
    "TOOL": "\n\n[controller] This turn you must call exactly one tool (type \"tool\").",
    "SKILL": "\n\n[controller] This turn you must engage exactly one skill (type \"skill\").",
    "DELEGATE": "\n\n[controller] This turn you must delegate a sub-task to one specialist (type \"delegate\").",
    "ANSWER": "\n\n[controller] You must finish now: output only an action of type \"answer\".",
}
RULE = ("Your reply had no valid action. Reply with exactly one JSON object inside <action></action> "
        "whose \"type\" is one of tool, skill, delegate, answer.")


# ------------------------------------------------------------------ service API
@dataclass
class EvoTask:
    task_id: str
    axis: str
    stage: int
    prompt: str
    benchmark: str = "eog"
    domain: str = ""
    split: str = "heldout"            # "adapt" (adaptation split) | "heldout"
    new_capability: bool = False      # introduced at this stage (FWT cohort) vs carried over (BWT cohort)
    meta: dict = field(default_factory=dict)


class Session(Protocol):
    def catalog(self) -> dict: ...                    # {"tools": [...], "skills": [...], "agents": [...]}
    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]: ...   # (observation, ok)
    def skill(self, name: str) -> tuple[str, bool]: ...
    def delegate(self, agent: str, instruction: str) -> tuple[str, bool]: ...
    def submit(self, answer: str) -> dict: ...       # {"pass": bool, "score": float, ...}
    def close(self) -> None: ...


class ServiceBackend(Protocol):
    def list_tasks(self, axis: str, stage: int) -> list[EvoTask]: ...
    def open(self, task: EvoTask) -> Session: ...


def api_key() -> str | None:
    return os.environ.get("EVAL_SERVICE_API_KEY")


def health(base_url: str = DEFAULT_BASE_URL, key: str | None = None, timeout: float = 20.0) -> dict:
    """GET /v1/health with the bearer key (the service's documented connectivity + key check)."""
    import urllib.request
    key = key or api_key()
    if not key:
        raise RuntimeError("EVAL_SERVICE_API_KEY is not set; see pace/experiments/evoharness/README.md")
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/health",
                                 headers={"Authorization": f"Bearer {key}", "ngrok-skip-browser-warning": "1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 -- fixed https URL
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


class SDKSession:
    """One live service session over the SDK (`simple_agentic_evals` >= 0.19).

    EOG acts through the gym's MCP servers: `task.mcp_session(server)`, `list_tools()`,
    `call_tool(name, arguments)`, `close()`. With `resource_mode="accumulative"` the task's
    `resources["names"]` is the cumulative tool universe of the stage (H_K, distractors
    included) and the service says to enforce it as the agent's allowlist, so the catalog is
    the MCP listing filtered to it. `grade` runs the hidden verifiers on the environment;
    `on_close` tears the session down (frees the gym DB). Skills / specialists are not served
    on the tools axis; `skill_fn` / `delegate_fn` stay None there."""

    def __init__(self, sdk_task: Any, servers: list[Any], grade: Any, skill_fn: Any = None, delegate_fn: Any = None,
                 allow: list[str] | None = None, on_close: Any = None):
        self.task, self._grade, self._on_close = sdk_task, grade, on_close
        self._skill, self._delegate = skill_fn, delegate_fn
        self.system_prompt = str(getattr(sdk_task, "system_prompt", "") or "")
        self.user_prompt = str(getattr(sdk_task, "user_prompt", "") or "")
        self.mcp: dict[str, Any] = {}
        self.tool_server: dict[str, str] = {}
        self._tools: list[dict] = []
        allowed = set(allow) if allow else None
        for srv in servers:
            name = getattr(srv, "name", srv)
            m = sdk_task.mcp_session(srv)
            self.mcp[name] = m
            for t in _tool_list(m.list_tools()):
                if allowed is not None and t["name"] not in allowed:
                    continue
                self.tool_server[t["name"]] = name
                self._tools.append(dict(t, server=name))

    def catalog(self) -> dict:
        return {"tools": list(self._tools), "skills": [], "agents": []}

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        srv = self.tool_server.get(name)
        if srv is None:
            return f"unknown or not-allowed tool {name!r}", False
        try:
            res = self.mcp[srv].call_tool(name, arguments)
        except Exception as e:  # noqa: BLE001 -- surface transport / tool errors to the agent
            return f"{type(e).__name__}: {str(e)[:500]}", False
        if isinstance(res, dict) and "error" in res and "result" not in res:
            return f"error: {json.dumps(res['error'])[:1000]}", False
        body = res.get("result", res) if isinstance(res, dict) else res
        is_err = bool(body.get("isError")) if isinstance(body, dict) else False
        return _text(body), not is_err

    def skill(self, name: str) -> tuple[str, bool]:
        return (self._skill(name), True) if self._skill else ("skills are not served on this axis", False)

    def delegate(self, agent: str, instruction: str) -> tuple[str, bool]:
        if not self._delegate:
            return "specialists are not served on this axis", False
        return self._delegate(agent, instruction), True

    def submit(self, answer: str) -> dict:
        g = self._grade(answer)
        if not isinstance(g, dict):    # GradeResult
            g = {"pass": bool(g.overall_success), "score": float(g.pass_rate), "n_passed": g.n_passed,
                 "n_total": g.n_total}
        return dict(g)

    def close(self) -> None:
        for m in self.mcp.values():
            try:
                m.close()
            except Exception:  # noqa: BLE001
                pass
        self.mcp = {}
        if self._on_close is not None:
            f, self._on_close = self._on_close, None
            f()


class SDKBackend:
    """The hosted service through `simple_agentic_evals.EvalClient` (key from
    $EVAL_SERVICE_API_KEY). Stage K of an axis = dataset version K of every in-scope EOG
    domain (the paper's EOG streams: calendar, csm, drive, email, hr, hybrid, itsm, teams);
    the `test` split is held out, the `train` split is the adaptation split. Each task is
    run under its own stage's cumulative harness (`resource_mode="accumulative"`)."""

    def __init__(self, client: Any = None, benchmark: str = "eog", domains: list[str] | None = None,
                 splits: tuple[str, ...] = ("test", "train"), resource_mode: str = "accumulative"):
        if client is None:
            from simple_agentic_evals import EvalClient
            client = EvalClient()
        self.client, self.benchmark, self.resource_mode = client, benchmark, resource_mode
        self.domains, self.splits = domains, tuple(splits)

    def stages(self, axis: str) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for ds in self.client.benchmarks()["datasets"]:
            if ds["dataset"] != f"evovling_{axis}":
                continue
            for b in ds["benchmarks"]:
                if b["benchmark"] != self.benchmark:
                    continue
                for d in b["domains"]:
                    dom = d["domain"]
                    if (self.domains is None and dom in PAPER_EOG_DOMAINS) or (self.domains and dom in self.domains):
                        out[dom] = sorted(int(v["version"]) for v in d["versions"])
        return out

    def list_tasks(self, axis: str, stage: int) -> list[EvoTask]:
        out = []
        for dom, versions in sorted(self.stages(axis).items()):
            if stage not in versions:
                continue
            for split in self.splits:
                for tid in self.client.task_ids(f"evovling_{axis}", self.benchmark, stage, split, dom):
                    out.append(EvoTask(task_id=f"{dom}/v{stage}/{split}/{tid}", axis=axis, stage=stage, prompt="",
                                       benchmark=self.benchmark, domain=dom,
                                       split="heldout" if split == "test" else "adapt", new_capability=True,
                                       meta={"sdk_task_id": tid, "sdk_split": split, "introduced": stage}))
        return out

    def open(self, task: EvoTask) -> SDKSession:
        t = self.client.task(f"evovling_{task.axis}", self.benchmark, task.stage, task.meta["sdk_task_id"],
                             split=task.meta["sdk_split"], domain=task.domain, resource_mode=self.resource_mode)
        t.start()
        try:
            names = list((t.resources or {}).get("names") or []) if task.axis == "tools" else None
            task.meta["oracle_tools"] = list(t.oracle_tools or [])
            return SDKSession(t, t.mcp_servers, grade=lambda _ans: t.grade(keep_alive=False), allow=names or None,
                              on_close=t.close)
        except Exception:
            t.close()
            raise


PAPER_EOG_DOMAINS = ["calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams"]


def _tool_list(res: Any) -> list[dict]:
    items = res.get("tools", res) if isinstance(res, dict) else getattr(res, "tools", res)
    out = []
    for t in items or []:
        g = (lambda k, t=t: t.get(k) if isinstance(t, dict) else getattr(t, k, None))
        out.append({"name": g("name"), "description": g("description") or "",
                    "input_schema": g("inputSchema") or g("input_schema") or {}})
    return out


def _text(res: Any) -> str:
    content = res.get("content") if isinstance(res, dict) else getattr(res, "content", None)
    if isinstance(content, list):
        return "\n".join(str(c.get("text") if isinstance(c, dict) else getattr(c, "text", c)) for c in content)
    return str(res if content is None else content)


# ------------------------------------------------------------------ mock backend
class MockSession:
    """A task needs its `required` tools called (with any arguments) and, on the skills /
    agents axes, its required skill engaged / specialist consulted; score = fraction of
    verifiers passed, pass = all passed."""

    def __init__(self, spec: dict):
        self.spec, self.called, self.skills, self.delegated = spec, set(), set(), set()
        self.closed = False

    def catalog(self) -> dict:
        return {k: list(self.spec.get(k, [])) for k in ("tools", "skills", "agents")}

    def call_tool(self, name, arguments):
        if name not in {t["name"] for t in self.spec["tools"]}:
            return f"unknown tool {name!r}", False
        self.called.add(name)
        return json.dumps({"ok": True, "tool": name}), True

    def skill(self, name):
        if name not in {s["name"] for s in self.spec.get("skills", [])}:
            return f"unknown skill {name!r}", False
        self.skills.add(name)
        return f"procedure for {name}: call the required tools in order", True

    def delegate(self, agent, instruction):
        if agent not in {a["name"] for a in self.spec.get("agents", [])}:
            return f"unknown specialist {agent!r}", False
        self.delegated.add(agent)
        return f"{agent} done: {instruction[:80]}", True

    def submit(self, answer):
        checks = [t in self.called for t in self.spec["required"]]
        checks += [s in self.skills for s in self.spec.get("required_skills", [])]
        checks += [a in self.delegated for a in self.spec.get("required_agents", [])]
        score = sum(checks) / len(checks) if checks else 0.0
        return {"pass": all(checks), "score": score, "verifiers": len(checks)}

    def close(self):
        self.closed = True


class MockBackend:
    """Deterministic offline service: stage k of an axis exposes k capabilities on that axis
    (cumulative), and each task needs the capability introduced at its stage."""

    def __init__(self, tasks_per_stage: int = 4, n_stages: int = 3):
        self.tasks_per_stage, self.n_stages = tasks_per_stage, n_stages
        self.opened: list[MockSession] = []

    def _spec(self, task: EvoTask) -> dict:
        k = task.stage
        tools = [{"name": f"tool_{i}", "description": f"tool {i}", "input_schema": {}} for i in range(1, 3 + k)]
        spec = {"tools": tools, "required": [f"tool_{1 + (task.meta['j'] % (2 + k))}"]}
        if task.axis == "skills":
            spec["skills"] = [{"name": f"skill_{i}", "description": f"skill {i}"} for i in range(1, k + 1)]
            spec["required_skills"] = [f"skill_{k}"] if task.new_capability else []
        if task.axis == "agents":
            spec["agents"] = [{"name": f"agent_{i}", "description": f"specialist {i}"} for i in range(1, k + 1)]
            spec["required_agents"] = [f"agent_{k}"] if task.new_capability else []
        return spec

    def list_tasks(self, axis: str, stage: int) -> list[EvoTask]:
        out = []
        for s in range(1, stage + 1):            # cumulative eval set under H_stage
            for j in range(self.tasks_per_stage):
                out.append(EvoTask(task_id=f"{axis}-s{s}-{j}", axis=axis, stage=stage, prompt=f"task {s}.{j}",
                                   domain="hr", split="adapt" if j == 0 else "heldout",
                                   new_capability=(s == stage), meta={"j": j, "introduced": s}))
        return out

    def open(self, task: EvoTask) -> MockSession:
        sess = MockSession(self._spec(task))
        self.opened.append(sess)
        return sess


# ------------------------------------------------------------------ parsing
def parse_action(content: str) -> dict | None:
    m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
    body = m.group(1) if m else content
    j = body.find("{")
    if j < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[j:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("type") not in ACTION_TYPES:
        return None
    return obj


def _render(items: list[dict], with_schema: bool = False) -> str:
    lines = []
    for it in items:
        s = f"- {it['name']}: {it.get('description', '')}"
        if with_schema and it.get("input_schema"):
            s += f" args={json.dumps(it['input_schema'])[:600]}"
        lines.append(s)
    return "\n".join(lines) or "(none)"


# ------------------------------------------------------------------ harness
class EvoHarness(Harness):
    version = VERSION

    def __init__(self, backend: ServiceBackend, axis: str, stage: int, ledger: SpendLedger | None = None,
                 llm: PricedLLM | None = None, max_turns: int = MAX_TURNS, max_tokens: int = 2048,
                 answer_max_tokens: int = 1024, obs_max_chars: int = 20000):
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}")
        self.backend, self.axis, self.stage = backend, axis, int(stage)
        self.ledger = ledger or SpendLedger()
        self.llm = llm or PricedLLM(self.ledger, prompt_cache=os.environ.get("PACE_PROMPT_CACHE", "1") == "1")
        self.max_turns, self.max_tokens, self.answer_max_tokens = int(max_turns), int(max_tokens), int(answer_max_tokens)
        self.obs_max_chars = int(obs_max_chars)
        self._tasks: list[EvoTask] | None = None
        self._live: Session | None = None

    @property
    def all_actions(self) -> list[str]:
        return list(ACTIONS)

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def abort(self) -> None:
        """Tear down the live session of an episode that raised (cap, model failure, service error)."""
        sess, self._live = self._live, None
        if sess is not None:
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass

    def tasks(self) -> list[EvoTask]:
        if self._tasks is None:
            self._tasks = list(self.backend.list_tasks(self.axis, self.stage))
        return self._tasks

    def sample_tasks(self, n: int, seed: int = 0) -> list[EvoTask]:
        """Seeded, stable-prefix (resume-safe) round-robin over (domain, split, cohort) groups,
        each shuffled, so a small n covers domains, held-out and adaptation tasks, and new and
        carried-over capabilities in proportion."""
        rng = random.Random(seed)
        groups: dict[tuple, list[EvoTask]] = {}
        for t in sorted(self.tasks(), key=lambda t: t.task_id):
            groups.setdefault((t.domain, t.split, t.new_capability), []).append(t)
        keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
        for k in keys:
            rng.shuffle(groups[k])
        sizes = {k: len(v) for k, v in groups.items()}
        total = sum(sizes.values())
        out: list[EvoTask] = []
        taken = {k: 0 for k in keys}
        while len(out) < min(n, total):
            # largest remaining deficit vs proportional share
            k = max((k for k in keys if taken[k] < sizes[k]),
                    key=lambda k: (sizes[k] * (len(out) + 1) / total - taken[k], sizes[k]))
            out.append(groups[k][taken[k]])
            taken[k] += 1
        return out

    # ------------------------------------------------------------ state
    def reset(self, task: EvoTask) -> HarnessState:
        sess = self.backend.open(task)
        cat = sess.catalog()
        skills = f"\nSkills (procedures you may engage):\n{_render(cat['skills'])}\n" if cat.get("skills") else ""
        agents = f"\nSpecialist agents you may delegate to:\n{_render(cat['agents'])}\n" if cat.get("agents") else ""
        sys_prompt = SYSTEM.format(tools=_render(cat.get("tools", []), True), skills=skills, agents=agents)
        domain_policy = getattr(sess, "system_prompt", "")
        if domain_policy:
            sys_prompt = domain_policy.strip() + "\n\n---\n\n" + sys_prompt
        prompt = getattr(sess, "user_prompt", "") or task.prompt
        self._live = sess
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt.strip()}]
        return HarnessState(task_id=task.task_id, version=self.version, extra={
            "task": task, "session": sess, "catalog": {k: len(v) for k, v in cat.items()},
            "messages": msgs, "tool_calls": 0, "skills": 0, "delegations": 0, "invalid": 0, "errors": 0,
            "last_obs": "none", "answer": None, "grade": None, "end_reason": None, "tiers": [], "ops": [],
            "types": [], "calls": [], "ctx_chars": sum(len(m["content"]) for m in msgs)})

    def legal_actions(self, state: HarnessState) -> list[str]:
        if state.done:
            return []
        if state.step >= self.max_turns - 1:
            return [a for a in ACTIONS if a.startswith("ANSWER@")]
        cat = state.extra["catalog"]
        ops = ["ACT", "TOOL"] + (["SKILL"] if cat.get("skills") else []) + \
              (["DELEGATE"] if cat.get("agents") else []) + ["ANSWER"]
        return [a for a in ACTIONS if a.split("@")[0] in ops]

    def featurize(self, state: HarnessState) -> list[float]:
        x = [0.0] * N_FEATURES
        task: EvoTask = state.extra["task"]
        e = state.extra
        x[0] = 1.0
        x[1 + AXES.index(task.axis)] = 1.0
        if task.benchmark in BENCHMARKS:
            x[4 + BENCHMARKS.index(task.benchmark)] = 1.0
        x[6] = task.stage / MAX_STAGE
        x[7] = 1.0 if task.new_capability else 0.0
        x[TURN0_IDX + min(state.step, 6)] = 1.0
        o = TURN0_IDX + 7
        x[o] = state.step / max(self.max_turns - 1, 1)
        x[o + 1], x[o + 2], x[o + 3] = e["tool_calls"] / 10, e["skills"] / 5, e["delegations"] / 5
        x[o + 4] = float(state.cost_so_far)
        o += 5
        for j, k in enumerate(["ok", "error", "invalid"]):
            x[o + j] = 1.0 if e["last_obs"] == k else 0.0
        x[o + 3] = sum(e["catalog"].values()) / 50
        o += 4
        x[o + sum(e["ctx_chars"] / 4 >= b for b in CONTEXT_BUCKETS)] = 1.0
        return x

    # ------------------------------------------------------------- step
    def _clip(self, obs: str) -> str:
        if self.obs_max_chars and len(obs) > self.obs_max_chars:
            return obs[:self.obs_max_chars] + f" ... [truncated {len(obs) - self.obs_max_chars} chars]"
        return obs

    def _finish(self, s: HarnessState, answer: str, reason: str) -> None:
        sess = s.extra["session"]
        try:
            s.extra["grade"] = sess.submit(answer or "None")
        finally:
            sess.close()
            self._live = None
        s.extra["answer"], s.extra["end_reason"] = answer, reason
        s.done = True

    def step(self, state: HarnessState, action: str):
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action} at turn {state.step}")
        op, tier = action.split("@")
        t0 = time.time()
        s = state.clone()
        e = s.extra
        for k in ("tiers", "ops", "types", "calls"):
            e[k] = list(e[k])
        e["tiers"].append(tier)
        e["ops"].append(op)
        msgs = list(e["messages"])
        send = msgs
        if op in FORCE:
            last = dict(msgs[-1])
            last["content"] = (last.get("content") or "") + FORCE[op]
            send = msgs[:-1] + [last]
        mt = self.answer_max_tokens if op == "ANSWER" else self.max_tokens
        r = self.llm.complete(tier, send, mt, s.task_id, f"agent:{tier}:{op.lower()}")
        content = r.text.strip()
        msgs.append({"role": "assistant", "content": content or "(empty)"})
        act = parse_action(content)
        if op == "ANSWER" and (act is None or act["type"] != "answer"):
            act = {"type": "answer", "content": (act or {}).get("content") or content}
        e["types"].append(act["type"] if act else "invalid")
        if act is None:
            msgs.append({"role": "user", "content": RULE})
            e["invalid"] += 1
            e["last_obs"] = "invalid"
        elif act["type"] == "answer":
            self._finish(s, str(act.get("content") or ""), "forced_answer" if op == "ANSWER" else "agent_answer")
        else:
            sess = e["session"]
            if act["type"] == "tool":
                obs, ok = sess.call_tool(str(act.get("name")), dict(act.get("arguments") or {}))
                e["tool_calls"] += 1
                e["calls"].append(["tool", act.get("name")])
            elif act["type"] == "skill":
                obs, ok = sess.skill(str(act.get("name")))
                e["skills"] += 1
                e["calls"].append(["skill", act.get("name")])
            else:
                obs, ok = sess.delegate(str(act.get("agent")), str(act.get("instruction") or ""))
                e["delegations"] += 1
                e["calls"].append(["delegate", act.get("agent")])
            e["errors"] += 0 if ok else 1
            e["last_obs"] = "ok" if ok else "error"
            msgs.append({"role": "user", "content": f"Observation ({act['type']}): {self._clip(obs)}"})
            s.evidence = e["tool_calls"] + e["skills"] + e["delegations"]
        e["messages"] = msgs
        e["ctx_chars"] = sum(len(m.get("content") or "") for m in msgs)
        s.step += 1
        lat = time.time() - t0
        s.cost_so_far += r.usd
        s.latency_so_far += lat
        return s, s.done, r.usd, lat

    # ---------------------------------------------------------- quality
    def quality(self, state: HarnessState) -> float:
        """The service's partial-credit score (mean verifier pass rate)."""
        g = state.extra.get("grade")
        if g is None:
            if not state.done:
                raise RuntimeError("quality() on an unfinished episode")
            return 0.0
        return max(0.0, min(1.0, float(g.get("score", 1.0 if g.get("pass") else 0.0))))

    def episode_meta(self, state: HarnessState) -> dict:
        task: EvoTask = state.extra["task"]
        e, g = state.extra, state.extra.get("grade") or {}
        return {"harness": self.version, "axis": task.axis, "stage": task.stage, "benchmark": task.benchmark,
                "domain": task.domain, "split": task.split, "new_capability": task.new_capability,
                "introduced": task.meta.get("introduced"), "pass": bool(g.get("pass", False)),
                "score": g.get("score"), "grade": g, "answer": e.get("answer"), "end_reason": e.get("end_reason"),
                "turns": state.step, "tool_calls": e["tool_calls"], "skills": e["skills"],
                "delegations": e["delegations"], "invalid_turns": e["invalid"], "errors": e["errors"],
                "tiers": e["tiers"], "ops": e["ops"], "types": e["types"], "calls": e["calls"],
                "catalog": e["catalog"]}
