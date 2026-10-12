# EvoHarnessBench hosted service

PACE's per-turn controller on **EvoHarnessBench** (Salesforce AI Research; project page
<https://mas-orchestra.salesforceresearch.ai/evoharness/>, datasets
[`ZixuanKe/evovling_tools`](https://huggingface.co/datasets/ZixuanKe/evovling_tools),
`evovling_skills`, `evovling_agents`, CC BY-SA 4.0). The service holds the gyms, sandboxes
and hidden verifiers; the agent acts over HTTP and gets a score.

**Status: wired to the live service** (SDK `simple_agentic_evals` 0.19.2, served at `GET /sdk`).
The key lives only in the gitignored `third_party/CRMArena/.env`; load it with
`export $(grep -E '^(EVAL_SERVICE_API_KEY|ANTHROPIC_API_KEY)=' third_party/CRMArena/.env)`.
Results of the runs are in `results/evoharness/` and `traces/evoharness/`.

## What the user must do (once)

1. Open the live demo, <https://ian-sections-none-closest.trycloudflare.com/?mode=enterprise&gym=csm&evolving=1>
   (linked from the project page; the tunnel URL may change, so take the current one from the page),
   log in in the browser, click **MyAuthtoken** and copy the key.
2. `export EVAL_SERVICE_API_KEY="<key>"` (keep it out of the repo, e.g. in a local `.env`).
3. Install the SDK the service serves (from the project page, verbatim):
   ```bash
   SDK_BASE="https://educator-marrow-cultural.ngrok-free.dev"
   WHEEL=$(curl -sSL -H "Authorization: Bearer $EVAL_SERVICE_API_KEY" "$SDK_BASE/sdk" \
     | python -c 'import json,sys; d=json.load(sys.stdin); print(d["path"])')
   curl -fsSL -H "Authorization: Bearer $EVAL_SERVICE_API_KEY" \
     "$SDK_BASE$WHEEL" -o "/tmp/${WHEEL##*/}" && python -m pip install "/tmp/${WHEEL##*/}"
   ```
4. Check connectivity and the key: `python -m pace.experiments.evoharness.run health`
   (`GET /v1/health` with the bearer key).
5. Open the "Full API Reference" notebook (Colab, login-gated:
   <https://colab.research.google.com/drive/1mYQEDCVFStXMRWYI2hpEGBXwyRx1NSFj>) and note the per-task
   calls listed under *Wiring after registration* below.

## Protocol found (public part)

* Base URL `https://educator-marrow-cultural.ngrok-free.dev` (an ngrok tunnel; `EvalClient()`
  resolves it itself). Auth: `Authorization: Bearer $EVAL_SERVICE_API_KEY` on all `/v1/*`.
* Import: `from simple_agentic_evals import EvalClient, run_benchmark, run_leaderboard`.
* Driver is callback style: `run_benchmark(my_agent, mode="deployment_eval", client=EvalClient(), benchmark="eog")`;
  `run_leaderboard(my_agent, "Method", client=..., benchmarks=["eog", "ale"], seeds=3)`.
* Per task the service creates a fresh isolated session; the agent acts; hidden verifiers read
  the environment state; the session is torn down on scope exit.
  * **EOG** (EnterpriseOps-Gym; hr, itsm, csm, email, calendar, drive, teams, hybrid): tools are MCP
    servers, `mcp = task.mcp_session(server); mcp.list_tools(); mcp.call_tool(name, arguments); mcp.close()`
    (close even on exception).
  * **ALE** (Agents' Last Exam): file I/O, `task.fetch_inputs_to(indir)` then `task.submit_dir(outdir)`.
* Harness stages `H_1 ⊆ H_2 ⊆ … ⊆ H_T` along three axes: tools (growing API catalog), skills (growing
  library of procedures), agents (growing pool of specialist agents bundled by domain). At stage t the
  agent gets the full cumulative `H_t`, not an oracle-pruned subset; tasks are feasible at their stage
  and need the newly introduced capability. HF configs are `{domain}_v{stage}` (v1…v12), splits
  `train` / `test`; rows carry `oracle_tools`, `cummulative_tools`, `task_split`
  (`near-term` / `full-spectrum` / `last-exam`).
* Modes: **deployment** (fresh agent on the cumulative eval set under `H_t`, no state) and
  **self-evolving** (state `z_t` updated from the adaptation split and carried across stages).
  Metrics: strict pass (all verifiers), score (mean verifier pass rate), FWT on new tasks,
  BWT on earlier cohorts, cost (hours, Mtokens, 3 seeds).
* No rate limits or service-side pricing are stated; our cost is only our own model calls.

### Per-task API (read from the SDK source)

* Catalog: `client.benchmarks()` lists datasets `evovling_{tools,skills,agents}` x benchmarks
  (`eog`, `ale`, plus loopback adapters) x domains x versions with `n_train` / `n_test`. A stage is a
  **per-domain dataset version**; the paper's EOG streams are calendar, csm, drive, email, hr, hybrid,
  itsm, teams (`enterprise_tri_hybrid` is excluded from the published tables). Tools axis, EOG, test
  split: 175 / 88 / 90 / 81 / 15 / 6 tasks at stages 1-6 (454 in the paper); `train` is the
  adaptation split.
* Tasks: `client.task_ids(dataset, "eog", version, split, domain)`;
  `t = client.task(dataset, "eog", version, task_id, split=..., domain=..., resource_mode="accumulative")`;
  `t.start()` provisions a fresh gym DB (`POST /v1/sessions`) and fills `t.system_prompt` (the domain
  policy), `t.user_prompt`, `t.oracle_tools`, `t.resources`, `t.mcp_servers`. `version` picks the task
  cohort; optional `at_stage` picks the harness independently (cohort x harness matrix for BWT).
* Harness exposure: `resource_mode="accumulative"` puts the cumulative stage universe in
  `t.resources = {"kind", "mode", "stage", "count", "names", ...}`; the gym's MCP server lists its full
  tool set (e.g. 89 for HR) and the service says to **enforce `names` as the agent's allowlist**
  (hr v1: 23 tools). Skills are `SKILL.md` bundles and specialists `.toml` definitions, inspectable via
  `client.resources(..., include_content=True)`.
* Acting: `t.mcp_session(server)` is a pre-authenticated streamable-HTTP MCP client
  (`list_tools()`, `call_tool(name, args)` returning the JSON-RPC envelope).
* Scoring: `t.grade()` (`POST /v1/sessions/{id}/grade`) returns `pass_rate` (score, partial credit),
  `overall_success` (strict pass), `per_verifier`; `t.close()` frees the session.
* `run_benchmark(agent, mode=...)` simply loops these calls with `act(task)`; provided server-side
  harnesses (`react`, `codex`) need an OpenAI key and are not used here.

## What is built

* `pace/harness/evoharness.py`: `EvoHarness(Harness)`. Actions
  `{ACT, TOOL, SKILL, DELEGATE, ANSWER} x {cheap, mid, frontier}`; one `step()` = one model call
  through `PricedLLM` (litellm behind the spend ledger, same path as CRMArena) emitting one JSON
  action, executed on the service session; `ANSWER` submits and the service grade (score, pass) is
  the quality. The action type the model actually took is logged per turn, so the decision points
  tier x {tool, skill, delegate, answer} are recorded. `MockBackend` (offline) and `SDKSession`
  (documented MCP calls) implement the `ServiceBackend` / `Session` protocol.
* `run.py collect`: counterfactual mode (every tier on every task, fresh session per (task, tier)),
  resume on (task_id, arm), hard cap on its own ledger dir, and a projection stop (exit 5) when the
  per-task cost (all tiers) times `--project-n` exceeds the cap.
* `run.py analyze`: per (axis, stage) per-tier score / pass / cost, new vs carried-over cohorts,
  per-task oracle (cheapest tier reaching the best score) and the protocol inputs: held-out and
  adaptation task lists per stage (overlap is an error), FWT / BWT cohorts.

## How our harness maps onto it

`SDKBackend.list_tasks(axis, K)` = every in-scope domain's version K, test (held-out) + train
(adaptation) splits; `open` starts the session in accumulative mode and returns an `SDKSession`
whose catalog is the MCP tool list filtered to the stage allowlist, with the domain policy prepended
to our action protocol. Each task is run under its own stage's harness (the diagonal of the
cohort x harness matrix: FWT-style cells); BWT cells need `at_stage` and are not collected here.
Sampling within a stage is proportional over (domain, split), so both splits stay represented.

## Run plan

```bash
python -m pace.experiments.evoharness.run collect --axis tools --stage 1 --n-tasks 10 --tiers all \
    --cap 50 --traces-dir traces/evoharness            # smoke; exit 5 if per-task cost x 60 > cap
python -m pace.experiments.evoharness.run analyze --traces-dir traces/evoharness
```
If the smoke prices a task (all three tiers) at or below $1, collect the full tools axis, every stage,
up to 200 tasks spread over stages in proportion to their size, under `--cap 150` on the same ledger
(rerun `collect` per stage; it resumes).

Full axis: `collect --axis tools --stage all --n-tasks 200 --tiers all --cap 150 --project-n 0`
(stages get tasks in proportion to their size; resume skips logged (task, tier) pairs).

Offline check: `python -m pace.experiments.evoharness.run collect --axis skills --stage 2 --n-tasks 4 --cap 5 --traces-dir /tmp/evo --dry-run`.
