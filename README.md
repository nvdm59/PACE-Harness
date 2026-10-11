# Priced runtime control over frozen agent harnesses

This repository accompanies the technical report **"Priced Runtime Control over Frozen Agent Harnesses: a Learned Orchestrator, an Off-Policy Promotion Gate, and Drift Metrics"** (draft v0.1, `paper/main.pdf`). It contains all code, logged data and result files behind every number in the report, and nothing else.

| what | where |
|---|---|
| the report (LaTeX source, figures script, PDF) | `paper/` |
| platform: harness interface, traces with propensities, fail-closed spend ledger, controller, trainers, gate, drift | `pace/` |
| the simulator pipeline (one command, ~1 min, no API) | `bash scripts/run_v0.sh` |
| CRMArena-Pro host harness and the 150-task counterfactual log (every task on every tier) | `pace/harness/crmarena.py`, `traces/crmarena_b2b_counterfactual.jsonl` |
| offline controller, gate and repricing analysis on that log (no API) | `python scripts/crmarena_offline.py` → `results/crmarena_150/` |
| LLM controller (GRPO/LoRA on Modal) | `pace/llm/`, `results/llm/` |
| process-verified multi-agent orchestration, orchestration reward model | `pace/experiments/mas/`, `results/mas/` |
| priced Socratic self-refinement | `pace/experiments/ssr/`, `results/ssr/`, `traces/ssr/` |
| continual orchestrator over a frozen, growing harness (simulator calibrated to the real logs), verifier ablation, RAO credit, meta-learned init | `pace/experiments/mas_continual/`, `results/mas_continual/` |
| skill handbook for the cheap tier, with its no-hints control | `pace/experiments/skills/`, `results/skills/`, `traces/skills_cheap_b2b.jsonl`, `traces/cheap_rerun_b2b.jsonl` |
| the same controller task on verl (Ray, vLLM) and NeMo-RL, and the three-way comparison on the run that failed in TRL | `pace/llm/backends/`, `results/llm/backends/COMPARISON.md` |
| tests (CPU, mocked; no network) | `python -m pytest tests -q` |

Data: the CRMArena-Pro benchmark and the logs derived from it are CC BY-NC 4.0 (research use); see the "Data and licenses" section at the end of this file. Nothing here calls a paid API unless you run a collection command with an explicit `--cap`.
---

# PACE — Priced Adaptive Control for Evolving harnesses

PACE is a **learned, cost-priced runtime controller** that sits on top of a **frozen agent
harness**. At every step it decides which `(op, tier)` the harness runs next (answer, retrieve,
verify or escalate, on a cheap, mid or frontier model) under a per-task cost budget whose price
λ is *learned*. The controller is **trained offline from logged traces**, is put into service only
through an **off-policy promotion gate** (IPS / SNIPS / doubly-robust estimates with bootstrap
CIs), and is **evaluated under drift** (new harness versions, repricing) with a priced
backward-transfer metric.

> **Clean-room rule.** This repository uses public assets only: a simulator written here, public
> model APIs, and the public CRMArena-Pro benchmark (fetched at setup, not copied). It contains
> no code, traces or data from any other project.

## What was measured, and how to reproduce it

| experiment | setting | what it showed | reproduce |
|---|---|---|---|
| **Simulator pipeline** | sim harness v1 → v2 drift; 3,000 + 6,000 logged episodes; budget 0.30 | the gate PROMOTEs the constrained-GRPO candidate (DR quality 0.851 [0.815, 0.887], ground truth 0.880); drift costs the frozen controller −0.045 in J, almost all of it **cost**; the gated retrain recovers +0.033 and restores the budget | `bash scripts/run_v0.sh` (CPU, about 2 min) |
| **CRMArena-Pro collection** | 50 b2b tasks × 3 tiers (Haiku 4.5 / Sonnet 5.5 / Opus 5.5), fixed continuation, $17.98, committed in `traces/` | with 2–3 tasks per category, **no learned linear controller beats the best fixed arm on held-out tasks** (5-fold CV, exact lookups); the per-task oracle reaches 0.769 at $0.092 against 0.663 at $0.164 for the best arm, so the headroom is per task, not per category | `python scripts/crmarena_offline.py` (CPU, about 15 s) |
| **Gate sample-size finding** | the same log, one logged arm per task (50 episodes), 200 redraws | DR halves IPS's quality RMSE (0.048 vs 0.094), but its 90% cost CI is $0.046 wide: a candidate whose exact cost is $0.0987 (under the $0.10 budget) is PROMOTEd in only **6%** of redraws. Certifying a $0.005 margin needs a CI near $0.01, i.e. on the order of 1,000 tasks | same script, section 3 |
| **LLM controller** | Qwen3-1.7B + LoRA, GRPO, priced reward with learned λ; sim (phases 1, 1b, 1c) and CRMArena fold 0 (phase 2) | GRPO alone never found the budgeted optimum (near-deterministic groups give no advantage signal); an SFT warm start did the work and GRPO's contribution was the **price** λ ≈ 0.44; on CRMArena the LLM ties fixed frontier at half the cost on the 10 held-out tasks of one fold, which is a single draw, not evidence of generalization | logged artifacts in `results/llm/` (one Modal run each); commands below |

```bash
pip install -r requirements.txt          # numpy only
bash scripts/run_v0.sh                   # simulator pipeline: collect -> train -> gate -> drift -> retrain -> gate
python scripts/crmarena_offline.py       # CRMArena-Pro Part A on the committed log (no API calls)
python -m pytest tests -q                # CPU, no network (or: python tests/test_smoke.py)
```

Every number in this README and in `results/` comes from one of those three commands or from a
logged artifact under `results/llm/` (the GPU runs, which ran once on Modal). Both CPU pipelines
are seeded end to end; the CRMArena script is byte-for-byte deterministic on the committed log.

## Design in one page

**Objective.** Maximize quality subject to a cost budget `B` per task. The budget is the product
spec. The price of cost, λ, is a Lagrange multiplier that is *learned* by dual ascent (with a
proportional damping term, a PID-Lagrangian without the derivative term) on an EMA of realized
episode cost, so the policy meets `B` after a repricing without anyone retuning λ. Primal-dual
iterates orbit the budget, so the served policy is the **iterate average** (the per-state mean of
weight snapshots from the second half of training).

**Policy.** A softmax-linear controller over state features: an observed difficulty proxy,
evidence count, a verified flag, the step index, cost so far, and harness-version crosses. Logits
are soft-clipped (`L·tanh(z/L)`) so the softmax stays steerable and importance ratios stay
bounded. Serving mixes in ε-exploration, and the logged propensity **includes** ε. That makes every
served log usable for off-policy evaluation. Training rollouts are sampled from the policy *as
served* (ε included) and the policy gradient uses the score of that ε-mixed policy, so the cost the
dual variable tracks is the served cost.

**Three cadences.**

| cadence | what runs | latency / frequency | code |
|---|---|---|---|
| serve-time | `Controller.act(features, legal)` per harness step; log `(features, action, propensity, cost, latency)` | microseconds per decision, every request | `controller.py`, `collect.py` |
| design-time, fast | retrain on fresh traces (constrained GRPO on replay, or AWR+IPS), then an off-policy gate (IPS / SNIPS / doubly-robust with bootstrap CIs) decides **PROMOTE or REVERT** | hours to daily | `train_offline.py`, `gate.py` |
| design-time, slow | the harness evolves (a new version, tools, a model swap, repricing). Measure priced-BWT of the frozen controller, re-collect with exploration, retrain and gate, and revisit the features, action space and reward model | per harness release / weeks | `drift.py`, `harness/` |

**Gate.** `PROMOTE iff DR-quality lower 90% CI ≥ q_prev − ε AND DR-cost upper 90% CI ≤ B + tol`. The
DR direct model is a per-feature-bucket mean fitted from the log, iterated under the candidate
(bucketed fitted-Q evaluation); on bandit logs it is cross-fitted (leave-one-task-out). The
bootstrap refits it on every resample, so the CI includes direct-model variance. `q_prev` is the
incumbent's quality. After a repricing that pushes the incumbent over budget, use the quality
**recorded at the incumbent's own promotion** (`--q-prev-from`). Otherwise an over-budget incumbent
sets a bar that no budget-respecting candidate can pass.

**Drift metric.** `J = quality − λ·cost`. priced-BWT = `J(frozen on v2) − J(frozen on v1)`. recovery =
`J(serving policy on v2 after retrain + gate) − J(frozen on v2)`.

**What each piece operationalizes.** Per-decision importance sampling (Precup, Sutton & Singh
2000), the self-normalized estimator (Swaminathan & Joachims 2015), doubly-robust evaluation for
bandits (Dudík, Langford & Li 2011, arXiv:1103.4601) and for sequences (Jiang & Li 2016,
arXiv:1511.03722), bucketed fitted-Q evaluation (Le, Voloshin & Yue 2019, arXiv:1903.08738),
cross-fitting of the direct model (Chernozhukov et al. 2018, arXiv:1608.00060), AWR (Peng et al.
2019, arXiv:1910.00177), GRPO (Shao et al. 2024, arXiv:2402.03300), the PID-Lagrangian (Stooke,
Achiam & Abbeel 2020, arXiv:2007.03964), backward transfer (Lopez-Paz & Ranzato 2017,
arXiv:1706.08840), and the propensity-logging discipline of Bottou et al. 2013 (arXiv:1209.2355).
Each module's docstring names what it implements.

### Where a research group's assets plug in

| PACE slot | interface | example asset |
|---|---|---|
| host harness | subclass `pace.harness.base.Harness` (`reset`, `legal_actions`, `step`, `quality`); its skills or agents are the *ops*, its model pool is the *tiers*, and its version string feeds the features | a skill orchestrator or an evolved agent harness as the frozen host; each harness release is a drift event for `drift.py` |
| quality signal | `Harness.quality(state) -> [0,1]` (and `expected_quality` when a calibrated score exists) | an orchestration reward model scoring the final episode, replacing exact match |
| tasks | `Harness.sample_tasks(n, seed)` | benchmark task suites with their checkers (CRMArena-Pro is the worked example) |

## Layout

```
pace/trace.py              TraceStep / Episode dataclasses, JSONL I/O
pace/harness/base.py       abstract frozen Harness + HarnessState
pace/harness/sim.py        SimHarness (v1) and SimHarnessV2 (drift), seeded
pace/harness/react_api.py  ReAct QA harness over API model tiers (lazy SDK import; never run live here)
pace/harness/crmarena.py   CRMArena-Pro host harness (per-turn tier x CONTINUE/ANSWER; CRMArena's own scorer)
pace/prices.py             the single model / price / tier table, imported by every harness and the ledger
pace/spend.py              spend ledger (traces/spend.jsonl) with a hard pre-call cap
pace/features.py           state -> feature vector; DR bucket keys (sim); CRM (category, org) context decoder
pace/controller.py         softmax-linear policy, ε-propensities, logit clipping, iterate mix, JSON I/O
pace/collect.py            run a behavior policy, log episodes; Monte Carlo ground truth (sim)
pace/train_offline.py      AWR+IPS (full batch) and constrained GRPO on replay / on a lookup table, learned λ
pace/gate.py               IPS / SNIPS / DR + bootstrap CIs, PROMOTE/REVERT (multi-step sim; first-step bandit on CRMArena)
pace/lookup.py             CRMArena counterfactual log as a (task x tier) table, K-fold split, repricing from the ledger
pace/drift.py              reprice / swap events, priced-BWT and recovery table
pace/llm/                  LLM controller: text env + strict parser, priced rewards (sim / CRMArena lookup),
                           learned λ, SFT warm start, GRPO training and eval as Modal apps
scripts/run_v0.sh          end-to-end sim pipeline (about 2 min)
scripts/crmarena_offline.py  linear controller on the CRMArena log: CV, DR gate, repricing drift (CPU, about 15 s)
pace/experiments/mas/     process-verified planner / worker / verifier orchestration on CRMArena-Pro, priced per request; orchestration reward model; real-run results in results/mas/ (see its README)
pace/experiments/mas_continual/  continually trained MAS orchestrator: the decision point learns online across days under scripted harness stages (tools / skills / agents / prices), gated promotion, orchestrator-level BWT / FWT vs frozen and SkillOrchestra-style routers; simulator calibrated to the real MAS log; results in results/mas_continual/ (see its README)
scripts/setup_crmarena.sh  fetch CRMArena into third_party/ (gitignored) + a Python 3.12 venv
scripts/estimate_cost.py   cost arithmetic from pace/prices.py (assumed, or measured from the ledger)
tests/                     sim pipeline, estimators and trainers, CRMArena harness (mocked), offline pieces, LLM pieces
traces/crmarena_b2b_counterfactual.jsonl (+ .spend.jsonl)   the committed CRMArena-Pro log and its ledger snapshot
results/crmarena/          tables.md / results.json / policies written by crmarena_offline.py; RESULTS.md = interpretation
results/llm/               train logs, run metadata, SFT reports and eval tables of the Modal runs
examples/policy_v1_sim.json  the v1 policy used as the linear baseline in the LLM evals (see Limitations)
```

## The simulator

Each task has a latent difficulty: easy (p = 0.65) or hard (p = 0.35). The controller sees only a
noisy proxy, which is correct 85% of the time. There are 12 actions, `{ANSWER, RETRIEVE, VERIFY,
ESCALATE} × {cheap, mid, frontier}`, and each tier has its own cost, latency and quality tables. An
episode lasts at most 4 steps, and only terminal ops are legal on the last one. **v2 (drift)** adds
`CACHED_RETRIEVE`, which is near-free evidence that only works on easy tasks. The verifier also *ages*,
so VERIFY adds nothing on hard tasks, and frontier is repriced ×1.5. `run_v0.sh` adds a further
scripted `reprice(mid, 1.2)` event. Costs are in sim units, not dollars.

The individual steps are CLIs: `python -m pace.collect | pace.train_offline | pace.gate | pace.drift --help`.

### Results (sim, from `scripts/run_v0.sh`)

**Gate v1.** The cgrpo candidate (budget 0.30, learned λ̄ = 0.343) is scored against the uniform
behavior policy on 3,000 logged v1 episodes:

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.754 [0.743, 0.765] | 0.293 [0.286, 0.301] |
| candidate | IPS | 0.742 [0.638, 0.866] | 0.267 [0.228, 0.309] |
| candidate | SNIPS | 0.761 [0.682, 0.832] | 0.272 [0.246, 0.297] |
| candidate | DR | 0.851 [0.815, 0.887] | 0.306 [0.302, 0.309] |

The decision is **PROMOTE**. Ground truth for the candidate (20,000 fresh tasks, expected success
probability) is quality 0.880 at cost 0.303: inside the DR interval. IPS and SNIPS sit low because
the uniform behavior over 12 actions and up to 4 steps gives cumulative ratios up to 326 against a
near-deterministic candidate, and the gate clips them at 50; DR is the estimator the decision uses.

**Gate v2.** After the drift, the adapted policy (retrained on 6,000 v2 episodes served by the frozen
policy with ε = 0.3) is scored against the frozen one, with the quality bar set at the frozen
policy's v1 promotion (0.851):

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (frozen, DR) | - | 0.860 [0.852, 0.868] | 0.408 [0.404, 0.411] |
| candidate | IPS | 0.869 [0.826, 0.908] | 0.312 [0.303, 0.322] |
| candidate | SNIPS | 0.872 [0.851, 0.891] | 0.313 [0.309, 0.317] |
| candidate | DR | 0.873 [0.854, 0.889] | 0.310 [0.309, 0.311] |

The decision is **PROMOTE** (DR cost upper CI 0.311 ≤ B + tol = 0.320).

| condition | quality | cost | within budget | J (λ = 0.343) |
|---|---|---|---|---|
| (i) frozen on v1 | 0.880 | 0.303 | yes | 0.776 |
| (ii) frozen on v2 | 0.870 | 0.406 | NO | 0.731 |
| (iii) adapted on v2 | 0.871 | 0.312 | yes | 0.764 |

priced-BWT is **−0.045** and recovery is **+0.033**. Drift mostly hits **cost**, not quality: the
frozen policy keeps paying for an aged verifier and a repriced frontier. The gated update restores
the budget (to within the gate's tolerance; the served ε mass itself costs about 0.01) at equal
quality. These are single-seed numbers (seeds 1–4 in the script); re-running with other seeds is a
one-line change and was not done.

## Plugging in a real harness

1. Subclass `Harness`. Expose `all_actions`, a `HarnessState` per step with the observable fields
   (difficulty proxy, evidence, verified, step, cost so far, version), and real per-step cost from
   token usage × price. `pace/harness/react_api.py` is a worked example over API model tiers. Its search
   tool is chosen by `EXA_API_KEY` or `TAVILY_API_KEY` and is stubbed if neither is set. Quality is
   exact or contains match, with an optional LLM-judge fallback. It has never been run live here.
2. Collect with exploration (`eps` 0.1–0.3) and **always log propensities**. That is what makes the gate
   possible.
3. Train. `cgrpo` needs `snapshot`/`restore`, so it can re-roll from logged states. That is true of a
   simulator or a deterministic replay sandbox, but not of live APIs. On a real harness without replay,
   use `--method awr`, optionally with `--use-counterfactual` when alternate arms were recorded, or
   collect every arm (counterfactual mode) and train on the lookup table as on CRMArena below.
4. Gate on held-out logged traffic before serving anything.

## Running on CRMArena-Pro

`pace/harness/crmarena.py` (harness version `crmarena-pro-v1`) makes
[CRMArena-Pro](https://github.com/SalesforceAIResearch/CRMArena) (Huang et al. 2025,
arXiv:2505.18878) the first real host harness. CRMArena is a dependency, not a copy:
`scripts/setup_crmarena.sh` clones it into `third_party/CRMArena` (gitignored, CC BY-NC 4.0,
research use) and PACE imports from there.

### How the per-turn decisions are exposed

On the CRMArena-Pro orgs (b2b, b2c), CRMArena supports only its ReAct `ChatAgent`. `tool_call`
asserts `org_type == "original"`. `ChatAgent.act()` is one closed `while turn < max_turns` loop:
it calls `litellm.completion` with one fixed model and parses `<execute>SOQL/SOSL</execute>` or
`<respond>answer</respond>`. On `<execute>` it runs the query through `SalesforceConnector`, pastes
`Salesforce instance output: {rows}` back as a user turn, and on `<respond>` scores the answer with
`Evaluator.evaluate`. The loop has no per-turn hook, the model is fixed for the whole episode, and
only the agent decides when to answer. It cannot be driven turn by turn from outside without
modification. So the harness re-expresses that ~30-line loop as `Harness.step()` and reuses
everything else unchanged: task records and schemas, the ReAct prompts and rule string, the schema
rendering, the internal/external agent split, the Salesforce connector and `Evaluator.evaluate`.
Their agent is not forked. During development the first two messages of every episode were checked
to be byte-identical to `ChatAgent`'s on the b2b task set; that check needs the CRMArena checkout and
the Hugging Face data and is not part of the test suite.

| PACE action | meaning |
|---|---|
| `CONTINUE@tier` | one normal ReAct turn on that tier. If the model emits `<respond>`, the episode ends with its answer. |
| `ANSWER@tier` | force the final answer now on that tier. An instruction is appended to the last user turn. |

Tiers are `cheap` = `claude-haiku-4-5`, `mid` = `claude-sonnet-5-5` and `frontier` = `claude-opus-5-5`,
called through litellm's Anthropic provider (`anthropic/<id>`). The turn cap defaults to 12, and
only `ANSWER@*` is legal on the last turn. Per step, cost is real usage × `pace/prices.py`, latency
is wall clock (model plus Salesforce), and quality is CRMArena's own `Evaluator.evaluate` for the
task's `reward_metric`: exact match, F1 for `fuzzy_match`, or the privacy judge. Its LLM answer
extractor and privacy judge run on the cheap tier through the same priced, capped path. They are
billed to the ledger (`purpose: extractor | privacy_judge`) and recorded in `Episode.meta`, but not
charged to the controller's episode cost.

Features (43 dims, layout in `crmarena.FEATURE_NAMES`) are: bias, a one-hot over the 22 task
categories (19 business and 3 confidentiality), org type, turn one-hot (0–5, ≥6), turn fraction,
tool calls so far, cost so far, whether the last query returned rows, came back empty or errored,
and a 5-bucket approximate context size.

These deviations from CRMArena's `run_tasks.py` are deliberate:

- **No `temperature` / `top_p`.** `ChatAgent` always sends `temperature=0.0, top_p=1.0`. At
  collection time the mid and frontier models rejected those sampling parameters (HTTP 400), so
  their agent could not run those tiers unmodified. The minimal edit to their script is to drop the
  two arguments from the `completion(...)` call in `crm_sandbox/agents/chat_agent.py`. Their
  `Evaluator` is hard-wired to `gpt-4o` on OpenAI.
- **Bounded calls.** `max_tokens` is capped per call (2048 per agent turn including thinking, 1024
  for a forced answer, 256 for the extractor), and every call has a timeout. Mid and frontier get
  `output_config.effort` (`--effort low` by default). The request body litellm 1.67.2 sends is
  checked offline with a fake HTTP client (`tests/test_crmarena.py`).
- **Observations truncated to 20,000 chars** (`--obs-max-chars`, with 0 turning it off). CRMArena
  pastes raw query results, which can run to hundreds of thousands of tokens in one episode.
- **The missing `import re` in `crm_sandbox/env/env.py` is supplied.** Without it, the extractor's
  fallback parser raises `NameError` (which it swallows) and scores a correct answer 0 whenever the
  extractor's reply is not strict JSON.
- **Non-interactive mode only**, with `--eval-mode aided` (the domain metadata, as in their run
  script).

### Setup

```bash
bash scripts/setup_crmarena.sh          # clone + pin CRMArena, Python 3.12 venv, .env from .env.example
source .venv-crmarena/bin/activate
```

Then put the keys in `third_party/CRMArena/.env`:

- `ANTHROPIC_API_KEY`: use a key from a **dedicated workspace** that has its own spend limit set in
  the Console. That limit is a second backstop, independent of `--cap`.
- `SALESFORCE_B2B_USERNAME`, `SALESFORCE_B2B_PASSWORD`, `SALESFORCE_B2B_SECURITY_TOKEN` (and the
  `B2C` set) are prefilled from CRMArena's `.env.example`, which carries the shared read-only org
  credentials they publish. Their README lists a different B2B password and token than the
  `.env.example` does. If the b2b login fails, try the README values.

The task data (`Salesforce/CRMArenaPro`) downloads from Hugging Face on first use.

### Collecting: a capped counterfactual collection

```bash
python -m pace.collect --harness crmarena --org b2b --categories all --n-tasks 150 \
    --tiers all --mode counterfactual --cap 21
```

`counterfactual` runs every task on all three tiers with a fixed op policy: CONTINUE until the agent
answers, with ANSWER forced on the last turn. It logs the behavior as a uniform tier choice, so each
episode has episode propensity 1/3. That is recorded per decision as 1/3 on the first step and
1.0 after it, because the tier is fixed once chosen. `--mode policy --policy <json|uniform> --eps 0.2`
instead serves a controller with ε-exploration and logs its propensities. Tasks come from a seeded
sample, stratified across categories. Output files:

```
traces/crmarena_b2b_counterfactual.jsonl   one Episode per (task, tier arm), appended as it finishes
traces/spend.jsonl                         one line per priced call: ts, model, in/out tokens, usd, task_id, purpose
traces/spend_total.json                    running totals by model and purpose
```

Every 10 tasks it prints one `[spend]` summary line. Before every call (agent turn or extractor),
the ledger prices that call's worst case: input tokens estimated at 2 chars per token, plus the full
`max_tokens`. If `spent + worst case > cap`, it raises `SpendCapReached` *before* sending; the
collector then flushes the ledger, writes `traces/STOPPED_AT_CAP` and exits with code **3**.
Realized spend never exceeds the cap. A call that errors after being sent is charged at its worst
case, because it may have been billed. The ledger is **cumulative** across runs and collection skips
every `(task_id, arm)` already in the trace file, so re-running the same command with a higher
`--cap` or `--n-tasks` resumes; keep `--org`, `--categories`, `--mode` and `--seed` the same so the
task order matches. The one episode interrupted by the cap is dropped and re-run; its partial spend
stays in the ledger as a sunk cost.

**The committed log.** The command above produced `traces/crmarena_b2b_counterfactual.jsonl`. The
run was stopped by hand once 50 tasks were complete (150 episodes; the 51st task's partial calls
stay in the ledger as sunk cost): 832 priced calls, $17.98 under the $21 cap. The ledger
snapshot of that run is committed as `traces/crmarena_b2b_counterfactual.spend.jsonl`
(`python scripts/estimate_cost.py --measured --ledger traces/crmarena_b2b_counterfactual.spend.jsonl`
summarizes it; `traces/spend.jsonl` itself is the live ledger every run appends to).
One org (b2b), one seed (0), one run. Episodes average 5.0 turns; 16 of 150 hit the 12-turn cap
and were force-answered.

### Budgeting a collection

`scripts/estimate_cost.py` prices a planned run from `pace/prices.py` under per-turn token
assumptions, or extrapolates from the ledger:

```
$ python scripts/estimate_cost.py --n-tasks 150
cheap    claude-haiku-4-5   $1/$5:  per episode 26,000 in x $1 +   600 out x $5  = $0.0290; x 150 = $ 4.35
mid      claude-sonnet-5-5  $2/$10: per episode 31,200 in x $2 + 2,200 out x $10 = $0.0844; x 150 = $12.66
frontier claude-opus-5-5    $4/$20: per episode 31,200 in x $4 + 2,200 out x $20 = $0.1688; x 150 = $25.32
extractor/judge (cheap): 450 episodes x $0.00075                                            = $ 0.34
TOTAL $42.67  ($0.284 per task across 3 arms)  ->  a $50 cap buys ~175 tasks
```

The default assumptions (about 6.5–7.8k input and 150–550 output tokens per turn, 4 turns) were
derived before the run from CRMArena's prompts and its published GPT-4o logs; that derivation is
not scripted here. Measured on the committed ledger snapshot (`--measured --ledger
traces/crmarena_b2b_counterfactual.spend.jsonl`): $0.35 per task across the three
arms (median $0.26, p90 $0.62, max $1.60), with 8–11k input tokens per call and only 140–220 output
tokens per call, i.e. the estimate was about 25% low on input and high on output. Input dominates
because the ~5k-token system prompt is re-sent every turn; prompt caching is not used (see
Limitations). Policy-mode collection runs one arm per task, so it costs roughly a third of a
counterfactual run.

## Results on CRMArena-Pro (50 tasks)

Full tables and interpretation: [`results/crmarena/RESULTS.md`](results/crmarena/RESULTS.md)
(its tables are `results/crmarena/tables.md`, which `scripts/crmarena_offline.py` writes). The
phase-2 LLM table: [`results/llm/eval_table_crm_phase2.md`](results/llm/eval_table_crm_phase2.md).
Everything below is offline, on the committed counterfactual log. No new API, Salesforce or judge
calls are made.

The decision learned is the **first-step tier**, because the continuation was fixed during
collection. With every arm logged, the held-out value of any policy is an exact lookup.

```bash
python scripts/crmarena_offline.py                       # Part A: CV, DR gate, repricing drift (about 15 s CPU)
python -m pace.train_offline --traces traces/crmarena_b2b_counterfactual.jsonl --method cgrpo \
    --budget 0.10 --serve-eps 0 --out policies/crm.json   # the CLIs accept CRMArena logs
python -m pace.gate --traces traces/crmarena_b2b_counterfactual.jsonl --candidate policies/crm.json \
    --budget 0.10 --tol 0.005
modal run pace/llm/train_grpo_modal.py --env crm_phase2  # Part B: SFT + GRPO + eval, one A100-40GB container (paid)
```

**Linear controller, 5-fold CV over tasks** (category-stratified folds, seed 0; train on 40,
evaluate on the held-out 10; means over folds; costs in USD per task; the across-fold SD of J is
0.05–0.3, as large as the gaps between rows):

| policy (held-out) | λ = 2: J | λ = 5: J | λ = 10: J | B = $0.10: quality / cost |
|---|---|---|---|---|
| AWR+IPS linear, served (stochastic) | 0.350 | 0.045 | −0.414 | – |
| AWR+IPS linear, greedy | 0.384 | 0.186 | −0.273 | – |
| constrained cgrpo-lookup (iterate mix, learned λ) | – | – | – | 0.553 / 0.131 (over) |
| uniform behavior | 0.337 | −0.015 | −0.601 | 0.571 / 0.117 |
| fixed cheap / mid / frontier | 0.308 / 0.368 / 0.334 | 0.104 / 0.010 / −0.159 | −0.236 / −0.588 / −0.980 | 0.444 / 0.068, 0.608 / 0.120, 0.663 / 0.164 |
| per-task oracle | 0.639 | 0.452 | 0.148 | 0.769 / 0.092 |

- **No learned rule clearly beats the best fixed arm on held-out tasks.** Each category has only
  2–3 tasks, so the category one-hot memorizes. AWR's J at λ = 2 is 0.475 in-sample against 0.350
  held-out. The constrained policy meets the budget in-sample (0.669 at $0.099) and overspends
  held-out. This is a negative result for the linear controller at this sample size.
- **Gate on the real log.** On a realized behavior log (one arm per task, 50 episodes), DR's
  quality RMSE over 200 redraws is 0.048, against 0.094 for IPS, with 99% coverage of its 90% CI.
  The DR cost CI is about $0.046 wide, though. The all-50 constrained policy (exact cost $0.0987,
  within B + $0.005) is therefore PROMOTEd in only 6% of 200 redraws: 50 tasks are too few to
  certify a $0.005 budget margin.
- **Repricing, with costs recomputed from logged tokens.** Frontier ×1.5 and cheap ×3 push the frozen
  policy over budget: priced-BWT is −0.040 and −0.048 at its λ̄ = 1.36/USD. In-sample retraining
  recovers +0.069 and +0.091, but held-out the retrained policy still overspends (about $0.125),
  and the gate on the realized repriced log REVERTs all three retrains. Harness-version drift cannot
  be measured on this log.

**Phase-2 LLM controller (fold 0, 574 s in one A100-40GB container, about $0.35).** Qwen3-1.7B was
trained with SFT on per-category oracle tiers at λ ∈ 0..20/USD, then 100 GRPO steps in lookup mode
with the learned λ in the prompt.

- The SFT policy ignores the shown price. It plays the same per-category rule at every λ, which
  costs about $0.075, under the budget.
- The dual λ therefore fell to 0 by step 17 and stayed there.
- GRPO left the held-out choices unchanged. On the 10 held-out tasks the result is 0.700 at
  $0.073, against fixed frontier 0.700 / $0.142 and the fold-0 linear policy 0.590 / $0.143.
- That is one 10-task fold, where 0.1 of quality is one task. Part A's 5-fold mean for the same
  category rule is 0.584 / $0.085.

The full table, with SFT and GRPO, greedy and sampled, on both splits, is
`results/llm/eval_table_crm_phase2.md`.

## LLM controller (GRPO on Modal)

`pace/llm/` trains the controller as a **small open LLM**, Qwen3-1.7B with a LoRA adapter (r = 16 on the
q/k/v/o and gate/up/down projections, bf16), using TRL's `GRPOTrainer`. The agent models stay **frozen**:
the harness tiers are the arms. Only the 1.7B controller's adapter is trained. The LLM sees a short
prompt with the observed difficulty proxy (never the latent difficulty), the state, the per-tier prices,
the budget and the legal actions. It answers with exactly one tag, `<action>TIER OP</action>`.

- **Parsing is strict.** The completion must be that single tag and nothing else, and the action must
  be legal. Anything else is invalid: the reward is a fixed −1, and the cost is booked as the cheapest
  safe action (`cheap ANSWER`).
- **Reward.** After the LLM's first action, a fixed continuation finishes the episode by answering on
  the same tier. On the simulator that is the one-step lookahead `SimHarness.counterfactual()` already
  prices. On CRMArena it is collect.py's counterfactual arm. The reward is
  `r = quality − λ·cost`, where quality is the sim's expected success probability.
- **Learned λ.** `DualAscentLambda` uses the same PI-damped dual ascent on an EMA of cost as
  `train_cgrpo`. It is updated once per GRPO step from the batch mean cost and logged.
- **Batches and rollouts.** Each step generates 64 completions (8 prompts × `num_generations` 8), with
  `max_completion_length` 16, lr 1e-5, KL β 0.02 and 300 steps. The task pool is 2,000 seeded sim tasks.
  The legal actions and the example action in the prompt are shuffled per task, with a seed: with a
  fixed order and example, the base model only ever sampled one or two arms, and GRPO cannot reinforce
  an arm it never samples.

```bash
pip install modal && modal setup                    # once
modal run pace/llm/train_grpo_modal.py --steps 300 --budget 0.30            # phase 1: simulator
modal run pace/llm/eval_modal.py --runs phase1                              # all checkpoints vs baselines
# knobs used for the second run (slower dual step, stronger KL, entropy bonus):
modal run pace/llm/train_grpo_modal.py --run phase1b --lam-lr 0.02 --lam-kp 0.5 --lam-ema 0.95 \
    --beta 0.05 --entropy-coef 0.01
# phase1c: SFT warm start on per-difficulty oracle labels, price-conditioned prompt, then GRPO
modal run pace/llm/train_grpo_modal.py --run phase1c --steps 200 --sft-steps 150 --price-in-prompt \
    --lam0 0.5 --num-generations 16 --temperature 1.2 --temp-escalate 1.5 --entropy-coef 0.01 \
    --beta 0.05 --lam-lr 0.02 --lam-kp 0.5 --lam-ema 0.95
modal run pace/llm/eval_modal.py --runs phase1c
# phase 2: lookup rewards from a CRMArena counterfactual log ((task_id, tier) -> (quality, cost));
# only tasks with all three tiers logged become prompts; the only legal op is CONTINUE
modal run pace/llm/train_grpo_modal.py --env crm_phase2
```

The same price-conditioned task (same prompts, same reward, fixed λ = 0.4) also has verl and NeMo-RL entry points, with 20-step smoke runs and a platform comparison, in [`pace/llm/backends/`](pace/llm/backends/README.md).

Each of these ran **once**; the training logs (`train_log_*.jsonl`, one line per step with reward,
quality, cost, λ, action histogram, invalid rate and `degenerate_group_rate`, the share of prompt
groups whose completions are identical and so give zero GRPO advantage), run metadata, SFT reports
and eval tables are in `results/llm/`. Adapter weights stay on the Modal volume. The GPU stages are
seeded, but no run was repeated, so the numbers below carry no seed variance.

**Cost.** Modal's list price for the A100-40GB was $0.000583/s ($2.10/h) at the time. The 300-step
`phase1` run measured 1,422 s of container wall clock, 1,391 s of it in the trainer loop (about 4.6 s
per step): about $0.83. The phase-1 eval of two runs' checkpoints took 463–465 s (about $0.27).
Pinned stack: torch 2.14.1, transformers 5.19.0, trl 1.15.0, peft 0.21.2, accelerate 1.15.0,
datasets 5.1.0.

### Phase 1 result (sim v1, budget 0.30): GRPO did not find the budgeted optimum

The eval uses 2,000 fresh tasks (seed 777). Each row runs the policy's first action and then ANSWER on
the same tier. The full tables are in `results/llm/eval_table_phase1*.md`.

| policy | quality | cost | within B + 0.02 | invalid |
|---|---|---|---|---|
| Qwen3-1.7B zero-shot (greedy, no adapter) | 0.586 | 0.048 | yes | 18.9% |
| run `phase1` (λ lr 0.1 / kp 2.0, β 0.02), all of steps 100, 200 and 300 | 0.748–0.749 | 0.155 | yes | 0% |
| run `phase1b` (λ lr 0.02 / kp 0.5 / EMA 0.95, β 0.05, entropy 0.01), step 100 | 0.885 | 0.603 | NO | 0% |
| run `phase1b`, step 300 | 0.832 | 0.434 | NO | 0.1% |
| linear controller `policy_v1_sim.json`, first action only | 0.573 | 0.035 | yes | 0% |
| fixed ANSWER on cheap / mid / frontier | 0.522 / 0.688 / 0.820 | 0.020 / 0.100 / 0.400 | yes / yes / NO | 0% |
| **best deterministic per-proxy split** (easy → mid ESCALATE, hard → frontier ANSWER) | **0.814** | **0.250** | yes | – |
| best randomized policy at cost exactly 0.30 (LP) | 0.835 | 0.300 | yes | – |

The linear controller's first action is `cheap RETRIEVE`, because it was trained to *keep going*
(retrieve and verify) afterwards. Under this table's fixed continuation it is truncated, so its row is
not its real performance (0.876 at 0.301 with its own continuation, in the run that produced that
file; see Limitations).

What happened:

- **Group degeneracy.** Within a prompt group the 8 completions are almost always identical: 83.5% of
  groups on average in `phase1b` (`train_log_phase1b.jsonl`). That gives a zero GRPO advantage, so
  the only learning signal comes from λ moving the mean reward between steps. The policy jumps
  between near-deterministic arms instead of learning the per-proxy split.
- **Primal-dual orbit.** In `phase1`, the policy swings to `frontier RETRIEVE` (cost 0.60) and λ climbs
  to 1.02 by step 67. It then collapses onto `mid RETRIEVE` (cost 0.155) and stays there. λ decays to 0
  because the policy is under budget, and nothing pulls it back. With the slower dual step in
  `phase1b`, λ is still rising at step 300 (1.37). The policy is leaving the frontier tier
  (frontier ANSWER at 0.43), but it has not reached the budget.
- **Mixing is not the bottleneck.** A deterministic per-context split, which greedy decoding can
  express, reaches 0.814 at 0.250. Mixing adds only 0.021. What the GRPO runs failed to do is
  condition the action on the difficulty proxy.

### Phase 1c: SFT warm start + price in the prompt

`phase1c` has three stages.

1. **SFT warm start.** 150 steps of plain cross-entropy LoRA fine-tuning (lr 2e-4, batch 16) on 2,000
   `prompt → action` pairs (`pace/llm/sft.py`). The prompt shows a price `λ` drawn from {0.0, 0.1, …, 2.0}.
   The label is the per-difficulty oracle at that price: the action that maximizes the mean of
   `quality − λ·cost` for the observed proxy, taken from the sim's own one-step counterfactual over 20k
   seeded tasks. The label is not drawn from a single fixed λ = 0.5 because, at λ = 0.5, the oracle is
   `mid ESCALATE` for **both** difficulties, so those labels could not teach a split by difficulty.
2. **Merge.** The SFT adapter is merged into the base model, so GRPO's KL reference is the SFT policy.
3. **GRPO.** 200 steps with a fresh LoRA, 16 generations per prompt (64 completions per step),
   temperature 1.2, entropy bonus 0.01, β 0.05, and a slow dual step (lr 0.02, kp 0.5, EMA 0.95,
   λ0 = 0.5). The prompt shows the current λ, rounded to 0.1, as `Price of cost: λ=…`, and the reward
   charges exactly the λ the prompt showed.

**SFT.** Held-out greedy agreement with the oracle labels is **100%** (1,000 fresh prompts, 0%
invalid; `results/llm/sft_report_phase1c.json`). The policy splits by difficulty when the oracle does.
Each cell below is the greedy action on 200 probe tasks:

| shown λ | easy proxy | hard proxy |
|---|---|---|
| 0.3 | mid ESCALATE 117/117 | frontier VERIFY 83/83 |
| 0.5 | mid ESCALATE 117/117 | mid ESCALATE 83/83 (the oracle is the same for both) |
| 1.2 | cheap ESCALATE 117/117 | mid ESCALATE 83/83 |

**GRPO.** The degenerate-group rate stayed above 60%, so the rollout temperature was raised from 1.2
to 1.5 at step 20 (logged as an event). That is the only change made during training. The policy
stays deterministic given the shown price: the degenerate-group rate averaged 93% over steps 1–100
and 97% over steps 101–200. λ settles at 0.44 (the mean over the last 100 steps is 0.443). The shown
price alternates between 0.4 and 0.5, and the policy switches between two rules as it does:

- at λ = 0.4: easy → mid ESCALATE, hard → frontier VERIFY (cost 0.33)
- at λ = 0.5: mid ESCALATE for both (cost 0.15)

The batch cost therefore averages **0.301** (steps 1–100) and **0.296** (steps 101–200). The budget is
met by price alternation over time, not by any single policy.

**Eval.** The table below uses 2,000 seed-777 tasks, with λ̄ = 0.443 shown as λ = 0.4. The full table
is `results/llm/eval_table_phase1c.md`.

| policy | quality | cost | J (λ̄) | within B + 0.02 |
|---|---|---|---|---|
| SFT only, and GRPO steps 100 and 200, greedy and sampled (all six identical) | 0.848 | 0.330 | 0.702 | NO (+0.03) |
| best deterministic per-difficulty rule (easy mid ESCALATE, hard frontier ANSWER) | 0.814 | 0.250 | 0.703 | yes |
| best randomized per-difficulty policy at cost B (LP) | 0.835 | 0.300 | 0.702 | yes |
| *derived:* the phase1c policy served with the price drawn as in training's last 100 steps (0.4 at 86%, 0.5 at 14%) | 0.837 | 0.305 | – | yes |

The last row is computed from the two greedy rules, not decoded by the LLM end to end. The λ = 0.4
rule's eval-task numbers are measured above. That the policy plays `mid ESCALATE` everywhere at λ = 0.5
comes from the 200-task SFT probe.

What this shows:

- **The supervised step did the work.** The GRPO checkpoints make exactly the same choices as the SFT
  policy on all 2,000 eval tasks. GRPO left the policy unchanged: with 16 near-identical samples per
  prompt there is almost no advantage signal.
- **GRPO's contribution is the price, not the policy.** The dual ascent found λ ≈ 0.44. On the
  Lagrangian, at J = 0.702 against 0.703 for the oracle rule, the SFT policy evaluated at that price
  ties both references. λ ≈ 0.44 is the slope where the hard-proxy action flips.
- **A single greedy policy cannot meet B here.** At any one shown price, the policy is one
  deterministic rule, either 0.33 or 0.15 in cost. Meeting B = 0.30 needs a mixture.
  - The LLM does not produce that mixture by sampling: it stays deterministic at T = 1, and at
    T = 1.5 in training.
  - Training reached it through time-mixing of the shown price.
  - Serving can reach it by drawing the shown price, the price-conditioned analogue of the linear
    controller's iterate averaging: 0.837 at 0.305, as derived above.
  - Alternatively, serve the best deterministic rule within budget, which is 0.814 at 0.250.
- **Prompt rounding matters.** Showing λ to 0.1 puts λ̄ = 0.443 on the over-budget side of the flip
  point.

**Cost of phase1c.** 1,134 s of training container wall clock (SFT 127 s, GRPO 955 s) plus 278 s of
eval: about 23.5 A100-40GB minutes, about $0.82 at list price, excluding container start-up.

**Experiment: priced Socratic self-refinement as a test-time compute policy.** [`pace/experiments/ssr/`](pace/experiments/ssr/README.md): SSR rounds (decompose → re-solve by SOQL workers → confidence → rewrite) priced per call on CRMArena-Pro answers, with `never` / `always` / uncertainty-gated / PACE-controller policies scored exactly on a logged refinement tree and compared with buying a better tier up front; real-run results (41 b2b tasks, $14) in [`results/ssr/RESULTS.md`](results/ssr/RESULTS.md).

**Experiment: a skill handbook for the cheap tier.** [`pace/experiments/skills/`](pace/experiments/skills/README.md): per-category SOQL notes distilled by the mid tier from mid / frontier successes in the counterfactual log, served leave-one-task-out into the cheap tier's system prompt; results in [`results/skills/RESULTS.md`](results/skills/RESULTS.md).

## Limitations

- **Fifty tasks.** The real-harness evidence is one b2b collection of 50 tasks (2–3 per category).
  Every held-out number is a mean over 10 tasks per fold, where one task moves quality by 0.1, and the
  across-fold SDs are as large as the differences between policies. The negative result for the
  linear controller is a statement about this sample size, not about the approach.
- **One seed.** The simulator pipeline uses one seed per stage and the CRMArena folds one split
  (seed 0); each GPU run ran once. No variance across seeds is reported anywhere.
- **Simulator vs real.** The simulator's difficulty structure, success probabilities and drift
  event are hand-written, so its gate and drift results demonstrate the mechanics, not the size of
  the effects on a real harness. On the real harness only the first-step tier is learned, because
  the counterfactual collection fixes the continuation; per-turn `(op, tier)` control needs a
  policy-mode collection with exploration at every turn, which was not run. The log holds one
  harness version, so harness-version drift could not be measured on it.
- **Gate margin vs sample size.** With one logged arm per task, a $0.005 margin on a $0.10 budget
  is inside the DR cost CI ($0.046 wide at n = 50), so the gate rejects a candidate that is in fact
  within budget 94% of the time. The gate errs on the safe side; buying power means on the order of
  1,000 tasks (CI width scales with 1/√n) or a wider tolerance. On the sim, DR quality CIs are about
  ±0.02–0.04 with a few thousand episodes, and IPS/SNIPS are biased low by the ratio clip when a
  near-deterministic candidate is scored against a uniform behavior log.
- **What the LLM-controller results do and do not show.** They show that GRPO with 8–16 near-identical
  samples per prompt has no advantage signal on this task, that an SFT warm start on oracle labels
  gives a price-conditioned policy, and that the dual ascent finds the price at which the policy's
  hard-proxy action flips. They do not show that a 1.7B LLM controller beats the linear one: the
  phase-1 comparison truncates the linear policy's continuation, and the phase-2 held-out advantage
  is one 10-task fold where the 5-fold mean of the same rule is lower than the fold-0 draw.
- **Direct-model choices.** The sim gate fits its bucketed direct model in-sample (buckets hold
  hundreds of transitions); the bandit gate cross-fits by task. The bandit redraw study reports the
  resulting bias and coverage empirically rather than assuming them.
- **`examples/policy_v1_sim.json`** is the v1 policy the LLM evaluations used as their linear
  baseline. It was trained before this review corrected the trainer's policy gradient for the served
  ε mass, so it is kept as the artifact those tables reference; `scripts/run_v0.sh` now writes a
  slightly different `policies/policy_v1.json` (the numbers in the sim section above).
- **Not run here.** `react_api.py` has never been run against live APIs. Prompt caching is not used
  on CRMArena, so the ~5k-token system prompt is paid on every turn; cache-token accounting is
  implemented in the ledger and the repricing join but has not been exercised by a live run.

## Data and licenses

CRMArena-Pro (Salesforce AI Research) is CC BY-NC 4.0, research use; it is fetched by
`scripts/setup_crmarena.sh` and never copied into this repository. The committed trace
`traces/crmarena_b2b_counterfactual.jsonl` is a derived research artifact: per-episode features,
actions, costs, the agent's answers and SOQL queries, and the benchmark's gold answers for 50 tasks,
under the same non-commercial terms. `traces/crmarena_b2b_counterfactual.spend.jsonl` is the
matching per-call ledger snapshot (token counts and prices) and contains no benchmark data.
