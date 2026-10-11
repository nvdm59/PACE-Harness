# A continually trained MAS orchestrator over a frozen multi-agent structure

The MAS experiment ([`../mas/`](../mas/README.md), real run on CRMArena-Pro) exposed the
orchestration decision point ({VERIFY, SKIP_VERIFY, REPLAN, ANSWER} × tier after every worker
step) as a PACE action and exercised the trainer and the gate on it once, on 30 exploration
episodes. What it lacked is the thing the PACE thesis is about: that decision point **learning
online across days of traffic while the harness around it changes**, promoted behind the
off-policy gate, with transfer measured at the orchestrator level. This package builds that loop
on a simulator calibrated to the real MAS log, runs it under scripted harness stages, and
compares the online learner with the design-time alternatives on the same traffic. Phase A
(here) is CPU, seeded, and free. Phase B (the real-run protocol, at the end) is a plan with a
cost table, not a run.

**Framing.** EvoHarnessBench (arXiv 2609.04280) grows a harness `H_t = {tools, skills,
specialist agents}` monotonically by stages and measures what a *fixed* agent policy forgets or
gains per axis (its headline: agent-pool expansion alone gives −34.7% BWT). SkillOrchestra
(arXiv 2602.19672) routes each query, per mode, to a typed agent in a pool by a learned
competence profile over fine-grained skills with an explicit cost penalty. Here the structure is
frozen at each stage and the **policy** at the decision point is what adapts, so the question is
EvoHarnessBench's turned around: when `H` expands under an orchestrator that keeps learning,
what happens to the policy's backward and forward transfer, and does the gate keep the learner
safe without making it too slow to matter?

## Question and hypotheses

Does an orchestrator trained continually at the decision point, served behind the gate, recover
from each harness change where a frozen controller and a design-time router do not, and at what
cost in transfer?

- **H1 (recovery).** After each stage, the continual orchestrator returns to within 0.02 of the
  per-stage oracle's held-out J within k days; the frozen controller does not, and the
  statically routed SkillOrchestra-style baseline does not either without a re-fit.
- **H2 (the gate's price).** The gate blocks candidates that would have violated the budget or
  regressed quality, at the cost of slower recovery: some REVERTs are false alarms on candidates
  the simulator says were fine, and recovery takes days that an ungated learner would not need.
- **H3 (bounded BWT).** Because the structure at each stage is frozen and later stages only add
  capabilities, the continual policy's BWT on earlier stages is bounded: a policy that learned to
  use a later tool keeps its earlier behaviour where that tool is not legal. (EvoHarnessBench's
  negative BWT is drift of the structure under a fixed policy; here the structure is fixed per
  stage and the policy moves.)

## Design

### The harness as `H = (tools, skills, agent pool)`

`simenv.py` expresses the real MAS as a typed harness. The mapping from the two papers' terms to
this package's fields:

| SkillOrchestra term | here |
|---|---|
| agent pool of typed specialists (agent_id, backbone, mode, per-token price) | `simenv.AGENTS`: `Agent(agent_id, tier, mode, stage)`; the price is the tier's list price in `pace/prices.py` × the stage's price factors |
| modes {search, reasoner, code-reasoner, answer, answer-math} | modes {planner, data-worker, verifier, answer}: the roles the real MAS run logged (a CRM data worker is the "search" specialist of this domain) |
| skill handbook: competence per fine-grained skill per agent, learned from execution traces | `router.Handbook`: competence[(action, task category)] = shrunk mean outcome of logged episodes in which that action was taken at that category; the categories are the fine-grained skills because the real run found they carry the signal (AUROC 0.82 for the category alone) |
| `final_score = weighted_competence − λ_c · avg_cost_usd`, cost breaks ties | `router.HandbookRouter.choose`, literally; `avg_cost` is the mean logged cost of the calls the action triggered |
| explore → learn → select → test | warm-up days 1–5 (explore, ε-served traffic) → `Handbook.learn` on days 1–4 → λ_c chosen on day 5 by episode-level SN-IPS (`pace.experiments.mas.offpolicy`) → served from day 6 |
| tools (retrieval index, code execution) | `simenv.TOOLS`: executable interfaces the orchestrator can `call@` (stage 3: `schema_lookup`, `aggregate_calc`) |

| EvoHarnessBench term | here |
|---|---|
| harness `H_t = {tools, skills, specialist agents}`, `H_1 ⊆ H_2 ⊆ …` | `drift.schedule`: five generations g0…g4, each a `Params` with `available_agents / available_tools / available_skills` that only grow (plus a price axis) |
| stages: frequent capabilities first, long tail later | g0 = the real run's pool; g2 adds agents, g3 tools, g4 skills |
| tool selection / skill engagement / agent delegation / implicit stop | actions `call@<tool>` / `engage@<skill>` / `delegate@<data agent>` / `answer@<answer agent>` (plus `verify@`, `replan@`) |
| delegation precision / recall / coverage | the served action mix per stage (section 8 of RESULTS.md) and the per-stage oracle's mix |
| adaptive state `z_t` (memory, learned skills, prompts), separate from `H_t` | the controller's weights `W` (and its learned λ); `z_t` is what the continual arm updates daily and the frozen arm never does |
| Pass / Score / BWT / FWT | expected quality (Score) and the priced `J_g`; BWT and FWT as defined below, per stage and so per axis |

Actions (18): `delegate@data-{1..4}`, `verify@verify-{1,2}`, `replan@plan-{1,2,3}`,
`answer@answer-{1..4}`, `call@{schema_lookup, aggregate_calc}`, `engage@{soql_aggregate,
date_window, decompose_fine}`. The real orchestrator's 12 macro actions map onto them
(`simenv.MAS_ACTION_MAP`: `VERIFY@t` = `verify@verify-1` then `delegate@data-<t>`, `SKIP_VERIFY@t`
= `delegate@data-<t>`, `REPLAN@t` = `replan@plan-<t>`, `ANSWER@t` = `answer@answer-<t>`). A
decision point follows every non-terminal role call; `delegate` after a verifier FAIL with a
retry available retries the current sub-query (the real orchestrator's rule), else runs the
next one. At most 8 role calls per episode, and the legal set shrinks so an answer always fits,
as in `Orchestrator.legal_actions`.

Features (88): the real orchestrator's 66 dims (`Orchestrator.featurize` is called on a view of
the simulated state, not re-implemented, so `pace.features.bucket_key` and the core gate's direct
model read the same base block), then availability flags for every agent / tool / skill, the
tools called on the current sub-query, the skills engaged, the stage, and a `last_fail` flag.
Unavailable capabilities have fixed zero slots, so a policy trained at stage g keeps its weights
when stage g+1 arrives and `Controller.ensure_actions` pads the new actions with zero logits.

### Calibration

Every parameter is in `simenv.M` (measured, with the count it came from) or `simenv.A`
(assumed). `simenv.calibrate()` recomputes the measured ones from `results/mas/mas_b2b.jsonl`
(130 episodes) and `traces/crmarena_b2b_counterfactual.jsonl` (150 episodes), and the tests
check they match. Measured: plan-length distribution (4 / 22 / 104 of 130), worker step kinds
per tier (rows / empty / error / invalid / respond; mid n = 331, cheap n = 11 and frontier n = 10
are thin), the retry re-draw, the verifier's per-call fail rates by step kind (72 of 124 on rows,
23 of 24 on empty, 12 of 12 structural at $0) and its episode-level P(flag | wrong) = 0.92 and
P(flag | correct) = 0.96, token means per role and tier (planner 5,029/279, worker 8,369/158 with
context growth 6,833 + 1,749 per earlier call, verifier 711/62, answer 12,250/91), per-category
accuracy of the MAS (answer on mid) and the single-agent (category, tier) table that sets the
cheap / frontier answer offsets, and the configuration (8 calls, 1 replan, 1 retry). Assumed:
whether a rows step is the *right* rows (`P(good | rows)` 0.60 / 0.75 / 0.85 by tier), how
coverage of the plan by good steps moves the answer (a logit slope of 1.5, centred on
never_verify's own coverage so that never_verify reproduces the category base rates), a
task-level logit noise, and everything that arrives in later stages (verify-2, data-4, answer-4,
the two tools, the three skills, planner v2), none of which exists in the logs. The full table
with status per parameter is section 10 of `results/mas_continual/RESULTS.md`.

The check that matters: on the stage-0 harness the two fixed modes reproduce the real run's
process numbers (always_verify: $0.089 vs $0.0906 per task, 7.83 vs 7.88 role calls, 2.92 vs
2.94 verifier calls, pass rate 0.345 vs 0.347, 93% vs 94% of episodes at the call cap,
P(flag | wrong) 0.92 vs 0.92; never_verify: $0.082 vs $0.0821, 4.74 vs 4.74 calls). Accuracy
lands at 0.43 / 0.48 (always / never) against 0.48 / 0.46 measured: the simulated verifier is
uninformative, so verify-and-retry re-draws good steps as often as bad ones and the call cap
truncates the plan; the real run's +1 task for verification was inside its own ±0.067 SE.

### Stages (`drift.py`, pre-registered)

| stage | day | axis | change |
|---|---|---|---|
| g0-baseline | 1 | – | the real run's pool: plan-1, data-1..3, verify-1, answer-1..3 (Haiku / Sonnet / Opus), list prices |
| g1-reprice | 9 | price | mid × 0.5, frontier × 1.5 |
| g2-agents | 17 | agents | + verify-2 (P(flag ∣ correct) 0.96 → 0.30 at the same recall: "a better verifier shipped"), + data-4 (database specialist, cheap price), + answer-4 (reporting specialist, cheap price) |
| g3-tools | 25 | tools | + schema_lookup (the next worker step cannot error), + aggregate_calc |
| g4-skills | 33 | skills | + soql_aggregate, date_window, decompose_fine; the planner now decomposes into 1–4 sub-queries (the planner change that alters sub-query counts) |

The `mild` schedule keeps the days and axes with smaller effects (`drift.SCHEDULES`). Both were
written down before the first full run and were not changed afterwards.

### The day loop (`run_continual.py`)

Each day is 96 tasks (seeded, the same for every arm). The continual arm serves its incumbent
with ε = 0.1 and logs propensities; a **learner** trains every day on replay from the last 5 days
of logged states (constrained GRPO: group-relative advantages, λ learned by the PI-damped dual
ascent against B = $0.090 per task, iterate averaging; `train.py` is `pace.train_offline.train_cgrpo`
with the harness's own feature map, nothing else re-implemented), warm-started from itself so
training accumulates; the **gate** (`pace.gate.gate`, DR with bootstrap CIs) then scores the
learner against the incumbent on the log served since the last harness event (up to 5 days) and
decides PROMOTE or REVERT. The policy-independent prefix of every episode (planner + first worker
call, $0.022 at list prices) is not a decision, so the gate's budget is B net of the log's mean
prefix. The per-decision estimators of `pace.experiments.mas.offpolicy` run alongside and their
ESS is logged. An AWR + IPS variant (`continual_awr`, fixed λ = 2) runs the same loop.

Baselines on the same traffic: `frozen` (the learner as of day 5, served without ε, restricted
to the stage-0 actions: a fixed policy under a growing harness), `so_static` (handbook from the
warm-up traces, λ_c tuned once, never updated; new capabilities are scored at the handbook's
prior), `so_relearn` (after every stage: 3 days served with ε = 0.3, then a refit and a re-tune;
the lag counts), `always_verify`, `never_verify`, and a per-stage `oracle` (constrained GRPO on
200 fresh exploration episodes of that stage, 300 iterations, no gate: a reference, not a proven
optimum).

### Metrics

Held-out: `n_heldout` = 150 fixed tasks per stage, simulator expected quality (the gate never
sees these). `J_g = quality − λ̄_g · cost` with λ̄_g the stage oracle's learned price, at which the
budgeted optimum maximizes the Lagrangian over every policy, so J_g has a ceiling. With `R[i, j]`
the held-out J of the policy in service on the last day of stage i under stage j:

- BWT_j = R[T, j] − R[j, j] (j < T), FWT_j = R[j−1, j] − b_j with b_j a uniform policy on stage j
  (Lopez-Paz & Ranzato 2017, as EvoHarnessBench uses them; `R[j−1, j]` is the event-day policy);
- hit_j = the event-day policy's J on stage j minus on stage j−1; recovery_j = R[j, j] − R[j−1, j];
- days-to-recover: first day of the stage within 0.02 of the oracle; stationary J: mean over the
  stage's last four days; cumulative regret: Σ over days 6–40 of the oracle's J minus the
  serving policy's, on the day's stage.

Per axis: each stage has one axis, so the per-stage rows are the per-axis rows.

## How to run

```bash
python -m pace.experiments.mas_continual.run_continual --out results/mas_continual                     # default schedule
python -m pace.experiments.mas_continual.run_continual --schedule mild --out results/mas_continual/mild
python -m pace.experiments.mas_continual.analyze --run results/mas_continual --mild results/mas_continual/mild \
    --prelude results/mas_continual/interpretation.md --out results/mas_continual/RESULTS.md
python -m pytest tests/test_mas_continual.py -q        # or: python tests/test_mas_continual.py
```

CPU only, numpy only, seeded end to end; each run is about three minutes. The run writes
`days.jsonl` (one row per day per arm: served quality / cost / J / λ, gate decision and
estimates, held-out values on every stage so far), `gates.jsonl` (every candidate: DR / SNIPS /
per-decision estimates with ESS, the decision, and the simulator's truth for it), `meta.jsonl`
(configuration, stages, oracles, handbooks, calibration check, runtime) and the policies.

## Results

Full tables and the authored interpretation: [`results/mas_continual/RESULTS.md`](../../../results/mas_continual/RESULTS.md).

Default schedule, one seed (the mild schedule gives the same ordering with smaller gaps):

| arm | cumulative regret vs oracle, days 6–40 | days to recover (g1 / g2 / g3 / g4) | stationary J g1 / g2 / g3 / g4 (oracle 0.460 / 0.445 / 0.458 / 0.466) | mean BWT | mean FWT |
|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | **1.26** | 6 / 6 / 1 / 1 | 0.427 / 0.429 / 0.453 / 0.448 | +0.031 | +0.081 |
| continual (AWR + IPS, gated) | 3.90 | – / – / – / – | 0.363 / 0.342 / 0.330 / 0.321 | −0.006 | +0.012 |
| frozen (trained days 1–5) | 2.60 | – / – / – / – | 0.411 / 0.370 / 0.374 / 0.350 | 0 | +0.049 |
| SkillOrchestra-style static router | 3.46 | – / – / – / – | 0.385 / 0.352 / 0.349 / 0.316 | 0 | +0.023 |
| SkillOrchestra-style router, re-learned per stage | 3.03 | – / – / – / – | 0.389 / 0.369 / 0.346 / 0.363 | +0.019 | +0.032 |
| always_verify | 1.99 | – / – / – / – | 0.426 / 0.389 / 0.397 / 0.376 | 0 | +0.069 |
| never_verify | **0.56** | 1 / – / – / – | 0.466 / 0.423 / 0.434 / 0.428 | 0 | +0.110 |

- **H1, with a caveat.** The continual orchestrator is the only learned arm that recovers after
  every stage and ends each stage within 0.014 of the oracle; the frozen controller and both
  routers never come within 0.02. But the fixed all-Sonnet mode `never_verify` is at the oracle
  after the repricing and within 0.02–0.04 afterwards without learning, so it has the lowest
  regret of all. The learner beats it only in the stationary periods of the capability-adding
  stages (by 0.006–0.020 J, one seed) and loses that back during recovery.
- **The headline is a cold-start artifact.** Served J, continual minus `never_verify`, by stage
  window: g0 −0.073, g1 −0.070, g2 −0.021, g3 +0.024, g4 +0.035; the gap stays positive from day
  24. Regret vs the oracle over days 17–40 is 0.42 vs 0.68 (held-out) and 0.07 vs 0.37 (served,
  where the learner also pays its ε = 0.1 exploration mass); from day 25, 0.22 vs 0.50 and −0.11 vs
  0.36. The deficit is the 12 days on the uniform policy before the first promotion plus the
  repricing recovery; after the crossover the learner wins every stage window while `never_verify`
  falls from above the reference oracle (+0.006 at g1) to −0.038 below it at g4 as capabilities it
  cannot use arrive. One seed; stage ≥ 2 capabilities are assumed.
- **H2, strongly.** 32 of 40 candidates were reverted: 9 rightly (every one a budget violation,
  caught by a DR cost estimate with RMSE $0.006) and 23 as false alarms on the quality criterion
  (DR quality CI 0.20 wide at n ≈ 500–1,000 episodes, 63% coverage). The first promotion came on
  day 13; for 12 days the served policy was the uniform exploration policy while the learner
  was already good. The AWR variant was promoted 14 times on a sequential-DR quality estimate
  biased by +0.68 and is the worst learned arm.
- **H3.** Orchestrator-level BWT of the continual arm is +0.111 / +0.023 / +0.009 / −0.019 per
  stage: later stages only add capabilities, and the policy keeps its earlier behaviour where
  they are not legal. The fixed policies lose at the agent-pool stage and never recover
  (frozen 0.411 → 0.350, static router 0.385 → 0.316 from g1 to g4).

### Ablation: when does verification help? ([`RESULTS_ablation_verifier.md`](../../../results/mas_continual/RESULTS_ablation_verifier.md))

Is "never_verify beats everything" an artifact? Four variants of the stream, 5 seeds each
(`run_continual.py --verifier verify-2 --max-calls 12 --verifier-exempt True --arms ablation`;
`ablation.py` aggregates): A the committed configuration; B the informative verifier (per-call
P(fail ∣ good rows) 0.58 → 0.114, recall unchanged) for every arm from day 1; C = B + call cap 8 → 12;
D = C + verifier calls exempt from the cap.

| variant | always_verify − never_verify: quality | J | cost | always cap hits | learner verify share (g0–g4) |
|---|---|---|---|---|---|
| A committed | −0.033 ± 0.001 | −0.036 ± 0.001 | +$0.0036 | 95% | 0.03–0.09 |
| B verify-2 | −0.004 ± 0.001 | −0.007 ± 0.001 | +$0.0030 | 91% | 0.07–0.15 |
| C + cap 12 | **+0.029 ± 0.001** | **+0.011 ± 0.001** | +$0.0148 (+18%) | 9% | 0.07–0.13 |
| D + exempt | +0.028 ± 0.001 | +0.010 ± 0.001 | +$0.0148 | 0% | 0.08–0.13 |

Verification pays only when the verifier's fail carries information about the step **and** the
retry does not come out of the plan's call budget; then it buys about +0.03 accuracy for +18% cost
(+0.01 J). Under the real run's verifier and cap it hurts. On the real log the per-step picture
matches the per-episode one: of 53 retries after a flag, 33 were of steps that had returned rows
and 26 returned rows again; 10 of 20 flagged empty/error steps were repaired; the 44 retried
episodes flipped 6 wrong → right and 5 right → wrong. Under C/D the fixed `always_verify` costs
$0.098–0.105, over the learner's budget, and has the lowest regret of all arms; the budgeted
learner's verify share rises to 0.10–0.13 and it still ends the capability stages above
`never_verify` under every variant (g4: +0.013 to +0.031, 5-seed SE 0.007–0.011). Variant A also
gives the main result its seed error bars: continual − never_verify stationary J g0 −0.050 ± 0.011,
g1 −0.009 ± 0.002, g2 −0.014 ± 0.007, g3 +0.011 ± 0.010, g4 +0.031 ± 0.007; regret days 6–40
continual 1.06 ± 0.21 vs never_verify 0.74 ± 0.08.

### RAO-style delegation credit and a meta-learned init ([`RESULTS_rao_metainit.md`](../../../results/mas_continual/RESULTS_rao_metainit.md))

Two options of the cgrpo arm, 5 seeds, `--arms ablation`. `--delegation-credit rao --lambda-d 0.5` (`rao_reward.py`):
a `delegate@` decision's node reward is the priced episode outcome + λ_d · mean success of the sub-call(s) it
spawned, against the root-anchored leave-one-out baseline. **Negative:** it over-delegates (share +0.05 to +0.10
per stage, more to data-4 and to the frontier data-3), and stationary J falls 0.02–0.04 per stage from g1. Regret
days 6–40 goes from 1.06 ± 0.21 to 1.76 ± 0.16. `--init meta` (`meta_init.py`): Reptile over stages g0–g3 (2 inner days
of cGRPO per stage, 3 rounds, never held-out sets or g4). It cuts regret days 6–40 to 0.47 ± 0.08, below never_verify's
0.74 ± 0.08. Served J on days 1–12 is 0.362 vs 0.325. The first-promotion day does not move, and there is no gain on
the unseen stage g4. Caveat: the meta-init saw the simulator's definitions of stages the run only reaches later.

## Limitations

- **A simulator calibrated on 130 episodes.** The process statistics are measured; the link from
  a step's latent quality to the final answer, and every capability that arrives after stage 0,
  are assumed (section 10 of RESULTS.md lists each). The sizes of the transfer effects are
  therefore the simulator's, not CRMArena's; the mechanics (what the gate does at this sample
  size, what a frozen policy and a static router do when the pool grows) are what the run shows.
- **Scripted drift, one seed for the full arm set.** Four stages on fixed days, one schedule plus a
  milder one; the all-arms run is one seed. The verifier ablation's variant A repeats the committed
  configuration over 5 seeds for the continual, always_verify and never_verify arms and reports SEs.
- **The gate at n ≈ 500.** With binary outcomes the DR quality CI is ±0.04 or wider, so ε_q is
  0.05 (the core default of 0.03 rejects the incumbent against itself at this n); the cost
  criterion keeps the core's $0.005 tolerance and is the binding one. Section 6a reports the
  estimators' bias and coverage against the simulator's truth.
- **The oracle is a reference.** Constrained GRPO on fresh exploration with more iterations, not
  a proven optimum; a learner can and occasionally does exceed it, so a regret can be slightly
  negative.
- **No real run.** Nothing here calls an API, Salesforce or a judge. Phase B is the plan.

## Phase B: the real-run protocol on CRMArena-Pro

The same loop on the real MAS orchestrator (`pace/experiments/mas/run.py --mode controller
--policy <json> --eps 0.1`), days of 20 b2b tasks, the learner trained with AWR + IPS on the day's
log (no replay exists on the real harness; `cgrpo` needs `snapshot / restore`) and gated with
the per-decision estimators, held-out evaluation by re-serving a fixed 20-task set per stage.
Which stages can be induced for real:

| stage | how, on the real harness | extra cost |
|---|---|---|
| repricing | free: costs are recomputed from the logged tokens (`pace.lookup.reprice_episodes` does this for the single-agent log) | $0 |
| a better verifier | swap the verifier prompt or tier (`--verifier-tier mid`, or a stricter prompt in `roles.VERIFIER_INSTRUCTIONS`) | a Sonnet verifier call is about $0.002 against $0.001 |
| agent pool | a new worker or answer backbone through litellm (any model id in `pace/prices.py`) | depends on the model |
| worker swap by tier | `--worker-tier cheap|frontier` for the default worker | cheap halves, frontier doubles the worker share |
| tools | a schema lookup is one `describe` call through the existing Salesforce connector; an aggregation tool is local | about $0 |
| skills | procedure text added to the worker steering (`Worker.steering`) | tokens only |

Cost, from the measured $0.084 per episode (planner Haiku, worker and answer Sonnet, verifier
Haiku; $0.076 under mixed-tier exploration):

| protocol | episodes | estimate |
|---|---|---|
| 10 days × 20 tasks × 1 served policy | 200 | $17 |
| the same with always_verify and never_verify on every day | 600 | $50 |
| 20 days × 20 tasks × 1 policy with two induced stages + held-out re-serves (20 tasks per stage) | 460 | $39 |
| 40 days × 20 tasks × 1 policy, four stages, held-out re-serves | 900 | $76 |
| 40 days, four stages, with the two fixed baselines | 2,500 | $210 |

Per day of 20 tasks, one policy: about $1.70; the two fixed baselines add $3.40. A $0.005 cost
margin on a $0.09 budget needs about 1,000 logged episodes to certify (the gate finding of
`results/crmarena/RESULTS.md`), so at 20 tasks a day the gate's cost criterion would run on a
rolling 5-day window and still be underpowered; the quality criterion at ε_q = 0.05 needs on the
order of 500 episodes. The honest Phase B at $50 therefore measures the mechanics (what the
learner serves, what the gate does, the frozen policy's hit at each stage) rather than
certifying improvements.
