# Process-verified multi-agent orchestration on CRM tasks

An enterprise service/sales agent that takes a CRM request, decomposes it across **planner /
data-worker / verifier** roles over the frozen [CRMArena-Pro](https://github.com/SalesforceAIResearch/CRMArena)
harness, prices every role call per request through the PACE spend ledger, and verifies the
process step by step before an answer is returned. The orchestration decisions (verify or not,
replan, answer, and on which model tier) are exposed as PACE actions, so the same offline
trainer and promotion gate that drive the single-agent controller drive the orchestrator.

Everything here runs offline against a deterministic fake backend (`--dry-run`); the real run is
one command with a hard spend cap. Nothing in this package changes the PACE core: the one hook
it needs (attributing the harness's own calls to `mas:worker:<tier>` in the ledger) is a proxy
inside the package.

## Question

Outcome scoring (CRMArena's exact match / F1 on the final answer) says whether a multi-step
agent got the answer, not whether its process was sound. For an enterprise agent the second
matters: an answer that is right by luck after a failed query will be wrong next time, and a
wrong answer after a clean process points at a different fix than a wrong answer after a
query error. Three questions, in order:

1. **What does step-level process verification catch that outcome scoring misses, and vice
   versa?** (cf. process verification for multi-agent systems, MAS-ProVe, arXiv 2602.03053)
2. **Does the logged process predict the outcome well enough to serve as a reward?** An
   orchestration reward model (ORM) over process features → P(correct) (cf. reward modeling for
   multi-agent orchestration, arXiv 2606.13598).
3. **Can a priced controller use that reward?** Decisions inside the orchestration loop priced
   per request, `quality_hat − λ·cost`, trained offline and gated, as in the rest of PACE.

## Hypotheses

- **H1 (silent failures exist in both directions).** A non-trivial share of correct answers
  comes with at least one failed verification (the retry fixed it, or the step was not needed),
  and a non-trivial share of wrong answers comes with every verification passed. The first is
  what outcome-only scoring over-credits; the second is what process verification cannot see
  (wrong plan, wrong synthesis). The report measures both as rates with the conditional views
  P(flagged | wrong) and P(wrong | all pass).
- **H2 (the process is predictive).** An ORM over counts of verify / replan / retry, the
  verifier pass rate, observation kinds, tiers used and cost has out-of-fold AUROC clearly above
  0.5 and a Brier score below the base-rate Brier, and the verifier verdicts add information
  beyond the raw observation kinds (the `− verifier features` ablation).
- **H3 (verification pays only when priced).** `always_verify` buys accuracy at a cost; whether
  it wins depends on λ. A controller that verifies selectively (after empty/error observations,
  on hard categories) should match `always_verify`'s accuracy at a cost closer to
  `never_verify`'s. H3 needs the real run and an on-policy evaluation; the dry run only shows
  the mechanism.

## Design

**Roles (`roles.py`).** Each is one priced call through `pace.spend.SpendLedger` (cap checked
before the call, real usage recorded after), with a `purpose` the ledger groups by:

| role | what it does | tier | purpose |
|---|---|---|---|
| PLANNER | decomposes the request into 1–3 sub-queries `{goal, soql}` (JSON; fallback parsing) | cheap (or mid) | `mas:planner` |
| WORKER | runs one sub-query through the harness's own step path (`CONTINUE@tier`): steering text on the last user turn, the model emits `<execute>SOQL</execute>`, the harness runs it and pastes the observation back | per call | `mas:worker:<tier>` |
| VERIFIER | checks one sub-result against the goal and the question: a structural pre-check (query error, no `<execute>`) fails at $0; else one call returns `{"pass", "reason"}` | cheap | `mas:verifier` |
| answer | the harness's `ANSWER@tier` op (forced `<respond>`) | per call | `mas:worker:<tier>:answer` |

The outcome goes through the harness's existing priced evaluate path (CRMArena's `Evaluator`
with its extractor / privacy judge on the cheap tier, purpose `extractor`).

**Decision points (`orchestrator.py`).** After every worker step the orchestrator asks a policy
for one of `{VERIFY, SKIP_VERIFY, REPLAN, ANSWER} × {cheap, mid, frontier}`; the tier is the tier
of the next call (the following worker call, the re-planner, or the answer). `VERIFY` on a fail
retries the same sub-query once on the chosen tier with the verifier's reason; `REPLAN` lets the
planner see the evidence and the verdicts so far. At most **8 role calls** per episode (planner
and answer included); the legal set shrinks so that an `ANSWER` always fits, and with one call
left only `ANSWER@*` is legal. Policies are the two fixed baselines `always_verify` /
`never_verify` (propensity 1) and `pace.controller.Controller` (propensity includes ε), so every
log is usable by `pace.gate` and `pace.train_offline` unchanged. Each decision is a `TraceStep`
(features, action, propensity, cost of the calls it triggered, legal set); each role call is a
process step `{role, tier, input summary, output summary, verdict, reason, cost, latency,
tokens}` in `Episode.meta["process"]`.

**Features.** The first 12 dims follow `pace.features`' base block (bias, observable
difficulty proxy = the last observation was empty/error, evidence count = worker steps that
returned rows, verified flag, decision index, cost so far), so the core's DR bucket key and AWR
baseline work on these logs; then the category and org one-hots, verification counts and pass
rate, plan progress, the last observation kind and the last worker tier (66 dims).

**Rewards (`process_verify.py`).** `R_out = q`; `R_mult = q × coverage` with coverage = share of
worker steps verified *and* passed (unverified steps are not covered, so skipping verification
is not rewarded); `R_add = q + 0.25·coverage − 0.25·open-fail rate` (an open fail is one never
followed by a pass on the same sub-query). The silent-failure analysis puts every episode in one
of seven cells (correct/wrong × all-pass / ≥1 fail recovered / ≥1 fail open / unverified) and
reports the rates above.

**ORM (`orm.py`).** Logistic regression and a 16-unit MLP in numpy over process features
computed from a prefix of the process log (so the same model scores a finished episode and a
partial one at a decision point). Evaluated by K-fold CV over tasks (category-stratified):
AUROC, Brier against the base-rate Brier, ECE, a calibration table. Default features exclude the
task-category one-hot: with 2–3 tasks per category it memorises training tasks, the same finding
as the linear controller on this benchmark (`results/crmarena/RESULTS.md`); it stays in the
table as an ablation. Exported as `quality_model.json` (`kind: mas_orm`).

**Priced controller hook.** `orm.relabel_episodes(log, quality_model)` replaces each episode's
quality by `quality_hat`; `pace.train_offline.train_awr` then trains on reward
`quality_hat − λ·cost`, unchanged. `run.py analyze` does this on the exploration log and saves
the policy, which `run --mode controller --policy <file>` serves on-policy. Off-policy estimates
of the trained policy come from `offpolicy.py`, not from the core gate: with 2–4 decisions per
episode over 12 actions the cumulative importance ratio explodes, so the report uses
**per-decision** estimators (each logged decision point as a contextual-bandit sample with its
own single-step ratio; SNIPS and a bucketed-DM DR; the value of changing one decision while the
rest of the episode stays as logged) and one **episode-level** self-normalised IPS with each
per-step ratio clipped at 5. Every estimate is printed with its effective sample size
(Σw)²/Σw², and any estimate whose ESS is below 10% of n is flagged as unreliable.

## How to run

```bash
# pipeline check on the deterministic fake backend (no network, no keys, ~10 s):
python -m pace.experiments.mas.run run --mode all --n-tasks 50 --dry-run
python -m pace.experiments.mas.run analyze --log results/mas/dryrun/mas_fake_b2b.jsonl \
    --quality-model-out results/mas/dryrun/quality_model.json --policy-out results/mas/dryrun/policy_awr_orm.json
python -m pace.experiments.mas.run run --mode controller --policy results/mas/dryrun/policy_awr_orm.json \
    --eps 0.05 --n-tasks 50 --dry-run            # serve the trained policy on-policy, then re-run analyze
python -m pytest tests/test_mas.py -q

# the real run as it was launched (CRMArena-Pro b2b; needs scripts/setup_crmarena.sh and its .env):
python -m pace.experiments.mas.run estimate --n-tasks 50 --modes always_verify,never_verify,controller
python -m pace.experiments.mas.run run --mode all --n-tasks 50 --org b2b --cap 28.98
#   the ledger traces/spend.jsonl already held $17.98 from the single-agent collection, so this cap
#   bought $11 of MAS calls: 50 always_verify + 50 never_verify + 30 controller:uniform episodes, then
#   "STOPPED AT CAP before b2b-145 mode controller:uniform" (traces/STOPPED_AT_CAP), exit code 3.
#   Resuming the remaining 20 exploration episodes is the same command with a higher --cap.
python -m pace.experiments.mas.run analyze --log results/mas/mas_b2b.jsonl --out results/mas/RESULTS.md \
    --single-agent-log traces/crmarena_b2b_counterfactual.jsonl --prelude results/mas/interpretation_b2b.md
```

Real runs share the cumulative `traces/` ledger with `pace.collect` (`--cap` is on the ledger
total, not on the run), are resumable per `(task, mode)`, write `traces/STOPPED_AT_CAP` and exit 3
at the cap (the episode in flight is dropped and re-run on resume; its partial spend stays in the
ledger), and exit 4 when a model call fails after retries. `analyze` takes `--single-agent-log`
for the side-by-side table on the shared task ids and `--prelude` for an authored interpretation
inserted as section 0, so regenerating the tables never loses the write-up. Dry runs never touch `traces/`: their ledger and log live
under `results/mas/dryrun/`. Role tiers are flags (`--planner-tier`, `--worker-tier`,
`--verifier-tier`, `--answer-tier`); defaults are planner cheap, worker mid, verifier cheap,
answer mid. The controller mode takes `--policy uniform` (an exploration log with propensities)
or a policy JSON.

## Cost estimate: 50 tasks × 2 fixed modes (+ the exploration mode)

Per-call token assumptions come from the measured CRMArena figures in the main README (ReAct
system prompt ≈ 4.1k o200k tokens, ×1.2 for Haiku and ×1.45 for Sonnet/Opus; a mid-tier agent
turn 7,800 in / 550 out including ~400 thinking tokens at effort low; a cheap turn 6,500 / 150;
the extractor 600 / 30) and list prices in `pace/prices.py` (Haiku $1/$5, Sonnet $2/$10, Opus
$4/$20 per M tokens). The planner sees the system prompt once (no accumulated context); the
verifier sees only the question, the sub-goal, the query and ≤ 3k chars of output; the answer
call carries the full context. `python -m pace.experiments.mas.run estimate` prints:

```
always_verify (planner cheap, 2.8 worker mid, 2.8 verifier cheap, answer mid, extractor cheap):
  planner    1   × (5,200×$1 + 300×$5)/1e6   = 1   × $0.0067 = $0.0067
  worker     2.8 × (7,800×$2 + 550×$10)/1e6  = 2.8 × $0.0211 = $0.0591
  verifier   2.8 × (1,500×$1 + 60×$5)/1e6    = 2.8 × $0.0018 = $0.0050
  answer     1   × (9,000×$2 + 550×$10)/1e6  = 1   × $0.0235 = $0.0235
  extractor  1   × (600×$1 + 30×$5)/1e6      = 1   × $0.0008 = $0.0008
  per task $0.0951  → 50 tasks $4.75
never_verify (2.5 worker calls, no verifier):      per task $0.0837  → 50 tasks $4.18
controller, uniform exploration over tiers:        per task $0.0992  → 50 tasks $4.96
TOTAL 50 × 2 fixed modes = $8.93;  with the exploration mode $13.90
```

**Measured on the real run** (130 episodes, $10.97, $0.084 per episode against the $0.084–0.099
estimated): planner on Haiku 5,029 in / 279 out per call, worker on Sonnet 8,369 / 158, verifier on
Haiku 711 / 62, answer on Sonnet 12,250 / 91. The input figures match the assumptions; the output
figures are a third of the assumed 550 (effort `low` thinking is short on these turns), which is
where the small saving comes from. Per task: `always_verify` $0.0906 (planner $0.0065, workers
$0.0547, verifiers $0.0031, answer $0.0264), `never_verify` $0.0821.

Cross-check against the realized per-episode costs of the single-agent run on the same tasks
(`results/crmarena/RESULTS.md`: fixed mid $0.120, fixed cheap $0.068 per episode over ~5 turns):
a mid worker call at $0.021–0.024 is in line, and the MAS spends about as much per task as one
mid-tier single-agent episode because its planner and verifier run on Haiku. Sensitivity: at
1,500 thinking tokens per mid call the two fixed modes cost $12.95 (`--thinking-out 1500`);
at 4 worker calls per task instead of 2.8, about +$0.025 per task. A `--cap 25` covers the
three modes with margin; the ledger stops the run before the cap is crossed.

## Results (real run, 50 CRMArena-Pro b2b tasks)

Full tables and the authored interpretation: [`results/mas/RESULTS.md`](../../../results/mas/RESULTS.md)
(section 0 is the interpretation, 1–7 are generated). The dry-run pipeline check is kept as
[`results/mas/RESULTS_dryrun.md`](../../../results/mas/RESULTS_dryrun.md); **its numbers carry no
scientific content** (the fake's latent model is in `fake_backend.py`). The 130-episode process log
is `results/mas/mas_b2b.jsonl`; `quality_model.json` and `policy_awr_orm.json` are the ORM and the
AWR policy it produced.

| system (same 50 tasks) | accuracy / quality | cost/task | J (λ=2) | J (λ=5) |
|---|---|---|---|---|
| single agent, fixed cheap (Haiku) | 0.444 | $0.068 | 0.308 | 0.104 |
| single agent, fixed mid (Sonnet) | **0.608** | $0.120 | **0.368** | 0.010 |
| single agent, fixed frontier (Opus) | 0.663 | $0.164 | 0.334 | −0.159 |
| single agent, per-task oracle | 0.769 | $0.092 | 0.639 (λ-oracle) | 0.452 (λ-oracle) |
| MAS `always_verify` (planner Haiku, worker Sonnet, verifier Haiku) | 0.480 | $0.091 | 0.299 | 0.027 |
| MAS `never_verify` | 0.460 | $0.082 | 0.297 | 0.051 |
| MAS `controller:uniform` (30 tasks, exploration, mixed tiers) | 0.533 | $0.076 | 0.384 | 0.156 |

- **H3 answered first, negatively: the decomposition scores below the single mid agent (0.48 vs
  0.61) at lower cost and is dominated on the priced frontier at both λ** (by mid at λ = 2, by cheap
  at λ = 5). Paired on the 50 tasks the single mid agent wins 11 tasks and loses 5. The
  compute-matched single-agent baseline that MAS-Orchestra (arXiv 2601.14652) reports as the bar for multi-agent
  systems is not cleared here. Mechanisms visible in the log: a cheap planner fixes the plan with
  no data in hand while the single ReAct agent picks each query after seeing the last result; the
  MAS gives Sonnet fewer turns (≈2.7 worker turns + a forced answer, against ≈5); and under
  `always_verify` 47/50 episodes hit the 8-call cap because 65% of verifier calls fail and each
  fail spends a retry.
- **H1, verification bought +1 task.** Paired on the same tasks: 6 flipped wrong → right, 5
  right → wrong, net +1 (+2.0 pp, Δquality +0.019 ± 0.067 SE) for +$0.0086 per task (+10%). Every
  flipped task went through a verifier fail and a retry, in both directions.
- **The verifier is an alarm, not a judge.** Per-call pass rate 0.347. It flags 24/26 wrong
  episodes (P(flag | wrong) = 0.92; silent failures 2/50) but also 23/24 correct ones
  (P(flag | correct) = 0.96), so P(wrong | flagged) = 0.51 against a base rate of 0.52: the
  step-level pass/fail carries no information about the outcome, and `R_mult` rewards coverage
  the outcome does not reflect.
- **H2 reversed relative to the fake backend: the process carries no signal, the category carries
  most of it.** ORM on process features: out-of-fold AUROC 0.469, Brier above the base rate;
  with the category one-hot 0.831 / Brier 0.165 / ECE 0.04; the category alone gives 0.821. CV is over tasks, so this is difficulty by category that transfers to unseen tasks, as in
  the single-agent study, resting on 2–3 training tasks per category. On this benchmark the
  orchestration signal is known before the first call.
- **The controller hook is exercised, not evaluated.** 30 exploration episodes, 64 decisions:
  per-decision SNIPS of the AWR policy 0.563 [0.373, 0.701] against the behavior's 0.567 (ESS 86%);
  episode-level ESS 7/30. Serving the policy on the 50 tasks costs about $4; finding H2 argues
  against pursuing it on this benchmark.

## Limitations

- **50 tasks, 2–3 per category.** The single-agent results on this log already show that
  category features memorise; the ORM therefore excludes them by default and reports CV over
  tasks. Even so, an ORM on 100–200 episodes is a small-sample estimate; the Brier score against
  the base rate is the honest headline, not AUROC alone.
- **The verifier is an LLM of the same family as the worker** (Haiku checking Sonnet). Its
  false-alarm and miss rates are what the silent-failure table measures, not a ground truth;
  there is no human process label.
- **Outcome is CRMArena's scorer.** Exact match for most categories; F1 for the fuzzy ones, so
  "correct" thresholds quality at 0.5.
- **The process reward variants are definitions, not findings.** `R_mult` and `R_add` are
  reported so that a reader can see how the ranking of modes changes with the reward; the
  weights 0.25 are not tuned.
- **The controller hook is a smoke test on a dry run.** With 12 actions and 2–4 decisions per
  episode, importance ratios on a 100-episode exploration log are large (the table prints the
  max ratio); the on-policy served row is the measurement, and both need the real log.
- **The 8-call cap interacts with a high-fail verifier.** Each verifier fail spends a retry, so
  `always_verify` reached the cap in 47/50 episodes and had its plan truncated; a run with a
  higher cap, or a verifier that fails less, is a different system. Six of the 50 tasks are
  confidentiality tasks where a refusal is the correct answer; every system scores 0 on them.
- **Sequential sub-queries only.** The worker runs one sub-query at a time in one growing
  context; parallel workers with separate contexts would change the cost profile (the system
  prompt is re-sent per call) and are not implemented.
- **CRMArena-Pro is CC BY-NC 4.0** (research use), vendored by `scripts/setup_crmarena.sh`,
  not copied.
