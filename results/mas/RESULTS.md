# MAS results on CRMArena-Pro b2b

## 0. Interpretation

Real run on CRMArena-Pro b2b: 50 tasks (2–3 per category, the same 50 task ids as the single-agent
counterfactual log in `traces/crmarena_b2b_counterfactual.jsonl`), planner Haiku 4.5, worker and
answer Sonnet 5.5, verifier Haiku 4.5. 130 episodes: 50 `always_verify`, 50 `never_verify`, 30
`controller:uniform` before the ledger cap stopped the run (`--cap 28.98` on a ledger that already
held $17.98; the MAS spent $10.97, $0.084 per episode). Five findings, in order of importance.

**1. Headline, negative: the planner / worker / verifier decomposition scores below the single
mid-tier agent on the same tasks, at lower cost, and is dominated on the priced frontier.**
`always_verify` reaches accuracy 0.480 at $0.0906 per task and `never_verify` 0.460 at $0.0821,
against the single Sonnet ReAct agent at **0.608 at $0.120** and the single Haiku agent at 0.444 at
$0.068 (section 7). Paired on the 50 tasks, the single mid agent is right on 11 tasks the MAS gets
wrong and wrong on 5 the MAS gets right (net −6 for the MAS). The MAS sits between cheap and mid in
both quality and cost, and J = quality − λ·cost says it is not a good trade at either price: at
λ = 2 per dollar the single mid agent has J 0.368 against 0.299 (`always_verify`) and 0.297
(`never_verify`); at λ = 5 the single cheap agent has 0.104 against 0.027 and 0.051. The per-task
single-agent oracle is at 0.769 / $0.092. This is the compute-matched single-agent comparison that
MAS-Orchestra (arXiv 2601.14652) reports as the baseline a multi-agent system has to beat, and here the decomposition
does not beat it. Three mechanisms are visible in the log: (i) the plan is fixed up front by a
cheap model with no data in hand, while the single ReAct agent on Sonnet chooses each next query
after seeing the previous result; (ii) `never_verify` runs about 2.7 worker turns and a forced
answer, fewer Sonnet turns than the single agent's ~5, so the MAS is not compute-matched upward
either, it simply gives the strong model less room; (iii) under `always_verify` 47 of 50 episodes
hit the 8-role-call cap because 65% of verifier calls fail and each fail spends a retry, so the
plan is truncated and the answer forced. The categories where the MAS loses outright are the ones
that need the agent to notice something in the data (`invalid_config` 0/2 against the single mid
agent's 2/2; `lead_qualification`, `quote_approval`, `sales_cycle_understanding` 0/2 against 1/2
each). The six confidentiality tasks (`privacy_rejection`, a refusal is the correct answer) score
0 for every system here, so the ceiling on these 50 tasks is 0.88.

**2. Verification bought +1 task for +10% cost, and the paired table shows it is noise.** On the
same 50 tasks, verification flipped 6 tasks wrong → right and 5 right → wrong; 18 were right and 21
wrong under both (section 2b). Net +1 task (+2.0 pp, paired Δquality +0.019 ± 0.067 SE) for
+$0.0086 per task. All 11 flipped tasks went through at least one verifier fail and a retry: the
retry is what moves the outcome, in both directions.

**3. The verifier as built is an alarm, not a judge: high recall, no precision.** Its per-call pass
rate is 0.347 (147 calls, 10 of them structural fails on query errors). At the episode level it
flags 24 of the 26 wrong episodes, P(flag | wrong) = 0.92, so the silent-failure rate (wrong with
every check passed) is only 2/50. But it also flags 23 of the 24 correct episodes, P(flag | correct)
= 0.96 (46% of all episodes are correct-with-a-fail; 20 of those 23 fails were never resolved). A
flag therefore carries no information about the outcome: P(wrong | flagged) = 24/47 = 0.51 against
a base rate of 0.52. The step-level check (well-formed, non-empty, consistent with the sub-goal)
does not predict whether the final synthesis will be right, because most sub-results it fails were
not needed for the answer and most it passes do not determine it. What process verification
"catches that outcome scoring misses" is therefore real but not useful as a reward here: the
`R_mult` ranking (0.203 vs 0 by construction) rewards coverage the outcome does not reflect.

**4. The orchestration reward model finds the reverse of the fake backend: process features carry
no signal, the task category carries most of it.** Out of fold over tasks, the logistic ORM on
process features has AUROC 0.469 and a Brier score above the base rate (0.281 vs 0.250), with or
without the verifier features (0.451 without). Adding the 22-way category one-hot lifts it to
AUROC 0.831, Brier 0.165, ECE 0.04; the category alone, with no process feature, gives
AUROC 0.821, Brier 0.166. The CV is over tasks, so this is difficulty that varies by category and
transfers to unseen tasks of the same category (the six confidentiality tasks and `knowledge_qa`
at 0 for every system, `handle_time` and `transfer_count` near 1), not memorisation of task ids.
It is the same structure the single-agent study found when it chose the tier (`results/crmarena/
RESULTS.md`), and it rests on 2–3 training tasks per category, so the per-category estimates are
thin. The consequence for the thesis: on this benchmark the right orchestration signal is
known before the first call (the category), not learned from the trace of the process.

**5. The priced-controller hook is exercised, not evaluated.** `controller:uniform` ran 30
episodes (0.533 at $0.0758, mixed tiers; an exploration log, not a tuned policy, and its 30 tasks
are an easier draw: the single mid agent scores 0.672 on them against 0.608 on all 50). The AWR
policy trained on the ORM-relabelled log changes nothing detectable per decision (SNIPS outcome
0.563 [0.373, 0.701] against the behavior's 0.567; ESS 55 of 64 decisions, 86%); the episode-level
estimate has ESS 7 of 30 (23%) and a CI of [0.24, 0.85]. Nothing can be concluded about the
policy from 30 exploration episodes; serving it on the 50 tasks costs about $4 and is the next
step if the hook is worth pursuing, which finding 4 argues against: with the outcome predicted by
the category rather than the process, a per-decision controller has nothing to react to beyond
the tier choice the single-agent study already covers.

Sections 1–7 below are generated by `run.py analyze`; section 0 is authored and passed in with `--prelude`.

Logs: results/mas/mas_b2b.jsonl. 130 episodes over 50 tasks, backend `crmarena`, host harness `crmarena-pro-v1`. Role tiers: planner = cheap, worker = mid, verifier = cheap, answer = mid. Cost/task is the sum of the role calls (planner, worker, verifier, answer); the outcome evaluator's extractor call is listed separately. Quality is the host harness's own score (CRMArena's evaluator); accuracy thresholds it at 0.5. R_mult = quality × verified-pass coverage, R_add = quality + 0.25·coverage − 0.25·open-fail rate (see `process_verify.py`).

## 1. Per-mode quality, cost and process rewards

| mode | n | accuracy (q ≥ 0.5) | quality | cost/task (USD) | calls/task | verifier calls/task | pass rate | coverage | R_mult | R_add | J (λ=2) | J (λ=5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_verify | 50 | 0.480 | 0.480 | 0.0906 | 7.88 | 2.94 | 0.347 | 0.470 | 0.203 | 0.465 | 0.299 | 0.027 |
| never_verify | 50 | 0.460 | 0.461 | 0.0821 | 4.74 | 0.00 | – | 0.000 | 0.000 | 0.461 | 0.297 | 0.051 |
| controller:uniform | 30 | 0.533 | 0.535 | 0.0758 | 5.00 | 0.43 | 0.154 | 0.028 | 0.017 | 0.485 | 0.384 | 0.156 |

Cost split per task by role:

| mode | planner | worker | verifier | answer | eval (extractor) | worker tiers used |
|---|---|---|---|---|---|---|
| always_verify | $0.0065 | $0.0547 | $0.0031 | $0.0264 | $0.0002 | mid 100% |
| never_verify | $0.0065 | $0.0515 | $0.0000 | $0.0241 | $0.0003 | mid 100% |
| controller:uniform | $0.0156 | $0.0418 | $0.0004 | $0.0181 | $0.0003 | cheap 28%, mid 57%, frontier 15% |

## 2. Silent-failure analysis: what process verification catches that the outcome misses, and vice versa

Rates are over episodes that had at least one verifier call (the never_verify row has none by construction).

| mode | n | verified eps | verifier calls | pass rate | silent failure: wrong & all pass | caught: wrong & ≥1 fail | correct & ≥1 fail (open / recovered) | P(flag ∣ wrong) | P(wrong ∣ all pass) |
|---|---|---|---|---|---|---|---|---|---|
| always_verify | 50 | 50 | 147 | 0.347 | 2 (0.040) | 24 (0.480) | 20 / 3 (0.460) | 0.923 | 0.667 |
| never_verify | 50 | 0 | 0 | – | 0 (–) | 0 (–) | 0 / 0 (–) | – | – |
| controller:uniform | 30 | 9 | 13 | 0.154 | 0 (0.000) | 3 (0.333) | 5 / 1 (0.667) | 1.000 | – |

## 2b. Paired per-task comparison: what verification bought

50 tasks run under both `always_verify` and `never_verify` (same task ids, same seeds).

| | `never_verify` correct | `never_verify` wrong |
|---|---|---|
| `always_verify` correct | 18 | **6** (wrong → right) |
| `always_verify` wrong | **5** (right → wrong) | 21 |

Net: 6 − 5 = **+1 task(s)** of 50; paired mean Δquality +0.019 ± 0.067 (SE), paired mean Δcost $+0.0086/task.

Flipped wrong → right under `always_verify`:

- `b2b-1041` (conversion_rate_comprehension): verifier 3 fail / 3 calls, 1 retry, 8 role calls
- `b2b-1086` (best_region_identification): verifier 2 fail / 2 calls, 1 retry, 6 role calls
- `b2b-1755` (lead_routing): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-176` (monthly_trend_analysis): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-1975` (named_entity_disambiguation): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-995` (conversion_rate_comprehension): verifier 3 fail / 3 calls, 1 retry, 8 role calls

Flipped right → wrong under `always_verify`:

- `b2b-1139` (best_region_identification): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-1397` (policy_violation_identification): verifier 1 fail / 3 calls, 1 retry, 8 role calls
- `b2b-1557` (case_routing): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-1708` (lead_routing): verifier 2 fail / 3 calls, 1 retry, 8 role calls
- `b2b-429` (sales_insight_mining): verifier 2 fail / 3 calls, 1 retry, 8 role calls


## 3. Orchestration reward model: process features → P(correct)

Pooled over all modes: 130 episodes over 50 tasks, base rate P(correct) = 0.485. 5-fold CV over tasks (category-stratified; a task's episodes never straddle folds); out-of-fold predictions. Default features: counts of role calls, verifier verdicts and pass rate, open failures, replans and retries, observation kinds, tiers used, cost. The '− verifier features' row drops n_verify/4, n_pass/4, n_fail/4, pass_rate, has_verify, open_fails, n_retry; the '+ category one-hot' row adds the 22-way task category.

| model | features | AUROC | Brier | Brier (base rate) | ECE (5 bins) |
|---|---|---|---|---|---|
| logreg | process features (default) | 0.469 | 0.281 | 0.250 | 0.114 |
| logreg | process − verifier features | 0.451 | 0.280 | 0.250 | 0.140 |
| logreg | process + category one-hot | 0.831 | 0.165 | 0.250 | 0.040 |
| mlp | process features (default) | 0.456 | 0.263 | 0.250 | 0.053 |
| mlp | process − verifier features | 0.430 | 0.261 | 0.250 | 0.045 |
| mlp | process + category one-hot | 0.831 | 0.164 | 0.250 | 0.035 |

Calibration of the default logistic ORM (out-of-fold):

| predicted bin | n | mean predicted | observed correct |
|---|---|---|---|
| [0.0, 0.2) | 4 | 0.155 | 0.500 |
| [0.2, 0.4) | 40 | 0.330 | 0.500 |
| [0.4, 0.6) | 65 | 0.506 | 0.477 |
| [0.6, 0.8) | 18 | 0.672 | 0.500 |
| [0.8, 1.0) | 3 | 0.892 | 0.333 |

Exported `results/mas/quality_model.json` (default logistic ORM on all 130 episodes; in-sample AUROC 0.669, Brier 0.227; out-of-fold above). Consumers: `orm.relabel_episodes` → `pace.train_offline.train_awr` / `pace.gate.estimate`.

Largest standardised logistic weights: n_rows/3 +0.22, n_error/3 -0.22, cost_so_far -0.19, pass_rate -0.17, answered -0.17, worker_tier=cheap/4 +0.14, n_retry +0.12, last_obs=empty -0.12.

## 4. Priced comparison: J = quality − λ·cost, outcome vs ORM quality

| mode | quality (outcome) | quality_hat (ORM) | cost/task | J_out (λ=2) | J_orm (λ=2) | J_out (λ=5) | J_orm (λ=5) |
|---|---|---|---|---|---|---|---|
| always_verify | 0.480 | 0.481 | 0.0906 | 0.299 | 0.300 | 0.027 | 0.028 |
| never_verify | 0.461 | 0.459 | 0.0821 | 0.297 | 0.295 | 0.051 | 0.048 |
| controller:uniform | 0.535 | 0.533 | 0.0758 | 0.384 | 0.381 | 0.156 | 0.154 |

## 5. Priced controller hook: AWR on the ORM-relabelled exploration log

AWR+IPS on the 30-episode uniform exploration log (64 logged decisions), reward quality_hat − 2·cost; saved to `results/mas/policy_awr_orm.json`.

Per-decision estimates treat each logged decision point as a contextual-bandit sample with its own single-step importance ratio (max 10.2 here): the value of changing one decision while the rest of the trajectory stays as logged, scored on quality_hat, on the outcome, and on the cost-to-go from the decision (the policy-independent prefix, planner + first worker, is $0.0214/task on top). The episode-level row is self-normalised IPS of the full policy with each per-step ratio clipped at 5. ESS = (Σw)²/Σw²; an estimate with ESS below 10% of n is flagged.

| policy | estimator | unit | n | ESS | quality_hat [90% CI] | outcome quality [90% CI] | cost [90% CI] |
|---|---|---|---|---|---|---|---|
| uniform behavior | on-policy mean | decision | 64 | 64 | 0.545 | 0.567 | cost-to-go 0.0447 |
| AWR (ORM reward) | per-decision SNIPS | decision | 64 | 55.0 | 0.565 [0.516, 0.619] | 0.563 [0.373, 0.701] | cost-to-go 0.0427 [0.0320, 0.0552] |
| AWR (ORM reward) | per-decision DR (bucket DM) | decision | 64 | 55.0 | 0.597 [0.538, 0.637] | 0.505 [0.307, 0.733] | cost-to-go 0.0303 [0.0170, 0.0448] |
| uniform behavior | on-policy mean | episode | 30 | 30 | 0.533 | 0.535 | total 0.0758 |
| AWR (ORM reward) | episode SN-IPS, step clip 5 | episode | 30 | 7.0 | 0.555 [0.472, 0.634] | 0.587 [0.235, 0.853] | total 0.0989 [0.0785, 0.1263] |

per-decision: ESS 55.0 of 64 (86%); episode-level: ESS 7.0 of 30 (23%).

Greedy action of the trained policy at the logged decision points: ANSWER@cheap 25%, VERIFY@mid 12%, VERIFY@cheap 9%, REPLAN@mid 8%, ANSWER@mid 8%. Serve it with `run --mode controller --policy <file>` to measure it on-policy.

## 6. Generated summary

- **Outcome vs process.** always_verify reaches accuracy 0.480 at $0.0906/task against never_verify 0.460 at $0.0821/task; the verifier costs $+0.0086/task. Priced: J(λ=2) 0.299 vs 0.297, J(λ=5) 0.027 vs 0.051.
- **Silent failures.** Among always_verify episodes, 0.040 are wrong with every verifier passing (process verification missed them) and 0.480 are wrong with at least one fail (caught); 0.460 are correct despite a fail (0.400 with the fail still open: false alarms or steps the answer did not need). P(flagged | wrong) = 0.923, P(wrong | all pass) = 0.667: the first is what process verification adds over outcome-only scoring, the second is the residual error a clean process still carries.
- **controller:uniform.** accuracy 0.533 at $0.0758/task: a uniform policy over 12 actions is the exploration log (propensities logged) the ORM and the AWR hook train on, not a candidate.
- **ORM.** Out of fold, the default logistic ORM has AUROC 0.469 and Brier 0.281 against a base-rate Brier of 0.250 (ECE 0.114). Dropping the verifier features moves AUROC to 0.451; adding the category one-hot moves it to 0.831. The category one-hot raises out-of-fold AUROC, and the CV is over tasks, so this is task difficulty that varies by category and transfers across tasks, not memorisation; the process features add nothing beyond it here. With 2–3 training tasks per category the per-category estimate is thin.
- **What this does not show.** Whether a priced controller over {VERIFY, SKIP_VERIFY, REPLAN, ANSWER} × tier beats the fixed baselines at equal budget needs the real run plus an on-policy evaluation of the trained policy; the per-decision estimates on the exploration log value one changed decision at a time and the clipped episode-level estimate is biased by construction.

## 7. The comparison that matters: decomposition vs the single agent on the same tasks

Single-agent rows are exact lookups of `traces/crmarena_b2b_counterfactual.jsonl` on the 50 MAS task ids (quality is the same CRMArena score; the single-agent continuation is CONTINUE on one tier until the agent answers, ≤ 12 turns). MAS rows are the modes above; `controller:uniform` ran on fewer tasks and is listed with its own single-agent reference below.

| system | tasks | accuracy / quality | cost/task (USD) | J (λ=2) | J (λ=5) |
|---|---|---|---|---|---|
| single agent, fixed cheap | 50 | 0.444 | 0.0680 | 0.308 | 0.104 |
| single agent, fixed mid | 50 | 0.608 | 0.1196 | 0.368 | 0.010 |
| single agent, fixed frontier | 50 | 0.663 | 0.1642 | 0.334 | -0.159 |
| single agent, per-task oracle (max quality, cheapest) | 50 | 0.769 | 0.0916 | 0.586 | 0.311 |
| single agent, per-task oracle (argmax q − 2·c) | 50 | 0.764 | 0.0624 | 0.639 | 0.452 |
| single agent, per-task oracle (argmax q − 5·c) | 50 | 0.764 | 0.0624 | 0.639 | 0.452 |
| MAS `always_verify` | 50 | 0.480 | 0.0906 | 0.299 | 0.027 |
| MAS `never_verify` | 50 | 0.461 | 0.0821 | 0.297 | 0.051 |

- `controller:uniform` (30 episodes on 30 of the 50 tasks): 0.535 at $0.0758; single mid on those 30 tasks: 0.672 at $0.1167, single cheap 0.568 at $0.0616, single frontier 0.738 at $0.1683.
- Paired `always_verify` vs single mid on the same 50 tasks: both right 19, MAS right / mid wrong 5, MAS wrong / mid right 11, both wrong 15 (net -6 for the MAS).
- Paired `never_verify` vs single mid on the same 50 tasks: both right 19, MAS right / mid wrong 4, MAS wrong / mid right 11, both wrong 16 (net -7 for the MAS).
