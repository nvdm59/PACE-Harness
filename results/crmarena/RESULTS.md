# Results on CRMArena-Pro b2b (50 tasks × 3 tiers)

**Data.** `traces/crmarena_b2b_counterfactual.jsonl` (committed): 50 b2b tasks (2–3 per category,
all 22 categories), each run on all three tiers (cheap = Haiku 4.5, mid = Sonnet 5.5, frontier =
Opus 5.5) with the fixed continuation "CONTINUE on that tier until the agent answers". The log
records propensity 1/3 on the first step and 1 after it. 150 episodes, $17.98 of model spend
(ledger snapshot `traces/crmarena_b2b_counterfactual.spend.jsonl`). Nothing below makes an API, Salesforce or LLM-judge call. Part A runs on
CPU in about 15 s and is byte-for-byte deterministic (`python scripts/crmarena_offline.py` writes
`tables.md`, `results.json` and `policies/`; the Part A tables below are `tables.md` verbatim).
Part B ran once on Modal (one A100-40GB container); its artifacts are `results/llm/*_crm_phase2.*`.

**Decision learned.** The first-step tier, a contextual bandit over three arms whose reward table
is fully observed. The policy is `tier | step-0 features`: the 43-dim CRMArena vector, which at
step 0 is bias, the category one-hot, org and the context bucket. Reward is `quality − λ·cost_usd`.
λ is either fixed (2, 5, 10 per dollar) or learned against a budget of B = $0.10 per task. The
per-task oracle reaches quality 0.769 at $0.092.

## Interpretation

- **On 50 tasks, the learned linear controllers do not generalize (a negative result).** On
  held-out tasks, the AWR+IPS policy as served (stochastic) has lower J than the best fixed arm at
  every λ: 0.350 vs mid 0.368 at λ = 2, 0.045 vs cheap 0.104 at λ = 5, and −0.414 vs cheap −0.236
  at λ = 10. Its greedy version and the per-category plug-in rule match or beat the best fixed arm
  by at most 0.08: 0.384 / 0.415 vs 0.368, 0.186 / 0.161 vs 0.104, and −0.273 / −0.214 vs −0.236.
  The fold SDs (0.05–0.3) are as large as the gaps between rows, so none of these orderings is
  established.
- **The features memorize.** With 2–3 tasks per category, a category one-hot fits the training
  tasks and transfers little. AWR's J is 0.475 in-sample against 0.350 held-out at λ = 2. The
  constrained policy meets B in-sample (0.669 at $0.099). Held out, it overspends (0.553 at $0.131)
  and is dominated by the uniform behavior (0.571 at $0.117).
- **Held-out cost is noisy.** Even the context-free arm mixture tuned to B on 40 tasks costs $0.115
  on the held-out 10. Per-task cost variance alone moves a 10-task mean by more than the $0.005
  tolerance.
- **The headroom is per task, not per category.** The per-task oracle reaches 0.769 at $0.092, which
  beats every arm on both quality and cost. What is missing is an observable that predicts which
  tier a given task needs. The category does not.
- **The gate is the safe side of an underpowered test.** With one logged arm per task (50
  episodes), over 200 redraws DR estimates quality with RMSE 0.048 and cost with RMSE $0.011. That
  is half the RMSE of IPS for quality, and its 90% CIs cover the exact value 99% / 92% of the time
  (the direct model is cross-fitted by task; fitted in-sample it covers 96% / 94% with a −0.006
  quality bias, a difference inside the noise at n = 50). The DR cost CI is about $0.046 wide,
  though, so a candidate whose exact cost is $0.0987 is PROMOTEd in only 6% of 200 redraws. CI width
  scales with 1/√n: certifying a $0.005 margin needs a CI near $0.01, which is roughly 20× the
  tasks (about 1,000), a back-of-the-envelope figure, not a measurement.
- **Repricing hits cost, as in the sim.** Frontier ×1.5 and cheap ×3 push the frozen policy over
  budget, by +$0.029 and +$0.035 per task. Retraining on the repriced log (costs recomputed from
  token counts) restores the budget in-sample: priced-BWT is −0.040 / −0.048 and recovery
  +0.069 / +0.091. Held out, the retrained policy still overspends (about $0.125). Recovery is
  −0.012 for frontier ×1.5, so the gate's REVERT on all three events is the correct call.
- **Harness-version drift (v1 → v2) cannot be measured on this log.** It holds a single harness
  version, and nothing here re-runs the harness.

## Part A: linear controller on the CRMArena-Pro b2b log (50 tasks × 3 tiers)

### 1–2. Cross-validated first-step tier policies

5-fold CV over task ids (category-stratified, seed 0): train on 40 tasks, evaluate on the held-out 10. All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.

#### λ = 2 per dollar (J = quality − 2·cost)

| policy | quality | cost/task (USD) | J (λ=2) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.571 | 0.1106 | 0.350 [0.047] | NO |
| AWR+IPS linear (greedy) | 0.584 | 0.1000 | 0.384 [0.076] | yes |
| per-category plug-in rule (train-fold argmax) | 0.584 | 0.0846 | 0.415 [0.066] | yes |
| uniform behavior | 0.571 | 0.1173 | 0.337 [0.066] | NO |
| fixed cheap | 0.444 | 0.0680 | 0.308 [0.179] | yes |
| fixed mid | 0.608 | 0.1196 | 0.368 [0.092] | NO |
| fixed frontier | 0.663 | 0.1642 | 0.334 [0.133] | NO |
| per-task oracle (argmax q - 2c) | 0.764 | 0.0624 | 0.639 [0.056] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.475 (held-out: 0.350).

#### λ = 5 per dollar (J = quality − 5·cost)

| policy | quality | cost/task (USD) | J (λ=5) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.553 | 0.1016 | 0.045 [0.126] | yes |
| AWR+IPS linear (greedy) | 0.584 | 0.0797 | 0.186 [0.126] | yes |
| per-category plug-in rule (train-fold argmax) | 0.584 | 0.0846 | 0.161 [0.104] | yes |
| uniform behavior | 0.571 | 0.1173 | -0.015 [0.164] | NO |
| fixed cheap | 0.444 | 0.0680 | 0.104 [0.227] | yes |
| fixed mid | 0.608 | 0.1196 | 0.010 [0.124] | NO |
| fixed frontier | 0.663 | 0.1642 | -0.159 [0.317] | NO |
| per-task oracle (argmax q - 5c) | 0.764 | 0.0624 | 0.452 [0.080] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.205 (held-out: 0.045).

#### λ = 10 per dollar (J = quality − 10·cost)

| policy | quality | cost/task (USD) | J (λ=10) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.530 | 0.0943 | -0.414 [0.246] | yes |
| AWR+IPS linear (greedy) | 0.524 | 0.0797 | -0.273 [0.272] | yes |
| per-category plug-in rule (train-fold argmax) | 0.544 | 0.0758 | -0.214 [0.154] | yes |
| uniform behavior | 0.571 | 0.1173 | -0.601 [0.340] | NO |
| fixed cheap | 0.444 | 0.0680 | -0.236 [0.308] | yes |
| fixed mid | 0.608 | 0.1196 | -0.588 [0.298] | NO |
| fixed frontier | 0.663 | 0.1642 | -0.980 [0.633] | NO |
| per-task oracle (argmax q - 10c) | 0.704 | 0.0556 | 0.148 [0.168] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): -0.243 (held-out: -0.414).

#### Constrained: budget B = $0.10 per task, λ learned by dual ascent

J uses each fold's own learned λ̄ (mean over folds 1.10 per dollar; per fold 1.66, 0.37, 0.28, 1.64, 1.51).

| policy | quality | cost/task (USD) | J (fold λ̄) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| constrained cgrpo-lookup linear (served: iterate mix) | 0.553 | 0.1306 | 0.435 [0.108] | NO |
| constrained cgrpo-lookup linear (greedy of the mix) | 0.542 | 0.1505 | 0.409 [0.160] | NO |
| uniform behavior | 0.571 | 0.1173 | 0.461 [0.078] | NO |
| fixed cheap | 0.444 | 0.0680 | 0.377 [0.151] | yes |
| fixed mid | 0.608 | 0.1196 | 0.487 [0.189] | NO |
| fixed frontier | 0.663 | 0.1642 | 0.520 [0.069] | NO |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.507 | 0.1146 | 0.399 [0.095] | NO |
| per-task oracle (max quality, cheapest among ties) | 0.769 | 0.0916 | 0.702 [0.075] | yes |

Constrained policy in-sample on its 40 training tasks (mean over folds): quality 0.669, cost $0.0994.

### 3. DR gate on the real log

Candidate: the constrained cgrpo-lookup policy trained on all 50 tasks (λ̄ = 1.36/USD). Exact value on the 50 tasks (all arms logged): **quality 0.665, cost $0.0987**. Uniform behavior exact: quality 0.571, cost $0.1173. Gate: eps_q = 0.03, budget $0.10 + tol $0.005.

### (a) Realized behavior log: one uniformly drawn arm per task (50 episodes, seed 0)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.627 [0.528, 0.724] | 0.1262 [0.0894, 0.1819] |
| candidate | IPS | 0.767 [0.597, 0.962] | 0.1104 [0.0884, 0.1362] |
| candidate | SNIPS | 0.742 [0.642, 0.824] | 0.1067 [0.0860, 0.1338] |
| candidate | DR | 0.740 [0.627, 0.855] | 0.0960 [0.0750, 0.1200] |

criteria: DR-quality lower CI 0.627 >= q_prev - eps = 0.627 (incumbent on this log) - 0.03 = 0.597 [ok];  DR-cost upper CI 0.1200 <= budget + tol = 0.1050 [FAIL]
(n = 50 logged episodes over 50 tasks, max cumulative ratio 2.4)

**DECISION: REVERT**

Estimator error vs exact on this draw: quality IPS +0.102, SNIPS +0.076, DR +0.075; cost IPS +0.0116, SNIPS +0.0079, DR -0.0027.

### (b) Full counterfactual log: all three arms per task (150 episodes)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.571 [0.505, 0.640] | 0.1173 [0.1005, 0.1353] |
| candidate | IPS | 0.665 [0.569, 0.751] | 0.0987 [0.0867, 0.1122] |
| candidate | SNIPS | 0.665 [0.569, 0.751] | 0.0987 [0.0867, 0.1122] |
| candidate | DR | 0.665 [0.569, 0.751] | 0.0987 [0.0867, 0.1122] |

criteria: DR-quality lower CI 0.569 >= q_prev - eps = 0.571 (incumbent on this log) - 0.03 = 0.541 [ok];  DR-cost upper CI 0.1122 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 50 tasks, max cumulative ratio 2.4)

**DECISION: REVERT**

With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value (the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure task-sampling uncertainty from 50 tasks.

(c) The same gate over 200 independent redraws of the realized behavior log (one uniform arm per task each time; bootstrap 200 per redraw):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.011 | 0.094 | 0.338 | 94% |
| quality | SNIPS | +0.001 | 0.048 | 0.238 | 100% |
| quality | DR | +0.004 | 0.048 | 0.256 | 99% |
| cost | IPS | +0.0001 | 0.0114 | 0.0414 | 92% |
| cost | SNIPS | -0.0009 | 0.0102 | 0.0381 | 92% |
| cost | DR | -0.0018 | 0.0114 | 0.0461 | 92% |

PROMOTE in 12/200 redraws (6%).

### 4. Repricing drift (no new calls)

Frozen policy: the all-50 constrained policy from the gate section (λ̄ = 1.36/USD; J = quality − 1.36·cost). Costs after each event are recomputed from the logged token counts (spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized log (seed 0), with the quality bar recorded at the frozen policy's promotion (0.740).

In-sample (all 50 tasks, exact lookups):

| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | (iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | gate on retrained (exact q / cost of candidate) |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.665 / $0.0987 / 0.531 | 0.665 / $0.1280 / 0.491 (NO budget) | 0.696 / $0.0997 / 0.560 (yes budget) | -0.040 | +0.069 | REVERT (DR q 0.793 [0.649, 0.929], DR cost $0.0951 [0.0553, 0.1264]; exact 0.696 / $0.0997) |
| mid ×0.5 | 0.665 / $0.0987 / 0.531 | 0.665 / $0.0874 / 0.547 (yes budget) | 0.706 / $0.0972 / 0.574 (yes budget) | +0.015 | +0.027 | REVERT (DR q 0.742 [0.603, 0.862], DR cost $0.0976 [0.0720, 0.1285]; exact 0.706 / $0.0972) |
| cheap ×3 | 0.665 / $0.0987 / 0.531 | 0.665 / $0.1341 / 0.483 (NO budget) | 0.715 / $0.1040 / 0.574 (yes budget) | -0.048 | +0.091 | REVERT (DR q 0.743 [0.565, 0.907], DR cost $0.1039 [0.0713, 0.1378]; exact 0.715 / $0.1040) |

Held-out (5-fold CV; frozen and retrained both trained on the 40 training tasks of each fold, evaluated on its 10 held-out tasks; J at each fold's frozen λ̄; means over folds):

| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J | priced-BWT | recovery | (iii) within budget |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.553 / $0.1306 / 0.435 | 0.553 / $0.1679 / 0.415 | 0.524 / $0.1252 / 0.402 | -0.020 | -0.012 | NO |
| mid ×0.5 | 0.553 / $0.1306 / 0.435 | 0.553 / $0.1112 / 0.465 | 0.592 / $0.1209 / 0.492 | +0.030 | +0.027 | NO |
| cheap ×3 | 0.553 / $0.1306 / 0.435 | 0.553 / $0.1649 / 0.400 | 0.627 / $0.1251 / 0.505 | -0.035 | +0.105 | NO |

Harness-version drift (v1 → v2) cannot be measured on this log: it holds a single harness version (crmarena-pro-v1) and nothing here re-runs the harness.

### Gate after frontier ×1.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.740 [0.627, 0.855] | 0.1236 [0.0912, 0.1564] |
| candidate | IPS | 0.826 [0.558, 1.054] | 0.1180 [0.0847, 0.1528] |
| candidate | SNIPS | 0.791 [0.671, 0.889] | 0.1129 [0.0915, 0.1393] |
| candidate | DR | 0.793 [0.649, 0.929] | 0.0951 [0.0553, 0.1264] |

criteria: DR-quality lower CI 0.649 >= q_prev - eps = 0.740 (recorded at incumbent's promotion) - 0.03 = 0.710 [FAIL];  DR-cost upper CI 0.1264 <= budget + tol = 0.1050 [FAIL]
(n = 50 logged episodes over 50 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

### Gate after mid ×0.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.740 [0.627, 0.855] | 0.0843 [0.0615, 0.1059] |
| candidate | IPS | 0.805 [0.582, 1.061] | 0.1176 [0.0838, 0.1631] |
| candidate | SNIPS | 0.736 [0.606, 0.836] | 0.1075 [0.0807, 0.1462] |
| candidate | DR | 0.742 [0.603, 0.862] | 0.0976 [0.0720, 0.1285] |

criteria: DR-quality lower CI 0.603 >= q_prev - eps = 0.740 (recorded at incumbent's promotion) - 0.03 = 0.710 [FAIL];  DR-cost upper CI 0.1285 <= budget + tol = 0.1050 [FAIL]
(n = 50 logged episodes over 50 tasks, max cumulative ratio 2.8)

**DECISION: REVERT**

### Gate after cheap ×3: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.740 [0.627, 0.855] | 0.1307 [0.0998, 0.1675] |
| candidate | IPS | 0.786 [0.520, 1.111] | 0.1158 [0.0804, 0.1588] |
| candidate | SNIPS | 0.735 [0.593, 0.866] | 0.1084 [0.0878, 0.1309] |
| candidate | DR | 0.743 [0.565, 0.907] | 0.1039 [0.0713, 0.1378] |

criteria: DR-quality lower CI 0.565 >= q_prev - eps = 0.740 (recorded at incumbent's promotion) - 0.03 = 0.710 [FAIL];  DR-cost upper CI 0.1378 <= budget + tol = 0.1050 [FAIL]
(n = 50 logged episodes over 50 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**


## Part B: phase-2 LLM controller (Qwen3-1.7B, lookup mode, fold 0)

This was one Modal run in one A100-40GB container: 574 s of wall clock inside the container,
about 10 GPU-minutes and about $0.35 at list price. The stages were zero-shot decoding, SFT
for 150 steps (63 s) on 840 (training task, λ ∈ 0..20/USD) → category-oracle pairs from the 40
training tasks of fold 0, then GRPO for 100 steps (403 s; 16 generations, temperature 1.2) with
the learned λ shown in the prompt, rounded to 1/USD. Logs and tables are in `results/llm/*_crm_phase2.*`
(`train_log`, `run_meta`, `sft_report`, `eval_results`, `eval_table`). The adapters stay on the
`pace-llm` volume under `crm_phase2/`. The run was not repeated.

What happened:

- **The SFT policy ignores the price.** Greedy agreement with the category-oracle labels is 90%
  on training prompts and 94% on held-out prompts (`sft_report_crm_phase2.json`). The tier the
  model picks for each held-out task is identical at every shown λ from 0 to 20. It learned the
  per-category rule for λ ≥ 1 (the majority label over the grid) and not the λ = 0 one. At λ = 0
  that rule is the max-quality rule, which costs $0.124 on the training tasks.
- **The budget was slack, so the dual price collapsed.** The SFT policy costs about $0.075 per task,
  under B = $0.10. λ fell from λ0 = 1.66 to 0 by step 17 and stayed there (`train_log_crm_phase2.jsonl`).
  λ̄ = 0, so the eval prompts show λ = 0 and J equals quality.
- **Temperature escalation never triggered.** The degenerate-group rate averaged 39% and never
  stayed above the 60% threshold for 20 consecutive steps. Unlike phase1c on the sim, there was
  within-group variety to learn from.
- **GRPO moved the training-set greedy policy a little and left the held-out one unchanged.**
  Training quality went from 0.705 to 0.730 at about $0.075. On the held-out tasks it stays at
  0.700 / $0.073, identical to SFT.
- **The held-out numbers rest on 10 tasks.** On those 10, the LLM's 0.700 at $0.073 ties fixed
  frontier on quality at half its cost. It beats the fold-0 linear policy (0.590 / $0.143) and
  its own λ = 0 SFT target (0.500 / $0.135). But one task is worth 0.1 of quality here, and in
  Part A's CV the same category rule at λ ≥ 1 averages 0.584 / $0.085 over all five folds. Fold 0's
  held-out tasks happen to favour it. Treat this row as a single draw, not an improvement over
  the linear controller.

Run `crm_phase2` (model Qwen/Qwen3-1.7B; SFT 150 steps on per-category oracle labels at λ ∈ 0..20/USD from the 40 training tasks of fold 0, then GRPO lookup mode 100 steps, 16 generations, temperature 1.2; dual step lr 2.0, kp 5.0, ema 0.95, λ0 1.664241986523554; β 0.05; entropy 0.01). Budget B = $0.10; 'within budget' = cost <= B + $0.005. λ̄ = 0.000/USD (mean learned λ over the last 50 GRPO steps; final 0.000); the LLM prompts show λ=0 (λ̄ rounded to the 1/USD grid) and J = quality − 0·cost. Quality and cost are exact lookups of the logged arm the first action picks. Sampled rows average 16 samples per task.

#### held-out (fold 0 test) (10 tasks)

| policy | quality | cost (USD) | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | 0.600 | 0.0911 | 0.600 | yes | 70.0% | INVALID 70%, mid 30% |
| SFT (greedy) | 0.700 | 0.0730 | 0.700 | yes | 0.0% | cheap 60%, frontier 20%, mid 20% |
| SFT (sampled T=1, 16/prompt) | 0.656 | 0.0852 | 0.656 | yes | 0.0% | cheap 57%, frontier 28%, mid 16% |
| GRPO step 100 (greedy) | 0.700 | 0.0730 | 0.700 | yes | 0.0% | cheap 60%, frontier 20%, mid 20% |
| GRPO step 100 (sampled T=1, 16/prompt) | 0.681 | 0.0924 | 0.681 | yes | 0.0% | cheap 46%, frontier 30%, mid 24% |
| fixed cheap | 0.500 | 0.0687 | 0.500 | yes | 0.0% | cheap 100% |
| fixed mid | 0.600 | 0.1654 | 0.600 | NO | 0.0% | mid 100% |
| fixed frontier | 0.700 | 0.1421 | 0.700 | NO | 0.0% | frontier 100% |
| linear constrained policy, Part A fold 0 (served mix) | 0.590 | 0.1431 | 0.590 | NO | 0.0% | cheap 23%, mid 62%, frontier 15% |
| category-oracle rule at λ=0 from the 40 training tasks (SFT target) | 0.500 | 0.1352 | 0.500 | NO | 0.0% | cheap 50%, mid 40%, frontier 10% |
| per-task oracle at λ=0 (argmax q − λc on each task) | 0.800 | 0.0702 | 0.800 | yes | 0.0% | cheap 70%, mid 30% |
| per-task oracle (max quality, cheapest among ties) | 0.800 | 0.0594 | 0.800 | yes | 0.0% | cheap 60%, mid 30%, frontier 10% |

#### training (fold 0 train) (40 tasks)

| policy | quality | cost (USD) | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | 0.484 | 0.0808 | 0.484 | yes | 65.0% | INVALID 65%, mid 22%, cheap 12% |
| SFT (greedy) | 0.705 | 0.0728 | 0.705 | yes | 0.0% | cheap 50%, frontier 38%, mid 12% |
| SFT (sampled T=1, 16/prompt) | 0.680 | 0.0773 | 0.680 | yes | 0.6% | cheap 52%, frontier 35%, mid 12% |
| GRPO step 100 (greedy) | 0.730 | 0.0758 | 0.730 | yes | 0.0% | cheap 48%, frontier 38%, mid 15% |
| GRPO step 100 (sampled T=1, 16/prompt) | 0.717 | 0.0850 | 0.717 | yes | 0.0% | cheap 45%, frontier 40%, mid 16% |
| fixed cheap | 0.430 | 0.0679 | 0.430 | yes | 0.0% | cheap 100% |
| fixed mid | 0.609 | 0.1081 | 0.609 | NO | 0.0% | mid 100% |
| fixed frontier | 0.653 | 0.1698 | 0.653 | NO | 0.0% | frontier 100% |
| linear constrained policy, Part A fold 0 (served mix) | 0.634 | 0.1012 | 0.634 | yes | 0.0% | cheap 21%, mid 61%, frontier 17% |
| category-oracle rule at λ=0 from the 40 training tasks (SFT target) | 0.736 | 0.1240 | 0.736 | NO | 0.0% | cheap 42%, mid 30%, frontier 28% |
| per-task oracle at λ=0 (argmax q − λc on each task) | 0.761 | 0.1168 | 0.761 | NO | 0.0% | cheap 60%, mid 30%, frontier 10% |
| per-task oracle (max quality, cheapest among ties) | 0.761 | 0.0997 | 0.761 | yes | 0.0% | cheap 40%, mid 38%, frontier 22% |

Greedy tier mix on the 10 held-out tasks by shown price (λ per USD):

| model | λ=0 | λ=1 | λ=2 | λ=5 | λ=10 | λ=20 |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 |
| SFT (greedy) | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 |
| GRPO step 100 (greedy) | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 |

Wall clock in the container: 574s on NVIDIA A100-SXM4-40GB (GRPO 403.2s).
