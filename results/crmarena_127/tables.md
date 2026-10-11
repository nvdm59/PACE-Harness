## Part A: linear controller on the CRMArena-Pro b2b log (127 tasks × 3 tiers)

### 1–2. Cross-validated first-step tier policies

5-fold CV over task ids (category-stratified, seed 0): train on 40 tasks, evaluate on the held-out 10. All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.

#### λ = 2 per dollar (J = quality − 2·cost)

| policy | quality | cost/task (USD) | J (λ=2) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.579 | 0.1031 | 0.373 [0.070] | yes |
| AWR+IPS linear (greedy) | 0.626 | 0.0921 | 0.442 [0.090] | yes |
| per-category plug-in rule (train-fold argmax) | 0.666 | 0.0838 | 0.498 [0.052] | yes |
| uniform behavior | 0.545 | 0.1074 | 0.330 [0.057] | NO |
| fixed cheap | 0.461 | 0.0627 | 0.335 [0.065] | yes |
| fixed mid | 0.541 | 0.1173 | 0.306 [0.068] | NO |
| fixed frontier | 0.634 | 0.1421 | 0.350 [0.100] | NO |
| per-task oracle (argmax q - 2c) | 0.729 | 0.0582 | 0.613 [0.048] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.411 (held-out: 0.373).

#### λ = 5 per dollar (J = quality − 5·cost)

| policy | quality | cost/task (USD) | J (λ=5) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.568 | 0.0941 | 0.097 [0.100] | yes |
| AWR+IPS linear (greedy) | 0.540 | 0.0830 | 0.125 [0.113] | yes |
| per-category plug-in rule (train-fold argmax) | 0.626 | 0.0817 | 0.218 [0.067] | yes |
| uniform behavior | 0.545 | 0.1074 | 0.008 [0.097] | NO |
| fixed cheap | 0.461 | 0.0627 | 0.147 [0.098] | yes |
| fixed mid | 0.541 | 0.1173 | -0.046 [0.099] | NO |
| fixed frontier | 0.634 | 0.1421 | -0.077 [0.182] | NO |
| per-task oracle (argmax q - 5c) | 0.729 | 0.0582 | 0.439 [0.063] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.150 (held-out: 0.097).

#### λ = 10 per dollar (J = quality − 10·cost)

| policy | quality | cost/task (USD) | J (λ=10) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.541 | 0.0900 | -0.358 [0.145] | yes |
| AWR+IPS linear (greedy) | 0.508 | 0.0799 | -0.291 [0.157] | yes |
| per-category plug-in rule (train-fold argmax) | 0.579 | 0.0781 | -0.202 [0.087] | yes |
| uniform behavior | 0.545 | 0.1074 | -0.529 [0.166] | NO |
| fixed cheap | 0.461 | 0.0627 | -0.166 [0.155] | yes |
| fixed mid | 0.541 | 0.1173 | -0.633 [0.155] | NO |
| fixed frontier | 0.634 | 0.1421 | -0.787 [0.320] | NO |
| per-task oracle (argmax q - 10c) | 0.705 | 0.0553 | 0.152 [0.091] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): -0.296 (held-out: -0.358).

#### Constrained: budget B = $0.10 per task, λ learned by dual ascent

J uses each fold's own learned λ̄ (mean over folds 0.97 per dollar; per fold 1.14, 0.68, 1.22, 0.65, 1.15).

| policy | quality | cost/task (USD) | J (fold λ̄) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| constrained cgrpo-lookup linear (served: iterate mix) | 0.645 | 0.1073 | 0.546 [0.044] | NO |
| constrained cgrpo-lookup linear (greedy of the mix) | 0.690 | 0.1070 | 0.591 [0.058] | NO |
| uniform behavior | 0.545 | 0.1074 | 0.444 [0.027] | NO |
| fixed cheap | 0.461 | 0.0627 | 0.402 [0.043] | yes |
| fixed mid | 0.541 | 0.1173 | 0.428 [0.065] | NO |
| fixed frontier | 0.634 | 0.1421 | 0.503 [0.045] | NO |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.542 | 0.1011 | 0.449 [0.034] | yes |
| per-task oracle (max quality, cheapest among ties) | 0.731 | 0.0700 | 0.668 [0.039] | yes |

Constrained policy in-sample on its 40 training tasks (mean over folds): quality 0.659, cost $0.0996.

### 3. DR gate on the real log

Candidate: the constrained cgrpo-lookup policy trained on all 50 tasks (λ̄ = 1.15/USD). Exact value on the 50 tasks (all arms logged): **quality 0.655, cost $0.0990**. Uniform behavior exact: quality 0.545, cost $0.1075. Gate: eps_q = 0.03, budget $0.10 + tol $0.005.

### (a) Realized behavior log: one uniformly drawn arm per task (50 episodes, seed 0)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.553 [0.489, 0.618] | 0.1132 [0.0935, 0.1318] |
| candidate | IPS | 0.734 [0.587, 0.874] | 0.1065 [0.0856, 0.1279] |
| candidate | SNIPS | 0.717 [0.630, 0.788] | 0.1041 [0.0854, 0.1233] |
| candidate | DR | 0.688 [0.597, 0.784] | 0.1044 [0.0859, 0.1263] |

criteria: DR-quality lower CI 0.597 >= q_prev - eps = 0.553 (incumbent on this log) - 0.03 = 0.523 [ok];  DR-cost upper CI 0.1263 <= budget + tol = 0.1050 [FAIL]
(n = 128 logged episodes over 128 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

Estimator error vs exact on this draw: quality IPS +0.079, SNIPS +0.062, DR +0.034; cost IPS +0.0075, SNIPS +0.0051, DR +0.0054.

### (b) Full counterfactual log: all three arms per task (150 episodes)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.542 [0.503, 0.582] | 0.1075 [0.0986, 0.1161] |
| candidate | IPS | 0.651 [0.591, 0.703] | 0.0986 [0.0891, 0.1087] |
| candidate | SNIPS | 0.653 [0.592, 0.705] | 0.0989 [0.0891, 0.1087] |
| candidate | DR | 0.653 [0.592, 0.705] | 0.0989 [0.0891, 0.1087] |

criteria: DR-quality lower CI 0.592 >= q_prev - eps = 0.542 (incumbent on this log) - 0.03 = 0.512 [ok];  DR-cost upper CI 0.1087 <= budget + tol = 0.1050 [FAIL]
(n = 383 logged episodes over 128 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value (the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure task-sampling uncertainty from 50 tasks.

(c) The same gate over 200 independent redraws of the realized behavior log (one uniform arm per task each time; bootstrap 200 per redraw):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | -0.007 | 0.069 | 0.278 | 96% |
| quality | SNIPS | -0.001 | 0.040 | 0.181 | 96% |
| quality | DR | -0.006 | 0.045 | 0.202 | 98% |
| cost | IPS | -0.0006 | 0.0113 | 0.0417 | 92% |
| cost | SNIPS | +0.0004 | 0.0076 | 0.0326 | 94% |
| cost | DR | +0.0005 | 0.0089 | 0.0356 | 97% |

PROMOTE in 15/200 redraws (8%).

### 4. Repricing drift (no new calls)

Frozen policy: the all-50 constrained policy from the gate section (λ̄ = 1.15/USD; J = quality − 1.15·cost). Costs after each event are recomputed from the logged token counts (spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized log (seed 0), with the quality bar recorded at the frozen policy's promotion (0.688).

In-sample (all 50 tasks, exact lookups):

| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | (iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | gate on retrained (exact q / cost of candidate) |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.655 / $0.0990 / 0.541 | 0.655 / $0.1381 / 0.496 (NO budget) | 0.621 / $0.0995 / 0.506 (yes budget) | -0.045 | +0.010 | REVERT (DR q 0.592 [0.491, 0.707], DR cost $0.0920 [0.0744, 0.1124]; exact 0.621 / $0.0995) |
| mid ×0.5 | 0.655 / $0.0990 / 0.541 | 0.655 / $0.0975 / 0.543 (yes budget) | 0.680 / $0.0985 / 0.567 (yes budget) | +0.002 | +0.025 | REVERT (DR q 0.730 [0.630, 0.830], DR cost $0.1028 [0.0838, 0.1268]; exact 0.680 / $0.0985) |
| cheap ×3 | 0.655 / $0.0990 / 0.541 | 0.655 / $0.1347 / 0.500 (NO budget) | 0.651 / $0.1004 / 0.536 (yes budget) | -0.041 | +0.036 | REVERT (DR q 0.670 [0.545, 0.783], DR cost $0.0830 [0.0708, 0.1095]; exact 0.651 / $0.1004) |

Held-out (5-fold CV; frozen and retrained both trained on the 40 training tasks of each fold, evaluated on its 10 held-out tasks; J at each fold's frozen λ̄; means over folds):

| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J | priced-BWT | recovery | (iii) within budget |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.645 / $0.1073 / 0.546 | 0.645 / $0.1509 / 0.506 | 0.603 / $0.1042 / 0.504 | -0.040 | -0.002 | yes |
| mid ×0.5 | 0.645 / $0.1073 / 0.546 | 0.645 / $0.1056 / 0.548 | 0.669 / $0.1079 / 0.570 | +0.002 | +0.022 | NO |
| cheap ×3 | 0.645 / $0.1073 / 0.546 | 0.645 / $0.1408 / 0.514 | 0.631 / $0.1186 / 0.521 | -0.032 | +0.007 | NO |

Harness-version drift (v1 → v2) cannot be measured on this log: it holds a single harness version (crmarena-pro-v1) and nothing here re-runs the harness.

### Gate after frontier ×1.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.688 [0.597, 0.784] | 0.1479 [0.1212, 0.1809] |
| candidate | IPS | 0.655 [0.512, 0.796] | 0.1025 [0.0841, 0.1240] |
| candidate | SNIPS | 0.638 [0.536, 0.745] | 0.0999 [0.0844, 0.1166] |
| candidate | DR | 0.592 [0.491, 0.707] | 0.0920 [0.0744, 0.1124] |

criteria: DR-quality lower CI 0.491 >= q_prev - eps = 0.688 (recorded at incumbent's promotion) - 0.03 = 0.658 [FAIL];  DR-cost upper CI 0.1124 <= budget + tol = 0.1050 [FAIL]
(n = 128 logged episodes over 128 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after mid ×0.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.688 [0.597, 0.784] | 0.1030 [0.0843, 0.1249] |
| candidate | IPS | 0.773 [0.600, 0.933] | 0.1054 [0.0821, 0.1287] |
| candidate | SNIPS | 0.754 [0.662, 0.826] | 0.1028 [0.0850, 0.1216] |
| candidate | DR | 0.730 [0.630, 0.830] | 0.1028 [0.0838, 0.1268] |

criteria: DR-quality lower CI 0.630 >= q_prev - eps = 0.688 (recorded at incumbent's promotion) - 0.03 = 0.658 [FAIL];  DR-cost upper CI 0.1268 <= budget + tol = 0.1050 [FAIL]
(n = 128 logged episodes over 128 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after cheap ×3: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.688 [0.597, 0.784] | 0.1336 [0.1093, 0.1597] |
| candidate | IPS | 0.723 [0.577, 0.891] | 0.1055 [0.0889, 0.1282] |
| candidate | SNIPS | 0.647 [0.538, 0.726] | 0.0944 [0.0813, 0.1056] |
| candidate | DR | 0.670 [0.545, 0.783] | 0.0830 [0.0708, 0.1095] |

criteria: DR-quality lower CI 0.545 >= q_prev - eps = 0.688 (recorded at incumbent's promotion) - 0.03 = 0.658 [FAIL];  DR-cost upper CI 0.1095 <= budget + tol = 0.1050 [FAIL]
(n = 128 logged episodes over 128 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

