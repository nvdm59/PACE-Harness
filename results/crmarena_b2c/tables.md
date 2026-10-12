## Part A: linear controller on the CRMArena-Pro b2c log (150 tasks × 3 tiers)

### 1–2. Cross-validated first-step tier policies

5-fold CV over task ids (category-stratified, seed 0): train on ~120 tasks, evaluate on the held-out ~30. All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.

#### λ = 2 per dollar (J = quality − 2·cost)

| policy | quality | cost/task (USD) | J (λ=2) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.569 | 0.0988 | 0.371 [0.095] | yes |
| AWR+IPS linear (greedy) | 0.598 | 0.0813 | 0.435 [0.057] | yes |
| per-category plug-in rule (train-fold argmax) | 0.636 | 0.0812 | 0.474 [0.067] | yes |
| uniform behavior | 0.538 | 0.1085 | 0.322 [0.090] | NO |
| fixed cheap | 0.469 | 0.0558 | 0.358 [0.078] | yes |
| fixed mid | 0.508 | 0.1263 | 0.255 [0.109] | NO |
| fixed frontier | 0.638 | 0.1433 | 0.352 [0.116] | NO |
| per-task oracle (argmax q - 2c) | 0.727 | 0.0607 | 0.606 [0.068] | yes |

AWR+IPS in-sample J on the ~120 training tasks of each fold (mean over folds): 0.399 (held-out: 0.371).

#### λ = 5 per dollar (J = quality − 5·cost)

| policy | quality | cost/task (USD) | J (λ=5) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.542 | 0.0897 | 0.093 [0.134] | yes |
| AWR+IPS linear (greedy) | 0.529 | 0.0608 | 0.226 [0.098] | yes |
| per-category plug-in rule (train-fold argmax) | 0.603 | 0.0716 | 0.245 [0.087] | yes |
| uniform behavior | 0.538 | 0.1085 | -0.004 [0.127] | NO |
| fixed cheap | 0.469 | 0.0558 | 0.191 [0.089] | yes |
| fixed mid | 0.508 | 0.1263 | -0.124 [0.164] | NO |
| fixed frontier | 0.638 | 0.1433 | -0.078 [0.165] | NO |
| per-task oracle (argmax q - 5c) | 0.719 | 0.0583 | 0.427 [0.092] | yes |

AWR+IPS in-sample J on the ~120 training tasks of each fold (mean over folds): 0.138 (held-out: 0.093).

#### λ = 10 per dollar (J = quality − 10·cost)

| policy | quality | cost/task (USD) | J (λ=10) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.521 | 0.0807 | -0.286 [0.182] | yes |
| AWR+IPS linear (greedy) | 0.523 | 0.0608 | -0.085 [0.136] | yes |
| per-category plug-in rule (train-fold argmax) | 0.543 | 0.0615 | -0.072 [0.126] | yes |
| uniform behavior | 0.538 | 0.1085 | -0.546 [0.191] | NO |
| fixed cheap | 0.469 | 0.0558 | -0.088 [0.113] | yes |
| fixed mid | 0.508 | 0.1263 | -0.756 [0.257] | NO |
| fixed frontier | 0.638 | 0.1433 | -0.795 [0.250] | NO |
| per-task oracle (argmax q - 10c) | 0.662 | 0.0516 | 0.147 [0.130] | yes |

AWR+IPS in-sample J on the ~120 training tasks of each fold (mean over folds): -0.231 (held-out: -0.286).

#### Constrained: budget B = $0.10 per task, λ learned by dual ascent

J uses each fold's own learned λ̄ (mean over folds 1.21 per dollar; per fold 1.49, 1.26, 0.81, 1.17, 1.33).

| policy | quality | cost/task (USD) | J (fold λ̄) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| constrained cgrpo-lookup linear (served: iterate mix) | 0.617 | 0.1033 | 0.493 [0.082] | yes |
| constrained cgrpo-lookup linear (greedy of the mix) | 0.677 | 0.1052 | 0.549 [0.100] | NO |
| uniform behavior | 0.538 | 0.1085 | 0.409 [0.077] | NO |
| fixed cheap | 0.469 | 0.0558 | 0.402 [0.071] | yes |
| fixed mid | 0.508 | 0.1263 | 0.357 [0.085] | NO |
| fixed frontier | 0.638 | 0.1433 | 0.466 [0.111] | NO |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.554 | 0.1005 | 0.433 [0.079] | yes |
| per-task oracle (max quality, cheapest among ties) | 0.728 | 0.0619 | 0.654 [0.063] | yes |

Constrained policy in-sample on its ~120 training tasks of each fold (mean over folds): quality 0.633, cost $0.1000.

### 3. DR gate on the real log

Candidate: the constrained cgrpo-lookup policy trained on all 150 tasks (λ̄ = 1.35/USD). Exact value on the 150 tasks (all arms logged): **quality 0.630, cost $0.0999**. Uniform behavior exact: quality 0.538, cost $0.1085. Gate: eps_q = 0.03, budget $0.10 + tol $0.005.

### (a) Realized behavior log: one uniformly drawn arm per task (150 episodes, seed 0)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.572 [0.520, 0.636] | 0.1024 [0.0913, 0.1141] |
| candidate | IPS | 0.685 [0.553, 0.803] | 0.1058 [0.0880, 0.1260] |
| candidate | SNIPS | 0.667 [0.593, 0.746] | 0.1032 [0.0909, 0.1172] |
| candidate | DR | 0.660 [0.562, 0.751] | 0.1033 [0.0900, 0.1160] |

criteria: DR-quality lower CI 0.562 >= q_prev - eps = 0.572 (incumbent on this log) - 0.03 = 0.542 [ok];  DR-cost upper CI 0.1160 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.6)

**DECISION: REVERT**

Estimator error vs exact on this draw: quality IPS +0.055, SNIPS +0.037, DR +0.031; cost IPS +0.0060, SNIPS +0.0033, DR +0.0035.

### (b) Full counterfactual log: all three arms per task (450 episodes)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.538 [0.502, 0.575] | 0.1085 [0.1015, 0.1180] |
| candidate | IPS | 0.630 [0.574, 0.681] | 0.0999 [0.0911, 0.1074] |
| candidate | SNIPS | 0.630 [0.574, 0.681] | 0.0999 [0.0911, 0.1074] |
| candidate | DR | 0.630 [0.574, 0.681] | 0.0999 [0.0911, 0.1074] |

criteria: DR-quality lower CI 0.574 >= q_prev - eps = 0.538 (incumbent on this log) - 0.03 = 0.508 [ok];  DR-cost upper CI 0.1074 <= budget + tol = 0.1050 [FAIL]
(n = 450 logged episodes over 150 tasks, max cumulative ratio 2.6)

**DECISION: REVERT**

With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value (the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure task-sampling uncertainty from 150 tasks.

(c) The same gate over 200 independent redraws of the realized behavior log (one uniform arm per task each time; bootstrap 200 per redraw):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.012 | 0.072 | 0.244 | 90% |
| quality | SNIPS | +0.006 | 0.039 | 0.161 | 94% |
| quality | DR | +0.007 | 0.044 | 0.181 | 98% |
| cost | IPS | +0.0012 | 0.0100 | 0.0360 | 95% |
| cost | SNIPS | +0.0005 | 0.0072 | 0.0278 | 94% |
| cost | DR | -0.0001 | 0.0075 | 0.0301 | 97% |

PROMOTE in 16/200 redraws (8%).

### 4. Repricing drift (no new calls)

Frozen policy: the all-50 constrained policy from the gate section (λ̄ = 1.35/USD; J = quality − 1.35·cost). Costs after each event are recomputed from the logged token counts (spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized log (seed 0), with the quality bar recorded at the frozen policy's promotion (0.660).

In-sample (all 150 tasks, exact lookups):

| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | (iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | gate on retrained (exact q / cost of candidate) |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.630 / $0.0999 / 0.495 | 0.630 / $0.1386 / 0.442 (NO budget) | 0.624 / $0.0991 / 0.490 (yes budget) | -0.052 | +0.048 | REVERT (DR q 0.664 [0.569, 0.772], DR cost $0.1147 [0.0966, 0.1274]; exact 0.624 / $0.0991) |
| mid ×0.5 | 0.630 / $0.0999 / 0.495 | 0.630 / $0.0984 / 0.497 (yes budget) | 0.664 / $0.0986 / 0.531 (yes budget) | +0.002 | +0.034 | REVERT (DR q 0.703 [0.585, 0.786], DR cost $0.1042 [0.0905, 0.1184]; exact 0.664 / $0.0986) |
| cheap ×3 | 0.630 / $0.0999 / 0.495 | 0.630 / $0.1389 / 0.442 (NO budget) | 0.627 / $0.1014 / 0.490 (yes budget) | -0.053 | +0.048 | REVERT (DR q 0.671 [0.561, 0.766], DR cost $0.1046 [0.0874, 0.1276]; exact 0.627 / $0.1014) |

Held-out (5-fold CV; frozen and retrained both trained on the ~120 training tasks of each fold, evaluated on its ~30 held-out tasks; J at each fold's frozen λ̄; means over folds):

| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J | priced-BWT | recovery | (iii) within budget |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.617 / $0.1033 / 0.493 | 0.617 / $0.1438 / 0.444 | 0.593 / $0.1016 / 0.471 | -0.049 | +0.027 | yes |
| mid ×0.5 | 0.617 / $0.1033 / 0.493 | 0.617 / $0.1018 / 0.495 | 0.650 / $0.1041 / 0.525 | +0.002 | +0.030 | yes |
| cheap ×3 | 0.617 / $0.1033 / 0.493 | 0.617 / $0.1420 / 0.446 | 0.595 / $0.1170 / 0.455 | -0.047 | +0.009 | NO |

Harness-version drift (v1 → v2) cannot be measured on this log: it holds a single harness version (crmarena-pro-v1) and nothing here re-runs the harness.

### Gate after frontier ×1.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.660 [0.562, 0.751] | 0.1406 [0.1236, 0.1569] |
| candidate | IPS | 0.709 [0.581, 0.836] | 0.1221 [0.0963, 0.1486] |
| candidate | SNIPS | 0.674 [0.596, 0.772] | 0.1160 [0.0978, 0.1312] |
| candidate | DR | 0.664 [0.569, 0.772] | 0.1147 [0.0966, 0.1274] |

criteria: DR-quality lower CI 0.569 >= q_prev - eps = 0.660 (recorded at incumbent's promotion) - 0.03 = 0.630 [FAIL];  DR-cost upper CI 0.1274 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after mid ×0.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.660 [0.562, 0.751] | 0.1022 [0.0888, 0.1149] |
| candidate | IPS | 0.725 [0.566, 0.868] | 0.1072 [0.0867, 0.1288] |
| candidate | SNIPS | 0.699 [0.616, 0.780] | 0.1032 [0.0909, 0.1174] |
| candidate | DR | 0.703 [0.585, 0.786] | 0.1042 [0.0905, 0.1184] |

criteria: DR-quality lower CI 0.585 >= q_prev - eps = 0.660 (recorded at incumbent's promotion) - 0.03 = 0.630 [FAIL];  DR-cost upper CI 0.1184 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after cheap ×3: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.660 [0.562, 0.751] | 0.1563 [0.1298, 0.1842] |
| candidate | IPS | 0.760 [0.618, 0.919] | 0.1214 [0.0990, 0.1473] |
| candidate | SNIPS | 0.663 [0.571, 0.754] | 0.1058 [0.0915, 0.1216] |
| candidate | DR | 0.671 [0.561, 0.766] | 0.1046 [0.0874, 0.1276] |

criteria: DR-quality lower CI 0.561 >= q_prev - eps = 0.660 (recorded at incumbent's promotion) - 0.03 = 0.630 [FAIL];  DR-cost upper CI 0.1276 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 3.0)

**DECISION: REVERT**

