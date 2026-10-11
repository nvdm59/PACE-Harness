## Part A: linear controller on the CRMArena-Pro b2b log (150 tasks × 3 tiers)

### 1–2. Cross-validated first-step tier policies

5-fold CV over task ids (category-stratified, seed 0): train on 40 tasks, evaluate on the held-out 10. All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.

#### λ = 2 per dollar (J = quality − 2·cost)

| policy | quality | cost/task (USD) | J (λ=2) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.571 | 0.1018 | 0.367 [0.073] | yes |
| AWR+IPS linear (greedy) | 0.597 | 0.0881 | 0.421 [0.081] | yes |
| per-category plug-in rule (train-fold argmax) | 0.623 | 0.0816 | 0.460 [0.094] | yes |
| uniform behavior | 0.537 | 0.1075 | 0.322 [0.074] | NO |
| fixed cheap | 0.463 | 0.0597 | 0.344 [0.027] | yes |
| fixed mid | 0.533 | 0.1195 | 0.294 [0.107] | NO |
| fixed frontier | 0.616 | 0.1434 | 0.329 [0.115] | NO |
| per-task oracle (argmax q - 2c) | 0.711 | 0.0567 | 0.597 [0.040] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.400 (held-out: 0.367).

#### λ = 5 per dollar (J = quality − 5·cost)

| policy | quality | cost/task (USD) | J (λ=5) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.561 | 0.0920 | 0.101 [0.092] | yes |
| AWR+IPS linear (greedy) | 0.583 | 0.0702 | 0.232 [0.073] | yes |
| per-category plug-in rule (train-fold argmax) | 0.610 | 0.0766 | 0.227 [0.115] | yes |
| uniform behavior | 0.537 | 0.1075 | -0.000 [0.126] | NO |
| fixed cheap | 0.463 | 0.0597 | 0.165 [0.034] | yes |
| fixed mid | 0.533 | 0.1195 | -0.065 [0.149] | NO |
| fixed frontier | 0.616 | 0.1434 | -0.101 [0.232] | NO |
| per-task oracle (argmax q - 5c) | 0.711 | 0.0567 | 0.427 [0.047] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): 0.146 (held-out: 0.101).

#### λ = 10 per dollar (J = quality − 10·cost)

| policy | quality | cost/task (USD) | J (λ=10) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.540 | 0.0884 | -0.344 [0.111] | yes |
| AWR+IPS linear (greedy) | 0.523 | 0.0731 | -0.207 [0.076] | yes |
| per-category plug-in rule (train-fold argmax) | 0.576 | 0.0694 | -0.117 [0.138] | yes |
| uniform behavior | 0.537 | 0.1075 | -0.538 [0.217] | NO |
| fixed cheap | 0.463 | 0.0597 | -0.134 [0.050] | yes |
| fixed mid | 0.533 | 0.1195 | -0.662 [0.228] | NO |
| fixed frontier | 0.616 | 0.1434 | -0.817 [0.439] | NO |
| per-task oracle (argmax q - 10c) | 0.683 | 0.0532 | 0.151 [0.059] | yes |

AWR+IPS in-sample J on the 40 training tasks (mean over folds): -0.305 (held-out: -0.344).

#### Constrained: budget B = $0.10 per task, λ learned by dual ascent

J uses each fold's own learned λ̄ (mean over folds 0.80 per dollar; per fold 0.69, 0.79, 0.82, 0.77, 0.95).

| policy | quality | cost/task (USD) | J (fold λ̄) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| constrained cgrpo-lookup linear (served: iterate mix) | 0.616 | 0.1062 | 0.531 [0.064] | NO |
| constrained cgrpo-lookup linear (greedy of the mix) | 0.644 | 0.1054 | 0.559 [0.080] | NO |
| uniform behavior | 0.537 | 0.1075 | 0.451 [0.053] | NO |
| fixed cheap | 0.463 | 0.0597 | 0.415 [0.022] | yes |
| fixed mid | 0.533 | 0.1195 | 0.437 [0.094] | NO |
| fixed frontier | 0.616 | 0.1434 | 0.500 [0.071] | NO |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.536 | 0.1038 | 0.453 [0.038] | yes |
| per-task oracle (max quality, cheapest among ties) | 0.714 | 0.0685 | 0.659 [0.042] | yes |

Constrained policy in-sample on its 40 training tasks (mean over folds): quality 0.635, cost $0.0995.

### 3. DR gate on the real log

Candidate: the constrained cgrpo-lookup policy trained on all 50 tasks (λ̄ = 0.91/USD). Exact value on the 50 tasks (all arms logged): **quality 0.632, cost $0.0990**. Uniform behavior exact: quality 0.537, cost $0.1075. Gate: eps_q = 0.03, budget $0.10 + tol $0.005.

### (a) Realized behavior log: one uniformly drawn arm per task (50 episodes, seed 0)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.550 [0.475, 0.608] | 0.1014 [0.0907, 0.1140] |
| candidate | IPS | 0.672 [0.525, 0.795] | 0.0887 [0.0726, 0.1058] |
| candidate | SNIPS | 0.685 [0.586, 0.756] | 0.0905 [0.0794, 0.1027] |
| candidate | DR | 0.693 [0.584, 0.757] | 0.0905 [0.0782, 0.1006] |

criteria: DR-quality lower CI 0.584 >= q_prev - eps = 0.550 (incumbent on this log) - 0.03 = 0.520 [ok];  DR-cost upper CI 0.1006 <= budget + tol = 0.1050 [ok]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.7)

**DECISION: PROMOTE**

Estimator error vs exact on this draw: quality IPS +0.040, SNIPS +0.054, DR +0.061; cost IPS -0.0104, SNIPS -0.0086, DR -0.0085.

### (b) Full counterfactual log: all three arms per task (150 episodes)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.537 [0.503, 0.572] | 0.1075 [0.0987, 0.1166] |
| candidate | IPS | 0.632 [0.570, 0.684] | 0.0990 [0.0896, 0.1080] |
| candidate | SNIPS | 0.632 [0.570, 0.684] | 0.0990 [0.0896, 0.1080] |
| candidate | DR | 0.632 [0.570, 0.684] | 0.0990 [0.0896, 0.1080] |

criteria: DR-quality lower CI 0.570 >= q_prev - eps = 0.537 (incumbent on this log) - 0.03 = 0.507 [ok];  DR-cost upper CI 0.1080 <= budget + tol = 0.1050 [FAIL]
(n = 450 logged episodes over 150 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value (the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure task-sampling uncertainty from 50 tasks.

(c) The same gate over 200 independent redraws of the realized behavior log (one uniform arm per task each time; bootstrap 200 per redraw):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.008 | 0.070 | 0.253 | 92% |
| quality | SNIPS | +0.002 | 0.040 | 0.167 | 97% |
| quality | DR | +0.003 | 0.044 | 0.190 | 98% |
| cost | IPS | +0.0006 | 0.0111 | 0.0385 | 91% |
| cost | SNIPS | -0.0003 | 0.0078 | 0.0304 | 93% |
| cost | DR | -0.0006 | 0.0080 | 0.0333 | 96% |

PROMOTE in 21/200 redraws (10%).

### 4. Repricing drift (no new calls)

Frozen policy: the all-50 constrained policy from the gate section (λ̄ = 0.91/USD; J = quality − 0.91·cost). Costs after each event are recomputed from the logged token counts (spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized log (seed 0), with the quality bar recorded at the frozen policy's promotion (0.693).

In-sample (all 50 tasks, exact lookups):

| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | (iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | gate on retrained (exact q / cost of candidate) |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.632 / $0.0990 / 0.542 | 0.632 / $0.1383 / 0.506 (NO budget) | 0.620 / $0.0995 / 0.530 (yes budget) | -0.036 | +0.024 | REVERT (DR q 0.658 [0.556, 0.740], DR cost $0.0943 [0.0802, 0.1063]; exact 0.620 / $0.0995) |
| mid ×0.5 | 0.632 / $0.0990 / 0.542 | 0.632 / $0.0973 / 0.543 (yes budget) | 0.654 / $0.0980 / 0.565 (yes budget) | +0.002 | +0.022 | REVERT (DR q 0.728 [0.599, 0.792], DR cost $0.0894 [0.0768, 0.1012]; exact 0.654 / $0.0980) |
| cheap ×3 | 0.632 / $0.0990 / 0.542 | 0.632 / $0.1328 / 0.511 (NO budget) | 0.642 / $0.1052 / 0.546 (NO budget) | -0.031 | +0.036 | REVERT (DR q 0.737 [0.614, 0.807], DR cost $0.1022 [0.0862, 0.1237]; exact 0.642 / $0.1052) |

Held-out (5-fold CV; frozen and retrained both trained on the 40 training tasks of each fold, evaluated on its 10 held-out tasks; J at each fold's frozen λ̄; means over folds):

| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J | priced-BWT | recovery | (iii) within budget |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.616 / $0.1062 / 0.531 | 0.616 / $0.1491 / 0.497 | 0.596 / $0.1033 / 0.513 | -0.034 | +0.017 | yes |
| mid ×0.5 | 0.616 / $0.1062 / 0.531 | 0.616 / $0.1044 / 0.532 | 0.634 / $0.1055 / 0.550 | +0.001 | +0.017 | NO |
| cheap ×3 | 0.616 / $0.1062 / 0.531 | 0.616 / $0.1398 / 0.504 | 0.644 / $0.1165 / 0.551 | -0.027 | +0.047 | NO |

Harness-version drift (v1 → v2) cannot be measured on this log: it holds a single harness version (crmarena-pro-v1) and nothing here re-runs the harness.

### Gate after frontier ×1.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.693 [0.584, 0.757] | 0.1261 [0.1076, 0.1402] |
| candidate | IPS | 0.648 [0.513, 0.777] | 0.0944 [0.0730, 0.1170] |
| candidate | SNIPS | 0.661 [0.555, 0.744] | 0.0964 [0.0799, 0.1129] |
| candidate | DR | 0.658 [0.556, 0.740] | 0.0943 [0.0802, 0.1063] |

criteria: DR-quality lower CI 0.556 >= q_prev - eps = 0.693 (recorded at incumbent's promotion) - 0.03 = 0.663 [FAIL];  DR-cost upper CI 0.1063 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after mid ×0.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.693 [0.584, 0.757] | 0.0887 [0.0763, 0.0988] |
| candidate | IPS | 0.701 [0.536, 0.845] | 0.0882 [0.0701, 0.1065] |
| candidate | SNIPS | 0.710 [0.601, 0.780] | 0.0893 [0.0773, 0.1025] |
| candidate | DR | 0.728 [0.599, 0.792] | 0.0894 [0.0768, 0.1012] |

criteria: DR-quality lower CI 0.599 >= q_prev - eps = 0.693 (recorded at incumbent's promotion) - 0.03 = 0.663 [FAIL];  DR-cost upper CI 0.1012 <= budget + tol = 0.1050 [ok]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after cheap ×3: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.693 [0.584, 0.757] | 0.1221 [0.1067, 0.1377] |
| candidate | IPS | 0.719 [0.568, 0.857] | 0.1013 [0.0802, 0.1231] |
| candidate | SNIPS | 0.725 [0.630, 0.795] | 0.1021 [0.0871, 0.1173] |
| candidate | DR | 0.737 [0.614, 0.807] | 0.1022 [0.0862, 0.1237] |

criteria: DR-quality lower CI 0.614 >= q_prev - eps = 0.693 (recorded at incumbent's promotion) - 0.03 = 0.663 [FAIL];  DR-cost upper CI 0.1237 <= budget + tol = 0.1050 [FAIL]
(n = 150 logged episodes over 150 tasks, max cumulative ratio 2.8)

**DECISION: REVERT**

