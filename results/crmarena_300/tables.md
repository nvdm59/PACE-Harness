## Part A: linear controller on the CRMArena-Pro b2b log (300 tasks × 3 tiers)

### 1–2. Cross-validated first-step tier policies

5-fold CV over task ids (category-stratified, seed 0): train on ~240 tasks, evaluate on the held-out ~60. All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.

#### λ = 2 per dollar (J = quality − 2·cost)

| policy | quality | cost/task (USD) | J (λ=2) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.548 | 0.0977 | 0.352 [0.029] | yes |
| AWR+IPS linear (greedy) | 0.596 | 0.0796 | 0.437 [0.028] | yes |
| per-category plug-in rule (train-fold argmax) | 0.644 | 0.0789 | 0.486 [0.035] | yes |
| uniform behavior | 0.510 | 0.1045 | 0.301 [0.028] | yes |
| fixed cheap | 0.434 | 0.0569 | 0.320 [0.063] | yes |
| fixed mid | 0.482 | 0.1261 | 0.230 [0.026] | NO |
| fixed frontier | 0.614 | 0.1305 | 0.353 [0.046] | NO |
| per-task oracle (argmax q - 2c) | 0.708 | 0.0584 | 0.592 [0.032] | yes |

AWR+IPS in-sample J on the ~240 training tasks of each fold (mean over folds): 0.369 (held-out: 0.352).

#### λ = 5 per dollar (J = quality − 5·cost)

| policy | quality | cost/task (USD) | J (λ=5) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.536 | 0.0879 | 0.096 [0.055] | yes |
| AWR+IPS linear (greedy) | 0.514 | 0.0680 | 0.174 [0.059] | yes |
| per-category plug-in rule (train-fold argmax) | 0.587 | 0.0724 | 0.225 [0.057] | yes |
| uniform behavior | 0.510 | 0.1045 | -0.012 [0.059] | yes |
| fixed cheap | 0.434 | 0.0569 | 0.150 [0.086] | yes |
| fixed mid | 0.482 | 0.1261 | -0.149 [0.039] | NO |
| fixed frontier | 0.614 | 0.1305 | -0.038 [0.091] | NO |
| per-task oracle (argmax q - 5c) | 0.705 | 0.0571 | 0.419 [0.031] | yes |

AWR+IPS in-sample J on the ~240 training tasks of each fold (mean over folds): 0.116 (held-out: 0.096).

#### λ = 10 per dollar (J = quality − 10·cost)

| policy | quality | cost/task (USD) | J (λ=10) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| AWR+IPS linear (served, stochastic) | 0.511 | 0.0829 | -0.318 [0.090] | yes |
| AWR+IPS linear (greedy) | 0.477 | 0.0646 | -0.169 [0.107] | yes |
| per-category plug-in rule (train-fold argmax) | 0.550 | 0.0653 | -0.103 [0.096] | yes |
| uniform behavior | 0.510 | 0.1045 | -0.535 [0.115] | yes |
| fixed cheap | 0.434 | 0.0569 | -0.135 [0.128] | yes |
| fixed mid | 0.482 | 0.1261 | -0.779 [0.071] | NO |
| fixed frontier | 0.614 | 0.1305 | -0.690 [0.186] | NO |
| per-task oracle (argmax q - 10c) | 0.677 | 0.0534 | 0.143 [0.054] | yes |

AWR+IPS in-sample J on the ~240 training tasks of each fold (mean over folds): -0.294 (held-out: -0.318).

#### Constrained: budget B = $0.10 per task, λ learned by dual ascent

J uses each fold's own learned λ̄ (mean over folds 1.22 per dollar; per fold 1.71, 0.51, 1.69, 1.07, 1.12).

| policy | quality | cost/task (USD) | J (fold λ̄) [fold SD] | within budget (<= $0.105) |
|---|---|---|---|---|
| constrained cgrpo-lookup linear (served: iterate mix) | 0.596 | 0.1016 | 0.475 [0.067] | yes |
| constrained cgrpo-lookup linear (greedy of the mix) | 0.644 | 0.1025 | 0.521 [0.073] | yes |
| uniform behavior | 0.510 | 0.1045 | 0.385 [0.055] | yes |
| fixed cheap | 0.434 | 0.0569 | 0.367 [0.047] | yes |
| fixed mid | 0.482 | 0.1261 | 0.330 [0.076] | NO |
| fixed frontier | 0.614 | 0.1305 | 0.458 [0.092] | NO |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.543 | 0.1008 | 0.423 [0.054] | yes |
| per-task oracle (max quality, cheapest among ties) | 0.711 | 0.0672 | 0.632 [0.059] | yes |

Constrained policy in-sample on its ~240 training tasks of each fold (mean over folds): quality 0.603, cost $0.0992.

### 3. DR gate on the real log

Candidate: the constrained cgrpo-lookup policy trained on all 300 tasks (λ̄ = 1.09/USD). Exact value on the 300 tasks (all arms logged): **quality 0.611, cost $0.0982**. Uniform behavior exact: quality 0.510, cost $0.1045. Gate: eps_q = 0.03, budget $0.10 + tol $0.005.

### (a) Realized behavior log: one uniformly drawn arm per task (300 episodes, seed 0)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.514 [0.466, 0.553] | 0.1015 [0.0930, 0.1135] |
| candidate | IPS | 0.545 [0.467, 0.624] | 0.0906 [0.0777, 0.1035] |
| candidate | SNIPS | 0.599 [0.532, 0.661] | 0.0996 [0.0878, 0.1118] |
| candidate | DR | 0.607 [0.525, 0.668] | 0.0953 [0.0866, 0.1125] |

criteria: DR-quality lower CI 0.525 >= q_prev - eps = 0.514 (incumbent on this log) - 0.03 = 0.484 [ok];  DR-cost upper CI 0.1125 <= budget + tol = 0.1050 [FAIL]
(n = 300 logged episodes over 300 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

Estimator error vs exact on this draw: quality IPS -0.066, SNIPS -0.012, DR -0.004; cost IPS -0.0076, SNIPS +0.0015, DR -0.0029.

### (b) Full counterfactual log: all three arms per task (900 episodes)

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| behavior (on-policy log mean) | - | 0.510 [0.484, 0.536] | 0.1045 [0.0992, 0.1104] |
| candidate | IPS | 0.611 [0.574, 0.649] | 0.0982 [0.0913, 0.1050] |
| candidate | SNIPS | 0.611 [0.574, 0.649] | 0.0982 [0.0913, 0.1050] |
| candidate | DR | 0.611 [0.574, 0.649] | 0.0982 [0.0913, 0.1050] |

criteria: DR-quality lower CI 0.574 >= q_prev - eps = 0.510 (incumbent on this log) - 0.03 = 0.480 [ok];  DR-cost upper CI 0.1050 <= budget + tol = 0.1050 [FAIL]
(n = 900 logged episodes over 300 tasks, max cumulative ratio 2.7)

**DECISION: REVERT**

With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value (the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure task-sampling uncertainty from 300 tasks.

(c) The same gate over 200 independent redraws of the realized behavior log (one uniform arm per task each time; bootstrap 200 per redraw):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.004 | 0.051 | 0.180 | 90% |
| quality | SNIPS | -0.000 | 0.030 | 0.120 | 94% |
| quality | DR | -0.001 | 0.029 | 0.135 | 100% |
| cost | IPS | +0.0000 | 0.0081 | 0.0277 | 92% |
| cost | SNIPS | -0.0005 | 0.0059 | 0.0217 | 91% |
| cost | DR | -0.0004 | 0.0058 | 0.0240 | 96% |

PROMOTE in 60/200 redraws (30%).

### 4. Repricing drift (no new calls)

Frozen policy: the all-50 constrained policy from the gate section (λ̄ = 1.09/USD; J = quality − 1.09·cost). Costs after each event are recomputed from the logged token counts (spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized log (seed 0), with the quality bar recorded at the frozen policy's promotion (0.607).

In-sample (all 300 tasks, exact lookups):

| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | (iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | gate on retrained (exact q / cost of candidate) |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.611 / $0.0982 / 0.504 | 0.611 / $0.1377 / 0.461 (NO budget) | 0.595 / $0.1001 / 0.486 (yes budget) | -0.043 | +0.025 | REVERT (DR q 0.592 [0.500, 0.660], DR cost $0.1000 [0.0870, 0.1143]; exact 0.595 / $0.1001) |
| mid ×0.5 | 0.611 / $0.0982 / 0.504 | 0.611 / $0.0966 / 0.506 (yes budget) | 0.639 / $0.0988 / 0.531 (yes budget) | +0.002 | +0.025 | REVERT (DR q 0.631 [0.544, 0.706], DR cost $0.0955 [0.0874, 0.1130]; exact 0.639 / $0.0988) |
| cheap ×3 | 0.611 / $0.0982 / 0.504 | 0.611 / $0.1300 / 0.470 (NO budget) | 0.625 / $0.1112 / 0.504 (NO budget) | -0.035 | +0.034 | REVERT (DR q 0.610 [0.526, 0.684], DR cost $0.1225 [0.0983, 0.1553]; exact 0.625 / $0.1112) |

Held-out (5-fold CV; frozen and retrained both trained on the ~240 training tasks of each fold, evaluated on its ~60 held-out tasks; J at each fold's frozen λ̄; means over folds):

| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J | priced-BWT | recovery | (iii) within budget |
|---|---|---|---|---|---|---|
| frontier ×1.5 | 0.596 / $0.1016 / 0.475 | 0.596 / $0.1431 / 0.425 | 0.583 / $0.1024 / 0.460 | -0.050 | +0.035 | yes |
| mid ×0.5 | 0.596 / $0.1016 / 0.475 | 0.596 / $0.1003 / 0.476 | 0.626 / $0.1002 / 0.507 | +0.002 | +0.031 | yes |
| cheap ×3 | 0.596 / $0.1016 / 0.475 | 0.596 / $0.1337 / 0.437 | 0.617 / $0.1148 / 0.480 | -0.038 | +0.044 | NO |

Harness-version drift (v1 → v2) cannot be measured on this log: it holds a single harness version (crmarena-pro-v1) and nothing here re-runs the harness.

### Gate after frontier ×1.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.607 [0.525, 0.668] | 0.1329 [0.1216, 0.1566] |
| candidate | IPS | 0.538 [0.467, 0.613] | 0.0931 [0.0782, 0.1079] |
| candidate | SNIPS | 0.583 [0.510, 0.645] | 0.1009 [0.0874, 0.1133] |
| candidate | DR | 0.592 [0.500, 0.660] | 0.1000 [0.0870, 0.1143] |

criteria: DR-quality lower CI 0.500 >= q_prev - eps = 0.607 (recorded at incumbent's promotion) - 0.03 = 0.577 [FAIL];  DR-cost upper CI 0.1143 <= budget + tol = 0.1050 [FAIL]
(n = 300 logged episodes over 300 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after mid ×0.5: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.607 [0.525, 0.668] | 0.0936 [0.0850, 0.1108] |
| candidate | IPS | 0.559 [0.464, 0.654] | 0.0903 [0.0754, 0.1051] |
| candidate | SNIPS | 0.631 [0.565, 0.700] | 0.1020 [0.0890, 0.1145] |
| candidate | DR | 0.631 [0.544, 0.706] | 0.0955 [0.0874, 0.1130] |

criteria: DR-quality lower CI 0.544 >= q_prev - eps = 0.607 (recorded at incumbent's promotion) - 0.03 = 0.577 [FAIL];  DR-cost upper CI 0.1130 <= budget + tol = 0.1050 [FAIL]
(n = 300 logged episodes over 300 tasks, max cumulative ratio 2.9)

**DECISION: REVERT**

### Gate after cheap ×3: retrained vs frozen, repriced realized log

| policy | estimator | quality [90% CI] | cost [90% CI] |
|---|---|---|---|
| incumbent (DR) | - | 0.607 [0.525, 0.668] | 0.1289 [0.1105, 0.1505] |
| candidate | IPS | 0.552 [0.454, 0.646] | 0.1106 [0.0867, 0.1361] |
| candidate | SNIPS | 0.616 [0.547, 0.706] | 0.1235 [0.1044, 0.1475] |
| candidate | DR | 0.610 [0.526, 0.684] | 0.1225 [0.0983, 0.1553] |

criteria: DR-quality lower CI 0.526 >= q_prev - eps = 0.607 (recorded at incumbent's promotion) - 0.03 = 0.577 [FAIL];  DR-cost upper CI 0.1553 <= budget + tol = 0.1050 [FAIL]
(n = 300 logged episodes over 300 tasks, max cumulative ratio 3.0)

**DECISION: REVERT**

