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

