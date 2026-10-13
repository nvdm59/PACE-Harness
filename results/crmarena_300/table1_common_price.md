## Table 1 at a common price: CRMArena-Pro B2B, 300 tasks, 5-fold CV (seed 0)

Held-out means over the 5 test folds (exact lookups). J(λ) = quality − λ·cost evaluated at the SAME λ for every row; [SD] is the across-fold SD of J (np.std, ddof 0). Feasibility flags are on the mean held-out cost: ≤ B = $0.100 and ≤ B + tol = $0.105; 'folds ok' counts folds whose own held-out cost is ≤ B + tol. Training λ̄ is shown for the constrained rows only (a training quantity, not the evaluation price). AWR+IPS and the per-category rule are trained at a fixed λ (2 or 5); each is still evaluated at both prices.

| policy | quality | cost (USD) | J(λ=2) [SD] | J(λ=5) [SD] | train λ̄ (mean; per fold) | cost ≤ B | cost ≤ B+tol | folds ok |
|---|---|---|---|---|---|---|---|---|
| fixed cheap | 0.433824 | 0.056861 | 0.320103 [0.062889] | 0.149520 [0.086068] | – | yes | yes | 5/5 |
| fixed mid | 0.481997 | 0.126139 | 0.229719 [0.026374] | -0.148700 [0.038794] | – | NO | NO | 0/5 |
| fixed frontier | 0.614337 | 0.130475 | 0.353387 [0.045979] | -0.038038 [0.091368] | – | NO | NO | 0/5 |
| uniform behavior | 0.510053 | 0.104492 | 0.301069 [0.027510] | -0.012406 [0.058792] | – | NO | yes | 3/5 |
| AWR+IPS linear, trained at λ=2 (served, stochastic) | 0.547673 | 0.097725 | 0.352224 [0.028707] | 0.059049 [0.053724] | – | yes | yes | 4/5 |
| AWR+IPS linear, trained at λ=2 (greedy) | 0.596417 | 0.079596 | 0.437225 [0.027917] | 0.198437 [0.024043] | – | yes | yes | 5/5 |
| per-category rule, train-fold argmax of q − 2c | 0.643824 | 0.078944 | 0.485936 [0.035347] | 0.249104 [0.033417] | – | yes | yes | 5/5 |
| AWR+IPS linear, trained at λ=5 (served, stochastic) | 0.535805 | 0.087911 | 0.359984 [0.033859] | 0.096252 [0.054801] | – | yes | yes | 4/5 |
| AWR+IPS linear, trained at λ=5 (greedy) | 0.513824 | 0.068046 | 0.377733 [0.045995] | 0.173596 [0.058758] | – | yes | yes | 5/5 |
| per-category rule, train-fold argmax of q − 5c | 0.587158 | 0.072404 | 0.442349 [0.042997] | 0.225136 [0.056923] | – | yes | yes | 5/5 |
| constrained cgrpo-lookup, B=$0.10 (served: iterate mix, stochastic) | 0.595910 | 0.101647 | 0.392617 [0.049478] | 0.087677 [0.097623] | 1.220572; 1.708155, 0.513796, 1.693236, 1.070859, 1.116812 | NO | yes | 4/5 |
| constrained cgrpo-lookup, B=$0.10 (greedy of the mix) | 0.643530 | 0.102525 | 0.438480 [0.081547] | 0.130907 [0.163865] | 1.220572; 1.708155, 0.513796, 1.693236, 1.070859, 1.116812 | NO | yes | 4/5 |
| per-task oracle (argmax q − 2c) | 0.708454 | 0.058359 | 0.591736 [0.031725] | 0.416658 [0.029926] | – | yes | yes | 5/5 |
| per-task oracle (argmax q − 5c) | 0.704917 | 0.057128 | 0.590661 [0.030891] | 0.419276 [0.031365] | – | yes | yes | 5/5 |
| per-task oracle (max quality, cheapest among ties) | 0.711332 | 0.067186 | 0.576960 [0.041983] | 0.375402 [0.071972] | – | yes | yes | 5/5 |

### Constrained rows: common price vs fold λ̄

- served (iterate mix): J(λ=2) = 0.392617; J at each fold's λ̄ = 0.474637
- greedy of the mix: J(λ=2) = 0.438480; J at each fold's λ̄ = 0.520514

### λ̄ provenance

- CV, per fold: 1.708155, 0.513796, 1.693236, 1.070859, 1.116812; mean **1.220572**
- the all-300-task constrained policy of the gate / drift sections (results.json gate.lambda_bar): **1.088964**
- lines of tables.md containing '1.35': none

### Reproduction check against the committed results.json

max |Δ| over re-derived q and c: 0.000e+00; max |Δ| fold λ̄: 0.000e+00.
