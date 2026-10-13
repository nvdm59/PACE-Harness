## Table 1 at a common price: CRMArena-Pro B2C, 150 tasks, 5-fold CV (seed 0)

Held-out means over the 5 test folds (exact lookups). J(λ) = quality − λ·cost evaluated at the SAME λ for every row; [SD] is the across-fold SD of J (np.std, ddof 0). Feasibility flags are on the mean held-out cost: ≤ B = $0.100 and ≤ B + tol = $0.105; 'folds ok' counts folds whose own held-out cost is ≤ B + tol. Training λ̄ is shown for the constrained rows only (a training quantity, not the evaluation price). AWR+IPS and the per-category rule are trained at a fixed λ (2 or 5); each is still evaluated at both prices.

| policy | quality | cost (USD) | J(λ=2) [SD] | J(λ=5) [SD] | train λ̄ (mean; per fold) | cost ≤ B | cost ≤ B+tol | folds ok |
|---|---|---|---|---|---|---|---|---|
| fixed cheap | 0.469482 | 0.055764 | 0.357954 [0.078046] | 0.190663 [0.088984] | – | yes | yes | 5/5 |
| fixed mid | 0.507556 | 0.126325 | 0.254905 [0.109258] | -0.124071 [0.163556] | – | NO | NO | 1/5 |
| fixed frontier | 0.638337 | 0.143305 | 0.351727 [0.116086] | -0.078189 [0.164964] | – | NO | NO | 0/5 |
| uniform behavior | 0.538459 | 0.108465 | 0.321529 [0.090305] | -0.003866 [0.127115] | – | NO | NO | 2/5 |
| AWR+IPS linear, trained at λ=2 (served, stochastic) | 0.569138 | 0.098823 | 0.371492 [0.094946] | 0.075022 [0.131700] | – | yes | yes | 4/5 |
| AWR+IPS linear, trained at λ=2 (greedy) | 0.598049 | 0.081300 | 0.435449 [0.056519] | 0.191549 [0.068860] | – | yes | yes | 5/5 |
| per-category rule, train-fold argmax of q − 2c | 0.636149 | 0.081156 | 0.473837 [0.067267] | 0.230371 [0.088105] | – | yes | yes | 5/5 |
| AWR+IPS linear, trained at λ=5 (served, stochastic) | 0.541539 | 0.089735 | 0.362069 [0.097186] | 0.092866 [0.134365] | – | yes | yes | 4/5 |
| AWR+IPS linear, trained at λ=5 (greedy) | 0.529482 | 0.060783 | 0.407915 [0.076585] | 0.225566 [0.097931] | – | yes | yes | 5/5 |
| per-category rule, train-fold argmax of q − 5c | 0.602815 | 0.071623 | 0.459570 [0.077774] | 0.244701 [0.087320] | – | yes | yes | 5/5 |
| constrained cgrpo-lookup, B=$0.10 (served: iterate mix, stochastic) | 0.617428 | 0.103341 | 0.410746 [0.096789] | 0.100725 [0.148954] | 1.212746; 1.487682, 1.257945, 0.809593, 1.174319, 1.334192 | NO | yes | 4/5 |
| constrained cgrpo-lookup, B=$0.10 (greedy of the mix) | 0.676803 | 0.105198 | 0.466407 [0.122821] | 0.150814 [0.202866] | 1.212746; 1.487682, 1.257945, 0.809593, 1.174319, 1.334192 | NO | NO | 4/5 |
| per-task oracle (argmax q − 2c) | 0.727060 | 0.060748 | 0.605564 [0.068344] | 0.423321 [0.090493] | – | yes | yes | 5/5 |
| per-task oracle (argmax q − 5c) | 0.718868 | 0.058314 | 0.602239 [0.066878] | 0.427296 [0.091997] | – | yes | yes | 5/5 |
| per-task oracle (max quality, cheapest among ties) | 0.728165 | 0.061860 | 0.604445 [0.067450] | 0.418865 [0.087246] | – | yes | yes | 5/5 |

### Constrained rows: common price vs fold λ̄

- served (iterate mix): J(λ=2) = 0.410746; J at each fold's λ̄ = 0.493005
- greedy of the mix: J(λ=2) = 0.466407; J at each fold's λ̄ = 0.549384

### λ̄ provenance

- CV, per fold: 1.487682, 1.257945, 0.809593, 1.174319, 1.334192; mean **1.212746**
- the all-150-task constrained policy of the gate / drift sections (results.json gate.lambda_bar): **1.351188**
- lines of tables.md containing '1.35': [71, 120]

### Reproduction check against the committed results.json

max |Δ| over re-derived q and c: 0.000e+00; max |Δ| fold λ̄: 0.000e+00.
