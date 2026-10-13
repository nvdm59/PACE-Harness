## Cross-fitted DR gate: CRMArena-Pro B2B, 300 tasks

Cross-fitted gate: per fold, candidate = constrained cgrpo-lookup trained on the training folds; gated on the held-out fold's realised one-arm log (200 redraws, bootstrap 200, eps_q = 0.03, budget $0.10 + tol $0.005, cross-fitted DR direct model within the held-out log). Correct decision: PROMOTE iff exact held-out cost ≤ B + tol and exact quality ≥ exact uniform quality − eps_q.

| fold | n test | cand. exact q | cand. exact cost | uniform exact q | correct | promote rate | decision acc. | acc. (realised bar) | DR q bias | DR q RMSE | DR q CI width | DR q cover | DR c bias | DR c RMSE | DR c CI width | DR c cover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 60 | 0.558047 | 0.091778 | 0.499120 | PROMOTE | 0.040 | 0.040 | 0.040 | -0.001918 | 0.060952 | 0.291233 | 0.990 | +0.000260 | 0.008593 | 0.034281 | 0.940 |
| 1 | 60 | 0.613769 | 0.097627 | 0.527639 | PROMOTE | 0.055 | 0.055 | 0.060 | +0.001258 | 0.061129 | 0.278536 | 0.970 | +0.000183 | 0.010757 | 0.041958 | 0.905 |
| 2 | 60 | 0.568135 | 0.086217 | 0.489435 | PROMOTE | 0.170 | 0.170 | 0.175 | +0.004527 | 0.057593 | 0.281880 | 0.980 | -0.000560 | 0.008689 | 0.036065 | 0.905 |
| 3 | 60 | 0.588040 | 0.137796 | 0.503373 | REVERT | 0.000 | 1.000 | 1.000 | -0.003390 | 0.061107 | 0.292620 | 1.000 | -0.001997 | 0.026488 | 0.097759 | 0.835 |
| 4 | 60 | 0.651562 | 0.094816 | 0.530698 | PROMOTE | 0.100 | 0.100 | 0.100 | +0.004767 | 0.064168 | 0.284335 | 0.975 | +0.000511 | 0.008827 | 0.039955 | 0.955 |
| pooled | 300 | – | – | – | – | 0.073 | 0.273 | 0.275 | +0.001049 | 0.061025 | 0.285721 | 0.983 | -0.000321 | 0.014454 | 0.050004 | 0.908 |

Pooled over the 1000 gates (5 folds × 200 redraws), all estimators (error vs the fold's exact held-out value):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.004133 | 0.107617 | 0.384517 | 0.920 |
| quality | SNIPS | -0.000432 | 0.062332 | 0.266741 | 0.965 |
| quality | DR | +0.001049 | 0.061025 | 0.285721 | 0.983 |
| cost | IPS | +0.000311 | 0.020351 | 0.063439 | 0.892 |
| cost | SNIPS | -0.000132 | 0.015055 | 0.048631 | 0.902 |
| cost | DR | -0.000321 | 0.014454 | 0.050004 | 0.908 |
