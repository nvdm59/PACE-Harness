## Cross-fitted DR gate: CRMArena-Pro B2C, 150 tasks

Cross-fitted gate: per fold, candidate = constrained cgrpo-lookup trained on the training folds; gated on the held-out fold's realised one-arm log (200 redraws, bootstrap 200, eps_q = 0.03, budget $0.10 + tol $0.005, cross-fitted DR direct model within the held-out log). Correct decision: PROMOTE iff exact held-out cost ≤ B + tol and exact quality ≥ exact uniform quality − eps_q.

| fold | n test | cand. exact q | cand. exact cost | uniform exact q | correct | promote rate | decision acc. | acc. (realised bar) | DR q bias | DR q RMSE | DR q CI width | DR q cover | DR c bias | DR c RMSE | DR c CI width | DR c cover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 30 | 0.672883 | 0.087235 | 0.593505 | PROMOTE | 0.065 | 0.065 | 0.090 | -0.000328 | 0.084173 | 0.384338 | 0.975 | +0.001807 | 0.013979 | 0.054328 | 0.935 |
| 1 | 30 | 0.574655 | 0.099174 | 0.456734 | PROMOTE | 0.005 | 0.005 | 0.005 | +0.005021 | 0.105672 | 0.430660 | 0.950 | -0.000755 | 0.017410 | 0.063073 | 0.900 |
| 2 | 30 | 0.649414 | 0.097186 | 0.547683 | PROMOTE | 0.010 | 0.010 | 0.020 | +0.005819 | 0.071650 | 0.353779 | 0.980 | -0.001125 | 0.013505 | 0.058112 | 0.945 |
| 3 | 30 | 0.513593 | 0.138305 | 0.466806 | REVERT | 0.000 | 1.000 | 1.000 | -0.000508 | 0.086421 | 0.448335 | 0.990 | -0.001918 | 0.028667 | 0.111452 | 0.910 |
| 4 | 30 | 0.676594 | 0.094803 | 0.627564 | PROMOTE | 0.035 | 0.035 | 0.080 | +0.003613 | 0.086917 | 0.399710 | 0.975 | +0.000139 | 0.012267 | 0.054043 | 0.940 |
| pooled | 150 | – | – | – | – | 0.023 | 0.223 | 0.239 | +0.002723 | 0.087645 | 0.403364 | 0.974 | -0.000370 | 0.018183 | 0.068202 | 0.926 |

Pooled over the 1000 gates (5 folds × 200 redraws), all estimators (error vs the fold's exact held-out value):

| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |
|---|---|---|---|---|---|
| quality | IPS | +0.007073 | 0.142366 | 0.529512 | 0.919 |
| quality | SNIPS | -0.000104 | 0.087030 | 0.361832 | 0.963 |
| quality | DR | +0.002723 | 0.087645 | 0.403364 | 0.974 |
| cost | IPS | +0.001000 | 0.024161 | 0.082371 | 0.886 |
| cost | SNIPS | +0.000273 | 0.018085 | 0.062950 | 0.895 |
| cost | DR | -0.000370 | 0.018183 | 0.068202 | 0.926 |
