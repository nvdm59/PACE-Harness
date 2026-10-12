# Simulator pipeline over 5 seeds

`scripts/run_sim_seeds.sh 0 1 2 3 4` (the `scripts/run_v0.sh` pipeline; replicate s uses component
seeds 10s+1..10s+4 for collect/train/collect/train and seed s for the gate bootstrap and the Monte Carlo
ground truth; s = 0 reproduces run_v0.sh). Aggregated by `scripts/aggregate_sim_seeds.py`. Mean ± SE over seeds;
budget B = 0.30, tol 0.02, eps_q 0.03; ground truth over 20,000 fresh tasks per condition.

## Summary

| quantity | mean ± SE |
|---|---|
| learned price λ̄ (v1) | 0.358 ± 0.027 |
| gate v1: DR quality (point / 90% CI lo / hi) | 0.884 ± 0.009 / 0.851 ± 0.011 / 0.910 ± 0.007 |
| gate v1: DR cost (point / hi) | 0.308 ± 0.001 / 0.311 ± 0.001 |
| gate v1: IPS / SNIPS quality | 0.760 ± 0.022 / 0.797 ± 0.014 |
| gate v1: behaviour log mean quality | 0.758 ± 0.003 |
| gate v1 decision | PROMOTE in 5/5; truth inside DR quality CI in 4/5 |
| truth (i) frozen on v1: quality / cost / J | 0.878 ± 0.001 / 0.306 ± 0.002 / 0.768 ± 0.008 |
| truth (ii) frozen on v2: quality / cost / J | 0.869 ± 0.001 / 0.413 ± 0.003 / 0.721 ± 0.011 |
| gate v2: DR quality (point / lo / hi) | 0.869 ± 0.004 / 0.847 ± 0.006 / 0.889 ± 0.002 |
| gate v2: DR cost (point / hi) | 0.298 ± 0.004 / 0.299 ± 0.004 |
| gate v2: incumbent (frozen) DR quality / cost | 0.865 ± 0.002 / 0.413 ± 0.003 |
| gate v2 decision | PROMOTE in 2/5; truth inside DR quality CI in 5/5 |
| truth, adapted policy on v2 (whether or not promoted): quality / cost / J | 0.867 ± 0.001 / 0.299 ± 0.004 / 0.760 ± 0.008 |
| truth (iii) policy served after the v2 gate: quality / cost / J | 0.868 ± 0.001 / 0.369 ± 0.028 / 0.737 ± 0.014 |
| priced-BWT = J(ii) − J(i) | -0.047 ± 0.003 |
| recovery = J(iii) − J(ii), served after the gate | 0.015 ± 0.009 |
| recovery the adapted policy would give if served | 0.039 ± 0.003 |

## Per seed

| seed | λ̄ | gate v1 DR q [CI] | DR cost hi | dec. | truth v1 q / c | gate v2 DR q [CI] | DR cost hi | dec. | truth v2 adapted q / c | priced-BWT | recovery (served) | recovery (adapted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343 | 0.851 [0.815, 0.887] | 0.309 | PROMOTE | 0.880 / 0.303 | 0.873 [0.854, 0.889] | 0.311 | PROMOTE | 0.871 / 0.312 | -0.045 | +0.033 | +0.033 |
| 1 | 0.379 | 0.880 [0.848, 0.911] | 0.310 | PROMOTE | 0.877 / 0.310 | 0.871 [0.851, 0.892] | 0.292 | PROMOTE | 0.864 / 0.292 | -0.049 | +0.042 | +0.042 |
| 2 | 0.278 | 0.901 [0.878, 0.921] | 0.313 | PROMOTE | 0.877 / 0.309 | 0.879 [0.860, 0.895] | 0.304 | REVERT | 0.869 / 0.304 | -0.041 | -0.000 | +0.034 |
| 3 | 0.443 | 0.883 [0.845, 0.905] | 0.313 | PROMOTE | 0.878 / 0.306 | 0.855 [0.825, 0.881] | 0.290 | REVERT | 0.864 / 0.288 | -0.057 | +0.001 | +0.051 |
| 4 | 0.346 | 0.903 [0.870, 0.925] | 0.308 | PROMOTE | 0.877 / 0.302 | 0.867 [0.847, 0.887] | 0.296 | REVERT | 0.866 / 0.296 | -0.042 | +0.000 | +0.033 |

## Reading

Gate v1 promotes in 5/5 seeds. Gate v2 promotes in 2/5. Every v2 revert fails on
quality only: the bar is the DR quality recorded at the v1 promotion minus eps_q, and in the reverting seeds
that v1 point estimate sits above the candidate's ground truth (seed 2: v1 DR 0.901 vs truth 0.877, seed 3: v1 DR 0.883 vs truth 0.878, seed 4: v1 DR 0.903 vs truth 0.877),
so the inherited bar is high while the adapted policy is within budget and within about 0.01 of the frozen
policy's quality. The reverts are false alarms of the quality criterion (the cost criterion passes in every seed).
Recovery is therefore reported two ways: for the policy served after the gate (zero when the gate reverts),
and for the adapted policy as if served (from drift_ungated.json, the same evaluation without the gate).
