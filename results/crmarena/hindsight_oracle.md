# Hindsight (luck) component of the realised per-task oracle: CRMArena-Pro B2B, first 50 tasks

**An estimate under stated assumptions, not a measurement.** Produced by `scripts/crmarena_hindsight_oracle.py` from `traces/cheap_rerun_b2b.jsonl` (a second cheap-tier run on the first 50 B2B tasks) and `traces/crmarena_b2b_counterfactual.jsonl` (run 1, all tiers).

## 1. Cheap-tier repeatability (measured)

- tasks with two cheap runs: 50 (task ids match the counterfactual log; arm = cheap in every rerun episode)
- correctness = quality ≥ 0.5 (the 50 × 3 cells hold 6 fractional text-similarity scores, all < 0.5)
- correctness agreement between the two runs: **0.800** (40/50); flip rate F = 0.200 (10 tasks: 8 wrong→right, 2 right→wrong; two-sided sign test p = 0.109). Exact raw-quality agreement: 0.760.
- cheap accuracy: run 1 0.440, run 2 0.560 (rerun better by +0.120; not significant at 10 flips, but drift between runs cannot be excluded)
- estimated per-task outcome variance (mean over tasks of (x1 − x2)²/2, unbiased for p(1 − p)): **0.1000** (0.25 would be a coin flip on every task; 0 fully repeatable)
- test-retest reliability (correlation of the two runs' correctness): ρ = **0.623** (task bootstrap 90% interval [0.445, 0.785], 2000 resamples, seed 0)

Realised on these 50 tasks (run 1, binarised): per-task oracle **0.7600**, best single tier 0.6600 (frontier); raw quality 0.7689 vs 0.6626. Realised oracle − best single = 0.1000.

## 2. Model and definitions

Each (task, tier) outcome X_it is Bernoulli(p_it). Luck (hindsight) = realised oracle − probability oracle, where the probability oracle picks per task the tier with the highest p_it (it knows each tier's success probability on the task, not the draw). Selection = probability oracle − best single tier. The realised oracle is observed (0.76 above), so the anchored luck estimate is R_obs − (estimated probability oracle); the model column E[realised oracle] = mean_i 1 − Π_t(1 − p_it) additionally assumes tiers independent given the task.

**Model K (primary): Kelley true-score shrinkage**, mean-preserving (each tier's mean accuracy is unchanged): p̂ = m_t + rel·(x̄ − m_t). Cheap: x̄ = mean of its two runs, rel = 2ρ/(1 + ρ) = 0.768 (Spearman-Brown). Mid / frontier: one run, reliability ρ_mf unknown, swept from 1 (no noise) over the cheap tier's ρ and its 90% interval.

| ρ_mf (mid/frontier reliability) | probability oracle | best single tier | selection | luck, anchored (R_obs − prob. oracle) | E[realised oracle] (model) | luck (model) |
|---|---|---|---|---|---|---|
| 1 (no noise on mid/frontier) | 0.7755 | 0.6600 (frontier) | 0.1155 | -0.0155 | 0.7755 | 0.0000 |
| 0.785 (cheap ρ, 95th pct) | 0.7279 | 0.6600 (frontier) | 0.0679 | 0.0321 | 0.8254 | 0.0975 |
| 0.623 (= cheap ρ) | 0.7164 | 0.6600 (frontier) | 0.0564 | 0.0436 | 0.8571 | 0.1407 |
| 0.445 (cheap ρ, 5th pct) | 0.7225 | 0.6600 (frontier) | 0.0625 | 0.0375 | 0.8862 | 0.1637 |

**Model F (the reviewer's band): realised outcome ± flip probability.** Cheap p = mean of its two runs; mid / frontier p = (1 − δ)x + δ(1 − x). This shrinks tier means toward 0.5, so only the anchored luck is reported.

| δ | probability oracle | luck, anchored (R_obs − prob. oracle) |
|---|---|---|
| 0 | 0.7500 | 0.0100 |
| δ* = 0.1127 (2δ(1−δ) = F) | 0.7387 | 0.0213 |
| F = 0.2000 | 0.7300 | 0.0300 |

## 3. Estimate

- **Hindsight (luck) component of the realised per-task oracle on these 50 tasks: about 0.021 to 0.044 quality points** (anchored estimates over Model K with ρ_mf in the cheap tier's 90% reliability interval, and Model F with δ ∈ {δ*, F}), against a realised oracle − best single tier gap of 0.100. With no noise on mid/frontier (ρ_mf = 1, δ = 0) it is -0.016–0.010, the cheap tier's own contribution.
- Upper sensitivity: the model-internal luck (E[realised oracle] − probability oracle, tiers independent) is 0.098–0.164 for ρ_mf < 1. But that model predicts a realised oracle of 0.825–0.886, above the observed 0.760: the data favour positively correlated outcomes across tiers (hard tasks fail everywhere) and/or more repeatable mid/frontier tiers, either of which lowers luck, so the anchored figures are the estimate and the model-internal ones an upper bound under independence.
- Expected realised best-of-three vs expected best single tier: best single tier stays 0.660 (frontier) under Model K (mean-preserving); the expected realised oracle is 0.760 as observed (anchored), 0.776–0.886 under the independent-tier model.
- Assumptions: (i) mid/frontier noise is not measured; ρ_mf / δ borrow the cheap tier's repeatability, and stronger tiers may be more (or less) repeatable; (ii) the probability oracle is estimated by a plug-in max over shrunk p̂, itself an estimate (max of noisy estimates is biased upward, so luck may be understated); (iii) the model E[realised oracle] assumes independence across tiers given the task; positively correlated outcomes make it smaller; (iv) 50 tasks, the first 50 of the collection, not a random sample of the 300; (v) the cheap rerun scored higher than run 1, so part of the flip rate may be drift rather than noise; (vi) correctness is binarised at 0.5.
- Implication: part of the realised per-task oracle's lead over any outcome-blind policy is luck; a fairer ceiling is the probability oracle, whose estimation needs repeated runs on every tier.
