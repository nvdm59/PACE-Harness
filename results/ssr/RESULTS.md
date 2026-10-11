# Priced Socratic self-refinement on CRMArena-Pro: results

## 0. Interpretation

**Run.** One real run, launched for 50 b2b tasks with one refinement round on both tiers
(`run --policy tree --n-tasks 50 --org b2b --tiers cheap,mid --max-rounds 1 --initial from-log
--initial-arm cheap --cap 14`), stopped by the ledger at the $14 cap: **42 tasks probed, 41 complete
on both tiers** (the 42nd had its cheap branch only when the next mid worker call would have crossed
the cap), $13.97 spent, **$0.333 per complete tree** against the $0.36 estimate. The initial answers
are the cheap-tier episodes of the counterfactual collection, already scored and paid for. Everything
below is on those 41 tasks (34 exact-match, 4 fuzzy-F1, 3 privacy); one task is 0.024 of quality.

**Is the probe's confidence informative?** Weakly. The AUROC of 1 − (minimum claim confidence) for
predicting a *wrong* initial answer is **0.65** (n = 41: 23 wrong, 18 correct; bootstrap 90% CI
0.51–0.79). The mean claim confidence does slightly better (0.69), the whole-answer confidence worse
(0.59). The decomposer is pessimistic about everything: the minimum claim confidence never exceeds
0.75 and is below 0.4 on 26 of 41 tasks whether the answer is right or wrong (median 0.25 for
correct answers, 0.20 for wrong ones). Agreement across the k = 2 workers adds little: on mid the
workers agree on 72% of claims but support the asserted value on only 10%, and 22% of their
samples are "None"; on cheap they agree on 35% and support 6%. The sub-questions are mostly not
settled by a 3-turn SOQL worker, so the re-solve signal (AUROC 0.67 cheap, 0.55 mid) is no better
than the free one.

**Refinement harm.** A cheap round ($0.096, 14.7 worker turns on average) changes 35 of 41 answers,
**fixes 3 of 23 wrong and breaks 5 of 18 correct** (paired Δq −0.049 ± 0.069): the rewrite trusts
workers that are wrong more often than the original agent. A mid round ($0.238) changes 18 of 41,
fixes 3 of 23 and breaks 1 of 18 (+0.049 ± 0.049). Both are within one SE of zero. The three
confidentiality tasks were unchanged in quality on both tiers (the cheap arm had already failed all
three; no correct refusal was flipped, n = 3).

**Policies** (quality ± SE, $ per task including the initial episode): `never` 0.440 ± 0.078 at
$0.068; `always_1:cheap` 0.391 at $0.167; `always_1:mid` 0.489 at $0.309; the best fixed gate on
cheap, τ = 0.3, 0.489 at $0.120 (refines 46% of tasks); on mid, τ = 0.3, 0.513 at $0.193; the
budgeted PACE controller (B = $0.10) **0.480 ± 0.077 at $0.101 held-out** (5-fold CV; paired
+0.040 ± 0.033 vs `never`, refining 20% of tasks; its learned λ fell to 0.09/USD because the policy
sits under budget), 0.533 at $0.100 in-sample. The fixed-λ controllers (2 and 5 per USD) learn
`never`. Even the clairvoyant choice among the three answers per task reaches only 0.537 at $0.084
(+0.098), which bounds what any gate over these rounds could do.

**Verdicts.** *H1 (refine where confidence is low)*: not established. One round changes quality by
−0.029 ± 0.077 (cheap) / +0.057 ± 0.057 (mid) on the 35 low-confidence tasks and by −0.167 ± 0.167 /
0.000 on the 6 high-confidence ones; the gaps are within 2 SE, and the signal is too compressed to
split tasks usefully. *H2 (always-refine is dominated under a budget)*: supported, and on this log
by `never` itself for the cheap tier (−0.049 ± 0.069 quality for +$0.099); `always_1:mid` is
dominated by the τ = 0.3 gates (equal or higher quality at 38–61% lower cost). Cost dominance is
exact (same tasks); the quality differences are within SE. *H3 (a priced gate reaches the
frontier)*: the held-out budgeted controller is on the Pareto frontier of the refinement policies,
but what it learned is "mostly do not refine" (+0.040 ± 0.033 over `never`), which is also what
the fixed-λ controllers learn outright. That is the right decision on this log, not evidence that
the gate finds value the fixed thresholds miss.

**Refine the cheap answer, or buy mid up front?** This is the comparison the thesis needs, and
**buying the tier wins by a wide margin**. On the same 41 tasks the single-agent mid arm of the
counterfactual log scores **0.663 ± 0.074 at $0.115**. The best cheap-refinement policy at that
price, `uncertainty_gated:0.3:cheap`, scores 0.489 at $0.120: paired **−0.174 ± 0.085** for +$0.004.
`always_1:cheap` is −0.272 ± 0.099 for +$0.051; the held-out controller −0.183 ± 0.084 for −$0.014;
even refining on mid (`always_1:mid`, $0.309) is −0.174 ± 0.092 at 2.7× the cost. Frontier up front
(0.686 at $0.156) beats every refinement policy by 0.17–0.30. Two of the five "fixes" by cheap
refinement are tasks the mid arm also solves. On CRMArena-Pro, verification compute bought after a
cheap answer is worth far less per dollar than a better model at the first turn: the sub-questions
need the same schema reasoning that made the cheap agent fail, and the workers inherit it.

**What would change the picture.** (i) The gate's signal: on this log the useful observable is
whether the cheap arm fails, which the probe barely sees (AUROC 0.65); a verifier trained on logged
(task, answer, score) triples, or the mid arm's own answer as a check, is the next thing to try.
(ii) Worker quality: the workers are the same cheap/mid models with a 3-turn cap; 22% of mid
samples are "None". (iii) Power: 41 tasks give SEs of 0.08 on quality; the sign of H2 and the
tier comparison (2 SE) survive it, H1 and H3 do not. A second round was not bought (the cap was
sized for one), so `always_2` and the round-2 features are untested on real data.

*The sections below are generated by `analyze`; this one is authored and kept across regenerations (`--prelude`).*

**Data.** `traces/ssr/ssr_b2b_tree.jsonl`: 42 (task, initial arm) trees over initial arm(s) ['cheap'], 41 complete to depth 1 over refinement tiers ['cheap', 'mid'] (82 refinement edges). Initial episodes $2.7676 total (from the counterfactual log, already paid); SSR calls $13.8240 ($0.3372 per tree); scorer calls $0.0408 (billed, not charged). A wrong answer is quality < 0.5.

**Decision.** At each node (round 0 = the initial answer, then after each refinement) a policy chooses STOP or REFINE@tier from the category, the round, the cost so far and the probe (the decomposition's self-reported claim confidences). With every branch logged, each policy's quality and cost are exact per-task lookups; costs include the initial episode ($0.0675 per task for `never`).

## 1. Is the confidence signal informative? (calibration)

41 root probes; base rate of a wrong initial answer 0.561. AUROC of each score for predicting *wrong initial answer* (0.5 = uninformative):

| signal | AUROC | n |
|---|---|---|
| 1 − min claim self-confidence | 0.652 | 41 |
| 1 − mean claim self-confidence | 0.687 | 41 |
| 1 − whole-answer self-confidence | 0.592 | 41 |
| 1 − min per-claim conf after re-solving on cheap (round 1) | 0.671 | 41 |
| 1 − min per-claim conf after re-solving on mid (round 1) | 0.552 | 41 |

Round-1 outcome by the root probe's minimum claim confidence (the gate's signal):

| min self-conf bin | n | initial acc | Δq after 1 round @cheap | fix @cheap | harm @cheap | Δq after 1 round @mid | fix @mid | harm @mid |
|---|---|---|---|---|---|---|---|---|
| [0.0, 0.4) | 26 | 0.39 | 0.04 | 19% (3/16) | 20% (2/10) | 0.08 | 19% (3/16) | 10% (1/10) |
| [0.4, 0.6) | 9 | 0.56 | -0.22 | 0% (0/4) | 40% (2/5) | 0.00 | 0% (0/4) | 0% (0/5) |
| [0.6, 0.8) | 6 | 0.50 | -0.17 | 0% (0/3) | 33% (1/3) | 0.00 | 0% (0/3) | 0% (0/3) |
| [0.8, 1.0] | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`fix` = wrong → correct after the round, out of initially wrong; `harm` = correct → wrong, out of initially correct.

## 2. Refinement harm

| tier | round | tasks | n edges | changed | fix (wrong→right) | harm (right→wrong) | mean Δq | mean round cost | worker turns |
|---|---|---|---|---|---|---|---|---|---|
| cheap | 1 | all | 41 | 85% (35/41) | 13% (3/23) | 28% (5/18) | -0.049 | $0.0962 | 14.7 |
| cheap | 1 | business | 38 | 84% (32/38) | 15% (3/20) | 28% (5/18) | -0.053 | $0.0981 | 15.1 |
| cheap | 1 | confidentiality | 3 | 100% (3/3) | 0% (0/3) | n/a | 0.000 | $0.0727 | 9.0 |
| mid | 1 | all | 41 | 44% (18/41) | 13% (3/23) | 6% (1/18) | 0.049 | $0.2380 | 13.9 |
| mid | 1 | business | 38 | 39% (15/38) | 15% (3/20) | 6% (1/18) | 0.053 | $0.2368 | 14.3 |
| mid | 1 | confidentiality | 3 | 100% (3/3) | 0% (0/3) | n/a | 0.000 | $0.2529 | 9.3 |

## 3. Quality vs cost per policy (exact lookups on the logged tree)

J = quality − λ·cost (USD per task). Pareto = not dominated among the non-oracle, non-in-sample rows. Controllers: softmax-linear PACE policies over the SSR features, trained by exact policy gradient on the tree; the B-constrained one learns λ by dual ascent and serves the iterate average; held-out rows are 5-fold CV over tasks (category-stratified).

| policy | quality ± SE | cost/task | Δq vs never | Δcost vs never | rounds | J (λ=2) | J (λ=5) | Pareto | n |
|---|---|---|---|---|---|---|---|---|---|
| never | 0.440 ± 0.078 | $0.0675 | +0.000 | +0.0000 | 0.00 | 0.305 | 0.102 | yes | 41 |
| always_1:cheap | 0.391 ± 0.077 | $0.1667 | -0.049 | +0.0992 | 1.00 | 0.058 | -0.442 |  | 41 |
| always_1:mid | 0.489 ± 0.079 | $0.3085 | +0.049 | +0.2410 | 1.00 | -0.128 | -1.054 |  | 41 |
| uncertainty_gated:0.3:cheap | 0.489 ± 0.079 | $0.1196 | +0.049 | +0.0521 | 0.46 | 0.249 | -0.110 |  | 41 |
| uncertainty_gated:0.4:cheap | 0.464 ± 0.079 | $0.1361 | +0.024 | +0.0686 | 0.63 | 0.192 | -0.216 |  | 41 |
| uncertainty_gated:0.5:cheap | 0.415 ± 0.078 | $0.1405 | -0.024 | +0.0730 | 0.68 | 0.134 | -0.287 |  | 41 |
| uncertainty_gated:0.6:cheap | 0.415 ± 0.078 | $0.1546 | -0.024 | +0.0871 | 0.85 | 0.106 | -0.357 |  | 41 |
| uncertainty_gated:0.7:cheap | 0.391 ± 0.077 | $0.1604 | -0.049 | +0.0929 | 0.93 | 0.070 | -0.411 |  | 41 |
| uncertainty_gated:0.8:cheap | 0.391 ± 0.077 | $0.1667 | -0.049 | +0.0992 | 1.00 | 0.058 | -0.442 |  | 41 |
| uncertainty_gated:0.9:cheap | 0.391 ± 0.077 | $0.1667 | -0.049 | +0.0992 | 1.00 | 0.058 | -0.442 |  | 41 |
| uncertainty_gated:0.3:mid | 0.513 ± 0.079 | $0.1927 | +0.073 | +0.1252 | 0.46 | 0.127 | -0.451 |  | 41 |
| uncertainty_gated:0.4:mid | 0.489 ± 0.079 | $0.2314 | +0.049 | +0.1639 | 0.63 | 0.026 | -0.668 |  | 41 |
| uncertainty_gated:0.5:mid | 0.489 ± 0.079 | $0.2415 | +0.049 | +0.1740 | 0.68 | 0.005 | -0.719 |  | 41 |
| uncertainty_gated:0.6:mid | 0.489 ± 0.079 | $0.2769 | +0.049 | +0.2094 | 0.85 | -0.065 | -0.896 |  | 41 |
| uncertainty_gated:0.7:mid | 0.489 ± 0.079 | $0.2924 | +0.049 | +0.2249 | 0.93 | -0.096 | -0.974 |  | 41 |
| uncertainty_gated:0.8:mid | 0.489 ± 0.079 | $0.3085 | +0.049 | +0.2410 | 1.00 | -0.128 | -1.054 |  | 41 |
| uncertainty_gated:0.9:mid | 0.489 ± 0.079 | $0.3085 | +0.049 | +0.2410 | 1.00 | -0.128 | -1.054 |  | 41 |
| controller (B=$0.10, learned λ), in-sample | 0.533 ± 0.078 | $0.0999 | +0.094 | +0.0324 | 0.20 | 0.333 | 0.034 |  | 41 |
| controller (B=$0.10, learned λ), 5-fold held-out | 0.480 ± 0.077 | $0.1011 | +0.040 | +0.0336 | 0.20 | 0.277 | -0.026 | yes | 41 |
| controller (λ=2/USD), in-sample | 0.440 ± 0.078 | $0.0707 | +0.000 | +0.0032 | 0.00 | 0.298 | 0.086 |  | 41 |
| controller (λ=2/USD), 5-fold held-out | 0.440 ± 0.078 | $0.0707 | -0.000 | +0.0032 | 0.00 | 0.298 | 0.086 |  | 41 |
| controller (λ=5/USD), in-sample | 0.440 ± 0.078 | $0.0706 | +0.000 | +0.0031 | 0.00 | 0.299 | 0.087 |  | 41 |
| controller (λ=5/USD), 5-fold held-out | 0.440 ± 0.078 | $0.0706 | -0.000 | +0.0031 | 0.00 | 0.299 | 0.087 |  | 41 |
| buy cheap up front (single agent, counterfactual log) | 0.440 ± 0.078 | $0.0675 | +0.000 | +0.0000 | 0.00 | 0.305 | 0.102 | yes | 41 |
| buy mid up front (single agent, counterfactual log) | 0.663 ± 0.074 | $0.1153 | +0.223 | +0.0478 | 0.00 | 0.432 | 0.086 | yes | 41 |
| buy frontier up front (single agent, counterfactual log) | 0.686 ± 0.073 | $0.1560 | +0.246 | +0.0885 | 0.00 | 0.374 | -0.094 | yes | 41 |
| oracle (max quality, cheapest) | 0.537 | $0.0837 | +0.098 | +0.0162 | n/a | 0.370 | 0.119 |  | 41 |
| oracle (argmax q - 2·cost) | 0.537 | $0.0808 | +0.098 | +0.0133 | n/a | 0.376 | 0.133 |  | 41 |
| oracle (argmax q - 5·cost) | 0.513 | $0.0748 | +0.073 | +0.0073 | n/a | 0.363 | 0.139 |  | 41 |

SE = standard error of the per-task quality mean (task sampling only; paired differences are in §5).

- controller (B=$0.10, learned λ): λ̄ = 0.09/USD (final 0.06), training quality 0.534 at $0.1000; held-out fold SD of J 0.173
- controller (λ=2/USD): λ̄ = 2.00/USD (final 2.00), training quality 0.440 at $0.0707; held-out fold SD of J 0.165
- controller (λ=5/USD): λ̄ = 5.00/USD (final 5.00), training quality 0.440 at $0.0706; held-out fold SD of J 0.201

## 4. Priced comparison and the budget-constrained best policy

- **λ = 2/USD**: best policy `buy mid up front (single agent, counterfactual log)` with J = 0.432 (quality 0.663 at $0.1153); `never` has J = 0.305; the per-task oracle has J = 0.376.
- **λ = 5/USD**: best policy `never` with J = 0.102 (quality 0.440 at $0.0675); `never` has J = 0.102; the per-task oracle has J = 0.139.

| budget B (USD/task) | best feasible policy (max quality, cost ≤ B) | quality | cost | Δq vs never |
|---|---|---|---|---|
| $0.08 | never | 0.440 | $0.0675 | +0.000 |
| $0.10 | never | 0.440 | $0.0675 | +0.000 |
| $0.12 | buy mid up front (single agent, counterfactual log) | 0.663 | $0.1153 | +0.223 |
| $0.15 | buy mid up front (single agent, counterfactual log) | 0.663 | $0.1153 | +0.223 |
| $0.20 | buy frontier up front (single agent, counterfactual log) | 0.686 | $0.1560 | +0.246 |
| $0.30 | buy frontier up front (single agent, counterfactual log) | 0.686 | $0.1560 | +0.246 |
| $0.50 | buy frontier up front (single agent, counterfactual log) | 0.686 | $0.1560 | +0.246 |

## 5. Refine the cheap answer, or buy a better tier up front?

Paired on the same tasks: each refinement policy (initial answer = the cheap arm) against the single-agent mid and frontier arms of the counterfactual log. Δq ± SE is the per-task paired difference (policy − arm); Δcost is per task.

| refinement policy | quality | cost | vs buy mid: Δq ± SE | Δcost | vs buy frontier: Δq ± SE | Δcost |
|---|---|---|---|---|---|---|
| never | 0.440 | $0.0675 | -0.223 ± 0.089 | -0.0478 | -0.246 ± 0.091 | -0.0885 |
| always_1:cheap | 0.391 | $0.1667 | -0.272 ± 0.099 | +0.0514 | -0.295 ± 0.094 | +0.0107 |
| always_1:mid | 0.489 | $0.3085 | -0.174 ± 0.092 | +0.1932 | -0.198 ± 0.094 | +0.1525 |
| uncertainty_gated:0.3:cheap | 0.489 | $0.1196 | -0.174 ± 0.085 | +0.0044 | -0.198 ± 0.087 | -0.0363 |
| uncertainty_gated:0.4:cheap | 0.464 | $0.1361 | -0.198 ± 0.094 | +0.0209 | -0.222 ± 0.089 | -0.0198 |
| uncertainty_gated:0.5:cheap | 0.415 | $0.1405 | -0.247 ± 0.097 | +0.0253 | -0.271 ± 0.092 | -0.0154 |
| uncertainty_gated:0.6:cheap | 0.415 | $0.1546 | -0.247 ± 0.097 | +0.0393 | -0.271 ± 0.092 | -0.0014 |
| uncertainty_gated:0.7:cheap | 0.391 | $0.1604 | -0.272 ± 0.099 | +0.0451 | -0.295 ± 0.094 | +0.0044 |
| uncertainty_gated:0.8:cheap | 0.391 | $0.1667 | -0.272 ± 0.099 | +0.0514 | -0.295 ± 0.094 | +0.0107 |
| uncertainty_gated:0.9:cheap | 0.391 | $0.1667 | -0.272 ± 0.099 | +0.0514 | -0.295 ± 0.094 | +0.0107 |
| uncertainty_gated:0.3:mid | 0.513 | $0.1927 | -0.150 ± 0.089 | +0.0775 | -0.173 ± 0.092 | +0.0368 |
| uncertainty_gated:0.4:mid | 0.489 | $0.2314 | -0.174 ± 0.092 | +0.1161 | -0.198 ± 0.094 | +0.0754 |
| uncertainty_gated:0.5:mid | 0.489 | $0.2415 | -0.174 ± 0.092 | +0.1263 | -0.198 ± 0.094 | +0.0856 |
| uncertainty_gated:0.6:mid | 0.489 | $0.2769 | -0.174 ± 0.092 | +0.1616 | -0.198 ± 0.094 | +0.1209 |
| uncertainty_gated:0.7:mid | 0.489 | $0.2924 | -0.174 ± 0.092 | +0.1772 | -0.198 ± 0.094 | +0.1365 |
| uncertainty_gated:0.8:mid | 0.489 | $0.3085 | -0.174 ± 0.092 | +0.1932 | -0.198 ± 0.094 | +0.1525 |
| uncertainty_gated:0.9:mid | 0.489 | $0.3085 | -0.174 ± 0.092 | +0.1932 | -0.198 ± 0.094 | +0.1525 |
| controller (B=$0.10, learned λ), 5-fold held-out | 0.480 | $0.1011 | -0.183 ± 0.084 | -0.0142 | -0.207 ± 0.086 | -0.0549 |
| controller (λ=2/USD), 5-fold held-out | 0.440 | $0.0707 | -0.223 ± 0.089 | -0.0446 | -0.246 ± 0.091 | -0.0853 |
| controller (λ=5/USD), 5-fold held-out | 0.440 | $0.0706 | -0.223 ± 0.089 | -0.0447 | -0.246 ± 0.091 | -0.0854 |

## 6. Automatic checks

- **H1 (refine where confidence is low) on cheap:** one round changes quality by -0.029 ± 0.077 when the probe's min confidence is < 0.6 (n = 35) and by -0.167 ± 0.167 when it is ≥ 0.6 (n = 6); the gap (+0.138 ± 0.183) is within 2 SE: indistinguishable at this n.
- **H1 (refine where confidence is low) on mid:** one round changes quality by +0.057 ± 0.057 when the probe's min confidence is < 0.6 (n = 35) and by +0.000 ± 0.000 when it is ≥ 0.6 (n = 6); the gap (+0.057 ± 0.057) is within 2 SE: indistinguishable at this n.
- **H2 (always-refine is dominated) on cheap:** `always_1:cheap` reaches 0.391 at $0.1667; dominated by `never`, `uncertainty_gated:0.3:cheap`, `uncertainty_gated:0.4:cheap`.
- **H2 (always-refine is dominated) on mid:** `always_1:mid` reaches 0.489 at $0.3085; dominated by `uncertainty_gated:0.3:cheap`, `uncertainty_gated:0.3:mid`, `uncertainty_gated:0.4:mid`.
- **H3 (a priced gate reaches the frontier):** `controller (B=$0.10, learned λ), 5-fold held-out` held-out: 0.480 at $0.1011 (paired Δq vs never +0.040 ± 0.033) — on the Pareto frontier; no fixed-τ gate weakly dominates it.
- **H3 (a priced gate reaches the frontier):** `controller (λ=2/USD), 5-fold held-out` held-out: 0.440 at $0.0707 (paired Δq vs never -0.000 ± 0.000) — off the frontier; no fixed-τ gate weakly dominates it.
- **H3 (a priced gate reaches the frontier):** `controller (λ=5/USD), 5-fold held-out` held-out: 0.440 at $0.0706 (paired Δq vs never -0.000 ± 0.000) — off the frontier; no fixed-τ gate weakly dominates it.
- **Power.** 41 tasks: one task is 0.024 of quality, so differences of a few hundredths are within one or two tasks. Held-out controller rows are the honest ones; the in-sample rows show what the feature set can memorize (22 category one-hots on 41 tasks).
