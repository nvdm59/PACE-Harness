# Priced Socratic self-refinement on CRMArena-Pro: results

> **DRY RUN.** Every number below comes from the deterministic fake world in `pace/experiments/ssr/fake.py` (no API, no Salesforce, no spend). The structure of the tables is real; the values are not evidence about SSR on CRMArena. Re-run `analyze` on a real tree log to replace this file.

**Data.** `results/ssr/dryrun/ssr_tree.jsonl`: 50 (task, initial arm) trees over initial arm(s) ['cheap'], 50 complete to depth 2 over refinement tiers ['cheap', 'mid'] (300 refinement edges). Initial episodes $1.5353 total (fake); SSR calls $35.4325 ($0.7087 per tree); scorer calls $0.0000 (billed, not charged). A wrong answer is quality < 0.5.

**Decision.** At each node (round 0 = the initial answer, then after each refinement) a policy chooses STOP or REFINE@tier from the category, the round, the cost so far and the probe (the decomposition's self-reported claim confidences). With every branch logged, each policy's quality and cost are exact per-task lookups; costs include the initial episode ($0.0307 per task for `never`).

## 1. Is the confidence signal informative? (calibration)

50 root probes; base rate of a wrong initial answer 0.640. AUROC of each score for predicting *wrong initial answer* (0.5 = uninformative):

| signal | AUROC | n |
|---|---|---|
| 1 − min claim self-confidence | 0.838 | 50 |
| 1 − mean claim self-confidence | 0.788 | 50 |
| 1 − whole-answer self-confidence | 0.816 | 50 |
| 1 − min per-claim conf after re-solving on cheap (round 1) | 0.627 | 50 |
| 1 − min per-claim conf after re-solving on mid (round 1) | 0.821 | 50 |

Round-1 outcome by the root probe's minimum claim confidence (the gate's signal):

| min self-conf bin | n | initial acc | Δq after 1 round @cheap | fix @cheap | harm @cheap | Δq after 1 round @mid | fix @mid | harm @mid |
|---|---|---|---|---|---|---|---|---|
| [0.0, 0.4) | 18 | 0.00 | 0.72 | 72% (13/18) | n/a | 0.89 | 89% (16/18) | n/a |
| [0.4, 0.6) | 22 | 0.59 | 0.09 | 22% (2/9) | 0% (0/13) | 0.27 | 78% (7/9) | 8% (1/13) |
| [0.6, 0.8) | 9 | 0.44 | 0.22 | 80% (4/5) | 50% (2/4) | 0.33 | 60% (3/5) | 0% (0/4) |
| [0.8, 1.0] | 1 | 1.00 | 0.00 | n/a | 0% (0/1) | 0.00 | n/a | 0% (0/1) |

`fix` = wrong → correct after the round, out of initially wrong; `harm` = correct → wrong, out of initially correct.

## 2. Refinement harm

| tier | round | tasks | n edges | changed | fix (wrong→right) | harm (right→wrong) | mean Δq | mean round cost | worker turns |
|---|---|---|---|---|---|---|---|---|---|
| cheap | 1 | all | 50 | 42% (21/50) | 59% (19/32) | 11% (2/18) | 0.340 | $0.0717 | 9.8 |
| cheap | 1 | business | 44 | 45% (20/44) | 62% (18/29) | 13% (2/15) | 0.364 | $0.0723 | 9.8 |
| cheap | 1 | confidentiality | 6 | 17% (1/6) | 33% (1/3) | 0% (0/3) | 0.167 | $0.0674 | 9.3 |
| cheap | 2 | all | 100 | 16% (16/100) | 36% (8/22) | 10% (8/78) | 0.000 | $0.0705 | 9.6 |
| cheap | 2 | business | 88 | 15% (13/88) | 28% (5/18) | 11% (8/70) | -0.034 | $0.0700 | 9.5 |
| cheap | 2 | confidentiality | 12 | 25% (3/12) | 75% (3/4) | 0% (0/8) | 0.250 | $0.0742 | 10.0 |
| mid | 1 | all | 50 | 54% (27/50) | 81% (26/32) | 6% (1/18) | 0.500 | $0.1651 | 9.8 |
| mid | 1 | business | 44 | 55% (24/44) | 83% (24/29) | 0% (0/15) | 0.545 | $0.1661 | 9.8 |
| mid | 1 | confidentiality | 6 | 50% (3/6) | 67% (2/3) | 33% (1/3) | 0.167 | $0.1576 | 9.3 |
| mid | 2 | all | 100 | 13% (13/100) | 59% (13/22) | 0% (0/78) | 0.130 | $0.1620 | 9.6 |
| mid | 2 | business | 88 | 14% (12/88) | 67% (12/18) | 0% (0/70) | 0.136 | $0.1612 | 9.5 |
| mid | 2 | confidentiality | 12 | 8% (1/12) | 25% (1/4) | 0% (0/8) | 0.083 | $0.1675 | 10.0 |

## 3. Quality vs cost per policy (exact lookups on the logged tree)

J = quality − λ·cost (USD per task). Pareto = not dominated among the non-oracle, non-in-sample rows. Controllers: softmax-linear PACE policies over the SSR features, trained by exact policy gradient on the tree; the B-constrained one learns λ by dual ascent and serves the iterate average; held-out rows are 5-fold CV over tasks (category-stratified).

| policy | quality | cost/task | Δq vs never | Δcost vs never | rounds | J (λ=2) | J (λ=5) | Pareto | n |
|---|---|---|---|---|---|---|---|---|---|
| never | 0.360 | $0.0307 | +0.000 | +0.0000 | 0.00 | 0.299 | 0.206 | yes | 50 |
| always_1:cheap | 0.700 | $0.1047 | +0.340 | +0.0740 | 1.00 | 0.491 | 0.176 |  | 50 |
| always_2:cheap | 0.740 | $0.1773 | +0.380 | +0.1465 | 2.00 | 0.385 | -0.146 |  | 50 |
| always_1:mid | 0.860 | $0.1981 | +0.500 | +0.1674 | 1.00 | 0.464 | -0.131 | yes | 50 |
| always_2:mid | 0.900 | $0.3615 | +0.540 | +0.3308 | 2.00 | 0.177 | -0.908 |  | 50 |
| uncertainty_gated:0.3:cheap | 0.440 | $0.0426 | +0.080 | +0.0119 | 0.12 | 0.355 | 0.227 | yes | 50 |
| uncertainty_gated:0.4:cheap | 0.640 | $0.0699 | +0.280 | +0.0392 | 0.46 | 0.500 | 0.291 | yes | 50 |
| uncertainty_gated:0.5:cheap | 0.660 | $0.1034 | +0.300 | +0.0727 | 0.94 | 0.453 | 0.143 | yes | 50 |
| uncertainty_gated:0.6:cheap | 0.680 | $0.1392 | +0.320 | +0.1085 | 1.44 | 0.402 | -0.016 |  | 50 |
| uncertainty_gated:0.7:cheap | 0.760 | $0.1623 | +0.400 | +0.1316 | 1.78 | 0.435 | -0.052 | yes | 50 |
| uncertainty_gated:0.8:cheap | 0.760 | $0.1737 | +0.400 | +0.1430 | 1.94 | 0.413 | -0.108 |  | 50 |
| uncertainty_gated:0.9:cheap | 0.740 | $0.1773 | +0.380 | +0.1465 | 2.00 | 0.385 | -0.146 |  | 50 |
| uncertainty_gated:0.3:mid | 0.460 | $0.0507 | +0.100 | +0.0200 | 0.10 | 0.359 | 0.207 | yes | 50 |
| uncertainty_gated:0.4:mid | 0.700 | $0.1038 | +0.340 | +0.0731 | 0.40 | 0.492 | 0.181 | yes | 50 |
| uncertainty_gated:0.5:mid | 0.800 | $0.1834 | +0.440 | +0.1527 | 0.88 | 0.433 | -0.117 | yes | 50 |
| uncertainty_gated:0.6:mid | 0.840 | $0.2711 | +0.480 | +0.2404 | 1.42 | 0.298 | -0.516 |  | 50 |
| uncertainty_gated:0.7:mid | 0.900 | $0.3338 | +0.540 | +0.3030 | 1.82 | 0.232 | -0.769 | yes | 50 |
| uncertainty_gated:0.8:mid | 0.900 | $0.3533 | +0.540 | +0.3226 | 1.94 | 0.193 | -0.867 |  | 50 |
| uncertainty_gated:0.9:mid | 0.900 | $0.3615 | +0.540 | +0.3308 | 2.00 | 0.177 | -0.908 |  | 50 |
| controller (B=$0.10, learned λ), in-sample | 0.834 | $0.1002 | +0.474 | +0.0695 | 0.74 | 0.634 | 0.333 |  | 50 |
| controller (B=$0.10, learned λ), 5-fold held-out | 0.656 | $0.1044 | +0.296 | +0.0737 | 0.72 | 0.447 | 0.134 |  | 50 |
| controller (λ=2/USD), in-sample | 0.931 | $0.1381 | +0.571 | +0.1073 | 1.24 | 0.655 | 0.241 |  | 50 |
| controller (λ=2/USD), 5-fold held-out | 0.724 | $0.1339 | +0.364 | +0.1032 | 1.01 | 0.456 | 0.055 | yes | 50 |
| controller (λ=5/USD), in-sample | 0.640 | $0.0631 | +0.280 | +0.0324 | 0.39 | 0.514 | 0.325 |  | 50 |
| controller (λ=5/USD), 5-fold held-out | 0.495 | $0.0655 | +0.135 | +0.0348 | 0.41 | 0.364 | 0.168 | yes | 50 |
| oracle (max quality, cheapest) | 0.960 | $0.0951 | +0.600 | +0.0644 | n/a | 0.770 | 0.485 |  | 50 |
| oracle (argmax q - 2·cost) | 0.960 | $0.0951 | +0.600 | +0.0644 | n/a | 0.770 | 0.485 |  | 50 |
| oracle (argmax q - 5·cost) | 0.880 | $0.0784 | +0.520 | +0.0477 | n/a | 0.723 | 0.488 |  | 50 |

- controller (B=$0.10, learned λ): λ̄ = 4.68/USD (final 4.78), training quality 0.841 at $0.1001; held-out fold SD of J 0.222
- controller (λ=2/USD): λ̄ = 2.00/USD (final 2.00), training quality 0.935 at $0.1349; held-out fold SD of J 0.077
- controller (λ=5/USD): λ̄ = 5.00/USD (final 5.00), training quality 0.646 at $0.0628; held-out fold SD of J 0.080

## 4. Priced comparison and the budget-constrained best policy

- **λ = 2/USD**: best policy `uncertainty_gated:0.4:cheap` with J = 0.500 (quality 0.640 at $0.0699); `never` has J = 0.299; the per-task oracle has J = 0.770.
- **λ = 5/USD**: best policy `uncertainty_gated:0.4:cheap` with J = 0.291 (quality 0.640 at $0.0699); `never` has J = 0.206; the per-task oracle has J = 0.488.

| budget B (USD/task) | best feasible policy (max quality, cost ≤ B) | quality | cost | Δq vs never |
|---|---|---|---|---|
| $0.08 | uncertainty_gated:0.4:cheap | 0.640 | $0.0699 | +0.280 |
| $0.10 | uncertainty_gated:0.4:cheap | 0.640 | $0.0699 | +0.280 |
| $0.12 | uncertainty_gated:0.4:mid | 0.700 | $0.1038 | +0.340 |
| $0.15 | controller (λ=2/USD), 5-fold held-out | 0.724 | $0.1339 | +0.364 |
| $0.20 | always_1:mid | 0.860 | $0.1981 | +0.500 |
| $0.30 | always_1:mid | 0.860 | $0.1981 | +0.500 |
| $0.50 | uncertainty_gated:0.7:mid | 0.900 | $0.3338 | +0.540 |

## 5. Interpretation

- **This is the dry run.** The fake world was built so that low confidence predicts wrong answers imperfectly, mid workers are better than cheap ones, and refinement harms correct refusals on the confidentiality categories. The bullets below show what the analysis *reports* given such a world; they are not findings.
- **H1 (refine where confidence is low) on cheap:** one round changes quality by +0.375 on average when the probe's min confidence is < 0.6 (n = 40) and by +0.200 when it is ≥ 0.6 (n = 10). Consistent with H1.
- **H1 (refine where confidence is low) on mid:** one round changes quality by +0.550 on average when the probe's min confidence is < 0.6 (n = 40) and by +0.300 when it is ≥ 0.6 (n = 10). Consistent with H1.
- **H2 (always-refine is dominated) on cheap:** `always_1:cheap` reaches 0.700 at $0.1047; dominated by `uncertainty_gated:0.4:mid`.
- **H2 (always-refine is dominated) on mid:** `always_1:mid` reaches 0.860 at $0.1981; not dominated by any evaluated policy.
- **H3 (a priced gate reaches the frontier):** `controller (B=$0.10, learned λ), 5-fold held-out` held-out: 0.656 at $0.1044 — off the frontier; 2 fixed-τ gate(s) weakly dominate it (`uncertainty_gated:0.5:cheap`, `uncertainty_gated:0.4:mid`).
- **H3 (a priced gate reaches the frontier):** `controller (λ=2/USD), 5-fold held-out` held-out: 0.724 at $0.1339 — on the Pareto frontier; no fixed-τ gate weakly dominates it.
- **H3 (a priced gate reaches the frontier):** `controller (λ=5/USD), 5-fold held-out` held-out: 0.495 at $0.0655 — on the Pareto frontier; no fixed-τ gate weakly dominates it.
- **Power.** 50 tasks: one task is 0.020 of quality, so differences of a few hundredths are within one or two tasks. Held-out controller rows are the honest ones; the in-sample rows show what the feature set can memorize (22 category one-hots on 50 tasks).
- **Harm concentrates where the correct answer is a refusal.** On the confidentiality categories, mid round 1 flips 33% (1/3) of correct answers to wrong. A gate that only looks at confidence cannot see this; the controller can, through the category feature.
