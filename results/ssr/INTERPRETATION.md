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
