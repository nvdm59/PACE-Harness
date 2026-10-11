# Continual MAS orchestrator: results

## 0. Interpretation

Simulator calibrated to the real MAS log (130 episodes), 40 days × 96 tasks per arm, four
harness stages (price, agents, tools, skills) on days 9 / 17 / 25 / 33, one pre-registered
schedule plus a milder one, one seed, 189 s on CPU. J_g = quality − λ̄_g·cost with λ̄_g the
per-stage oracle's learned price (0.66–1.25 per USD). Seven findings, in order of importance.

**1. Against the learned alternatives the continual orchestrator wins on every metric; against
the right fixed mode it does not.** Cumulative regret to the per-stage oracle over days 6–40:
continual (cgrpo on replay, gated) 1.26, always_verify 1.99, frozen 2.60, SkillOrchestra-style
router re-learned per stage 3.03, static router 3.46, continual AWR 3.90, and `never_verify`
**0.56** (section 3). The continual arm ends every stage within 0.001–0.014 of the oracle
(section 2, "gap to oracle at stage end") and is the only learned arm that recovers at all
(days to recover 6 / 6 / 1 / 1); the frozen controller, the static router and the re-learned
router never come within 0.02 of the oracle on any stage. But `never_verify`, the all-Sonnet
fixed mode, is at the oracle on stage g1 (0.466 vs 0.460) and within 0.02–0.04 on g2–g4 without
learning anything: after the mid repricing, running every sub-query and the answer on the
half-price mid tier is the budgeted optimum or near it in this simulator. The continual arm
beats it only in the stationary periods of the stages that add capabilities (g2 0.429 vs 0.423,
g3 0.453 vs 0.434, g4 0.448 vs 0.428, section 4) and loses it back during the recovery days.
On a single seed those stationary margins (0.006–0.020) are not established. The honest answer
to "what are the chances we beat the alternatives": the online learner beats every design-time
learner here, and the thing that beats the online learner is knowing the right fixed policy in
advance, which the growing pool makes less likely with every stage (the fixed mode's gap to the
oracle grows from −0.006 at g1 to −0.038 at g4). The mild schedule gives the same ordering with
smaller gaps (section 11: continual 1.10, never_verify 0.39, frozen 1.97, static 2.90).

**2. The total-regret headline is a cold-start artifact: the learner crosses `never_verify` around
day 22–25 and wins every stage window after that, while `never_verify` falls further below the
oracle at each capability stage.** Served J, continual minus `never_verify`, by stage window
(section 2b): g0 −0.073, g1 −0.070, g2 −0.021, g3 +0.024, g4 +0.035; on the held-out sets
−0.085, −0.087, −0.004, +0.016, +0.019. The served gap is positive and stays positive from day 24.
Cumulative regret vs the oracle restricted to days 17–40 is 0.42 (continual) against 0.68
(`never_verify`) on the held-out sets and 0.07 against 0.37 on the served stream; from day 25 it is
0.22 against 0.50 held-out and −0.11 against 0.36 served (section 2b; the served stream is a
different task draw from the held-out set, so a served J can exceed the oracle's held-out J). The
learner's served J includes its ε = 0.1 exploration mass, which `never_verify` does not pay, and
it still wins after the crossover. The whole deficit in the whole-stream number is the 12 days on
the uniform policy before the first promotion (finding 3) plus the repricing recovery (days 9–13,
where the half-price mid tier makes `never_verify` the optimum before the learner has moved):
`never_verify` is above the reference oracle at g0 and g1 (+0.023, +0.006 on the held-out sets)
and falls to −0.022, −0.024, −0.038 at g2–g4 as capabilities it cannot use arrive. The caveats
stand: one seed, and the stage ≥ 2 capabilities are assumed parameters (section 10), so the size
of the post-crossover margin is the simulator's.

**3. H2 holds and its price is larger than its benefit on this stream: the gate reverted 32 of
40 candidates, 9 of them rightly and 23 as false alarms, and the first promotion came on day
13.** Every one of the 9 correct REVERTs was a budget violation (candidate true cost $0.095–0.101
against B + tol = $0.095, section 6b); no promoted candidate was bad. The DR cost estimate is the
criterion that works: RMSE $0.006 against the simulator's truth (section 6a), and the "what the
gate prevented" table is exactly the set of over-budget candidates. The quality criterion is
what produces the false alarms: 18 of the 32 REVERTs failed on quality alone, with the candidate
in fact acceptable. At n ≈ 500–1,000 logged episodes the DR quality 90% CI is 0.20 wide and
covers the truth 63% of the time, so a candidate whose lower bound must reach the incumbent's
point estimate − 0.05 passes only when it is visibly better. During days 1–12 the served policy
was therefore the uniform ε-policy while the learner was already good (finding 2): the frozen arm, which is
the learner's day-5 snapshot, scores 0.411 on g1 against the served incumbent's 0.247 on day 9
(section 2, 5). That is the cost of promotion behind a gate at this sample size, and it is the
same sample-size finding as `results/crmarena/RESULTS.md` (a $0.005 margin needs ~1,000 tasks):
the cost side is now certifiable at ~500 episodes a day, the quality side is not.

**4. The AWR + IPS variant is the cautionary row: it was promoted 14 times and is the worst
learned arm (stationary J 0.32–0.36), because the gate's sequential DR over-estimated its
quality by +0.68 on average (coverage 6%).** AWR's exponentiated, inverse-propensity weights
move the policy far from the behavior in one full-batch step; the core gate's cumulative ratios
(clipped at 50) then carry the estimate with a per-decision ESS of 25% and a bias the clip does
not remove. This is the failure mode that made the MAS package introduce per-decision
estimators; their bias on the same candidates is +0.18, better but still not a safe bar. The
cgrpo candidates, trained on rollouts sampled from the ε-mixed served policy, stay close to the
behavior (ESS 54%) and the same estimator is nearly unbiased on them (−0.005). The lesson for the
real run is that the gate is only as good as the overlap between candidate and behavior, and a
trainer that preserves that overlap is part of the safety story.

**5. H3 holds: orchestrator-level BWT is bounded and mostly positive.** The continual arm's BWT
per stage is +0.111 (g0), +0.023 (g1), +0.009 (g2), −0.019 (g3); the mild schedule gives +0.120,
+0.026, +0.002, −0.009. Later training did not undo earlier behaviour because the harness only
grows: a capability that arrives at stage g has a zero feature slot and a zero logit before it,
and the policy's stage-g weights are still what runs where that capability is not legal. The
large g0 number is not transfer in the usual sense, it is the day-8 incumbent being the uniform
policy (finding 3). Contrast with EvoHarnessBench's −34.7% BWT for agent-pool expansion under a
fixed policy: here the pool expansion costs the fixed policies at the event (frozen hit −0.041,
static −0.033 at g2) and they never recover (stationary J falls from 0.411 / 0.385 at g1 to
0.350 / 0.316 at g4), while the policy that moves loses nothing on the old stages.

**6. FWT is a property of the policy in service, not of anticipation.** New capabilities arrive
with zero logits, so the event-day policy is the previous stage's; the continual arm's FWT on
g2–g4 (+0.085 / +0.130 / +0.128) exceeds the frozen arm's (+0.046 / +0.057 / +0.032) because it
is a better policy, and the static router's FWT falls to −0.002 at g4 because its handbook prices
unseen capabilities at a prior. Hits at the events are small for every arm except the price
event, where the halved mid price lifts every mid-heavy policy (+0.05 to +0.10).

**7. The SkillOrchestra-style router is the weakest design-time baseline here, and the reason is
diagnostic.** Its handbook is the literal rule, competence[(action, category)] minus λ_c × mean
logged cost with cost tie-break, learned from the warm-up exploration traces. Two things go wrong
on this log. The competence of an action is the mean outcome of episodes in which it was taken,
which under per-step exploration is confounded by everything else in the episode (the handbook
over-selects `verify@verify-1`, 28% of decisions, an action the real run showed buys nothing); and
λ_c is tuned by episode-level SN-IPS of a deterministic router against ε = 0.1 traffic, whose ESS
is 1–8 of 96 (section 9), so the choice of λ_c is noise and the re-learned router's λ_c swings
between 0 and 20 across stages. Re-learning per stage recovers some of the agent-pool loss (+0.043
at g4) at the price of three exploration days per stage, and still ends 0.07–0.11 below the
oracle. This is a statement about applying the competence − λ_c·cost rule to multi-step
orchestration traces with a per-step router, not about SkillOrchestra's own handbook learning
from richer per-skill traces on NQ / FRAMES.

**What this does not show.** Effect sizes are the simulator's: the stage-2+ capabilities are
assumed, as is the link from a worker step's latent quality to the answer (section 10 marks every
assumed parameter). One seed, one schedule plus a mild one. The per-stage oracle is constrained
GRPO on fresh exploration, a reference and not a proven optimum (a learner exceeds it at g1 in the
mild run). The real-run protocol and its cost are in the package README (Phase B).

Sections 1–11 below are generated by `analyze.py`; section 0 is authored and passed in with `--prelude`.

Run: schedule `default`, 40 days × 96 tasks per arm, budget B = $0.090 per task (gate tol $0.005, ε_q 0.05), served ε = 0.1, held-out 150 tasks per stage, seed 0; 18 actions, 88 features; runtime 189.1 s on CPU. J_g = quality − λ̄_g · cost (USD) with λ̄_g the per-stage oracle's learned price (g0 1.25, g1 0.85, g2 0.81, g3 0.66, g4 0.73 per USD): at the price where the budget binds, the budgeted optimum maximizes the Lagrangian over every policy, so J_g has a ceiling and regret is well defined.

## 1. Harness stages

| stage | axis | day | date | what changed |
|---|---|---|---|---|
| g0-baseline | none | 1 | 2026-09-01 | the real run's harness (plan-1, data-1..3, verify-1, answer-1..3), list prices |
| g1-reprice | price | 9 | 2026-09-09 | repricing: mid x0.5, frontier x1.5 |
| g2-agents | agents | 17 | 2026-09-17 | agent pool + verify-2 (per-call P(fail | good rows) 0.114), data-4 (rows 0.85, good 0.8), answer-4 (8 strong categories) |
| g3-tools | tools | 25 | 2026-09-25 | tools + schema_lookup, aggregate_calc (bonus scale 1.0) |
| g4-skills | skills | 33 | 2026-10-03 | skills + soql_aggregate, date_window, decompose_fine; planner v2 plan length p = (0.0, 0.1, 0.4, 0.5) |

## 2. Per-stage transfer: hit, FWT, recovery, days to recover, stationary J, BWT

Held-out J of the policy in service each day (simulator expected quality on the stage's fixed task set). `hit` = J on the new stage minus J on the old stage the day before the event (what the event cost the policy in service); `FWT` = that same event-day J minus a uniform policy's; `recovery` = J at the stage's end minus J on the event day; `BWT` = J on this stage at day 40 minus J on it at the stage's end (the frozen arm's BWT is 0 by construction, its policy never changes).

**Stage g1 (day 9; oracle held-out J 0.460, uniform 0.351)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.048 | -0.020 | +0.128 | 6 | 0.427 | -0.001 | +0.023 | $0.0811 |
| continual (AWR+IPS, gated, λ = 2) | +0.071 | +0.012 | -0.005 | – | 0.363 | -0.102 | -0.002 | $0.0790 |
| frozen (trained days 1–5) | +0.067 | +0.060 | +0.000 | – | 0.411 | -0.050 | +0.000 | $0.0540 |
| SkillOrchestra-style static router | +0.053 | +0.034 | +0.000 | – | 0.385 | -0.075 | +0.000 | $0.0957 |
| SkillOrchestra-style router, re-learned per stage | +0.053 | +0.034 | +0.004 | – | 0.389 | -0.071 | +0.013 | $0.0879 |
| always_verify | +0.097 | +0.075 | +0.000 | – | 0.426 | -0.034 | +0.000 | $0.0494 |
| never_verify | +0.098 | +0.115 | +0.000 | 1 | 0.466 | +0.006 | +0.000 | $0.0446 |

**Stage g2 (day 17; oracle held-out J 0.445, uniform 0.324)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | -0.050 | +0.085 | +0.026 | 6 | 0.429 | -0.010 | +0.009 | $0.0771 |
| continual (AWR+IPS, gated, λ = 2) | -0.019 | +0.016 | -0.002 | – | 0.342 | -0.107 | -0.022 | $0.0662 |
| frozen (trained days 1–5) | -0.041 | +0.046 | +0.000 | – | 0.370 | -0.075 | +0.000 | $0.0516 |
| SkillOrchestra-style static router | -0.033 | +0.028 | +0.000 | – | 0.352 | -0.093 | +0.000 | $0.0720 |
| SkillOrchestra-style router, re-learned per stage | -0.041 | +0.024 | +0.020 | – | 0.369 | -0.076 | +0.010 | $0.0577 |
| always_verify | -0.037 | +0.065 | +0.000 | – | 0.389 | -0.056 | +0.000 | $0.0492 |
| never_verify | -0.043 | +0.099 | +0.000 | – | 0.423 | -0.022 | +0.000 | $0.0446 |

**Stage g3 (day 25; oracle held-out J 0.458, uniform 0.317)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.012 | +0.130 | +0.007 | 1 | 0.453 | -0.004 | -0.019 | $0.0881 |
| continual (AWR+IPS, gated, λ = 2) | -0.007 | +0.014 | -0.002 | – | 0.330 | -0.130 | +0.011 | $0.0654 |
| frozen (trained days 1–5) | +0.004 | +0.057 | +0.000 | – | 0.374 | -0.084 | +0.000 | $0.0520 |
| SkillOrchestra-style static router | -0.002 | +0.032 | +0.000 | – | 0.349 | -0.109 | +0.000 | $0.0709 |
| SkillOrchestra-style router, re-learned per stage | +0.015 | +0.066 | -0.037 | – | 0.346 | -0.112 | +0.035 | $0.0596 |
| always_verify | +0.008 | +0.080 | +0.000 | – | 0.397 | -0.062 | +0.000 | $0.0492 |
| never_verify | +0.011 | +0.117 | +0.000 | – | 0.434 | -0.024 | +0.000 | $0.0449 |

**Stage g4 (day 33; oracle held-out J 0.466, uniform 0.318)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | -0.008 | +0.128 | +0.006 | 1 | 0.448 | -0.014 | n/a | $0.0945 |
| continual (AWR+IPS, gated, λ = 2) | -0.004 | +0.006 | -0.014 | – | 0.321 | -0.155 | n/a | $0.0692 |
| frozen (trained days 1–5) | -0.024 | +0.032 | +0.000 | – | 0.350 | -0.116 | n/a | $0.0512 |
| SkillOrchestra-style static router | -0.033 | -0.002 | +0.000 | – | 0.316 | -0.150 | n/a | $0.0724 |
| SkillOrchestra-style router, re-learned per stage | -0.026 | +0.002 | +0.043 | – | 0.363 | -0.103 | n/a | $0.0628 |
| always_verify | -0.021 | +0.057 | +0.000 | – | 0.376 | -0.090 | n/a | $0.0498 |
| never_verify | -0.006 | +0.109 | +0.000 | – | 0.428 | -0.038 | n/a | $0.0527 |

## 2b. Crossover: the continual arm against never_verify over the stream

**By stage window** (mean over the stage's days):

| stage | days | continual − never_verify, served J | continual − never_verify, held-out J | never_verify − oracle, held-out |
|---|---|---|---|---|
| g0 | 1–8 | -0.073 | -0.085 | +0.023 |
| g1 | 9–16 | -0.070 | -0.087 | +0.006 |
| g2 | 17–24 | -0.021 | -0.004 | -0.022 |
| g3 | 25–32 | +0.024 | +0.016 | -0.024 |
| g4 | 33–40 | +0.035 | +0.019 | -0.038 |

First day from which the served gap stays positive for three consecutive days: **day 24**.
First promotion of the continual arm: day 13; promotions on days [13, 21, 23, 27, 28, 30, 31, 39].

**Cumulative regret vs the per-stage oracle, restricted to later windows** (held-out, as in section 3; and on the served stream, where the learner's J includes its ε = 0.1 exploration mass and `never_verify` pays none; served tasks are a different draw from the held-out set, so the served column is indicative):

| arm | held-out, days 6–40 | held-out, days 17–40 | held-out, days 25–40 | served, days 6–40 | served, days 17–40 | served, days 25–40 |
|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | 1.262 | 0.423 | 0.217 | 1.149 | 0.068 | -0.109 |
| continual (AWR+IPS, gated, λ = 2) | 3.896 | 2.977 | 2.136 | 3.514 | 2.387 | 1.678 |
| frozen (trained days 1–5) | 2.597 | 2.198 | 1.600 | 2.276 | 1.598 | 1.214 |
| SkillOrchestra-style static router | 3.459 | 2.820 | 2.074 | 3.233 | 2.244 | 1.620 |
| SkillOrchestra-style router, re-learned per stage | 3.029 | 2.410 | 1.737 | 2.792 | 1.807 | 1.351 |
| always_verify | 1.988 | 1.667 | 1.217 | 1.850 | 1.178 | 0.944 |
| never_verify | 0.559 | 0.675 | 0.501 | 0.663 | 0.369 | 0.363 |

**Day by day** (`*` event day, `P` the continual arm promoted a candidate that day; the bar is the served gap, one block per 0.01):

| day | stage | continual served J | never_verify served J | gap, served | gap, held-out | |
|---|---|---|---|---|---|---|
| 1 | g0-baseline | 0.302 | 0.364 | -0.061 | -0.085 | `------` |
| 2 | g0-baseline | 0.277 | 0.342 | -0.065 | -0.085 | `------` |
| 3 | g0-baseline | 0.256 | 0.357 | -0.101 | -0.085 | `----------` |
| 4 | g0-baseline | 0.303 | 0.399 | -0.096 | -0.085 | `----------` |
| 5 | g0-baseline | 0.305 | 0.333 | -0.029 | -0.085 | `---` |
| 6 | g0-baseline | 0.233 | 0.301 | -0.069 | -0.085 | `-------` |
| 7 | g0-baseline | 0.299 | 0.380 | -0.081 | -0.085 | `--------` |
| 8 | g0-baseline | 0.241 | 0.321 | -0.080 | -0.085 | `--------` |
| 9 * | g1-reprice | 0.371 | 0.481 | -0.110 | -0.135 | `-----------` |
| 10 | g1-reprice | 0.315 | 0.422 | -0.107 | -0.135 | `-----------` |
| 11 | g1-reprice | 0.311 | 0.406 | -0.095 | -0.135 | `----------` |
| 12 | g1-reprice | 0.321 | 0.417 | -0.096 | -0.135 | `----------` |
| 13 P | g1-reprice | 0.316 | 0.439 | -0.123 | -0.135 | `------------` |
| 14 | g1-reprice | 0.407 | 0.431 | -0.024 | -0.007 | `--` |
| 15 | g1-reprice | 0.417 | 0.424 | -0.007 | -0.007 | `-` |
| 16 | g1-reprice | 0.406 | 0.401 | +0.006 | -0.007 | `+` |
| 17 * | g2-agents | 0.419 | 0.461 | -0.042 | -0.014 | `----` |
| 18 | g2-agents | 0.359 | 0.400 | -0.041 | -0.014 | `----` |
| 19 | g2-agents | 0.392 | 0.437 | -0.045 | -0.014 | `----` |
| 20 | g2-agents | 0.401 | 0.448 | -0.047 | -0.014 | `-----` |
| 21 P | g2-agents | 0.408 | 0.439 | -0.031 | -0.014 | `---` |
| 22 | g2-agents | 0.461 | 0.421 | +0.040 | +0.014 | `++++` |
| 23 P | g2-agents | 0.461 | 0.482 | -0.020 | +0.014 | `--` |
| 24 | g2-agents | 0.482 | 0.467 | +0.014 | +0.011 | `+` |
| 25 * | g3-tools | 0.461 | 0.441 | +0.020 | +0.013 | `++` |
| 26 | g3-tools | 0.466 | 0.461 | +0.005 | +0.013 | `+` |
| 27 P | g3-tools | 0.499 | 0.484 | +0.015 | +0.013 | `+` |
| 28 P | g3-tools | 0.459 | 0.412 | +0.046 | +0.018 | `+++++` |
| 29 | g3-tools | 0.484 | 0.480 | +0.004 | +0.026 | `` |
| 30 P | g3-tools | 0.475 | 0.431 | +0.043 | +0.026 | `++++` |
| 31 P | g3-tools | 0.488 | 0.462 | +0.026 | +0.003 | `+++` |
| 32 | g3-tools | 0.503 | 0.469 | +0.034 | +0.020 | `+++` |
| 33 * | g4-skills | 0.449 | 0.432 | +0.018 | +0.018 | `++` |
| 34 | g4-skills | 0.407 | 0.390 | +0.017 | +0.018 | `++` |
| 35 | g4-skills | 0.509 | 0.456 | +0.053 | +0.018 | `+++++` |
| 36 | g4-skills | 0.471 | 0.420 | +0.051 | +0.018 | `+++++` |
| 37 | g4-skills | 0.465 | 0.411 | +0.055 | +0.018 | `+++++` |
| 38 | g4-skills | 0.441 | 0.405 | +0.036 | +0.018 | `++++` |
| 39 P | g4-skills | 0.437 | 0.433 | +0.005 | +0.018 | `` |
| 40 | g4-skills | 0.491 | 0.448 | +0.043 | +0.024 | `++++` |

## 3. Whole-stream summary

| arm | mean BWT (stages < T) | mean FWT | cumulative regret vs oracle (days 6–40) | regret per day |
|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.031 | +0.081 | 1.262 | 0.0361 |
| continual (AWR+IPS, gated, λ = 2) | -0.006 | +0.012 | 3.896 | 0.1113 |
| frozen (trained days 1–5) | +0.000 | +0.049 | 2.597 | 0.0742 |
| SkillOrchestra-style static router | +0.000 | +0.023 | 3.459 | 0.0988 |
| SkillOrchestra-style router, re-learned per stage | +0.019 | +0.032 | 3.029 | 0.0865 |
| always_verify | +0.000 | +0.069 | 1.988 | 0.0568 |
| never_verify | +0.000 | +0.110 | 0.559 | 0.0160 |

## 4. Stationary periods: mean held-out J over the last 4 days of each stage

| arm | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| oracle | 0.345 | 0.460 | 0.445 | 0.458 | 0.466 |
| continual (cgrpo on replay, gated, learned λ) | 0.283 | 0.427 | 0.429 | 0.453 | 0.448 |
| continual (AWR+IPS, gated, λ = 2) | 0.294 | 0.363 | 0.342 | 0.330 | 0.321 |
| frozen (trained days 1–5) | – | 0.411 | 0.370 | 0.374 | 0.350 |
| SkillOrchestra-style static router | – | 0.385 | 0.352 | 0.349 | 0.316 |
| SkillOrchestra-style router, re-learned per stage | – | 0.389 | 0.369 | 0.346 | 0.363 |
| always_verify | 0.329 | 0.426 | 0.389 | 0.397 | 0.376 |
| never_verify | 0.368 | 0.466 | 0.423 | 0.434 | 0.428 |

## 5. The continual (cgrpo) arm, day by day

`*` marks an event day. λ̄ is the served policy's learned price of cost (per USD). `truth` is whether the candidate was acceptable on the simulator (within budget + tol and not more than ε_q below the incumbent's true quality).

| day | stage | served quality (exp.) | cost/request | J | λ̄ (served) | gate | DR quality [90% CI] | DR cost upper | ESS (per-decision) | truth |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | g0-baseline | 0.405 | $0.0821 | 0.302 | 0.00 | REVERT | 0.311 [0.198, 0.489] | 0.0410 | 61/199 | ok |
| 2 | g0-baseline | 0.375 | $0.0776 | 0.277 | 0.00 | REVERT | 0.583 [0.353, 0.832] | 0.0790 | 130/394 | BAD |
| 3 | g0-baseline | 0.361 | $0.0834 | 0.256 | 0.00 | REVERT | 0.484 [0.358, 0.621] | 0.0763 | 199/606 | BAD |
| 4 | g0-baseline | 0.403 | $0.0797 | 0.303 | 0.00 | REVERT | 0.401 [0.321, 0.495] | 0.0483 | 298/805 | ok |
| 5 | g0-baseline | 0.408 | $0.0824 | 0.305 | 0.00 | REVERT | 0.389 [0.309, 0.474] | 0.0494 | 356/991 | ok |
| 6 | g0-baseline | 0.334 | $0.0813 | 0.233 | 0.00 | REVERT | 0.308 [0.213, 0.415] | 0.0572 | 302/999 | ok |
| 7 | g0-baseline | 0.402 | $0.0822 | 0.299 | 0.00 | REVERT | 0.292 [0.186, 0.411] | 0.0547 | 292/1013 | ok |
| 8 | g0-baseline | 0.343 | $0.0814 | 0.241 | 0.00 | REVERT | 0.312 [0.228, 0.415] | 0.0703 | 259/1002 | ok |
| 9 * | g1-reprice | 0.439 | $0.0802 | 0.371 | 0.00 | REVERT | 0.406 [0.094, 0.784] | 0.1109 | 42/192 | ok |
| 10 | g1-reprice | 0.375 | $0.0718 | 0.315 | 0.00 | REVERT | 0.594 [0.377, 0.802] | 0.0863 | 81/386 | ok |
| 11 | g1-reprice | 0.379 | $0.0805 | 0.311 | 0.00 | REVERT | 0.483 [0.259, 0.690] | 0.0741 | 116/587 | ok |
| 12 | g1-reprice | 0.385 | $0.0755 | 0.321 | 0.00 | REVERT | 0.488 [0.342, 0.637] | 0.0763 | 158/788 | ok |
| 13 | g1-reprice | 0.381 | $0.0772 | 0.316 | 0.00 | PROMOTE | 0.552 [0.415, 0.694] | 0.0790 | 190/995 | ok |
| 14 | g1-reprice | 0.482 | $0.0898 | 0.407 | 0.81 | REVERT | 0.553 [0.433, 0.687] | 0.0759 | 254/1070 | ok |
| 15 | g1-reprice | 0.491 | $0.0874 | 0.417 | 0.81 | REVERT | 0.536 [0.436, 0.629] | 0.0761 | 347/1143 | ok |
| 16 | g1-reprice | 0.479 | $0.0864 | 0.406 | 0.81 | REVERT | 0.550 [0.459, 0.645] | 0.0757 | 473/1200 | ok |
| 17 * | g2-agents | 0.472 | $0.0645 | 0.419 | 0.81 | REVERT | 0.528 [0.372, 0.658] | 0.0744 | 143/283 | ok |
| 18 | g2-agents | 0.420 | $0.0756 | 0.359 | 0.81 | REVERT | 0.484 [0.378, 0.597] | 0.0812 | 199/561 | BAD |
| 19 | g2-agents | 0.449 | $0.0700 | 0.392 | 0.81 | REVERT | 0.422 [0.344, 0.525] | 0.0763 | 236/829 | BAD |
| 20 | g2-agents | 0.458 | $0.0699 | 0.401 | 0.81 | REVERT | 0.443 [0.364, 0.518] | 0.0788 | 307/1101 | BAD |
| 21 | g2-agents | 0.470 | $0.0768 | 0.408 | 0.81 | PROMOTE | 0.442 [0.370, 0.518] | 0.0747 | 408/1395 | ok |
| 22 | g2-agents | 0.531 | $0.0870 | 0.461 | 0.75 | REVERT | 0.445 [0.380, 0.504] | 0.0805 | 509/1461 | BAD |
| 23 | g2-agents | 0.531 | $0.0857 | 0.461 | 0.75 | PROMOTE | 0.501 [0.443, 0.556] | 0.0751 | 570/1507 | ok |
| 24 | g2-agents | 0.552 | $0.0869 | 0.482 | 1.03 | REVERT | 0.537 [0.482, 0.579] | 0.0735 | 900/1593 | ok |
| 25 * | g3-tools | 0.519 | $0.0883 | 0.461 | 1.03 | REVERT | 0.561 [0.483, 0.641] | 0.0830 | 339/345 | ok |
| 26 | g3-tools | 0.524 | $0.0875 | 0.466 | 1.03 | REVERT | 0.558 [0.503, 0.638] | 0.0775 | 665/680 | ok |
| 27 | g3-tools | 0.559 | $0.0914 | 0.499 | 1.03 | PROMOTE | 0.568 [0.515, 0.615] | 0.0753 | 997/1022 | ok |
| 28 | g3-tools | 0.511 | $0.0788 | 0.459 | 0.77 | PROMOTE | 0.541 [0.499, 0.590] | 0.0778 | 1275/1371 | ok |
| 29 | g3-tools | 0.548 | $0.0971 | 0.484 | 0.66 | REVERT | 0.560 [0.519, 0.612] | 0.0814 | 1410/1736 | ok |
| 30 | g3-tools | 0.535 | $0.0910 | 0.475 | 0.66 | PROMOTE | 0.572 [0.516, 0.626] | 0.0802 | 1363/1747 | ok |
| 31 | g3-tools | 0.543 | $0.0832 | 0.488 | 0.53 | PROMOTE | 0.576 [0.519, 0.659] | 0.0801 | 1368/1774 | ok |
| 32 | g3-tools | 0.561 | $0.0872 | 0.503 | 0.41 | REVERT | 0.581 [0.527, 0.651] | 0.0765 | 1459/1809 | ok |
| 33 * | g4-skills | 0.518 | $0.0941 | 0.449 | 0.41 | REVERT | 0.472 [0.394, 0.583] | 0.0830 | 412/432 | ok |
| 34 | g4-skills | 0.470 | $0.0874 | 0.407 | 0.41 | REVERT | 0.452 [0.390, 0.518] | 0.0811 | 810/848 | BAD |
| 35 | g4-skills | 0.585 | $0.1041 | 0.509 | 0.41 | REVERT | 0.478 [0.418, 0.537] | 0.0818 | 1165/1277 | BAD |
| 36 | g4-skills | 0.542 | $0.0986 | 0.471 | 0.41 | REVERT | 0.512 [0.465, 0.574] | 0.0829 | 1540/1720 | ok |
| 37 | g4-skills | 0.539 | $0.1012 | 0.465 | 0.41 | REVERT | 0.522 [0.480, 0.573] | 0.0825 | 1959/2164 | ok |
| 38 | g4-skills | 0.506 | $0.0900 | 0.441 | 0.41 | REVERT | 0.570 [0.512, 0.630] | 0.0842 | 1824/2145 | ok |
| 39 | g4-skills | 0.505 | $0.0926 | 0.437 | 0.41 | PROMOTE | 0.600 [0.536, 0.666] | 0.0771 | 1765/2142 | ok |
| 40 | g4-skills | 0.555 | $0.0881 | 0.491 | 0.88 | REVERT | 0.603 [0.547, 0.661] | 0.0816 | 1789/2103 | BAD |

## 6. Gate history

| arm | candidates | PROMOTE | REVERT | REVERT right (candidate bad) | REVERT false alarm (candidate fine) | PROMOTE right | PROMOTE missed (candidate bad) | correct | mean per-decision ESS / n | mean DR quality CI width | mean DR cost CI width |
|---|---|---|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | 40 | 8 | 32 | 9 | 23 | 8 | 0 | 42% | 0.54 | 0.200 | $0.0079 |
| continual (AWR+IPS, gated, λ = 2) | 40 | 14 | 26 | 0 | 26 | 14 | 0 | 35% | 0.25 | 0.653 | $0.0144 |

### 6a. Gate estimators against simulator truth (the candidate's true held-out value on the current stage)

| arm | days | n | DR quality: bias / RMSE / CI covers truth | DR cost: bias / RMSE / CI covers truth | per-decision SNIPS quality: bias / RMSE | mean n per gate |
|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | warm-up | 5 | -0.009 / 0.063 / 100% | -0.0008 / 0.0008 / 0% | -0.034 / 0.039 | 288 |
| continual (cgrpo on replay, gated, learned λ) | after warm-up | 35 | -0.005 / 0.067 / 63% | +0.0003 / 0.0061 / 54% | -0.020 / 0.053 | 370 |
| continual (AWR+IPS, gated, λ = 2) | warm-up | 5 | +1.569 / 1.642 / 0% | -0.0044 / 0.0064 / 60% | +0.268 / 0.271 | 288 |
| continual (AWR+IPS, gated, λ = 2) | after warm-up | 35 | +0.676 / 0.809 / 6% | +0.0000 / 0.0056 / 69% | +0.178 / 0.187 | 370 |

### 6b. What the gate prevented (candidates that were over budget or regressed quality on the simulator)

**continual (cgrpo on replay, gated, learned λ)**

| day | stage | candidate true quality / cost | incumbent true quality / cost | over budget | regressed | gate said |
|---|---|---|---|---|---|---|
| 2 | g0-baseline | 0.482 / $0.1009 | 0.382 / $0.0794 | yes | – | REVERT |
| 3 | g0-baseline | 0.470 / $0.0981 | 0.382 / $0.0794 | yes | – | REVERT |
| 18 | g2-agents | 0.515 / $0.0974 | 0.467 / $0.0721 | yes | – | REVERT |
| 19 | g2-agents | 0.508 / $0.0953 | 0.467 / $0.0721 | yes | – | REVERT |
| 20 | g2-agents | 0.528 / $0.0996 | 0.467 / $0.0721 | yes | – | REVERT |
| 22 | g2-agents | 0.527 / $0.0962 | 0.512 / $0.0926 | yes | – | REVERT |
| 34 | g4-skills | 0.521 / $0.0966 | 0.513 / $0.0914 | yes | – | REVERT |
| 35 | g4-skills | 0.522 / $0.0951 | 0.513 / $0.0914 | yes | – | REVERT |
| 40 | g4-skills | 0.523 / $0.0954 | 0.517 / $0.0904 | yes | – | REVERT |

**continual (AWR+IPS, gated, λ = 2)**

(no candidate would have violated the budget or regressed quality)

## 7. Cost per request over time (served, every other day and event days)

| day | stage | continual (cgrpo) | continual (AWR) | frozen | router static | router re-learned | always_verify | never_verify |
|---|---|---|---|---|---|---|---|---|
| 2 | g0-baseline | $0.0776 | $0.0770 | – | – | – | $0.0884 | $0.0822 |
| 4 | g0-baseline | $0.0797 | $0.0674 | – | – | – | $0.0901 | $0.0816 |
| 6 | g0-baseline | $0.0813 | $0.0821 | $0.0710 | $0.1102 | $0.1083 | $0.0898 | $0.0830 |
| 8 | g0-baseline | $0.0814 | $0.0786 | $0.0758 | $0.0983 | $0.0980 | $0.0898 | $0.0810 |
| 9 * | g1-reprice | $0.0802 | $0.0871 | $0.0582 | $0.0954 | $0.0912 | $0.0501 | $0.0436 |
| 10 | g1-reprice | $0.0718 | $0.0752 | $0.0519 | $0.0951 | $0.0909 | $0.0495 | $0.0455 |
| 12 | g1-reprice | $0.0755 | $0.0851 | $0.0526 | $0.1091 | $0.1004 | $0.0489 | $0.0447 |
| 14 | g1-reprice | $0.0898 | $0.0828 | $0.0544 | $0.1045 | $0.0865 | $0.0490 | $0.0453 |
| 16 | g1-reprice | $0.0864 | $0.0752 | $0.0555 | $0.0861 | $0.0832 | $0.0494 | $0.0444 |
| 17 * | g2-agents | $0.0645 | $0.0640 | $0.0566 | $0.0698 | $0.0806 | $0.0493 | $0.0446 |
| 18 | g2-agents | $0.0756 | $0.0729 | $0.0505 | $0.0720 | $0.0841 | $0.0495 | $0.0439 |
| 20 | g2-agents | $0.0699 | $0.0695 | $0.0547 | $0.0642 | $0.0441 | $0.0494 | $0.0452 |
| 22 | g2-agents | $0.0870 | $0.0658 | $0.0473 | $0.0736 | $0.0460 | $0.0490 | $0.0450 |
| 24 | g2-agents | $0.0869 | $0.0705 | $0.0567 | $0.0730 | $0.0448 | $0.0495 | $0.0445 |
| 25 * | g3-tools | $0.0883 | $0.0580 | $0.0487 | $0.0724 | $0.0530 | $0.0488 | $0.0449 |
| 26 | g3-tools | $0.0875 | $0.0654 | $0.0559 | $0.0775 | $0.0642 | $0.0493 | $0.0458 |
| 28 | g3-tools | $0.0788 | $0.0650 | $0.0469 | $0.0719 | $0.0616 | $0.0494 | $0.0452 |
| 30 | g3-tools | $0.0910 | $0.0685 | $0.0527 | $0.0634 | $0.0646 | $0.0494 | $0.0449 |
| 32 | g3-tools | $0.0872 | $0.0632 | $0.0571 | $0.0630 | $0.0598 | $0.0498 | $0.0452 |
| 33 * | g4-skills | $0.0941 | $0.0643 | $0.0516 | $0.0713 | $0.0545 | $0.0492 | $0.0521 |
| 34 | g4-skills | $0.0874 | $0.0724 | $0.0505 | $0.0770 | $0.0621 | $0.0498 | $0.0527 |
| 36 | g4-skills | $0.0986 | $0.0718 | $0.0571 | $0.0812 | $0.0577 | $0.0500 | $0.0525 |
| 38 | g4-skills | $0.0900 | $0.0710 | $0.0529 | $0.0723 | $0.0652 | $0.0496 | $0.0527 |
| 40 | g4-skills | $0.0881 | $0.0687 | $0.0454 | $0.0700 | $0.0714 | $0.0500 | $0.0518 |

## 8. What each policy serves at the end of each stage

| stage | arm | top actions served (share of decisions, last day of the stage) |
|---|---|---|
| g0-baseline | continual (cgrpo) | answer@answer-1 18%, answer@answer-3 14%, answer@answer-2 12%, delegate@data-3 11%, verify@verify-1 11%, replan@plan-1 9% |
| g0-baseline | frozen | delegate@data-2 38%, answer@answer-2 37%, answer@answer-3 20%, replan@plan-1 1%, answer@answer-1 1%, delegate@data-1 1% |
| g0-baseline | router static | verify@verify-1 28%, answer@answer-2 17%, delegate@data-3 11%, delegate@data-2 11%, answer@answer-3 11%, replan@plan-1 9% |
| g0-baseline | router re-learned | verify@verify-1 28%, answer@answer-2 17%, answer@answer-3 11%, delegate@data-3 11%, delegate@data-2 10%, replan@plan-1 10% |
| g1-reprice | continual (cgrpo) | delegate@data-2 53%, answer@answer-2 18%, answer@answer-3 17%, replan@plan-1 4%, replan@plan-2 2%, answer@answer-1 2% |
| g1-reprice | frozen | answer@answer-2 50%, delegate@data-2 29%, answer@answer-3 18%, delegate@data-3 1%, verify@verify-1 1%, replan@plan-1 1% |
| g1-reprice | router static | verify@verify-1 27%, answer@answer-2 15%, delegate@data-2 14%, answer@answer-3 11%, replan@plan-2 9%, delegate@data-3 7% |
| g1-reprice | router re-learned | answer@answer-3 19%, answer@answer-2 18%, delegate@data-1 17%, verify@verify-1 16%, replan@plan-3 15%, answer@answer-1 7% |
| g2-agents | continual (cgrpo) | delegate@data-2 26%, delegate@data-4 24%, answer@answer-2 15%, replan@plan-1 11%, answer@answer-3 10%, verify@verify-2 7% |
| g2-agents | frozen | answer@answer-2 44%, delegate@data-2 36%, answer@answer-3 15%, answer@answer-1 2%, delegate@data-3 1%, delegate@data-1 1% |
| g2-agents | router static | verify@verify-2 19%, verify@verify-1 14%, answer@answer-4 13%, replan@plan-2 9%, delegate@data-4 8%, delegate@data-2 7% |
| g2-agents | router re-learned | verify@verify-1 42%, delegate@data-2 28%, answer@answer-2 19%, replan@plan-2 4%, answer@answer-1 2%, delegate@data-4 2% |
| g3-tools | continual (cgrpo) | delegate@data-2 22%, delegate@data-4 21%, call@aggregate_calc 13%, answer@answer-2 12%, answer@answer-3 10%, verify@verify-2 10% |
| g3-tools | frozen | answer@answer-2 46%, delegate@data-2 33%, answer@answer-3 17%, delegate@data-3 1%, replan@plan-2 1%, answer@answer-1 1% |
| g3-tools | router static | verify@verify-2 20%, verify@verify-1 13%, answer@answer-4 12%, replan@plan-1 8%, delegate@data-2 7%, delegate@data-4 7% |
| g3-tools | router re-learned | call@schema_lookup 24%, answer@answer-4 14%, answer@answer-2 13%, call@aggregate_calc 12%, verify@verify-1 9%, answer@answer-3 9% |
| g4-skills | continual (cgrpo) | delegate@data-2 23%, delegate@data-4 23%, answer@answer-2 14%, call@aggregate_calc 12%, replan@plan-1 10%, answer@answer-3 8% |
| g4-skills | frozen | answer@answer-2 52%, delegate@data-2 35%, answer@answer-3 8%, verify@verify-1 1%, replan@plan-1 1%, replan@plan-3 1% |
| g4-skills | router static | verify@verify-2 16%, verify@verify-1 16%, answer@answer-4 12%, delegate@data-2 11%, replan@plan-2 9%, delegate@data-4 7% |
| g4-skills | router re-learned | engage@date_window 21%, answer@answer-2 21%, verify@verify-1 13%, answer@answer-3 8%, delegate@data-3 7%, delegate@data-4 7% |

## 9. SkillOrchestra-style handbooks

| handbook | fitted on day | episodes (learn / tune) | λ_c chosen | SN-IPS quality / cost at λ_c (ESS) | within budget |
|---|---|---|---|---|---|
| static | 5 | 384 / 96 | 0.0 | 1.000 / $0.0787 (ESS 4/96) | yes |
| relearn_g1-reprice | 11 | 192 / 96 | 0.0 | 0.597 / $0.0911 (ESS 7/96) | yes |
| relearn_g2-agents | 19 | 192 / 96 | 20.0 | 1.000 / $0.0278 (ESS 1/96) | yes |
| relearn_g3-tools | 27 | 192 / 96 | 0.0 | 0.705 / $0.0407 (ESS 3/96) | yes |
| relearn_g4-skills | 35 | 192 / 96 | 0.0 | 0.750 / $0.0690 (ESS 8/96) | yes |

## 10. Simulator calibration

Stage-0 harness, 1,000 simulated tasks per mode, against the real run's 50 tasks per mode:

| mode | accuracy sim / measured | cost/task sim / measured | role calls sim / measured | verifier calls sim / measured | cap hits sim / measured | P(flag ∣ wrong) sim / measured | P(flag ∣ correct) sim / measured |
|---|---|---|---|---|---|---|---|
| always_verify | 0.446 / 0.480 | $0.0894 / $0.0906 | 7.87 / 7.88 | 2.94 / 2.94 | 95% / 94% | 0.95 / 0.92 | 0.93 / 0.96 |
| never_verify | 0.459 / 0.460 | $0.0827 / $0.0821 | 4.75 / 4.74 | 0.00 / 0.00 | 0% / 0% | – / – | – / – |

Every parameter, with its status:

| parameter | value | status | source / note |
|---|---|---|---|
| initial plan length P(1, 2, 3) | 0.031 / 0.169 / 0.800 | measured | 130 planner calls, results/mas/mas_b2b.jsonl (4 / 22 / 104) |
| re-plan length | 1 sub-query | measured | 15 re-plans, all emitted 1 |
| data-2 (Sonnet) step kind rows / empty / error / invalid / respond | 0.761 / 0.151 / 0.057 / 0.027 / 0.003 | measured | 331 worker calls |
| data-1 (Haiku) step kind | 0.36 / 0.09 / 0.09 / 0.00 / 0.45 | measured, thin | 11 worker calls (controller:uniform) |
| data-3 (Opus) step kind | 0.80 / 0.10 / 0.10 / 0.00 / 0.00 | measured, thin | 10 worker calls |
| retried step: same kind distribution as a first attempt | rows 0.68 vs 0.76 | measured-consistent | 53 retries vs 299 first attempts; difference inside the noise |
| tokens planner / worker / verifier / answer (in, out) | 5,029/279 · 8,369/158 · 711/62 · 12,250/91 | measured | ledger means per role (cheap / mid / cheap / mid) |
| worker input tokens by position | 6833 + 1748·k | measured | mid worker calls at positions 0 / 1 / 2: 6,833 / 8,571 / 10,330 |
| cheap / frontier tokens per role | planner 6,996/159 · 6,902/311; worker 6,647/115 · 9,598/196; answer 7,393/245 · 15,227/332 | measured, thin | 5–15 calls each; priced at pace/prices.py list prices |
| verifier P(fail ∣ empty step) | 0.958 | measured | 23 of 24 verifier calls on empty steps |
| verifier P(fail ∣ rows step) | 0.581 | measured | 72 of 124; the same whether the step was good or bad (the measured verifier is uninformative: P(flag∣wrong) 0.92, P(flag∣correct) 0.96) |
| verifier structural fail on error / invalid steps | 1.0 at $0 | measured | 12 of 12 |
| answer competence per (agent, category) | logit of the MAS per-category accuracy (answer on mid) + single-agent (category, tier) offset | measured, shrunk | 130 MAS episodes, 150 single-agent episodes; pseudo-count 2.0 |
| P(good ∣ rows) per data agent | 0.60 / 0.75 / 0.85 | ASSUMED | not identifiable from the logs; ordered like the single-agent tier accuracies |
| coverage → answer logit | 1.5 × (coverage − 0.571) | ASSUMED (slope); centre self-calibrated | centre = never_verify's expected coverage, so never_verify reproduces the category base rates |
| task-level logit noise | sd 0.4 | ASSUMED |  |
| max role calls / replans / retries | 8 / 1 / 1 | measured (configuration) | the real orchestrator's defaults |
| worker 'respond' on the prefix call | disallowed | ASSUMED | the first worker call is always data-2 and always queries |
| verify-2 (cheap, verifier, stage 2) | ASSUMED: a better verifier, P(flag|correct) 0.30, same recall | ASSUMED |  |
| data-4 (cheap, data-worker, stage 2) | ASSUMED: database specialist at cheap price, rows 0.85 / good 0.80 | ASSUMED |  |
| answer-4 (cheap, answer, stage 2) | ASSUMED: reporting specialist at cheap price, mid-grade on 8 categories | ASSUMED |  |
| tool schema_lookup (stage 3) | ASSUMED: the next worker step cannot error (error/invalid mass moves to rows) | ASSUMED | priced as a verifier-sized cheap call (711/62 tokens) |
| tool aggregate_calc (stage 3) | ASSUMED: +0.34 coverage on aggregation categories, else nothing | ASSUMED | priced as a verifier-sized cheap call (711/62 tokens) |
| skill soql_aggregate (stage 4) | ASSUMED: +0.15 P(good | rows) on aggregation categories for the rest of the episode | ASSUMED | same price as a tool call |
| skill date_window (stage 4) | ASSUMED: +0.15 P(good | rows) on date-window categories | ASSUMED | same price as a tool call |
| skill decompose_fine (stage 4) | ASSUMED: re-plan into 3-4 finer sub-queries (the planner change that alters sub-query counts) | ASSUMED | same price as a tool call |
| planner v2 plan length P(1..4) | (0.0, 0.1, 0.4, 0.5) | ASSUMED | stage 4 |

## 11. Sensitivity: the mild schedule

Same days and axes, smaller effects (see `drift.SCHEDULES['mild']`); runtime 182.7 s.

| arm | mean BWT (stages < T) | mean FWT | cumulative regret vs oracle (days 6–40) | regret per day |
|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.035 | +0.075 | 1.100 | 0.0314 |
| continual (AWR+IPS, gated, λ = 2) | -0.005 | +0.002 | 3.668 | 0.1048 |
| frozen (trained days 1–5) | +0.000 | +0.053 | 1.970 | 0.0563 |
| SkillOrchestra-style static router | +0.000 | +0.025 | 2.902 | 0.0829 |
| SkillOrchestra-style router, re-learned per stage | +0.017 | +0.014 | 3.057 | 0.0873 |
| always_verify | +0.000 | +0.062 | 1.731 | 0.0495 |
| never_verify | +0.000 | +0.100 | 0.394 | 0.0113 |

**Stage g1 (day 9; oracle held-out J 0.406, uniform 0.325)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.022 | -0.020 | +0.108 | 7 | 0.359 | +0.006 | +0.026 | $0.0785 |
| continual (AWR+IPS, gated, λ = 2) | +0.045 | +0.012 | -0.010 | – | 0.339 | -0.079 | -0.001 | $0.0894 |
| frozen (trained days 1–5) | +0.039 | +0.058 | +0.000 | – | 0.383 | -0.023 | +0.000 | $0.0630 |
| SkillOrchestra-style static router | +0.021 | +0.029 | +0.000 | – | 0.354 | -0.053 | +0.000 | $0.0987 |
| SkillOrchestra-style router, re-learned per stage | +0.021 | +0.029 | -0.063 | – | 0.291 | -0.116 | +0.035 | $0.0601 |
| always_verify | +0.056 | +0.059 | +0.000 | – | 0.384 | -0.022 | +0.000 | $0.0732 |
| never_verify | +0.058 | +0.101 | +0.000 | 1 | 0.426 | +0.020 | +0.000 | $0.0675 |

**Stage g2 (day 17; oracle held-out J 0.387, uniform 0.299)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | -0.055 | +0.059 | +0.039 | 3 | 0.391 | +0.010 | +0.002 | $0.0868 |
| continual (AWR+IPS, gated, λ = 2) | -0.054 | -0.015 | -0.005 | – | 0.281 | -0.108 | +0.001 | $0.0552 |
| frozen (trained days 1–5) | -0.045 | +0.040 | +0.000 | – | 0.339 | -0.049 | +0.000 | $0.0608 |
| SkillOrchestra-style static router | -0.038 | +0.017 | +0.000 | – | 0.316 | -0.071 | +0.000 | $0.0770 |
| SkillOrchestra-style router, re-learned per stage | -0.034 | -0.042 | +0.096 | – | 0.353 | -0.034 | -0.031 | $0.0669 |
| always_verify | -0.041 | +0.044 | +0.000 | – | 0.343 | -0.044 | +0.000 | $0.0733 |
| never_verify | -0.047 | +0.081 | +0.000 | 1 | 0.380 | -0.007 | +0.000 | $0.0672 |

**Stage g3 (day 25; oracle held-out J 0.416, uniform 0.283)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.011 | +0.134 | -0.002 | 1 | 0.411 | -0.001 | -0.009 | $0.0883 |
| continual (AWR+IPS, gated, λ = 2) | +0.033 | +0.016 | -0.008 | – | 0.297 | -0.125 | -0.002 | $0.0700 |
| frozen (trained days 1–5) | +0.009 | +0.064 | +0.000 | – | 0.347 | -0.069 | +0.000 | $0.0628 |
| SkillOrchestra-style static router | +0.002 | +0.035 | +0.000 | – | 0.318 | -0.098 | +0.000 | $0.0772 |
| SkillOrchestra-style router, re-learned per stage | +0.005 | +0.075 | -0.052 | – | 0.306 | -0.110 | +0.046 | $0.0608 |
| always_verify | +0.014 | +0.074 | +0.000 | – | 0.357 | -0.059 | +0.000 | $0.0727 |
| never_verify | +0.017 | +0.114 | +0.000 | 1 | 0.397 | -0.019 | +0.000 | $0.0671 |

**Stage g4 (day 33; oracle held-out J 0.454, uniform 0.297)**

| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |
|---|---|---|---|---|---|---|---|---|
| continual (cgrpo on replay, gated, learned λ) | +0.013 | +0.126 | +0.002 | – | 0.428 | -0.029 | n/a | $0.0935 |
| continual (AWR+IPS, gated, λ = 2) | -0.000 | -0.006 | +0.004 | – | 0.297 | -0.159 | n/a | $0.0701 |
| frozen (trained days 1–5) | +0.001 | +0.051 | +0.000 | – | 0.348 | -0.106 | n/a | $0.0608 |
| SkillOrchestra-style static router | -0.001 | +0.020 | +0.000 | – | 0.317 | -0.136 | n/a | $0.0781 |
| SkillOrchestra-style router, re-learned per stage | -0.016 | -0.007 | +0.063 | – | 0.353 | -0.100 | n/a | $0.0761 |
| always_verify | +0.011 | +0.071 | +0.000 | – | 0.368 | -0.086 | n/a | $0.0740 |
| never_verify | +0.005 | +0.105 | +0.000 | – | 0.402 | -0.051 | n/a | $0.0696 |

Stationary J per stage:

| arm | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| oracle | 0.345 | 0.406 | 0.387 | 0.416 | 0.454 |
| continual (cgrpo on replay, gated, learned λ) | 0.283 | 0.359 | 0.391 | 0.411 | 0.428 |
| continual (AWR+IPS, gated, λ = 2) | 0.294 | 0.339 | 0.281 | 0.297 | 0.297 |
| frozen (trained days 1–5) | – | 0.383 | 0.339 | 0.347 | 0.348 |
| SkillOrchestra-style static router | – | 0.354 | 0.316 | 0.318 | 0.317 |
| SkillOrchestra-style router, re-learned per stage | – | 0.291 | 0.353 | 0.306 | 0.353 |
| always_verify | 0.329 | 0.384 | 0.343 | 0.357 | 0.368 |
| never_verify | 0.368 | 0.426 | 0.380 | 0.397 | 0.402 |
