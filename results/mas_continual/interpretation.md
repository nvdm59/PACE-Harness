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
