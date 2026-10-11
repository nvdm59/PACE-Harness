Four variants × 5 seeds (seeds 0–4), `--arms ablation` (continual cgrpo, always_verify,
never_verify, per-stage oracle), about 220 s per run on CPU (6 runs in parallel on one machine). Variant A is the committed
configuration; it also supplies the seed error bars the main run lacks.

**1. "never_verify beats everything" is not an artifact of a single seed, but it is an artifact of
the measured verifier and the 8-call cap together.** Under the committed configuration (A, 5
seeds) always_verify is below never_verify by 0.033 ± 0.001 in expected quality and 0.036 ± 0.001
in priced J, for +$0.0036 per task: verification *hurts* at the real run's verifier and cap, and
the real run's +1 task (+0.019 ± 0.067 SE) is consistent with that at n = 50. Giving every arm the
informative verifier (B: per-call P(fail ∣ good rows) 0.58 → 0.114, recall unchanged) closes the
gap to −0.004 ± 0.001 but does not reverse it, because 91% of always_verify episodes still hit
the cap and the plan is truncated. Raising the cap to 12 (C) is what flips the sign: always_verify
is above never_verify by **+0.029 ± 0.001 in quality and +0.011 ± 0.001 in J**, for
+$0.0148 per task (+18%); with 3.65 verifier calls and 0.76 retries per episode and 9% cap hits
(the stage-4 four-step plans). Exempting verifier calls from the cap (D) changes nothing beyond
removing the last cap hits (+0.028 / +0.010): with one retry per sub-query a 3-step plan fits in
12 calls already. Each effect is many standard errors over 5 seeds.

**2. The conditions, stated plainly.** Verification pays when (i) a verifier fail carries
information about the step (the measured one does not: on the real log a flagged rows step stays
rows on retry 26 of 33 times, and the retried step is failed again at least as often as a fresh
one, 23% vs 37% pass), and (ii) the retry budget does not come out of the plan's budget. Under
(i) alone the retry repairs what it can (on the real log 10 of 20 flagged empty/error/invalid
steps became rows) but the cap eats the plan. Under (i) and (ii) the gain is about +0.03 in
accuracy for +18% cost, a priced gain of +0.01 J at the stage prices λ̄_g ≈ 0.7–1.3 per USD. That
is the size of the effect in this simulator, where a step's latent quality is the assumed link to
the answer (RESULTS.md section 10).

**3. Why the learner does not buy it.** Under C and D the fixed always_verify costs $0.098–0.105
per task, above the learner's budget B = $0.090 + $0.005; the continual arm is budget-constrained
and the oracle is too, so always_verify can exceed the constrained oracle at a stage (g1 in C:
0.443 vs 0.434). The learner's verify share rises from 0.03–0.09 (A) to 0.10–0.13 (B–D) of its
decisions and no further. Within budget, the learner still ends the capability stages above
never_verify under every variant (g4: A +0.031 ± 0.007, B +0.017 ± 0.011, C +0.016 ± 0.007,
D +0.013 ± 0.010) and below it in the warm-up and repricing stages, which is the crossover of
RESULTS.md finding 2, now with error bars. The regret ordering over the whole stream under C/D is
always_verify 0.30 < never_verify 0.60 < continual 1.05–1.20: at a verifier worth using and a cap
that does not truncate, the right fixed mode is `always_verify`, it is over budget, and the
budgeted learner is third, with its cold start still the whole deficit.

**4. The per-step number beside the per-episode one (real log, section 4).** 107 verifier fails,
53 followed by a retry. The retry changed the observation kind in 10 of 53 cases (19%), always
from empty/error to rows; in 33 of 53 the flagged step had returned rows and 26 of those returned
rows again. Paired with never_verify on the same tasks, the 44 always_verify episodes that
retried flipped 6 wrong → right and 5 right → wrong (25% flipped, symmetric). Per step and per
episode say the same thing: the measured verifier's flag is not information about correctness.
Note also that the informative verifier's episode-level P(flag ∣ correct) in the simulator is
0.71, not the 0.30 its per-call parameter was derived from, because correct episodes still contain
empty and wrong steps that are rightly flagged; the per-call parameter is what the ablation sets.
