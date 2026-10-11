# Continual orchestrator: RAO-style delegation credit and a meta-learned controller init

Same simulator, same pre-registered 40-day default schedule (events on days 9 / 17 / 25 / 33), 5 seeds (0–4), the
arm set of the 5-seed runs (`--arms ablation`: continual cgrpo, always_verify, never_verify, per-stage oracle). The
cgrpo arm is run three ways: committed (`--delegation-credit none --init zero`), `--delegation-credit rao --lambda-d 0.5`,
and `--init meta`. Only the cgrpo arm changes; the fixed arms and oracles are re-run in each variant. The `none / zero`
runs reproduce variant A of the verifier ablation (regret days 6–40 1.061 ± 0.205 against 1.06 ± 0.21). Mean ± SE over
5 seeds. Wall time: 15 runs in 9 min 56 s on 8 CPU cores (meta-init training 28 s per seed). $0, no API.

## Interpretation

1. **RAO credit causes over-delegation (negative).** Delegation share rises at every stage (paired Δ +0.05 to +0.10;
   +0.102 ± 0.008 at g3) and stationary J falls at every stage from g1 (Δ −0.023 to −0.036, SE 0.005–0.016);
   regret days 6–40 rises from 1.06 ± 0.21 to 1.76 ± 0.16 (paired +0.69 ± 0.26). The bonus is not in the
   root-anchored baseline, so every delegate decision that spawns a good sub-call gets a positive push regardless of
   whether delegating was the better choice. That bias is the trap.
2. **It changes WHEN and to WHOM.** The learner delegates to the new specialist data-4 sooner after g2 (share in the
   stage's first 4 days 0.076 vs 0.047). It ends at 0.43 vs 0.21 of decisions on data-4 at g4. It also moves mass to
   the frontier worker data-3 (0.14–0.18 vs 0.006 at g2–g3), whose higher sub-call success rate the credit rewards and
   whose price the episode outcome does not offset. More delegation to better workers, paid for in cost and answers.
3. **The meta-init removes most of the cold start.** Served J over days 1–12 is 0.362 ± 0.002 vs 0.325 ± 0.008 (never_verify
   0.378). Regret days 6–40 is 0.47 ± 0.08 vs 1.06 ± 0.21 (paired −0.59 ± 0.15). That is below never_verify's 0.74 ± 0.08.
   Regret days 17–40 is 0.43 vs 0.76. Stationary J is higher at g0–g3 and unchanged at g4 (0.454 vs 0.456).
4. **The first-promotion day does not move.** It is 5.2 ± 1.6 (zero) vs 5.2 ± 1.8 (meta). With this arm set the zero
   init promotes on day 2–11, not the committed seed-0 all-arms run's day 13. The gain comes from what is served
   before and between promotions, not from a faster gate.
5. **On the one stage it never saw (g4) the meta-init is no better.** Days to recover are 6.5 vs 3.5 (4 of 5 seeds
   recover in both), and stationary J is equal. Its mean BWT is smaller (+0.008 vs +0.026) because it starts closer
   to each stage's optimum, so there is less to win back.
6. **Caveat.** The meta-init was trained on the same simulator, and that includes the *definitions* of stages g1–g3,
   which the run only reaches later. It never read a held-out task set, the daily traffic, an oracle or stage g4
   (audited per seed). On real data it would need logged or simulated past harness generations (older price sheets,
   agent pools, tool sets) with the deployment's future stages excluded, plus enough episodes per generation to run
   cGRPO inner loops (here 2 × 96 episodes per stage per outer round, 3 rounds). Whether a past-generations-only
   meta-init keeps this gain is untested here.
7. The baseline delegation shares in the brief (0.07 → 0.30 → 0.52 → 0.51 → 0.54) do not reproduce from the committed
   `days.jsonl`. That file logs only the top-6 `action_mix` (a lower bound), which gives 0.06 / 0.25 / 0.48 / 0.45 / 0.44
   as stage means. The new runs log the full `delegate_share` per day.

## A. RAO-style delegation credit

Node reward of a `delegate@<agent>` decision = R + λ_d · mean(success of the sub-call(s) it spawned). R = the
priced episode outcome (quality − λ·cost-to-go). Success = the simulator's latent "right rows" flag of the worker
step. Every decision of rollout k uses the root-anchored leave-one-out baseline mean_{j≠k} R_j (`rao_reward.py`).

Delegation share of served decisions (stage mean of daily shares), by agent as share of all decisions:

| row | g0 | g1 | g2 | g3 | g4 |
|---|---|---|---|---|---|
| committed run (seed 0, all arms; top-6 action_mix, stage mean) | 0.06 | 0.25 | 0.48 | 0.45 | 0.44 |
| continual, credit none: delegation share | 0.225 ± 0.031 | 0.439 ± 0.038 | 0.480 ± 0.010 | 0.461 ± 0.008 | 0.475 ± 0.008 |
| continual, credit none: share, first 4 days of stage | 0.224 ± 0.016 | 0.367 ± 0.076 | 0.476 ± 0.013 | 0.463 ± 0.009 | 0.476 ± 0.012 |
| continual, credit none: → data-1 | 0.048 ± 0.007 | 0.017 ± 0.004 | 0.006 ± 0.001 | 0.004 ± 0.001 | 0.005 ± 0.000 |
| continual, credit none: → data-2 | 0.127 ± 0.035 | 0.408 ± 0.042 | 0.382 ± 0.023 | 0.288 ± 0.036 | 0.251 ± 0.023 |
| continual, credit none: → data-3 | 0.050 ± 0.004 | 0.015 ± 0.003 | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.005 ± 0.001 |
| continual, credit none: → data-4 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.086 ± 0.028 | 0.162 ± 0.041 | 0.214 ± 0.026 |
| continual, credit none: → data-4, first 4 days of stage | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.047 ± 0.016 | 0.141 ± 0.049 | 0.194 ± 0.043 |
| continual, credit RAO (λ_d 0.5): delegation share | 0.323 ± 0.053 | 0.485 ± 0.070 | 0.534 ± 0.016 | 0.563 ± 0.003 | 0.551 ± 0.005 |
| continual, credit RAO (λ_d 0.5): share, first 4 days of stage | 0.265 ± 0.032 | 0.470 ± 0.069 | 0.512 ± 0.031 | 0.557 ± 0.004 | 0.525 ± 0.005 |
| continual, credit RAO (λ_d 0.5): → data-1 | 0.049 ± 0.009 | 0.022 ± 0.013 | 0.008 ± 0.003 | 0.005 ± 0.000 | 0.005 ± 0.000 |
| continual, credit RAO (λ_d 0.5): → data-2 | 0.216 ± 0.071 | 0.387 ± 0.087 | 0.272 ± 0.055 | 0.160 ± 0.026 | 0.042 ± 0.029 |
| continual, credit RAO (λ_d 0.5): → data-3 | 0.058 ± 0.009 | 0.076 ± 0.027 | 0.143 ± 0.035 | 0.179 ± 0.023 | 0.079 ± 0.034 |
| continual, credit RAO (λ_d 0.5): → data-4 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.110 ± 0.021 | 0.219 ± 0.013 | 0.425 ± 0.052 |
| continual, credit RAO (λ_d 0.5): → data-4, first 4 days of stage | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.076 ± 0.015 | 0.186 ± 0.009 | 0.375 ± 0.044 |
| never_verify: delegation share | 0.640 ± 0.001 | 0.638 ± 0.001 | 0.641 ± 0.001 | 0.641 ± 0.001 | 0.705 ± 0.001 |
| always_verify: delegation share | 0.330 ± 0.000 | 0.331 ± 0.000 | 0.330 ± 0.000 | 0.331 ± 0.000 | 0.333 ± 0.000 |

Held-out J, transfer and regret:

| row | stationary J g0 | g1 | g2 | g3 | g4 | mean BWT | mean FWT | regret 6–40 | regret 17–40 |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 0.336 ± 0.007 | 0.434 ± 0.012 | 0.441 ± 0.010 | 0.476 ± 0.013 | 0.474 ± 0.015 | | | | |
| continual, credit none | 0.304 ± 0.009 | 0.428 ± 0.014 | 0.407 ± 0.005 | 0.453 ± 0.019 | 0.456 ± 0.011 | 0.026 ± 0.006 | 0.090 ± 0.006 | 1.061 ± 0.205 | 0.757 ± 0.180 |
| continual, credit RAO | 0.309 ± 0.020 | 0.392 ± 0.026 | 0.384 ± 0.007 | 0.418 ± 0.015 | 0.432 ± 0.013 | 0.014 ± 0.008 | 0.074 ± 0.004 | 1.755 ± 0.159 | 1.347 ± 0.085 |
| never_verify | 0.354 ± 0.006 | 0.437 ± 0.013 | 0.420 ± 0.008 | 0.442 ± 0.012 | 0.425 ± 0.014 | 0.000 ± 0.000 | 0.109 ± 0.002 | 0.743 ± 0.084 | 0.824 ± 0.063 |
| always_verify | 0.317 ± 0.007 | 0.404 ± 0.012 | 0.388 ± 0.008 | 0.408 ± 0.013 | 0.385 ± 0.015 | 0.000 ± 0.000 | 0.074 ± 0.003 | 1.974 ± 0.087 | 1.675 ± 0.064 |
| RAO − none (paired by seed): Δ stationary J | 0.005 ± 0.021 | -0.036 ± 0.016 | -0.023 ± 0.005 | -0.035 ± 0.011 | -0.024 ± 0.011 | | | 0.694 ± 0.264 | |
| RAO − none (paired by seed): Δ delegation share | 0.098 ± 0.074 | 0.046 ± 0.095 | 0.054 ± 0.014 | 0.102 ± 0.008 | 0.076 ± 0.011 | | | | |

## B. Meta-learned controller init

Reptile / first-order MAML (`meta_init.py`). Inner: 2 days × 96 fresh tasks of stage g served by the current φ
(ε = 0.1), then the committed cGRPO replay trainer warm-started from it. Outer: φ ← mean of the adapted
(iterate-averaged) weights over g0..g3, and λ₀ ← mean λ̄. 3 rounds, β = 1. The meta-init sees only the harness
definitions of g0..g3 and task seeds ≥ 7,000,000. It never sees held-out seeds (seed + 500 + g), daily-traffic seeds,
oracle data or g4. All 5 per-seed audits report no overlap.

| init | first promotion day | days to recover g1 / g2 / g3 / g4 (never = –; n recovered of 5) | served J days 1–12 | regret 6–40 | regret 17–40 | mean BWT | mean FWT |
|---|---|---|---|---|---|---|---|
| zero (uniform policy) | 5.200 ± 1.562 ([2, 3, 5, 5, 11]) | 3.4 (5) / 4.7 (3) / 3.7 (3) / 3.5 (4) | 0.325 ± 0.008 | 1.061 ± 0.205 | 0.757 ± 0.180 | 0.026 ± 0.006 | 0.090 ± 0.006 |
| meta (Reptile over g0–g3) | 5.200 ± 1.828 ([2, 3, 3, 6, 12]) | 3.0 (5) / 1.8 (4) / 3.4 (5) / 6.5 (4) | 0.362 ± 0.002 | 0.471 ± 0.076 | 0.427 ± 0.066 | 0.008 ± 0.002 | 0.114 ± 0.003 |
| never_verify (reference) | – | – | 0.378 ± 0.002 | 0.743 ± 0.084 | 0.824 ± 0.063 | | |
| meta − zero (paired by seed) | | | | -0.590 ± 0.147 | -0.330 ± 0.125 | | |

| init | stationary J g0 | stationary J g1 | stationary J g2 | stationary J g3 | stationary J g4 |
|---|---|---|---|---|---|
| zero | 0.304 ± 0.009 | 0.428 ± 0.014 | 0.407 ± 0.005 | 0.453 ± 0.019 | 0.456 ± 0.011 |
| meta | 0.344 ± 0.007 | 0.433 ± 0.011 | 0.425 ± 0.006 | 0.462 ± 0.012 | 0.454 ± 0.015 |

Run directories (untracked): `results/mas_continual/rao_metainit/{base,rao,meta}/seed{0..4}`; tables:
`python -m pace.experiments.mas_continual.analyze --run results/mas_continual --rao-metainit results/mas_continual/rao_metainit`.

## Control: RAO baseline with λ_d = 0 (5 seeds, added after review)

The RAO mode changes two things at once — the delegation credit and the baseline (leave-one-out instead of group mean). Running the RAO mode with λ_d = 0 isolates the baseline change.

| condition | regret days 6–40 (served, mean ± SE) | delegation share g0 … g4 |
|---|---|---|
| none (group-mean baseline), zero init | 0.94 ± 0.27 | 0.17, 0.44, 0.50, 0.51, 0.55 |
| RAO baseline, λ_d = 0 | 0.99 ± 0.24 | 0.23, 0.35, 0.51, 0.51, 0.56 |
| RAO credit, λ_d = 0.5 | 1.74 ± 0.18 | 0.26, 0.47, 0.57, 0.62, 0.62 |
| none, meta-learned init | 0.39 ± 0.15 | 0.51, 0.54, 0.53, 0.51, 0.54 |

The baseline change alone leaves regret and delegation share where they were; the credit is what raises delegation and regret. The RAO negative result stands: on this simulator a delegation bonus outside the baseline buys over-delegation. (Regret here is recomputed from the served-J rows of each run's days.jsonl; the agent's table above used its own aggregation, hence 1.06 vs 0.94 for the same `none` runs — same ordering, same conclusion.)
