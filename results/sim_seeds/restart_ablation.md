# Restart-state ablation for `train_cgrpo` (simulator pipeline, v1 stage)

`scripts/restart_ablation.py` (raw numbers: `restart_ablation.json`, policies: `restart_ablation/seed*/`).
For each replicate seed s = 0..4, with the component seeds of `scripts/run_sim_seeds.sh`:

1. log: 3000 uniform-behaviour v1 episodes, collect seed 10s+1 (the same log as the committed run; regenerated
   into a scratch directory because `traces/sim_seeds/` is not committed);
2. `train_cgrpo`, budget B = 0.30, train seed 10s+2, everything else at the defaults (batch 32, group 8, lr 1.5,
   PI-damped dual ascent, serve eps 0.05, iterate average over the second half), with `restart_mix` ρ:
   - **A** ρ = 0.5, 300 iterations: the committed default. It reproduces the committed
     `results/sim_seeds/seed{s}/policy_v1.json` exactly (W and all snapshots, 5/5 seeds).
   - **B** ρ = 0.0, matched simulator budget: `max_transitions` = A's transition count; training stops at the
     first iteration that has used up the budget, and the iterate average covers the iterations after half the
     budget was spent. Matched to within one iteration (+0.15% to +0.32% over A).
   - **C** ρ = 0.0, 300 iterations (matched iterations, not transitions; shown for reference).
3. ground truth: `pace.collect.true_eval` on a fresh v1 simulator, 20,000 tasks, seed 4242, expected quality
   (the evaluation of `pace.drift` row (i)); J = q − λ_ref·c with λ_ref = arm A's learned λ̄ for that seed (a
   common price, so the arms are compared on one objective);
4. gate v1 against the behaviour on the step-1 log (`pace.gate.gate`, bootstrap seed s, eps_q 0.03, tol 0.02).

A transition is one `harness.step` call inside training (every re-rolled step from every restart state).
Episode starts are longer to roll out than interior restarts, so ρ = 0 consumes about 8% more transitions per
iteration (C vs A); the matched arm B therefore ran 281 ± 5 iterations instead of 300.

## Summary (mean ± SE over 5 seeds)

| arm | ρ | iterations | transitions | λ̄ | truth quality | truth cost | J (λ_ref) | gate v1 DR quality | gate DR cost hi | PROMOTE |
|---|---|---|---|---|---|---|---|---|---|---|
| A (default) | 0.5 | 300 | 181,097 ± 3,465 | 0.358 ± 0.027 | 0.878 ± 0.001 | 0.306 ± 0.002 | 0.768 ± 0.008 | 0.884 ± 0.009 | 0.311 ± 0.001 | 5/5 |
| B (matched budget) | 0.0 | 281 ± 5 | 181,473 ± 3,455 | 0.391 ± 0.014 | 0.880 ± 0.001 | 0.308 ± 0.002 | 0.770 ± 0.008 | 0.883 ± 0.012 | 0.314 ± 0.002 | 5/5 |
| C (matched iters) | 0.0 | 300 | 196,204 ± 4,624 | 0.392 ± 0.013 | 0.879 ± 0.001 | 0.307 ± 0.002 | 0.770 ± 0.009 | 0.883 ± 0.012 | 0.313 ± 0.003 | 4/5 |

Paired differences against A (same seed, same log, same evaluation tasks), mean ± SE:

| | Δ quality | Δ cost | Δ J (λ_ref) |
|---|---|---|---|
| B − A (matched transitions) | +0.0020 ± 0.0011 | +0.0020 ± 0.0021 | +0.0014 ± 0.0008 |
| C − A (matched iterations) | +0.0016 ± 0.0013 | +0.0009 ± 0.0025 | +0.0015 ± 0.0009 |

All three arms are within budget on ground truth (cost ≤ B + tol = 0.32 in every seed). The only gate
difference is C, seed 2: DR cost upper bound 0.32000 against the 0.32 threshold (a knife-edge REVERT; truth cost
0.315). On ground truth, training on episode starts only is not distinguishable from the mixed-restart default
at this sample size: the paired J difference is +0.0014 ± 0.0008 (under 2 SE, and about 0.2% of J). The
learned price moves (λ̄ 0.358 → 0.391; its seed-to-seed SE halves), consistent with the dual variable now being
driven by every sampled start rather than by about 77% of them.

## Per seed

| seed | arm | iters | transitions | λ̄ | truth q / c | J (λ_ref) | gate DR q [90% CI] | DR cost hi | decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | A ρ=0.5 | 300 | 194,406 | 0.343 | 0.880 / 0.303 | 0.776 | 0.851 [0.815, 0.887] | 0.3090 | PROMOTE |
| 0 | B ρ=0, matched | 287 | 194,697 | 0.392 | 0.881 / 0.309 | 0.775 | 0.844 [0.804, 0.880] | 0.3176 | PROMOTE |
| 0 | C ρ=0, 300 it | 300 | 204,992 | 0.390 | 0.882 / 0.309 | 0.776 | 0.843 [0.803, 0.879] | 0.3166 | PROMOTE |
| 1 | A ρ=0.5 | 300 | 175,131 | 0.379 | 0.877 / 0.310 | 0.760 | 0.880 [0.848, 0.911] | 0.3104 | PROMOTE |
| 1 | B ρ=0, matched | 270 | 175,507 | 0.435 | 0.878 / 0.306 | 0.762 | 0.870 [0.831, 0.907] | 0.3087 | PROMOTE |
| 1 | C ρ=0, 300 it | 300 | 198,417 | 0.430 | 0.878 / 0.303 | 0.763 | 0.871 [0.831, 0.907] | 0.3060 | PROMOTE |
| 2 | A ρ=0.5 | 300 | 177,308 | 0.278 | 0.877 / 0.309 | 0.791 | 0.901 [0.878, 0.921] | 0.3135 | PROMOTE |
| 2 | B ρ=0, matched | 275 | 177,637 | 0.370 | 0.883 / 0.316 | 0.795 | 0.907 [0.887, 0.925] | 0.3199 | PROMOTE |
| 2 | C ρ=0, 300 it | 300 | 197,201 | 0.379 | 0.882 / 0.315 | 0.795 | 0.908 [0.888, 0.926] | 0.3200 | REVERT |
| 3 | A ρ=0.5 | 300 | 177,486 | 0.443 | 0.878 / 0.306 | 0.743 | 0.883 [0.845, 0.905] | 0.3130 | PROMOTE |
| 3 | B ρ=0, matched | 299 | 177,792 | 0.356 | 0.878 / 0.304 | 0.743 | 0.884 [0.848, 0.905] | 0.3090 | PROMOTE |
| 3 | C ρ=0, 300 it | 300 | 178,532 | 0.353 | 0.876 / 0.303 | 0.741 | 0.881 [0.846, 0.903] | 0.3090 | PROMOTE |
| 4 | A ρ=0.5 | 300 | 181,155 | 0.346 | 0.877 / 0.302 | 0.773 | 0.903 [0.870, 0.925] | 0.3081 | PROMOTE |
| 4 | B ρ=0, matched | 274 | 181,732 | 0.401 | 0.879 / 0.304 | 0.774 | 0.909 [0.875, 0.928] | 0.3124 | PROMOTE |
| 4 | C ρ=0, 300 it | 300 | 201,876 | 0.407 | 0.880 / 0.305 | 0.775 | 0.910 [0.875, 0.929] | 0.3120 | PROMOTE |

Arm A's truth rows equal the committed `RESULTS.md` "truth v1 q / c" column for every seed (same policy, same
evaluation). The gate bar (on-policy behaviour mean) is the same for all arms of a seed.

## What the mixed-restart gradient optimises

**Notation.** With θ the weights, π = π_θ the served (ε-mixed) policy, and λ the current price, let
V_π(s) = E_π[q − λ·c_{s→T} | s] be the Lagrangian value from state s. Here c_{s→T} is the cost from s to the end
of the episode, which is exactly the `c_rem` the trainer uses. d_0 is the empirical distribution of logged
episode starts (the 3000 logged tasks, i.e. d_tasks), and d_L is the empirical distribution of all logged states.
d_L is the **behaviour's** state visitation, and 53% of its states are step-0 states on these logs. Each update
draws its B restart states from the fixed mixture

  d_restart = (1 − ρ)·d_0 + ρ·d_L,

which puts 1 − ρ + 0.53ρ ≈ 0.77 of the mass on starts at ρ = 0.5. From each restart state s the update re-rolls G
continuations from π. The update is ĝ = (1/BG) Σ_s Σ_k (R_k − R̄_s) Σ_t ∇log π(a_t|x_t), where R_k is the
continuation's return and R̄_s is the group mean.

**Primal.** d_restart does not depend on θ: the restart states come from the log, not from π. Inside one update
it is therefore a fixed distribution. The group-mean baseline introduces only the factor (G − 1)/G, because each
continuation sits in its own group mean. So

  E[ĝ] = ((G − 1)/G) · E_{s∼d_restart}[∇_θ V_π(s)] = ((G − 1)/G) · ∇_θ F_ρ(θ),
  F_ρ(θ) = (1 − ρ)·E_{s0∼d_0}[V_π(s0)] + ρ·E_{s∼d_L}[V_π(s)].

This is the **restart-distribution objective** F_ρ, not the episode-start objective
F_0(θ) = E_{s0∼d_tasks}[V_π(s0)]. By the policy-gradient theorem, ∇F_0 weights each state by **π's own**
visitation from the starts. The ρ-term instead adds weight on states the uniform behaviour reached, which π may
rarely or never reach (for example states after a frontier-tier VERIFY). The two objectives share a maximiser
when the policy class can be optimal at every state simultaneously: an optimal policy for fixed λ is optimal from
every state (Bellman optimality), so maximising V at extra restart states costs nothing. They can differ under
function approximation. Here π is one linear-softmax map shared across steps, and the base features are crossed
only with the hard proxy and the version, so improving V on behaviour-only states can trade off against the states
π visits. The ρ-term then changes the fixed point, not only the speed.

**Dual.** The multiplier update λ ← [λ_I + η(Ĉ − B)]₊ (plus the proportional term) uses Ĉ, an EMA of
**full-episode** costs c_{0→T}, computed only from rollouts whose restart state is an episode start
(`base.step == 0`). Ĉ estimates C(π) = E_{s0∼d_tasks}[c_{0→T}], so the constraint is enforced on the
episode-start objective, which is correct. The primal, however, ascends the restart-weighted Lagrangian
L_ρ(θ, λ) = E_{d_restart}[V_q − λV_c], while the dual descends ∂_λ of L_0(θ, λ) = E_{d_0}[V_q − λV_c] + λB. The
coupled iteration seeks a point where ∇_θ L_ρ = 0 and the episode-start constraint binds. That is a saddle point
of neither Lagrangian unless the two share a maximiser in θ (see above). At ρ = 0 the primal and the dual refer to
the same Lagrangian L_0. Interior restarts are then a variance and coverage device that changes the objective;
they are not an unbiased estimator of ∇F_0.

**Empirically**, on this simulator the mismatch is not measurable: at matched transitions, ρ = 0 gives
ΔJ = +0.0014 ± 0.0008 against ρ = 0.5, with ground-truth cost within budget in both arms.
