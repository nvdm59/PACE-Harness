## Exact vs sampled constrained training: CRMArena-Pro B2B, 300 tasks, 5-fold CV (seed 0)

Held-out means over the 5 test folds (exact lookups). J at a common λ (2 and 5 per dollar). Budget B = $0.10, tol $0.005; flags on the mean held-out cost; 'folds ok' = folds with held-out cost ≤ B + tol. 'table reads' = (task, tier) cells read by the trainer: sampled = iters·batch·group = 300·32·8 sampled transitions; exact = iters·tasks·3; LP = one pass over the training cells. Multiplier: learned λ̄ (trainers) or the LP's dual price μ of the budget constraint. Wall-clock is training time per fold on this machine (CPU).

| policy | quality | cost (USD) | J(λ=2) [SD] | J(λ=5) [SD] | cost ≤ B | cost ≤ B+tol | folds ok | multiplier (mean) | wall s / fold | table reads / fold |
|---|---|---|---|---|---|---|---|---|---|---|
| sampled cgrpo-lookup (served: iterate mix) | 0.595910 | 0.101647 | 0.392617 [0.049478] | 0.087677 [0.097623] | NO | yes | 4/5 | 1.220572 | 0.075 | 76800 |
| sampled cgrpo-lookup (greedy of the mix) | 0.643530 | 0.102525 | 0.438480 [0.081547] | 0.130907 [0.163865] | NO | yes | 4/5 | 1.220572 | 0.075 | 76800 |
| exact-gradient, full batch (served: iterate mix) | 0.609981 | 0.102607 | 0.404768 [0.047351] | 0.096948 [0.097447] | NO | yes | 4/5 | 0.657319 | 0.100 | 216000 |
| exact-gradient, full batch (greedy of the mix) | 0.640196 | 0.101058 | 0.438081 [0.081488] | 0.134908 [0.165122] | NO | yes | 4/5 | 0.657319 | 0.100 | 216000 |
| exact-gradient, minibatch 32 (served: iterate mix) | 0.602642 | 0.103126 | 0.396389 [0.046673] | 0.087010 [0.094911] | NO | yes | 4/5 | 0.997891 | 0.055 | 28800 |
| exact-gradient, minibatch 32 (greedy of the mix) | 0.638064 | 0.105910 | 0.426244 [0.074380] | 0.108514 [0.152557] | NO | NO | 4/5 | 0.997891 | 0.055 | 28800 |
| category LP, train-fold means (served, stochastic) | 0.643632 | 0.094070 | 0.455491 [0.052096] | 0.173280 [0.063760] | yes | yes | 4/5 | 0.000000 | 0.003 | 720 |
| category LP, train-fold means (greedy: argmax tier per category) | 0.643632 | 0.094070 | 0.455491 [0.052096] | 0.173280 [0.063760] | yes | yes | 4/5 | 0.000000 | 0.003 | 720 |
| best fixed-arm mixture at cost <= B (train-fold LP, context-free) | 0.542760 | 0.100771 | 0.341219 [0.035137] | 0.038907 [0.091294] | NO | yes | 4/5 | – | 0.000 | 720 |

Category LP per fold (training-fold objective): fold 0: q 0.6760, c 0.0994, randomized categories 0, unseen test categories 0; fold 1: q 0.6584, c 0.0895, randomized categories 0, unseen test categories 0; fold 2: q 0.6692, c 0.0957, randomized categories 0, unseen test categories 0; fold 3: q 0.6641, c 0.0898, randomized categories 0, unseen test categories 0; fold 4: q 0.6478, c 0.0944, randomized categories 0, unseen test categories 0.

The budget constraint is slack in every fold's LP (μ = 0, no randomized category): on training-fold category means, the highest-quality tier of every category already costs ≤ B on average, so the LP optimum is the per-category max-quality tier (cheaper on quality ties), and its stochastic and greedy rows coincide.
