Run `crm_phase2` (model Qwen/Qwen3-1.7B; SFT 150 steps on per-category oracle labels at λ ∈ 0..20/USD from the 40 training tasks of fold 0, then GRPO lookup mode 100 steps, 16 generations, temperature 1.2; dual step lr 2.0, kp 5.0, ema 0.95, λ0 1.664241986523554; β 0.05; entropy 0.01). Budget B = $0.10; 'within budget' = cost <= B + $0.005. λ̄ = 0.000/USD (mean learned λ over the last 50 GRPO steps; final 0.000); the LLM prompts show λ=0 (λ̄ rounded to the 1/USD grid) and J = quality − 0·cost. Quality and cost are exact lookups of the logged arm the first action picks. Sampled rows average 16 samples per task.

#### held-out (fold 0 test) (10 tasks)

| policy | quality | cost (USD) | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | 0.600 | 0.0911 | 0.600 | yes | 70.0% | INVALID 70%, mid 30% |
| SFT (greedy) | 0.700 | 0.0730 | 0.700 | yes | 0.0% | cheap 60%, frontier 20%, mid 20% |
| SFT (sampled T=1, 16/prompt) | 0.656 | 0.0852 | 0.656 | yes | 0.0% | cheap 57%, frontier 28%, mid 16% |
| GRPO step 100 (greedy) | 0.700 | 0.0730 | 0.700 | yes | 0.0% | cheap 60%, frontier 20%, mid 20% |
| GRPO step 100 (sampled T=1, 16/prompt) | 0.681 | 0.0924 | 0.681 | yes | 0.0% | cheap 46%, frontier 30%, mid 24% |
| fixed cheap | 0.500 | 0.0687 | 0.500 | yes | 0.0% | cheap 100% |
| fixed mid | 0.600 | 0.1654 | 0.600 | NO | 0.0% | mid 100% |
| fixed frontier | 0.700 | 0.1421 | 0.700 | NO | 0.0% | frontier 100% |
| linear constrained policy, Part A fold 0 (served mix) | 0.590 | 0.1431 | 0.590 | NO | 0.0% | cheap 23%, mid 62%, frontier 15% |
| category-oracle rule at λ=0 from the 40 training tasks (SFT target) | 0.500 | 0.1352 | 0.500 | NO | 0.0% | cheap 50%, mid 40%, frontier 10% |
| per-task oracle at λ=0 (argmax q − λc on each task) | 0.800 | 0.0702 | 0.800 | yes | 0.0% | cheap 70%, mid 30% |
| per-task oracle (max quality, cheapest among ties) | 0.800 | 0.0594 | 0.800 | yes | 0.0% | cheap 60%, mid 30%, frontier 10% |

#### training (fold 0 train) (40 tasks)

| policy | quality | cost (USD) | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | 0.484 | 0.0808 | 0.484 | yes | 65.0% | INVALID 65%, mid 22%, cheap 12% |
| SFT (greedy) | 0.705 | 0.0728 | 0.705 | yes | 0.0% | cheap 50%, frontier 38%, mid 12% |
| SFT (sampled T=1, 16/prompt) | 0.680 | 0.0773 | 0.680 | yes | 0.6% | cheap 52%, frontier 35%, mid 12% |
| GRPO step 100 (greedy) | 0.730 | 0.0758 | 0.730 | yes | 0.0% | cheap 48%, frontier 38%, mid 15% |
| GRPO step 100 (sampled T=1, 16/prompt) | 0.717 | 0.0850 | 0.717 | yes | 0.0% | cheap 45%, frontier 40%, mid 16% |
| fixed cheap | 0.430 | 0.0679 | 0.430 | yes | 0.0% | cheap 100% |
| fixed mid | 0.609 | 0.1081 | 0.609 | NO | 0.0% | mid 100% |
| fixed frontier | 0.653 | 0.1698 | 0.653 | NO | 0.0% | frontier 100% |
| linear constrained policy, Part A fold 0 (served mix) | 0.634 | 0.1012 | 0.634 | yes | 0.0% | cheap 21%, mid 61%, frontier 17% |
| category-oracle rule at λ=0 from the 40 training tasks (SFT target) | 0.736 | 0.1240 | 0.736 | NO | 0.0% | cheap 42%, mid 30%, frontier 28% |
| per-task oracle at λ=0 (argmax q − λc on each task) | 0.761 | 0.1168 | 0.761 | NO | 0.0% | cheap 60%, mid 30%, frontier 10% |
| per-task oracle (max quality, cheapest among ties) | 0.761 | 0.0997 | 0.761 | yes | 0.0% | cheap 40%, mid 38%, frontier 22% |

Greedy tier mix on the 10 held-out tasks by shown price (λ per USD):

| model | λ=0 | λ=1 | λ=2 | λ=5 | λ=10 | λ=20 |
|---|---|---|---|---|---|---|
| zero-shot (greedy) | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 | INVALID 7, mid 3 |
| SFT (greedy) | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 |
| GRPO step 100 (greedy) | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 | cheap 6, frontier 2, mid 2 |

Wall clock in the container: 574s on NVIDIA A100-SXM4-40GB (GRPO 403.2s).
