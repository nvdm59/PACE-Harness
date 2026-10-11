Run `phase1c` (model Qwen/Qwen3-1.7B, 200 steps; dual step lr 0.02, kp 0.5, ema 0.95; beta 0.05; entropy_coef 0.01; num_generations 16; SFT warm start 150 steps; events [{'step': 20, 'event': 'temperature_raised', 'from': 1.2, 'to': 1.5, 'reason': 'degenerate_group_rate > 0.6 for 20 consecutive steps'}]). Sim v1, 2000 fresh tasks (seed 777; training used seed 0), budget B = 0.30. J = quality - lambda_bar * cost with lambda_bar = 0.443, the mean learned lambda over the last 100 training steps (final lambda 0.436). First action from the policy, then ANSWER on the same tier. Quality = expected success probability. 'within budget' = cost <= B + 0.02. The LLM prompts show the price λ=0.4 (lambda_bar rounded).

| policy | quality | cost | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| base model zero-shot (greedy, no adapter) | 0.592 | 0.054 | 0.567 | yes | 11.4% | cheap RETRIEVE 58%, cheap ANSWER 14%, mid RETRIEVE 13%, INVALID 11% |
| LLM adapter_sft (greedy) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| LLM adapter_sft (sampled T=1) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| LLM adapter_step100 (greedy) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| LLM adapter_step100 (sampled T=1) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| LLM adapter_step200 (greedy) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| LLM adapter_step200 (sampled T=1) | 0.848 | 0.330 | 0.702 | NO | 0.0% | mid ESCALATE 60%, frontier VERIFY 40% |
| linear controller (policy_v1_sim.json, greedy) | 0.573 | 0.035 | 0.557 | yes | 0.0% | cheap RETRIEVE 100% |
| fixed cheap ANSWER | 0.522 | 0.020 | 0.513 | yes | 0.0% | cheap ANSWER 100% |
| fixed mid ANSWER | 0.688 | 0.100 | 0.643 | yes | 0.0% | mid ANSWER 100% |
| fixed frontier ANSWER | 0.820 | 0.400 | 0.643 | NO | 0.0% | frontier ANSWER 100% |
| best fixed action per proxy under lambda_bar (oracle ref.) | 0.814 | 0.250 | 0.703 | yes | 0.0% | mid ESCALATE 60%, frontier ANSWER 40% |
| best deterministic per-difficulty rule, cost <= B (easy mid ESCALATE, hard frontier ANSWER) | 0.814 | 0.250 | 0.703 | yes | 0.0% | mid ESCALATE 60%, frontier ANSWER 40% |
| best randomized per-difficulty policy at cost B (LP; 0.38 x easy mid ESCALATE/hard frontier ANSWER + rest easy mid ESCALATE/hard frontier VERIFY) | 0.835 | 0.300 | 0.702 | yes | 0.0% |  |

Best LLM checkpoint within budget (by J): **none** of the LLM checkpoints is within budget

Base model zero-shot raw outputs (distinct, up to 6): ["<action>ANSWER</action>", "<action>RETRIEVE</action>", "<action>VERIFY</action>", "<action>cheap ANSWER</action>", "<action>cheap ESCALATE</action>", "<action>cheap RETRIEVE</action>"]

Eval (all runs, generation + scoring) 278s on NVIDIA A100-SXM4-40GB.
