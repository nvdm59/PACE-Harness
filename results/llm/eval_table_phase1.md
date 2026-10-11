Run `phase1` (model Qwen/Qwen3-1.7B, 300 steps; dual step lr 0.1, kp 2.0, ema 0.9; beta 0.02; entropy_coef 0.0). Sim v1, 2000 fresh tasks (seed 777; training used seed 0), budget B = 0.30. J = quality - lambda_bar * cost with lambda_bar = 0.000, the mean learned lambda over the last 100 training steps (final lambda 0.000). First action from the policy, then ANSWER on the same tier. Quality = expected success probability. 'within budget' = cost <= B + 0.02.

| policy | quality | cost | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| base model zero-shot (greedy, no adapter) | 0.586 | 0.048 | 0.586 | yes | 18.9% | cheap RETRIEVE 51%, INVALID 19%, cheap ANSWER 13%, mid RETRIEVE 13% |
| LLM adapter_step100 (greedy) | 0.748 | 0.155 | 0.748 | yes | 0.0% | mid RETRIEVE 100%, mid ESCALATE 0% |
| LLM adapter_step100 (sampled T=1) | 0.748 | 0.155 | 0.748 | yes | 0.1% | mid RETRIEVE 100%, mid ESCALATE 0%, frontier RETRIEVE 0%, cheap RETRIEVE 0% |
| LLM adapter_step200 (greedy) | 0.748 | 0.155 | 0.748 | yes | 0.0% | mid RETRIEVE 100%, mid ESCALATE 0% |
| LLM adapter_step200 (sampled T=1) | 0.749 | 0.155 | 0.749 | yes | 0.0% | mid RETRIEVE 99%, mid ESCALATE 1% |
| LLM adapter_step300 (greedy) | 0.749 | 0.155 | 0.749 | yes | 0.0% | mid RETRIEVE 99%, mid ESCALATE 1% |
| LLM adapter_step300 (sampled T=1) | 0.749 | 0.155 | 0.749 | yes | 0.0% | mid RETRIEVE 98%, mid ESCALATE 2% |
| linear controller (policy_v1_sim.json, greedy) | 0.573 | 0.035 | 0.573 | yes | 0.0% | cheap RETRIEVE 100% |
| fixed cheap ANSWER | 0.522 | 0.020 | 0.522 | yes | 0.0% | cheap ANSWER 100% |
| fixed mid ANSWER | 0.688 | 0.100 | 0.688 | yes | 0.0% | mid ANSWER 100% |
| fixed frontier ANSWER | 0.820 | 0.400 | 0.820 | NO | 0.0% | frontier ANSWER 100% |
| best fixed action per proxy under lambda_bar (oracle ref.) | 0.897 | 0.600 | 0.897 | NO | 0.0% | frontier ESCALATE 60%, frontier VERIFY 40% |

Best LLM checkpoint within budget (by J): **LLM adapter_step300 (sampled T=1)**: quality 0.749, cost 0.155, J 0.749

Base model zero-shot raw outputs (distinct, up to 6): ["<action>ANSWER</action>", "<action>RETRIEVE</action>", "<action>cheap ANSWER</action>", "<action>cheap ESCALATE</action>", "<action>cheap RETRIEVE</action>", "<action>cheap VERIFY</action>"]

Eval (all runs, generation + scoring) 463s on NVIDIA A100-SXM4-40GB.
