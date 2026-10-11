Run `phase1b` (model Qwen/Qwen3-1.7B, 300 steps; dual step lr 0.02, kp 0.5, ema 0.95; beta 0.05; entropy_coef 0.01). Sim v1, 2000 fresh tasks (seed 777; training used seed 0), budget B = 0.30. J = quality - lambda_bar * cost with lambda_bar = 1.214, the mean learned lambda over the last 100 training steps (final lambda 1.369). First action from the policy, then ANSWER on the same tier. Quality = expected success probability. 'within budget' = cost <= B + 0.02.

| policy | quality | cost | J | within budget | invalid | first-action mix |
|---|---|---|---|---|---|---|
| base model zero-shot (greedy, no adapter) | 0.586 | 0.048 | 0.528 | yes | 18.9% | cheap RETRIEVE 51%, INVALID 19%, cheap ANSWER 13%, mid RETRIEVE 13% |
| LLM adapter_step100 (greedy) | 0.885 | 0.603 | 0.153 | NO | 0.0% | frontier RETRIEVE 82%, frontier ESCALATE 16%, frontier VERIFY 1%, frontier ANSWER 1% |
| LLM adapter_step100 (sampled T=1) | 0.885 | 0.602 | 0.153 | NO | 0.1% | frontier RETRIEVE 82%, frontier ESCALATE 16%, frontier VERIFY 1%, frontier ANSWER 1% |
| LLM adapter_step200 (greedy) | 0.884 | 0.599 | 0.157 | NO | 0.0% | frontier RETRIEVE 78%, frontier ESCALATE 19%, frontier ANSWER 2%, frontier VERIFY 1% |
| LLM adapter_step200 (sampled T=1) | 0.884 | 0.598 | 0.157 | NO | 0.0% | frontier RETRIEVE 77%, frontier ESCALATE 19%, frontier ANSWER 3%, frontier VERIFY 1% |
| LLM adapter_step300 (greedy) | 0.832 | 0.434 | 0.306 | NO | 0.1% | frontier ANSWER 83%, frontier ESCALATE 13%, frontier RETRIEVE 4%, frontier VERIFY 0% |
| LLM adapter_step300 (sampled T=1) | 0.832 | 0.433 | 0.306 | NO | 0.0% | frontier ANSWER 83%, frontier ESCALATE 12%, frontier RETRIEVE 4%, frontier VERIFY 0% |
| linear controller (policy_v1_sim.json, greedy) | 0.573 | 0.035 | 0.530 | yes | 0.0% | cheap RETRIEVE 100% |
| fixed cheap ANSWER | 0.522 | 0.020 | 0.498 | yes | 0.0% | cheap ANSWER 100% |
| fixed mid ANSWER | 0.688 | 0.100 | 0.566 | yes | 0.0% | mid ANSWER 100% |
| fixed frontier ANSWER | 0.820 | 0.400 | 0.335 | NO | 0.0% | frontier ANSWER 100% |
| best fixed action per proxy under lambda_bar (oracle ref.) | 0.690 | 0.078 | 0.595 | yes | 0.0% | cheap ESCALATE 60%, mid ESCALATE 40% |

Best LLM checkpoint within budget (by J): **none** of the LLM checkpoints is within budget

Base model zero-shot raw outputs (distinct, up to 6): ["<action>ANSWER</action>", "<action>RETRIEVE</action>", "<action>cheap ANSWER</action>", "<action>cheap ESCALATE</action>", "<action>cheap RETRIEVE</action>", "<action>cheap VERIFY</action>"]

Eval (all runs, generation + scoring) 465s on NVIDIA A100-SXM4-40GB.
