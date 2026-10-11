# verl run `verl_smoke`

Status: **ok**; container wall clock 795.2 s on NVIDIA A100-SXM4-40GB; first step logged 481.4003713130951 s after container start; versions {'verl': '0.9.1', 'vllm': '0.24.0', 'torch': '2.11.0+cu130', 'transformers': '5.5.3', 'ray': '2.56.1', 'peft': '0.19.1'}.

Fixed price λ = 0.4 in the prompt and in the reward; 8 prompts x 8 completions per step; every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.

### verl: per step

| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |
|---|---|---|---|---|---|---|---|
| 1 | 0.476 | 0.539 | 0.063 | 3.1% | 25% | 101.7 | cheap RETRIEVE 30, mid RETRIEVE 17, cheap ANSWER 15 |
| 2 | 0.532 | 0.772 | 0.067 | 12.5% | 75% | 19.1 | cheap RETRIEVE 27, cheap ANSWER 16, mid RETRIEVE 10 |
| 3 | 0.355 | 0.518 | 0.031 | 12.5% | 88% | 14.5 | cheap RETRIEVE 39, cheap ANSWER 16, INVALID 8 |
| 4 | 0.417 | 0.654 | 0.054 | 12.5% | 50% | 15.2 | cheap RETRIEVE 27, cheap ANSWER 16, mid RETRIEVE 13 |
| 5 | 0.530 | 0.583 | 0.065 | 1.6% | 12% | 16.4 | cheap RETRIEVE 34, mid RETRIEVE 17, cheap ANSWER 11 |
| 6 | 0.343 | 0.551 | 0.127 | 12.5% | 0% | 14.4 | cheap RETRIEVE 32, mid RETRIEVE 13, INVALID 8 |
| 7 | 0.569 | 0.603 | 0.084 | 0.0% | 12% | 15.4 | cheap RETRIEVE 29, mid RETRIEVE 24, cheap ANSWER 5 |
| 8 | 0.671 | 0.706 | 0.088 | 0.0% | 50% | 14.8 | mid RETRIEVE 23, cheap RETRIEVE 18, cheap ANSWER 17 |
| 9 | 0.278 | 0.506 | 0.134 | 10.9% | 62% | 14.8 | mid RETRIEVE 26, cheap RETRIEVE 23, INVALID 7 |
| 10 | 0.680 | 0.732 | 0.131 | 0.0% | 12% | 15.8 | cheap RETRIEVE 33, mid RETRIEVE 23, frontier RETRIEVE 6 |
| 11 | 0.660 | 0.720 | 0.148 | 0.0% | 62% | 14.9 | mid RETRIEVE 39, cheap RETRIEVE 18, frontier ANSWER 7 |
| 12 | 0.527 | 0.584 | 0.143 | 0.0% | 50% | 15.6 | mid RETRIEVE 41, cheap RETRIEVE 15, frontier ANSWER 5 |
| 13 | 0.679 | 0.731 | 0.129 | 0.0% | 62% | 14.4 | mid RETRIEVE 40, cheap RETRIEVE 15, mid ANSWER 6 |
| 14 | 0.701 | 0.749 | 0.118 | 0.0% | 25% | 15.0 | mid RETRIEVE 42, cheap RETRIEVE 14, mid ANSWER 5 |
| 15 | 0.683 | 0.747 | 0.161 | 0.0% | 12% | 17.5 | mid RETRIEVE 53, cheap RETRIEVE 5, mid ANSWER 3 |
| 16 | 0.698 | 0.757 | 0.149 | 0.0% | 62% | 17.3 | mid RETRIEVE 56, cheap RETRIEVE 7, frontier RETRIEVE 1 |
| 17 | 0.753 | 0.811 | 0.144 | 0.0% | 75% | 17.5 | mid RETRIEVE 54, mid ANSWER 8, cheap RETRIEVE 1 |
| 18 | 0.760 | 0.821 | 0.153 | 0.0% | 62% | 20.1 | mid RETRIEVE 55, mid ANSWER 7, frontier ANSWER 1 |
| 19 | 0.644 | 0.703 | 0.147 | 0.0% | 62% | 15.8 | mid RETRIEVE 57, mid ANSWER 5, cheap ANSWER 1 |
| 20 | 0.702 | 0.760 | 0.145 | 0.0% | 62% | 16.1 | mid RETRIEVE 52, mid ANSWER 11, mid ESCALATE 1 |


| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |
|---|---|---|---|---|---|---|
| verl | 20 | 0.583 | 0.677 | 0.114 | 3.1% → 0.0% | 46% |
| TRL phase 1 (λ learned from 0) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | n/a |
