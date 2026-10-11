# NeMo-RL run `nemo_rl_dapo_1b`

Status: **ok**; container wall clock 693.4 s on NVIDIA A100-SXM4-40GB; first step logged 270.26013588905334 s after container start; versions {'nemo_rl': '0.6.0+19244a094', 'vllm': None, 'torch': '2.11.0+cu130', 'transformers': '5.8.1', 'ray': '2.55.1'}.

Fixed price λ = 0.44 in the prompt and in the reward; 4 prompts x 16 completions (after group filtering) per step; every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.

### NeMo-RL: per step

| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |
|---|---|---|---|---|---|---|---|
| 1 | 0.100 | 0.377 | 0.033 | 21.9% | 0% | 27.1 | cheap RETRIEVE 29, cheap ANSWER 18, INVALID 14 |
| 2 | 0.486 | 0.551 | 0.149 | 0.0% | 0% | 15.8 | cheap RETRIEVE 38, frontier ANSWER 14, mid RETRIEVE 7 |
| 3 | 0.647 | 0.688 | 0.091 | 0.0% | 0% | 14.5 | mid RETRIEVE 20, cheap ANSWER 14, mid ANSWER 8 |
| 4 | 0.500 | 0.662 | 0.066 | 7.8% | 0% | 16.7 | cheap RETRIEVE 37, mid RETRIEVE 17, INVALID 5 |
| 5 | 0.544 | 0.586 | 0.096 | 0.0% | 0% | 15.0 | mid RETRIEVE 28, cheap ANSWER 27, cheap RETRIEVE 4 |
| 6 | 0.609 | 0.661 | 0.119 | 0.0% | 0% | 14.4 | mid RETRIEVE 36, cheap RETRIEVE 11, mid ANSWER 9 |
| 7 | 0.568 | 0.620 | 0.118 | 0.0% | 0% | 17.0 | mid ANSWER 40, mid RETRIEVE 19, cheap ANSWER 3 |
| 8 | 0.565 | 0.626 | 0.140 | 0.0% | 0% | 16.9 | mid RETRIEVE 46, mid ANSWER 18 |
| 9 | 0.749 | 0.801 | 0.118 | 0.0% | 0% | 17.2 | mid RETRIEVE 36, mid ANSWER 15, cheap RETRIEVE 13 |
| 10 | 0.600 | 0.646 | 0.104 | 0.0% | 0% | 16.9 | mid ANSWER 42, mid RETRIEVE 15, cheap ANSWER 6 |
| 11 | 0.763 | 0.828 | 0.148 | 0.0% | 0% | 16.8 | mid RETRIEVE 44, mid ANSWER 2, cheap RETRIEVE 2 |
| 12 | 0.684 | 0.733 | 0.112 | 0.0% | 0% | 14.4 | mid RETRIEVE 26, cheap RETRIEVE 20, cheap ANSWER 1 |
| 13 | 0.631 | 0.727 | 0.217 | 0.0% | 0% | 17.2 | mid RETRIEVE 54, frontier RETRIEVE 9, mid ANSWER 1 |
| 14 | 0.592 | 0.784 | 0.436 | 0.0% | 0% | 16.3 | frontier RETRIEVE 30, mid RETRIEVE 18 |
| 15 | 0.517 | 0.597 | 0.183 | 0.0% | 0% | 16.1 | mid RETRIEVE 15, frontier RETRIEVE 1 |
| 16 | 0.759 | 0.875 | 0.265 | 0.0% | 0% | 16.2 | mid RETRIEVE 35, frontier RETRIEVE 12, cheap RETRIEVE 1 |
| 17 | 0.776 | 0.906 | 0.296 | 0.0% | 0% | 18.0 | mid RETRIEVE 22, frontier RETRIEVE 10 |
| 18 | 0.667 | 0.764 | 0.221 | 0.0% | 0% | 18.2 | mid RETRIEVE 41, frontier RETRIEVE 7 |
| 19 | 0.652 | 0.792 | 0.317 | 0.0% | 0% | 16.6 | mid RETRIEVE 39, frontier RETRIEVE 23, mid ESCALATE 2 |
| 20 | 0.475 | 0.643 | 0.380 | 0.0% | 0% | 18.2 | frontier RETRIEVE 32, mid RETRIEVE 32 |
| 21 | 0.630 | 0.723 | 0.211 | 0.0% | 0% | 14.4 | mid RETRIEVE 56, frontier RETRIEVE 7, frontier VERIFY 1 |
| 22 | 0.672 | 0.802 | 0.296 | 0.0% | 0% | 16.6 | mid RETRIEVE 44, frontier RETRIEVE 20 |
| 23 | 0.534 | 0.706 | 0.389 | 0.0% | 0% | 18.6 | frontier RETRIEVE 25, mid RETRIEVE 23 |
| 24 | 0.598 | 0.722 | 0.282 | 0.0% | 0% | 17.4 | mid RETRIEVE 46, frontier RETRIEVE 18 |
| 25 | 0.662 | 0.777 | 0.260 | 0.0% | 0% | 20.5 | mid RETRIEVE 49, frontier RETRIEVE 15 |


| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |
|---|---|---|---|---|---|---|
| NeMo-RL | 20 | 0.594 | 0.693 | 0.180 | 21.9% → 0.0% | 0% |
| TRL phase 1 (λ learned from 0) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | n/a |
