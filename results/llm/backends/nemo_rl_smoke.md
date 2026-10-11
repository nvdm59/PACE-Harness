# NeMo-RL run `nemo_rl_smoke`

Status: **ok**; container wall clock 592.1 s on NVIDIA A100-SXM4-40GB; first step logged 270.0779526233673 s after container start; versions {'nemo_rl': '0.6.0+19244a094', 'vllm': None, 'torch': '2.11.0+cu130', 'transformers': '5.8.1', 'ray': '2.55.1'}.

Fixed price λ = 0.4 in the prompt and in the reward; 8 prompts x 8 completions per step; every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.

### NeMo-RL: per step

| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |
|---|---|---|---|---|---|---|---|
| 1 | 0.326 | 0.542 | 0.037 | 15.6% | 38% | 26.0 | cheap RETRIEVE 38, cheap ANSWER 12, INVALID 10 |
| 2 | 0.592 | 0.640 | 0.120 | 0.0% | 25% | 16.6 | cheap RETRIEVE 27, mid RETRIEVE 12, mid ANSWER 9 |
| 3 | 0.420 | 0.625 | 0.087 | 12.5% | 25% | 15.4 | cheap RETRIEVE 24, mid RETRIEVE 20, cheap ANSWER 9 |
| 4 | 0.561 | 0.601 | 0.100 | 0.0% | 38% | 15.7 | mid RETRIEVE 32, cheap ANSWER 22, cheap RETRIEVE 8 |
| 5 | 0.536 | 0.586 | 0.125 | 0.0% | 38% | 15.3 | mid RETRIEVE 40, mid ANSWER 11, cheap ANSWER 9 |
| 6 | 0.715 | 0.760 | 0.113 | 0.0% | 12% | 15.9 | mid RETRIEVE 38, cheap RETRIEVE 18, mid ANSWER 7 |
| 7 | 0.596 | 0.642 | 0.114 | 0.0% | 50% | 15.3 | mid RETRIEVE 34, mid ANSWER 18, cheap ANSWER 12 |
| 8 | 0.668 | 0.730 | 0.155 | 0.0% | 38% | 15.0 | mid RETRIEVE 45, cheap ANSWER 7, cheap RETRIEVE 6 |
| 9 | 0.709 | 0.758 | 0.124 | 0.0% | 62% | 15.1 | mid RETRIEVE 47, cheap RETRIEVE 15, cheap ANSWER 1 |
| 10 | 0.645 | 0.717 | 0.181 | 0.0% | 50% | 15.7 | mid RETRIEVE 57, frontier RETRIEVE 4, mid ANSWER 3 |
| 11 | 0.677 | 0.742 | 0.162 | 0.0% | 88% | 15.4 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 12 | 0.660 | 0.739 | 0.197 | 0.0% | 88% | 15.7 | mid RETRIEVE 58, frontier RETRIEVE 6 |
| 13 | 0.711 | 0.776 | 0.162 | 0.0% | 88% | 15.3 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 14 | 0.633 | 0.695 | 0.155 | 0.0% | 100% | 15.2 | mid RETRIEVE 64 |
| 15 | 0.625 | 0.684 | 0.149 | 0.0% | 88% | 15.1 | mid RETRIEVE 61, cheap RETRIEVE 3 |
| 16 | 0.688 | 0.755 | 0.169 | 0.0% | 88% | 15.2 | mid RETRIEVE 62, frontier RETRIEVE 2 |
| 17 | 0.739 | 0.827 | 0.218 | 0.0% | 50% | 15.4 | mid RETRIEVE 55, frontier RETRIEVE 9 |
| 18 | 0.637 | 0.725 | 0.218 | 0.0% | 50% | 15.2 | mid RETRIEVE 54, frontier RETRIEVE 9, mid ESCALATE 1 |
| 19 | 0.607 | 0.698 | 0.225 | 0.0% | 75% | 15.5 | mid RETRIEVE 54, frontier RETRIEVE 10 |
| 20 | 0.660 | 0.731 | 0.176 | 0.0% | 75% | 15.3 | mid RETRIEVE 61, frontier RETRIEVE 3 |


| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |
|---|---|---|---|---|---|---|
| NeMo-RL | 20 | 0.620 | 0.699 | 0.149 | 15.6% → 0.0% | 58% |
| TRL phase 1 (λ learned from 0) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | n/a |
