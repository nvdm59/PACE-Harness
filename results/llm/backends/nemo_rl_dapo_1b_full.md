# NeMo-RL run `nemo_rl_dapo_1b_full`

Status: **killed: container time limit**; container wall clock 2657.9 s on NVIDIA A100-SXM4-40GB; first step logged 260.26646900177 s after container start; versions {'nemo_rl': '0.6.0+19244a094', 'vllm': None, 'torch': '2.11.0+cu130', 'transformers': '5.8.1', 'ray': '2.55.1'}.

Fixed price λ = 0.44 in the prompt and in the reward; 4 prompts x 16 completions (after group filtering) per step; every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.

### NeMo-RL: per step

| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |
|---|---|---|---|---|---|---|---|
| 1 | 0.100 | 0.377 | 0.033 | 21.9% | 0% | 27.3 | cheap RETRIEVE 29, cheap ANSWER 18, INVALID 14 |
| 2 | 0.486 | 0.557 | 0.161 | 0.0% | 0% | 16.4 | cheap RETRIEVE 37, frontier ANSWER 10, mid ANSWER 6 |
| 3 | 0.683 | 0.718 | 0.080 | 0.0% | 25% | 15.1 | mid RETRIEVE 22, cheap RETRIEVE 21, cheap ANSWER 14 |
| 4 | 0.504 | 0.668 | 0.070 | 7.8% | 0% | 17.3 | cheap RETRIEVE 38, mid RETRIEVE 19, INVALID 5 |
| 5 | 0.550 | 0.606 | 0.128 | 0.0% | 0% | 15.2 | mid RETRIEVE 27, cheap ANSWER 26, frontier RETRIEVE 5 |
| 6 | 0.535 | 0.572 | 0.084 | 0.0% | 0% | 17.5 | mid ANSWER 24, mid RETRIEVE 15, cheap ANSWER 15 |
| 7 | 0.611 | 0.665 | 0.124 | 0.0% | 50% | 17.1 | mid RETRIEVE 39, mid ANSWER 17, cheap ANSWER 7 |
| 8 | 0.775 | 0.835 | 0.137 | 0.0% | 0% | 17.4 | mid RETRIEVE 47, mid ANSWER 14, cheap ANSWER 2 |
| 9 | 0.603 | 0.650 | 0.108 | 0.0% | 25% | 17.3 | mid ANSWER 35, mid RETRIEVE 21, cheap ANSWER 8 |
| 10 | 0.800 | 0.867 | 0.151 | 0.0% | 75% | 17.1 | mid RETRIEVE 62, cheap RETRIEVE 2 |
| 11 | 0.725 | 0.810 | 0.194 | 0.0% | 0% | 17.5 | mid RETRIEVE 52, frontier RETRIEVE 7, cheap ANSWER 3 |
| 12 | 0.595 | 0.686 | 0.208 | 0.0% | 0% | 17.0 | mid RETRIEVE 51, frontier RETRIEVE 8, mid ANSWER 4 |
| 13 | 0.696 | 0.764 | 0.155 | 0.0% | 100% | 17.3 | mid RETRIEVE 64 |
| 14 | 0.670 | 0.784 | 0.260 | 0.0% | 25% | 17.1 | mid RETRIEVE 49, frontier RETRIEVE 15 |
| 15 | 0.651 | 0.741 | 0.204 | 0.0% | 50% | 19.2 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 16 | 0.584 | 0.675 | 0.149 | 1.6% | 50% | 17.3 | mid RETRIEVE 61, cheap RETRIEVE 2, INVALID 1 |
| 17 | 0.682 | 0.787 | 0.238 | 0.0% | 25% | 21.5 | mid RETRIEVE 39, mid ESCALATE 13, frontier RETRIEVE 12 |
| 18 | 0.471 | 0.614 | 0.324 | 0.0% | 25% | 19.9 | mid RETRIEVE 40, frontier RETRIEVE 24 |
| 19 | 0.774 | 0.873 | 0.225 | 0.0% | 25% | 17.8 | mid RETRIEVE 54, frontier RETRIEVE 10 |
| 20 | 0.629 | 0.805 | 0.357 | 1.6% | 25% | 20.0 | mid RETRIEVE 34, frontier RETRIEVE 29, INVALID 1 |
| 21 | 0.794 | 0.862 | 0.155 | 0.0% | 100% | 16.9 | mid RETRIEVE 64 |
| 22 | 0.465 | 0.589 | 0.281 | 0.0% | 0% | 19.2 | mid RETRIEVE 46, frontier RETRIEVE 15, frontier VERIFY 3 |
| 23 | 0.543 | 0.643 | 0.225 | 0.0% | 50% | 19.5 | mid RETRIEVE 54, frontier RETRIEVE 10 |
| 24 | 0.635 | 0.750 | 0.202 | 1.6% | 50% | 17.5 | mid RETRIEVE 56, frontier RETRIEVE 5, frontier ESCALATE 2 |
| 25 | 0.688 | 0.806 | 0.268 | 0.0% | 50% | 20.8 | mid RETRIEVE 48, frontier RETRIEVE 16 |
| 26 | 0.708 | 0.789 | 0.183 | 0.0% | 50% | 17.0 | mid RETRIEVE 54, mid VERIFY 6, frontier RETRIEVE 4 |
| 27 | 0.526 | 0.631 | 0.239 | 0.0% | 25% | 17.2 | mid RETRIEVE 52, frontier RETRIEVE 12 |
| 28 | 0.734 | 0.821 | 0.197 | 0.0% | 50% | 17.3 | mid RETRIEVE 58, frontier RETRIEVE 6 |
| 29 | 0.677 | 0.788 | 0.253 | 0.0% | 25% | 17.4 | mid RETRIEVE 50, frontier RETRIEVE 14 |
| 30 | 0.688 | 0.797 | 0.246 | 0.0% | 50% | 19.3 | mid RETRIEVE 51, frontier RETRIEVE 13 |
| 31 | 0.844 | 0.915 | 0.162 | 0.0% | 75% | 19.4 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 32 | 0.846 | 0.920 | 0.169 | 0.0% | 50% | 19.6 | mid RETRIEVE 62, frontier RETRIEVE 2 |
| 33 | 0.694 | 0.762 | 0.155 | 0.0% | 100% | 24.0 | mid RETRIEVE 64 |
| 34 | 0.546 | 0.649 | 0.232 | 0.0% | 25% | 18.2 | mid RETRIEVE 52, frontier RETRIEVE 11, mid ESCALATE 1 |
| 35 | 0.733 | 0.823 | 0.204 | 0.0% | 75% | 26.0 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 36 | 0.667 | 0.742 | 0.169 | 0.0% | 50% | 17.2 | mid RETRIEVE 61, frontier RETRIEVE 2, mid ESCALATE 1 |
| 37 | 0.806 | 0.877 | 0.162 | 0.0% | 75% | 22.9 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 38 | 0.726 | 0.804 | 0.176 | 0.0% | 50% | 19.1 | mid RETRIEVE 61, frontier RETRIEVE 3 |
| 39 | 0.619 | 0.690 | 0.162 | 0.0% | 75% | 17.4 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 40 | 0.809 | 0.887 | 0.176 | 0.0% | 75% | 17.3 | mid RETRIEVE 61, frontier RETRIEVE 3 |
| 41 | 0.805 | 0.877 | 0.162 | 0.0% | 75% | 16.9 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 42 | 0.800 | 0.880 | 0.183 | 0.0% | 50% | 17.4 | mid RETRIEVE 60, frontier RETRIEVE 4 |
| 43 | 0.679 | 0.751 | 0.162 | 0.0% | 75% | 22.0 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 44 | 0.544 | 0.616 | 0.162 | 0.0% | 75% | 21.5 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 45 | 0.705 | 0.774 | 0.155 | 0.0% | 75% | 23.9 | mid RETRIEVE 63, mid ESCALATE 1 |
| 46 | 0.706 | 0.777 | 0.162 | 0.0% | 75% | 19.0 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 47 | 0.737 | 0.808 | 0.162 | 0.0% | 75% | 27.7 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 48 | 0.779 | 0.847 | 0.155 | 0.0% | 100% | 16.8 | mid RETRIEVE 64 |
| 49 | 0.755 | 0.826 | 0.162 | 0.0% | 75% | 16.8 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 50 | 0.686 | 0.767 | 0.183 | 0.0% | 75% | 29.9 | mid RETRIEVE 60, frontier RETRIEVE 4 |
| 51 | 0.625 | 0.696 | 0.162 | 0.0% | 75% | 19.0 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 52 | 0.590 | 0.658 | 0.155 | 0.0% | 100% | 19.8 | mid RETRIEVE 64 |
| 53 | 0.744 | 0.812 | 0.155 | 0.0% | 100% | 19.5 | mid RETRIEVE 64 |
| 54 | 0.690 | 0.774 | 0.190 | 0.0% | 75% | 24.3 | mid RETRIEVE 59, frontier RETRIEVE 5 |
| 55 | 0.718 | 0.786 | 0.155 | 0.0% | 100% | 19.8 | mid RETRIEVE 64 |
| 56 | 0.705 | 0.780 | 0.169 | 0.0% | 75% | 19.5 | mid RETRIEVE 62, frontier RETRIEVE 2 |
| 57 | 0.567 | 0.638 | 0.162 | 0.0% | 75% | 19.8 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 58 | 0.641 | 0.712 | 0.162 | 0.0% | 75% | 25.5 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 59 | 0.602 | 0.673 | 0.162 | 0.0% | 75% | 23.7 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 60 | 0.624 | 0.698 | 0.169 | 0.0% | 50% | 18.9 | mid RETRIEVE 62, frontier RETRIEVE 2 |
| 61 | 0.741 | 0.812 | 0.162 | 0.0% | 75% | 24.5 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 62 | 0.487 | 0.614 | 0.289 | 0.0% | 0% | 17.0 | mid RETRIEVE 45, frontier RETRIEVE 19 |
| 63 | 0.592 | 0.675 | 0.190 | 0.0% | 50% | 16.9 | mid RETRIEVE 58, frontier RETRIEVE 5, mid VERIFY 1 |
| 64 | 0.693 | 0.858 | 0.373 | 0.0% | 25% | 17.7 | mid RETRIEVE 33, frontier RETRIEVE 31 |
| 65 | 0.647 | 0.802 | 0.352 | 0.0% | 0% | 17.5 | mid RETRIEVE 36, frontier RETRIEVE 28 |
| 66 | 0.695 | 0.816 | 0.275 | 0.0% | 50% | 19.4 | mid RETRIEVE 47, frontier RETRIEVE 17 |
| 67 | 0.732 | 0.843 | 0.253 | 0.0% | 50% | 17.3 | mid RETRIEVE 50, frontier RETRIEVE 14 |
| 68 | 0.670 | 0.825 | 0.352 | 0.0% | 0% | 17.2 | mid RETRIEVE 36, frontier RETRIEVE 28 |
| 69 | 0.650 | 0.832 | 0.415 | 0.0% | 0% | 15.0 | frontier RETRIEVE 37, mid RETRIEVE 27 |
| 70 | 0.690 | 0.810 | 0.274 | 0.0% | 50% | 17.8 | mid RETRIEVE 47, frontier RETRIEVE 10, frontier ESCALATE 7 |
| 71 | 0.770 | 0.860 | 0.204 | 0.0% | 0% | 17.5 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 72 | 0.657 | 0.828 | 0.387 | 0.0% | 75% | 19.5 | frontier RETRIEVE 33, mid RETRIEVE 31 |
| 73 | 0.631 | 0.817 | 0.421 | 0.0% | 50% | 17.4 | frontier RETRIEVE 29, mid RETRIEVE 26, frontier VERIFY 9 |
| 74 | 0.616 | 0.746 | 0.295 | 0.0% | 25% | 15.2 | mid RETRIEVE 44, frontier RETRIEVE 13, frontier VERIFY 7 |
| 75 | 0.779 | 0.850 | 0.162 | 0.0% | 75% | 17.4 | mid RETRIEVE 63, frontier RETRIEVE 1 |
| 76 | 0.709 | 0.836 | 0.289 | 0.0% | 0% | 17.6 | mid RETRIEVE 45, frontier RETRIEVE 18, frontier VERIFY 1 |
| 77 | 0.618 | 0.831 | 0.483 | 0.0% | 25% | 19.6 | frontier VERIFY 26, frontier RETRIEVE 21, mid RETRIEVE 17 |
| 78 | 0.730 | 0.897 | 0.380 | 0.0% | 50% | 17.4 | mid RETRIEVE 32, frontier RETRIEVE 30, frontier VERIFY 2 |
| 79 | 0.595 | 0.788 | 0.354 | 3.1% | 25% | 17.3 | frontier VERIFY 27, mid RETRIEVE 25, mid ANSWER 7 |
| 80 | 0.540 | 0.767 | 0.474 | 1.6% | 25% | 17.4 | frontier VERIFY 26, frontier RETRIEVE 20, mid RETRIEVE 17 |
| 81 | 0.668 | 0.848 | 0.348 | 1.6% | 25% | 14.9 | frontier VERIFY 24, mid RETRIEVE 22, frontier ANSWER 8 |
| 82 | 0.762 | 0.876 | 0.259 | 0.0% | 50% | 17.1 | mid RETRIEVE 36, frontier VERIFY 16, mid ANSWER 8 |
| 83 | 0.623 | 0.803 | 0.368 | 1.6% | 50% | 17.5 | mid RETRIEVE 31, frontier VERIFY 31, mid VERIFY 1 |
| 84 | 0.768 | 0.888 | 0.273 | 0.0% | 75% | 19.2 | mid RETRIEVE 47, frontier VERIFY 16, frontier RETRIEVE 1 |
| 85 | 0.690 | 0.847 | 0.357 | 0.0% | 25% | 19.8 | frontier VERIFY 31, mid ANSWER 20, mid RETRIEVE 12 |
| 86 | 0.541 | 0.752 | 0.478 | 0.0% | 75% | 19.4 | frontier VERIFY 48, mid ANSWER 12, mid RETRIEVE 4 |
| 87 | 0.666 | 0.878 | 0.482 | 0.0% | 50% | 19.3 | frontier VERIFY 44, mid RETRIEVE 16, frontier RETRIEVE 3 |
| 88 | 0.642 | 0.758 | 0.262 | 0.0% | 50% | 19.5 | mid RETRIEVE 47, frontier VERIFY 15, frontier ANSWER 1 |
| 89 | 0.746 | 0.863 | 0.266 | 0.0% | 100% | 28.2 | mid RETRIEVE 48, frontier VERIFY 16 |
| 90 | 0.665 | 0.880 | 0.489 | 0.0% | 100% | 19.1 | frontier VERIFY 48, mid RETRIEVE 16 |
| 91 | 0.774 | 0.842 | 0.155 | 0.0% | 100% | 25.0 | mid RETRIEVE 64 |
| 92 | 0.748 | 0.865 | 0.266 | 0.0% | 75% | 21.7 | mid RETRIEVE 48, frontier VERIFY 14, frontier RETRIEVE 2 |
| 93 | 0.551 | 0.761 | 0.351 | 4.7% | 50% | 19.1 | mid RETRIEVE 32, frontier VERIFY 20, frontier RETRIEVE 9 |
| 94 | 0.783 | 0.900 | 0.266 | 0.0% | 100% | 19.1 | mid RETRIEVE 48, frontier VERIFY 16 |
| 95 | 0.752 | 0.868 | 0.263 | 0.0% | 75% | 19.8 | mid RETRIEVE 48, frontier VERIFY 15, frontier ANSWER 1 |
| 96 | 0.737 | 0.903 | 0.378 | 0.0% | 100% | 18.8 | frontier VERIFY 32, mid RETRIEVE 32 |
| 97 | 0.734 | 0.852 | 0.266 | 0.0% | 100% | 19.5 | mid RETRIEVE 48, frontier VERIFY 16 |
| 98 | 0.649 | 0.815 | 0.378 | 0.0% | 75% | 21.8 | mid RETRIEVE 32, frontier VERIFY 31, frontier RETRIEVE 1 |
| 99 | 0.551 | 0.668 | 0.266 | 0.0% | 100% | 18.9 | mid RETRIEVE 48, frontier VERIFY 16 |
| 100 | 0.675 | 0.841 | 0.378 | 0.0% | 100% | 25.8 | frontier VERIFY 32, mid RETRIEVE 32 |
| 101 | 0.700 | 0.866 | 0.378 | 0.0% | 100% | 25.9 | mid RETRIEVE 32, frontier VERIFY 32 |
| 102 | 0.770 | 0.884 | 0.259 | 0.0% | 75% | 17.0 | mid RETRIEVE 48, frontier VERIFY 15, mid VERIFY 1 |
| 103 | 0.683 | 0.849 | 0.378 | 0.0% | 100% | 20.0 | frontier VERIFY 32, mid RETRIEVE 32 |
| 104 | 0.620 | 0.786 | 0.378 | 0.0% | 100% | 22.5 | frontier VERIFY 32, mid RETRIEVE 32 |
| 105 | 0.705 | 0.871 | 0.378 | 0.0% | 100% | 23.5 | frontier VERIFY 32, mid RETRIEVE 32 |
| 106 | 0.722 | 0.836 | 0.259 | 0.0% | 75% | 19.3 | mid RETRIEVE 49, frontier VERIFY 15 |
| 107 | 0.562 | 0.728 | 0.378 | 0.0% | 100% | 19.2 | frontier VERIFY 32, mid RETRIEVE 32 |
| 108 | 0.785 | 0.853 | 0.155 | 0.0% | 100% | 17.1 | mid RETRIEVE 64 |
| 109 | 0.622 | 0.859 | 0.480 | 1.6% | 75% | 28.9 | frontier VERIFY 47, mid RETRIEVE 16, INVALID 1 |
| 110 | 0.620 | 0.860 | 0.487 | 1.6% | 75% | 22.9 | frontier VERIFY 48, mid RETRIEVE 15, INVALID 1 |
| 111 | 0.532 | 0.698 | 0.378 | 0.0% | 100% | 19.0 | frontier VERIFY 32, mid RETRIEVE 32 |
| 112 | 0.804 | 0.872 | 0.155 | 0.0% | 100% | 24.1 | mid RETRIEVE 64 |
| 113 | 0.556 | 0.771 | 0.489 | 0.0% | 100% | 22.7 | frontier VERIFY 48, mid RETRIEVE 16 |
| 114 | 0.734 | 0.802 | 0.155 | 0.0% | 100% | 25.4 | mid RETRIEVE 64 |
| 115 | 0.807 | 0.925 | 0.266 | 0.0% | 100% | 17.1 | mid RETRIEVE 48, frontier VERIFY 16 |
| 116 | 0.743 | 0.860 | 0.266 | 0.0% | 100% | 22.2 | mid RETRIEVE 48, frontier VERIFY 16 |
| 117 | 0.788 | 0.856 | 0.155 | 0.0% | 100% | 21.5 | mid RETRIEVE 64 |
| 118 | 0.648 | 0.863 | 0.489 | 0.0% | 100% | 22.8 | frontier VERIFY 48, mid RETRIEVE 16 |
| 119 | 0.684 | 0.850 | 0.378 | 0.0% | 100% | 26.1 | frontier VERIFY 32, mid RETRIEVE 32 |
| 120 | 0.781 | 0.849 | 0.155 | 0.0% | 100% | 30.8 | mid RETRIEVE 64 |


| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |
|---|---|---|---|---|---|---|
| NeMo-RL | 20 | 0.606 | 0.703 | 0.170 | 21.9% → 1.6% | 25% |
| TRL phase 1 (λ learned from 0) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | n/a |
