# verl run `verl_dapo_1b`

Status: **killed: container time limit**; container wall clock 2669.5 s on NVIDIA A100-SXM4-40GB; first step logged 363.32226300239563 s after container start; versions {'verl': '0.9.1', 'vllm': '0.24.0', 'torch': '2.11.0+cu130', 'transformers': '5.5.3', 'ray': '2.56.1', 'peft': '0.19.1'}.

Fixed price λ = 0.44 in the prompt and in the reward; 4 prompts x 16 completions (after group filtering) per step; every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.

### verl: per step

| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |
|---|---|---|---|---|---|---|---|
| 1 | 0.383 | 0.480 | 0.094 | 4.7% | 0% | 88.3 | cheap RETRIEVE 26, mid RETRIEVE 19, cheap ANSWER 13 |
| 2 | 0.622 | 0.692 | 0.099 | 1.6% | 0% | 13.4 | cheap RETRIEVE 23, mid RETRIEVE 22, cheap ANSWER 15 |
| 3 | 0.253 | 0.681 | 0.126 | 21.9% | 0% | 13.5 | cheap RETRIEVE 29, INVALID 14, mid RETRIEVE 12 |
| 4 | 0.419 | 0.652 | 0.045 | 12.5% | 0% | 20.1 | cheap RETRIEVE 34, cheap ANSWER 14, INVALID 8 |
| 5 | 0.560 | 0.589 | 0.066 | 0.0% | 0% | 16.6 | cheap RETRIEVE 43, mid RETRIEVE 17, cheap ANSWER 4 |
| 6 | 0.478 | 0.574 | 0.091 | 4.7% | 0% | 15.5 | cheap ANSWER 27, cheap RETRIEVE 18, mid RETRIEVE 8 |
| 7 | 0.544 | 0.572 | 0.064 | 0.0% | 0% | 16.4 | cheap RETRIEVE 31, mid RETRIEVE 15, cheap ANSWER 14 |
| 8 | 0.620 | 0.671 | 0.116 | 0.0% | 0% | 11.3 | mid RETRIEVE 36, cheap RETRIEVE 23, mid ANSWER 3 |
| 9 | 0.584 | 0.659 | 0.169 | 0.0% | 0% | 14.0 | mid RETRIEVE 28, cheap ANSWER 16, frontier RETRIEVE 7 |
| 10 | 0.513 | 0.577 | 0.086 | 1.6% | 0% | 13.7 | cheap RETRIEVE 32, mid RETRIEVE 27, cheap ANSWER 3 |
| 11 | 0.764 | 0.852 | 0.200 | 0.0% | 0% | 13.6 | mid RETRIEVE 31, cheap RETRIEVE 20, frontier RETRIEVE 12 |
| 12 | 0.659 | 0.757 | 0.222 | 0.0% | 0% | 13.9 | mid RETRIEVE 25, cheap RETRIEVE 20, frontier RETRIEVE 10 |
| 13 | 0.546 | 0.619 | 0.166 | 0.0% | 0% | 16.3 | mid RETRIEVE 36, cheap RETRIEVE 12, frontier ANSWER 10 |
| 14 | 0.752 | 0.800 | 0.110 | 0.0% | 0% | 13.6 | mid RETRIEVE 33, cheap RETRIEVE 24, mid ESCALATE 3 |
| 15 | 0.683 | 0.731 | 0.109 | 0.0% | 0% | 12.5 | mid RETRIEVE 32, cheap RETRIEVE 21, mid ANSWER 6 |
| 16 | 0.765 | 0.820 | 0.124 | 0.0% | 0% | 14.9 | mid RETRIEVE 45, cheap RETRIEVE 14, mid ANSWER 5 |
| 17 | 0.790 | 0.855 | 0.147 | 0.0% | 0% | 13.4 | mid RETRIEVE 50, cheap RETRIEVE 12, frontier RETRIEVE 2 |
| 18 | 0.785 | 0.841 | 0.128 | 0.0% | 0% | 18.8 | mid RETRIEVE 42, mid ANSWER 15, cheap ANSWER 4 |
| 19 | 0.652 | 0.720 | 0.155 | 0.0% | 0% | 19.8 | mid RETRIEVE 49, mid ANSWER 10, cheap RETRIEVE 3 |
| 20 | 0.602 | 0.699 | 0.177 | 1.6% | 0% | 15.6 | mid RETRIEVE 55, frontier RETRIEVE 4, mid ANSWER 2 |
| 21 | 0.735 | 0.797 | 0.142 | 0.0% | 0% | 18.5 | mid RETRIEVE 43, mid ANSWER 14, mid ESCALATE 7 |
| 22 | 0.716 | 0.793 | 0.175 | 0.0% | 0% | 18.7 | mid RETRIEVE 60, frontier RETRIEVE 2, frontier ESCALATE 1 |
| 23 | 0.717 | 0.817 | 0.226 | 0.0% | 0% | 17.8 | mid RETRIEVE 47, frontier RETRIEVE 11, mid ANSWER 3 |
| 24 | 0.677 | 0.797 | 0.271 | 0.0% | 0% | 16.0 | mid RETRIEVE 47, frontier RETRIEVE 14, frontier VERIFY 2 |
| 25 | 0.687 | 0.789 | 0.232 | 0.0% | 0% | 17.6 | mid RETRIEVE 53, frontier RETRIEVE 11 |
| 26 | 0.700 | 0.782 | 0.188 | 0.0% | 0% | 17.5 | mid RETRIEVE 58, frontier RETRIEVE 5, cheap RETRIEVE 1 |
| 27 | 0.509 | 0.595 | 0.195 | 0.0% | 0% | 22.9 | mid RETRIEVE 56, frontier RETRIEVE 6, mid ANSWER 2 |
| 28 | 0.703 | 0.782 | 0.180 | 0.0% | 0% | 15.5 | mid RETRIEVE 60, frontier RETRIEVE 2, frontier ANSWER 1 |
| 29 | 0.639 | 0.717 | 0.178 | 0.0% | 0% | 14.0 | mid RETRIEVE 45, mid ANSWER 14, frontier RETRIEVE 5 |
| 30 | 0.564 | 0.656 | 0.211 | 0.0% | 0% | 17.6 | mid RETRIEVE 56, frontier RETRIEVE 8 |
| 31 | 0.497 | 0.604 | 0.202 | 1.6% | 0% | 15.5 | mid RETRIEVE 56, frontier RETRIEVE 7, INVALID 1 |
| 32 | 0.598 | 0.697 | 0.225 | 0.0% | 0% | 18.3 | mid RETRIEVE 53, frontier RETRIEVE 10, mid ESCALATE 1 |
| 33 | 0.624 | 0.705 | 0.183 | 0.0% | 0% | 17.7 | mid RETRIEVE 59, frontier RETRIEVE 4, mid ESCALATE 1 |
| 34 | 0.550 | 0.689 | 0.317 | 0.0% | 0% | 17.5 | mid RETRIEVE 41, frontier RETRIEVE 22, frontier VERIFY 1 |
| 35 | 0.557 | 0.659 | 0.232 | 0.0% | 0% | 18.0 | mid RETRIEVE 53, frontier RETRIEVE 11 |
| 36 | 0.599 | 0.713 | 0.260 | 0.0% | 0% | 18.1 | mid RETRIEVE 49, frontier RETRIEVE 15 |
| 37 | 0.624 | 0.717 | 0.211 | 0.0% | 0% | 15.7 | mid RETRIEVE 55, frontier RETRIEVE 8, mid ESCALATE 1 |
| 38 | 0.550 | 0.640 | 0.204 | 0.0% | 0% | 18.7 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 39 | 0.682 | 0.800 | 0.268 | 0.0% | 0% | 17.5 | mid RETRIEVE 48, frontier RETRIEVE 16 |
| 40 | 0.658 | 0.751 | 0.211 | 0.0% | 0% | 22.0 | mid RETRIEVE 56, frontier RETRIEVE 7, frontier VERIFY 1 |
| 41 | 0.530 | 0.660 | 0.295 | 0.0% | 0% | 20.6 | mid RETRIEVE 42, frontier RETRIEVE 20, mid ESCALATE 2 |
| 42 | 0.690 | 0.783 | 0.211 | 0.0% | 0% | 17.9 | mid RETRIEVE 56, frontier RETRIEVE 8 |
| 43 | 0.624 | 0.726 | 0.232 | 0.0% | 0% | 24.5 | mid RETRIEVE 53, frontier RETRIEVE 11 |
| 44 | 0.723 | 0.826 | 0.174 | 1.6% | 0% | 20.6 | mid RETRIEVE 60, frontier RETRIEVE 3, INVALID 1 |
| 45 | 0.475 | 0.577 | 0.232 | 0.0% | 0% | 17.6 | mid RETRIEVE 52, frontier RETRIEVE 8, frontier VERIFY 3 |
| 46 | 0.558 | 0.664 | 0.181 | 1.6% | 0% | 23.2 | mid RETRIEVE 58, frontier RETRIEVE 4, INVALID 1 |
| 47 | 0.568 | 0.655 | 0.197 | 0.0% | 0% | 19.8 | mid RETRIEVE 58, frontier RETRIEVE 5, frontier VERIFY 1 |
| 48 | 0.753 | 0.855 | 0.174 | 1.6% | 0% | 19.9 | mid RETRIEVE 60, frontier RETRIEVE 3, INVALID 1 |
| 49 | 0.831 | 0.917 | 0.197 | 0.0% | 0% | 24.5 | mid RETRIEVE 58, frontier RETRIEVE 6 |
| 50 | 0.535 | 0.636 | 0.188 | 1.6% | 0% | 39.5 | mid RETRIEVE 58, frontier RETRIEVE 5, INVALID 1 |
| 51 | 0.718 | 0.808 | 0.204 | 0.0% | 0% | 23.5 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 52 | 0.630 | 0.717 | 0.197 | 0.0% | 0% | 28.1 | mid RETRIEVE 58, frontier RETRIEVE 6 |
| 53 | 0.659 | 0.743 | 0.190 | 0.0% | 0% | 21.9 | mid RETRIEVE 59, frontier RETRIEVE 5 |
| 54 | 0.612 | 0.702 | 0.204 | 0.0% | 0% | 26.0 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 55 | 0.583 | 0.686 | 0.188 | 1.6% | 0% | 22.5 | mid RETRIEVE 58, frontier RETRIEVE 5, INVALID 1 |
| 56 | 0.480 | 0.563 | 0.190 | 0.0% | 0% | 21.8 | mid RETRIEVE 59, frontier RETRIEVE 5 |
| 57 | 0.657 | 0.767 | 0.188 | 1.6% | 0% | 37.9 | mid RETRIEVE 58, frontier RETRIEVE 5, INVALID 1 |
| 58 | 0.637 | 0.727 | 0.204 | 0.0% | 0% | 22.5 | mid RETRIEVE 57, frontier RETRIEVE 7 |
| 59 | 0.572 | 0.671 | 0.225 | 0.0% | 0% | 21.8 | mid RETRIEVE 54, frontier RETRIEVE 10 |
| 60 | 0.534 | 0.621 | 0.197 | 0.0% | 0% | 19.6 | mid RETRIEVE 57, frontier RETRIEVE 5, mid VERIFY 1 |
| 61 | 0.445 | 0.592 | 0.293 | 1.6% | 0% | 13.9 | mid RETRIEVE 43, frontier RETRIEVE 11, frontier VERIFY 9 |
| 62 | 0.703 | 0.793 | 0.204 | 0.0% | 0% | 19.5 | mid RETRIEVE 57, frontier RETRIEVE 6, frontier VERIFY 1 |
| 63 | 0.603 | 0.733 | 0.296 | 0.0% | 0% | 18.1 | mid RETRIEVE 44, frontier RETRIEVE 20 |
| 64 | 0.626 | 0.740 | 0.260 | 0.0% | 0% | 18.6 | mid RETRIEVE 46, frontier RETRIEVE 15, mid ESCALATE 3 |
| 65 | 0.655 | 0.763 | 0.246 | 0.0% | 0% | 15.5 | mid RETRIEVE 51, frontier RETRIEVE 13 |
| 66 | 0.615 | 0.773 | 0.359 | 0.0% | 0% | 18.3 | mid RETRIEVE 35, frontier RETRIEVE 29 |
| 67 | 0.536 | 0.638 | 0.232 | 0.0% | 0% | 17.5 | mid RETRIEVE 53, frontier RETRIEVE 11 |
| 68 | 0.615 | 0.723 | 0.246 | 0.0% | 0% | 16.3 | mid RETRIEVE 51, frontier RETRIEVE 13 |
| 69 | 0.642 | 0.819 | 0.401 | 0.0% | 0% | 16.3 | frontier RETRIEVE 35, mid RETRIEVE 28, mid VERIFY 1 |
| 70 | 0.605 | 0.735 | 0.296 | 0.0% | 0% | 13.5 | mid RETRIEVE 44, frontier RETRIEVE 20 |
| 71 | 0.723 | 0.828 | 0.239 | 0.0% | 0% | 13.4 | mid RETRIEVE 52, frontier RETRIEVE 10, frontier VERIFY 2 |
| 72 | 0.508 | 0.647 | 0.316 | 0.0% | 0% | 16.5 | mid RETRIEVE 41, frontier RETRIEVE 19, frontier VERIFY 4 |
| 73 | 0.683 | 0.877 | 0.440 | 0.0% | 0% | 13.4 | frontier RETRIEVE 40, mid RETRIEVE 23, frontier ANSWER 1 |
| 74 | 0.562 | 0.751 | 0.429 | 0.0% | 0% | 13.5 | frontier RETRIEVE 39, mid RETRIEVE 25 |
| 75 | 0.478 | 0.670 | 0.436 | 0.0% | 0% | 14.3 | frontier RETRIEVE 39, mid RETRIEVE 24, frontier VERIFY 1 |
| 76 | 0.598 | 0.769 | 0.387 | 0.0% | 0% | 16.9 | frontier RETRIEVE 33, mid RETRIEVE 31 |
| 77 | 0.618 | 0.764 | 0.331 | 0.0% | 0% | 14.5 | mid RETRIEVE 39, frontier RETRIEVE 24, frontier VERIFY 1 |
| 78 | 0.631 | 0.827 | 0.445 | 0.0% | 0% | 14.6 | frontier RETRIEVE 36, mid RETRIEVE 20, frontier ANSWER 6 |
| 79 | 0.627 | 0.816 | 0.429 | 0.0% | 0% | 15.5 | frontier RETRIEVE 39, mid RETRIEVE 25 |
| 80 | 0.688 | 0.858 | 0.387 | 0.0% | 0% | 16.2 | frontier RETRIEVE 33, mid RETRIEVE 31 |
| 81 | 0.680 | 0.847 | 0.379 | 0.0% | 0% | 11.5 | mid RETRIEVE 32, frontier RETRIEVE 21, frontier VERIFY 11 |
| 82 | 0.780 | 0.901 | 0.275 | 0.0% | 0% | 15.7 | mid RETRIEVE 47, frontier RETRIEVE 17 |
| 83 | 0.689 | 0.881 | 0.436 | 0.0% | 0% | 16.4 | frontier RETRIEVE 39, mid RETRIEVE 24, frontier VERIFY 1 |
| 84 | 0.596 | 0.856 | 0.590 | 0.0% | 0% | 15.7 | frontier RETRIEVE 53, frontier VERIFY 9, mid RETRIEVE 2 |
| 85 | 0.562 | 0.788 | 0.513 | 0.0% | 0% | 13.6 | frontier RETRIEVE 38, frontier VERIFY 13, mid RETRIEVE 13 |
| 86 | 0.628 | 0.795 | 0.379 | 0.0% | 0% | 16.5 | mid RETRIEVE 32, frontier RETRIEVE 17, frontier VERIFY 15 |
| 87 | 0.715 | 0.900 | 0.421 | 0.0% | 0% | 16.1 | mid RETRIEVE 26, frontier RETRIEVE 21, frontier VERIFY 13 |
| 88 | 0.710 | 0.834 | 0.282 | 0.0% | 0% | 15.9 | mid RETRIEVE 46, frontier RETRIEVE 18 |
| 89 | 0.638 | 0.808 | 0.387 | 0.0% | 0% | 14.3 | mid RETRIEVE 31, frontier RETRIEVE 30, frontier VERIFY 3 |
| 90 | 0.524 | 0.788 | 0.599 | 0.0% | 0% | 17.5 | frontier VERIFY 33, frontier RETRIEVE 30, frontier ANSWER 1 |
| 91 | 0.634 | 0.863 | 0.520 | 0.0% | 0% | 15.3 | frontier RETRIEVE 32, frontier VERIFY 18, mid RETRIEVE 8 |
| 92 | 0.681 | 0.838 | 0.357 | 0.0% | 0% | 16.8 | mid RETRIEVE 26, frontier RETRIEVE 19, frontier VERIFY 9 |
| 93 | 0.570 | 0.836 | 0.603 | 0.0% | 0% | 15.7 | frontier RETRIEVE 39, frontier VERIFY 25 |
| 94 | 0.753 | 0.870 | 0.265 | 0.0% | 0% | 13.9 | mid RETRIEVE 44, frontier VERIFY 14, frontier RETRIEVE 2 |
| 95 | 0.580 | 0.794 | 0.487 | 0.0% | 0% | 16.7 | frontier VERIFY 42, mid RETRIEVE 14, frontier RETRIEVE 4 |
| 96 | 0.482 | 0.762 | 0.593 | 1.6% | 0% | 24.0 | frontier VERIFY 35, frontier RETRIEVE 26, frontier ESCALATE 2 |
| 97 | 0.620 | 0.828 | 0.473 | 0.0% | 0% | 39.3 | frontier RETRIEVE 42, mid VERIFY 11, mid RETRIEVE 6 |
| 98 | 0.484 | 0.828 | 0.422 | 9.4% | 0% | 16.4 | frontier VERIFY 30, mid ESCALATE 14, frontier RETRIEVE 10 |
| 99 | 0.552 | 0.776 | 0.340 | 6.2% | 0% | 15.7 | mid RETRIEVE 29, frontier VERIFY 28, INVALID 4 |
| 100 | 0.456 | 0.747 | 0.580 | 3.1% | 0% | 33.2 | frontier VERIFY 51, frontier RETRIEVE 10, INVALID 2 |
| 101 | 0.580 | 0.843 | 0.599 | 0.0% | 0% | 14.1 | frontier VERIFY 40, frontier RETRIEVE 23, frontier ANSWER 1 |
| 102 | 0.571 | 0.848 | 0.586 | 1.6% | 0% | 15.9 | frontier VERIFY 45, frontier RETRIEVE 16, frontier ANSWER 2 |
| 103 | 0.689 | 0.819 | 0.253 | 1.6% | 0% | 21.3 | mid RETRIEVE 42, frontier VERIFY 15, mid ANSWER 5 |
| 104 | 0.580 | 0.810 | 0.480 | 1.6% | 0% | 15.6 | frontier VERIFY 37, mid RETRIEVE 12, frontier RETRIEVE 10 |
| 105 | 0.519 | 0.780 | 0.594 | 0.0% | 0% | 15.6 | frontier VERIFY 48, frontier RETRIEVE 15, mid VERIFY 1 |
| 106 | 0.787 | 0.855 | 0.154 | 0.0% | 0% | 18.4 | mid RETRIEVE 54, mid VERIFY 10 |
| 107 | 0.633 | 0.864 | 0.481 | 1.6% | 0% | 19.7 | frontier VERIFY 24, frontier RETRIEVE 23, mid RETRIEVE 12 |
| 108 | 0.605 | 0.797 | 0.376 | 1.6% | 0% | 17.5 | mid RETRIEVE 30, frontier VERIFY 17, frontier RETRIEVE 15 |
| 109 | 0.628 | 0.746 | 0.266 | 0.0% | 0% | 20.3 | mid RETRIEVE 44, frontier VERIFY 10, frontier RETRIEVE 6 |
| 110 | 0.627 | 0.758 | 0.257 | 1.6% | 0% | 21.9 | mid RETRIEVE 41, frontier VERIFY 15, mid VERIFY 7 |
| 111 | 0.798 | 0.866 | 0.154 | 0.0% | 0% | 31.6 | mid RETRIEVE 49, mid VERIFY 15 |
| 112 | 0.719 | 0.787 | 0.154 | 0.0% | 0% | 27.7 | mid RETRIEVE 49, mid VERIFY 15 |
| 113 | 0.673 | 0.840 | 0.379 | 0.0% | 0% | 22.2 | mid RETRIEVE 28, frontier RETRIEVE 17, frontier VERIFY 15 |
| 114 | 0.813 | 0.880 | 0.152 | 0.0% | 0% | 19.8 | mid RETRIEVE 46, mid VERIFY 16, mid ANSWER 2 |
| 115 | 0.755 | 0.848 | 0.152 | 1.6% | 0% | 18.1 | mid RETRIEVE 49, mid VERIFY 14, INVALID 1 |
| 116 | 0.618 | 0.800 | 0.369 | 1.6% | 0% | 51.6 | mid RETRIEVE 28, frontier VERIFY 18, frontier RETRIEVE 13 |
| 117 | 0.617 | 0.725 | 0.150 | 3.1% | 0% | 44.6 | mid RETRIEVE 56, mid VERIFY 6, INVALID 2 |


| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |
|---|---|---|---|---|---|---|
| verl | 20 | 0.599 | 0.692 | 0.125 | 4.7% → 1.6% | 0% |
| TRL phase 1 (λ learned from 0) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | n/a |
