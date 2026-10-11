"""The price-conditioned controller task as entry points for other public RL stacks.

The task, prompts and reward are the ones `pace.llm.train_grpo_modal` trains with TRL;
only the trainer changes, so the stacks can be compared on identical data and reward.

common.py        fixed-price task set (phase-1c seed / prompts), completion cleanup,
                 re-scoring of logged rollouts with SimEpisodeReward, step summaries
verl_grpo.py     verl (volcengine/verl) dataset builder, custom reward function, Modal launcher
nemo_rl_grpo.py  NeMo-RL (NVIDIA-NeMo/RL) environment, dataset, Modal launcher
summarize.py     merge platform logs into results/llm/backends/*_smoke.jsonl + markdown tables

Nothing here imports modal, ray, verl, nemo_rl, torch or pandas at module import time.
"""
