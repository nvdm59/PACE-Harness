"""PACE controller as a small open LLM trained with GRPO.

Qwen3-1.7B (Qwen team 2025, arXiv:2505.09388) with a LoRA adapter (Hu et al. 2021,
arXiv:2106.09685), trained with TRL's GRPOTrainer (GRPO: Shao et al. 2024, arXiv:2402.03300)
on the same priced, dual-ascent reward as the linear controller.

The frozen harness tiers are the arms; the LLM only picks the first action
`<action>TIER OP</action>`; a fixed continuation policy finishes the episode;
the reward is priced (quality - lambda * cost, lambda learned by dual ascent).

env_text.py    prompt rendering, strict action parser, invalid-action fallback
rewards.py     SimEpisodeReward, LookupReward (CRMArena counterfactual logs),
               DualAscentLambda, and the TRL reward-function adapter
train_grpo_modal.py / eval_modal.py   Modal apps (GPU); not imported by tests
"""
