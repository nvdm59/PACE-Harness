# One task, three RL stacks: TRL, verl and NeMo-RL

The price-conditioned controller task of `pace/llm/` is set up as entry points for two more public RL
stacks, so all three can be compared on **identical data and reward**:

- **Prompt.** One rendered simulator state plus a price line: `sim_prompt(..., price=True)` with the
  price filled in, under the same system message. Qwen3's thinking block is off.
- **Completion.** At most 16 tokens, parsed strictly as one `<action>TIER OP</action>`.
- **Reward.** `pace.llm.rewards.SimEpisodeReward` gives `r = quality − λ·cost`, or −1 if the completion
  is invalid.
- **Task pool.** The phase-1c pool: `sim_rows(2000, seed=0, budget=0.30, price=True)`.

In the backend smoke runs λ is **fixed at 0.4**: phase 1c's λ̄ rounded to the grid shown in the prompt.
The same λ is written into the prompt and charged in the reward. The dual-ascent λ update is not ported.
That loop sits outside the policy-gradient step, and each stack would need its own hook for it (TRL: a
callback; verl: a custom reward manager with state; NeMo-RL: environment state).

Every per-step number in `results/llm/backends/` is **re-scored from the logged completions** with
`common.step_stats`. The numbers therefore come from one code path, whatever each stack logs about itself.

| file | what |
|---|---|
| `common.py` | task pool, fixed price, completion cleanup (strips a trailing `<|im_end|>`), re-scoring, step summaries |
| `verl_grpo.py`, `verl_config.yaml` | parquet builder, verl `compute_score` reward function, Modal launcher (Hydra overrides in the YAML) |
| `nemo_rl_grpo.py`, `nemo_rl_config.yaml` | NeMo-RL `EnvironmentInterface` env, DatumSpec builder, in-container driver, Modal launcher |
| `summarize.py` | writes `results/llm/backends/<run>.jsonl`, `<run>_meta.json`, `<run>.md` |
| `trl_grpo.py` | TRL arm with hand-written dynamic sampling (phase-1b reproduction), plus the **shared evaluator** used for every stack's adapter |
| `compare.py` | the tables of `results/llm/backends/COMPARISON.md` |

## Commands

```bash
modal run pace/llm/backends/verl_grpo.py                     # verl: 20-step smoke, one A100-40GB
modal run pace/llm/backends/nemo_rl_grpo.py::versions        # NeMo-RL: build the image, no GPU needed
modal run pace/llm/backends/nemo_rl_grpo.py                  # NeMo-RL: 20-step smoke, one A100-40GB
uv run --no-project --with pytest --with pyyaml --with numpy python -m pytest -q tests/test_backends.py
# phase-1b reproduction (results/llm/backends/COMPARISON.md)
modal run pace/llm/backends/trl_grpo.py::main --run trl_dapo_1b --steps 200        # dynamic sampling + clip 0.2/0.28
modal run pace/llm/backends/verl_grpo.py::main --run verl_dapo_1b --recipe dapo_1b --steps 120 --time-cap-s 2700
modal run pace/llm/backends/nemo_rl_grpo.py::main --run nemo_rl_dapo_1b --recipe dapo_1b --steps 25 --time-cap-s 900
modal run pace/llm/backends/trl_grpo.py::eval_main --adapters "name=/vol/path/to/adapter,..."
python -m pace.llm.backends.compare
```

Inside the containers the launchers run the following. `<overrides>` is every entry under
`overrides:` in `verl_config.yaml`.

```bash
# verl
VERL_FILE_LOGGER_PATH=/vol/backends/verl/verl_smoke/metrics.jsonl python -m verl.trainer.main_ppo <overrides>
# NeMo-RL (cwd /opt/nemo-rl; the config is copied to examples/configs/pace_grpo_smoke.yaml)
python /root/src/pace/llm/backends/nemo_rl_grpo.py --config examples/configs/pace_grpo_smoke.yaml \
    --metrics /vol/backends/nemo_rl/nemo_rl_smoke/metrics.jsonl grpo.max_num_steps=20 ...
```

Both Modal functions have `timeout=1320` s. Each also runs a watchdog that kills the run when no
training step has been logged 15 minutes after the container started.

## The three stacks

| | TRL (what the phase-1 runs used) | verl 0.9.1 | NeMo-RL v0.7.0 |
|---|---|---|---|
| install footprint | pip: torch 2.14.1, transformers 5.19, trl 1.15.0, peft 0.21.2, accelerate, datasets; `debian_slim` image | `pip install "verl[vllm]==0.9.1"` pins torch 2.11, vLLM 0.24.0, transformers 5.9, plus Ray, TransferQueue and flash-attn 2.8.3 (fsdp extra). Used here: verl's own image `verlai/verl:vllm024.dev2` (12 GB) + `verl[verl-core]==0.9.1` | Python 3.13 only, uv-managed: Ray workers each get their own prebuilt venv. vLLM 0.20.0, torch 2.11. Used here: NGC image `nvcr.io/nvidia/nemo-rl:v0.7.0`, with the code in `/opt/nemo-rl` and the venvs in `/opt/ray_venvs` |
| rollout engine | HF `generate` in our runs (`use_vllm=False`); vLLM optional (`vllm_mode` colocate or server) | vLLM, SGLang or TRT-LLM as async servers driven by an agent loop | vLLM (sync or async engine), Megatron inference, or SGLang |
| parallelism | accelerate / DeepSpeed / FSDP; one process here | Ray single controller; FSDP/FSDP2, Megatron, TorchTitan or VeOmni engines; rollout TP/DP/EP | Ray; DTensor (FSDP2 + TP/SP/CP) or Megatron (TP/PP/CP/EP); colocated or dedicated generation GPUs |
| parametrization in the smoke runs | LoRA r16/α32 on q,k,v,o,gate,up,down | LoRA r16/α32 `all-linear` (`lora_rank=0` gives full fine-tuning) | LoRA r16/α32 `match_all_linear` (`lora_cfg.enabled=false` gives full fine-tuning) |
| advantage | group mean, divided by the group std (`scale_rewards="group"`) | `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=true`; also rloo, reinforce++, opo, gpg, … | `grpo` with `use_leave_one_out_baseline` (default **true**; false here) and `normalize_rewards`; also reinforce++ |
| clip bounds | `epsilon=0.2`, `epsilon_high` defaults to epsilon; optional `delta` two-sided clip | `clip_ratio_low/high=0.2/0.2`, dual-clip `clip_ratio_c=3.0` | `ratio_clip_min/max=0.2/0.2`, optional dual-clip `ratio_clip_c` |
| dynamic sampling / group filtering | none | `algorithm.filter_groups` (DAPO), honoured by the V1 sync trainer's replay buffer (`max_num_gen_batches`) | `grpo.use_dynamic_sampling` (`dynamic_sampling_max_gen_batches`) |
| KL: estimator, placement | k3, in the loss; `beta` default **0.0** (0.02 / 0.05 in our runs) | `use_kl_loss` + `kl_loss_type` (k1/k2/k3=`low_var_kl`, abs; k3 clamped to ±10), or KL in the reward (`use_kl_in_reward`, fixed or adaptive controller); default off, coefficient 0.001 when on | `reference_policy_kl_penalty` (default 0.01), `reference_policy_kl_type` k3 by default, in the loss with input/output clamps 20/10; `use_kl_in_reward` for reinforce++ |
| loss aggregation | `loss_type="dapo"` by default (token-level over the global batch); also grpo, bnpo, dr_grpo, cispo, … | `loss_agg_mode`: `token-mean` (default), `seq-mean-token-sum`, `seq-mean-token-mean`, `seq-mean-token-sum-norm` | `token_level_loss=true` (default) or sequence-level |
| overlong handling | `mask_truncated_completions` (default off) | DAPO reward manager with `overlong_buffer` length penalty | `grpo.overlong_filtering` (mask truncated) and `reward_shaping` with a DAPO overlong buffer |
| rollout vs trainer mismatch | vLLM importance-sampling correction (default on when vLLM is used) | rollout correction / rollout IS options | `use_importance_sampling_correction`, truncated IS |
| async / staleness | synchronous; `steps_per_generation` / `num_iterations` reuse a generation batch (bounded off-policy) | V1 `trainer_mode`: `sync`, `colocate_async` or `separate_async` with `max_off_policy_threshold` (default 8, strategy `drop`); also fully-async and one-step-off-policy modules | `grpo.async_grpo` with a maximum rollout age in training steps (default 1) and in-flight weight updates |
| MoE router replay | none | `enable_rollout_routing_replay` (rollout-recorded routing) and Megatron `router_replay` modes | Megatron `policy.router_replay` (vLLM records the routing, training replays it) |

## What to expect to differ even on identical data and reward

- **Algorithm defaults.**
  - NeMo-RL's leave-one-out baseline is on by default. It is turned off here so that, like TRL, the
    mean includes the sample itself.
  - The KL coefficient defaults differ: TRL 0.0, verl 0.001 when enabled, NeMo-RL 0.01. All three are
    set to 0.02 here.
  - verl and NeMo-RL clamp the k3 term. TRL does not.
  - TRL's `dapo` loss normalizes by the tokens in the global batch. verl's `token-mean` and NeMo-RL's
    token-level loss normalize per mini- or micro-batch. With one mini-batch per step they agree up to
    gradient-accumulation details.
  - The learning-rate schedule differs too: TRL decays linearly by default, the two smoke configs keep
    it constant, and NeMo-RL's example config warms up for 50 steps.
- **Sampling distribution.** Our TRL runs sampled with HF `generate`. verl and NeMo-RL sample with
  vLLM, which uses different kernels and bf16 numerics. At temperature 1.0 the empirical arm mix of a
  near-deterministic 16-token policy can differ, and the logprobs the trainer recomputes are not the
  sampler's. The stacks correct for that mismatch in different ways or not at all.
  `degenerate_group_rate` (identical groups, zero advantage) is the quantity to watch: it was the
  failure mode of the TRL runs.
- **Precision.** All three train bf16 with LoRA adapters. TRL's adapters were fp32 under autocast
  (`modal_common.run_sft`) or PEFT's default; verl's FSDP mixed precision keeps fp32 master weights;
  NeMo-RL's DTensor policy follows `policy.precision`.
- **Async staleness.** All three smoke runs are synchronous and on-policy: one optimizer step per
  generation batch. Turning on verl's async modes or NeMo-RL's `async_grpo` makes the gradient
  off-policy by up to the configured age, so their curves are then not comparable step for step.
- **What is not compared.** λ is fixed here and learned in TRL phase 1 (from 0, so its early steps
  charge almost nothing). The TRL phase-1 prompt had no price line. The side-by-side table below is a
  sanity check that the plumbing gives numbers in the same range, not a platform benchmark: one seed,
  20 steps each.

## Smoke runs (20 steps, Qwen3-1.7B + LoRA, 8 prompts × 8 completions, λ = 0.4)

Each smoke run ran once, on one A100-40GB. The verl run took 795 s of container time; its first
training step came 481 s after the container started (Ray, vLLM start-up and model load), then about
15 s/step. The NeMo-RL run took 592 s; its first step came at 270 s, then about 15.5 s/step. Per-step
tables are in `results/llm/backends/{verl,nemo_rl}_smoke.md`.

| stack | steps | reward (mean of 20) | quality | cost | invalid (step 1 → 20) | identical-completion groups |
|---|---|---|---|---|---|---|
| verl 0.9.1 (vLLM 0.24) | 20 | 0.583 | 0.677 | 0.114 | 3.1% → 0.0% | 46% |
| NeMo-RL v0.7.0 image (vLLM 0.20) | 20 | 0.620 | 0.699 | 0.149 | 15.6% → 0.0% | 58% |
| TRL 1.15 phase 1, first 20 steps (HF generate; λ learned from 0; no price line) | 20 | 0.548 | 0.628 | 0.104 | 14.1% → 0.0% | not logged |

All three move the same way in 20 steps: invalid outputs disappear, and the mass shifts from
`cheap RETRIEVE` to `mid RETRIEVE`. The TRL row is not a like-for-like control: its price started at 0
and was not shown in the prompt.

What it took to get the runs going (the first attempts failed in ways the configs now encode):

- **verl.** Modal mounts the entry file at `/root`, so the repo-relative path logic needed a guard.
- **NeMo-RL scheduler.** A single-entry scheduler list fails in v0.7.0: it needs the `SequentialLR`
  form with `milestones`.
- **NeMo-RL LoRA targets.**
  - Bare module names match nothing, so the optimizer got an empty parameter list. They must be
    wildcard patterns (`*.q_proj`).
  - `match_all_linear` also adapts `lm_head`, unlike TRL.
- **NeMo-RL version string.** The image reports `nemo_rl` 0.6.0+19244a0 even though it is tagged
  v0.7.0.
- **GPU type.** Modal served one of the A100-40GB requests on an 80 GB card.

**Evaluating a NeMo-RL checkpoint.** A DTensor-v2 LoRA checkpoint needs no conversion.
`ckpt/step_N/policy/weights/model/` is a standard PEFT adapter: `adapter_config.json` (r 16, α 32) and
`adapter_model.safetensors` with `base_model.model.…lora_{A,B}.weight` keys. NeMo's
`automodel_peft_config.json` sits beside them. `nemo_rl_grpo.nemo_rl_adapter_dir(step_dir)` returns
that path, and the shared evaluator loads it with `PeftModel.from_pretrained`. A full-weight run would
save a sharded DCP checkpoint instead, which needs NeMo-RL's `examples/converters/convert_dcp_to_hf.py`.

**NeMo-RL dynamic sampling leaks constant groups.** Its filter keeps a group when the reward std is
`!= 0`, with the std computed in fp32 as sqrt(E[x²] − E[x]²). Groups of exactly equal rewards often
pass on rounding noise. In the 120-step run, 58–62% of the trained groups had zero true advantage
(see COMPARISON.md).

### Phase-1b reproduction

See `results/llm/backends/COMPARISON.md`: TRL as logged, TRL + dynamic sampling, verl's DAPO recipe,
and NeMo-RL curves, all evaluated with one evaluator.

- **Fixed price.** With λ fixed at 0.44 in the prompt, TRL with dynamic sampling and verl's DAPO
  recipe both learn an easy/hard split (easy → `mid RETRIEVE`, hard → frontier). Phase 1b never did.
- **Same outcome on both stacks.** They land within 0.001 J of each other, so the platform is not
  what changed the outcome.
- **The change is not attributed to dynamic sampling alone yet.** A fixed-price TRL control without
  dynamic sampling would separate it from the fixed-price prompt, and it was not run.
