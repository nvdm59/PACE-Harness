# The TRL failure case on three stacks: TRL, TRL + dynamic sampling, verl (DAPO recipe), NeMo-RL, and the fixed-price control

**Question.** Does a policy trained on the phase-1b setup learn to act differently by difficulty? TRL
phase 1b never did. The setup: no SFT warm start, Qwen3-1.7B + LoRA r16/α32, temperature 1.0, KL β 0.05,
entropy bonus 0.01, lr 1e-5, at most 16 tokens.

**Arms.**

1. **TRL phase 1b, as logged.** No new run.
2. **TRL + dynamic sampling + clip 0.2/0.28.** Groups whose 16 rewards are all equal are dropped and
   redrawn: up to 8 generation rounds per step, topped up with dropped groups if still short. Same stack
   as phase 1 (TRL 1.15, HF `generate`).
3. **verl 0.9.1, DAPO recipe.** `filter_groups` on, `clip_ratio_low/high` 0.2/0.28, `token-mean` loss,
   k3 KL in the loss, vLLM 0.24 rollouts.
4. **NeMo-RL (NGC v0.7.0 image), equivalent options, 120 steps.** `use_dynamic_sampling`, `ratio_clip` 0.2/0.28,
   token-level loss, k3 KL in the loss, vLLM 0.20 rollouts.

**Shared setup for arms 2–4.**

- The price is **fixed at λ = 0.44**, shown in the prompt line and charged in the reward.
- Group size 16 with 4 prompts per step, so 64 completions per step.
- The task pool is the phase-1c one (2,000 tasks, seed 0).
- Every per-step number is re-scored from the logged completions with `SimEpisodeReward`.
- **Every adapter is evaluated by one evaluator**: HF `generate`, the phase-1 eval engine, greedy and
  sampled at T = 1, on the 2,000 seed-777 tasks with the λ = 0.44 prompt
  (`trl_grpo.py::evaluate`).

## Train-time: first 20 steps

| arm | steps logged | reward (first 20) | quality | cost | identical-reward groups pre → post filter | entropy | s/step | reward curve (steps 1, 5, 9, 13, 17) |
|---|---|---|---|---|---|---|---|---|
| (1) TRL phase 1b, as logged | 20 | 0.549 | 0.628 | 0.104 | 48% → 48% | – | – | 0.297, 0.481, 0.477, 0.547, 0.737 |
| (2) TRL + dynamic sampling + clip 0.2/0.28 | 20 | 0.525 | 0.634 | 0.099 | 44% → 0% | 0.048 | 8.3 | 0.155, 0.728, 0.333, 0.577, 0.721 |
| (3) verl, DAPO recipe | 20 | 0.599 | 0.692 | 0.125 | 7% → 0% | 0.049 | 18.8 | 0.383, 0.56, 0.584, 0.546, 0.79 |
| (4) NeMo-RL, DAPO recipe (120-step run) | 20 | 0.606 | 0.703 | 0.170 | 48% → 25% | 0.038 | 18.1 | 0.1, 0.55, 0.603, 0.696, 0.682 |
| control: TRL, fixed price, no dynamic sampling | 20 | 0.472 | 0.573 | 0.079 | 42% → 42% | 0.033 | 5.7 | 0.214, 0.442, 0.332, 0.444, 0.6 |

Notes on the train-time table:

- For arm 1 the identical-group column is the identical-*completion* rate from its log. Rewards were
  not logged per sample, so it is a lower bound on the identical-reward rate. Arm 1 also charged a
  *learned* λ starting at 0, so its reward column is not on the same scale.
- verl's pre-filter rate is reconstructed from its `filter_groups/evicted_samples` counter, so it is
  approximate. verl refills one prompt at a time and has no round cap: 0.9.1's V1 buffer ignores
  `max_num_gen_batches`.
- NeMo-RL's pre-filter rate is computed over all generation rounds of the step. Its post-filter
  rate is the EXACT identical-reward rate of the groups it trained on: 25% in the first 20 steps and
  58–62% over the whole run. NeMo-RL's own `baseline_reward/pct_mixed` reads 100% on every step.
  - **Why the two disagree.** NeMo-RL's filter keeps a group when its reward std `!= 0`, and it
    computes that std in fp32 on the GPU as sqrt(E[x²] − E[x]²). For a group of exactly equal rewards
    the difference often rounds to a tiny positive number, so the group passes. Its advantage is then
    rounding noise.
  - **How the trained groups were identified.** They are matched to NeMo's logged
    `train/filtered_reward` by `nemo_rl_grpo._trained_groups`. Every step matches, and the 41 steps
    with more than one match change the rate by at most 3 points.
  - **Effect.** NeMo-RL's dynamic sampling trains, on average, on about 1.6 groups with signal out of
    4, versus 4 of 4 for TRL + DS and verl. It is not equivalent to theirs.
- The control's columns are its own log: no filter, so pre = post.
- Entropy is TRL's logged `entropy`, verl's `actor/entropy` and NeMo-RL's `approx_entropy`. They are
  computed differently and are only comparable within an arm.

## Steps, wall clock and GPU minutes (A100-40GB)

| arm | steps planned → done | s/step (all steps) | container wall clock | GPU-min | adapter evaluated |
|---|---|---|---|---|---|
| (1) TRL 1b as logged | 300 → 300 (group 8, not 16) | 4.6 | 1,422 s (earlier run) | – (not rerun) | step 300 (`eval_table_phase1b.md`) |
| (2) TRL + DS | 200 → 186; stopped by the 2,400 s function timeout | 12.7 | 2,400 s | 40.0 | step 150 (last saved) |
| (3) verl DAPO | 120 → 117; stopped by the launcher's time cap. A first 200-step attempt OOM'd at start-up | 19.5 | 2,670 s (+384 s for the OOM attempt) | 44.5 + 6.4 | step 100 (last checkpoint, merged with `verl.model_merger`) |
| (4) NeMo-RL DAPO | 120 → 120 (a 25-step pilot came first: 693 s) | 19.8 (1–7 generation rounds per step) | 2,658 s + 693 s pilot | 44.3 + 11.6 | step 120 (`ckpt/step_120/policy/weights/model`, a PEFT adapter as saved) |
| control: TRL, fixed price, no DS | 150 → 150 | 5.4 | 845 s | 14.1 | step 150 (`adapter_final`) |
| shared evaluator | – | – | 227 s + 99 s (NeMo-RL) + 154 s (control) of generation | ≈ 10 | – |

The first write-up used ≈ 107.5 GPU-minutes. The NeMo-RL 120-step run, the control and their
evaluations were run afterwards, on a separate allowance. The 117/120/150/186-step lengths follow from
the time caps. They are not a design choice.

## Final evaluation (2,000 tasks, seed 777, B = 0.30)

| policy (2,000 tasks, seed 777, λ = 0.44 in the prompt) | mode | quality | cost | J | within B | invalid | easy: top actions | hard: top actions | acts by difficulty |
|---|---|---|---|---|---|---|---|---|---|
| base zero-shot | greedy | 0.594 | 0.056 | 0.569 | yes | 9.1% | cheap RETRIEVE 63%, cheap ANSWER 21% | cheap RETRIEVE 57%, mid RETRIEVE 21% | no |
| base zero-shot | sampled | 0.593 | 0.055 | 0.569 | yes | 9.0% | cheap RETRIEVE 60%, cheap ANSWER 23% | cheap RETRIEVE 56%, mid RETRIEVE 20% | no |
| TRL+DS (step 150) | greedy | 0.828 | 0.333 | 0.681 | NO | 0.0% | mid RETRIEVE 99%, mid ESCALATE 1% | frontier RETRIEVE 87%, frontier VERIFY 8% | yes |
| TRL+DS (step 150) | sampled | 0.824 | 0.325 | 0.681 | NO | 0.1% | mid RETRIEVE 99%, mid ESCALATE 1% | frontier RETRIEVE 83%, frontier VERIFY 9% | yes |
| verl DAPO (step 100) | greedy | 0.829 | 0.334 | 0.682 | NO | 0.0% | mid RETRIEVE 100%, mid VERIFY 0% | frontier VERIFY 82%, frontier RETRIEVE 18% | yes |
| verl DAPO (step 100) | sampled | 0.827 | 0.330 | 0.682 | NO | 0.4% | mid RETRIEVE 99%, mid VERIFY 1% | frontier VERIFY 79%, frontier RETRIEVE 18% | yes |
| NeMo-RL DAPO (step 120) | greedy | 0.829 | 0.333 | 0.682 | NO | 0.0% | mid RETRIEVE 100% | frontier VERIFY 100% | yes |
| NeMo-RL DAPO (step 120) | sampled | 0.829 | 0.333 | 0.682 | NO | 0.0% | mid RETRIEVE 100%, mid VERIFY 0% | frontier VERIFY 100% | yes |
| trl_fixedprice_1b | greedy | 0.796 | 0.268 | 0.679 | yes | 0.0% | mid RETRIEVE 95%, frontier RETRIEVE 4% | frontier RETRIEVE 53%, mid RETRIEVE 43% | yes |
| trl_fixedprice_1b | sampled | 0.794 | 0.268 | 0.676 | yes | 0.1% | mid RETRIEVE 92%, frontier RETRIEVE 6% | frontier RETRIEVE 49%, mid RETRIEVE 46% | yes |
| (1) TRL 1b step 300, from `eval_table_phase1b.md` (prompt without a price line, λ̄ = 1.21) | greedy | 0.832 | 0.434 | – | NO | 0.1% | frontier ANSWER 83% overall, ≥ 96% frontier on every task | (easy/hard split not logged) | no: one tier for every task |
| reference: uniform `mid ESCALATE` | – | 0.768 | 0.150 | 0.702 | yes | 0% | – | – | no |
| reference: best fixed action per proxy at λ = 0.44 (easy `mid ESCALATE`, hard `frontier ANSWER`) | – | 0.814 | 0.250 | 0.704 | yes | 0% | mid ESCALATE 100% | frontier ANSWER 100% | yes |

## Interpretation (four arms + control)

- **All three stacks learn the same split with the DAPO recipe.**
  - Easy tasks go to `mid RETRIEVE`: ≥ 98.6% on TRL+DS and verl, 100% on NeMo-RL.
  - Hard tasks go to the frontier tier: 98% on TRL+DS (RETRIEVE/VERIFY), 99% on verl (mostly VERIFY),
    100% `frontier VERIFY` on NeMo-RL.
  - Priced J is 0.681 / 0.682 / 0.682, and cost is 0.333 / 0.334 / 0.333. All three are 0.01 over
    B + 0.02.
  - **Platform effect:** within 0.001 J and 0.001 in cost across HF generate + TRL, vLLM 0.24 + verl
    and vLLM 0.20 + NeMo-RL. The one difference is RETRIEVE vs VERIFY on hard tasks: a 0.003-J near-tie
    in the simulator.
- **The control shows the price, not the recipe, is what enables learning.** TRL at the fixed price
  without dynamic sampling (symmetric clip) also splits. Easy tasks go 95% to `mid RETRIEVE`; hard tasks
  go 53% frontier and 43% mid. It reaches J 0.679, within budget at cost 0.268. It got there with
  identical-reward groups at 62% on average after step 20, and ≥ 75% on half of the steps: the same
  starvation as TRL-1b. TRL-1b moved λ by
  dual ascent every step and never split.
- **Recipe effect, separated from the platform.** Dynamic sampling + clip-higher sharpens the hard-task
  split from 53% to 98–100% frontier. That costs +0.065 and pushes the policy over budget, for
  +0.002–0.003 J. It does not enable the split.
  - The recipe's effect replicates across stacks even though NeMo-RL's filter leaks: it trained on
    exactly-constant groups 58–62% of the time, against 0% for TRL+DS and verl.
  - So the sharpening does not need a perfect filter. It needs only that some starved batches are
    replaced.
- **Rollout engine, precision, async.**
  - All runs were synchronous and on-policy, so there is no staleness.
  - Training samples came from HF generate (TRL) or bf16 vLLM (verl, NeMo-RL). Evaluation used one
    engine for every adapter, so the eval numbers carry no engine effect.
  - The precision effect that was found is the fp32 std test in NeMo-RL's filter. It changes what is
    trained on, but not the end point.
- **No parallelism effect is measured.** Every run used one A100-40GB: world size 1. FSDP (verl),
  DTensor/FSDP2 (NeMo-RL) and the Megatron options are nominal here. Nothing is sharded, and no tensor
  or pipeline parallelism runs.
- **Statement.**
  - The phase-1b failure was the moving price.
  - With λ fixed, every stack learns the difficulty→tier mapping without a warm start.
  - The DAPO recipe sharpens the mapping on every stack, at a small budget cost and no priced-J gain.
  - The platform changes nothing measurable at this resolution: one seed per arm, a 0.002-J margin
    between the split and uniform `mid ESCALATE`.

Files: `trl_dapo_1b.jsonl`, `verl_dapo_1b.jsonl`, `nemo_rl_dapo_1b_full.jsonl` (+ its `.metrics.jsonl`), `nemo_rl_dapo_1b.jsonl`
(the 25-step pilot) and `trl_fixedprice_1b.jsonl` (per step) with their `_meta.json`; `eval_phase1b_repro.json`, `eval_nemo_rl_full.json`
and `eval_fixedprice_control.json` (all eval numbers, including the easy/hard mixes). Tables are
regenerated with `python -m pace.llm.backends.compare`. Every run is a single seed.
