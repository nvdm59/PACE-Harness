# The TRL failure case on three stacks: TRL, TRL + dynamic sampling, verl (DAPO recipe), NeMo-RL

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
4. **NeMo-RL (NGC v0.7.0 image), equivalent options.** `use_dynamic_sampling`, `ratio_clip` 0.2/0.28,
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
| (4) NeMo-RL, DAPO recipe | 20 | 0.594 | 0.693 | 0.180 | 39% → 0% | 0.036 | 17.0 | 0.1, 0.544, 0.749, 0.631, 0.776 |

Notes on the train-time table:

- For arm 1 the identical-group column is the identical-*completion* rate from its log. Rewards were
  not logged per sample, so it is a lower bound on the identical-reward rate. Arm 1 also charged a
  *learned* λ starting at 0, so its reward column is not on the same scale.
- verl's pre-filter rate is reconstructed from its `filter_groups/evicted_samples` counter, so it is
  approximate. verl refills one prompt at a time and has no round cap: 0.9.1's V1 buffer ignores
  `max_num_gen_batches`.
- NeMo-RL's pre-filter rate is computed over all generation rounds of the step.
- Entropy is TRL's logged `entropy`, verl's `actor/entropy` and NeMo-RL's `approx_entropy`. They are
  computed differently and are only comparable within an arm.

## Steps, wall clock and GPU minutes (A100-40GB)

| arm | steps planned → done | s/step (all steps) | container wall clock | GPU-min | adapter evaluated |
|---|---|---|---|---|---|
| (1) TRL 1b as logged | 300 → 300 (group 8, not 16) | 4.6 | 1,422 s (earlier run) | – (not rerun) | step 300 (`eval_table_phase1b.md`) |
| (2) TRL + DS | 200 → 186; stopped by the 2,400 s function timeout | 12.7 | 2,400 s | 40.0 | step 150 (last saved) |
| (3) verl DAPO | 120 → 117; stopped by the launcher's time cap. A first 200-step attempt OOM'd at start-up | 19.5 | 2,670 s (+384 s for the OOM attempt) | 44.5 + 6.4 | step 100 (last checkpoint, merged with `verl.model_merger`) |
| (4) NeMo-RL DAPO | 25 → 25 (curves only; budget) | 17.0 | 693 s | 11.6 | none: no final eval |
| shared evaluator | – | – | 227 s of generation + load | ≈ 4.5 | – |

Phase-2 total: ≈ 107.5 GPU-minutes of the 120 allowed. The 120-step and 25-step lengths follow from
that budget. They are not a design choice.

## Final evaluation (2,000 tasks, seed 777, B = 0.30)

| policy (2,000 tasks, seed 777, λ = 0.44 in the prompt) | mode | quality | cost | J | within B | invalid | easy: top actions | hard: top actions | acts by difficulty |
|---|---|---|---|---|---|---|---|---|---|
| base zero-shot | greedy | 0.594 | 0.056 | 0.569 | yes | 9.1% | cheap RETRIEVE 63%, cheap ANSWER 21% | cheap RETRIEVE 57%, mid RETRIEVE 21% | no |
| base zero-shot | sampled | 0.593 | 0.055 | 0.569 | yes | 9.0% | cheap RETRIEVE 60%, cheap ANSWER 23% | cheap RETRIEVE 56%, mid RETRIEVE 20% | no |
| TRL+DS (step 150) | greedy | 0.828 | 0.333 | 0.681 | NO | 0.0% | mid RETRIEVE 99%, mid ESCALATE 1% | frontier RETRIEVE 87%, frontier VERIFY 8% | yes |
| TRL+DS (step 150) | sampled | 0.824 | 0.325 | 0.681 | NO | 0.1% | mid RETRIEVE 99%, mid ESCALATE 1% | frontier RETRIEVE 83%, frontier VERIFY 9% | yes |
| verl DAPO (step 100) | greedy | 0.829 | 0.334 | 0.682 | NO | 0.0% | mid RETRIEVE 100%, mid VERIFY 0% | frontier VERIFY 82%, frontier RETRIEVE 18% | yes |
| verl DAPO (step 100) | sampled | 0.827 | 0.330 | 0.682 | NO | 0.4% | mid RETRIEVE 99%, mid VERIFY 1% | frontier VERIFY 79%, frontier RETRIEVE 18% | yes |
| (1) TRL 1b step 300, from `eval_table_phase1b.md` (prompt without a price line, λ̄ = 1.21) | greedy | 0.832 | 0.434 | – | NO | 0.1% | frontier ANSWER 83% overall, ≥ 96% frontier on every task | (easy/hard split not logged) | no: one tier for every task |
| reference: uniform `mid ESCALATE` | – | 0.768 | 0.150 | 0.702 | yes | 0% | – | – | no |
| reference: best fixed action per proxy at λ = 0.44 (easy `mid ESCALATE`, hard `frontier ANSWER`) | – | 0.814 | 0.250 | 0.704 | yes | 0% | mid ESCALATE 100% | frontier ANSWER 100% | yes |

## Interpretation

- **Arms 2 and 3 both act by difficulty, and arm 1 never did.** Easy tasks go to `mid RETRIEVE` (≥ 98.6%)
  and hard tasks to frontier RETRIEVE/VERIFY (≥ 94% frontier, greedy or sampled). Quality is 0.828 / 0.829 at cost 0.333 /
  0.334, with J = 0.681 / 0.682. Both are 0.01 over B + 0.02 and 0.02 J below the per-proxy optimum.
  They do not find `ESCALATE` on easy tasks, nor plain `ANSWER` on hard ones.
- **The platform effect is nil at this resolution.** TRL with HF generation (arm 2) and verl with vLLM
  (arm 3) land on the same easy/hard split and within 0.001 J. The only visible difference is the arm
  they settle on for hard tasks: RETRIEVE in TRL, VERIFY in verl. That is a near-tie in the simulator
  (on hard tasks frontier RETRIEVE vs VERIFY differ by 0.003 in J). The first-20-step curves of verl and NeMo-RL
  are close to each other (reward 0.60 / 0.59, quality 0.69 / 0.69). NeMo-RL is on a costlier path,
  0.18 vs 0.13. One seed and 20 steps cannot separate that from noise.
- **What explains the change from TRL-1b is the recipe arm (2), not the platform.** Arm 2 differs from
  arm 3 only in stack and reproduces arm 3's outcome. This is not a clean recipe effect, though: arm 2
  also differs from arm 1 in the price. Arm 1 learned λ by dual ascent from 0, reaching 1.2–1.4, and
  its prompt carried no price line. Arm 2 has λ fixed at 0.44 in the prompt and group 16 instead of 8.
  The control that isolates dynamic sampling + clip-higher from the fixed-price prompt is TRL with
  λ = 0.44 and no dynamic sampling. It was not run: no budget left. Until it is, the honest statement
  is: recipe + fixed price together, on either platform; the platform is not the explanation.
- **Dynamic sampling did what it is for.** In arm 2, 44% of generated groups had identical rewards
  in the first 20 steps, and 71% on average over steps 21–186 (`identical_reward_group_rate_pre` in
  `trl_dapo_1b.jsonl`). It needed 1–8 generation rounds per step. After filtering the rate was 0% on
  most steps, and up to 50% on the steps where 8 rounds were not enough and dropped groups topped the
  batch up. That is also why it cost 12.7 s/step against 4.6.
- **Rollout engine, precision, async.** All three arms ran synchronously and on-policy, so there is no
  async staleness. verl and NeMo-RL sample with vLLM in bf16; TRL with HF `generate`. The evaluator is
  the same for all, so the eval numbers carry no engine effect. Only the training samples do.
- **No parallelism effect is measured.** Every run used one GPU: world size 1. verl's FSDP, NeMo-RL's
  DTensor/FSDP2 and the Megatron options are nominal here. Nothing is sharded, and no tensor or
  pipeline parallelism runs.
- **Why "by difficulty" is a weak target at λ = 0.44.** The per-proxy optimum (J 0.704) beats uniform
  `mid ESCALATE` (0.702) by 0.002 J. The reward barely pays for the split, so it is an easy pattern for
  a learner to miss and a noisy one to read across single runs.

Files: `trl_dapo_1b.jsonl`, `verl_dapo_1b.jsonl`, `nemo_rl_dapo_1b.jsonl` (per step) with their
`_meta.json`; `eval_phase1b_repro.json` (all eval numbers, including the easy/hard mixes). Tables are
regenerated with `python -m pace.llm.backends.compare`. Every run is a single seed.
