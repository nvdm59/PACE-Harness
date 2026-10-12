# τ-bench (retail / airline) as a second non-CRM suite

[τ-bench](https://github.com/sierra-research/tau-bench) (Yao et al. 2024, arXiv:2406.12045;
Sierra, MIT license) puts a tool-using customer-service agent in front of an LLM-simulated user
in two domains, **retail** (115 test tasks) and **airline** (50 test tasks). The agent reads and
writes an in-memory database through domain tools under a written policy (the "wiki"). The
episode reward is τ-bench's own `Env.calculate_reward`: 1 iff the final database equals the one
produced by replaying the gold actions **and** every required output string appears in an agent
reply, else 0.

This package runs the same study as the CRMArena-Pro collection (`pace.collect --harness
crmarena`, `scripts/crmarena_offline.py`) on that suite: every task on every tier, so first-step
tier policies, the per-task oracle and the DR gate are exact lookups.

## Setup

    bash scripts/setup_crmarena.sh       # the Python 3.12 venv with litellm (once)
    bash scripts/setup_taubench.sh       # clone into third_party/tau-bench (gitignored), pip install --no-deps
    source .venv-crmarena/bin/activate

τ-bench is installed into `.venv-crmarena` without its provider SDKs (anthropic, mistralai,
google-generativeai), which only its `model_utils` uses; the code path used here needs litellm,
pydantic and tenacity, already present. No second venv. `ANTHROPIC_API_KEY` is read from the
environment or `third_party/CRMArena/.env`.

## Harness (`pace/harness/taubench.py`, version `taubench-v1`)

`TauBenchHarness` wraps one τ-bench task episode. τ-bench's env provides the tools, database,
wiki and user simulator; our tier model is the agent's LLM, called through litellm via the
CRM harness's `PricedLLM` and the spend ledger (the tool schemas are added by a completion
wrapper, so the guarded / priced / recorded call path is the same code).

* **Decision points**, one per agent step: tier × {CONTINUE, ANSWER}. CONTINUE@t is one normal
  τ-bench agent step on tier t (first tool call executed, or a reply to the user). ANSWER@t
  ends the conversation: tier t writes its final reply with tools disabled, the reply is
  recorded as a `respond` action and the episode is scored. Only ANSWER@* is legal on the last
  step; the step limit is τ-bench's default `max_num_steps = 30`.
* **User simulator**: τ-bench's `LLMUserSimulationEnv`, fixed to the **cheap tier (Haiku 4.5)
  for every arm and policy**, called through the same ledger under purpose `user_sim`. Its cost
  is logged per episode (`meta.user_cost_usd`) and excluded from the episode cost.
* **Quality** = τ-bench's reward (0/1). **Cost** = our agent's token cost only.
  `meta.reward_strict` keeps τ-bench's runner semantics (0 unless the env itself ended the
  episode by a user stop or a terminate tool).
* **Features** (16): bias, domain one-hot, turn index one-hot (0–5, ≥6), turn fraction, tool
  calls so far / 10, cost so far, last observation kind (none / user / tool_ok / tool_error /
  invalid). At step 0 only the domain varies across tasks, so a first-step tier policy is
  context-free within a domain.
* **Metric**: pass^1 only (one trial per task per tier). τ-bench's pass^k (all of k i.i.d.
  trials succeed) needs k trials per task and is not estimated.

Deviations from τ-bench's runner are listed in the module docstring (no temperature, bounded
max_tokens, cheap-tier user simulator instead of gpt-4o, scoring at the step limit, malformed
tool arguments answered with an error instead of aborting the episode).

## Commands

    python -m pace.experiments.taubench.run collect --domain retail --n-tasks 10 --tiers all \
        --mode counterfactual --seed 0 --cap 8 --traces-dir traces/taubench \
        --out traces/taubench_retail_counterfactual.jsonl
    python -m pace.experiments.taubench.run analyze \
        --log traces/taubench_retail_counterfactual.jsonl --log traces/taubench_airline_counterfactual.jsonl \
        --out results/taubench/RESULTS.md

`collect`: seeded shuffle of the domain's test-split task ids (a larger `--n-tasks` extends a
smaller run with the same seed); resumable per (task, tier); the `--cap` is on the cumulative
`--traces-dir` ledger; exit 3 and `STOPPED_AT_CAP` at the cap, exit 4 when a model call fails
after retries. Behavior = uniform tier per task, CONTINUE until the episode ends (propensity
1/3 at step 0, 1 after), as in the CRMArena collection.

`analyze`: offline. Imports `cv_tables` and `gate_section` from `scripts/crmarena_offline.py`
(not copied) and runs them per domain and pooled, with the budget set to the uniform
behavior's mean agent cost unless `--budget` is given. Writes `results/taubench/RESULTS.md`
(+ `.json`).

Tests: `python -m pytest tests/test_taubench.py -q` (fake env and completion; no network).
