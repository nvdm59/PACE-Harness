# EvoHarnessBench, tools axis, EOG, stage 1: 10-task smoke (hosted service)

Live run on the hosted service (SDK `simple_agentic_evals` 0.19.2), counterfactual mode: every tier
on every task, one fresh gym session per (task, tier), tools restricted to the stage-1 cumulative
allowlist, graded by the service's hidden verifiers. Seeded proportional sample over
(domain, split): 7 held-out (test) + 3 adaptation (train) tasks across calendar, csm, drive,
hybrid, itsm, teams. Max 20 turns, `effort=low` on mid / frontier. Traces:
`traces/evoharness/evoharness_tools_s1.jsonl`; ledger `traces/evoharness/spend*.json*`.

| arm | score (mean verifier pass rate) | strict pass | cost / task |
|---|---|---|---|
| cheap (Haiku 4.5) | 0.601 | 0.30 | $0.140 |
| mid (Sonnet 5.5) | 0.777 | 0.50 | $0.298 |
| frontier (Opus 5.5) | 0.868 | 0.70 | $0.679 |
| per-task oracle (cheapest tier at best score) | 0.868 | 0.70 | $0.374 |

Oracle tier labels: cheap 4, mid 4, frontier 2. The oracle matches frontier quality at 55% of its cost.
n = 10, so these are smoke numbers, not estimates.

**Spend: $11.17 total, $1.117 per task for all three tiers** (4.34M input vs 0.05M output tokens).
That is above the $1/task threshold set for the full tools-axis collection, so the full collection
was not started. Input tokens dominate: every turn resends the domain policy (~9k chars) plus the
stage tool schemas. Anthropic prompt caching on that fixed prefix (reads billed at 0.1x) would cut
the per-task cost by an estimated 60-75%, but the shared `PricedLLM` ledger path does not yet send
`cache_control` or record cache tokens, so it was not used.

The projection guard (per-task cost x 60 > $50 cap) stopped the first pass after 3 tasks at
$0.91/task; the remaining 7 were completed with the guard off to finish the 10-task measurement
asked for (resume, no task re-run).


# Tools axis with prompt caching: partial collection (stopped by the Anthropic account limit)

`PricedLLM` now sends Anthropic `cache_control` breakpoints on the system prompt (domain policy,
action protocol, stage tool schemas) and on the last message, and prices cache writes at 1.25x
and cache reads at 0.1x of the input price (ledger rows carry `cache_write_tokens` /
`cache_read_tokens`). It is on for EvoHarness, off by default elsewhere (`PACE_PROMPT_CACHE=1`
opts in), so the CRM collections keep their accounting.

* 3-task check (stage 1, after the smoke): $0.258 per task for all three tiers, vs $1.117 uncached.
* Over the whole cached collection: $0.377 / $0.433 / $0.518 per task (all tiers) at stages 1 / 2 / 4.

Full run launched as `collect --axis tools --stage all --n-tasks 300 --tiers all --project-n 0`
(allocation 115 / 58 / 60 / 53 / 10 / 4 over stages 1-6) in three parallel workers. Each worker had
its own ledger and a slice of the $150 cap: stage 1 on `traces/evoharness` to $72 cumulative,
stages 2-3 on `ledger_w2` $45, stages 4-6 on `ledger_w3` $33. All three aborted (exit 4) on the
same HTTP 400: "You have reached your specified API usage limits. You will regain access on
2026-11-01". The $150 cap did not bind: total spend is $31.30 ($18.36 + $5.93 + $7.01, the
smoke included). Rerunning the same commands resumes.

Collected, tasks with all three tiers (held-out / adaptation):

| stage | tasks | cheap score / pass / $ | mid | frontier | oracle (score / pass / $) | oracle labels c/m/f |
|---|---|---|---|---|---|---|
| H1 | 29 (20 / 9) | 0.502 / 0.24 / 0.074 | 0.649 / 0.31 / 0.178 | 0.645 / 0.41 / 0.381 | 0.731 / 0.48 / 0.179 | 16 / 9 / 4 |
| H2 | 13 (9 / 4) | 0.219 / 0.15 / 0.048 | 0.613 / 0.31 / 0.124 | 0.523 / 0.23 / 0.260 | 0.661 / 0.38 / 0.138 | 5 / 6 / 2 |
| H4 | 13 (9 / 4) | 0.392 / 0.08 / 0.056 | 0.761 / 0.38 / 0.162 | 0.700 / 0.23 / 0.300 | 0.793 / 0.38 / 0.146 | 4 / 9 / 0 |

Stages 3, 5 and 6 have no tasks yet; H2 and H4 each have one task with only some tiers. H1 costs
include the 10 uncached smoke tasks. Every task is run under its own stage's harness, so all
held-out tasks are in the FWT group; BWT cells, which re-run earlier tasks under later stages
via `at_stage`, were not collected. The per-stage oracle beats every single tier on score, at
roughly mid-tier cost or less. Mid matches or beats frontier on score at H1, H2 and H4. Sample
sizes are small (13-29 tasks per stage). Full report: `results/evoharness/tools_axis_partial.json`.

# Tools axis, 150 tasks: 50 at each of stages 1, 2 and 4 (prompt caching on)

Resumed after the account limit was raised, with the per-stage target cut from the proportional
allocation (115 / 58 / 53) to 50 tasks per stage on stages 1, 2 and 4; stages 3, 5 and 6 were not
collected. Same seeded round-robin sample, so the earlier tasks are a prefix of this one. Every
task has all three tiers (0 partial). Analysis: `results/evoharness/tools_axis_150.json`
(`python -m pace.experiments.evoharness.run analyze --traces-dir traces/evoharness`).

| stage | tasks (held-out / adaptation) | cheap score / pass / $ | mid | frontier | oracle (score / pass / $) | oracle labels c/m/f |
|---|---|---|---|---|---|---|
| H1 | 50 (36 / 14) | 0.491 / 0.24 / 0.062 | 0.702 / 0.42 / 0.146 | 0.619 / 0.40 / 0.298 | 0.753 / 0.52 / 0.136 | 26 / 20 / 4 |
| H2 | 50 (34 / 16) | 0.370 / 0.16 / 0.044 | 0.699 / 0.40 / 0.114 | 0.610 / 0.30 / 0.227 | 0.741 / 0.48 / 0.106 | 21 / 25 / 4 |
| H4 | 50 (35 / 15) | 0.403 / 0.14 / 0.056 | 0.643 / 0.28 / 0.147 | 0.631 / 0.30 / 0.262 | 0.738 / 0.38 / 0.138 | 18 / 28 / 4 |

Costs are per task at cached prices (H1 includes the 10 uncached smoke tasks). At every stage the
mid tier beats the frontier tier on score at about half its cost; the per-task oracle beats every
single tier at or below mid-tier cost and picks the frontier tier for 4 tasks in 50. The ordering
is the same at the three harness sizes. As before, every task ran under its own stage's allowlist
(FWT group only; no BWT cells), and no controller was trained on these logs.
