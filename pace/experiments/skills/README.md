# Skill handbook for the cheap tier

**Question.** On the 50-task CRMArena-Pro b2b sample, the cheap tier (Haiku 4.5) scores 0.44. It
fails mostly on schema reasoning: which object to query first, how records join, which filters
matter. The mid and frontier tiers solved many of the same tasks, and their successful SOQL is in
the counterfactual log. Can a short per-category handbook distilled from those successes lift the
cheap tier, and at what cost compared with buying the mid tier?

**Method.** This follows Procedural Memory Distillation (arXiv 2607.01480) and Just-in-Time Memory
(arXiv 2609.27334). The difference is that the notes are written by a strong writer (mid tier,
Sonnet 5.5), not by the small model.

1. `distill.py` reads `traces/crmarena_b2b_counterfactual.jsonl` (read only). It keeps the first
   episode per (task, arm), restricted to the collector's seeded 50-task sample (`sample_tasks(50,
   seed=0)`). For each category it takes the mid and frontier episodes with quality ≥ 0.5 (never the
   cheap arm) as exemplars: question → SOQL/SOSL sequence (`meta.queries`) → final answer and score.
   One writer call per category (purpose `skills:distill`, priced through the ledger) turns them
   into ≤ 3 procedural notes of ≤ 40 words each, with no task-specific values. Salesforce-Id-like
   tokens are masked, notes are cut at 40 words, and the output is deterministic in order.
   The result goes to `results/skills/handbook.json`: category → notes, source task ids, and every
   variant with its exemplars and cost.
2. `serve.py` runs `SkilledCRMArenaHarness`, a subclass of `CRMArenaHarness` that wraps the
   backend's prompt builder (`system_prompt`). It appends one "Procedural hints for this category"
   block to the system prompt, and nothing else changes. `PrefixedLLM` wraps `harness.llm` in the
   style of `ssr.refine.tag_purpose`, so every served call is billed as `skills:<purpose>`. No core
   file is edited.
3. `run.py analyze` writes `results/skills/RESULTS.md`. It compares cheap+skills with cheap, mid
   and frontier on the same tasks: quality, cost per task, paired Δ ± SE vs cheap, flips, a
   per-category table, and the per-task oracle with the skilled arm added as a 4th arm.

**Leakage control (leave one task out).** When the cheap tier is served task t, its hints come only
from notes whose source task ids exclude t. It therefore never sees notes distilled from the mid or
frontier solution of its own task. A category with sources S gets one writer call for S (the
deployable "full" entry) and one for S \ {t} for each source task t. A task that was not itself a
source is served the full entry, which already excludes it. When S \ {t} is empty, task t gets no
hints. Variants are cached by (category, source ids), so a rerun makes no new calls.
`notes_for_task` asserts that t is not among the sources of the variant it serves.

*Cost of the control:* 22 calls would build the plain handbook. Leave-one-out needs 49 calls on
this log (18 of 22 categories have at least one solved task; 5 have three source tasks, at 4 calls each). The real
distill run cost **$0.27** in total, against $0.11 for the 18 plain-handbook calls alone. With 2–3 tasks per
category, removing t also leaves 0–2 source tasks, so the served notes are weaker than the
deployable handbook would be. The measured lift is a lower bound for that reason.

## Run

```
# .venv-crmarena; ANTHROPIC_API_KEY and the Salesforce creds come from third_party/CRMArena/.env
python -m pace.experiments.skills.run distill --cap 3                       # ledger traces/skills/distill
python -m pace.experiments.skills.run serve --tier cheap --n-tasks 50 --cap 7 --traces-dir traces/skills
python -m pace.experiments.skills.run analyze                               # -> results/skills/RESULTS.md
python -m pace.experiments.skills.run {distill,serve,analyze} --dry-run --n-tasks 3   # fake world, no spend
python -m pytest tests/test_skills.py -q
```

Each paid step uses its own ledger directory, so each `--cap` is an absolute cap on that step and
never touches `traces/spend.jsonl`. Cap, resume and exit-code semantics match `pace.collect`
(serve calls `collect_crmarena` directly). Exit codes: 0 = done; 3 = stopped at the cap, with
`STOPPED_AT_CAP` written; 4 = a model call failed after retries. Served episodes are written to
`traces/skills_cheap_b2b.jsonl`, and `meta.skills` records the variant, its source task ids and
the number of notes.

## Cost estimate

| step | calls | estimate | measured |
|---|---|---|---|
| distill (mid writer) | 22 plain, 49 with leave-one-out | ≈ $0.50 | $0.27 |
| serve (50 cheap episodes) | ≈ 50 × 6 turns | ≈ $3.50 (at the measured $0.068/episode) | $2.39 ($0.048/episode, 262 calls) |

The measured total was $2.66, well under $10.

## Limitations

- n = 50 tasks with 2–3 per category, so per-category numbers are anecdotes and the overall
  paired SE is about ±0.07.
- Leave-one-out handbooks have 0–2 source tasks. In 14 of 50 tasks the category had no other
  solved task, so the cheap tier ran without hints, which dilutes the overall Δ (RESULTS.md
  reports hinted and unhinted tasks separately).
- The writer often exceeds 40 words; notes are cut at 40 words and can end mid-sentence.
- The "no task-specific values" rule is partly prompt-enforced. Only Salesforce-Id-like tokens are
  masked mechanically, and a note can still carry a literal filter value such as a status string
  seen in the source SOQL.
- One serve run per task. Cheap-tier CRMArena episodes are stochastic, so a single paired run mixes
  the hint effect with run-to-run noise. In the real run, cheap+skills scored 0.581 against 0.444
  for the logged cheap arm (paired Δ +0.137 ± 0.070, 10 upward flips and 3 downward). The 14 tasks
  that got no hints rose by about the same amount as the 36 hinted ones (+0.131 vs +0.139;
  difference +0.008 ± 0.134). The lift is therefore not attributable to the handbook on this
  evidence. A resampled cheap control without hints is the missing arm.
- Only `meta.queries` (the SOQL text) and the final answer are distilled. The log holds no
  per-step observations (`state` is null in counterfactual mode), so the writer never sees which
  queries returned rows.
