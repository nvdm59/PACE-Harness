# Priced Socratic self-refinement as a test-time compute policy

**Question.** Socratic Self-Refine (SSR, [arXiv 2511.10621](https://arxiv.org/abs/2511.10621))
improves an LLM response by decomposing it into verifiable sub-questions, re-solving each,
estimating confidence per step and rewriting where confidence is low. On an agent that queries a
live CRM, each of those steps is a priced call. So the question is not *whether* SSR helps on
CRMArena-Pro, but **how much of it to buy for a given request**: refine or not, how many rounds,
on which tier. That is a PACE decision: a per-request policy over `{STOP, REFINE@cheap,
REFINE@mid}` with cost in the objective, evaluated as a quality-cost frontier and under a budget.
The gating signal is the confidence/uncertainty estimate of the current answer, in the two
standard families, verbalized confidence and agreement across samples (cf.
[arXiv 2506.01114](https://arxiv.org/abs/2506.01114)).

**Hypotheses.**

- **H1.** Refinement helps mainly when the confidence of the initial answer is low. When it is
  high, a round mostly costs money and sometimes flips a correct answer ("refinement harm").
- **H2.** Under a budget, `always refine` is dominated: a gate that spends the same money on the
  uncertain tasks reaches higher quality at lower cost.
- **H3.** A priced gate (a controller with the price of cost in its objective, over the category,
  the confidence signal, the cost so far and the round index) reaches the quality-cost frontier
  traced by the sweep of fixed thresholds, and does so at a budget nobody tuned a threshold for.

## Design

One SSR round on a CRMArena-Pro answer (`refine.py`), every call priced through the spend ledger
with its own `purpose`:

| step | call(s) | ledger purpose | what it produces |
|---|---|---|---|
| probe | 1 call on the cheap tier | `ssr:decompose` | 1–3 sub-claims, each with a Socratic question, the value it asserts and a self-reported confidence; a whole-answer confidence |
| re-solve | per claim, k = 2 worker mini-episodes on the chosen tier through `CRMArenaHarness.step` (`CONTINUE@tier` ≤ 3 turns, `ANSWER@tier` forced on the last), so the SOQL runs on the real org | `ssr:resolve:<tier>` | k answers per claim; per-claim confidence = mean(self-reported, agreement across the k samples, support for the asserted value) |
| rewrite | 1 call on the chosen tier that sees the claims and the workers' findings | `ssr:refine:<tier>` | the refined answer |
| score | CRMArena's own `Evaluator.evaluate` on the refined answer | `extractor` / `privacy_judge` | quality in [0, 1]; billed, not charged (as in the core harness) |

The probe is split from the rest so that a gate can **look before it pays**: the decomposition
costs about $0.003 and its minimum claim confidence is the gating signal; the re-solve is where the
money goes. The harness is not modified: `tag_purpose` wraps `harness.llm` during a worker
episode, and the worker's sub-task is a `CRMTask` whose query is the Socratic question, so it gets
the same system prompt, schema and connector as the main agent.

**Decision and policies** (`policy.py`). At each node (round 0 = the initial answer; round r = the
answer after r refinements) the policy sees the task category, the round, the cost so far, the
probe's confidences and what the last round did, and chooses `STOP` or `REFINE@tier`, up to 2
rounds.

| policy | rule | pays the probe? |
|---|---|---|
| `never` | stop at the initial answer | no |
| `always_1:t`, `always_2:t` | 1 or 2 rounds on tier t | only where it refines (the decomposition is part of the round) |
| `uncertainty_gated:τ:t` | refine on t iff the probe's minimum claim confidence < τ; τ swept 0.3–0.9 | at every visited node |
| `controller:<json>` | a PACE softmax-linear policy (`pace.controller.Controller`) over 36 features: bias, category one-hot, round one-hot, probe min / mean / whole-answer confidence, last round's post-re-solve confidence, whether the last round changed the answer, cost so far, initial arm | at every visited node |

The controller is trained **offline on the logged tree by exact policy gradient**: every branch is
enumerated, so the policy's expected quality and cost are computed, not sampled. With a budget B,
the price of cost λ is a Lagrange multiplier updated by damped dual ascent on an EMA of the exact
expected cost, and the served policy is the iterate average of the second half of training, as in
`pace.train_offline`. It is also trained at fixed λ ∈ {2, 5}/USD for the priced comparison.
Held-out numbers are 5-fold CV over tasks, category-stratified.

**Why a tree.** `run --policy tree` logs, per task, the probe at the initial answer, a refinement
on **every** tier, the probe of each refined answer, and a second refinement on every tier below
each (1 + 2 + 4 terminal answers over {cheap, mid}). Every policy above is a function of the
observable node state, so once the tree is logged the quality and cost of each are **exact
lookups**, the same trick `pace.lookup` uses for the first-step tier on the counterfactual
collection. The τ sweep, the controller's CV and the budget table all come from one collection,
and no policy is ever re-run. A single policy can also be served live (`--policy
uncertainty_gated:0.6:cheap`); then only its branch is logged.

**Initial answers are free.** By default (`--initial from-log --initial-arm cheap`) the initial
answer of a task is the already-collected and already-scored cheap-tier episode in
`traces/crmarena_b2b_counterfactual.jsonl` (answer, queries, quality, cost). The cheap arm is the
one with headroom (quality 0.444 vs 0.608 mid and 0.663 frontier), and it makes the comparison
pointed: is it better to buy refinement on a cheap answer, or a better tier up front (the
counterfactual log has that answer too)? `--initial-arm all` refines all three arms;
`--initial fresh` re-runs the initial episodes (charged as `agent:<tier>`).

## How to run

```bash
# dry run: fake world, no key, no network, no spend; writes results/ssr/dryrun/
python -m pace.experiments.ssr.run run --dry-run --n-tasks 50 --tiers cheap,mid
python -m pace.experiments.ssr.run analyze --log results/ssr/dryrun/ssr_tree.jsonl \
    --out results/ssr/RESULTS.md --csv results/ssr/frontier.csv

# real run (paid; same cap, ledger and resume semantics as pace.collect), staged:
source .venv-crmarena/bin/activate            # keys in third_party/CRMArena/.env
python -m pace.experiments.ssr.run run --policy tree --n-tasks 50 --org b2b --tiers cheap,mid \
    --traces-dir traces/ssr --max-rounds 1 --cap 25     # stage A: one round on both tiers
python -m pace.experiments.ssr.run run --policy tree --n-tasks 50 --org b2b --tiers cheap,mid \
    --traces-dir traces/ssr --max-rounds 2 --cap 60     # stage B: extend the logged tree to two rounds
python -m pace.experiments.ssr.run analyze --log traces/ssr/ssr_b2b_tree.jsonl

python -m pytest tests/test_ssr.py -q         # mocked: ledger purposes, gate, harm accounting, controller, dry run
```

`--cap` is a hard cap on cumulative ledger spend. `--traces-dir traces/ssr` gives the experiment
its own ledger (`traces/ssr/spend.jsonl`), so the caps above are absolute; without it the ledger is
`traces/spend.jsonl`, which already holds the $17.98 of the collection runs, and a cap must be set
relative to that. The dedicated Anthropic workspace limit remains the second backstop. Before each call the ledger prices
its worst case (input at 2 chars/token plus the full `max_tokens`) and raises before sending if it
would cross the cap; the run then writes `traces/STOPPED_AT_CAP` and exits 3. The tree log is
resumable: logged branches are skipped, an interrupted branch is re-run, and a tree logged to
depth 1 is extended to depth 2 by the second command (its leaves are probed and expanded).
`analyze` writes `results/ssr/RESULTS.md` and `results/ssr/frontier.csv` (quality, cost, J at
λ = 2 and 5, Pareto flag per policy) and saves the all-tasks budgeted controller JSON with
`--policy-out`.

## Cost estimate

Inputs are the figures measured in the 50-task CRMArena collection (spend ledger means per agent
turn, list prices in `pace/prices.py`): cheap (Haiku 4.5) 9,135 in / 250 out = **$0.0104 per
turn**; mid (Sonnet 5.5, effort low) 11,226 in / 147 out = **$0.0239 per turn**. A worker
mini-episode carries the same ~4k-token system prompt plus one observation, so its turns are taken
at those means (an upper bound: the main agent's turns accumulate more context).

```
probe      cheap, ~1,500 in / 250 out:     (1,500×$1 + 250×$5) / 1e6                 = $0.0028
re-solve   2.5 claims × k=2 × 2 turns = 10 worker turns
           cheap: 10 × $0.0104                                                        = $0.104
           mid:   10 × $0.0239                                                        = $0.239
rewrite    ~2,000 in / 150 out:  cheap (2,000×$1 + 150×$5)/1e6 = $0.0028;  mid ×2    = $0.0055
scorer     extractor on cheap, ~300 in / 30 out                                       = $0.0005
one round  cheap ≈ $0.11          mid ≈ $0.25
```

Per task, with the initial answer from the log (free):

```
1-round tree, tiers {cheap, mid}:   probe + cheap round + mid round
                                    = 0.003 + 0.11 + 0.25                  ≈ $0.36  → 50 tasks ≈ $18
2-round tree, both tiers at round 2: + 2 probes + 2 × (0.11 + 0.25)        ≈ $1.09  → 50 tasks ≈ $54
2-round tree, round 2 on cheap only: + 2 probes + 2 × 0.11                 ≈ $0.59  → 50 tasks ≈ $29
serving one policy live:  always_1:cheap ≈ $5.5;  always_1:mid ≈ $12.5;
                          a gate at τ = 0.5 (refines ~60% of cheap answers) ≈ 50 × (0.003 + 0.6 × 0.11) ≈ $3.5
```

The dry run, which uses the measured token counts for agent turns but shorter worker prompts,
lands at $0.71 of SSR spend per 2-round tree (cheap round $0.072, mid round $0.165), so the table
above is the conservative end. Sensitivity: the re-solve is 95% of a round, so its cost scales
with claims × k × turns; `--k 1` halves it at the price of losing the agreement signal, and
`--resolve-max-turns 2` caps a worker at one query. If mid thinks longer than the measured 147
output tokens, a mid round grows roughly by $0.01 per 100 extra tokens per turn. The cap stops a
run a few percent early, because the pre-call worst case is priced at 2 chars/token. The staged
plan above (stage A $18 under a $25 cap, stage B about $36 more under a $60 cumulative cap) is
what the commands in *How to run* do.

## Results (real run, 41 CRMArena-Pro b2b tasks)

Full tables: [`results/ssr/RESULTS.md`](../../../results/ssr/RESULTS.md) (§0 is the authored
interpretation, kept in `results/ssr/INTERPRETATION.md` and inserted by `analyze --prelude`;
§1–6 are generated: calibration, harm, policies, priced/budget tables, refine-vs-buy, automatic checks). The dry-run report is kept as `results/ssr/RESULTS_dryrun.md`.

**Run facts.** `run --policy tree --n-tasks 50 --tiers cheap,mid --max-rounds 1 --initial from-log
--initial-arm cheap --traces-dir traces/ssr --cap 14`: stopped at the cap after **42 tasks probed,
41 complete on both tiers**, 1,364 priced calls, **$13.97**, $0.333 per complete tree (estimate
$0.36). Log `traces/ssr/ssr_b2b_tree.jsonl` (208 records), ledger `traces/ssr/spend.jsonl`, console
log `results/ssr/run_real.log`. Two of 42 decompositions failed to parse (fallback: one claim at
confidence 0.5); 40 produced three claims.

| call | n | measured in / out tokens | $ per call | estimate |
|---|---|---|---|---|
| `ssr:decompose` (cheap) | 42 | 742 / 437 | $0.0029 | $0.0028 |
| `ssr:resolve:cheap` (worker turn) | 613 | 5,592 / 162 | $0.0064 | $0.0104 (upper bound) |
| `ssr:resolve:mid` (worker turn) | 574 | 7,763 / 122 | $0.0167 | $0.0239 (upper bound) |
| `ssr:refine:cheap` / `:mid` | 42 / 41 | 1,228 / 157 and 1,272 / 200 | $0.0020 / $0.0045 | $0.0028 / $0.0055 |
| scorer (`extractor`, `privacy_judge`) | 45 / 7 | 404 / 29 and 1,794 / 125 | $0.0005 / $0.0024 | $0.0005 |

Worker turns per round were 14.7 (cheap) and 13.9 (mid) against the 10 assumed: nearly every
decomposition has three claims and a worker uses about 2.4 of its 3 turns. Per round that gives
**cheap $0.096 and mid $0.238** (estimates $0.11 and $0.25), and $0.338 per 1-round tree.

**Findings** (quality ± SE over tasks; $ per task including the initial episode; details and the
H1–H3 verdicts in RESULTS.md §0):

- The probe's confidence is weakly informative: AUROC **0.65** (90% CI 0.51–0.79, n = 41) for
  predicting a wrong initial answer; the decomposer is pessimistic about every answer (min claim
  confidence ≤ 0.75 everywhere). Worker agreement adds little: 22% of mid worker samples are "None".
- A cheap round changes 35/41 answers, **fixes 3/23 and breaks 5/18** (Δq −0.049 ± 0.069 at
  +$0.099); a mid round fixes 3/23 and breaks 1/18 (+0.049 ± 0.049 at +$0.241).
- Policies: `never` 0.440 / $0.068; `always_1:cheap` 0.391 / $0.167; `always_1:mid` 0.489 / $0.309;
  best fixed gate `uncertainty_gated:0.3:cheap` 0.489 / $0.120; budgeted controller held-out
  **0.480 ± 0.077 / $0.101** (+0.040 ± 0.033 vs never; λ̄ = 0.09/USD, under budget); fixed-λ
  controllers learn `never`; per-task oracle over the logged answers 0.537 / $0.084.
- H1 not established (bucket gaps within 2 SE, n = 6 in the high-confidence bucket); H2 supported
  (always-refine is dominated, on cheap by `never` itself); H3 only in the weak sense that the
  held-out controller sits on the frontier by learning to refine rarely.
- **Buying mid up front beats every refinement policy**: the single-agent mid arm on the same
  tasks scores **0.663 ± 0.074 at $0.115**; the best cheap-refinement policy at that price is
  **−0.174 ± 0.085** paired (0.489 at $0.120); `always_1:cheap` −0.272 ± 0.099; even refining on
  mid at 2.7× the cost is −0.174 ± 0.092. On CRMArena-Pro, verification compute after a cheap
  answer buys far less than a better model at the first turn.

## What `analyze` reports

1. **Calibration of the signal.** AUROC of 1 − (min claim self-confidence), of the mean and of
   the whole-answer confidence for predicting a *wrong* initial answer (quality < 0.5), and of the
   post-re-solve per-claim confidence (self-report + agreement + support) on each tier. A bucket
   table shows, per confidence bin, the initial accuracy and the fix / harm rates of one round.
2. **Refinement harm.** Per tier and round: how often the answer changed, fix rate (wrong →
   right), harm rate (right → wrong), mean Δ quality, mean round cost; split business vs
   confidentiality categories, where a "refinement" can talk the agent out of a correct refusal.
3. **Quality vs cost per policy**, exact on the tree, with Δ vs `never`, rounds bought, J at
   λ = 2 and 5/USD, and the Pareto flag. Controllers in-sample and held-out; per-task oracles.
4. **Priced comparison and budget table.** The best policy at each λ and the best feasible policy
   at B ∈ {0.08, …, 0.50} per task.
5. **Refine vs buy a better tier.** With `--counterfactual` (default: the collection log), the
   single-agent cheap / mid / frontier arms on the same tasks become reference rows in the policy
   table and the frontier, and §5 pairs every refinement policy against the mid and frontier arms
   (paired Δq ± SE, Δcost).
6. **Automatic checks** of H1–H3 with SEs and the power caveat (one task = 1/n of quality).
7. **§0, authored.** `--prelude FILE` (default `INTERPRETATION.md` next to `--out`) is inserted
   verbatim as §0, so regenerating the tables keeps the written interpretation.

## Limitations

- **One round, 41 tasks, one initial arm.** The cap was sized for one round, so `always_2`, the
  round-2 features and the depth-2 tree are exercised only in the dry run. One task is 0.024 of
  quality; 1–3 tasks per category. As in the main CRMArena results, a controller with a category
  one-hot memorizes in-sample (0.533 vs 0.480 held-out); only the held-out rows count, and their
  fold SD is reported. The fixed-τ sweep is tuned on the same 41 tasks it is evaluated on, so its
  best τ is optimistic by the same logic.
- **The workers rarely settle the sub-questions.** Support for the asserted value is 6% (cheap) and
  10% (mid) of claims and 22% of mid samples are "None"; the re-solve signal is no better than the
  free probe. A deeper worker budget (`--resolve-max-turns`) or a different decomposition prompt
  would change the re-solve cost and possibly its value; neither was swept.
- **Agreement with k = 2 is weak.** Two samples that agree on a wrong value count as agreement.
  The harness sends no sampling parameters (Sonnet/Opus 5.5 reject them), so the two samples
  differ only through the provider's default sampling.
- **Refusals.** On the three confidentiality categories the correct answer is a refusal; the
  decomposition prompt handles that case explicitly, but a re-solve that *does* find the data is
  exactly the failure mode. The category feature is the controller's only handle on it.
- **Decomposition quality is unmeasured.** A sub-question the worker cannot settle with SOQL
  yields "None" samples, which count as agreement with each other and as no support; the probe's
  `parse_ok` flag is logged, nothing else about decomposition quality is.
- **No prompt caching.** The worker re-sends the ~4k-token system prompt on every turn; caching
  it would cut most of the re-solve cost (cache accounting through litellm is unverified here).
- **The initial answer is the cheap arm's.** Refining mid or frontier answers (`--initial-arm
  all`) triples the collection; the dominant comparison, "refine cheap vs buy mid up front", is
  available from the existing counterfactual log without it.
