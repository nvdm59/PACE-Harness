Real run on CRMArena-Pro b2b: 50 tasks (2–3 per category, the same 50 task ids as the single-agent
counterfactual log in `traces/crmarena_b2b_counterfactual.jsonl`), planner Haiku 4.5, worker and
answer Sonnet 5.5, verifier Haiku 4.5. 130 episodes: 50 `always_verify`, 50 `never_verify`, 30
`controller:uniform` before the ledger cap stopped the run (`--cap 28.98` on a ledger that already
held $17.98; the MAS spent $10.97, $0.084 per episode). Five findings, in order of importance.

**1. Headline, negative: the planner / worker / verifier decomposition scores below the single
mid-tier agent on the same tasks, at lower cost, and is dominated on the priced frontier.**
`always_verify` reaches accuracy 0.480 at $0.0906 per task and `never_verify` 0.460 at $0.0821,
against the single Sonnet ReAct agent at **0.608 at $0.120** and the single Haiku agent at 0.444 at
$0.068 (section 7). Paired on the 50 tasks, the single mid agent is right on 11 tasks the MAS gets
wrong and wrong on 5 the MAS gets right (net −6 for the MAS). The MAS sits between cheap and mid in
both quality and cost, and J = quality − λ·cost says it is not a good trade at either price: at
λ = 2 per dollar the single mid agent has J 0.368 against 0.299 (`always_verify`) and 0.297
(`never_verify`); at λ = 5 the single cheap agent has 0.104 against 0.027 and 0.051. The per-task
single-agent oracle is at 0.769 / $0.092. This is the compute-matched single-agent comparison that
MAS-Orchestra (arXiv 2601.14652) reports as the baseline a multi-agent system has to beat, and here the decomposition
does not beat it. Three mechanisms are visible in the log: (i) the plan is fixed up front by a
cheap model with no data in hand, while the single ReAct agent on Sonnet chooses each next query
after seeing the previous result; (ii) `never_verify` runs about 2.7 worker turns and a forced
answer, fewer Sonnet turns than the single agent's ~5, so the MAS is not compute-matched upward
either, it simply gives the strong model less room; (iii) under `always_verify` 47 of 50 episodes
hit the 8-role-call cap because 65% of verifier calls fail and each fail spends a retry, so the
plan is truncated and the answer forced. The categories where the MAS loses outright are the ones
that need the agent to notice something in the data (`invalid_config` 0/2 against the single mid
agent's 2/2; `lead_qualification`, `quote_approval`, `sales_cycle_understanding` 0/2 against 1/2
each). The six confidentiality tasks (`privacy_rejection`, a refusal is the correct answer) score
0 for every system here, so the ceiling on these 50 tasks is 0.88.

**2. Verification bought +1 task for +10% cost, and the paired table shows it is noise.** On the
same 50 tasks, verification flipped 6 tasks wrong → right and 5 right → wrong; 18 were right and 21
wrong under both (section 2b). Net +1 task (+2.0 pp, paired Δquality +0.019 ± 0.067 SE) for
+$0.0086 per task. All 11 flipped tasks went through at least one verifier fail and a retry: the
retry is what moves the outcome, in both directions.

**3. The verifier as built is an alarm, not a judge: high recall, no precision.** Its per-call pass
rate is 0.347 (147 calls, 10 of them structural fails on query errors). At the episode level it
flags 24 of the 26 wrong episodes, P(flag | wrong) = 0.92, so the silent-failure rate (wrong with
every check passed) is only 2/50. But it also flags 23 of the 24 correct episodes, P(flag | correct)
= 0.96 (46% of all episodes are correct-with-a-fail; 20 of those 23 fails were never resolved). A
flag therefore carries no information about the outcome: P(wrong | flagged) = 24/47 = 0.51 against
a base rate of 0.52. The step-level check (well-formed, non-empty, consistent with the sub-goal)
does not predict whether the final synthesis will be right, because most sub-results it fails were
not needed for the answer and most it passes do not determine it. What process verification
"catches that outcome scoring misses" is therefore real but not useful as a reward here: the
`R_mult` ranking (0.203 vs 0 by construction) rewards coverage the outcome does not reflect.

**4. The orchestration reward model finds the reverse of the fake backend: process features carry
no signal, the task category carries most of it.** Out of fold over tasks, the logistic ORM on
process features has AUROC 0.469 and a Brier score above the base rate (0.281 vs 0.250), with or
without the verifier features (0.451 without). Adding the 22-way category one-hot lifts it to
AUROC 0.831, Brier 0.165, ECE 0.04; the category alone, with no process feature, gives
AUROC 0.821, Brier 0.166. The CV is over tasks, so this is difficulty that varies by category and
transfers to unseen tasks of the same category (the six confidentiality tasks and `knowledge_qa`
at 0 for every system, `handle_time` and `transfer_count` near 1), not memorisation of task ids.
It is the same structure the single-agent study found when it chose the tier (`results/crmarena/
RESULTS.md`), and it rests on 2–3 training tasks per category, so the per-category estimates are
thin. The consequence for the thesis: on this benchmark the right orchestration signal is
known before the first call (the category), not learned from the trace of the process.

**5. The priced-controller hook is exercised, not evaluated.** `controller:uniform` ran 30
episodes (0.533 at $0.0758, mixed tiers; an exploration log, not a tuned policy, and its 30 tasks
are an easier draw: the single mid agent scores 0.672 on them against 0.608 on all 50). The AWR
policy trained on the ORM-relabelled log changes nothing detectable per decision (SNIPS outcome
0.563 [0.373, 0.701] against the behavior's 0.567; ESS 55 of 64 decisions, 86%); the episode-level
estimate has ESS 7 of 30 (23%) and a CI of [0.24, 0.85]. Nothing can be concluded about the
policy from 30 exploration episodes; serving it on the 50 tasks costs about $4 and is the next
step if the hook is worth pursuing, which finding 4 argues against: with the outcome predicted by
the category rather than the process, a per-decision controller has nothing to react to beyond
the tier choice the single-agent study already covers.
