# Skill handbook for the cheap tier: CRMArena-Pro b2b, 50 tasks

Arms on the same 50 tasks (the collector's seeded sample, seed 0): `cheap`, `mid`, `frontier` from `traces/crmarena_b2b_counterfactual.jsonl` (first episode per task and arm), `cheap+skills` = the cheap tier with the leave-one-task-out procedural hints, from `traces/skills_cheap_b2b.jsonl`. Success = quality >= 0.5. Cost/task = agent-call spend of the episode (as in the base log; the CRMArena extractor is excluded for every arm). Paired Δ is per task vs `cheap`, ± its standard error.

## Overall

| arm | quality | ± SE | success | cost/task | Δ quality vs cheap ± SE | tasks up / down |
|---|---|---|---|---|---|---|
| cheap | 0.444 | 0.070 | 22/50 | $0.0680 | — | — |
| mid | 0.608 | 0.069 | 30/50 | $0.1196 | +0.163 ± 0.083 | 15 / 5 |
| frontier | 0.663 | 0.067 | 33/50 | $0.1642 | +0.219 ± 0.077 | 16 / 4 |
| cheap+skills | 0.581 | 0.070 | 29/50 | $0.0475 | +0.137 ± 0.070 | 11 / 4 |

**Flips (success threshold 0.5).** cheap fail → cheap+skills success: 10 (b2b-1755, b2b-1236, b2b-1397, b2b-842, b2b-1708, b2b-2057, b2b-1833, b2b-80, b2b-656, b2b-891); cheap success → cheap+skills fail: 3 (b2b-1975, b2b-995, b2b-1086); net +7.

- tasks served hints: 36; Δ quality vs cheap +0.139 ± 0.090
- tasks with no leave-one-out notes: 14; Δ quality vs cheap +0.131 ± 0.099

**Attribution.** The tasks that got no hints are an in-run control: same prompt, same tier, no hint block. Hinted Δ − unhinted Δ = +0.008 ± 0.134. 2 of the 10 upward flips are on tasks that were served NO hints (b2b-1833, b2b-80). When the difference sits inside its SE, the lift over the logged cheap run is not attributable to the hints. It is consistent with run-to-run variance of the cheap tier, and a resampled cheap control is needed to separate the two.

## Per-task oracle

Per task, the arm with the highest quality, ties broken by lower cost.

| arms | quality | cost/task | picks |
|---|---|---|---|
| cheap, mid, frontier | 0.769 | $0.0916 | cheap 22, mid 18, frontier 10 |
| + cheap+skills | 0.789 | $0.0748 | cheap 12, mid 7, frontier 6, cheap+skills 25 |

## Per category (mean quality)

| category | n | notes served | cheap | cheap+skills | mid | frontier |
|---|---|---|---|---|---|---|
| handle_time | 3 | 3/3 | 0.33 | 0.67 | 0.67 | 1.00 |
| transfer_count | 3 | 3/3 | 0.67 | 1.00 | 0.67 | 1.00 |
| knowledge_qa | 3 | 0/3 | 0.07 | 0.01 | 0.12 | 0.02 |
| best_region_identification | 3 | 3/3 | 0.67 | 0.33 | 0.67 | 1.00 |
| case_routing | 3 | 3/3 | 1.00 | 1.00 | 1.00 | 1.00 |
| named_entity_disambiguation | 3 | 3/3 | 1.00 | 0.67 | 0.67 | 1.00 |
| policy_violation_identification | 2 | 2/2 | 0.00 | 0.50 | 1.00 | 1.00 |
| top_issue_identification | 2 | 2/2 | 1.00 | 1.00 | 1.00 | 1.00 |
| monthly_trend_analysis | 2 | 2/2 | 0.50 | 0.50 | 1.00 | 1.00 |
| sales_amount_understanding | 2 | 1/2 | 0.50 | 1.00 | 0.50 | 0.00 |
| lead_routing | 2 | 2/2 | 0.00 | 1.00 | 1.00 | 1.00 |
| sales_cycle_understanding | 2 | 2/2 | 0.00 | 0.50 | 0.50 | 1.00 |
| conversion_rate_comprehension | 2 | 2/2 | 0.50 | 0.00 | 1.00 | 1.00 |
| wrong_stage_rectification | 2 | 1/2 | 0.50 | 0.50 | 0.50 | 0.50 |
| sales_insight_mining | 2 | 0/2 | 0.00 | 0.00 | 0.00 | 0.03 |
| quote_approval | 2 | 1/2 | 0.50 | 1.00 | 0.50 | 0.00 |
| lead_qualification | 2 | 1/2 | 0.50 | 1.00 | 0.50 | 0.00 |
| activity_priority | 2 | 2/2 | 1.00 | 1.00 | 1.00 | 1.00 |
| invalid_config | 2 | 2/2 | 0.50 | 1.00 | 1.00 | 1.00 |
| private_customer_information | 2 | 1/2 | 0.00 | 0.00 | 0.00 | 0.50 |
| internal_operation_data | 2 | 0/2 | 0.00 | 0.00 | 0.00 | 0.00 |
| confidential_company_knowledge | 2 | 0/2 | 0.00 | 0.00 | 0.00 | 0.00 |

## Cost

- Distillation: 49 writer calls (mid tier), $0.2715 total (ledger: $0.2715).
- Serving: $2.3899 on the serve ledger (skills:agent:cheap $2.2592, skills:agent:cheap:answer $0.1173, skills:extractor $0.0049, skills:privacy_judge $0.0084).
- cheap+skills cost/task $0.0475 vs cheap $0.0680 and mid $0.1196; with distillation amortized over these 50 tasks $0.0530/task.

## Control: a fresh cheap run with no hints (same 50 tasks, same seed; added after review)

| arm | quality | cost/task |
|---|---|---|
| cheap, original run (Oct 9) | 0.444 | $0.0680 |
| cheap, re-run with no hints | 0.561 | $0.0581 |
| cheap + skills | 0.581 | $0.0475 |

Paired: re-run − original +0.117 ± 0.062 (9 up / 3 down); skills − re-run **+0.020 ± 0.067** (7 up / 5 down). The skill handbook's apparent lift is run-to-run variation of the cheap tier, not the hints. The finding that survives is the size of that variation: the same tier on the same 50 tasks moves by ±0.12 between runs a day apart, as large as the gap between tiers — a second reason the 50-task gate refused to certify anything.
