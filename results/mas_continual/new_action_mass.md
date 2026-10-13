# Probability the served policy put on newly introduced actions

Default schedule, seed 0, re-run of the committed configuration into `results/mas_continual/gate_v2/default_seed0_rerun` with `--log-new-action-mass 1` (logging only; nothing the run decides depends on it). The re-run reproduces the committed run exactly: 265 of 265 `days.jsonl` rows (every committed served and held-out field, compared for equality) and 80 of 80 `gates.jsonl` rows (every field but wall-clock timings). The re-run's served records additionally carry fields added to the code after the committed run (`delegate_share`, verifier/retry counters) and the `new_action_mass` field.

For each stage that adds actions, on every logged decision of the stage's days at which at least one newly introduced action was legal: the served policy's probability (ε = 0.1 included) summed over the new legal actions, against the ε-exploration floor ε·|new ∩ legal| / |legal| (what the new actions get from exploration alone); and the mean probability of the single most likely new action against ε/|legal|. `no-ε` is the policy's own softmax mass. The policy serving a stage's first day is the incumbent promoted under the previous stage (with zero-logit rows for the new actions).

| arm | stage (event day) | new actions | stage day | decisions with a new action legal / all | mass on new actions | ε floor | mass / floor | no-ε mass | max single new action | ε/\|legal\| | share of decisions that chose a new action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| continual | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 1 | 283 / 283 | 0.228 | 0.023 | 9.94× | 0.228 | 0.106 | 0.010 | 0.226 |
| continual | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 2 | 278 / 278 | 0.226 | 0.023 | 9.76× | 0.226 | 0.105 | 0.010 | 0.205 |
| continual | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 3 | 268 / 268 | 0.224 | 0.023 | 9.72× | 0.223 | 0.104 | 0.010 | 0.269 |
| continual | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 8 | 354 / 354 | 0.327 | 0.024 | 13.49× | 0.336 | 0.287 | 0.011 | 0.328 |
| continual | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 1 | 205 / 345 | 0.053 | 0.013 | 3.98× | 0.044 | 0.027 | 0.007 | 0.032 |
| continual | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 2 | 189 / 335 | 0.053 | 0.013 | 3.98× | 0.044 | 0.027 | 0.007 | 0.024 |
| continual | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 3 | 198 / 342 | 0.053 | 0.013 | 3.98× | 0.044 | 0.027 | 0.007 | 0.020 |
| continual | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 8 | 232 / 377 | 0.178 | 0.012 | 14.74× | 0.185 | 0.169 | 0.007 | 0.130 |
| continual | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 1 | 330 / 432 | 0.099 | 0.018 | 5.55× | 0.090 | 0.035 | 0.006 | 0.086 |
| continual | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 2 | 322 / 416 | 0.099 | 0.018 | 5.64× | 0.091 | 0.036 | 0.006 | 0.106 |
| continual | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 3 | 330 / 429 | 0.103 | 0.018 | 5.74× | 0.095 | 0.036 | 0.006 | 0.089 |
| continual | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 8 | 311 / 390 | 0.032 | 0.018 | 1.73× | 0.015 | 0.011 | 0.006 | 0.015 |
| continual_awr | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 1 | 203 / 203 | 0.437 | 0.024 | 17.90× | 0.458 | 0.191 | 0.011 | 0.429 |
| continual_awr | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 2 | 208 / 208 | 0.350 | 0.025 | 14.13× | 0.361 | 0.240 | 0.011 | 0.327 |
| continual_awr | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 3 | 200 / 200 | 0.381 | 0.024 | 15.67× | 0.396 | 0.263 | 0.010 | 0.425 |
| continual_awr | g2-agents (day 17) | answer@answer-4, delegate@data-4, verify@verify-2 | 8 | 237 / 237 | 0.279 | 0.025 | 11.28× | 0.282 | 0.171 | 0.011 | 0.266 |
| continual_awr | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 1 | 214 / 333 | 0.549 | 0.011 | 48.33× | 0.597 | 0.369 | 0.007 | 0.381 |
| continual_awr | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 2 | 192 / 303 | 0.544 | 0.011 | 47.97× | 0.592 | 0.362 | 0.007 | 0.327 |
| continual_awr | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 3 | 212 / 335 | 0.550 | 0.011 | 50.53× | 0.599 | 0.379 | 0.007 | 0.382 |
| continual_awr | g3-tools (day 25) | call@aggregate_calc, call@schema_lookup | 8 | 173 / 244 | 0.186 | 0.012 | 15.04× | 0.193 | 0.131 | 0.007 | 0.164 |
| continual_awr | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 1 | 245 / 347 | 0.675 | 0.014 | 47.99× | 0.735 | 0.324 | 0.006 | 0.496 |
| continual_awr | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 2 | 250 / 323 | 0.355 | 0.016 | 22.64× | 0.377 | 0.209 | 0.007 | 0.337 |
| continual_awr | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 3 | 229 / 285 | 0.356 | 0.016 | 22.39× | 0.378 | 0.204 | 0.007 | 0.274 |
| continual_awr | g4-skills (day 33) | engage@date_window, engage@decompose_fine, engage@soql_aggregate | 8 | 216 / 264 | 0.246 | 0.017 | 14.85× | 0.254 | 0.142 | 0.007 | 0.208 |

Rows are the 1st, 2nd, 3rd and 8th (last) day of each stage that introduces actions (g1 is a repricing and adds none).

Per new action, mean served probability on the stage's first day (over decisions where it was legal):

- continual, g2-agents: answer@answer-4 0.106, delegate@data-4 0.046, verify@verify-2 0.104
- continual, g3-tools: call@aggregate_calc 0.027, call@schema_lookup 0.027
- continual, g4-skills: engage@date_window 0.035, engage@decompose_fine 0.034, engage@soql_aggregate 0.033
- continual_awr, g2-agents: answer@answer-4 0.191, delegate@data-4 0.161, verify@verify-2 0.167
- continual_awr, g3-tools: call@aggregate_calc 0.340, call@schema_lookup 0.335
- continual_awr, g4-skills: engage@date_window 0.287, engage@decompose_fine 0.309, engage@soql_aggregate 0.282
