# MAS results — DRY RUN on the fake backend (a pipeline check; these numbers carry no scientific content)

Logs: results/mas/dryrun/mas_fake_b2b.jsonl. 200 episodes over 50 tasks, backend `fake`, host harness `crmarena-pro-v1`. Role tiers: planner = cheap, worker = mid, verifier = cheap, answer = mid. Cost/task is the sum of the role calls (planner, worker, verifier, answer); the outcome evaluator's extractor call is listed separately. Quality is the host harness's own score (CRMArena's evaluator); accuracy thresholds it at 0.5. R_mult = quality × verified-pass coverage, R_add = quality + 0.25·coverage − 0.25·open-fail rate (see `process_verify.py`).

## 1. Per-mode quality, cost and process rewards

| mode | n | accuracy (q ≥ 0.5) | quality | cost/task (USD) | calls/task | verifier calls/task | pass rate | coverage | R_mult | R_add | J (λ=2) | J (λ=5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_verify | 50 | 0.720 | 0.720 | 0.0605 | 6.36 | 2.18 | 0.661 | 0.830 | 0.617 | 0.885 | 0.599 | 0.417 |
| never_verify | 50 | 0.560 | 0.560 | 0.0492 | 3.80 | 0.00 | – | 0.000 | 0.000 | 0.560 | 0.462 | 0.314 |
| controller:awr_orm_lam2 | 50 | 0.560 | 0.560 | 0.0669 | 5.40 | 0.66 | 0.667 | 0.273 | 0.183 | 0.613 | 0.426 | 0.226 |
| controller:uniform | 50 | 0.660 | 0.660 | 0.0712 | 5.50 | 0.66 | 0.667 | 0.220 | 0.177 | 0.688 | 0.518 | 0.304 |

Cost split per task by role:

| mode | planner | worker | verifier | answer | eval (extractor) | worker tiers used |
|---|---|---|---|---|---|---|
| always_verify | $0.0057 | $0.0329 | $0.0032 | $0.0187 | $0.0006 | mid 100% |
| never_verify | $0.0057 | $0.0262 | $0.0000 | $0.0172 | $0.0006 | mid 100% |
| controller:awr_orm_lam2 | $0.0125 | $0.0344 | $0.0010 | $0.0190 | $0.0006 | cheap 31%, mid 52%, frontier 18% |
| controller:uniform | $0.0123 | $0.0380 | $0.0010 | $0.0199 | $0.0006 | cheap 24%, mid 59%, frontier 17% |

## 2. Silent-failure analysis: what process verification catches that the outcome misses, and vice versa

Rates are over episodes that had at least one verifier call (the never_verify row has none by construction).

| mode | n | verified eps | verifier calls | pass rate | silent failure: wrong & all pass | caught: wrong & ≥1 fail | correct & ≥1 fail (open / recovered) | P(flag ∣ wrong) | P(wrong ∣ all pass) |
|---|---|---|---|---|---|---|---|---|---|
| always_verify | 50 | 50 | 109 | 0.661 | 8 (0.160) | 6 (0.120) | 8 / 11 (0.380) | 0.429 | 0.320 |
| never_verify | 50 | 0 | 0 | – | 0 (–) | 0 (–) | 0 / 0 (–) | – | – |
| controller:awr_orm_lam2 | 50 | 22 | 33 | 0.667 | 3 (0.136) | 4 (0.182) | 3 / 3 (0.273) | 0.571 | 0.250 |
| controller:uniform | 50 | 24 | 33 | 0.667 | 3 (0.125) | 6 (0.250) | 1 / 2 (0.125) | 0.667 | 0.200 |

## 3. Orchestration reward model: process features → P(correct)

Pooled over all modes: 200 episodes over 50 tasks, base rate P(correct) = 0.625. 5-fold CV over tasks (category-stratified; a task's episodes never straddle folds); out-of-fold predictions. Default features: counts of role calls, verifier verdicts and pass rate, open failures, replans and retries, observation kinds, tiers used, cost. The '− verifier features' row drops n_verify/4, n_pass/4, n_fail/4, pass_rate, has_verify, open_fails, n_retry; the '+ category one-hot' row adds the 22-way task category.

| model | features | AUROC | Brier | Brier (base rate) | ECE (5 bins) |
|---|---|---|---|---|---|
| logreg | process features (default) | 0.698 | 0.213 | 0.234 | 0.098 |
| logreg | process − verifier features | 0.691 | 0.213 | 0.234 | 0.100 |
| logreg | process + category one-hot | 0.562 | 0.284 | 0.234 | 0.188 |
| mlp | process features (default) | 0.712 | 0.208 | 0.234 | 0.053 |
| mlp | process − verifier features | 0.707 | 0.208 | 0.234 | 0.051 |
| mlp | process + category one-hot | 0.592 | 0.247 | 0.234 | 0.111 |

Calibration of the default logistic ORM (out-of-fold):

| predicted bin | n | mean predicted | observed correct |
|---|---|---|---|
| [0.0, 0.2) | 7 | 0.116 | 0.429 |
| [0.2, 0.4) | 27 | 0.324 | 0.481 |
| [0.4, 0.6) | 67 | 0.512 | 0.448 |
| [0.6, 0.8) | 46 | 0.712 | 0.804 |
| [0.8, 1.0) | 53 | 0.881 | 0.792 |

Exported `results/mas/dryrun/quality_model.json` (default logistic ORM on all 200 episodes; in-sample AUROC 0.772, Brier 0.184; out-of-fold above). Consumers: `orm.relabel_episodes` → `pace.train_offline.train_awr` / `pace.gate.estimate`.

Largest standardised logistic weights: bias +0.65, open_fails -0.37, n_empty/3 -0.35, n_rows/3 +0.31, answer_tier=frontier +0.26, n_fail/4 +0.21, n_verify/4 +0.20, answer_tier=cheap -0.19.

## 4. Priced comparison: J = quality − λ·cost, outcome vs ORM quality

| mode | quality (outcome) | quality_hat (ORM) | cost/task | J_out (λ=2) | J_orm (λ=2) | J_out (λ=5) | J_orm (λ=5) |
|---|---|---|---|---|---|---|---|
| always_verify | 0.720 | 0.701 | 0.0605 | 0.599 | 0.580 | 0.417 | 0.398 |
| never_verify | 0.560 | 0.571 | 0.0492 | 0.462 | 0.473 | 0.314 | 0.325 |
| controller:awr_orm_lam2 | 0.560 | 0.601 | 0.0669 | 0.426 | 0.467 | 0.226 | 0.266 |
| controller:uniform | 0.660 | 0.628 | 0.0712 | 0.518 | 0.485 | 0.304 | 0.272 |

## 5. Priced controller hook: AWR on the ORM-relabelled exploration log

AWR+IPS on the 100-episode uniform exploration log (253 logged decisions), reward quality_hat − 2·cost; saved to `results/mas/dryrun/policy_awr_orm.json`.

Per-decision estimates treat each logged decision point as a contextual-bandit sample with its own single-step importance ratio (max 72.7 here): the value of changing one decision while the rest of the trajectory stays as logged, scored on quality_hat, on the outcome, and on the cost-to-go from the decision (the policy-independent prefix, planner + first worker, is $0.0181/task on top). The episode-level row is self-normalised IPS of the full policy with each per-step ratio clipped at 5. ESS = (Σw)²/Σw²; an estimate with ESS below 10% of n is flagged.

| policy | estimator | unit | n | ESS | quality_hat [90% CI] | outcome quality [90% CI] | cost [90% CI] |
|---|---|---|---|---|---|---|---|
| uniform behavior | on-policy mean | decision | 253 | 253 | 0.662 | 0.648 | cost-to-go 0.0408 |
| AWR (ORM reward) | per-decision SNIPS | decision | 253 | 67.8 | 0.667 [0.621, 0.722] | 0.671 [0.552, 0.799] | cost-to-go 0.0445 [0.0377, 0.0515] |
| AWR (ORM reward) | per-decision DR (bucket DM) | decision | 253 | 67.8 | 0.733 [0.683, 0.758] | 0.865 [0.723, 0.927] | cost-to-go 0.0444 [0.0389, 0.0471] |
| uniform behavior | on-policy mean | episode | 100 | 100 | 0.614 | 0.610 | total 0.0690 |
| AWR (ORM reward) | episode SN-IPS, step clip 5 | episode | 100 | 16.4 | 0.726 [0.653, 0.795] | 0.653 [0.480, 0.852] | total 0.0984 [0.0887, 0.1063] |
| AWR (ORM reward), served | on-policy mean | decision | 125 | 125 | 0.601 | 0.560 | cost-to-go 0.0402 |
| AWR (ORM reward), served | on-policy mean | episode | 50 | 50 | 0.601 | 0.560 | total 0.0669 |

per-decision: ESS 67.8 of 253 (27%); episode-level: ESS 16.4 of 100 (16%).

Greedy action of the trained policy at the logged decision points: VERIFY@mid 15%, ANSWER@cheap 14%, REPLAN@mid 12%, ANSWER@frontier 10%, ANSWER@mid 9%. 

## 6. Interpretation

- **These numbers come from the deterministic fake backend** (`fake_backend.py`): they show that the pipeline runs end to end, that every role call is priced, and that the tables render. They say nothing about CRM tasks. The generator's latent model is written down in the fake's docstring, so the ORM's AUROC here measures how well a logistic model recovers that generator, not anything about Salesforce data.
- **Outcome vs process.** always_verify reaches accuracy 0.720 at $0.0605/task against never_verify 0.560 at $0.0492/task; the verifier costs $+0.0113/task. Priced: J(λ=2) 0.599 vs 0.462, J(λ=5) 0.417 vs 0.314.
- **Silent failures.** Among always_verify episodes, 0.160 are wrong with every verifier passing (process verification missed them) and 0.120 are wrong with at least one fail (caught); 0.380 are correct despite a fail (0.160 with the fail still open: false alarms or steps the answer did not need). P(flagged | wrong) = 0.429, P(wrong | all pass) = 0.320: the first is what process verification adds over outcome-only scoring, the second is the residual error a clean process still carries.
- **controller:awr_orm_lam2, served on-policy.** accuracy 0.560 at $0.0669/task, J(λ=2) 0.426 against always_verify 0.599 and the uniform exploration log 0.518. On the fake this is a null result: one AWR pass on 100 exploration episodes with an ORM reward does not beat the fixed baselines. Section 5 reports each off-policy estimate with its effective sample size; where the ESS is a small fraction of n the estimate is flagged and the served rows are the ones to read.
- **controller:uniform.** accuracy 0.660 at $0.0712/task: a uniform policy over 12 actions is the exploration log (propensities logged) the ORM and the AWR hook train on, not a candidate.
- **ORM.** Out of fold, the default logistic ORM has AUROC 0.698 and Brier 0.213 against a base-rate Brier of 0.234 (ECE 0.098). Dropping the verifier features moves AUROC to 0.691; adding the category one-hot moves it to 0.562. On the fake backend the verifier verdict is a noisy function of the observation kind, which is already a feature, so the verifier features cannot add information here; the category carries none by construction and only adds 22 dimensions to memorise. Whether verifier verdicts carry information beyond the observation on real CRM data is exactly what the real run measures.
- **What this does not show.** Whether a priced controller over {VERIFY, SKIP_VERIFY, REPLAN, ANSWER} × tier beats the fixed baselines at equal budget needs the real run plus an on-policy evaluation of the trained policy; the per-decision estimates on the exploration log value one changed decision at a time and the clipped episode-level estimate is biased by construction.
