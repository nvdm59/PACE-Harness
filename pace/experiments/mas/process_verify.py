"""Outcome-only vs process-verified rewards, and the silent-failure analysis. Offline: reads the
process log in `Episode.meta["process"]`; makes no calls.

Rewards (all per episode, q = the harness's outcome quality in [0, 1]):

  R_out   = q                                   outcome only (what CRMArena scores)
  R_mult  = q x coverage                        coverage = (# worker steps that were verified AND
                                                passed) / (# worker steps). Unverified steps count
                                                as not covered, so skipping verification is not
                                                rewarded; a correct answer reached through a
                                                process with open failures is discounted.
  R_add   = q + alpha x coverage - beta x open_fail_rate      (alpha = beta = 0.25 by default)
                                                open_fail_rate = (# verifier fails whose sub-query
                                                was never subsequently passed) / (# worker steps).
                                                Additive, so a wrong answer with a clean process
                                                still earns something, and a correct answer with
                                                an open failure is penalised but not zeroed.

Silent-failure cells (threshold: "correct" iff q >= CORRECT_THRESHOLD; CRMArena's exact-match
metrics are 0/1, its fuzzy metrics are an F1):

  correct & >= 1 verifier fail     the process flagged a step, the outcome was right anyway:
                                   either the retry fixed it (recovered) or the flag was a false
                                   alarm / the answer did not need that step (open)
  wrong & all verifiers passed     the SILENT FAILURE: verification saw nothing wrong, the answer
                                   is wrong (what process verification misses)
  wrong & >= 1 verifier fail       caught: the process flagged a step and the answer is wrong
                                   (what outcome-only scoring would attribute to nothing)
  correct / wrong & unverified     no verifier ran (never_verify, or a controller that skipped)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ...trace import Episode

CORRECT_THRESHOLD = 0.5
ALPHA, BETA = 0.25, 0.25


def process(ep: Episode) -> list[dict]:
    return list((ep.meta or {}).get("process") or [])


def is_correct(ep: Episode, threshold: float = CORRECT_THRESHOLD) -> bool:
    return float(ep.quality) >= threshold


def process_stats(ep: Episode) -> dict:
    """Counts over the process log: calls per role, verifier verdicts, coverage, open failures,
    tiers used, cost split."""
    calls = process(ep)
    workers = [c for c in calls if c["role"] == "worker"]
    verifs = [c for c in calls if c["role"] == "verifier"]
    n_pass = sum(1 for c in verifs if c["verdict"] == "pass")
    n_fail = len(verifs) - n_pass
    # per sub-query: did a verifier pass ever follow the last fail?
    last_verdict_by_sub: dict = {}
    for c in verifs:
        last_verdict_by_sub[c.get("sub_query")] = c["verdict"]
    open_fails = sum(1 for v in last_verdict_by_sub.values() if v == "fail")
    covered_subs = {s for s, v in last_verdict_by_sub.items() if v == "pass"}
    worker_subs = [c.get("sub_query") for c in workers]
    n_worker = len(workers)
    covered = sum(1 for s in set(worker_subs) if s in covered_subs)
    n_subs = max(len(set(worker_subs)), 1)
    tiers = Counter(c["tier"] for c in calls if c["role"] in ("worker", "answer"))
    cost_by_role = Counter()
    for c in calls:
        cost_by_role[c["role"]] += float(c["cost"])
    return {"n_calls": len(calls), "n_worker": n_worker, "n_verify": len(verifs), "n_pass": n_pass, "n_fail": n_fail,
            "pass_rate": n_pass / len(verifs) if verifs else float("nan"),
            "coverage": covered / n_subs, "open_fails": open_fails, "open_fail_rate": open_fails / n_subs,
            "n_recovered": sum(1 for c in verifs if c["verdict"] == "fail") - open_fails,
            "n_replan": sum(1 for c in calls if c["role"] == "planner" and c["verdict"] == "replan"),
            "n_retry": sum(1 for c in workers if c.get("retry")),
            "n_rows": sum(1 for c in workers if c["verdict"] == "rows"),
            "n_empty": sum(1 for c in workers if c["verdict"] == "empty"),
            "n_error": sum(1 for c in workers if c["verdict"] in ("error", "invalid")),
            "tiers": dict(tiers), "cost_by_role": {k: round(v, 6) for k, v in cost_by_role.items()},
            "cost": float(ep.total_cost), "eval_cost": float((ep.meta or {}).get("eval_cost_usd", 0.0))}


# ------------------------------------------------------------------- rewards
def outcome_reward(ep: Episode) -> float:
    return float(ep.quality)


def process_reward_mult(ep: Episode) -> float:
    return float(ep.quality) * process_stats(ep)["coverage"]


def process_reward_add(ep: Episode, alpha: float = ALPHA, beta: float = BETA) -> float:
    s = process_stats(ep)
    return float(ep.quality) + alpha * s["coverage"] - beta * s["open_fail_rate"]


def priced(reward: float, cost: float, lam: float) -> float:
    return reward - lam * cost


# ----------------------------------------------------------- silent failures
CELLS = ["correct_all_pass", "correct_some_fail_recovered", "correct_some_fail_open", "correct_unverified",
         "wrong_all_pass", "wrong_some_fail", "wrong_unverified"]


def classify(ep: Episode, threshold: float = CORRECT_THRESHOLD) -> str:
    s = process_stats(ep)
    ok = is_correct(ep, threshold)
    if s["n_verify"] == 0:
        return "correct_unverified" if ok else "wrong_unverified"
    if s["n_fail"] == 0:
        return "correct_all_pass" if ok else "wrong_all_pass"
    if not ok:
        return "wrong_some_fail"
    return "correct_some_fail_open" if s["open_fails"] > 0 else "correct_some_fail_recovered"


def silent_failure_analysis(episodes: list[Episode], threshold: float = CORRECT_THRESHOLD) -> dict:
    """Rates of what process verification catches that the outcome misses, and vice versa."""
    n = len(episodes)
    cells = Counter(classify(e, threshold) for e in episodes)
    verified = [e for e in episodes if process_stats(e)["n_verify"] > 0]
    nv = len(verified)
    wrong_v = sum(1 for e in verified if not is_correct(e, threshold))
    correct_v = nv - wrong_v
    silent_fail = cells["wrong_all_pass"]
    caught = cells["wrong_some_fail"]
    correct_flagged = cells["correct_some_fail_recovered"] + cells["correct_some_fail_open"]
    stats = [process_stats(e) for e in episodes]
    nver = sum(s["n_verify"] for s in stats)
    npass = sum(s["n_pass"] for s in stats)
    return {"n": n, "n_verified_episodes": nv, "cells": {c: cells.get(c, 0) for c in CELLS},
            "accuracy": float(np.mean([is_correct(e, threshold) for e in episodes])) if n else float("nan"),
            "verifier_calls": nver, "verifier_pass_rate": npass / nver if nver else float("nan"),
            # among verified episodes
            "silent_failure_rate": silent_fail / nv if nv else float("nan"),          # wrong & all pass
            "caught_rate": caught / nv if nv else float("nan"),                       # wrong & >=1 fail
            "correct_despite_fail_rate": correct_flagged / nv if nv else float("nan"),
            "correct_despite_open_fail_rate": cells["correct_some_fail_open"] / nv if nv else float("nan"),
            # conditional views: P(flagged | wrong) is the verifier's recall on wrong answers;
            # P(wrong | all pass) is the residual error a clean process still carries
            "p_flagged_given_wrong": caught / wrong_v if wrong_v else float("nan"),
            "p_wrong_given_all_pass": silent_fail / (silent_fail + cells["correct_all_pass"])
            if (silent_fail + cells["correct_all_pass"]) else float("nan"),
            "p_flagged_given_correct": correct_flagged / correct_v if correct_v else float("nan"),
            "coverage_mean": float(np.mean([s["coverage"] for s in stats])) if n else float("nan"),
            "R_out": float(np.mean([outcome_reward(e) for e in episodes])) if n else float("nan"),
            "R_mult": float(np.mean([process_reward_mult(e) for e in episodes])) if n else float("nan"),
            "R_add": float(np.mean([process_reward_add(e) for e in episodes])) if n else float("nan")}


def fmt(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{d}f}"


def silent_failure_table(by_mode: dict[str, list[Episode]]) -> str:
    lines = ["| mode | n | verified eps | verifier calls | pass rate | silent failure: wrong & all pass | "
             "caught: wrong & ≥1 fail | correct & ≥1 fail (open / recovered) | P(flag ∣ wrong) | P(wrong ∣ all pass) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, eps in by_mode.items():
        a = silent_failure_analysis(eps)
        c = a["cells"]
        lines.append(f"| {mode} | {a['n']} | {a['n_verified_episodes']} | {a['verifier_calls']} | "
                     f"{fmt(a['verifier_pass_rate'])} | {c['wrong_all_pass']} ({fmt(a['silent_failure_rate'])}) | "
                     f"{c['wrong_some_fail']} ({fmt(a['caught_rate'])}) | "
                     f"{c['correct_some_fail_open']} / {c['correct_some_fail_recovered']} ({fmt(a['correct_despite_fail_rate'])}) | "
                     f"{fmt(a['p_flagged_given_wrong'])} | {fmt(a['p_wrong_given_all_pass'])} |")
    return "\n".join(lines)
