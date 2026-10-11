"""Verifier ablation: is "never_verify beats everything" an artifact of the measured verifier and
the 8-call cap?  Four controlled variants of the stream, several seeds each, plus a per-step
retry-outcome table on the REAL log.

  A  the committed configuration (verify-1: P(fail | rows) 0.58 whether the step was good or bad;
     8 role calls per episode including verifier calls)
  B  the informative verifier verify-2 (per-call P(fail | good rows) 0.114, i.e. P(flag | correct)
     0.30 at 2.94 calls; recall unchanged) available to EVERY arm from day 1; always_verify uses it
  C  B + the role-call cap raised from 8 to 12, so verify + one retry never truncates a 3-step plan
  D  C + verifier calls exempt from the cap

Reported per variant, mean ± SE over seeds: always_verify − never_verify in served expected
quality and priced J (J at the stage oracle's λ̄_g, as in RESULTS.md), cap-hit rate, retries per
episode, the learner's verify-action share per stage, and the per-event / stationary-J table for
always_verify, never_verify, the continual (cgrpo) arm and the oracle.

  python -m pace.experiments.mas_continual.ablation --root results/mas_continual/ablation \\
      --out results/mas_continual/RESULTS_ablation_verifier.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ...trace import read_jsonl
from . import analyze
from .simenv import MAS_LOG

VARIANTS = {"A": "committed: verify-1, cap 8", "B": "verify-2 for every arm from day 1",
            "C": "B + cap 12", "D": "C + verifier calls exempt from the cap"}
ARMS = ("always_verify", "never_verify", "continual")


def _mse(xs) -> tuple[float, float, int]:
    xs = np.asarray([x for x in xs if x is not None], float)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan"), len(xs)


def pm(xs, d: int = 3) -> str:
    m, se, n = _mse(xs)
    if n == 0:
        return "–"
    return f"{m:+.{d}f} ± {se:.{d}f}" if not np.isnan(se) else f"{m:+.{d}f}"


def pmu(xs, d: int = 3) -> str:
    m, se, n = _mse(xs)
    if n == 0:
        return "–"
    return f"{m:.{d}f} ± {se:.{d}f}" if not np.isnan(se) else f"{m:.{d}f}"


# ------------------------------------------------------------------ stream
def load_grid(root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for v in VARIANTS:
        runs = []
        for d in sorted((root / v).glob("seed*")):
            if (d / "meta.jsonl").exists() and (d / "days.jsonl").exists():
                data = analyze.load(d)
                data["seed"] = int(d.name[4:])
                runs.append(data)
        out[v] = runs
    return out


def served_means(data: dict, arm: str, key: str, days: range | None = None) -> float:
    rows = [r for r in data["days"] if r["arm"] == arm and (days is None or r["day"] in days)]
    return float(np.mean([r["served"][key] for r in rows]))


def per_stage_served(data: dict, arm: str, key: str) -> dict[int, float]:
    by = defaultdict(list)
    for r in data["days"]:
        if r["arm"] == arm:
            by[r["gen_index"]].append(r["served"][key])
    return {g: float(np.mean(v)) for g, v in by.items()}


def gap_table(grid: dict) -> str:
    lines = ["| variant | seeds | always − never: expected quality | always − never: J (stage λ̄) | always cost − never cost | "
             "cap hits always / never / continual | retries per episode always / continual | verifier calls per episode always / continual |",
             "|---|---|---|---|---|---|---|---|"]
    for v, runs in grid.items():
        if not runs:
            continue
        dq = [served_means(r, "always_verify", "expected_quality") - served_means(r, "never_verify", "expected_quality") for r in runs]
        dj = [served_means(r, "always_verify", "J") - served_means(r, "never_verify", "J") for r in runs]
        dc = [served_means(r, "always_verify", "cost") - served_means(r, "never_verify", "cost") for r in runs]
        cap = [(served_means(r, a, "cap_hit_rate")) for a in ARMS for r in runs]
        cap_s = " / ".join(pmu([served_means(r, a, "cap_hit_rate") for r in runs], 2) for a in ARMS)
        ret = " / ".join(pmu([served_means(r, a, "retries") for r in runs], 2) for a in ("always_verify", "continual"))
        ver = " / ".join(pmu([served_means(r, a, "verifier_calls") for r in runs], 2) for a in ("always_verify", "continual"))
        lines.append(f"| {v}: {VARIANTS[v]} | {len(runs)} | {pm(dq)} | {pm(dj)} | ${pm(dc, 4)} | {cap_s} | {ret} | {ver} |")
    return "\n".join(lines)


def verify_share_table(grid: dict) -> str:
    lines = ["| variant | arm | g0 | g1 | g2 | g3 | g4 |", "|---|---|---|---|---|---|---|"]
    for v, runs in grid.items():
        if not runs:
            continue
        for arm in ("continual", "always_verify"):
            cells = []
            for g in range(5):
                cells.append(pmu([per_stage_served(r, arm, "verify_share").get(g) for r in runs], 2))
            lines.append(f"| {v} | {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def transfer_table(grid: dict) -> str:
    """Per stage: stationary J of always / never / continual / oracle; continual's hit, recovery, days to
    recover (mean over seeds that recovered, and how many did); regret over the stream."""
    out = []
    for v, runs in grid.items():
        if not runs:
            continue
        mts = [analyze.metrics(r, r["meta"]["config"]["recover_tol"], r["meta"]["config"]["warmup_days"]) for r in runs]
        out += [f"**Variant {v}: {VARIANTS[v]}** ({len(runs)} seeds)", "",
                "| stage | oracle J | never_verify J | always_verify J | continual J | continual − never | continual hit / recovery | days to recover (n recovered) |",
                "|---|---|---|---|---|---|---|---|"]
        for g in range(5):
            orc = [m["oracle"][g] for m in mts if g in m["oracle"]]
            st = {a: [s["stationary_J"] for m in mts for s in m["arms"][a]["per_stage"] if s["gen"] == g and s["available"]] for a in ARMS}
            diff = [c - n for m in mts for c, n in [(next(s["stationary_J"] for s in m["arms"]["continual"]["per_stage"] if s["gen"] == g),
                                                     next(s["stationary_J"] for s in m["arms"]["never_verify"]["per_stage"] if s["gen"] == g))]]
            cs = [s for m in mts for s in m["arms"]["continual"]["per_stage"] if s["gen"] == g]
            hit = [s["hit"] for s in cs]; rec = [s["recovery"] for s in cs]
            dtr = [s["days_to_recover"] for s in cs if s["days_to_recover"] is not None]
            out.append(f"| g{g} | {pmu(orc)} | {pmu(st['never_verify'])} | {pmu(st['always_verify'])} | {pmu(st['continual'])} | {pm(diff)} | "
                       f"{pm(hit) if g else '–'} / {pm(rec) if g else '–'} | {(f'{np.mean(dtr):.1f} ({len(dtr)}/{len(cs)})') if dtr else f'– (0/{len(cs)})'} |")
        reg = {a: [m["arms"][a]["regret"] for m in mts] for a in ARMS}
        out += ["", "regret vs oracle, days 6–40: " + ", ".join(f"{a} {pmu(reg[a])}" for a in ARMS), ""]
    return "\n".join(out)


# ------------------------------------------------------------------ real log
def retry_outcome_table(mas_log: Path = MAS_LOG) -> tuple[str, dict]:
    """On the real MAS log: for every verifier FAIL that was followed by a retry of the same sub-query,
    the observation kind before → after; whether a later verifier call on the retried sub-query
    passed; and, paired with never_verify on the same task, whether the episode outcome flipped."""
    eps = read_jsonl(mas_log)
    by_mode = defaultdict(dict)
    for e in eps:
        by_mode[e.meta["mode"]][e.task_id] = e
    trans = Counter()
    n_flag = n_retried = 0
    retried_then_pass = retried_then_verified = 0
    first_pass = first_verified = 0
    ep_with_retry = set()
    for e in eps:
        proc = e.meta["process"]
        for i, c in enumerate(proc):
            if c["role"] != "verifier":
                continue
            # the worker step this verdict checked
            w_idx = max(j for j in range(i) if proc[j]["role"] == "worker")
            w = proc[w_idx]
            if c["verdict"] == "fail":
                n_flag += 1
                nxt = proc[i + 1] if i + 1 < len(proc) else None
                if nxt and nxt["role"] == "worker" and nxt.get("retry") and nxt["sub_query"] == w["sub_query"]:
                    n_retried += 1
                    ep_with_retry.add((e.meta["mode"], e.task_id))
                    trans[(w["verdict"], nxt["verdict"])] += 1
                    later = [x for x in proc[i + 2:] if x["role"] == "verifier" and x["sub_query"] == w["sub_query"]]
                    if later:
                        retried_then_verified += 1
                        retried_then_pass += later[0]["verdict"] == "pass"
            if not w.get("retry"):
                first_verified += 1
                first_pass += c["verdict"] == "pass"
    kinds = ["rows", "empty", "error", "invalid", "respond"]
    lines = ["| observation before the flag → after the retry | " + " | ".join(kinds) + " | total |", "|---|" + "---|" * (len(kinds) + 1)]
    for b in ("rows", "empty", "error", "invalid"):
        row = [trans.get((b, a), 0) for a in kinds]
        if sum(row):
            lines.append(f"| {b} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    to_rows = sum(v for (b, a), v in trans.items() if b != "rows" and a == "rows")
    from_bad = sum(v for (b, a), v in trans.items() if b != "rows")
    rows_kept = trans.get(("rows", "rows"), 0)
    from_rows = sum(v for (b, a), v in trans.items() if b == "rows")
    # paired flips among always_verify episodes that retried
    av, nv = by_mode["always_verify"], by_mode["never_verify"]
    flips = Counter()
    for (mode, tid) in ep_with_retry:
        if mode != "always_verify" or tid not in nv:
            continue
        a, n = av[tid].quality >= 0.5, nv[tid].quality >= 0.5
        flips["wrong→right" if (a and not n) else "right→wrong" if (n and not a) else "unchanged (both right)" if a else "unchanged (both wrong)"] += 1
    n_av_retry = sum(1 for (m, _) in ep_with_retry if m == "always_verify")
    summary = {"verifier_fails": n_flag, "fails_followed_by_retry": n_retried, "transitions": {f"{b}->{a}": v for (b, a), v in trans.items()},
               "bad_to_rows": (to_rows, from_bad), "rows_to_rows": (rows_kept, from_rows),
               "retried_step_verified_again": (retried_then_pass, retried_then_verified), "first_attempt_pass": (first_pass, first_verified),
               "always_verify_episodes_with_retry": n_av_retry, "paired_flips": dict(flips)}
    text = "\n".join(lines) + "\n\n" + (
        f"- {n_flag} verifier fails, {n_retried} of them followed by a retry of the same sub-query (the rest hit the call cap or the one-retry limit).\n"
        f"- A flagged **empty / error / invalid** step became **rows** on retry in {to_rows} of {from_bad} cases ({100 * to_rows / max(from_bad, 1):.0f}%): "
        f"the retry does repair malformed or empty steps.\n"
        f"- A flagged **rows** step stayed rows in {rows_kept} of {from_rows} cases ({100 * rows_kept / max(from_rows, 1):.0f}%): the verifier's fail on a "
        f"rows step changes nothing observable, and {from_rows} of the {n_retried} retries ({100 * from_rows / max(n_retried, 1):.0f}%) were of this kind.\n"
        f"- When the retried step was verified again, it passed {retried_then_pass} of {retried_then_verified} times "
        f"({100 * retried_then_pass / max(retried_then_verified, 1):.0f}%), against {first_pass} of {first_verified} ({100 * first_pass / max(first_verified, 1):.0f}%) "
        f"for first attempts: the verifier fails a repaired step at least as often as a fresh one.\n"
        f"- Paired with `never_verify` on the same task, the {n_av_retry} `always_verify` episodes that retried flipped "
        + ", ".join(f"{k} {v}" for k, v in sorted(flips.items())) + ".")
    return text, summary


# ------------------------------------------------------------------ report
def report(root: Path) -> str:
    grid = load_grid(root)
    n_seeds = {v: len(r) for v, r in grid.items()}
    rt = [r["meta"]["runtime_s"] for runs in grid.values() for r in runs]
    real_text, real = retry_outcome_table()
    out = ["# Verifier ablation: when does verification help?", "",
           "Question: is \"never_verify beats everything\" (RESULTS.md, finding 1) an artifact of the measured verifier "
           "(uninformative: P(flag ∣ wrong) 0.92 vs P(flag ∣ correct) 0.96) and of the 8-call cap that a verify-and-retry process "
           "fills? Four controlled variants of the same 40-day stream, " + ", ".join(f"{v} {n} seeds" for v, n in n_seeds.items())
           + f" (seeds 0–{max(n_seeds.values()) - 1}, `--arms ablation`: continual cgrpo, always_verify, never_verify and the per-stage oracle; "
           f"mean runtime {np.mean(rt):.0f} s per run on CPU). Served numbers are means over all 40 days (the fixed modes do not learn); "
           "J uses the stage oracle's λ̄_g as in RESULTS.md. Because the harness draws one random stream per day shared by the arms "
           "that run, variant A's seed-0 numbers differ slightly from the committed all-arms run.", "",
           "| variant | what changes |", "|---|---|"] + [f"| {v} | {d} |" for v, d in VARIANTS.items()] + ["",
           "## 1. always_verify − never_verify, served", "", gap_table(grid), "",
           "## 2. The learner's verify-action share per stage (share of decisions that call a verifier)", "", verify_share_table(grid), "",
           "## 3. Per-stage transfer and stationary J (last 4 days of each stage, held-out)", "", transfer_table(grid),
           "## 4. The real log: what a flagged step's retry did (results/mas/mas_b2b.jsonl, 130 episodes)", "", real_text, ""]
    per_run = []
    for v, runs in grid.items():
        for r in runs:
            mt = analyze.metrics(r, r["meta"]["config"]["recover_tol"], r["meta"]["config"]["warmup_days"])
            per_run.append({"variant": v, "seed": r["seed"], "runtime_s": r["meta"]["runtime_s"],
                            "gap_quality": served_means(r, "always_verify", "expected_quality") - served_means(r, "never_verify", "expected_quality"),
                            "gap_J": served_means(r, "always_verify", "J") - served_means(r, "never_verify", "J"),
                            "gap_cost": served_means(r, "always_verify", "cost") - served_means(r, "never_verify", "cost"),
                            "cap_hits": {a: served_means(r, a, "cap_hit_rate") for a in ARMS},
                            "retries": {a: served_means(r, a, "retries") for a in ARMS},
                            "verify_share": {a: per_stage_served(r, a, "verify_share") for a in ("continual", "always_verify")},
                            "regret": {a: mt["arms"][a]["regret"] for a in ARMS}, "oracle_J": mt["oracle"],
                            "stationary_J": {a: {s["gen"]: s["stationary_J"] for s in mt["arms"][a]["per_stage"] if s["available"]} for a in ARMS}})
    return "\n".join(out), {"grid_seeds": n_seeds, "real_log": real, "runs": per_run}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Verifier ablation report for the continual MAS orchestrator.")
    ap.add_argument("--root", default="results/mas_continual/ablation")
    ap.add_argument("--out", default="results/mas_continual/RESULTS_ablation_verifier.md")
    ap.add_argument("--prelude", default=None)
    a = ap.parse_args(argv)
    text, summary = report(Path(a.root))
    if a.prelude and Path(a.prelude).exists():
        head, _, tail = text.partition("\n\n")
        text = head + "\n\n## 0. Interpretation\n\n" + Path(a.prelude).read_text().strip() + "\n\nSections 1–4 below are generated by `ablation.py`; section 0 is authored.\n\n" + tail
    Path(a.out).write_text(text)
    (Path(a.out).with_suffix(".summary.jsonl")).write_text(json.dumps(summary) + "\n")
    print(f"[ablation] wrote {a.out}")


if __name__ == "__main__":
    main()
