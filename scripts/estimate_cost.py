#!/usr/bin/env python3
"""Cost estimator for CRMArena-Pro collection runs, priced from pace/prices.py.

Two modes:

  assumed (default)  arithmetic from per-turn token assumptions; the defaults are
                     the README assumptions derived from CRMArena's prompts and
                     its published GPT-4o b2b run logs (see README,
                     "Running on CRMArena-Pro").
      python scripts/estimate_cost.py --n-tasks 150 --tiers all
      python scripts/estimate_cost.py --n-tasks 150 --turns 6 --in-tokens-per-turn 9000 \\
          --out-tokens-per-turn cheap=200,mid=800,frontier=800

  measured           reads traces/spend.jsonl and extrapolates from what the
                     ledger has actually recorded per task.
      python scripts/estimate_cost.py --measured --n-tasks 500

Per-turn token flags take one number (all tiers) or tier=value pairs.
No network, no API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pace.prices import MODELS, TIERS, call_cost, price_of  # noqa: E402

# README assumptions (per tier, per agent turn). See README for the derivation.
DEFAULT_TURNS = 4.0
DEFAULT_IN = {"cheap": 6500, "mid": 7800, "frontier": 7800}
DEFAULT_OUT = {"cheap": 150, "mid": 550, "frontier": 550}
DEFAULT_EXTRACTOR = (600, 30)      # (in, out) tokens per episode on the cheap tier


def per_tier(spec: str | None, default: dict) -> dict:
    if spec is None:
        return dict(default)
    if "=" not in spec:
        return {t: float(spec) for t in TIERS}
    out = dict(default)
    for part in spec.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in TIERS:
            raise SystemExit(f"unknown tier {k!r}; use {TIERS}")
        out[k] = float(v)
    return out


def assumed(a) -> None:
    tiers = TIERS if a.tiers == "all" else [t.strip() for t in a.tiers.split(",")]
    tin, tout = per_tier(a.in_tokens_per_turn, DEFAULT_IN), per_tier(a.out_tokens_per_turn, DEFAULT_OUT)
    ex_in, ex_out = a.extractor_in_tokens, a.extractor_out_tokens
    print(f"assumed: {a.n_tasks} tasks x {len(tiers)} tier arm(s) x {a.turns:g} turns/episode; "
          f"prices from pace/prices.py (USD per M tokens in/out)")
    total = 0.0
    for t in tiers:
        m = MODELS[t]
        pin, pout = price_of(m)
        tin_ep, tout_ep = tin[t] * a.turns, tout[t] * a.turns
        ep = call_cost(m, tin_ep, tout_ep)
        arm = ep * a.n_tasks
        total += arm
        print(f"  {t:8s} {m:18s} ${pin:g}/${pout:g}: per episode {tin_ep:,.0f} in x ${pin:g} + {tout_ep:,.0f} out x "
              f"${pout:g} = ${ep:.4f}; x {a.n_tasks} = ${arm:,.2f}")
    ex = call_cost(MODELS["cheap"], ex_in, ex_out)
    n_ep = a.n_tasks * len(tiers)
    print(f"  extractor/judge (cheap): {n_ep} episodes x ({ex_in} in, {ex_out} out) = {n_ep} x ${ex:.5f} "
          f"= ${ex * n_ep:,.2f}")
    total += ex * n_ep
    per_task = total / max(a.n_tasks, 1)
    print(f"  TOTAL ${total:,.2f}  (${per_task:.4f} per task across {len(tiers)} arm(s))")
    if a.cap:
        print(f"  a ${a.cap:g} cap buys ~{int(a.cap / per_task)} tasks at this rate")


def measured(a) -> None:
    path = Path(a.ledger)
    if not path.exists():
        raise SystemExit(f"{path} not found; run a capped collection first")
    per_task = defaultdict(float)
    per_model = defaultdict(float)
    per_purpose = defaultdict(float)
    tok = defaultdict(lambda: [0, 0, 0])
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        per_task[str(r.get("task_id"))] += r["usd"]
        per_model[r["model"]] += r["usd"]
        per_purpose[":".join(r.get("purpose", "?").split(":")[:2])] += r["usd"]   # agent:<tier>, extractor, ...
        tk = tok[r["model"]]
        tk[0] += 1; tk[1] += r.get("in_tokens", 0); tk[2] += r.get("out_tokens", 0)
    costs = sorted(per_task.values())
    if not costs:
        raise SystemExit("ledger is empty")
    total = sum(costs)
    mean, med = st.mean(costs), st.median(costs)
    p90 = costs[min(len(costs) - 1, int(0.9 * len(costs)))]
    print(f"measured from {path}: ${total:.4f} over {len(costs)} task ids")
    print(f"  per task: mean ${mean:.4f}  median ${med:.4f}  p90 ${p90:.4f}  max ${costs[-1]:.4f}")
    for m, (n, i, o) in sorted(tok.items()):
        print(f"  {m:18s} {n:5d} calls  mean in {i / n:,.0f}  mean out {o / n:,.0f} tok/call  ${per_model[m]:.4f}")
    for k, v in sorted(per_purpose.items()):
        print(f"  purpose {k:28s} ${v:.4f}")
    print(f"  extrapolated: {a.n_tasks} tasks x ${mean:.4f} = ${a.n_tasks * mean:,.2f}  "
          f"(at p90 per task: ${a.n_tasks * p90:,.2f})")
    print("  note: per-task cost covers whatever arms were run for those tasks; a task cut off by the cap "
          "counts as partial.")
    if a.cap:
        print(f"  a ${a.cap:g} cap buys ~{int(a.cap / mean)} tasks at the measured mean")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-tasks", type=int, default=150)
    ap.add_argument("--tiers", default="all")
    ap.add_argument("--turns", type=float, default=DEFAULT_TURNS)
    ap.add_argument("--in-tokens-per-turn", default=None, help="N or cheap=N,mid=N,frontier=N")
    ap.add_argument("--out-tokens-per-turn", default=None, help="N or cheap=N,mid=N,frontier=N (incl. thinking)")
    ap.add_argument("--extractor-in-tokens", type=int, default=DEFAULT_EXTRACTOR[0])
    ap.add_argument("--extractor-out-tokens", type=int, default=DEFAULT_EXTRACTOR[1])
    ap.add_argument("--cap", type=float, default=50.0, help="report how many tasks this cap buys (0 = skip)")
    ap.add_argument("--measured", action="store_true", help="extrapolate from traces/spend.jsonl instead")
    ap.add_argument("--ledger", default="traces/spend.jsonl")
    a = ap.parse_args(argv)
    (measured if a.measured else assumed)(a)


if __name__ == "__main__":
    main()
