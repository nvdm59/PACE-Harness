"""τ-bench (retail / airline) collection and offline analysis.

  python -m pace.experiments.taubench.run collect --domain retail --n-tasks 10 --tiers all \
      --mode counterfactual --seed 0 --cap 8 --traces-dir traces/taubench \
      --out traces/taubench_retail_counterfactual.jsonl
  python -m pace.experiments.taubench.run analyze \
      --log traces/taubench_retail_counterfactual.jsonl [--log traces/taubench_airline_counterfactual.jsonl] \
      --out results/taubench/RESULTS.md

collect: paid API calls behind a hard cap on the `--traces-dir` ledger (cumulative across runs).
Seeded task order over the domain's test-split task ids; resumable per (task, tier); exit 0 done,
3 stopped at the cap (STOPPED_AT_CAP written in --traces-dir), 4 a model call failed after retries.

analyze: offline, CPU only. Reuses scripts/crmarena_offline.py's CV tables and DR gate (imported,
not copied) on the τ-bench log; τ-bench's features are decoded by category = org = domain.
Metric: pass^1 (one trial per task per tier); τ-bench's pass^k for k > 1 is not estimated.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...collect import (EXIT_LLM_FAILED, EXIT_STOPPED_AT_CAP, FixedArmPolicy, done_keys, load_policy, parse_tiers,
                        run_episode)
from ...harness.crmarena import LLMCallFailed
from ...harness.taubench import DOMAINS, MAX_TURNS, TURN0_IDX, TauBenchHarness, crm_layout
from ...prices import TIERS
from ...spend import SpendCapReached, SpendLedger
from ...trace import append_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ collect
def collect_taubench(harness: TauBenchHarness, out: str | Path, n_tasks: int, mode: str = "counterfactual",
                     tiers: list[str] | None = None, policy=None, eps: float | None = None, seed: int = 0,
                     summary_every: int = 5, log=print) -> int:
    ledger = harness.ledger
    tiers = list(tiers or TIERS)
    if mode == "counterfactual":
        arms = [(t, FixedArmPolicy(t, harness.all_actions, len(tiers), TURN0_IDX)) for t in tiers]
    elif mode == "policy":
        if policy is None:
            raise ValueError("policy mode needs a controller")
        arms = [("policy", policy)]
    else:
        raise ValueError(mode)
    done = done_keys(out)
    ledger.clear_stop_marker()
    if ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd:
        p = ledger.write_stop_marker()
        log(f"[taubench] ledger already at ${ledger.total:.4f} >= cap ${ledger.cap.cap_usd:g}; nothing run ({p})")
        return EXIT_STOPPED_AT_CAP
    rng = random.Random(seed + 7919)
    n_new = n_skipped = 0
    for i, task in enumerate(harness.sample_tasks(n_tasks, seed), 1):
        for arm, pol in arms:
            if (task.task_id, arm) in done:
                n_skipped += 1
                continue
            try:
                ep = run_episode(harness, pol, task, rng, log_counterfactual=False, log_state=False, eps=eps)
            except SpendCapReached as e:
                p = ledger.write_stop_marker(e)
                log(f"[taubench] STOPPED AT CAP before/in task {task.task_id} arm {arm}: {e}")
                log(ledger.summary())
                log(f"[taubench] wrote {p}; {n_new} new episodes -> {out}. Resume with a higher --cap.")
                return EXIT_STOPPED_AT_CAP
            except LLMCallFailed as e:
                ledger.write_total()
                log(f"[taubench] ABORTED on task {task.task_id} arm {arm}: {e}")
                log(ledger.summary())
                return EXIT_LLM_FAILED
            ep.meta = dict(ep.meta or {})
            ep.meta.update({"arm": arm, "mode": mode, "seed": seed,
                            "behavior": ("uniform_tier_continue_until_answer" if mode == "counterfactual"
                                         else policy.meta.get("name", "controller")),
                            "eps": eps if mode == "policy" else None})
            append_jsonl(ep, out)
            n_new += 1
            log(f"[taubench] {task.task_id:12s} arm={arm:8s} r={ep.quality:.0f} cost=${ep.total_cost:.4f} "
                f"user=${ep.meta['user_cost_usd']:.4f} turns={len(ep.steps)} ({ep.meta.get('end_reason')})")
        if summary_every and i % summary_every == 0:
            log(ledger.summary())
    log(f"[taubench] done: {n_new} new episodes, {n_skipped} already logged -> {out}")
    log(ledger.summary())
    return 0


def _collect(a) -> int:
    if a.cap is None and not a.no_cap:
        raise SystemExit("collect makes paid API calls: pass --cap USD (or --no-cap explicitly)")
    tiers = parse_tiers(a.tiers)
    try:
        from dotenv import load_dotenv
        load_dotenv(a.env_file)
    except ImportError:
        pass
    ledger = SpendLedger(a.traces_dir, None if a.no_cap else a.cap)
    at_cap = ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd
    if not os.environ.get("ANTHROPIC_API_KEY") and not at_cap:
        raise SystemExit(f"ANTHROPIC_API_KEY is not set (looked in the environment and {a.env_file})")
    h = TauBenchHarness(domain=a.domain, ledger=ledger, max_turns=a.max_turns, max_tokens=a.max_tokens,
                        answer_max_tokens=a.answer_max_tokens, user_max_tokens=a.user_max_tokens,
                        timeout=a.timeout, effort=None if a.effort == "none" else a.effort)
    out = a.out or str(Path("traces") / f"taubench_{a.domain}_{a.mode}.jsonl")
    pol = None
    if a.mode == "policy":
        pol = load_policy(a.policy, h, a.eps if a.eps is not None else 0.2)
    print(f"[taubench] domain={a.domain} mode={a.mode} tiers={tiers} n_tasks={a.n_tasks} "
          f"cap={'none' if a.no_cap else f'${a.cap:g}'} ledger=${ledger.total:.4f} user_sim=cheap -> {out}")
    return collect_taubench(h, out, a.n_tasks, a.mode, tiers, pol, pol.eps if pol else None, a.seed)


# ------------------------------------------------------------------ analyze
@contextlib.contextmanager
def tau_layout():
    """pace.features.crm_context decodes (category, org) from a feature vector through
    pace.features._crm_layout, and pace.gate routes only crmarena logs to the bandit estimators;
    point both at the τ-bench log (category = org = domain) while the CRM analysis machinery runs."""
    import pace.features as F
    import pace.gate as G
    saved, saved_bandit = F._crm_layout, G.is_bandit_log
    F._crm_layout = crm_layout
    # the τ-bench counterfactual log is the same first-step contextual bandit as the CRM log
    G.is_bandit_log = lambda eps: bool(eps) and eps[0].harness_version.startswith(("crmarena", "taubench"))
    try:
        yield
    finally:
        F._crm_layout, G.is_bandit_log = saved, saved_bandit


def _offline():
    sys.path.insert(0, str(ROOT / "scripts"))
    import crmarena_offline as off   # noqa: E402  (scripts/ is not a package)
    return off


def arm_summary(eps) -> list[str]:
    by: dict[str, list] = defaultdict(list)
    for e in eps:
        by[e.meta["arm"]].append(e)
    lines = ["| tier (agent model) | episodes | pass^1 | pass^1 (runner-strict) | agent cost/task (USD) | "
             "user-sim cost/task (USD, cheap tier) | mean agent steps | end: user stop / terminate tool / max steps |",
             "|---|---|---|---|---|---|---|---|"]
    from ...prices import MODELS
    for t in TIERS:
        es = by.get(t, [])
        if not es:
            continue
        ends = defaultdict(int)
        for e in es:
            ends[e.meta.get("end_reason")] += 1
        lines.append(f"| {t} ({MODELS[t]}) | {len(es)} | {np.mean([e.quality for e in es]):.3f} | "
                     f"{np.mean([e.meta.get('reward_strict') or 0.0 for e in es]):.3f} | "
                     f"{np.mean([e.total_cost for e in es]):.4f} | {np.mean([e.meta['user_cost_usd'] for e in es]):.4f} | "
                     f"{np.mean([len(e.steps) for e in es]):.1f} | {ends['user_stop']} / {ends['terminate_tool']} / "
                     f"{ends['max_turns'] + ends['forced_end']} |")
    return lines


def analyze_log(eps, title: str, budget: float | None, redraws: int, boot: int) -> tuple[list[str], dict]:
    off = _offline()
    from pace.lookup import LookupTable, kfold_tasks, uniform
    with tau_layout():
        T = LookupTable.from_episodes(eps)
        n = len(T)
        full = [e for e in eps if e.task_id in set(T.task_ids)]
        b = float(budget) if budget is not None else T.value(uniform(n))[1]
        off.BUDGET, off.TOL = b, round(0.05 * b, 4)
        k = min(5, n)
        folds = kfold_tasks(T, k, seed=0)
        assert all(not (set(tr) & set(te)) for tr, te in folds)
        q_or, c_or = T.value(T.oracle())
        q_or2, c_or2 = T.value(T.oracle(2.0))
        lines = [f"## {title}: {n} tasks × 3 tiers ({len(full)} episodes)", "",
                 "### Fixed tiers (pass^1, exact)", ""] + arm_summary(full) + [""]
        lines += ["### Per-task oracle (exact, all arms logged)", "",
                  "| oracle | pass^1 | agent cost/task (USD) |", "|---|---|---|",
                  f"| max reward, cheapest among ties | {q_or:.3f} | {c_or:.4f} |",
                  f"| argmax reward − 2·cost | {q_or2:.3f} | {c_or2:.4f} |",
                  f"| uniform behavior | {T.value(uniform(n))[0]:.3f} | {T.value(uniform(n))[1]:.4f} |", "",
                  f"Tasks solved by at least one tier: {int((T.Q.max(1) > 0.5).sum())}/{n}; by all three: "
                  f"{int((T.Q.min(1) > 0.5).sum())}/{n}.", ""]
        lines += [f"### Held-out first-step tier policies ({k}-fold CV over task ids, domain-stratified, seed 0)", "",
                  f"Budget B = ${b:.4f}/task (default: the uniform behavior's mean agent cost), tol ${off.TOL:.4f}. "
                  "Exact held-out lookups (every arm logged). Note: at step 0 the observable features are the "
                  "domain one-hot only (turn 0, no tool calls, cost 0), so within one domain every learned "
                  "first-step policy is context-free.", ""]
        cv, cv_lines, _, _ = off.cv_tables(full, T, folds)
        ntr = n - n // k
        lines += [ln.replace("the 40 training tasks", f"the ~{ntr} training tasks")
                  .replace("its 40 training tasks", f"its ~{ntr} training tasks") for ln in cv_lines]
        lines += ["### DR gate on the realised log (one uniformly drawn tier per task)", ""]
        g, g_lines, _, _ = off.gate_section(full, T, redraws, boot)
        lines += [ln.replace("trained on all 50 tasks", f"trained on all {n} tasks")
                  .replace("on the 50 tasks", f"on the {n} tasks").replace("(50 episodes, seed 0)", f"({n} episodes, seed 0)")
                  .replace("(150 episodes)", f"({3 * n} episodes)").replace("from 50 tasks", f"from {n} tasks")
                  for ln in g_lines]
    return lines, {"n_tasks": n, "budget": b, "cv": cv, "gate": g, "oracle": [q_or, c_or]}


def _analyze(a) -> int:
    eps_all = []
    by_dom: dict[str, list] = defaultdict(list)
    for p in a.log:
        for e in read_jsonl(p):
            eps_all.append(e)
            by_dom[e.meta.get("domain", "?")].append(e)
    lines = ["# τ-bench results (pass^1)", "",
             "Agent = our tier model (litellm, spend ledger); user simulator fixed to the cheap tier (Haiku 4.5) "
             "for every arm, its cost logged separately and excluded from agent cost. Quality = τ-bench's own "
             "`Env.calculate_reward` (0/1). Reported metric is pass^1 (one trial per task per tier); τ-bench's "
             "pass^k (k > 1) is not estimated. Behavior: uniform tier per task, CONTINUE until the episode ends, "
             f"END forced at step {MAX_TURNS} (τ-bench's max_num_steps). Generated by "
             "`python -m pace.experiments.taubench.run analyze`.", ""]
    res = {}
    doms = [d for d in DOMAINS if by_dom.get(d)]
    for d in doms:
        ls, r = analyze_log(by_dom[d], f"τ-bench {d}", a.budget, a.redraws, a.boot)
        lines += ls
        res[d] = r
    if len(doms) > 1:
        ls, r = analyze_log(eps_all, "τ-bench pooled (" + " + ".join(doms) + ")", a.budget, a.redraws, a.boot)
        lines += ls
        res["pooled"] = r
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="τ-bench collection and analysis for PACE.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run τ-bench episodes (paid API calls; spend is capped)")
    c.add_argument("--domain", required=True, choices=DOMAINS)
    c.add_argument("--n-tasks", type=int, required=True)
    c.add_argument("--tiers", default="all")
    c.add_argument("--mode", default="counterfactual", choices=["counterfactual", "policy"])
    c.add_argument("--policy", default="uniform")
    c.add_argument("--eps", type=float, default=None)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--cap", type=float, default=None)
    c.add_argument("--no-cap", action="store_true")
    c.add_argument("--traces-dir", default="traces/taubench")
    c.add_argument("--out", default=None)
    c.add_argument("--max-turns", type=int, default=MAX_TURNS)
    c.add_argument("--max-tokens", type=int, default=2048)
    c.add_argument("--answer-max-tokens", type=int, default=1024)
    c.add_argument("--user-max-tokens", type=int, default=512)
    c.add_argument("--effort", default="low")
    c.add_argument("--timeout", type=float, default=120.0)
    c.add_argument("--env-file", default="third_party/CRMArena/.env")
    z = sub.add_parser("analyze", help="offline tier-policy table, oracle and gate")
    z.add_argument("--log", action="append", required=True)
    z.add_argument("--out", default="results/taubench/RESULTS.md")
    z.add_argument("--budget", type=float, default=None, help="USD/task; default the uniform behavior's cost")
    z.add_argument("--redraws", type=int, default=200)
    z.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)
    sys.exit(_collect(a) if a.cmd == "collect" else _analyze(a))


if __name__ == "__main__":
    main()
