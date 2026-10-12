"""EvoHarnessBench hosted service: capped counterfactual collection and analysis.

  python -m pace.experiments.evoharness.run health
  python -m pace.experiments.evoharness.run collect --axis tools --stage 1 --n-tasks 10 --tiers all \
      --cap 50 --traces-dir traces/evoharness
  python -m pace.experiments.evoharness.run analyze --traces-dir traces/evoharness

collect runs every tier on every task (one fresh service session per (task, tier)), appends
each Episode as it finishes, and skips (task_id, arm) pairs already logged (resume). It
stops with exit 3 at the spend cap (STOPPED_AT_CAP written), 4 when a model call fails after
retries, and 5 when the observed per-task cost projects over the cap for --project-n tasks.
`--dry-run` uses the offline MockBackend and a scripted model (no network, no spend).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from ...collect import EXIT_LLM_FAILED, EXIT_STOPPED_AT_CAP, done_keys, parse_tiers, run_episode
from ...harness.crmarena import LLMCallFailed, PricedLLM
from ...harness.evoharness import AXES, TURN0_IDX, EvoHarness, MockBackend, api_key, health
from ...spend import SpendCapReached, SpendLedger
from ...trace import append_jsonl, iter_jsonl

EXIT_PROJECTED_OVER_CAP = 5
EXIT_NO_SERVICE = 2
EXIT_SERVICE_ERRORS = 6


class TierArm:
    """Counterfactual behavior: tier drawn uniformly per task, then ACT on it (the model picks
    tool / skill / delegate / answer) until it answers; ANSWER is forced on the last turn.
    Logged propensity 1/|tiers| on the first step, 1.0 after."""

    def __init__(self, tier: str, actions: list[str], n_tiers: int):
        self.tier, self.actions, self.p0 = tier, list(actions), 1.0 / n_tiers
        self._idx = {a: i for i, a in enumerate(self.actions)}

    def act(self, x, legal, rng=None, eps=None):
        name = f"ACT@{self.tier}" if f"ACT@{self.tier}" in legal else f"ANSWER@{self.tier}"
        return self._idx[name], (self.p0 if x[TURN0_IDX] > 0.5 else 1.0)


def collect(h: EvoHarness, out: str | Path, n_tasks: int, tiers: list[str], seed: int = 0,
            project_n: int = 60, project_after: int = 3, max_service_errors: int = 10, log=print) -> int:
    ledger = h.ledger
    n_service_err = 0
    arms = [(t, TierArm(t, h.all_actions, len(tiers))) for t in tiers]
    done = done_keys(out)
    ledger.clear_stop_marker()
    cap = ledger.cap.cap_usd
    if cap is not None and ledger.total >= cap:
        log(f"[evo] ledger ${ledger.total:.4f} >= cap ${cap:g}; nothing run ({ledger.write_stop_marker()})")
        return EXIT_STOPPED_AT_CAP
    rng = random.Random(seed + 7919)
    task_cost: list[float] = []
    n_new = 0
    for task in h.sample_tasks(n_tasks, seed):
        c_task, fresh = 0.0, False
        for arm, pol in arms:
            if (task.task_id, arm) in done:
                continue
            try:
                ep = run_episode(h, pol, task, rng, log_counterfactual=False, log_state=False)
            except SpendCapReached as e:
                h.abort()
                log(f"[evo] STOPPED AT CAP before {task.task_id} arm {arm}: {e} ({ledger.write_stop_marker(e)})")
                log(ledger.summary())
                return EXIT_STOPPED_AT_CAP
            except LLMCallFailed as e:
                h.abort()
                ledger.write_total()
                log(f"[evo] ABORTED on {task.task_id} arm {arm}: {e}")
                return EXIT_LLM_FAILED
            except Exception as e:  # noqa: BLE001 -- a service-side failure on one task must not lose the run
                h.abort()
                n_service_err += 1
                log(f"[evo] SERVICE ERROR on {task.task_id} arm {arm} (skipped, rerun resumes it): "
                    f"{type(e).__name__}: {str(e)[:300]}")
                if n_service_err >= max_service_errors:
                    ledger.write_total()
                    log(f"[evo] {n_service_err} service errors; stopping")
                    return EXIT_SERVICE_ERRORS
                continue
            ep.meta = dict(ep.meta or {}, arm=arm, mode="counterfactual", seed=seed,
                           behavior="uniform_tier_act_until_answer")
            append_jsonl(ep, out)
            n_new += 1
            c_task += ep.total_cost
            fresh = True
            log(f"[evo] {task.task_id:28s} arm={arm:8s} score={ep.quality:.3f} pass={ep.meta['pass']} "
                f"cost=${ep.total_cost:.4f} turns={len(ep.steps)} types={ep.meta['types']}")
        if fresh:
            task_cost.append(c_task)
        if project_n and cap is not None and len(task_cost) >= project_after:
            per = sum(task_cost) / len(task_cost)
            if per * project_n > cap:
                ledger.write_total()
                log(f"[evo] STOP: ${per:.4f}/task (all tiers) x {project_n} tasks = ${per * project_n:.2f} "
                    f"> cap ${cap:g}")
                return EXIT_PROJECTED_OVER_CAP
    per = sum(task_cost) / len(task_cost) if task_cost else 0.0
    log(f"[evo] done: {n_new} new episodes -> {out}; ${per:.4f}/task (all tiers) "
        f"-> ${per * project_n:.2f} projected for {project_n} tasks")
    log(ledger.summary())
    return 0


# ------------------------------------------------------------------ analysis
def analyze(paths: list[Path], tiers: list[str]) -> dict:
    """Per (axis, stage): per-tier score / strict pass / cost, the per-task oracle (cheapest
    tier reaching the best score), FWT / BWT cohorts (new vs carried-over capability), and
    the held-out / adaptation task lists the continual-orchestrator protocol consumes."""
    rows: dict[tuple[str, int], dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for ep in iter_jsonl(p):
            m = ep.meta or {}
            if m.get("mode") != "counterfactual":
                continue
            rows[(m["axis"], int(m["stage"]))][ep.task_id][m["arm"]] = {
                "score": ep.quality, "pass": bool(m.get("pass")), "cost": ep.total_cost,
                "split": m.get("split"), "new": bool(m.get("new_capability")), "introduced": m.get("introduced")}
    report: dict[str, dict] = {}
    for (axis, stage), by_task in sorted(rows.items()):
        complete = {t: a for t, a in by_task.items() if all(x in a for x in tiers)}
        per_tier = {}
        for tier in tiers:
            v = [a[tier] for a in complete.values()]
            per_tier[tier] = _agg(v)
            for name, flag in (("new", True), ("carried", False)):
                per_tier[tier][name] = _agg([x for x in v if x["new"] == flag])
        oracle, labels = [], {}
        for t, a in sorted(complete.items()):
            best = max(a[x]["score"] for x in tiers)
            pick = min((x for x in tiers if a[x]["score"] >= best), key=lambda x: a[x]["cost"])
            labels[t] = pick
            oracle.append(a[pick])
        held = sorted(t for t, a in complete.items() if a[tiers[0]]["split"] != "adapt")
        adapt = sorted(t for t, a in complete.items() if a[tiers[0]]["split"] == "adapt")
        overlap = set(held) & set(adapt)
        if overlap:
            raise ValueError(f"held-out / adaptation overlap at {axis} stage {stage}: {sorted(overlap)}")
        report[f"{axis}/H{stage}"] = {
            "axis": axis, "stage": stage, "n_tasks_complete": len(complete), "n_tasks_partial": len(by_task) - len(complete),
            "per_tier": per_tier, "oracle": _agg(oracle), "oracle_tier_counts": _counts(labels.values(), tiers),
            "oracle_labels": labels,
            "protocol": {"heldout": held, "adaptation": adapt,
                         "fwt_cohort": sorted(t for t in held if complete[t][tiers[0]]["new"]),
                         "bwt_cohort": sorted(t for t in held if not complete[t][tiers[0]]["new"])}}
    return report


def _agg(v: list[dict]) -> dict:
    n = len(v)
    if not n:
        return {"n": 0}
    return {"n": n, "score": round(sum(x["score"] for x in v) / n, 4),
            "pass": round(sum(x["pass"] for x in v) / n, 4), "cost": round(sum(x["cost"] for x in v) / n, 6)}


def _counts(vals, tiers) -> dict:
    vals = list(vals)
    return {t: vals.count(t) for t in tiers}


def _print_report(rep: dict, tiers: list[str]) -> None:
    for key, r in rep.items():
        print(f"== {key}: {r['n_tasks_complete']} tasks with all tiers ({r['n_tasks_partial']} partial)")
        for t in tiers + ["oracle"]:
            a = r["oracle"] if t == "oracle" else r["per_tier"][t]
            if a.get("n"):
                print(f"   {t:9s} score {a['score']:.3f}  pass {a['pass']:.3f}  cost ${a['cost']:.4f}")
        print(f"   oracle tiers {r['oracle_tier_counts']}; held-out {len(r['protocol']['heldout'])} "
              f"(FWT {len(r['protocol']['fwt_cohort'])} / BWT {len(r['protocol']['bwt_cohort'])}), "
              f"adaptation {len(r['protocol']['adaptation'])}")


# ------------------------------------------------------------------ dry run
def scripted_completion(**kw):
    """Offline stand-in for litellm.completion: call the first listed tool, then answer."""
    from types import SimpleNamespace
    msgs = kw["messages"]
    if any(m["role"] == "user" and m["content"].startswith("Observation") for m in msgs) or "must finish" in msgs[-1]["content"]:
        text = '<action>{"type": "answer", "content": "done"}</action>'
    else:
        name = msgs[0]["content"].split("\n- ", 1)[1].split(":", 1)[0]
        text = '<action>{"type": "tool", "name": "%s", "arguments": {}}</action>' % name
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=800, completion_tokens=60))


def _service_backend(splits: tuple[str, ...] = ("test", "train")):
    """The hosted service through the SDK it serves (simple_agentic_evals, wheel at GET /sdk)."""
    if not api_key():
        raise SystemExit("EVAL_SERVICE_API_KEY is not set. Obtain it by logging in to the EvoHarnessBench demo "
                         "and clicking MyAuthtoken; see pace/experiments/evoharness/README.md")
    try:
        import simple_agentic_evals  # noqa: F401
    except ImportError as e:
        raise SystemExit("simple_agentic_evals is not installed; install the wheel served at /sdk "
                         "(README, step 3)") from e
    from ...harness.evoharness import SDKBackend
    return SDKBackend(splits=splits)


def allocate(sizes: dict[int, int], n_total: int) -> dict[int, int]:
    """Split n_total tasks over stages in proportion to their sizes (largest remainder)."""
    tot = sum(sizes.values())
    if n_total >= tot:
        return dict(sizes)
    raw = {k: n_total * v / tot for k, v in sizes.items()}
    out = {k: int(r) for k, r in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:n_total - sum(out.values())]:
        out[k] += 1
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health", help="GET /v1/health with EVAL_SERVICE_API_KEY")
    c = sub.add_parser("collect")
    c.add_argument("--axis", required=True, choices=AXES)
    c.add_argument("--stage", required=True, help="stage K, or 'all' (n-tasks spread over stages by size)")
    c.add_argument("--splits", default="test,train", help="service splits: test (held-out), train (adaptation)")
    c.add_argument("--n-tasks", type=int, required=True)
    c.add_argument("--tiers", default="all")
    c.add_argument("--cap", type=float, required=True, help="hard USD cap on this ledger dir")
    c.add_argument("--traces-dir", default="traces/evoharness")
    c.add_argument("--ledger-dir", default=None, help="spend ledger dir (default: --traces-dir); parallel workers "
                   "each take their own ledger dir and a slice of the cap")
    c.add_argument("--only-stages", default=None, help="with --stage all: run only these stages (comma list), "
                   "keeping the all-stage proportional allocation")
    c.add_argument("--out", default=None)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--project-n", type=int, default=60, help="stop if per-task cost x this > cap (0 = off)")
    c.add_argument("--max-turns", type=int, default=20)
    c.add_argument("--max-tokens", type=int, default=2048)
    c.add_argument("--effort", default="low")
    c.add_argument("--timeout", type=float, default=120.0)
    c.add_argument("--dry-run", action="store_true", help="MockBackend + scripted model; no network, no spend")
    z = sub.add_parser("analyze")
    z.add_argument("--traces-dir", default="traces/evoharness")
    z.add_argument("--tiers", default="all")
    z.add_argument("--out", default=None, help="JSON report path (default <traces-dir>/analysis.json)")
    a = ap.parse_args(argv)
    if a.cmd == "health":
        print(json.dumps(health(), indent=1))
        return
    if a.cmd == "analyze":
        tiers = parse_tiers(a.tiers)
        paths = sorted(Path(a.traces_dir).glob("evoharness_*.jsonl"))
        rep = analyze(paths, tiers)
        _print_report(rep, tiers)
        out = Path(a.out or Path(a.traces_dir) / "analysis.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1) + "\n")
        print(f"[evo] report -> {out}")
        return
    tiers = parse_tiers(a.tiers)
    ledger = SpendLedger(a.ledger_dir or a.traces_dir, a.cap)
    splits = tuple(x.strip() for x in a.splits.split(",") if x.strip())
    if a.dry_run:
        n_st = 3 if a.stage == "all" else max(int(a.stage), 1)
        backend, llm = MockBackend(n_stages=n_st), PricedLLM(ledger, completion_fn=scripted_completion)
        stage_list = list(range(1, n_st + 1)) if a.stage == "all" else [int(a.stage)]
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        backend = _service_backend(splits)
        llm = PricedLLM(ledger, timeout=a.timeout, effort=None if a.effort == "none" else a.effort,
                        prompt_cache=os.environ.get("PACE_PROMPT_CACHE", "1") == "1")   # on unless =0
        stage_list = (sorted({v for vs in backend.stages(a.axis).values() for v in vs}) if a.stage == "all"
                      else [int(a.stage)])
    hs = {k: EvoHarness(backend, a.axis, k, ledger=ledger, llm=llm, max_turns=a.max_turns, max_tokens=a.max_tokens)
          for k in stage_list}
    n_by_stage = (allocate({k: len(h.tasks()) for k, h in hs.items()}, a.n_tasks) if a.stage == "all"
                  else {stage_list[0]: a.n_tasks})
    if a.only_stages:
        keep = {int(x) for x in a.only_stages.split(",") if x.strip()}
        hs = {k: h for k, h in hs.items() if k in keep}
        n_by_stage = {k: v for k, v in n_by_stage.items() if k in keep}
    print(f"[evo] axis={a.axis} stages={n_by_stage} tiers={tiers} cap=${a.cap:g} ledger=${ledger.total:.4f}")
    for k, h in hs.items():
        out = a.out or str(Path(a.traces_dir) / f"evoharness_{a.axis}_s{k}.jsonl")
        print(f"[evo] stage {k}: {n_by_stage[k]} of {len(h.tasks())} tasks -> {out}")
        rc = collect(h, out, n_by_stage[k], tiers, a.seed, a.project_n)
        if rc:
            sys.exit(rc)
    print(ledger.summary())
    sys.exit(0)


if __name__ == "__main__":
    main()
