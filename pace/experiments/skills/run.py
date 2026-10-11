"""CLI for the skill-handbook experiment.

  python -m pace.experiments.skills.run distill --cap 3
  python -m pace.experiments.skills.run serve --tier cheap --n-tasks 50 --cap 7 --traces-dir traces/skills
  python -m pace.experiments.skills.run analyze
  (any command) --dry-run   fake world, no key / network / spend, outputs under results/skills/dryrun/

Paid runs use their own ledgers (distill: traces/skills/distill, serve: --traces-dir), so each --cap
is an absolute cap on that step. Cap / resume / exit-code semantics are pace.collect's: the ledger is
cumulative, finished work is skipped on rerun, exit 0 done, 3 stopped at the cap (STOPPED_AT_CAP
written), 4 a model call failed after retries.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ...collect import EXIT_LLM_FAILED, EXIT_STOPPED_AT_CAP, collect_crmarena
from ...harness.crmarena import CRMArenaHarness, LLMCallFailed, PricedLLM
from ...spend import SpendCapReached, SpendLedger
from . import analyze as an
from .distill import distill, extract_exemplars, load_handbook, load_log, save_handbook, total_usd
from .serve import PrefixedLLM, SkilledCRMArenaHarness

DRYRUN = Path("results/skills/dryrun")
BASE_LOG = "traces/crmarena_b2b_counterfactual.jsonl"


def _world(a):
    """(backend, tasks, completion_fn, base log path) for the real or the fake world."""
    if not a.dry_run:
        try:
            from dotenv import load_dotenv
            load_dotenv(a.env_file)
        except ImportError:
            pass
        return None, None, None, a.log
    from .fake import FakeBackend, FakeCompletion, make_fake_tasks, write_fake_log
    tasks = make_fake_tasks()
    DRYRUN.mkdir(parents=True, exist_ok=True)
    log = DRYRUN / "fake_counterfactual.jsonl"
    write_fake_log(tasks, log)
    return FakeBackend(tasks), tasks, FakeCompletion(tasks), str(log)


def _paths(a):
    if a.dry_run:
        return {"handbook": str(DRYRUN / "handbook.json"), "distill_dir": str(DRYRUN / "ledger_distill"),
                "serve_dir": str(DRYRUN / "ledger_serve"), "out": str(DRYRUN / f"skills_{a.tier}_{a.org}.jsonl"),
                "results": str(DRYRUN / "RESULTS.md")}
    return {"handbook": a.handbook, "distill_dir": a.distill_dir, "serve_dir": a.traces_dir,
            "out": a.out or f"traces/skills_{a.tier}_{a.org}.jsonl", "results": a.results}


def _check_cap(a, ledger) -> int | None:
    if a.cap is None:
        raise SystemExit("paid API calls: pass --cap USD")
    ledger.clear_stop_marker()
    if ledger.total >= ledger.cap.cap_usd:
        ledger.write_stop_marker()
        print(f"[skills] ledger already at ${ledger.total:.4f} >= cap ${a.cap:g}; nothing run.")
        return EXIT_STOPPED_AT_CAP
    if not a.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"ANTHROPIC_API_KEY is not set (looked in the environment and {a.env_file})")
    return None


def cmd_distill(a) -> int:
    backend, tasks, completion, log = _world(a)
    P = _paths(a)
    ledger = SpendLedger(P["distill_dir"], a.cap)
    rc = _check_cap(a, ledger)
    if rc is not None:
        return rc
    h = CRMArenaHarness(org=a.org, ledger=ledger, llm=PricedLLM(ledger, completion_fn=completion),
                        backend=backend, tasks=tasks)
    sample = h.sample_tasks(a.n_tasks, a.seed)
    questions = {t.task_id: t.query for t in sample}
    by_cat: dict[str, list[str]] = {}
    for t in sample:
        by_cat.setdefault(t.category, []).append(t.task_id)
    ex = extract_exemplars(load_log(log, questions), questions)
    hb = load_handbook(P["handbook"])
    hb["meta"] = {"log": str(log), "org": a.org, "n_tasks": a.n_tasks, "seed": a.seed,
                  "source_arms": ["mid", "frontier"], "success_quality": 0.5, "writer": "mid",
                  "n_exemplars": sum(len(v) for v in ex.values())}
    print(f"[distill] {len(sample)} tasks, {len(by_cat)} categories, {hb['meta']['n_exemplars']} exemplars "
          f"from {log}; ledger ${ledger.total:.4f} cap ${a.cap:g}")
    try:
        distill(h.llm, by_cat, ex, hb, save=lambda x: save_handbook(x, P["handbook"]))
    except SpendCapReached as e:
        save_handbook(hb, P["handbook"])
        p = ledger.write_stop_marker(e)
        print(f"[distill] STOPPED AT CAP: {e} ({p}); rerun with a higher --cap to resume")
        return EXIT_STOPPED_AT_CAP
    except LLMCallFailed as e:
        save_handbook(hb, P["handbook"])
        ledger.write_total()
        print(f"[distill] ABORTED: {e}")
        return EXIT_LLM_FAILED
    save_handbook(hb, P["handbook"])
    print(f"[distill] done: {len(hb['variants'])} variants, ${total_usd(hb):.4f} -> {P['handbook']}")
    print(ledger.summary())
    return 0


def cmd_serve(a) -> int:
    backend, tasks, completion, _ = _world(a)
    P = _paths(a)
    hb = load_handbook(P["handbook"])
    if not hb.get("categories"):
        raise SystemExit(f"no handbook at {P['handbook']}; run distill first")
    ledger = SpendLedger(P["serve_dir"], a.cap)
    rc = _check_cap(a, ledger)
    if rc is not None:
        return rc
    llm = PrefixedLLM(PricedLLM(ledger, completion_fn=completion, timeout=a.timeout))
    h = SkilledCRMArenaHarness(hb, org=a.org, ledger=ledger, llm=llm, backend=backend, tasks=tasks)
    print(f"[serve] tier={a.tier} n_tasks={a.n_tasks} cap=${a.cap:g} ledger=${ledger.total:.4f} -> {P['out']}")
    return collect_crmarena(h, P["out"], a.n_tasks, "counterfactual", [a.tier], seed=a.seed)


def cmd_analyze(a) -> int:
    P = _paths(a)
    log = str(DRYRUN / "fake_counterfactual.jsonl") if a.dry_run else a.log
    text, _ = an.analyze(log, P["out"], P["handbook"], P["serve_dir"], P["distill_dir"],
                         arm_name=f"{a.tier}+skills", tier=a.tier)
    an.write(P["results"], text)
    print(text)
    print(f"[analyze] -> {P['results']}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Skill handbook for the cheap tier (CRMArena-Pro).")
    ap.add_argument("command", choices=["distill", "serve", "analyze"])
    ap.add_argument("--cap", type=float, default=None, help="hard USD cap on this step's own ledger")
    ap.add_argument("--tier", default="cheap", choices=["cheap", "mid", "frontier"])
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="the collector's task-sample seed")
    ap.add_argument("--org", default="b2b", choices=["b2b", "b2c"])
    ap.add_argument("--log", default=BASE_LOG, help="counterfactual log (read only)")
    ap.add_argument("--handbook", default="results/skills/handbook.json")
    ap.add_argument("--distill-dir", default="traces/skills/distill", help="distill ledger dir")
    ap.add_argument("--traces-dir", default="traces/skills", help="serve ledger dir")
    ap.add_argument("--out", default=None, help="serve episodes (default traces/skills_<tier>_<org>.jsonl)")
    ap.add_argument("--results", default="results/skills/RESULTS.md")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--env-file", default="third_party/CRMArena/.env")
    ap.add_argument("--dry-run", action="store_true", help="fake world: no key, no network, no spend")
    a = ap.parse_args(argv)
    sys.exit({"distill": cmd_distill, "serve": cmd_serve, "analyze": cmd_analyze}[a.command](a))


if __name__ == "__main__":
    main()
