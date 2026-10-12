"""CLI for the process-verified MAS experiment.

  python -m pace.experiments.mas.run run --mode always_verify|never_verify|controller|all \
      --n-tasks 50 --org b2b --cap 25                   # real CRMArena-Pro run, hard-capped
  python -m pace.experiments.mas.run run --mode all --n-tasks 50 --dry-run    # deterministic fake backend
  python -m pace.experiments.mas.run analyze --log results/mas/mas_b2b.jsonl --out results/mas/RESULTS.md
  python -m pace.experiments.mas.run train-controller --log ... --quality-model ... --lam 2 --out policy.json
  python -m pace.experiments.mas.run estimate --n-tasks 50

Real runs: `--cap` is required, the ledger is the shared `traces/` ledger (cumulative, like
pace.collect) and ANTHROPIC_API_KEY plus the CRMArena .env must be present. Exit codes mirror
pace.collect: 0 done, 3 stopped at the spend cap, 4 a model call failed after retries.
Dry runs: never touch `traces/`; the ledger and the log live under `results/mas/dryrun/`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...controller import Controller
from ...harness.crmarena import CRMArenaHarness, PricedLLM
from ...prices import MODELS, TIERS, call_cost
from ...spend import SpendCapReached, SpendLedger
from ...trace import Episode, append_jsonl, read_jsonl
from ...train_offline import train_awr
from .orchestrator import ACTIONS, FEATURE_NAMES, N_FEATURES, Orchestrator, make_policy
from .orm import (CATEGORY_FEATURES, VERIFIER_FEATURES, QualityModel, cross_validate, fit_quality_model,
                  relabel_episodes)
from .process_verify import (fmt, is_correct, outcome_reward, process_reward_add, process_reward_mult,
                             process_stats, silent_failure_analysis, silent_failure_table)
from .roles import WorkerLLM

EXIT_STOPPED_AT_CAP, EXIT_LLM_FAILED = 3, 4
MODES = ["always_verify", "never_verify", "controller"]
DRYRUN_DIR = Path("results/mas/dryrun")


# --------------------------------------------------------------------- build
def build(dry_run: bool, org: str, cap: float | None, ledger_dir: str | None, seed: int, timeout: float = 120.0,
          effort: str | None = "low", n_fake_tasks: int = 66, **harness_kw):
    """(harness, ledger, backend_name). The harness's llm is the purpose-rewriting WorkerLLM."""
    if dry_run:
        from .fake_backend import FakeBackend, FakeCompletion
        root = Path(ledger_dir or DRYRUN_DIR / "ledger")
        ledger = SpendLedger(root, cap)
        backend = FakeBackend(org, n_fake_tasks, seed)
        llm = PricedLLM(ledger, completion_fn=FakeCompletion(backend, seed), timeout=timeout, effort=effort)
        h = CRMArenaHarness(org=org, ledger=ledger, llm=WorkerLLM(llm), backend=backend, **harness_kw)
        return h, ledger, "fake"
    from ...harness.crmarena import CRMArenaBackend
    ledger = SpendLedger(ledger_dir or "traces", cap)
    llm = PricedLLM(ledger, timeout=timeout, effort=effort)
    h = CRMArenaHarness(org=org, ledger=ledger, llm=WorkerLLM(llm), backend=CRMArenaBackend(org), **harness_kw)
    return h, ledger, "crmarena"


def done_keys(path: Path) -> set[tuple[str, str]]:
    out = set()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.add((str(d["task_id"]), str((d.get("meta") or {}).get("mode", "?"))))
    return out


def run_modes(h: CRMArenaHarness, ledger: SpendLedger, out: Path, modes: list[str], n_tasks: int, seed: int,
              backend_name: str, policy_path: str | None, eps: float, tiers: dict, max_calls: int = 8,
              log=print) -> int:
    orc = Orchestrator(h, tiers["planner"], tiers["worker"], tiers["verifier"], tiers["answer"], max_calls=max_calls)
    done = done_keys(out)
    ledger.clear_stop_marker()
    if ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd:
        p = ledger.write_stop_marker()
        log(f"[mas] ledger already at ${ledger.total:.4f} >= cap ${ledger.cap.cap_usd:g}; nothing run ({p})")
        return EXIT_STOPPED_AT_CAP
    from ...harness.crmarena import LLMCallFailed
    tasks = h.sample_tasks(n_tasks, seed)
    n_new = n_skip = 0
    for mode in modes:
        pol = make_policy(mode, tiers["worker"], tiers["answer"], policy_path, eps)
        mode_name = mode if mode != "controller" else f"controller:{pol.meta.get('name', 'policy')}"
        rng = random.Random(seed + 7919)
        for task in tasks:
            if (task.task_id, mode_name) in done:
                n_skip += 1
                continue
            try:
                ep = orc.run(task, pol, rng, eps if mode == "controller" else None)
            except SpendCapReached as e:
                p = ledger.write_stop_marker(e)
                log(f"[mas] STOPPED AT CAP before {task.task_id} mode {mode_name}: {e}")
                log(ledger.summary())
                log(f"[mas] wrote {p}; {n_new} new episodes -> {out}. Resume with a higher --cap.")
                return EXIT_STOPPED_AT_CAP
            except LLMCallFailed as e:
                ledger.write_total()
                log(f"[mas] ABORTED on {task.task_id} mode {mode_name}: {e}")
                return EXIT_LLM_FAILED
            ep.meta.update({"mode": mode_name, "backend": backend_name, "seed": seed, "eps": eps if mode == "controller" else None})
            append_jsonl(ep, out)
            n_new += 1
            s = process_stats(ep)
            log(f"[mas] {mode_name:22s} {task.task_id:8s} {task.category:30s} q={ep.quality:.2f} cost=${ep.total_cost:.4f} "
                f"calls={s['n_calls']} verify={s['n_pass']}/{s['n_verify']} replan={s['n_replan']} "
                f"steps={[st.action_name for st in ep.steps]}")
        log(ledger.summary())
    log(f"[mas] done: {n_new} new episodes, {n_skip} already logged -> {out}")
    return 0


# ------------------------------------------------------------------ estimate
# Per-call token assumptions, from the main README's measured CRMArena figures (see the package
# README for the derivation): the ReAct system prompt is ~4.1k o200k tokens (x1.2 Haiku, x1.45
# Sonnet/Opus); a mid-tier agent turn averages 7,800 in / 550 out (incl. ~400 thinking at effort low);
# a cheap turn 6,500 in / 150 out; the extractor 600 in / 30 out.
CALL_TOKENS = {"planner": {"cheap": (5200, 300), "mid": (6300, 450)},
               "worker": {"cheap": (6500, 150), "mid": (7800, 550), "frontier": (7800, 550)},
               "verifier": {"cheap": (1500, 60)},
               "answer": {"cheap": (7500, 150), "mid": (9000, 550), "frontier": (9000, 550)},
               "extractor": {"cheap": (600, 30)}}
CALLS_PER_TASK = {"always_verify": {"planner": 1, "worker": 2.8, "verifier": 2.8, "answer": 1, "extractor": 1},
                  "never_verify": {"planner": 1, "worker": 2.5, "verifier": 0, "answer": 1, "extractor": 1},
                  "controller": {"planner": 1.3, "worker": 2.6, "verifier": 1.3, "answer": 1, "extractor": 1}}


def estimate(n_tasks: int, modes: list[str], tiers: dict, thinking_out: int | None = None) -> tuple[str, float]:
    lines = [f"cost estimate: {n_tasks} tasks x {len(modes)} mode(s); prices from pace/prices.py; tiers "
             f"planner={tiers['planner']} worker={tiers['worker']} verifier={tiers['verifier']} answer={tiers['answer']}"]
    total = 0.0
    for mode in modes:
        per_task = 0.0
        lines.append(f"  {mode}:")
        for role, n in CALLS_PER_TASK[mode].items():
            if n == 0:
                continue
            tier = {"extractor": "cheap"}.get(role, tiers.get(role, "cheap"))
            if mode == "controller" and role in ("worker", "answer"):
                # uniform exploration over tiers for the worker / answer calls
                per_call = np.mean([call_cost(MODELS[t], *_tok(role, t, thinking_out)) for t in TIERS])
                desc = "mean over cheap/mid/frontier"
            else:
                tin, tout = _tok(role, tier, thinking_out)
                per_call = call_cost(MODELS[tier], tin, tout)
                desc = f"{tier} {tin:,} in / {tout} out"
            lines.append(f"    {role:9s} {n:>4} call(s) x ${per_call:.4f} ({desc}) = ${n * per_call:.4f}")
            per_task += n * per_call
        lines.append(f"    per task ${per_task:.4f}; x {n_tasks} = ${per_task * n_tasks:.2f}")
        total += per_task * n_tasks
    lines.append(f"  TOTAL ${total:.2f}")
    return "\n".join(lines), total


def _tok(role: str, tier: str, thinking_out: int | None) -> tuple[int, int]:
    table = CALL_TOKENS[role]
    tin, tout = table.get(tier) or table[next(iter(table))]
    if thinking_out is not None and role in ("worker", "answer") and tier != "cheap":
        tout = thinking_out + 150
    return tin, tout


# ------------------------------------------------------------------- analyze
def mode_table(by_mode: dict[str, list[Episode]], lams: list[float]) -> str:
    hdr = ("| mode | n | accuracy (q ≥ 0.5) | quality | cost/task (USD) | calls/task | verifier calls/task | pass rate | "
           "coverage | R_mult | R_add | " + " | ".join(f"J (λ={l:g})" for l in lams) + " |")
    lines = [hdr, "|" + "---|" * (11 + len(lams))]
    for mode, eps in by_mode.items():
        st = [process_stats(e) for e in eps]
        a = silent_failure_analysis(eps)
        q = np.mean([e.quality for e in eps]); c = np.mean([e.total_cost for e in eps])
        js = " | ".join(fmt(q - l * c) for l in lams)
        lines.append(f"| {mode} | {len(eps)} | {fmt(a['accuracy'])} | {fmt(q)} | {c:.4f} | "
                     f"{np.mean([s['n_calls'] for s in st]):.2f} | {np.mean([s['n_verify'] for s in st]):.2f} | "
                     f"{fmt(a['verifier_pass_rate'])} | {fmt(a['coverage_mean'])} | {fmt(a['R_mult'])} | {fmt(a['R_add'])} | {js} |")
    return "\n".join(lines)


def cost_split_table(by_mode: dict[str, list[Episode]]) -> str:
    roles = ["planner", "worker", "verifier", "answer"]
    lines = ["| mode | " + " | ".join(roles) + " | eval (extractor) | worker tiers used |", "|" + "---|" * (len(roles) + 3)]
    for mode, eps in by_mode.items():
        st = [process_stats(e) for e in eps]
        cells = [f"${np.mean([s['cost_by_role'].get(r, 0.0) for s in st]):.4f}" for r in roles]
        ev = np.mean([s["eval_cost"] for s in st])
        tiers = defaultdict(int)
        for s in st:
            for t, n in s["tiers"].items():
                tiers[t] += n
        tot = sum(tiers.values()) or 1
        mix = ", ".join(f"{t} {100 * tiers[t] / tot:.0f}%" for t in TIERS if tiers[t])
        lines.append(f"| {mode} | " + " | ".join(cells) + f" | ${ev:.4f} | {mix} |")
    return "\n".join(lines)


ORM_ROWS = [("process features (default)", list(CATEGORY_FEATURES)),
            ("process − verifier features", list(CATEGORY_FEATURES) + list(VERIFIER_FEATURES)),
            ("process + category one-hot", [])]


def orm_section(episodes: list[Episode], k: int, seed: int, qm_out: Path | None) -> tuple[str, QualityModel | None, dict]:
    y = np.array([is_correct(e) for e in episodes])
    n_tasks = len({e.task_id for e in episodes})
    lines = [f"Pooled over all modes: {len(episodes)} episodes over {n_tasks} tasks, base rate P(correct) = {y.mean():.3f}. "
             f"{k}-fold CV over tasks (category-stratified; a task's episodes never straddle folds); out-of-fold "
             f"predictions. Default features: counts of role calls, verifier verdicts and pass rate, open failures, "
             f"replans and retries, observation kinds, tiers used, cost. The '− verifier features' row drops "
             f"{', '.join(VERIFIER_FEATURES)}; the '+ category one-hot' row adds the 22-way task category.", "",
             "| model | features | AUROC | Brier | Brier (base rate) | ECE (5 bins) |", "|---|---|---|---|---|---|"]
    results: dict = {}
    if y.sum() < 2 or (1 - y).sum() < 2 or n_tasks < k:
        lines.append("| – | – | not enough episodes of both classes / tasks for CV | | | |")
        return "\n".join(lines), None, results
    for model in ("logreg", "mlp"):
        for name, drop in ORM_ROWS:
            r = cross_validate(episodes, model, k, seed, drop)
            results[(model, name)] = r
            lines.append(f"| {model} | {name} | {fmt(r['auroc'])} | {fmt(r['brier'])} | {fmt(r['brier_base_rate'])} | {fmt(r['ece'])} |")
    r = results[("logreg", ORM_ROWS[0][0])]
    lines += ["", "Calibration of the default logistic ORM (out-of-fold):", "",
              "| predicted bin | n | mean predicted | observed correct |", "|---|---|---|---|"]
    for c in r["calibration"]:
        lines.append(f"| [{c['lo']:.1f}, {c['hi']:.1f}) | {c['n']} | {fmt(c['mean_pred'])} | {fmt(c['frac_pos'])} |")
    qm = fit_quality_model(episodes, "logreg", seed)
    if qm_out is not None:
        qm.save(qm_out)
        lines += ["", f"Exported `{qm_out}` (default logistic ORM on all {len(episodes)} episodes; in-sample AUROC "
                  f"{fmt(qm.meta['train_auroc'])}, Brier {fmt(qm.meta['train_brier'])}; out-of-fold above). Consumers: "
                  "`orm.relabel_episodes` → `pace.train_offline.train_awr` / `pace.gate.estimate`."]
    top = sorted(zip(qm.feature_names, qm.model.w), key=lambda t: -abs(t[1]))[:8]
    lines += ["", "Largest standardised logistic weights: " + ", ".join(f"{n} {v:+.2f}" for n, v in top) + "."]
    return "\n".join(lines), qm, results


def priced_table(by_mode: dict[str, list[Episode]], qm: QualityModel | None, lams: list[float]) -> str:
    hdr = "| mode | quality (outcome) | quality_hat (ORM) | cost/task | " + " | ".join(
        f"J_out (λ={l:g}) | J_orm (λ={l:g})" for l in lams) + " |"
    lines = [hdr, "|" + "---|" * (4 + 2 * len(lams))]
    for mode, eps in by_mode.items():
        q = np.mean([e.quality for e in eps]); c = np.mean([e.total_cost for e in eps])
        qh = np.mean([qm.predict_episode(e) for e in eps]) if qm else float("nan")
        cells = " | ".join(f"{fmt(q - l * c)} | {fmt(qh - l * c)}" for l in lams)
        lines.append(f"| {mode} | {fmt(q)} | {fmt(qh)} | {c:.4f} | {cells} |")
    return "\n".join(lines)


def controller_hook(by_mode: dict[str, list[Episode]], qm: QualityModel | None, lam: float, out_policy: Path | None,
                    seed: int) -> str:
    """AWR on the ORM-relabelled exploration log (reward quality_hat - lam x cost), then the
    package's off-policy estimators (offpolicy.py) with effective sample sizes; the served row,
    when the policy was run with `--mode controller --policy`, is the on-policy measurement."""
    from .offpolicy import STEP_CLIP, cost_to_go, episode_level, ess_warning, per_decision
    explore = [e for m, eps in by_mode.items() if m.startswith("controller") for e in eps
               if any(s.propensity < 1.0 for s in e.steps)]
    if qm is None or len(explore) < 10:
        return "No exploration log with logged propensities (run `--mode controller --policy uniform`), so the hook is not exercised."
    rel = relabel_episodes(explore, qm)
    pol = train_awr(rel, ACTIONS, lam=lam, n_features=N_FEATURES)
    name = f"awr_orm_lam{lam:g}"
    pol.meta.update({"name": name, "reward": "quality_hat - lam*cost (mas_orm)"})
    if out_policy is not None:
        pol.save(out_policy)
    prefix = float(np.mean([(e.meta or {}).get("prefix_cost", 0.0) for e in explore]))
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    fc = lambda t: f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"
    pd_hat = per_decision(rel, pol, "quality", 200, seed)
    pd_out = per_decision(explore, pol, "quality", 200, seed)
    pd_cost = per_decision(explore, pol, "cost_to_go", 200, seed)
    el_hat = episode_level(rel, pol, "quality", STEP_CLIP, 200, seed)
    el_out = episode_level(explore, pol, "quality", STEP_CLIP, 200, seed)
    el_cost = episode_level(explore, pol, "cost", STEP_CLIP, 200, seed)
    lines = [f"AWR+IPS on the {len(explore)}-episode uniform exploration log ({pd_out['n']} logged decisions), reward "
             f"quality_hat − {lam:g}·cost; {'saved to `' + str(out_policy) + '`' if out_policy else 'not saved'}.",
             "",
             "Per-decision estimates treat each logged decision point as a contextual-bandit sample with its own "
             "single-step importance ratio (max {:.1f} here): the value of changing one decision while the rest of the "
             "trajectory stays as logged, scored on quality_hat, on the outcome, and on the cost-to-go from the decision "
             "(the policy-independent prefix, planner + first worker, is ${:.4f}/task on top). The episode-level row is "
             "self-normalised IPS of the full policy with each per-step ratio clipped at {:g}. ESS = (Σw)²/Σw²; an "
             "estimate with ESS below 10% of n is flagged.".format(pd_out["max_ratio"], prefix, STEP_CLIP), "",
             "| policy | estimator | unit | n | ESS | quality_hat [90% CI] | outcome quality [90% CI] | cost [90% CI] |",
             "|---|---|---|---|---|---|---|---|",
             f"| uniform behavior | on-policy mean | decision | {pd_out['n']} | {pd_out['n']} | {pd_hat['behavior']:.3f} | "
             f"{pd_out['behavior']:.3f} | cost-to-go {pd_cost['behavior']:.4f} |",
             f"| AWR (ORM reward) | per-decision SNIPS | decision | {pd_out['n']} | {pd_out['ess']:.1f} | {f(pd_hat['snips'])} | "
             f"{f(pd_out['snips'])} | cost-to-go {fc(pd_cost['snips'])} |",
             f"| AWR (ORM reward) | per-decision DR (bucket DM) | decision | {pd_out['n']} | {pd_out['ess']:.1f} | {f(pd_hat['dr'])} | "
             f"{f(pd_out['dr'])} | cost-to-go {fc(pd_cost['dr'])} |",
             f"| uniform behavior | on-policy mean | episode | {el_out['n']} | {el_out['n']} | {el_hat['behavior']:.3f} | "
             f"{el_out['behavior']:.3f} | total {el_cost['behavior']:.4f} |",
             f"| AWR (ORM reward) | episode SN-IPS, step clip {STEP_CLIP:g} | episode | {el_out['n']} | {el_out['ess']:.1f} | "
             f"{f(el_hat['snips'])} | {f(el_out['snips'])} | total {fc(el_cost['snips'])} |"]
    served = by_mode.get(f"controller:{name}") or by_mode.get(f"controller:{out_policy.stem}" if out_policy else "")
    if served:
        q = np.mean([e.quality for e in served]); c = np.mean([e.total_cost for e in served])
        qh = np.mean([qm.predict_episode(e) for e in served])
        ctg = np.mean([v for e in served for v in cost_to_go(e)])
        nd = sum(len(e.steps) for e in served)
        lines.append(f"| AWR (ORM reward), served | on-policy mean | decision | {nd} | {nd} | {qh:.3f} | {q:.3f} | cost-to-go {ctg:.4f} |")
        lines.append(f"| AWR (ORM reward), served | on-policy mean | episode | {len(served)} | {len(served)} | {qh:.3f} | {q:.3f} | total {c:.4f} |")
    warns = [f"{lbl}: {ess_warning(e)}" for lbl, e in (("per-decision", pd_out), ("episode-level", el_out))]
    flagged = [w for w in warns if "unreliable" in w]
    lines += ["", ("WARNING — " + "; ".join(flagged) + ". Read the served rows." if flagged else "; ".join(warns) + ".")]
    greedy = np.array([pol.probs(s.features, s.legal, eps=0.0) for e in explore for s in e.steps])
    mix = defaultdict(float)
    for p in greedy:
        mix[ACTIONS[int(p.argmax())]] += 1
    tot = sum(mix.values())
    top = ", ".join(f"{a} {100 * n / tot:.0f}%" for a, n in sorted(mix.items(), key=lambda t: -t[1])[:5])
    lines += ["", f"Greedy action of the trained policy at the logged decision points: {top}. "
              + ("" if served else "Serve it with `run --mode controller --policy <file>` to measure it on-policy.")]
    return "\n".join(lines)


def paired_table(by_mode: dict[str, list[Episode]], a: str = "always_verify", b: str = "never_verify",
                 threshold: float = 0.5) -> tuple[str, dict]:
    """Per-task paired comparison of two modes on their shared tasks: the 2x2 of correct/wrong, the
    tasks verification flipped in each direction (with what the verifier did on them), and the
    paired mean difference in quality and cost with its standard error."""
    A = {e.task_id: e for e in by_mode.get(a, [])}
    B = {e.task_id: e for e in by_mode.get(b, [])}
    ids = sorted(set(A) & set(B))
    if not ids:
        return f"(no tasks shared by {a} and {b})", {}
    ok = lambda e: float(e.quality) >= threshold
    both_r = [t for t in ids if ok(A[t]) and ok(B[t])]
    both_w = [t for t in ids if not ok(A[t]) and not ok(B[t])]
    a_only = [t for t in ids if ok(A[t]) and not ok(B[t])]
    b_only = [t for t in ids if not ok(A[t]) and ok(B[t])]
    dq = np.array([A[t].quality - B[t].quality for t in ids])
    dc = np.array([A[t].total_cost - B[t].total_cost for t in ids])
    se = float(dq.std(ddof=1) / np.sqrt(len(dq))) if len(dq) > 1 else float("nan")
    stats = {"n": len(ids), "both_right": len(both_r), "both_wrong": len(both_w), f"{a}_only": len(a_only),
             f"{b}_only": len(b_only), "net": len(a_only) - len(b_only), "dq_mean": float(dq.mean()), "dq_se": se,
             "dc_mean": float(dc.mean())}
    lines = [f"{len(ids)} tasks run under both `{a}` and `{b}` (same task ids, same seeds).", "",
             f"| | `{b}` correct | `{b}` wrong |", "|---|---|---|",
             f"| `{a}` correct | {len(both_r)} | **{len(a_only)}** (wrong → right) |",
             f"| `{a}` wrong | **{len(b_only)}** (right → wrong) | {len(both_w)} |", "",
             f"Net: {len(a_only)} − {len(b_only)} = **{len(a_only) - len(b_only):+d} task(s)** of {len(ids)}; paired mean Δquality "
             f"{dq.mean():+.3f} ± {se:.3f} (SE), paired mean Δcost ${dc.mean():+.4f}/task.", ""]

    def desc(t):
        e = A[t]
        return (f"`{t}` ({e.meta.get('category')}): verifier {e.meta.get('n_verify_fail', 0)} fail / {e.meta.get('n_verify', 0)} calls, "
                f"{e.meta.get('n_retry', 0)} retry, {e.meta.get('n_calls')} role calls")
    if a_only:
        lines += [f"Flipped wrong → right under `{a}`:", ""] + [f"- {desc(t)}" for t in a_only] + [""]
    if b_only:
        lines += [f"Flipped right → wrong under `{a}`:", ""] + [f"- {desc(t)}" for t in b_only] + [""]
    return "\n".join(lines), stats


def single_agent_table(by_mode: dict[str, list[Episode]], sa_log: Path, lams: list[float], threshold: float = 0.5) -> str | None:
    """The single-agent tiers of the counterfactual log (pace.lookup.LookupTable) restricted to the
    MAS task ids, side by side with the MAS modes, J = quality − λ·cost for each λ."""
    from ...lookup import LookupTable
    tab = LookupTable.from_episodes(read_jsonl(sa_log))
    base = next((m for m in ("always_verify", "never_verify") if m in by_mode), None)
    if base is None:
        return "(no fixed-mode MAS log to compare against)"
    mas_ids = sorted({e.task_id for e in by_mode[base]})
    pos = {t: i for i, t in enumerate(tab.task_ids)}
    common = [t for t in mas_ids if t in pos]
    if not common:
        return None
    sub = tab.subset(common)
    hdr = "| system | tasks | accuracy / quality | cost/task (USD) | " + " | ".join(f"J (λ={l:g})" for l in lams) + " |"
    lines = [f"Single-agent rows are exact lookups of `{sa_log}` on the {len(common)} MAS task ids (quality is the same "
             f"CRMArena score; the single-agent continuation is CONTINUE on one tier until the agent answers, ≤ 12 turns). "
             f"MAS rows are the modes above; `controller:uniform` ran on fewer tasks and is listed with its own single-agent "
             f"reference below.", "", hdr, "|" + "---|" * (4 + len(lams))]
    def row(name, n, q, c):
        return f"| {name} | {n} | {q:.3f} | {c:.4f} | " + " | ".join(f"{q - l * c:.3f}" for l in lams) + " |"
    for j, tier in enumerate(TIERS):
        lines.append(row(f"single agent, fixed {tier}", len(common), sub.Q[:, j].mean(), sub.C[:, j].mean()))
    q, c = sub.value(sub.oracle(None))
    lines.append(row("single agent, per-task oracle (max quality, cheapest)", len(common), q, c))
    for l in lams:
        q, c = sub.value(sub.oracle(l))
        lines.append(row(f"single agent, per-task oracle (argmax q − {l:g}·c)", len(common), q, c))
    extra = []
    for mode, eps in by_mode.items():
        on_common = [e for e in eps if e.task_id in set(common)]
        if {e.task_id for e in on_common} == set(common):
            lines.append(row(f"MAS `{mode}`", len(on_common), np.mean([e.quality for e in on_common]),
                             np.mean([e.total_cost for e in on_common])))
        elif on_common:
            ids = sorted({e.task_id for e in on_common})
            s2 = tab.subset(ids)
            q, c = np.mean([e.quality for e in on_common]), np.mean([e.total_cost for e in on_common])
            extra.append(f"`{mode}` ({len(on_common)} episodes on {len(ids)} of the {len(common)} tasks): {q:.3f} at ${c:.4f}; "
                         f"single mid on those {len(ids)} tasks: {s2.Q[:, 1].mean():.3f} at ${s2.C[:, 1].mean():.4f}, single cheap "
                         f"{s2.Q[:, 0].mean():.3f} at ${s2.C[:, 0].mean():.4f}, single frontier {s2.Q[:, 2].mean():.3f} at ${s2.C[:, 2].mean():.4f}.")
    # paired MAS vs single mid
    for mode in ("always_verify", "never_verify"):
        if mode in by_mode:
            M = {e.task_id: e for e in by_mode[mode]}
            cells = defaultdict(int)
            for t in common:
                cells[(M[t].quality >= threshold, sub.Q[pos_common(sub, t), 1] >= threshold)] += 1
            extra.append(f"Paired `{mode}` vs single mid on the same {len(common)} tasks: both right {cells[(True, True)]}, "
                         f"MAS right / mid wrong {cells[(True, False)]}, MAS wrong / mid right {cells[(False, True)]}, "
                         f"both wrong {cells[(False, False)]} (net {cells[(True, False)] - cells[(False, True)]:+d} for the MAS).")
    return "\n".join(lines + [""] + [f"- {x}" for x in extra])


def pos_common(sub, t: str) -> int:
    return sub.task_ids.index(t)


def interpretation(by_mode: dict[str, list[Episode]], cv: dict, backend: str, lams: list[float]) -> str:
    rows = {m: silent_failure_analysis(eps) for m, eps in by_mode.items()}
    cost = {m: float(np.mean([e.total_cost for e in eps])) for m, eps in by_mode.items()}
    lines = []
    if backend == "fake":
        lines.append("- **These numbers come from the deterministic fake backend** (`fake_backend.py`): they show that "
                     "the pipeline runs end to end, that every role call is priced, and that the tables render. "
                     "They say nothing about CRM tasks. The generator's latent model is written down in the fake's "
                     "docstring, so the ORM's AUROC here measures how well a logistic model recovers that generator, "
                     "not anything about Salesforce data.")
    av, nv = rows.get("always_verify"), rows.get("never_verify")
    if av and nv:
        lines.append(f"- **Outcome vs process.** always_verify reaches accuracy {fmt(av['accuracy'])} at ${cost['always_verify']:.4f}/task "
                     f"against never_verify {fmt(nv['accuracy'])} at ${cost['never_verify']:.4f}/task; the verifier costs "
                     f"${cost['always_verify'] - cost['never_verify']:+.4f}/task. Priced: J(λ=2) "
                     f"{fmt(av['R_out'] - 2 * cost['always_verify'])} vs {fmt(nv['R_out'] - 2 * cost['never_verify'])}, "
                     f"J(λ=5) {fmt(av['R_out'] - 5 * cost['always_verify'])} vs {fmt(nv['R_out'] - 5 * cost['never_verify'])}.")
        lines.append(f"- **Silent failures.** Among always_verify episodes, {fmt(av['silent_failure_rate'])} are wrong with every "
                     f"verifier passing (process verification missed them) and {fmt(av['caught_rate'])} are wrong with at least "
                     f"one fail (caught); {fmt(av['correct_despite_fail_rate'])} are correct despite a fail "
                     f"({fmt(av['correct_despite_open_fail_rate'])} with the fail still open: false alarms or steps the answer "
                     f"did not need). P(flagged | wrong) = {fmt(av['p_flagged_given_wrong'])}, "
                     f"P(wrong | all pass) = {fmt(av['p_wrong_given_all_pass'])}: the first is what process verification "
                     f"adds over outcome-only scoring, the second is the residual error a clean process still carries.")
    ctrl = [m for m in by_mode if m.startswith("controller")]
    for m in ctrl:
        r = rows[m]
        if "uniform" in m:
            lines.append(f"- **{m}.** accuracy {fmt(r['accuracy'])} at ${cost[m]:.4f}/task: a uniform policy over 12 actions "
                         "is the exploration log (propensities logged) the ORM and the AWR hook train on, not a candidate.")
            continue
        best = max((mm for mm in by_mode if not mm.startswith("controller")), key=lambda mm: rows[mm]["R_out"] - lams[0] * cost[mm])
        lines.append(f"- **{m}, served on-policy.** accuracy {fmt(r['accuracy'])} at ${cost[m]:.4f}/task, J(λ={lams[0]:g}) "
                     f"{fmt(r['R_out'] - lams[0] * cost[m])} against {best} {fmt(rows[best]['R_out'] - lams[0] * cost[best])}"
                     + (f" and the uniform exploration log {fmt(rows['controller:uniform']['R_out'] - lams[0] * cost['controller:uniform'])}"
                        if "controller:uniform" in rows else "") + ". "
                     + ("On the fake this is a null result: one AWR pass on 100 exploration episodes with an ORM reward does not "
                        "beat the fixed baselines. Section 5 reports each off-policy estimate with its effective sample size; "
                        "where the ESS is a small fraction of n the estimate is flagged and the served rows are the ones to read."
                        if backend == "fake" else
                        "Read the served rows over any off-policy row whose ESS is flagged in section 5."))
    if cv:
        d, nv, pc = (cv.get(("logreg", ORM_ROWS[0][0])), cv.get(("logreg", ORM_ROWS[1][0])), cv.get(("logreg", ORM_ROWS[2][0])))
        lines.append(f"- **ORM.** Out of fold, the default logistic ORM has AUROC {fmt(d['auroc'])} and Brier {fmt(d['brier'])} "
                     f"against a base-rate Brier of {fmt(d['brier_base_rate'])} (ECE {fmt(d['ece'])}). Dropping the verifier "
                     f"features moves AUROC to {fmt(nv['auroc'])}; adding the category one-hot moves it to {fmt(pc['auroc'])}. "
                     + ("On the fake backend the verifier verdict is a noisy function of the observation kind, which is "
                        "already a feature, so the verifier features cannot add information here; the category carries none "
                        "by construction and only adds 22 dimensions to memorise. Whether verifier verdicts carry information "
                        "beyond the observation on real CRM data is exactly what the real run measures."
                        if backend == "fake" else
                        ("The category one-hot raises out-of-fold AUROC, and the CV is over tasks, so this is task difficulty "
                         "that varies by category and transfers across tasks, not memorisation; the process features add "
                         "nothing beyond it here. With 2–3 training tasks per category the per-category estimate is thin."
                         if pc["auroc"] > d["auroc"] + 0.05 else
                         "A gap between the first two rows is the information the verifier verdicts carry beyond the raw "
                         "observation kinds; a drop from adding the category is the same memorisation the linear controller "
                         "showed on 50 tasks.")))
    lines.append("- **What this does not show.** Whether a priced controller over {VERIFY, SKIP_VERIFY, REPLAN, ANSWER} × tier "
                 "beats the fixed baselines at equal budget needs the real run plus an on-policy evaluation of the trained "
                 "policy; the per-decision estimates on the exploration log value one changed decision at a time and the "
                 "clipped episode-level estimate is biased by construction.")
    return "\n".join(lines)


def analyze(logs: list[Path], out: Path, lams: list[float], k: int, seed: int, qm_out: Path | None,
            policy_out: Path | None, hook_lam: float, sa_log: Path | None = None, prelude: Path | None = None) -> str:
    episodes: list[Episode] = []
    for p in logs:
        episodes += read_jsonl(p)
    if not episodes:
        raise SystemExit("no episodes in " + ", ".join(map(str, logs)))
    by_mode: dict[str, list[Episode]] = defaultdict(list)
    for e in episodes:
        by_mode[(e.meta or {}).get("mode", "?")].append(e)
    by_mode = dict(sorted(by_mode.items(), key=lambda t: (t[0].startswith("controller"), t[0])))
    backend = (episodes[0].meta or {}).get("backend", "?")
    tiers = (episodes[0].meta or {}).get("role_tiers", {})
    orm_txt, qm, cv = orm_section(episodes, k, seed, qm_out)
    title = "# MAS results" + (" — DRY RUN on the fake backend (a pipeline check; these numbers carry no scientific content)" if backend == "fake" else
                               f" on CRMArena-Pro {(episodes[0].meta or {}).get('org', '')}")
    paired_txt, _ = paired_table(by_mode)
    sa_txt = single_agent_table(by_mode, sa_log, lams) if (sa_log and Path(sa_log).exists()) else None
    parts = [title, ""]
    if prelude is not None and Path(prelude).exists():
        parts += ["## 0. Interpretation", "", Path(prelude).read_text().strip(), "",
                  "Sections 1–7 below are generated by `run.py analyze`; section 0 is authored and passed in with `--prelude`.", ""]
    parts += [f"Logs: {', '.join(str(p) for p in logs)}. {len(episodes)} episodes over {len({e.task_id for e in episodes})} tasks, "
             f"backend `{backend}`, host harness `{(episodes[0].meta or {}).get('host_harness', '?')}`. Role tiers: "
             + ", ".join(f"{r} = {t}" for r, t in tiers.items()) + ". Cost/task is the sum of the role calls "
             "(planner, worker, verifier, answer); the outcome evaluator's extractor call is listed separately. "
             "Quality is the host harness's own score (CRMArena's evaluator); accuracy thresholds it at 0.5. "
             "R_mult = quality × verified-pass coverage, R_add = quality + 0.25·coverage − 0.25·open-fail rate "
             "(see `process_verify.py`).", "",
             "## 1. Per-mode quality, cost and process rewards", "", mode_table(by_mode, lams), "",
             "Cost split per task by role:", "", cost_split_table(by_mode), "",
             "## 2. Silent-failure analysis: what process verification catches that the outcome misses, and vice versa", "",
             "Rates are over episodes that had at least one verifier call (the never_verify row has none by construction).", "",
             silent_failure_table(by_mode), "",
             "## 2b. Paired per-task comparison: what verification bought", "", paired_txt, "",
             "## 3. Orchestration reward model: process features → P(correct)", "", orm_txt, "",
             "## 4. Priced comparison: J = quality − λ·cost, outcome vs ORM quality", "", priced_table(by_mode, qm, lams), "",
             "## 5. Priced controller hook: AWR on the ORM-relabelled exploration log", "",
             controller_hook(by_mode, qm, hook_lam, policy_out, seed), "",
             "## 6. Generated summary", "", interpretation(by_mode, cv, backend, lams), ""]
    if sa_txt is not None:
        parts += ["## 7. The comparison that matters: decomposition vs the single agent on the same tasks", "", sa_txt, ""]
    text = "\n".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


# ----------------------------------------------------------------------- CLI
def _tiers(a) -> dict:
    return {"planner": a.planner_tier, "worker": a.worker_tier, "verifier": a.verifier_tier,
            "answer": a.answer_tier or a.worker_tier}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Process-verified MAS on CRMArena-Pro: run / analyze / estimate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one or all modes (real: --cap required; --dry-run: fake backend)")
    r.add_argument("--mode", default="all", choices=MODES + ["all"])
    r.add_argument("--n-tasks", type=int, required=True)
    r.add_argument("--org", default="b2b", choices=["b2b", "b2c"])
    r.add_argument("--cap", type=float, default=None, help="hard USD cap on cumulative ledger spend")
    r.add_argument("--dry-run", action="store_true", help="deterministic fake LLM + fake Salesforce; no spend")
    r.add_argument("--policy", default="uniform", help="controller mode: 'uniform' or a policy JSON")
    r.add_argument("--eps", type=float, default=0.2, help="controller exploration eps (logged in the propensity)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="episode log (default results/mas/mas_<org>.jsonl, dry run under results/mas/dryrun/)")
    r.add_argument("--ledger-dir", default=None, help="real: traces (shared, cumulative); dry run: results/mas/dryrun/ledger")
    r.add_argument("--planner-tier", default="cheap", choices=TIERS)
    r.add_argument("--worker-tier", default="mid", choices=TIERS)
    r.add_argument("--verifier-tier", default="cheap", choices=TIERS)
    r.add_argument("--answer-tier", default=None, choices=TIERS)
    r.add_argument("--max-calls", type=int, default=8)
    r.add_argument("--n-fake-tasks", type=int, default=66)
    r.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--effort", default="low")
    r.add_argument("--env-file", default="third_party/CRMArena/.env")

    z = sub.add_parser("analyze", help="tables -> RESULTS.md (offline)")
    z.add_argument("--log", action="append", required=True)
    z.add_argument("--out", default="results/mas/RESULTS.md")
    z.add_argument("--lam", type=float, nargs="*", default=[2.0, 5.0])
    z.add_argument("--k", type=int, default=5)
    z.add_argument("--seed", type=int, default=0)
    z.add_argument("--quality-model-out", default="results/mas/quality_model.json")
    z.add_argument("--policy-out", default="results/mas/policy_awr_orm.json")
    z.add_argument("--hook-lam", type=float, default=2.0)
    z.add_argument("--single-agent-log", default="traces/crmarena_b2b_counterfactual.jsonl",
                   help="counterfactual single-agent log (pace.collect --mode counterfactual) for the side-by-side table; skipped if absent")
    z.add_argument("--prelude", default=None, help="authored interpretation (markdown) inserted as section 0")

    t = sub.add_parser("train-controller", help="AWR on an ORM-relabelled log (reward quality_hat - lam*cost)")
    t.add_argument("--log", action="append", required=True)
    t.add_argument("--quality-model", required=True)
    t.add_argument("--lam", type=float, default=2.0)
    t.add_argument("--out", required=True)
    t.add_argument("--serve-eps", type=float, default=0.05)

    e = sub.add_parser("estimate", help="cost arithmetic from pace/prices.py")
    e.add_argument("--n-tasks", type=int, default=50)
    e.add_argument("--modes", default="always_verify,never_verify")
    e.add_argument("--thinking-out", type=int, default=None, help="sensitivity: thinking tokens per mid/frontier call")
    for p in (e,):
        p.add_argument("--planner-tier", default="cheap", choices=TIERS)
        p.add_argument("--worker-tier", default="mid", choices=TIERS)
        p.add_argument("--verifier-tier", default="cheap", choices=TIERS)
        p.add_argument("--answer-tier", default=None, choices=TIERS)
    a = ap.parse_args(argv)

    if a.cmd == "estimate":
        txt, _ = estimate(a.n_tasks, [m.strip() for m in a.modes.split(",") if m.strip()], _tiers(a), a.thinking_out)
        print(txt)
        return 0
    if a.cmd == "analyze":
        txt = analyze([Path(p) for p in a.log], Path(a.out), a.lam, a.k, a.seed,
                      Path(a.quality_model_out) if a.quality_model_out else None,
                      Path(a.policy_out) if a.policy_out else None, a.hook_lam,
                      Path(a.single_agent_log) if a.single_agent_log else None, Path(a.prelude) if a.prelude else None)
        print(txt)
        return 0
    if a.cmd == "train-controller":
        eps_ = [e for p in a.log for e in read_jsonl(p) if any(s.propensity < 1.0 for s in e.steps)]
        qm = QualityModel.load(a.quality_model)
        pol = train_awr(relabel_episodes(eps_, qm), ACTIONS, lam=a.lam, serve_eps=a.serve_eps, n_features=N_FEATURES)
        pol.meta.update({"name": Path(a.out).stem, "reward": "quality_hat - lam*cost (mas_orm)", "n_episodes": len(eps_)})
        pol.save(a.out)
        print(f"[mas] AWR on {len(eps_)} exploration episodes, lam={a.lam:g} -> {a.out}")
        return 0

    # run
    modes = MODES if a.mode == "all" else [a.mode]
    if not a.dry_run:
        if a.cap is None:
            raise SystemExit("real MAS runs make paid API calls: pass --cap USD (or --dry-run)")
        try:
            from dotenv import load_dotenv
            load_dotenv(a.env_file)
        except ImportError:
            pass
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(f"ANTHROPIC_API_KEY is not set (looked in the environment and {a.env_file})")
    h, ledger, backend = build(a.dry_run, a.org, a.cap, a.ledger_dir, a.seed, a.timeout,
                               None if a.effort == "none" else a.effort, a.n_fake_tasks)
    out = Path(a.out) if a.out else (DRYRUN_DIR / f"mas_fake_{a.org}.jsonl" if a.dry_run else Path(f"results/mas/mas_{a.org}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mas] backend={backend} org={a.org} modes={modes} n_tasks={a.n_tasks} cap={'none' if a.cap is None else f'${a.cap:g}'} "
          f"ledger={ledger.root} (${ledger.total:.4f} so far) -> {out}")
    return run_modes(h, ledger, out, modes, a.n_tasks, a.seed, backend, a.policy, a.eps, _tiers(a), a.max_calls)


if __name__ == "__main__":
    sys.exit(main())
