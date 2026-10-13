"""Gate lifecycle error rates under four gate configurations (the reviewer's points A-C).

Configurations (everything else is the committed default schedule and Config):
  v1        the committed gate (candidate DR lower bound vs incumbent's DR point; DR decision cost
            vs the budget net of the log's mean prefix), with the v2 record computed in SHADOW
            (logged, never acted on) so its intervals can be scored too
  v1_h30    v1 + `--gate-holdout 0.3` (trainers see 70% of each day's episodes, the gate the other 30%)
  v2        pace.gate.gate_v2: joint-bootstrap quality difference >= -eps_q, total cost (prefix
            included) upper bound <= B + tol, overlap guard (ESS/n < 0.2 or clip fraction > 0.1 ->
            UNEVALUABLE, served as REVERT)
  v2_h30    v2 + holdout 0.3

Every run uses `--arms all` so that the v1 seed-0 run is the committed run (the harness draws one
random stream per day shared by the arms that run; dropping arms changes it). Only the continual
(cgrpo), continual_awr, never_verify and always_verify arms and the per-stage oracle are reported.

"Truly unacceptable" (simulator, held-out reference, as in run_continual's `truth`): the candidate's
held-out quality on the current stage < the incumbent's held-out quality - eps_q, OR its held-out
total cost per task > B + tol. Coverage is against that held-out Monte Carlo reference (150 tasks),
not against an exact truth.

  python -m pace.experiments.mas_continual.gate_lifecycle run --seeds 0 1 2 3 4 --workers 7
  python -m pace.experiments.mas_continual.gate_lifecycle report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import analyze

ROOT = Path("results/mas_continual/gate_v2")
CONFIGS = {
    "v1": {"gate": "v1", "gate_holdout": 0.0, "gate_shadow": 1},
    "v1_h30": {"gate": "v1", "gate_holdout": 0.3, "gate_shadow": 1},
    "v2": {"gate": "v2", "gate_holdout": 0.0},
    "v2_h30": {"gate": "v2", "gate_holdout": 0.3},
}
LABEL = {"v1": "v1 (as committed)", "v1_h30": "v1 + holdout 0.3", "v2": "v2", "v2_h30": "v2 + holdout 0.3"}
GATED = ("continual", "continual_awr")
REF = ("never_verify", "always_verify")


# ------------------------------------------------------------------ running
def _one(config: str, seed: int, root: str) -> tuple[str, int, float]:
    from .run_continual import Config, run
    out = Path(root) / config / f"seed{seed}"
    cfg = Config(seed=seed, arms="all", **CONFIGS[config])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "run.log").open("w") as fh:
        run(cfg, out, log=lambda m: (fh.write(m + "\n"), fh.flush()))
    return config, seed, time.time() - t0


def run_grid(configs: list[str], seeds: list[int], workers: int, root: Path = ROOT) -> None:
    todo = [(c, s) for c in configs for s in seeds if not (root / c / f"seed{s}" / "meta.jsonl").exists()]
    print(f"[grid] {len(todo)} runs to do ({len(configs) * len(seeds) - len(todo)} already complete)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, c, s, str(root)) for c, s in todo]
        for f in as_completed(futs):
            c, s, dt = f.result()
            print(f"[grid] {c} seed {s} done in {dt:.0f} s", flush=True)


# ------------------------------------------------------------------ per-seed summary
def _covers(iv, x) -> bool:
    return bool(iv[1] <= x <= iv[2])


def _v2rec(g: dict) -> dict | None:
    return g.get("gate_v2") or g.get("gate_v2_shadow")


def summarise(run: Path) -> dict:
    data = analyze.load(run)
    cfg = data["meta"]["config"]
    B, tol, eps_q = cfg["budget"], cfg["tol"], cfg["eps_q"]
    mt = analyze.metrics(data, cfg["recover_tol"], cfg["warmup_days"])
    n_days = mt["n_days"]
    out = {"run": str(run), "seed": cfg["seed"], "gate": cfg.get("gate", "v1"), "gate_holdout": cfg.get("gate_holdout", 0.0),
           "runtime_s": data["meta"]["runtime_s"], "regret_6_40": {a: mt["arms"][a]["regret"] for a in GATED + REF if a in mt["arms"]},
           "arms": {}}
    H = analyze.heldout_table(data["days"])
    gen_of_day = {d: gi for gi, a, b in mt["ranges"] for d in range(a, b + 1)}
    for arm in GATED:
        rows = sorted([g for g in data["gates"] if g["arm"] == arm], key=lambda g: g["day"])
        dec = [g["decision"] for g in rows]
        unacc = []
        for g in rows:
            t = g["truth"]
            if g["decision"] == "PROMOTE" and not t["acceptable"]:
                crit = [c for c, k in (("over budget", "over_budget"), ("quality regressed", "regressed")) if t[k]]
                unacc.append({"day": g["day"], "stage": g["generation"], "criteria": crit,
                              "cand_true_quality": t["candidate"]["quality"], "cand_true_cost": t["candidate"]["cost"],
                              "inc_true_quality": t["incumbent"]["quality"]})
        # days served by a policy that entered service through an unacceptable promotion
        prom_days = [g["day"] for g in rows if g["decision"] == "PROMOTE"]
        bad = {u["day"] for u in unacc}
        days_bad = 0
        for k, d in enumerate(prom_days):
            end = prom_days[k + 1] if k + 1 < len(prom_days) else n_days
            if d in bad:
                days_bad += max(0, min(end, n_days) - d)        # served on days d+1 .. end
        over_budget_days = sum(1 for d in H[arm] if H[arm][d][gen_of_day[d]]["cost"] > B + tol)
        over_budget_days_6 = sum(1 for d in H[arm] if d > cfg["warmup_days"] and H[arm][d][gen_of_day[d]]["cost"] > B + tol)
        false_rev = [g["day"] for g in rows if g["decision"] == "REVERT" and g["truth"]["acceptable"]]
        unev_acc = [g["day"] for g in rows if g["decision"] == "UNEVALUABLE" and g["truth"]["acceptable"]]
        right_rev = [g["day"] for g in rows if g["decision"] != "PROMOTE" and not g["truth"]["acceptable"]]
        # intervals: v1's own (candidate quality; decision cost net of prefix) and the v2 record (acted on or shadow)
        v1w = [g["dr_quality"][2] - g["dr_quality"][1] for g in rows]
        v1cq = [_covers(g["dr_quality"], g["truth"]["candidate"]["quality"]) for g in rows]
        v1cc = [_covers(g["dr_cost"], g["truth"]["candidate"]["cost"] - g["prefix_cost"]) for g in rows]
        v2 = [(g, _v2rec(g)) for g in rows if _v2rec(g) is not None]
        diffw = [r["estimates"]["diff_quality"][2] - r["estimates"]["diff_quality"][1] for g, r in v2]
        cov_d = [_covers(r["estimates"]["diff_quality"], g["truth"]["candidate"]["quality"] - g["truth"]["incumbent"]["quality"]) for g, r in v2]
        cov_t = [_covers(r["estimates"]["cand_total_cost"], g["truth"]["candidate"]["cost"]) for g, r in v2]
        cov_joint = [a and b for a, b in zip(cov_d, cov_t)]
        ess = [r["diagnostics"]["ess_frac"] for g, r in v2]
        clipf = [r["diagnostics"]["clip_frac"] for g, r in v2]
        # what the v2 rule would have decided (shadow, or the acted-on record)
        v2dec = [r["decision"] for g, r in v2]
        out["arms"][arm] = {
            "n_gates": len(rows), "PROMOTE": dec.count("PROMOTE"), "REVERT": dec.count("REVERT"), "UNEVALUABLE": dec.count("UNEVALUABLE"),
            "unacceptable_promotions": unacc, "n_unacceptable_promotions": len(unacc),
            "days_in_unacceptable_state": days_bad, "days_serving_over_budget": over_budget_days,
            "days_serving_over_budget_6_40": over_budget_days_6,
            "false_rejections_revert": false_rev, "n_false_rejections_revert": len(false_rev),
            "unevaluable_of_acceptable": unev_acc, "n_unevaluable_of_acceptable": len(unev_acc),
            "n_rejections_of_unacceptable": len(right_rev),
            "n_acceptable_candidates": sum(1 for g in rows if g["truth"]["acceptable"]),
            "first_promotion_day": min(prom_days) if prom_days else None,
            "v1_quality_ci_width": float(np.mean(v1w)), "v1_quality_ci_coverage": float(np.mean(v1cq)),
            "v1_decision_cost_ci_coverage": float(np.mean(v1cc)),
            "v2_record": "acted on" if cfg.get("gate") == "v2" else "shadow (not acted on)",
            "diff_ci_width": float(np.mean(diffw)) if diffw else None,
            "diff_ci_coverage": float(np.mean(cov_d)) if cov_d else None,
            "total_cost_ci_coverage": float(np.mean(cov_t)) if cov_t else None,
            "joint_coverage": float(np.mean(cov_joint)) if cov_joint else None,
            "mean_ess_frac": float(np.mean(ess)) if ess else None, "mean_clip_frac": float(np.mean(clipf)) if clipf else None,
            "v2_rule_decisions": {k: v2dec.count(k) for k in ("PROMOTE", "REVERT", "UNEVALUABLE")},
            "regret_6_40": mt["arms"][arm]["regret"],
        }
    return out


# ------------------------------------------------------------------ report
def _ms(xs, d=2) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    m = float(np.mean(xs))
    if len(xs) == 1:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} ± {float(np.std(xs, ddof=1) / np.sqrt(len(xs))):.{d}f}"


def report(root: Path = ROOT) -> str:
    S: dict[str, list[dict]] = {}
    for c in CONFIGS:
        runs = []
        for d in sorted((root / c).glob("seed*")):
            if d.is_dir() and (d / "meta.jsonl").exists():
                s = summarise(d)
                (root / c / f"{d.name}.summary.json").write_text(json.dumps(s, indent=1))
                runs.append(s)
        S[c] = sorted(runs, key=lambda r: r["seed"])
    (root / "summary.json").write_text(json.dumps(S, indent=1))
    seeds = sorted({r["seed"] for v in S.values() for r in v})
    rt = [r["runtime_s"] for v in S.values() for r in v]
    L = ["# Gate lifecycle error rates: v1 vs v2, with and without a held-out gate slice", "",
         f"Default schedule, 40 days, seeds {', '.join(map(str, seeds))}; `--arms all` (so v1 seed 0 is the committed run), "
         f"reported arms: continual (cgrpo), continual_awr, never_verify, always_verify; the per-stage oracle is the regret reference. "
         f"Runtime per run {np.mean(rt):.0f} s mean ({min(rt):.0f}–{max(rt):.0f} s), single-threaded NumPy. "
         "Generated by `python -m pace.experiments.mas_continual.gate_lifecycle report`.", "",
         "| config | gate rule | gate data |", "|---|---|---|",
         "| v1 | candidate DR-quality lower bound ≥ incumbent DR point − ε_q; DR decision-cost upper bound ≤ (B − mean prefix) + tol | the whole served window (also used for training) |",
         "| v1_h30 | same | 30% of each day's served episodes (by task id), never used by the trainers |",
         "| v2 | joint bootstrap: lower bound of (q_cand − q_inc) ≥ −ε_q; upper bound of total cost (prefix inside the resampling) ≤ B + tol; "
         "UNEVALUABLE if per-decision ESS/n < 0.2 or > 10% of cumulative ratios hit the clip (50) | whole window |",
         "| v2_h30 | same | 30% slice |", "",
         f"B = 0.090, tol = 0.005, ε_q = 0.05, 200 bootstrap resamples, gate window = since the last harness event, at most 5 days "
         "(with holdout, the same window over the 30% slice: ≈ 29 episodes per day). ",
         "**Unacceptable** (simulator, against the held-out reference of 150 tasks per stage): candidate held-out quality < incumbent held-out "
         "quality − ε_q, OR candidate held-out total cost > B + tol. Gate counts are over all 40 days (the gate runs from day 1); "
         "regret is days 6–40. Coverage is the share of gates whose 90% interval contains the held-out reference value, "
         "**against the held-out reference**, which itself carries Monte Carlo error; it is not coverage of an exact truth.", ""]

    # ---- table 1: headline per config x arm, mean ± SE over seeds
    L += ["## 1. Lifecycle summary (mean ± SE over seeds)", "",
          "| config | arm | regret 6–40 | PROMOTE | REVERT | UNEVALUABLE | unacceptable promotions | P(≥1 unacceptable promotion) | "
          "days in unacceptable state | false rejections (REVERT of acceptable) | UNEVALUABLE of acceptable | first promotion day | "
          "decision-interval width (quality) | diff CI coverage | total-cost CI coverage | joint coverage |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, runs in S.items():
        if not runs:
            continue
        for arm in GATED:
            A = [r["arms"][arm] for r in runs]
            k = sum(1 for a in A if a["n_unacceptable_promotions"] > 0)
            fp = [a["first_promotion_day"] for a in A]
            fp_s = _ms([x for x in fp if x is not None], 1) + (f" ({sum(x is None for x in fp)} never)" if any(x is None for x in fp) else "")
            width = _ms([a["diff_ci_width"] for a in A], 3) if c.startswith("v2") else (
                _ms([a["v1_quality_ci_width"] for a in A], 3) + " (cand. q)")
            cov_lbl = "" if c.startswith("v2") else " (shadow)"
            L.append(f"| {LABEL[c]} | {arm} | {_ms([a['regret_6_40'] for a in A])} | {_ms([a['PROMOTE'] for a in A], 1)} | "
                     f"{_ms([a['REVERT'] for a in A], 1)} | {_ms([a['UNEVALUABLE'] for a in A], 1)} | {_ms([a['n_unacceptable_promotions'] for a in A], 1)} | "
                     f"{k}/{len(A)} | {_ms([a['days_in_unacceptable_state'] for a in A], 1)} | {_ms([a['n_false_rejections_revert'] for a in A], 1)} | "
                     f"{_ms([a['n_unevaluable_of_acceptable'] for a in A], 1)} | {fp_s} | {width} | "
                     f"{_ms([100 * a['diff_ci_coverage'] for a in A], 0)}%{cov_lbl} | {_ms([100 * a['total_cost_ci_coverage'] for a in A], 0)}%{cov_lbl} | "
                     f"{_ms([100 * a['joint_coverage'] for a in A], 0)}%{cov_lbl} |")
    L += ["", "Reference arms, regret 6–40 (same runs): " + "; ".join(
        f"{LABEL[c]}: " + ", ".join(f"{a} {_ms([r['regret_6_40'][a] for r in runs])}" for a in REF)
        for c, runs in S.items() if runs) + ". The oracle's regret is 0 by definition.", "",
        "`decision-interval width`: for v2 the width of the 90% interval of q_cand − q_inc (the quantity tested); for v1 the width of the "
        "candidate's DR-quality interval (v1 tests its lower bound against the incumbent's point estimate, so there is no difference interval). "
        "For v1 configurations the coverage columns score the v2 intervals computed in shadow on the same logs.", ""]

    # ---- table 2: per seed
    L += ["## 2. Per seed", "",
          "| config | arm | seed | regret 6–40 | P / R / U | unacceptable promotions (day: criterion) | days unacceptable | false REVERT | U of acceptable | first promotion | width | diff cov | cost cov | mean ESS/n | v1 cand-q cov / v1 decision-cost cov |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, runs in S.items():
        for arm in GATED:
            for r in runs:
                a = r["arms"][arm]
                ua = "; ".join(f"d{u['day']}: {' + '.join(u['criteria'])}" for u in a["unacceptable_promotions"]) or "–"
                w = a["diff_ci_width"] if c.startswith("v2") else a["v1_quality_ci_width"]
                L.append(f"| {c} | {arm} | {r['seed']} | {a['regret_6_40']:.2f} | {a['PROMOTE']} / {a['REVERT']} / {a['UNEVALUABLE']} | {ua} | "
                         f"{a['days_in_unacceptable_state']} | {a['n_false_rejections_revert']} | {a['n_unevaluable_of_acceptable']} | "
                         f"{a['first_promotion_day'] if a['first_promotion_day'] is not None else '–'} | {w:.3f} | "
                         f"{100 * a['diff_ci_coverage']:.0f}% | {100 * a['total_cost_ci_coverage']:.0f}% | {a['mean_ess_frac']:.2f} | "
                         f"{100 * a['v1_quality_ci_coverage']:.0f}% / {100 * a['v1_decision_cost_ci_coverage']:.0f}% |")
    L += ["", "P / R / U = PROMOTE / REVERT / UNEVALUABLE. `days unacceptable` = days served by a policy that entered service through an "
          "unacceptable promotion (from the day after the promotion to the next promotion or day 40).", ""]

    # ---- table 3: every unacceptable promotion
    L += ["## 3. Every unacceptable promotion", "",
          "| config | arm | seed | day | stage | violated | candidate held-out quality / cost | incumbent held-out quality |", "|---|---|---|---|---|---|---|---|"]
    n_any = 0
    for c, runs in S.items():
        for arm in GATED:
            for r in runs:
                for u in r["arms"][arm]["unacceptable_promotions"]:
                    n_any += 1
                    L.append(f"| {c} | {arm} | {r['seed']} | {u['day']} | {u['stage']} | {' + '.join(u['criteria'])} | "
                             f"{u['cand_true_quality']:.3f} / ${u['cand_true_cost']:.4f} | {u['inc_true_quality']:.3f} |")
    if not n_any:
        L.append("| (none) | | | | | | | |")
    L += ["", "## 4. Over-budget service and what the v2 rule would have said (shadow in v1 configs)", "",
          "| config | arm | days serving a policy with held-out cost > B + tol (days 1–40 / 6–40) | v2-rule decisions P / R / U (summed over seeds) | mean clip fraction |",
          "|---|---|---|---|---|"]
    for c, runs in S.items():
        for arm in GATED:
            A = [r["arms"][arm] for r in runs]
            if not A:
                continue
            tot = {k: sum(a["v2_rule_decisions"][k] for a in A) for k in ("PROMOTE", "REVERT", "UNEVALUABLE")}
            L.append(f"| {c} | {arm} | {_ms([a['days_serving_over_budget'] for a in A], 1)} / {_ms([a['days_serving_over_budget_6_40'] for a in A], 1)} | "
                     f"{tot['PROMOTE']} / {tot['REVERT']} / {tot['UNEVALUABLE']} | {_ms([a['mean_clip_frac'] for a in A], 3)} |")
    L += ["", "Raw gate records: `<config>/seed<k>/gates.jsonl` (v1 configs carry the shadow v2 record under `gate_v2_shadow`, "
          "v2 configs the acted-on record under `gate_v2`); per-seed summaries `<config>/seed<k>.summary.json`; all of them in `summary.json`.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Gate lifecycle grid (v1/v2 x holdout) and report.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--configs", nargs="+", default=list(CONFIGS))
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--root", default=str(ROOT))
    p = sub.add_parser("report")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--prelude", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_grid(a.configs, a.seeds, a.workers, Path(a.root))
    else:
        text = report(Path(a.root))
        if a.prelude and Path(a.prelude).exists():
            head, _, tail = text.partition("\n\n")
            text = head + "\n\n## 0. Findings\n\n" + Path(a.prelude).read_text().strip() + "\n\n" + tail
        (Path(a.root) / "RESULTS.md").write_text(text)
        print(f"[gate_lifecycle] wrote {Path(a.root) / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
