"""Table 1 at a COMMON evaluation price: CRMArena-Pro held-out tier policies, B2B and B2C.

Reproduces the 5-fold category-stratified CV of scripts/crmarena_offline.py (same folds, seed 0,
same trainers, imported from there) and reports every policy row at the same price of cost:
J = quality - lam * cost at lam = 2 and lam = 5 per dollar, for every row INCLUDING the
constrained cgrpo-lookup rows (crmarena_offline.py's constrained table reports J at each fold's
own learned lam_bar instead). The learned multiplier is reported separately, for the constrained
rows only, as a training quantity.

  python scripts/crmarena_table1.py            # both orgs, ~1-2 min on a laptop CPU, deterministic

Writes results/crmarena_300/table1_common_price.{json,md} (B2B, 300 tasks) and
results/crmarena_b2c/table1_common_price.{json,md} (B2C, 150 tasks). Reads only the logs; the
spend snapshot path is recorded for provenance (no repricing happens here). Nothing committed
is overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import crmarena_offline as off  # noqa: E402
from pace.lookup import LookupTable, fixed, kfold_tasks, uniform  # noqa: E402
from pace.prices import TIERS  # noqa: E402
from pace.trace import read_jsonl  # noqa: E402

EVAL_LAMS = (2.0, 5.0)
ORGS = {
    "b2b": {"log": ROOT / "traces" / "crmarena_b2b_counterfactual.jsonl",
            "spend": ROOT / "traces" / "crmarena_b2b_counterfactual.spend.jsonl",
            "out": ROOT / "results" / "crmarena_300"},
    "b2c": {"log": ROOT / "traces" / "crmarena_b2c_counterfactual.jsonl",
            "spend": ROOT / "traces" / "b2c" / "spend.jsonl",
            "out": ROOT / "results" / "crmarena_b2c"},
}
ROW_ORDER = [
    "fixed cheap", "fixed mid", "fixed frontier", "uniform behavior",
    "AWR+IPS linear, trained at λ=2 (served, stochastic)", "AWR+IPS linear, trained at λ=2 (greedy)",
    "per-category rule, train-fold argmax of q − 2c",
    "AWR+IPS linear, trained at λ=5 (served, stochastic)", "AWR+IPS linear, trained at λ=5 (greedy)",
    "per-category rule, train-fold argmax of q − 5c",
    "constrained cgrpo-lookup, B=$0.10 (served: iterate mix, stochastic)",
    "constrained cgrpo-lookup, B=$0.10 (greedy of the mix)",
    "per-task oracle (argmax q − 2c)", "per-task oracle (argmax q − 5c)",
    "per-task oracle (max quality, cheapest among ties)",
]
CONSTRAINED = ROW_ORDER[10:12]


def load_org(org: str):
    cfg = ORGS[org]
    eps = read_jsonl(cfg["log"])
    T = LookupTable.from_episodes(eps)
    folds = kfold_tasks(T, 5, seed=0)
    assert all(not (set(tr) & set(te)) and len(set(tr) | set(te)) == len(T) for tr, te in folds)
    return eps, T, folds


def summarize(per_fold: list[tuple[float, float]]) -> dict:
    """Means over folds of q and c, J at each evaluation price with its across-fold SD (np.std,
    ddof=0, as in crmarena_offline.py), and the feasibility flags on the mean cost."""
    q = np.array([x[0] for x in per_fold]); c = np.array([x[1] for x in per_fold])
    r = {"q": float(q.mean()), "c": float(c.mean()), "per_fold_q": q.tolist(), "per_fold_c": c.tolist()}
    for lam in EVAL_LAMS:
        J = q - lam * c
        r[f"J_lam{lam:g}"] = float(J.mean())
        r[f"J_lam{lam:g}_sd"] = float(J.std())
    r["cost_le_B"] = bool(r["c"] <= off.BUDGET + 1e-12)
    r["cost_le_B_tol"] = bool(r["c"] <= off.BUDGET + off.TOL + 1e-12)
    r["folds_cost_le_B_tol"] = int((c <= off.BUDGET + off.TOL + 1e-12).sum())
    return r


def table1(eps, T, folds) -> dict:
    d = T.X.shape[1]
    acc: dict[str, list] = {k: [] for k in ROW_ORDER}
    lams = []
    for fi, (tr, te) in enumerate(folds):
        Ttr, Tte = T.subset(tr), T.subset(te)
        n = len(Tte)
        Ps = {f"fixed {t}": fixed(t, n) for t in TIERS}
        Ps["uniform behavior"] = uniform(n)
        for lam in (2.0, 5.0):
            pol = off.awr_policy(eps, tr, lam, d)
            P = Tte.probs(pol)
            Ps[f"AWR+IPS linear, trained at λ={lam:g} (served, stochastic)"] = P
            Ps[f"AWR+IPS linear, trained at λ={lam:g} (greedy)"] = off.greedy(P)
            Ps[f"per-category rule, train-fold argmax of q − {lam:g}c"] = off.plugin_rule(Ttr, Tte, lam)
            Ps[f"per-task oracle (argmax q − {lam:g}c)"] = Tte.oracle(lam)
        pol, lb = off.cgrpo_policy(Ttr, seed=fi)
        lams.append(float(lb))
        P = Tte.probs(pol)
        Ps[CONSTRAINED[0]] = P
        Ps[CONSTRAINED[1]] = off.greedy(P)
        Ps["per-task oracle (max quality, cheapest among ties)"] = Tte.oracle()
        for k in ROW_ORDER:
            acc[k].append(Tte.value(Ps[k]))
    rows = {k: summarize(v) for k, v in acc.items()}
    for k in CONSTRAINED:
        rows[k]["train_lambda_bar_per_fold"] = lams
        rows[k]["train_lambda_bar_mean"] = float(np.mean(lams))
        # what crmarena_offline.py's constrained table reports: J at each fold's own lam_bar
        rows[k]["J_at_fold_lambda_bar"] = float(np.mean([q - lb * c for (q, c), lb in zip(acc[k], lams)]))
    return {"rows": rows, "lambda_bar_per_fold": lams, "lambda_bar_mean": float(np.mean(lams))}


def check_committed(res: dict, committed: Path) -> dict:
    """Max |diff| of q and c against the committed results.json rows that this table re-derives
    (lam=2 AWR / rule / fixed / uniform / oracle rows, the constrained rows, the fold lam_bars)."""
    if not committed.exists():
        return {"available": False}
    old = json.loads(committed.read_text())["cv"]
    pairs = {"fixed cheap": ("lam2", "fixed cheap"), "fixed mid": ("lam2", "fixed mid"),
             "fixed frontier": ("lam2", "fixed frontier"), "uniform behavior": ("lam2", "uniform behavior"),
             "AWR+IPS linear, trained at λ=2 (served, stochastic)": ("lam2", "AWR+IPS linear (served, stochastic)"),
             "AWR+IPS linear, trained at λ=2 (greedy)": ("lam2", "AWR+IPS linear (greedy)"),
             "per-category rule, train-fold argmax of q − 2c": ("lam2", "per-category plug-in rule (train-fold argmax)"),
             "AWR+IPS linear, trained at λ=5 (greedy)": ("lam5", "AWR+IPS linear (greedy)"),
             "per-category rule, train-fold argmax of q − 5c": ("lam5", "per-category plug-in rule (train-fold argmax)"),
             "per-task oracle (argmax q − 2c)": ("lam2", "per-task oracle (argmax q - 2c)"),
             CONSTRAINED[0]: ("constrained", "constrained cgrpo-lookup linear (served: iterate mix)"),
             CONSTRAINED[1]: ("constrained", "constrained cgrpo-lookup linear (greedy of the mix)"),
             "per-task oracle (max quality, cheapest among ties)": ("constrained", "per-task oracle (max quality, cheapest among ties)")}
    diff = 0.0
    for new, (sec, name) in pairs.items():
        o = old[sec]["rows"][name]
        diff = max(diff, abs(o["q"] - res["rows"][new]["q"]), abs(o["c"] - res["rows"][new]["c"]))
    dl = float(np.max(np.abs(np.array(old["constrained"]["lambda_bar_per_fold"]) - np.array(res["lambda_bar_per_fold"]))))
    return {"available": True, "max_abs_diff_q_c": diff, "max_abs_diff_lambda_bar": dl,
            "committed_J_constrained_served_at_fold_lambda": old["constrained"]["rows"]
            ["constrained cgrpo-lookup linear (served: iterate mix)"]["J"],
            "committed_J_constrained_greedy_at_fold_lambda": old["constrained"]["rows"]
            ["constrained cgrpo-lookup linear (greedy of the mix)"]["J"]}


def lambda_note(org: str, res: dict, out: Path) -> dict:
    """Where a reported lam_bar comes from: the CV fold mean vs the all-task (gate-section) policy."""
    note = {"cv_lambda_bar_per_fold": res["lambda_bar_per_fold"], "cv_lambda_bar_mean": res["lambda_bar_mean"]}
    rj = out / "results.json"
    if rj.exists():
        g = json.loads(rj.read_text())["gate"]
        note["all_task_gate_policy_lambda_bar"] = g["lambda_bar"]
    tm = out / "tables.md"
    if tm.exists():
        note["lines_in_tables_md_with_1.35"] = [i + 1 for i, s in enumerate(tm.read_text().splitlines()) if "1.35" in s]
    return note


def fmt(x: float, k: int = 6) -> str:
    return f"{x:.{k}f}"


def to_md(org: str, n: int, res: dict, chk: dict, note: dict) -> str:
    B, tol = off.BUDGET, off.TOL
    L = [f"## Table 1 at a common price: CRMArena-Pro {org.upper()}, {n} tasks, 5-fold CV (seed 0)", "",
         "Held-out means over the 5 test folds (exact lookups). J(λ) = quality − λ·cost evaluated at the SAME "
         "λ for every row; [SD] is the across-fold SD of J (np.std, ddof 0). Feasibility flags are on the "
         f"mean held-out cost: ≤ B = ${B:.3f} and ≤ B + tol = ${B + tol:.3f}; 'folds ok' counts folds whose own "
         "held-out cost is ≤ B + tol. Training λ̄ is shown for the constrained rows only (a training "
         "quantity, not the evaluation price). AWR+IPS and the per-category rule are trained at a fixed "
         "λ (2 or 5); each is still evaluated at both prices.", "",
         "| policy | quality | cost (USD) | J(λ=2) [SD] | J(λ=5) [SD] | train λ̄ (mean; per fold) | cost ≤ B | cost ≤ B+tol | folds ok |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in ROW_ORDER:
        r = res["rows"][k]
        lb = "–"
        if "train_lambda_bar_mean" in r:
            lb = fmt(r["train_lambda_bar_mean"]) + "; " + ", ".join(fmt(x) for x in r["train_lambda_bar_per_fold"])
        L.append(f"| {k} | {fmt(r['q'])} | {fmt(r['c'])} | {fmt(r['J_lam2'])} [{fmt(r['J_lam2_sd'])}] | "
                 f"{fmt(r['J_lam5'])} [{fmt(r['J_lam5_sd'])}] | {lb} | {'yes' if r['cost_le_B'] else 'NO'} | "
                 f"{'yes' if r['cost_le_B_tol'] else 'NO'} | {r['folds_cost_le_B_tol']}/5 |")
    s, g = res["rows"][CONSTRAINED[0]], res["rows"][CONSTRAINED[1]]
    L += ["", "### Constrained rows: common price vs fold λ̄", "",
          f"- served (iterate mix): J(λ=2) = {fmt(s['J_lam2'])}; J at each fold's λ̄ = {fmt(s['J_at_fold_lambda_bar'])}",
          f"- greedy of the mix: J(λ=2) = {fmt(g['J_lam2'])}; J at each fold's λ̄ = {fmt(g['J_at_fold_lambda_bar'])}",
          "", "### λ̄ provenance", "",
          f"- CV, per fold: {', '.join(fmt(x) for x in note['cv_lambda_bar_per_fold'])}; mean **{fmt(note['cv_lambda_bar_mean'])}**"]
    if "all_task_gate_policy_lambda_bar" in note:
        L.append(f"- the all-{n}-task constrained policy of the gate / drift sections (results.json gate.lambda_bar): "
                 f"**{fmt(note['all_task_gate_policy_lambda_bar'])}**")
    if "lines_in_tables_md_with_1.35" in note:
        L.append(f"- lines of tables.md containing '1.35': {note['lines_in_tables_md_with_1.35'] or 'none'}")
    if chk.get("available"):
        L += ["", "### Reproduction check against the committed results.json", "",
              f"max |Δ| over re-derived q and c: {chk['max_abs_diff_q_c']:.3e}; max |Δ| fold λ̄: "
              f"{chk['max_abs_diff_lambda_bar']:.3e}."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orgs", nargs="*", default=["b2b", "b2c"], choices=list(ORGS))
    a = ap.parse_args(argv)
    for org in a.orgs:
        eps, T, folds = load_org(org)
        out = ORGS[org]["out"]
        res = table1(eps, T, folds)
        chk = check_committed(res, out / "results.json")
        note = lambda_note(org, res, out)
        doc = {"org": org, "n_tasks": len(T), "log": str(ORGS[org]["log"].relative_to(ROOT)),
               "spend_snapshot": str(ORGS[org]["spend"].relative_to(ROOT)), "budget": off.BUDGET, "tol": off.TOL,
               "eval_lambdas": list(EVAL_LAMS), "folds": [{"train": tr, "test": te} for tr, te in folds],
               **res, "lambda_provenance": note, "reproduction_check": chk}
        (out / "table1_common_price.json").write_text(json.dumps(doc, indent=1))
        md = to_md(org, len(T), res, chk, note)
        (out / "table1_common_price.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
