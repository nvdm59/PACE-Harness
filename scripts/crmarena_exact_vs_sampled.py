"""Exact-gradient constrained trainer and category LP vs sampled cgrpo-lookup, CRMArena-Pro B2B + B2C.

Every arm of every task is logged, so the Lagrangian objective on the training tasks is a finite
sum and needs no sampling. Under the 5-fold category-stratified CV of scripts/crmarena_offline.py
(same folds, seed 0) this compares, per org:

  sampled cgrpo-lookup     train_cgrpo_lookup (as in the committed tables; seed = fold index)
  exact, full batch        pace.train_exact.train_exact_lookup, batch=None (deterministic)
  exact, minibatch 32      same, 32 tasks per iteration (seed = fold index), exact over arms
  category LP              pace.train_exact.category_lp on training-fold (category, tier) means
  context-free arm mix     crmarena_offline.arm_mixture (the committed reference row)

each served (stochastic) and greedy, with held-out q, cost, J at lam = 2 (and 5), fold SD,
feasibility at B and B + tol, wall-clock per fold, and the number of table reads used.

  python scripts/crmarena_exact_vs_sampled.py        # both orgs, deterministic

Writes results/crmarena_300/exact_vs_sampled.{json,md} and results/crmarena_b2c/exact_vs_sampled.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import crmarena_offline as off  # noqa: E402
from crmarena_table1 import ORGS, load_org, summarize  # noqa: E402
from pace.train_exact import category_lp, lp_probs, train_exact_lookup  # noqa: E402
from pace.train_offline import train_cgrpo_lookup  # noqa: E402

ITERS, BATCH, GROUP = 300, 32, 8          # train_cgrpo_lookup defaults


def greedy_rows(P: np.ndarray) -> np.ndarray:
    return off.greedy(P)


def run(eps, T, folds) -> dict:
    acc: dict[str, list] = {}
    meta: dict[str, dict] = {}

    def add(name, Tte, P, secs, reads, lam=None, extra=None):
        acc.setdefault(name, []).append(Tte.value(P))
        m = meta.setdefault(name, {"wall_s": [], "reads": [], "lambda": [], "extra": []})
        m["wall_s"].append(secs); m["reads"].append(reads)
        if lam is not None:
            m["lambda"].append(float(lam))
        if extra is not None:
            m["extra"].append(extra)

    for fi, (tr, te) in enumerate(folds):
        Ttr, Tte = T.subset(tr), T.subset(te)
        # sampled cgrpo-lookup (the committed trainer, same call as crmarena_offline.cgrpo_policy)
        t0 = time.perf_counter()
        pol, lb = train_cgrpo_lookup(Ttr, off.BUDGET, seed=fi, serve_eps=0.0)
        s = time.perf_counter() - t0
        P = Tte.probs(pol)
        n_tr = ITERS * min(BATCH, len(Ttr)) * GROUP
        add("sampled cgrpo-lookup (served: iterate mix)", Tte, P, s, n_tr, lb)
        add("sampled cgrpo-lookup (greedy of the mix)", Tte, greedy_rows(P), s, n_tr, lb)
        # exact gradient, full batch
        t0 = time.perf_counter()
        pol, lb = train_exact_lookup(Ttr, off.BUDGET, iters=ITERS, batch=None, seed=fi)
        s = time.perf_counter() - t0
        P = Tte.probs(pol)
        add("exact-gradient, full batch (served: iterate mix)", Tte, P, s, pol.meta["n_lookups"], lb)
        add("exact-gradient, full batch (greedy of the mix)", Tte, greedy_rows(P), s, pol.meta["n_lookups"], lb)
        # exact gradient, minibatch of 32 tasks (task sampling only)
        t0 = time.perf_counter()
        pol, lb = train_exact_lookup(Ttr, off.BUDGET, iters=ITERS, batch=BATCH, seed=fi)
        s = time.perf_counter() - t0
        P = Tte.probs(pol)
        add("exact-gradient, minibatch 32 (served: iterate mix)", Tte, P, s, pol.meta["n_lookups"], lb)
        add("exact-gradient, minibatch 32 (greedy of the mix)", Tte, greedy_rows(P), s, pol.meta["n_lookups"], lb)
        # category LP on training-fold means
        t0 = time.perf_counter()
        lp = category_lp(Ttr.cats, Ttr.Q, Ttr.C, off.BUDGET)
        fb = off.arm_mixture(Ttr, 1)[0][0]
        P = lp_probs(lp, Tte.cats, fb)
        s = time.perf_counter() - t0
        n_rand = int(((lp["p"] > 1e-12).sum(1) > 1).sum())
        ex = {"train_q": lp["value_q"], "train_c": lp["value_c"], "feasible": lp["feasible"],
              "n_randomized_categories": n_rand,
              "n_test_tasks_unseen_category": int(sum(c not in lp["cats"] for c in Tte.cats))}
        add("category LP, train-fold means (served, stochastic)", Tte, P, s, Ttr.Q.size, lp["mu"], ex)
        add("category LP, train-fold means (greedy: argmax tier per category)", Tte, greedy_rows(P), s,
            Ttr.Q.size, lp["mu"], ex)
        t0 = time.perf_counter()
        P = off.arm_mixture(Ttr, len(Tte))[0]
        s = time.perf_counter() - t0
        add("best fixed-arm mixture at cost <= B (train-fold LP, context-free)", Tte, P, s, Ttr.Q.size)
    rows = {}
    for k, v in acc.items():
        r = summarize(v)
        m = meta[k]
        r["wall_s_per_fold"] = m["wall_s"]; r["wall_s_mean"] = float(np.mean(m["wall_s"]))
        r["table_reads_per_fold"] = m["reads"]; r["table_reads_mean"] = float(np.mean(m["reads"]))
        if m["lambda"]:
            r["multiplier_per_fold"] = m["lambda"]; r["multiplier_mean"] = float(np.mean(m["lambda"]))
        if m["extra"]:
            r["lp_per_fold"] = m["extra"]
        rows[k] = r
    return rows


def to_md(org: str, n: int, rows: dict) -> str:
    f = lambda x: f"{x:.6f}"
    L = [f"## Exact vs sampled constrained training: CRMArena-Pro {org.upper()}, {n} tasks, 5-fold CV (seed 0)", "",
         f"Held-out means over the 5 test folds (exact lookups). J at a common λ (2 and 5 per dollar). Budget "
         f"B = ${off.BUDGET:.2f}, tol ${off.TOL:.3f}; flags on the mean held-out cost; 'folds ok' = folds with "
         "held-out cost ≤ B + tol. 'table reads' = (task, tier) cells read by the trainer: sampled = "
         f"iters·batch·group = {ITERS}·{BATCH}·{GROUP} sampled transitions; exact = iters·tasks·3; LP = one pass over "
         "the training cells. Multiplier: learned λ̄ (trainers) or the LP's dual price μ of the budget "
         "constraint. Wall-clock is training time per fold on this machine (CPU).", "",
         "| policy | quality | cost (USD) | J(λ=2) [SD] | J(λ=5) [SD] | cost ≤ B | cost ≤ B+tol | folds ok | "
         "multiplier (mean) | wall s / fold | table reads / fold |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        mult = f(r["multiplier_mean"]) if "multiplier_mean" in r else "–"
        L.append(f"| {k} | {f(r['q'])} | {f(r['c'])} | {f(r['J_lam2'])} [{f(r['J_lam2_sd'])}] | "
                 f"{f(r['J_lam5'])} [{f(r['J_lam5_sd'])}] | {'yes' if r['cost_le_B'] else 'NO'} | "
                 f"{'yes' if r['cost_le_B_tol'] else 'NO'} | {r['folds_cost_le_B_tol']}/5 | {mult} | "
                 f"{r['wall_s_mean']:.3f} | {r['table_reads_mean']:.0f} |")
    lp = rows["category LP, train-fold means (served, stochastic)"]["lp_per_fold"]
    L += ["", "Category LP per fold (training-fold objective): " + "; ".join(
        f"fold {i}: q {x['train_q']:.4f}, c {x['train_c']:.4f}, randomized categories {x['n_randomized_categories']}, "
        f"unseen test categories {x['n_test_tasks_unseen_category']}" for i, x in enumerate(lp)) + "."]
    mus = rows["category LP, train-fold means (served, stochastic)"]["multiplier_per_fold"]
    if all(m == 0.0 for m in mus):
        L += ["", "The budget constraint is slack in every fold's LP (μ = 0, no randomized category): on training-fold "
              "category means, the highest-quality tier of every category already costs ≤ B on average, so the "
              "LP optimum is the per-category max-quality tier (cheaper on quality ties), and its stochastic and "
              "greedy rows coincide."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orgs", nargs="*", default=["b2b", "b2c"], choices=list(ORGS))
    a = ap.parse_args(argv)
    for org in a.orgs:
        eps, T, folds = load_org(org)
        rows = run(eps, T, folds)
        out = ORGS[org]["out"]
        doc = {"org": org, "n_tasks": len(T), "budget": off.BUDGET, "tol": off.TOL, "iters": ITERS,
               "batch": BATCH, "group": GROUP, "rows": rows}
        (out / "exact_vs_sampled.json").write_text(json.dumps(doc, indent=1))
        md = to_md(org, len(T), rows)
        (out / "exact_vs_sampled.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()
