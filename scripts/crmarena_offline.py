"""Linear controller on the real CRMArena-Pro b2b log: CV, DR gate, repricing drift.

Offline only: reads traces/crmarena_b2b_counterfactual.jsonl (50 tasks x 3 tier arms) and
traces/crmarena_b2b_counterfactual.spend.jsonl (the collection's spend-ledger snapshot: token
counts, for repricing). No API, no Salesforce, no GPU.

  python scripts/crmarena_offline.py            # about 20 s on a laptop CPU, deterministic (seed 0)

Writes results/crmarena/tables.md, results/crmarena/results.json and the policies
(results/crmarena/policies/*.json; fold 0's constrained policy is Part B's linear row).
Everything is seeded (folds, trainers, redraws, bootstraps), so the output is byte-identical
across runs on the same log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.gate import estimate_bandit, format_table, gate, on_policy  # noqa: E402
from pace.lookup import (DECISION_ACTIONS, LookupTable, fixed, fixed_arm_log, kfold_tasks,  # noqa: E402
                         reprice_episodes, uniform)
from pace.prices import TIERS  # noqa: E402
from pace.trace import read_jsonl  # noqa: E402
from pace.train_offline import train_awr, train_cgrpo_lookup  # noqa: E402

LAMS = [2.0, 5.0, 10.0]
BUDGET = 0.10
TOL = 0.005            # USD over budget still counted as within budget (gate tol too)
EPS_Q = 0.03
EVENTS = [("frontier", 1.5), ("mid", 0.5), ("cheap", 3.0)]


def plugin_rule(train: LookupTable, test: LookupTable, lam: float) -> np.ndarray:
    """Per-category argmax tier of the train-fold mean quality - lam * cost (global argmax for a
    category absent from the train fold). This is the label the phase-2 SFT stage imitates."""
    g = (train.Q - lam * train.C).mean(0)
    best = {}
    for c in set(train.cats):
        ix = [i for i, x in enumerate(train.cats) if x == c]
        s = (train.Q[ix] - lam * train.C[ix]).mean(0)
        best[c] = int(np.argmax(s - 1e-9 * np.arange(3)))       # ties -> the cheaper tier
    P = np.zeros((len(test), 3))
    for i, c in enumerate(test.cats):
        P[i, best.get(c, int(np.argmax(g)))] = 1.0
    return P


def arm_mixture(train: LookupTable, n: int) -> tuple[np.ndarray, str]:
    """Best context-free randomized mix of the fixed arms with train-fold mean cost <= B
    (a two-arm LP on the train fold), as tier probabilities for n test tasks."""
    q, c = train.Q.mean(0), train.C.mean(0)
    best = (-1.0, None, "")
    for i in range(3):
        if c[i] <= BUDGET and q[i] > best[0]:
            best = (q[i], np.eye(3)[i], f"{TIERS[i]} only")
        for j in range(3):
            if c[i] <= BUDGET < c[j]:
                w = (c[j] - BUDGET) / (c[j] - c[i])
                qq = w * q[i] + (1 - w) * q[j]
                if qq > best[0]:
                    p = np.zeros(3); p[i], p[j] = w, 1 - w
                    best = (qq, p, f"{w:.2f} {TIERS[i]} + {1 - w:.2f} {TIERS[j]}")
    return np.tile(best[1], (n, 1)), best[2]


def greedy(P: np.ndarray) -> np.ndarray:
    G = np.zeros_like(P)
    G[np.arange(len(P)), P.argmax(1)] = 1.0
    return G


def awr_policy(eps, train_ids, lam, d):
    s = set(train_ids)
    return train_awr([e for e in eps if e.task_id in s], DECISION_ACTIONS, lam=lam, first_step_only=True,
                     legal=DECISION_ACTIONS, n_features=d, serve_eps=0.0)


def cgrpo_policy(table, seed=0, init=None):
    return train_cgrpo_lookup(table, BUDGET, seed=seed, init=init, serve_eps=0.0)


def ok(c):
    return "yes" if c <= BUDGET + TOL + 1e-12 else "NO"


def fmt_rows(rows: dict, lam_label: str) -> list[str]:
    out = [f"| policy | quality | cost/task (USD) | J ({lam_label}) [fold SD] | within budget (<= ${BUDGET + TOL:.3f}) |",
           "|---|---|---|---|---|"]
    for name, r in rows.items():
        out.append(f"| {name} | {r['q']:.3f} | {r['c']:.4f} | {r['J']:.3f} [{r['J_sd']:.3f}] | {ok(r['c'])} |")
    return out


def cv_tables(eps, T, folds):
    d = T.X.shape[1]
    res = {}
    lines = []
    # ---------------------------------------------------------- fixed lambda: AWR+IPS
    for lam in LAMS:
        acc: dict[str, list] = {}
        ins = []
        for tr, te in folds:
            Ttr, Tte = T.subset(tr), T.subset(te)
            pol = awr_policy(eps, tr, lam, d)
            Ps = {"AWR+IPS linear (served, stochastic)": Tte.probs(pol),
                  "AWR+IPS linear (greedy)": greedy(Tte.probs(pol)),
                  "per-category plug-in rule (train-fold argmax)": plugin_rule(Ttr, Tte, lam),
                  "uniform behavior": uniform(len(Tte))}
            Ps.update({f"fixed {t}": fixed(t, len(Tte)) for t in TIERS})
            Ps[f"per-task oracle (argmax q - {lam:g}c)"] = Tte.oracle(lam)
            for k, P in Ps.items():
                q, c = Tte.value(P)
                acc.setdefault(k, []).append((q, c, q - lam * c))
            qi, ci = Ttr.value(Ttr.probs(pol))
            ins.append(qi - lam * ci)
        rows = {k: {"q": float(np.mean([x[0] for x in v])), "c": float(np.mean([x[1] for x in v])),
                    "J": float(np.mean([x[2] for x in v])), "J_sd": float(np.std([x[2] for x in v]))}
                for k, v in acc.items()}
        res[f"lam{lam:g}"] = {"rows": rows, "awr_J_in_sample": float(np.mean(ins))}
        lines += [f"#### λ = {lam:g} per dollar (J = quality − {lam:g}·cost)", ""] + fmt_rows(rows, f"λ={lam:g}")
        lines += ["", f"AWR+IPS in-sample J on the ~{int(round(0.8*len(T)))} training tasks of each fold (mean over folds): {np.mean(ins):.3f} "
                      f"(held-out: {rows['AWR+IPS linear (served, stochastic)']['J']:.3f}).", ""]
    # ---------------------------------------------------- constrained: B = $0.10
    acc, lams, ins, fold0 = {}, [], [], None
    for fi, (tr, te) in enumerate(folds):
        Ttr, Tte = T.subset(tr), T.subset(te)
        pol, lb = cgrpo_policy(Ttr, seed=fi)
        lams.append(lb)
        if fi == 0:
            fold0 = pol
        P = Tte.probs(pol)
        Ps = {"constrained cgrpo-lookup linear (served: iterate mix)": P,
              "constrained cgrpo-lookup linear (greedy of the mix)": greedy(P),
              "uniform behavior": uniform(len(Tte))}
        Ps.update({f"fixed {t}": fixed(t, len(Tte)) for t in TIERS})
        Ps["best fixed-arm mixture at cost <= B (train-fold LP, context-free)"] = arm_mixture(Ttr, len(Tte))[0]
        Ps["per-task oracle (max quality, cheapest among ties)"] = Tte.oracle()
        for k, PP in Ps.items():
            q, c = Tte.value(PP)
            acc.setdefault(k, []).append((q, c, lb))
        ins.append(Ttr.value(Ttr.probs(pol)))
    lam_bar = float(np.mean(lams))
    rows = {k: {"q": float(np.mean([x[0] for x in v])), "c": float(np.mean([x[1] for x in v])),
                "J": float(np.mean([x[0] - x[2] * x[1] for x in v])),
                "J_sd": float(np.std([x[0] - x[2] * x[1] for x in v]))} for k, v in acc.items()}
    res["constrained"] = {"rows": rows, "lambda_bar_per_fold": lams, "lambda_bar_mean": lam_bar,
                          "in_sample_q_c": [list(x) for x in ins]}
    lines += [f"#### Constrained: budget B = ${BUDGET:.2f} per task, λ learned by dual ascent", "",
              f"J uses each fold's own learned λ̄ (mean over folds {lam_bar:.2f} per dollar; per fold "
              + ", ".join(f"{x:.2f}" for x in lams) + ")."] + [""] + fmt_rows(rows, "fold λ̄")
    lines += ["", f"Constrained policy in-sample on its ~{int(round(0.8*len(T)))} training tasks of each fold (mean over folds): quality "
              f"{np.mean([x[0] for x in ins]):.3f}, cost ${np.mean([x[1] for x in ins]):.4f}.", ""]
    return res, lines, fold0, lams[0]


def exact_row(T, pol):
    return T.value(T.probs(pol))


def gate_section(eps, T, n_redraws: int, boot: int):
    lines = []
    pol, lb = cgrpo_policy(T, seed=0)
    q_ex, c_ex = exact_row(T, pol)
    qb, cb = T.value(uniform(len(T)))
    out = {"lambda_bar": lb, "exact": {"quality": q_ex, "cost": c_ex}, "behavior_exact": {"quality": qb, "cost": cb}}
    # (a) the realized log of the uniform behavior (one arm per task, seed 0)
    log1 = fixed_arm_log(eps, seed=0)
    g1 = gate(log1, pol, "behavior", BUDGET, eps_q=EPS_Q, tol=TOL, boot=boot, seed=0)
    out["realized_log"] = g1
    lines += [f"Candidate: the constrained cgrpo-lookup policy trained on all {len(T)} tasks (λ̄ = {lb:.2f}/USD). "
              f"Exact value on the {len(T)} tasks (all arms logged): **quality {q_ex:.3f}, cost ${c_ex:.4f}**. "
              f"Uniform behavior exact: quality {qb:.3f}, cost ${cb:.4f}. "
              f"Gate: eps_q = {EPS_Q}, budget ${BUDGET:.2f} + tol ${TOL:.3f}.", "",
              format_table(g1, f"(a) Realized behavior log: one uniformly drawn arm per task ({len(T)} episodes, seed 0)",
                           cost_digits=4), "",
              f"Estimator error vs exact on this draw: quality IPS {g1['candidate']['quality']['ips'][0] - q_ex:+.3f}, "
              f"SNIPS {g1['candidate']['quality']['snips'][0] - q_ex:+.3f}, DR {g1['candidate']['quality']['dr'][0] - q_ex:+.3f}; "
              f"cost IPS {g1['candidate']['cost']['ips'][0] - c_ex:+.4f}, SNIPS {g1['candidate']['cost']['snips'][0] - c_ex:+.4f}, "
              f"DR {g1['candidate']['cost']['dr'][0] - c_ex:+.4f}.", ""]
    # (b) the full counterfactual log: every arm of every task (150 episodes)
    g2 = gate(eps, pol, "behavior", BUDGET, eps_q=EPS_Q, tol=TOL, boot=boot, seed=0)
    out["full_log"] = g2
    lines += [format_table(g2, f"(b) Full counterfactual log: all three arms per task ({len(eps)} episodes)", cost_digits=4), "",
              "With every arm logged at propensity 1/3, IPS, SNIPS and DR all reduce to the exact value "
              "(the importance weights over a task's three episodes sum to 3·Σπ = 3); the CI is then pure "
              f"task-sampling uncertainty from {len(T)} tasks.", ""]
    # (c) estimator error over redraws of the realized log
    errs = {m: {e: [] for e in ("ips", "snips", "dr")} for m in ("quality", "cost")}
    cover = {m: {e: 0 for e in ("ips", "snips", "dr")} for m in ("quality", "cost")}
    width = {m: {e: [] for e in ("ips", "snips", "dr")} for m in ("quality", "cost")}
    promote = 0
    exact = {"quality": q_ex, "cost": c_ex}
    for k in range(n_redraws):
        lg = fixed_arm_log(eps, seed=1000 + k)
        est = estimate_bandit(lg, pol, boot=boot, seed=k)
        for m in ("quality", "cost"):
            for e in ("ips", "snips", "dr"):
                pt, lo, hi = est[m][e]
                errs[m][e].append(pt - exact[m])
                cover[m][e] += int(lo <= exact[m] <= hi)
                width[m][e].append(hi - lo)
        qprev = float(np.mean([x.quality for x in lg]))
        promote += int(est["quality"]["dr"][1] >= qprev - EPS_Q and est["cost"]["dr"][2] <= BUDGET + TOL)
    tab = ["| metric | estimator | bias | RMSE | mean 90% CI width | CI covers exact |", "|---|---|---|---|---|---|"]
    stats = {}
    for m in ("quality", "cost"):
        for e in ("ips", "snips", "dr"):
            a = np.array(errs[m][e])
            stats[f"{m}/{e}"] = {"bias": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())),
                                 "ci_width": float(np.mean(width[m][e])), "coverage": cover[m][e] / n_redraws}
            dg = 3 if m == "quality" else 4
            s = stats[f"{m}/{e}"]
            tab.append(f"| {m} | {e.upper()} | {s['bias']:+.{dg}f} | {s['rmse']:.{dg}f} | {s['ci_width']:.{dg}f} | "
                       f"{s['coverage']:.0%} |")
    out["redraws"] = {"n": n_redraws, "stats": stats, "promote_rate": promote / n_redraws}
    lines += [f"(c) The same gate over {n_redraws} independent redraws of the realized behavior log "
              f"(one uniform arm per task each time; bootstrap {boot} per redraw):", ""] + tab
    lines += ["", f"PROMOTE in {promote}/{n_redraws} redraws ({promote / n_redraws:.0%}).", ""]
    return out, lines, pol, lb


def drift_section(eps, T, folds, frozen, lam_f, gate_out, spend: str, boot: int):
    lines = []
    out = {}
    q_prev = gate_out["realized_log"]["candidate"]["quality"]["dr"][0]
    lines += [f"Frozen policy: the all-50 constrained policy from the gate section (λ̄ = {lam_f:.2f}/USD; "
              f"J = quality − {lam_f:.2f}·cost). Costs after each event are recomputed from the logged token counts "
              "(spend ledger joined on task_id + purpose, matched on the logged usd), not by scaling. Quality is "
              "unchanged by a repricing. The retrained policy is cgrpo-lookup on the repriced log, warm-started "
              "from the frozen one, same budget. Its gate runs against the frozen incumbent on the repriced realized "
              f"log (seed 0), with the quality bar recorded at the frozen policy's promotion ({q_prev:.3f}).", "",
              f"In-sample (all {len(T)} tasks, exact lookups):", "",
              "| event | (i) frozen, old prices: q / cost / J | (ii) frozen, new prices: q / cost / J | "
              "(iii) retrained, new prices: q / cost / J | priced-BWT J(ii)−J(i) | recovery J(iii)−J(ii) | "
              "gate on retrained (exact q / cost of candidate) |",
              "|---|---|---|---|---|---|---|"]
    cv_lines = [f"Held-out (5-fold CV; frozen and retrained both trained on the ~{int(round(0.8*len(T)))} training tasks of each fold, "
                f"evaluated on its ~{int(round(0.2*len(T)))} held-out tasks; J at each fold's frozen λ̄; means over folds):", "",
                "| event | (i) frozen old: q / cost / J | (ii) frozen new: q / cost / J | (iii) retrained new: q / cost / J "
                "| priced-BWT | recovery | (iii) within budget |", "|---|---|---|---|---|---|---|"]
    qf, cf = T.value(T.probs(frozen))
    for tier, f in EVENTS:
        ev = reprice_episodes(eps, spend, {tier: f})
        Tn = LookupTable.from_episodes(ev)
        q2, c2 = Tn.value(Tn.probs(frozen))
        new, lb_new = train_cgrpo_lookup(Tn, BUDGET, seed=0, init=frozen, serve_eps=0.0)
        q3, c3 = Tn.value(Tn.probs(new))
        J1, J2, J3 = qf - lam_f * cf, q2 - lam_f * c2, q3 - lam_f * c3
        g = gate(fixed_arm_log(ev, seed=0), new, frozen, BUDGET, eps_q=EPS_Q, tol=TOL, boot=boot, seed=0,
                 q_prev=q_prev)
        out[f"{tier}x{f:g}"] = {"frozen_old": [qf, cf, J1], "frozen_new": [q2, c2, J2], "retrained_new": [q3, c3, J3],
                               "retrained_lambda_bar": lb_new, "priced_bwt": J2 - J1, "recovery": J3 - J2, "gate": g,
                               "uniform_new_cost": Tn.value(uniform(len(Tn)))[1]}
        lines.append(f"| {tier} ×{f:g} | {qf:.3f} / ${cf:.4f} / {J1:.3f} | {q2:.3f} / ${c2:.4f} / {J2:.3f} "
                     f"({ok(c2)} budget) | {q3:.3f} / ${c3:.4f} / {J3:.3f} ({ok(c3)} budget) | {J2 - J1:+.3f} | "
                     f"{J3 - J2:+.3f} | {g['decision']} (DR q {g['candidate']['quality']['dr'][0]:.3f} "
                     f"[{g['candidate']['quality']['dr'][1]:.3f}, {g['candidate']['quality']['dr'][2]:.3f}], DR cost "
                     f"${g['candidate']['cost']['dr'][0]:.4f} [{g['candidate']['cost']['dr'][1]:.4f}, "
                     f"{g['candidate']['cost']['dr'][2]:.4f}]; exact {q3:.3f} / ${c3:.4f}) |")
        # held-out version
        rows = []
        for fi, (tr, te) in enumerate(folds):
            fz, lf = train_cgrpo_lookup(T.subset(tr), BUDGET, seed=fi, serve_eps=0.0)
            rt, _ = train_cgrpo_lookup(Tn.subset(tr), BUDGET, seed=fi, init=fz, serve_eps=0.0)
            To, Tq = T.subset(te), Tn.subset(te)
            a = To.value(To.probs(fz)); b = Tq.value(Tq.probs(fz)); c = Tq.value(Tq.probs(rt))
            rows.append([a[0], a[1], a[0] - lf * a[1], b[0], b[1], b[0] - lf * b[1], c[0], c[1], c[0] - lf * c[1]])
        m = np.mean(rows, axis=0)
        out[f"{tier}x{f:g}"]["cv"] = m.tolist()
        cv_lines.append(f"| {tier} ×{f:g} | {m[0]:.3f} / ${m[1]:.4f} / {m[2]:.3f} | {m[3]:.3f} / ${m[4]:.4f} / {m[5]:.3f} "
                        f"| {m[6]:.3f} / ${m[7]:.4f} / {m[8]:.3f} | {m[5] - m[2]:+.3f} | {m[8] - m[5]:+.3f} | {ok(m[7])} |")
    lines += [""] + cv_lines + ["", "Harness-version drift (v1 → v2) cannot be measured on this log: it holds a "
                                "single harness version (crmarena-pro-v1) and nothing here re-runs the harness.", ""]
    for tier, f in EVENTS:
        g = out[f"{tier}x{f:g}"]["gate"]
        lines += [format_table(g, f"Gate after {tier} ×{f:g}: retrained vs frozen, repriced realized log", cost_digits=4), ""]
    return out, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default=str(ROOT / "traces" / "crmarena_b2b_counterfactual.jsonl"),
                    help="counterfactual CRMArena log (every task on every tier)")
    ap.add_argument("--spend", default=str(ROOT / "traces" / "crmarena_b2b_counterfactual.spend.jsonl"),
                    help="spend-ledger snapshot of the collection, with per-call token counts (repricing)")
    ap.add_argument("--out", default=str(ROOT / "results" / "crmarena"), help="output directory")
    ap.add_argument("--redraws", type=int, default=200, help="redraws of the realized one-arm-per-task log")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples per gate")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "policies").mkdir(parents=True, exist_ok=True)
    eps = read_jsonl(a.log)
    T = LookupTable.from_episodes(eps)
    folds = kfold_tasks(T, 5, seed=0)
    assert all(not (set(tr) & set(te)) and len(set(tr) | set(te)) == len(T) for tr, te in folds)
    res = {"n_tasks": len(T), "n_episodes": len(eps), "folds": [{"train": tr, "test": te} for tr, te in folds],
           "budget": BUDGET, "tol": TOL, "eps_q": EPS_Q}
    org = "b2c" if "b2c" in str(a.log) else "b2b"
    lines = [f"## Part A: linear controller on the CRMArena-Pro {org} log ({len(T)} tasks × 3 tiers)", "",
             "### 1–2. Cross-validated first-step tier policies", "",
             f"5-fold CV over task ids (category-stratified, seed 0): train on ~{int(round(0.8*len(T)))} tasks, evaluate on the held-out ~{int(round(0.2*len(T)))}. "
             "All three arms are logged for every task, so held-out quality and cost of any policy are exact lookups. "
             "Means over the 5 folds; [fold SD] is the across-fold standard deviation of J.", ""]
    cv, cv_lines, fold0_pol, fold0_lam = cv_tables(eps, T, folds)
    res["cv"] = cv
    lines += cv_lines
    fold0_pol.meta.update({"fold": 0, "train_ids": folds[0][0], "test_ids": folds[0][1]})
    fold0_pol.save(out / "policies" / "fold0_constrained.json")
    P0 = T.probs(fold0_pol)
    (out / "fold0_for_llm.json").write_text(json.dumps({
        "train": folds[0][0], "test": folds[0][1], "lambda_bar": float(fold0_pol.meta["lambda"]),
        "linear_probs": {t: [round(float(x), 6) for x in P0[i]] for i, t in enumerate(T.task_ids)}}, indent=1))
    lines += ["### 3. DR gate on the real log", ""]
    g, g_lines, pol50, lam50 = gate_section(eps, T, a.redraws, a.boot)
    pol50.save(out / "policies" / "all50_constrained.json")
    res["gate"] = g
    lines += g_lines
    lines += ["### 4. Repricing drift (no new calls)", ""]
    dr, d_lines = drift_section(eps, T, folds, pol50, lam50, g, a.spend, a.boot)
    res["drift"] = dr
    lines += d_lines
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
