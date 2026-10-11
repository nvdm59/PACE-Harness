"""Exact policy evaluation on a logged refinement tree -> results/ssr/RESULTS.md + frontier.csv.

Sections: data; calibration of the confidence signal (does low confidence predict a wrong
initial answer? AUROC); refinement harm (correct -> wrong after refining) by tier, round and
confidence bucket; the quality-cost table and Pareto frontier of every policy; the priced
comparison J = quality - lambda * cost at lambda in {2, 5} per USD; the best policy under a
per-task budget; and an interpretation written from the numbers, with the hypotheses H1-H3 of
the package README checked explicitly.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from ...prices import TIERS
from ...trace import read_jsonl
from .policy import (Always, ControllerPolicy, Never, UncertaintyGated, auroc, fit_controller, pareto)
from .tree import Tree, build_trees, evaluate_policy_on, kfold, max_rounds_in, read_records, tiers_in

LAMBDAS = (2.0, 5.0)
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BUDGETS = (0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50)
CONF_BINS = ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
WRONG_BELOW = 0.5          # quality < 0.5 counts as a wrong answer (F1 tasks included)


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _usd(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:.4f}"


def _pct(num, den):
    return "n/a" if not den else f"{100.0 * num / den:.0f}% ({num}/{den})"


def _mean(xs):
    xs = list(xs)
    return float("nan") if not xs else sum(xs) / len(xs)


# ------------------------------------------------------------- oracles
def leaf_outcomes(tree: Tree) -> list[tuple[list[str], float, float]]:
    """(path, quality, cost) of every terminal node reachable by a clairvoyant policy: probes are
    charged only where a refinement was actually taken (the oracle does not need to look)."""
    out = []

    def rec(node, cost):
        out.append((node.path, node.quality, cost))
        p = (node.probe or {}).get("usd", 0.0)
        for t, (e, c) in node.children.items():
            rec(c, cost + p + float(e["refine"]["usd"]))

    rec(tree.root, tree.initial_cost)
    return out


class Oracle:
    needs_probe = False

    def __init__(self, lam: float | None):
        self.lam = lam
        self.name = "oracle (max quality, cheapest)" if lam is None else f"oracle (argmax q - {lam:g}·cost)"


def eval_oracle(trees: list[Tree], lam: float | None) -> dict:
    q = c = 0.0
    for t in trees:
        leaves = leaf_outcomes(t)
        key = (lambda l: (l[1], -l[2])) if lam is None else (lambda l: (l[1] - lam * l[2], -l[2]))
        best = max(leaves, key=key)
        q += best[1]
        c += best[2]
    n = len(trees)
    return {"n": n, "n_total": n, "quality": q / n, "cost": c / n, "eval_usd": 0.0, "rounds": float("nan")}


# ------------------------------------------------------ tier arms (log)
def load_tier_arms(path: str | Path | None, task_ids: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """arm -> task_id -> (quality, cost) from the counterfactual collection: what the same tasks
    score when a better tier is bought up front instead of refining the cheap answer."""
    out: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    if not path or not Path(path).exists():
        return {}
    want = set(task_ids)
    for e in read_jsonl(path):
        arm = (e.meta or {}).get("arm") or e.steps[0].action_name.split("@")[1]
        if e.task_id in want:
            out[arm][e.task_id] = (float(e.quality), float(e.total_cost))
    return {k: v for k, v in out.items() if len(v) == len(want)}


def arm_row(arm: str, table: dict[str, tuple[float, float]]) -> dict:
    qs = [q for q, _ in table.values()]
    n = len(qs)
    mean = sum(qs) / n
    se = (sum((q - mean) ** 2 for q in qs) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan")
    return {"n": n, "n_total": n, "quality": mean, "cost": sum(c for _, c in table.values()) / n, "eval_usd": 0.0,
            "rounds": 0.0, "q_se": se, "per_task": dict(table)}


# --------------------------------------------------------------- tables
def policy_rows(trees: list[Tree], tiers: list[str], max_rounds: int, budget: float, folds: int, seed: int,
                policy_out: str | None, arms: dict | None = None) -> tuple[list[dict], dict]:
    rows = []

    def add(name, family, res, split="all", tier="", tau=""):
        rows.append({"policy": name, "family": family, "tier": tier, "tau": tau, "split": split, **res})

    add("never", "never", evaluate_policy_on(trees, Never(), tiers, max_rounds))
    for t in tiers:
        for n in range(1, max_rounds + 1):
            p = Always(n, t)
            add(p.name, "always", evaluate_policy_on(trees, p, tiers, max_rounds), tier=t)
    for t in tiers:
        for tau in TAUS:
            p = UncertaintyGated(tau, t, max_rounds)
            add(p.name, "uncertainty_gated", evaluate_policy_on(trees, p, tiers, max_rounds), tier=t, tau=tau)
    ctrl_meta = {}
    specs = [("B", budget, None)] + [("lam", None, lam) for lam in LAMBDAS]
    for kind, B, lam in specs:
        label = f"controller (B=${B:.2f}, learned λ)" if kind == "B" else f"controller (λ={lam:g}/USD)"
        try:
            ctrl = fit_controller(trees, tiers, max_rounds, budget=B, lam=(2.0 if lam is None else lam), seed=seed)
        except ValueError as e:
            ctrl_meta[label] = {"error": str(e)}
            continue
        if kind == "B" and policy_out:
            ctrl.save(policy_out)
        ctrl_meta[label] = dict(ctrl.meta)
        pol = ControllerPolicy(ctrl, label)
        add(f"{label}, in-sample", "controller", evaluate_policy_on(trees, pol, tiers, max_rounds), split="in-sample")
        # held-out: K-fold over trees
        if len(trees) >= folds and folds > 1:
            agg = defaultdict(float)
            n_tot = 0
            fold_j = []
            per_task = {}
            for tr, te in kfold(trees, folds, seed):
                c = fit_controller(tr, tiers, max_rounds, budget=B, lam=(2.0 if lam is None else lam), seed=seed)
                r = evaluate_policy_on(te, ControllerPolicy(c, label), tiers, max_rounds)
                if r["n"] == 0:
                    continue
                for k in ("quality", "cost", "eval_usd", "rounds"):
                    agg[k] += r[k] * r["n"]
                n_tot += r["n"]
                per_task.update(r["per_task"])
                fold_j.append(r["quality"] - (lam or c.meta.get("lambda_bar", 0.0)) * r["cost"])
            if n_tot:
                res = {k: agg[k] / n_tot for k in ("quality", "cost", "eval_usd", "rounds")}
                qs = [q for q, _ in per_task.values()]
                mq = sum(qs) / len(qs)
                res.update({"n": n_tot, "n_total": len(trees), "per_task": per_task,
                            "q_se": (sum((q - mq) ** 2 for q in qs) / max(len(qs) - 1, 1)) ** 0.5 / len(qs) ** 0.5})
                add(f"{label}, {folds}-fold held-out", "controller", res, split="held-out")
                ctrl_meta[label]["heldout_fold_sd_J"] = (sum((j - _mean(fold_j)) ** 2 for j in fold_j) / max(len(fold_j) - 1, 1)) ** 0.5
    for arm in TIERS:
        if arms and arm in arms:
            add(f"buy {arm} up front (single agent, counterfactual log)", "tier_arm", arm_row(arm, arms[arm]), tier=arm)
    add(Oracle(None).name, "oracle", eval_oracle(trees, None))
    for lam in LAMBDAS:
        add(Oracle(lam).name, "oracle", eval_oracle(trees, lam))
    for r in rows:
        for lam in LAMBDAS:
            r[f"J{lam:g}"] = r["quality"] - lam * r["cost"]
    comparable = [(r["policy"], r["quality"], r["cost"]) for r in rows if r["family"] != "oracle" and r["n"] > 0
                  and r["split"] != "in-sample"]          # tier arms from the log compete on the frontier too
    front = pareto(comparable)
    for r in rows:
        r["pareto"] = r["policy"] in front
    return rows, ctrl_meta


def calibration(trees: list[Tree], tiers: list[str]) -> dict:
    roots = [t.root for t in trees if t.root.probed]
    wrong = [int(n.quality < WRONG_BELOW) for n in roots]
    out = {"n": len(roots), "base_wrong": _mean(wrong)}
    for key, label in (("min_conf", "1 − min claim self-confidence"), ("mean_conf", "1 − mean claim self-confidence"),
                       ("answer_conf", "1 − whole-answer self-confidence")):
        out[label] = auroc([1.0 - float(n.probe[key]) for n in roots], wrong)
    for tier in tiers:
        sc, lb = [], []
        for t in trees:
            if tier in t.root.children:
                e = t.root.children[tier][0]["refine"]
                sc.append(1.0 - float(e["conf_min"]))
                lb.append(int(t.root.quality < WRONG_BELOW))
        out[f"1 − min per-claim conf after re-solving on {tier} (round 1)"] = auroc(sc, lb)
        out[f"n_{tier}"] = len(sc)
    # bucket table on the root probe
    buckets = []
    for lo, hi in CONF_BINS:
        ts = [t for t in trees if t.root.probed and lo <= float(t.root.probe["min_conf"]) < hi]
        row = {"bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}{')' if hi <= 1.0 else ']'}", "n": len(ts),
               "acc0": _mean(t.root.quality for t in ts)}
        for tier in tiers:
            es = [t.root.children[tier][0]["refine"] for t in ts if tier in t.root.children]
            row[f"dq_{tier}"] = _mean(e["quality_after"] - e["quality_before"] for e in es)
            row[f"fix_{tier}"] = (sum(e["quality_after"] >= WRONG_BELOW for e in es if e["quality_before"] < WRONG_BELOW),
                                  sum(e["quality_before"] < WRONG_BELOW for e in es))
            row[f"harm_{tier}"] = (sum(e["quality_after"] < WRONG_BELOW for e in es if e["quality_before"] >= WRONG_BELOW),
                                   sum(e["quality_before"] >= WRONG_BELOW for e in es))
        buckets.append(row)
    out["buckets"] = buckets
    return out


def harm_table(trees: list[Tree]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in trees:
        for parent, e, child in t.edges():
            rf = e["refine"]
            conf = "confidentiality" if t.category in ("private_customer_information", "internal_operation_data",
                                                        "confidential_company_knowledge") else "business"
            groups[(rf["tier"], e["round"], "all")].append(rf)
            groups[(rf["tier"], e["round"], conf)].append(rf)
    rows = []
    for (tier, rnd, grp) in sorted(groups):
        es = groups[(tier, rnd, grp)]
        n_wrong = sum(e["quality_before"] < WRONG_BELOW for e in es)
        n_right = len(es) - n_wrong
        rows.append({"tier": tier, "round": rnd, "group": grp, "n": len(es),
                     "changed": sum(bool(e["changed"]) for e in es),
                     "fix": sum(e["quality_after"] >= WRONG_BELOW and e["quality_before"] < WRONG_BELOW for e in es),
                     "n_wrong": n_wrong,
                     "harm": sum(e["quality_after"] < WRONG_BELOW and e["quality_before"] >= WRONG_BELOW for e in es),
                     "n_right": n_right,
                     "dq": _mean(e["quality_after"] - e["quality_before"] for e in es),
                     "usd": _mean(e["usd"] for e in es),
                     "turns": _mean(e["worker_turns"] for e in es)})
    return rows


# ------------------------------------------------------------- report
def write_report(path: str | Path, log: str, trees_all, trees, tiers, max_rounds, rows, ctrl_meta, cal, harm,
                 budget, backend, folds, prelude: str | None = None, arms: dict | None = None):
    fake = backend == "fake"
    L = []
    L.append("# Priced Socratic self-refinement on CRMArena-Pro: results\n")
    if prelude:
        L.append("## 0. Interpretation\n")
        L.append(prelude.strip() + "\n")
        L.append("*The sections below are generated by `analyze`; this one is authored and kept across "
                 "regenerations (`--prelude`).*\n")
    if fake:
        L.append("> **DRY RUN.** Every number below comes from the deterministic fake world in "
                 "`pace/experiments/ssr/fake.py` (no API, no Salesforce, no spend). The structure of the tables is "
                 "real; the values are not evidence about SSR on CRMArena. Re-run `analyze` on a real tree log "
                 "to replace this file.\n")
    init_arms = sorted({t.initial_arm for t in trees_all})
    n_edges = sum(1 for t in trees for _ in t.edges())
    spend = sum(t.initial_cost for t in trees)
    spend_ssr = sum(float(e["refine"]["usd"]) for t in trees for _, e, _ in t.edges()) \
        + sum(float((n.probe or {}).get("usd", 0.0)) for t in trees for n in t.nodes())
    spend_eval = sum(float(e["refine"].get("usd_eval", 0.0)) for t in trees for _, e, _ in t.edges())
    L.append(f"**Data.** `{log}`: {len(trees_all)} (task, initial arm) trees over initial arm(s) {init_arms}, "
             f"{len(trees)} complete to depth {max_rounds} over refinement tiers {tiers} ({n_edges} refinement edges). "
             f"Initial episodes {_usd(spend)} total ({'fake' if fake else 'from the counterfactual log, already paid'}); "
             f"SSR calls {_usd(spend_ssr)} ({_usd(spend_ssr / max(len(trees), 1))} per tree); scorer calls "
             f"{_usd(spend_eval)} (billed, not charged). A wrong answer is quality < {WRONG_BELOW}.\n")
    never = next(r for r in rows if r["policy"] == "never")
    L.append(f"**Decision.** At each node (round 0 = the initial answer, then after each refinement) a policy "
             f"chooses STOP or REFINE@tier from the category, the round, the cost so far and the probe (the "
             f"decomposition's self-reported claim confidences). With every branch logged, each policy's "
             f"quality and cost are exact per-task lookups; costs include the initial episode "
             f"({_usd(never['cost'])} per task for `never`).\n")

    # ---- calibration
    L.append("## 1. Is the confidence signal informative? (calibration)\n")
    L.append(f"{cal['n']} root probes; base rate of a wrong initial answer {_f(cal['base_wrong'])}. AUROC of each "
             f"score for predicting *wrong initial answer* (0.5 = uninformative):\n")
    L.append("| signal | AUROC | n |\n|---|---|---|")
    for k, v in cal.items():
        if k.startswith("1 −"):
            n = cal.get(f"n_{k.split(' on ')[-1].split(' ')[0]}", cal["n"]) if "re-solving" in k else cal["n"]
            L.append(f"| {k} | {_f(v)} | {n} |")
    L.append("")
    L.append("Round-1 outcome by the root probe's minimum claim confidence (the gate's signal):\n")
    hdr = "| min self-conf bin | n | initial acc |" + "".join(f" Δq after 1 round @{t} | fix @{t} | harm @{t} |" for t in tiers)
    L.append(hdr)
    L.append("|---|---|---|" + "---|---|---|" * len(tiers))
    for b in cal["buckets"]:
        cells = "".join(f" {_f(b[f'dq_{t}'], 2)} | {_pct(*b[f'fix_{t}'])} | {_pct(*b[f'harm_{t}'])} |" for t in tiers)
        L.append(f"| {b['bin']} | {b['n']} | {_f(b['acc0'], 2)} |{cells}")
    L.append("\n`fix` = wrong → correct after the round, out of initially wrong; `harm` = correct → wrong, out of "
             "initially correct.\n")

    # ---- harm
    L.append("## 2. Refinement harm\n")
    L.append("| tier | round | tasks | n edges | changed | fix (wrong→right) | harm (right→wrong) | mean Δq | mean round cost | worker turns |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for h in harm:
        L.append(f"| {h['tier']} | {h['round']} | {h['group']} | {h['n']} | {_pct(h['changed'], h['n'])} | "
                 f"{_pct(h['fix'], h['n_wrong'])} | {_pct(h['harm'], h['n_right'])} | {_f(h['dq'], 3)} | "
                 f"{_usd(h['usd'])} | {_f(h['turns'], 1)} |")
    L.append("")

    # ---- policies
    L.append("## 3. Quality vs cost per policy (exact lookups on the logged tree)\n")
    L.append(f"J = quality − λ·cost (USD per task). Pareto = not dominated among the non-oracle, non-in-sample rows. "
             f"Controllers: softmax-linear PACE policies over the {len(ctrl_meta) and 'SSR'} features, trained by exact "
             f"policy gradient on the tree; the B-constrained one learns λ by dual ascent and serves the iterate "
             f"average; held-out rows are {folds}-fold CV over tasks (category-stratified).\n")
    L.append("| policy | quality ± SE | cost/task | Δq vs never | Δcost vs never | rounds | J (λ=2) | J (λ=5) | Pareto | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        q_se = "" if math.isnan(r.get("q_se", float("nan"))) else f" ± {r['q_se']:.3f}"
        L.append(f"| {r['policy']} | {_f(r['quality'])}{q_se} | {_usd(r['cost'])} | {r['quality'] - never['quality']:+.3f} | "
                 f"{r['cost'] - never['cost']:+.4f} | {_f(r['rounds'], 2)} | {_f(r['J2'])} | {_f(r['J5'])} | "
                 f"{'yes' if r['pareto'] else ''} | {r['n']} |")
    L.append("")
    L.append("SE = standard error of the per-task quality mean (task sampling only; paired differences are in §5).\n")
    for label, m in ctrl_meta.items():
        if "error" in m:
            L.append(f"- {label}: not trained ({m['error']})")
        else:
            L.append(f"- {label}: λ̄ = {m.get('lambda_bar', float('nan')):.2f}/USD (final {m.get('lambda_final', float('nan')):.2f}), "
                     f"training quality {m.get('train_quality', float('nan')):.3f} at {_usd(m.get('train_cost'))}"
                     + (f"; held-out fold SD of J {m['heldout_fold_sd_J']:.3f}" if "heldout_fold_sd_J" in m else ""))
    L.append("")

    # ---- priced + budget
    L.append("## 4. Priced comparison and the budget-constrained best policy\n")
    cand = [r for r in rows if r["family"] != "oracle" and r["split"] != "in-sample" and r["n"] > 0]
    for lam in LAMBDAS:
        best = max(cand, key=lambda r: r[f"J{lam:g}"])
        orc = next(r for r in rows if r["policy"] == Oracle(lam).name)
        L.append(f"- **λ = {lam:g}/USD**: best policy `{best['policy']}` with J = {best[f'J{lam:g}']:.3f} "
                 f"(quality {best['quality']:.3f} at {_usd(best['cost'])}); `never` has J = {never[f'J{lam:g}']:.3f}; "
                 f"the per-task oracle has J = {orc[f'J{lam:g}']:.3f}.")
    L.append("\n| budget B (USD/task) | best feasible policy (max quality, cost ≤ B) | quality | cost | Δq vs never |")
    L.append("|---|---|---|---|---|")
    for B in BUDGETS:
        feas = [r for r in cand if r["cost"] <= B + 1e-9]
        if not feas:
            L.append(f"| ${B:.2f} | none feasible | | | |")
            continue
        b = max(feas, key=lambda r: (r["quality"], -r["cost"]))
        L.append(f"| ${B:.2f} | {b['policy']} | {_f(b['quality'])} | {_usd(b['cost'])} | {b['quality'] - never['quality']:+.3f} |")
    L.append("")

    # ---- refine vs buy a better tier
    if arms:
        L.append("## 5. Refine the cheap answer, or buy a better tier up front?\n")
        L.append("Paired on the same tasks: each refinement policy (initial answer = the cheap arm) against the "
                 "single-agent mid and frontier arms of the counterfactual log. Δq ± SE is the per-task paired "
                 "difference (policy − arm); Δcost is per task.\n")
        L.append("| refinement policy | quality | cost | vs buy mid: Δq ± SE | Δcost | vs buy frontier: Δq ± SE | Δcost |")
        L.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if r["family"] in ("oracle", "tier_arm") or r["split"] == "in-sample" or not r.get("per_task"):
                continue
            cells = []
            for arm in ("mid", "frontier"):
                if arm not in arms:
                    cells.append("n/a | n/a")
                    continue
                d = [r["per_task"][t][0] - arms[arm][t][0] for t in r["per_task"] if t in arms[arm]]
                dc = [r["per_task"][t][1] - arms[arm][t][1] for t in r["per_task"] if t in arms[arm]]
                m = sum(d) / len(d)
                se = (sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5 / len(d) ** 0.5
                cells.append(f"{m:+.3f} ± {se:.3f} | {sum(dc) / len(dc):+.4f}")
            L.append(f"| {r['policy']} | {_f(r['quality'])} | {_usd(r['cost'])} | {' | '.join(cells)} |")
        L.append("")

    # ---- interpretation
    L.append("## 6. Automatic checks\n" if arms else "## 5. Automatic checks\n")
    bul = []
    if fake:
        bul.append("**This is the dry run.** The fake world was built so that low confidence predicts wrong answers "
                   "imperfectly, mid workers are better than cheap ones, and refinement harms correct refusals "
                   "on the confidentiality categories. The bullets below show what the analysis *reports* given such "
                   "a world; they are not findings.")
    # H1
    low = [b for b in cal["buckets"] if b["bin"].startswith("[0.0") or b["bin"].startswith("[0.4")]
    high = [b for b in cal["buckets"] if b["bin"].startswith("[0.6") or b["bin"].startswith("[0.8")]
    for t in tiers:
        dq_low = [e["refine"]["quality_after"] - e["refine"]["quality_before"] for tr in trees
                  if tr.root.probed and float(tr.root.probe["min_conf"]) < 0.6 and t in tr.root.children
                  for e in [tr.root.children[t][0]]]
        dq_high = [e["refine"]["quality_after"] - e["refine"]["quality_before"] for tr in trees
                   if tr.root.probed and float(tr.root.probe["min_conf"]) >= 0.6 and t in tr.root.children
                   for e in [tr.root.children[t][0]]]

        def mse(xs):
            if not xs:
                return float("nan"), float("nan")
            m = sum(xs) / len(xs)
            return m, ((sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5 / len(xs) ** 0.5 if len(xs) > 1 else float("nan"))
        (dl, sl), (dh, sh) = mse(dq_low), mse(dq_high)
        gap = dl - dh
        gap_se = (sl ** 2 + sh ** 2) ** 0.5 if not (math.isnan(sl) or math.isnan(sh)) else float("nan")
        if math.isnan(gap) or math.isnan(gap_se):
            verdict = "too few tasks in one of the groups to compare."
        elif abs(gap) < 2 * gap_se:
            verdict = f"the gap ({gap:+.3f} ± {gap_se:.3f}) is within 2 SE: indistinguishable at this n."
        else:
            verdict = f"gap {gap:+.3f} ± {gap_se:.3f}; " + ("consistent with H1." if gap > 0 else "contrary to H1.")
        bul.append(f"**H1 (refine where confidence is low) on {t}:** one round changes quality by {dl:+.3f} ± {sl:.3f} when "
                   f"the probe's min confidence is < 0.6 (n = {len(dq_low)}) and by {dh:+.3f} ± {sh:.3f} when it is "
                   f"≥ 0.6 (n = {len(dq_high)}); {verdict}")
    # H2
    for t in tiers:
        a1 = next((r for r in rows if r["policy"] == f"always_1:{t}"), None)
        if a1:
            dom = [r for r in cand if r["quality"] >= a1["quality"] and r["cost"] <= a1["cost"] and r["policy"] != a1["policy"]
                   and (r["quality"] > a1["quality"] or r["cost"] < a1["cost"])]
            bul.append(f"**H2 (always-refine is dominated) on {t}:** `always_1:{t}` reaches {a1['quality']:.3f} at "
                       f"{_usd(a1['cost'])}; " + (f"dominated by {', '.join('`' + d['policy'] + '`' for d in dom[:3])}."
                                                 if dom else "not dominated by any evaluated policy."))
    # H3
    ho = [r for r in rows if r["family"] == "controller" and r["split"] == "held-out"]
    if ho:
        for r in ho:
            better_gates = [g for g in cand if g["family"] == "uncertainty_gated" and g["quality"] >= r["quality"] and g["cost"] <= r["cost"]]
            dn = [r["per_task"][t][0] - never["per_task"][t][0] for t in r["per_task"] if t in never.get("per_task", {})]
            mdn = sum(dn) / len(dn) if dn else float("nan")
            sdn = ((sum((x - mdn) ** 2 for x in dn) / max(len(dn) - 1, 1)) ** 0.5 / len(dn) ** 0.5) if len(dn) > 1 else float("nan")
            bul.append(f"**H3 (a priced gate reaches the frontier):** `{r['policy']}` held-out: {r['quality']:.3f} at {_usd(r['cost'])} "
                       f"(paired Δq vs never {mdn:+.3f} ± {sdn:.3f})"
                       f"{' — on the Pareto frontier' if r['pareto'] else ' — off the frontier'}; "
                       + (f"{len(better_gates)} fixed-τ gate(s) weakly dominate it ({', '.join('`' + g['policy'] + '`' for g in better_gates[:3])})."
                          if better_gates else "no fixed-τ gate weakly dominates it."))
    n_tasks = len(trees)
    bul.append(f"**Power.** {n_tasks} tasks: one task is {1.0 / max(n_tasks, 1):.3f} of quality, so differences of a few "
               f"hundredths are within one or two tasks. Held-out controller rows are the honest ones; the in-sample "
               f"rows show what the feature set can memorize (22 category one-hots on {n_tasks} tasks).")
    hm = [h for h in harm if h["group"] == "confidentiality" and h["n_right"]]
    if hm:
        worst = max(hm, key=lambda h: h["harm"] / h["n_right"])
        bul.append(f"**Harm concentrates where the correct answer is a refusal.** On the confidentiality categories, "
                   f"{worst['tier']} round {worst['round']} flips {_pct(worst['harm'], worst['n_right'])} of correct "
                   f"answers to wrong. A gate that only looks at confidence cannot see this; the controller can, "
                   f"through the category feature.")
    L.extend(f"- {b}" for b in bul)
    L.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["policy", "family", "tier", "tau", "split", "quality", "q_se", "cost", "eval_usd", "rounds", "J2", "J5", "pareto", "n"]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r.get(k):.6f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in cols})


def analyze(log: str, out_md: str, out_csv: str, budget: float = 0.10, folds: int = 5, seed: int = 0,
            policy_out: str | None = None, prelude: str | None = None,
            counterfactual: str | None = None) -> tuple[list[dict], dict]:
    """`prelude`: a markdown file inserted as §0 (authored interpretation that survives regeneration);
    `counterfactual`: the collection log whose single-agent tier arms become reference rows."""
    recs = read_records(log)
    if not recs:
        raise SystemExit(f"no records in {log}")
    backend = recs[0].get("backend", "crmarena")
    trees_all = build_trees(recs)
    tiers = tiers_in(trees_all) or ["cheap"]
    max_rounds = max_rounds_in(trees_all)
    trees = [t for t in trees_all if t.complete(tiers, max_rounds)]
    if not trees:
        trees = trees_all        # a single-policy log: evaluate what is evaluable
    arms = load_tier_arms(counterfactual, [t.task_id for t in trees]) if backend != "fake" else {}
    rows, ctrl_meta = policy_rows(trees, tiers, max_rounds, budget, folds, seed, policy_out, arms)
    cal = calibration(trees, tiers)
    harm = harm_table(trees)
    pre = Path(prelude).read_text() if prelude and Path(prelude).exists() else None
    write_report(out_md, log, trees_all, trees, tiers, max_rounds, rows, ctrl_meta, cal, harm, budget, backend, folds,
                 pre, arms)
    write_csv(out_csv, rows)
    return rows, ctrl_meta
