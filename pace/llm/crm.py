"""Phase 2: the LLM controller on the CRMArena-Pro counterfactual log (lookup mode).

Pure Python (no torch): SFT labels, prompt rows and the eval table. The GPU stages
(SFT, GRPO, decoding) run in pace/llm/train_grpo_modal.py::crm_phase2.

SFT labels: for a shown price lam (per USD) the label of a task is its category's
oracle tier on the TRAINING tasks: argmax_tier mean(quality - lam * cost) over the
training tasks of that category (ties -> the cheaper tier; a category absent from
training falls back to the all-training argmax). It conditions only on what the
prompt shows (category, org, lam), so it is learnable to 100% agreement.
"""
from __future__ import annotations

import json
from collections import Counter

from ..harness.sim import TIERS
from .env_text import format_action, parse_action, round_lam, with_price
from .rewards import LookupReward, lookup_rows

LAM_GRID_USD = [float(i) for i in range(21)]          # 0 .. 20 per USD, step 1
LAM_STEP_USD = 1.0


def category_oracle(lookup: LookupReward, train_ids: list[str], lam: float) -> dict[str, str]:
    """category -> oracle tier at price lam over the training tasks; key None = the global fallback."""
    lam = round_lam(lam, LAM_STEP_USD)

    def best(ids):
        means = {t: sum(lookup.table[(i, t)][0] - lam * lookup.table[(i, t)][1] for i in ids) / len(ids)
                 for t in TIERS}
        return max(TIERS, key=lambda t: (round(means[t], 12), -TIERS.index(t)))
    by: dict[str, list[str]] = {}
    for i in train_ids:
        by.setdefault(lookup.features[i]["category"], []).append(i)
    out = {c: best(ids) for c, ids in by.items()}
    out[None] = best(list(train_ids))
    return out


def oracle_tier(lookup: LookupReward, train_ids: list[str], category: str, lam: float) -> str:
    o = category_oracle(lookup, train_ids, lam)
    return o.get(category, o[None])


def price_rows(lookup: LookupReward, task_ids: list[str], budget: float, train_ids: list[str],
               shuffle_seed: int = 0) -> list[dict]:
    """Price-conditioned prompt rows (placeholder unfilled) for `task_ids`; the tier costs shown
    are the TRAINING tasks' means."""
    tc = {t: sum(lookup.table[(i, t)][1] for i in train_ids) / len(train_ids) for t in TIERS}
    return lookup_rows(lookup, budget, price=True, task_ids=task_ids, tier_cost=tc, shuffle_seed=shuffle_seed)


def sft_rows_crm(lookup: LookupReward, train_ids: list[str], budget: float,
                 lam_grid: list[float] | None = None) -> list[dict]:
    """Every (training task, grid price) pair -> its category-oracle label. Each pair gets its own
    seeded shuffle of the legal actions."""
    grid = list(lam_grid or LAM_GRID_USD)
    orc = {lam: category_oracle(lookup, train_ids, lam) for lam in grid}
    out = []
    for k, lam in enumerate(grid):
        for r in price_rows(lookup, train_ids, budget, train_ids, shuffle_seed=1000 + k):
            task = json.loads(r["task"])
            task["lam"] = lam
            tier = orc[lam].get(task["category"], orc[lam][None])
            out.append({"user": with_price(r["user"], lam, LAM_STEP_USD), "task": json.dumps(task, sort_keys=True),
                        "completion": format_action(tier, "CONTINUE"), "category": task["category"], "lam": lam})
    return out


# ------------------------------------------------------------------ eval
def score(lookup: LookupReward, tasks: list[dict], completions: list[list[str]], lam: float) -> dict:
    """completions[i] = one or more completions for task i (several = samples; each task's value
    is the mean over its samples). Exact lookups of the logged (quality, cost)."""
    qs, cs, inv, hist = [], [], 0, Counter()
    n_out = 0
    for t, comps in zip(tasks, completions):
        q = c = 0.0
        for comp in comps:
            o = lookup(t, comp, lam)
            q += o.quality; c += o.cost
            inv += int(not o.valid)
            hist[o.action.split()[0] if o.valid else "INVALID"] += 1
            n_out += 1
        qs.append(q / len(comps)); cs.append(c / len(comps))
    q, c = sum(qs) / len(qs), sum(cs) / len(cs)
    return {"quality": q, "cost": c, "J": q - lam * c, "invalid": inv / n_out,
            "mix": {k: v / n_out for k, v in hist.most_common()}}


def tier_rule_completions(tasks: list[dict], tier_of) -> list[list[str]]:
    return [[format_action(tier_of(t), "CONTINUE")] for t in tasks]


def table(rows: dict[str, dict], budget: float, tol: float) -> str:
    lines = ["| policy | quality | cost (USD) | J | within budget | invalid | first-action mix |",
             "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        mix = ", ".join(f"{k} {v:.0%}" for k, v in list(r["mix"].items())[:3])
        lines.append(f"| {name} | {r['quality']:.3f} | {r['cost']:.4f} | {r['J']:.3f} | "
                     f"{'yes' if r['cost'] <= budget + tol + 1e-12 else 'NO'} | {r['invalid']:.1%} | {mix} |")
    return "\n".join(lines)


def eval_tables(lookup: LookupReward, comps: dict, splits: dict[str, list[str]], train_ids: list[str],
                lam_bar: float, budget: float, tol: float, linear_probs: dict[str, list[float]],
                header: str) -> tuple[str, dict]:
    """comps: name -> {task_id: [completions]} for the LLM rows (already decoded at the shown
    price round(lam_bar)). Adds fixed arms, the fold's linear policy (exact expectation under its
    tier probabilities), the category-oracle rule (the SFT target) and the per-task oracle."""
    lam = round_lam(lam_bar, LAM_STEP_USD)
    out, md = {}, [header, ""]
    for split, ids in splits.items():
        tasks = [{"env": "lookup", **lookup.features[i]} for i in ids]
        rows = {}
        for name, by_task in comps.items():
            rows[name] = score(lookup, tasks, [by_task[i] for i in ids], lam)
        for t in TIERS:
            rows[f"fixed {t}"] = score(lookup, tasks, tier_rule_completions(tasks, lambda _t, t=t: t), lam)
        # linear policy: exact expectation over its first-step tier probabilities
        q = sum(sum(p * lookup.table[(i, t)][0] for p, t in zip(linear_probs[i], TIERS)) for i in ids) / len(ids)
        c = sum(sum(p * lookup.table[(i, t)][1] for p, t in zip(linear_probs[i], TIERS)) for i in ids) / len(ids)
        rows["linear constrained policy, Part A fold 0 (served mix)"] = {
            "quality": q, "cost": c, "J": q - lam * c, "invalid": 0.0,
            "mix": {t: sum(linear_probs[i][k] for i in ids) / len(ids) for k, t in enumerate(TIERS)}}
        rows[f"category-oracle rule at λ={lam:g} from the 40 training tasks (SFT target)"] = score(
            lookup, tasks, tier_rule_completions(tasks, lambda t: oracle_tier(lookup, train_ids, t["category"], lam)), lam)
        rows[f"per-task oracle at λ={lam:g} (argmax q − λc on each task)"] = score(
            lookup, tasks, tier_rule_completions(tasks, lambda t: max(
                TIERS, key=lambda k: (lookup.table[(t["task_id"], k)][0] - lam * lookup.table[(t["task_id"], k)][1],
                                      -TIERS.index(k)))), lam)
        rows["per-task oracle (max quality, cheapest among ties)"] = score(
            lookup, tasks, tier_rule_completions(tasks, lambda t: max(
                TIERS, key=lambda k: (lookup.table[(t["task_id"], k)][0], -lookup.table[(t["task_id"], k)][1]))), lam)
        out[split] = rows
        md += [f"#### {split} ({len(ids)} tasks)", "", table(rows, budget, tol), ""]
    return "\n".join(md), out


def parsed_tier(text: str) -> str:
    a = parse_action(text, [(t, "CONTINUE") for t in TIERS])
    return a[0] if a else "INVALID"
