"""Cheap-with-skills vs cheap / mid / frontier on the same tasks (offline, no calls)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .distill import SUCCESS_Q, _ordered, load_log, total_usd

BASE_ARMS = ["cheap", "mid", "frontier"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _se(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def oracle(table: dict[str, dict[str, tuple[float, float]]], arms: list[str]) -> dict:
    """Per task: the arm with the highest quality, ties broken by lower cost."""
    q = c = 0.0
    picks: dict[str, int] = {a: 0 for a in arms}
    for tid, row in table.items():
        best = min(arms, key=lambda a: (-row[a][0], row[a][1]))
        q += row[best][0]
        c += row[best][1]
        picks[best] += 1
    n = len(table)
    return {"quality": q / n, "cost": c / n, "picks": picks, "n": n}


def _ledger(dirpath) -> dict:
    p = Path(dirpath) / "spend.jsonl"
    tot, by = 0.0, {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tot += float(r["usd"])
            by[r.get("purpose", "?")] = by.get(r.get("purpose", "?"), 0.0) + float(r["usd"])
    return {"total": tot, "by_purpose": by}


def analyze(base_log, skilled_log, handbook_path, serve_ledger_dir=None, distill_ledger_dir=None,
            arm_name: str = "cheap+skills", tier: str = "cheap") -> tuple[str, dict]:
    sk_rows = load_log(skilled_log)
    sk = {d["task_id"]: d for d in sk_rows}
    ids = list(sk)
    base = load_log(base_log, ids)
    tab: dict[str, dict] = {}
    cat: dict[str, str] = {}
    meta_sk: dict[str, dict] = {}
    for d in base:
        arm = d["meta"]["arm"]
        if arm in BASE_ARMS:
            tab.setdefault(d["task_id"], {})[arm] = (float(d["quality"]), float(d["total_cost"]))
            cat[d["task_id"]] = d["meta"]["category"]
    for tid, d in sk.items():
        tab.setdefault(tid, {})[arm_name] = (float(d["quality"]), float(d["total_cost"]))
        meta_sk[tid] = d["meta"]
        cat.setdefault(tid, d["meta"]["category"])
    arms = BASE_ARMS + [arm_name]
    tids = [t for t in ids if all(a in tab[t] for a in arms)]
    tab = {t: tab[t] for t in tids}
    hb = json.loads(Path(handbook_path).read_text())
    distill_usd = total_usd(hb)
    n = len(tids)
    L = []
    L.append(f"# Skill handbook for the cheap tier: CRMArena-Pro b2b, {n} tasks\n")
    L.append(f"Arms on the same {n} tasks (the collector's seeded sample, seed 0): `cheap`, `mid`, `frontier` from "
             f"`{base_log}` (first episode per task and arm), `{arm_name}` = the {tier} tier with the "
             f"leave-one-task-out procedural hints, from `{skilled_log}`. Success = quality >= {SUCCESS_Q}. "
             f"Cost/task = agent-call spend of the episode (as in the base log; the CRMArena extractor is "
             f"excluded for every arm). Paired Δ is per task vs `cheap`, ± its standard error.\n")
    L.append("## Overall\n")
    L.append("| arm | quality | ± SE | success | cost/task | Δ quality vs cheap ± SE | tasks up / down |")
    L.append("|---|---|---|---|---|---|---|")
    res = {"n": n, "arms": {}}
    cheap_q = [tab[t]["cheap"][0] for t in tids]
    for a in arms:
        qs = [tab[t][a][0] for t in tids]
        cs = [tab[t][a][1] for t in tids]
        d = [x - y for x, y in zip(qs, cheap_q)]
        up = sum(x > 1e-9 for x in d)
        dn = sum(x < -1e-9 for x in d)
        succ = sum(x >= SUCCESS_Q for x in qs)
        dl = "—" if a == "cheap" else f"{_mean(d):+.3f} ± {_se(d):.3f}"
        ud = "—" if a == "cheap" else f"{up} / {dn}"
        L.append(f"| {a} | {_mean(qs):.3f} | {_se(qs):.3f} | {succ}/{n} | ${_mean(cs):.4f} | {dl} | {ud} |")
        res["arms"][a] = {"quality": _mean(qs), "se": _se(qs), "success": succ, "cost": _mean(cs),
                          "delta_vs_cheap": _mean(d), "delta_se": _se(d), "up": up, "down": dn}
    # flips
    fl_up = [t for t in tids if tab[t]["cheap"][0] < SUCCESS_Q <= tab[t][arm_name][0]]
    fl_dn = [t for t in tids if tab[t][arm_name][0] < SUCCESS_Q <= tab[t]["cheap"][0]]
    hinted = [t for t in tids if (meta_sk[t].get("skills") or {}).get("n_notes", 0) > 0]
    unh = [t for t in tids if t not in hinted]
    L.append(f"\n**Flips (success threshold {SUCCESS_Q}).** cheap fail → {arm_name} success: {len(fl_up)} "
             f"({', '.join(fl_up) or '-'}); cheap success → {arm_name} fail: {len(fl_dn)} "
             f"({', '.join(fl_dn) or '-'}); net {len(fl_up) - len(fl_dn):+d}.\n")
    res["flips_up"], res["flips_down"] = fl_up, fl_dn
    for name, sub in (("tasks served hints", hinted), ("tasks with no leave-one-out notes", unh)):
        if sub:
            d = [tab[t][arm_name][0] - tab[t]["cheap"][0] for t in sub]
            L.append(f"- {name}: {len(sub)}; Δ quality vs cheap {_mean(d):+.3f} ± {_se(d):.3f}")
    if hinted and unh:
        dh = _mean([tab[t][arm_name][0] - tab[t]["cheap"][0] for t in hinted])
        du = _mean([tab[t][arm_name][0] - tab[t]["cheap"][0] for t in unh])
        se = math.sqrt(_se([tab[t][arm_name][0] - tab[t]["cheap"][0] for t in hinted]) ** 2
                       + _se([tab[t][arm_name][0] - tab[t]["cheap"][0] for t in unh]) ** 2)
        up_u = [t for t in fl_up if t in unh]
        L.append(f"\n**Attribution.** The tasks that got no hints are an in-run control: same prompt, same tier, "
                 f"no hint block. Hinted Δ − unhinted Δ = {dh - du:+.3f} ± {se:.3f}. "
                 f"{len(up_u)} of the {len(fl_up)} upward flips are on tasks that were served NO hints "
                 f"({', '.join(up_u) or '-'}). When the difference sits inside its SE, the lift over the logged "
                 f"cheap run is not attributable to the hints. It is consistent with run-to-run variance of "
                 f"the cheap tier, and a resampled cheap control is needed to separate the two.")
        res["hinted_minus_unhinted"], res["hinted_minus_unhinted_se"] = dh - du, se
    # oracle
    o3, o4 = oracle(tab, BASE_ARMS), oracle(tab, arms)
    L.append("\n## Per-task oracle\n")
    L.append("Per task, the arm with the highest quality, ties broken by lower cost.\n")
    L.append("| arms | quality | cost/task | picks |")
    L.append("|---|---|---|---|")
    for nm, o in (("cheap, mid, frontier", o3), (f"+ {arm_name}", o4)):
        L.append(f"| {nm} | {o['quality']:.3f} | ${o['cost']:.4f} | "
                 + ", ".join(f"{k} {v}" for k, v in o["picks"].items()) + " |")
    res["oracle3"], res["oracle4"] = o3, o4
    # per category
    L.append("\n## Per category (mean quality)\n")
    L.append(f"| category | n | notes served | cheap | {arm_name} | mid | frontier |")
    L.append("|---|---|---|---|---|---|---|")
    for c in _ordered({cat[t] for t in tids}):
        ts = [t for t in tids if cat[t] == c]
        ns = sum((meta_sk[t].get("skills") or {}).get("n_notes", 0) > 0 for t in ts)
        L.append(f"| {c} | {len(ts)} | {ns}/{len(ts)} | "
                 + " | ".join(f"{_mean([tab[t][a][0] for t in ts]):.2f}" for a in ["cheap", arm_name, "mid",
                                                                                     "frontier"]) + " |")
    # cost
    L.append("\n## Cost\n")
    sv = _ledger(serve_ledger_dir) if serve_ledger_dir else None
    dv = _ledger(distill_ledger_dir) if distill_ledger_dir else None
    nv = len(hb.get("variants", {}))
    L.append(f"- Distillation: {nv} writer calls (mid tier), ${distill_usd:.4f} total "
             f"(ledger: ${dv['total']:.4f})." if dv else f"- Distillation: {nv} writer calls, ${distill_usd:.4f}.")
    if sv:
        L.append(f"- Serving: ${sv['total']:.4f} on the serve ledger ("
                 + ", ".join(f"{k} ${v:.4f}" for k, v in sorted(sv["by_purpose"].items())) + ").")
    sk_cost = res["arms"][arm_name]["cost"]
    L.append(f"- {arm_name} cost/task ${sk_cost:.4f} vs cheap ${res['arms']['cheap']['cost']:.4f} "
             f"and mid ${res['arms']['mid']['cost']:.4f}; with distillation amortized over these {n} tasks "
             f"${sk_cost + distill_usd / max(n, 1):.4f}/task.")
    res["distill_usd"] = distill_usd
    return "\n".join(L) + "\n", res


def write(out, text: str) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
