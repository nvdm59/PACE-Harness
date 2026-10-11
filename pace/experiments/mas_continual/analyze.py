"""results/mas_continual/RESULTS.md from a run's JSONL logs (days.jsonl, gates.jsonl, meta.jsonl).

Metrics, with R[i, j] = held-out J (quality − λ_J·cost, simulator expected quality, n_heldout fixed
tasks) of the policy IN SERVICE on the last day of stage i, evaluated under stage j, and
R[j−1, j] taken on stage j's first day (the policy that served the event day is the end of
stage j−1):

  BWT_j   = R[T, j] − R[j, j]                  (j < T; Lopez-Paz & Ranzato 2017, as EvoHarnessBench uses it)
  FWT_j   = R[j−1, j] − b_j                    (b_j = a uniform policy over stage j's legal set)
  hit_j   = R[j−1, j] − J of that same event-day policy on stage j−1   what the event cost the policy in
                                                   service (the priced-BWT of pace.drift), both evaluated on the event day
  rec_j   = R[j, j] − R[j−1, j]                what the stage's days won back
  days-to-recover_j   first day of stage j on which the serving policy's held-out J on stage j is
                      within `recover_tol` of the per-generation oracle's; "–" if never
  stationary_j        mean held-out J on stage j over the stage's last 4 days
  regret              Σ over days ≥ warmup+1 of (oracle J of the day's stage − serving policy's J on it)

Section 0 is an authored interpretation passed in with --prelude; everything after it is generated.
  python -m pace.experiments.mas_continual.analyze --run results/mas_continual --mild results/mas_continual/mild \\
      --prelude results/mas_continual/interpretation.md --out results/mas_continual/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import simenv
from .simenv import AGENTS, SKILLS, TOOLS, A, M

ARM_LABEL = {"continual": "continual (cgrpo on replay, gated, learned λ)", "continual_awr": "continual (AWR+IPS, gated, λ = 2)",
             "frozen": "frozen (trained days 1–5)", "so_static": "SkillOrchestra-style static router",
             "so_relearn": "SkillOrchestra-style router, re-learned per stage", "always_verify": "always_verify",
             "never_verify": "never_verify"}
SHORT = {"continual": "continual (cgrpo)", "continual_awr": "continual (AWR)", "frozen": "frozen", "so_static": "router static",
         "so_relearn": "router re-learned", "always_verify": "always_verify", "never_verify": "never_verify"}
ARMS = list(ARM_LABEL)


def load(run: Path) -> dict:
    days = [json.loads(l) for l in (run / "days.jsonl").read_text().splitlines() if l.strip()]
    gates = [json.loads(l) for l in (run / "gates.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((run / "meta.jsonl").read_text().splitlines()[0])
    return {"days": days, "gates": gates, "meta": meta}


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}" if isinstance(x, float) and abs(x) < 10 else f"{x}"


def f3u(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def f4(x) -> str:
    return "–" if x is None else f"{x:.4f}"


# ------------------------------------------------------------------ metrics
def stage_ranges(meta: dict, n_days: int) -> list[tuple[int, int, int]]:
    """[(gen index, first day, last day)]"""
    gens = [g for g in meta["generations"] if g["day"] <= n_days]
    out = []
    for k, g in enumerate(gens):
        last = gens[k + 1]["day"] - 1 if k + 1 < len(gens) else n_days
        out.append((g["index"], g["day"], last))
    return out


def heldout_table(days: list[dict]) -> dict:
    """H[arm][day][gen] -> {quality, cost, J}"""
    H: dict = defaultdict(dict)
    for r in days:
        H[r["arm"]][r["day"]] = {int(k): v for k, v in r["heldout"].items()}
    return H


def metrics(data: dict, recover_tol: float, warmup: int) -> dict:
    days, meta = data["days"], data["meta"]
    n_days = max(r["day"] for r in days)
    H = heldout_table(days)
    ranges = stage_ranges(meta, n_days)
    oracle = {int(k): v["J"] for k, v in meta["oracle"].items()}
    uniform = {int(k): v["J"] for k, v in meta.get("uniform", {}).items()}
    gen_of_day = {d: gi for gi, a, b in ranges for d in range(a, b + 1)}
    T = ranges[-1][0]
    out: dict = {}
    for arm in H:
        first_day = min(H[arm])
        R = {}
        for gi, a, b in ranges:
            for gj, _, _ in ranges:
                if gj <= gi and b in H[arm] and gj in H[arm][b]:
                    R[(gi, gj)] = H[arm][b][gj]["J"]
        per_stage = []
        for gi, a, b in ranges:
            if a < first_day:
                per_stage.append({"gen": gi, "available": False})
                continue
            prev = H[arm].get(a, {})
            R_prev_j = prev.get(gi, {}).get("J")
            R_prev_prev = prev.get(gi - 1, {}).get("J") if gi > 0 else None
            R_jj = R.get((gi, gi))
            bwt = (R[(T, gi)] - R_jj) if gi < T and (T, gi) in R and R_jj is not None else None
            fwt = (R_prev_j - uniform[gi]) if gi > 0 and R_prev_j is not None and gi in uniform else None
            hit = (R_prev_j - R_prev_prev) if gi > 0 and R_prev_j is not None and R_prev_prev is not None else None
            rec = (R_jj - R_prev_j) if R_jj is not None and R_prev_j is not None else None
            dtr = None
            for d in range(a, b + 1):
                if d in H[arm] and gi in H[arm][d] and H[arm][d][gi]["J"] >= oracle[gi] - recover_tol:
                    dtr = d - a + 1
                    break
            tail = [H[arm][d][gi]["J"] for d in range(max(a, b - 3), b + 1) if d in H[arm] and gi in H[arm][d]]
            stat = float(np.mean(tail)) if tail else None
            gap_end = (R_jj - oracle[gi]) if R_jj is not None else None
            cost = [r["served"]["cost"] for r in days if r["arm"] == arm and a <= r["day"] <= b]
            q = [r["served"]["expected_quality"] for r in days if r["arm"] == arm and a <= r["day"] <= b]
            per_stage.append({"gen": gi, "available": True, "R_prev_j": R_prev_j, "R_jj": R_jj, "bwt": bwt, "fwt": fwt, "hit": hit,
                              "recovery": rec, "days_to_recover": dtr, "stationary_J": stat, "gap_to_oracle_end": gap_end,
                              "served_cost": float(np.mean(cost)) if cost else None, "served_quality": float(np.mean(q)) if q else None,
                              "oracle_J": oracle[gi]})
        regret = sum(oracle[gen_of_day[d]] - H[arm][d][gen_of_day[d]]["J"] for d in H[arm] if d > warmup and gen_of_day[d] in H[arm][d])
        bwts = [s["bwt"] for s in per_stage if s.get("bwt") is not None]
        fwts = [s["fwt"] for s in per_stage if s.get("fwt") is not None]
        out[arm] = {"per_stage": per_stage, "regret": regret, "mean_bwt": float(np.mean(bwts)) if bwts else None,
                    "mean_fwt": float(np.mean(fwts)) if fwts else None, "n_days_counted": sum(1 for d in H[arm] if d > warmup)}
    return {"arms": out, "ranges": ranges, "oracle": oracle, "uniform": uniform, "T": T, "n_days": n_days}


# ------------------------------------------------------------------ tables
def table_generations(meta: dict) -> str:
    lines = ["| stage | axis | day | date | what changed |", "|---|---|---|---|---|"]
    for g in meta["generations"]:
        lines.append(f"| {g['name']} | {g['axis']} | {g['day']} | {g['date']} | {g['description']} |")
    return "\n".join(lines)


def table_events(mt: dict, which: list[str] | None = None) -> str:
    arms = [a for a in ARMS if a in mt["arms"]] if which is None else which
    lines = []
    for gi, a, b in mt["ranges"][1:]:
        axis = [g for g in mt["ranges"] if g[0] == gi][0]
        lines += [f"**Stage g{gi} (day {a}; oracle held-out J {mt['oracle'][gi]:.3f}, uniform {mt['uniform'].get(gi, float('nan')):.3f})**", "",
                  "| arm | hit at event | FWT | recovery | days to recover | stationary J (last 4 days) | gap to oracle at stage end | BWT (final) | cost/request (stage) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            s = [x for x in mt["arms"][arm]["per_stage"] if x["gen"] == gi][0]
            if not s["available"]:
                continue
            lines.append(f"| {ARM_LABEL[arm]} | {f3(s['hit'])} | {f3(s['fwt'])} | {f3(s['recovery'])} | "
                         f"{s['days_to_recover'] if s['days_to_recover'] is not None else '–'} | {f3u(s['stationary_J'])} | {f3(s['gap_to_oracle_end'])} | "
                         f"{f3(s['bwt']) if gi < mt['T'] else 'n/a'} | ${f4(s['served_cost'])} |")
        lines.append("")
    return "\n".join(lines)


def table_summary(mt: dict, warmup: int) -> str:
    lines = [f"| arm | mean BWT (stages < T) | mean FWT | cumulative regret vs oracle (days {warmup + 1}–{mt['n_days']}) | regret per day |",
             "|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in mt["arms"]:
            continue
        m = mt["arms"][arm]
        lines.append(f"| {ARM_LABEL[arm]} | {f3(m['mean_bwt'])} | {f3(m['mean_fwt'])} | {m['regret']:.3f} | {m['regret'] / max(m['n_days_counted'], 1):.4f} |")
    return "\n".join(lines)


def table_stationary(mt: dict) -> str:
    lines = ["| arm | " + " | ".join(f"g{gi}" for gi, _, _ in mt["ranges"]) + " |", "|---|" + "---|" * len(mt["ranges"])]
    lines.append("| oracle | " + " | ".join(f"{mt['oracle'][gi]:.3f}" for gi, _, _ in mt["ranges"]) + " |")
    for arm in ARMS:
        if arm not in mt["arms"]:
            continue
        cells = []
        for s in mt["arms"][arm]["per_stage"]:
            cells.append(f3u(s["stationary_J"]) if s["available"] else "–")
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_days(days: list[dict], arm: str = "continual") -> str:
    lines = ["| day | stage | served quality (exp.) | cost/request | J | λ̄ (served) | gate | DR quality [90% CI] | DR cost upper | ESS (per-decision) | truth |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in days:
        if r["arm"] != arm:
            continue
        g = r.get("gate", {})
        s = r["served"]
        lines.append(f"| {r['day']}{' *' if r['event'] else ''} | {r['generation']} | {s['expected_quality']:.3f} | ${s['cost']:.4f} | {s['J']:.3f} | "
                     f"{r['lambda']:.2f} | {g.get('decision', '–')} | "
                     + (f"{g['dr_quality'][0]:.3f} [{g['dr_quality'][1]:.3f}, {g['dr_quality'][2]:.3f}]" if g else "–")
                     + f" | {f4(g['dr_cost'][2]) if g else '–'} | {f'{g['per_decision_ess']:.0f}/{g['per_decision_n']}' if g else '–'} | "
                     + (("ok" if g["truth"] else "BAD") if g else "–") + " |")
    return "\n".join(lines)


def table_cost_over_time(days: list[dict]) -> str:
    arms = [a for a in ARMS if any(r["arm"] == a for r in days)]
    lines = ["| day | stage | " + " | ".join(SHORT[a] for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
    by = defaultdict(dict)
    gen = {}
    for r in days:
        by[r["day"]][r["arm"]] = r["served"]["cost"]; gen[r["day"]] = (r["generation"], r["event"])
    for d in sorted(by):
        if d % 2 and not gen[d][1]:
            continue
        lines.append(f"| {d}{' *' if gen[d][1] else ''} | {gen[d][0]} | " + " | ".join(f"${by[d][a]:.4f}" if a in by[d] else "–" for a in arms) + " |")
    return "\n".join(lines)


def table_gates(gates: list[dict], arm: str) -> tuple[str, str]:
    rows = [g for g in gates if g["arm"] == arm]
    n = len(rows)
    prom = [g for g in rows if g["decision"] == "PROMOTE"]
    rev = [g for g in rows if g["decision"] == "REVERT"]
    right_rev = [g for g in rev if not g["truth"]["acceptable"]]
    false_alarm = [g for g in rev if g["truth"]["acceptable"]]
    right_prom = [g for g in prom if g["truth"]["acceptable"]]
    missed = [g for g in prom if not g["truth"]["acceptable"]]
    summary = (f"| {ARM_LABEL[arm]} | {n} | {len(prom)} | {len(rev)} | {len(right_rev)} | {len(false_alarm)} | {len(right_prom)} | {len(missed)} | "
               f"{100 * (len(right_rev) + len(right_prom)) / max(n, 1):.0f}% | {np.mean([g['per_decision_ess'] / g['per_decision_n'] for g in rows]):.2f} | "
               f"{np.mean([g['dr_quality'][2] - g['dr_quality'][1] for g in rows]):.3f} | ${np.mean([g['dr_cost'][2] - g['dr_cost'][1] for g in rows]):.4f} |")
    prevented = ["| day | stage | candidate true quality / cost | incumbent true quality / cost | over budget | regressed | gate said |",
                 "|---|---|---|---|---|---|---|"]
    for g in rows:
        t = g["truth"]
        if t["over_budget"] or t["regressed"]:
            prevented.append(f"| {g['day']} | {g['generation']} | {t['candidate']['quality']:.3f} / ${t['candidate']['cost']:.4f} | "
                             f"{t['incumbent']['quality']:.3f} / ${t['incumbent']['cost']:.4f} | {'yes' if t['over_budget'] else '–'} | "
                             f"{'yes' if t['regressed'] else '–'} | {g['decision']}{' (missed)' if g['decision'] == 'PROMOTE' else ''} |")
    return summary, "\n".join(prevented) if len(prevented) > 2 else "(no candidate would have violated the budget or regressed quality)"


def table_estimator_accuracy(gates: list[dict], warmup: int) -> str:
    """Bias, RMSE and 90%-CI coverage of the gate's DR quality / cost (decision cost, net of the prefix)
    and of the per-decision SNIPS quality, against the candidate's true value."""
    lines = ["| arm | days | n | DR quality: bias / RMSE / CI covers truth | DR cost: bias / RMSE / CI covers truth | per-decision SNIPS quality: bias / RMSE | mean n per gate |",
             "|---|---|---|---|---|---|---|"]
    for arm in ("continual", "continual_awr"):
        for label, sel in (("warm-up", lambda g: g["day"] <= warmup), ("after warm-up", lambda g: g["day"] > warmup)):
            rows = [g for g in gates if g["arm"] == arm and sel(g)]
            if not rows:
                continue
            tq = np.array([g["truth"]["candidate"]["quality"] for g in rows])
            tc = np.array([g["truth"]["candidate"]["cost"] - g["prefix_cost"] for g in rows])
            dq = np.array([g["dr_quality"][0] for g in rows]); dc = np.array([g["dr_cost"][0] for g in rows])
            sq = np.array([g["per_decision_snips"][0] for g in rows])
            cov_q = np.mean([g["dr_quality"][1] <= t <= g["dr_quality"][2] for g, t in zip(rows, tq)])
            cov_c = np.mean([g["dr_cost"][1] <= t <= g["dr_cost"][2] for g, t in zip(rows, tc)])
            lines.append(f"| {ARM_LABEL[arm]} | {label} | {len(rows)} | {np.mean(dq - tq):+.3f} / {np.sqrt(np.mean((dq - tq) ** 2)):.3f} / {100 * cov_q:.0f}% | "
                         f"{np.mean(dc - tc):+.4f} / {np.sqrt(np.mean((dc - tc) ** 2)):.4f} / {100 * cov_c:.0f}% | "
                         f"{np.mean(sq - tq):+.3f} / {np.sqrt(np.mean((sq - tq) ** 2)):.3f} | {np.mean([g['n'] for g in rows]):.0f} |")
    return "\n".join(lines)


def crossover_tables(days: list[dict], gates: list[dict], warmup: int) -> str:
    """continual − never_verify, day by day and by stage window, on the served stream (expected quality,
    J at the stage λ̄_g) and on the held-out set of the day's stage; cumulative regret vs the oracle
    restricted to later windows, for every arm."""
    by = defaultdict(dict)
    for r in days:
        by[r["arm"]][r["day"]] = r
    n_days = max(r["day"] for r in days)
    gen = {r["day"]: (r["gen_index"], r["generation"], r["event"]) for r in days}
    prom = {g["day"] for g in gates if g["arm"] == "continual" and g["decision"] == "PROMOTE"}
    gap_s = lambda d: by["continual"][d]["served"]["J"] - by["never_verify"][d]["served"]["J"]
    gap_h = lambda d: by["continual"][d]["heldout"][str(gen[d][0])]["J"] - by["never_verify"][d]["heldout"][str(gen[d][0])]["J"]
    lines = ["**By stage window** (mean over the stage's days):", "",
             "| stage | days | continual − never_verify, served J | continual − never_verify, held-out J | never_verify − oracle, held-out |",
             "|---|---|---|---|---|"]
    for g in sorted({v[0] for v in gen.values()}):
        ds = [d for d in range(1, n_days + 1) if gen[d][0] == g]
        nv_gap = np.mean([by["never_verify"][d]["heldout"][str(g)]["J"] - by["never_verify"][d]["oracle_J"] for d in ds])
        lines.append(f"| g{g} | {ds[0]}–{ds[-1]} | {np.mean([gap_s(d) for d in ds]):+.3f} | {np.mean([gap_h(d) for d in ds]):+.3f} | {nv_gap:+.3f} |")
    first_pos = next((d for d in range(1, n_days + 1) if all(gap_s(x) > 0 for x in range(d, min(d + 3, n_days + 1)))), None)
    lines += ["", f"First day from which the served gap stays positive for three consecutive days: **day {first_pos}**." if first_pos else "",
              f"First promotion of the continual arm: day {min(prom) if prom else '–'}; promotions on days {sorted(prom)}.", "",
              "**Cumulative regret vs the per-stage oracle, restricted to later windows** (held-out, as in section 3; and on the served "
              "stream, where the learner's J includes its ε = 0.1 exploration mass and `never_verify` pays none; served tasks are a "
              "different draw from the held-out set, so the served column is indicative):", "",
              "| arm | held-out, days " + f"{warmup + 1}–{n_days} | held-out, days 17–{n_days} | held-out, days 25–{n_days} | served, days {warmup + 1}–{n_days} | served, days 17–{n_days} | served, days 25–{n_days} |",
              "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        if arm not in by:
            continue
        cells = []
        for key in ("heldout", "served"):
            for lo in (warmup + 1, 17, 25):
                ds = [d for d in range(lo, n_days + 1) if d in by[arm]]
                val = sum(by[arm][d]["oracle_J"] - (by[arm][d]["heldout"][str(gen[d][0])]["J"] if key == "heldout" else by[arm][d]["served"]["J"]) for d in ds)
                cells.append(f"{val:.3f}")
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines += ["", "**Day by day** (`*` event day, `P` the continual arm promoted a candidate that day; the bar is the served gap, one block per 0.01):", "",
              "| day | stage | continual served J | never_verify served J | gap, served | gap, held-out | |", "|---|---|---|---|---|---|---|"]
    for d in range(1, n_days + 1):
        gs = gap_s(d)
        bar = ("+" * int(round(gs / 0.01))) if gs > 0 else ("-" * int(round(-gs / 0.01)))
        lines.append(f"| {d}{' *' if gen[d][2] else ''}{' P' if d in prom else ''} | {gen[d][1]} | {by['continual'][d]['served']['J']:.3f} | "
                     f"{by['never_verify'][d]['served']['J']:.3f} | {gs:+.3f} | {gap_h(d):+.3f} | `{bar}` |")
    return "\n".join(lines)


def table_calibration(meta: dict) -> str:
    cc = meta.get("calibration_check", {})
    sim, mea = cc.get("simulated", {}), cc.get("measured", {})
    lines = ["| mode | accuracy sim / measured | cost/task sim / measured | role calls sim / measured | verifier calls sim / measured | cap hits sim / measured | P(flag ∣ wrong) sim / measured | P(flag ∣ correct) sim / measured |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ("always_verify", "never_verify"):
        if m not in sim:
            continue
        s, r = sim[m], mea[m]
        fw = f"{s['p_flag_given_wrong']:.2f} / {M.flag_given_wrong[0] / M.flag_given_wrong[1]:.2f}" if m == "always_verify" else "– / –"
        fc = f"{s['p_flag_given_correct']:.2f} / {M.flag_given_correct[0] / M.flag_given_correct[1]:.2f}" if m == "always_verify" else "– / –"
        lines.append(f"| {m} | {s['accuracy']:.3f} / {r['accuracy']:.3f} | ${s['cost']:.4f} / ${r['cost']:.4f} | {s['calls']:.2f} / {r['calls']:.2f} | "
                     f"{s['verifier_calls']:.2f} / {r['verifier_calls']:.2f} | {100 * s['cap_hits']:.0f}% / {100 * r['cap_hits'] / r['n']:.0f}% | {fw} | {fc} |")
    return "\n".join(lines)


def table_params() -> str:
    p = simenv.measured_params()
    k = M.worker_kind_counts["mid"]
    n_mid = sum(k.values())
    rows = [
        ("initial plan length P(1, 2, 3)", f"{p.plan_len_p[0]:.3f} / {p.plan_len_p[1]:.3f} / {p.plan_len_p[2]:.3f}", "measured", "130 planner calls, results/mas/mas_b2b.jsonl (4 / 22 / 104)"),
        ("re-plan length", "1 sub-query", "measured", "15 re-plans, all emitted 1"),
        ("data-2 (Sonnet) step kind rows / empty / error / invalid / respond", " / ".join(f"{k[x] / n_mid:.3f}" for x in simenv.KINDS), "measured", f"{n_mid} worker calls"),
        ("data-1 (Haiku) step kind", " / ".join(f"{v / 11:.2f}" for v in M.worker_kind_counts['cheap'].values()), "measured, thin", "11 worker calls (controller:uniform)"),
        ("data-3 (Opus) step kind", " / ".join(f"{v / 10:.2f}" for v in M.worker_kind_counts['frontier'].values()), "measured, thin", "10 worker calls"),
        ("retried step: same kind distribution as a first attempt", "rows 0.68 vs 0.76", "measured-consistent", "53 retries vs 299 first attempts; difference inside the noise"),
        ("tokens planner / worker / verifier / answer (in, out)", "5,029/279 · 8,369/158 · 711/62 · 12,250/91", "measured", "ledger means per role (cheap / mid / cheap / mid)"),
        ("worker input tokens by position", f"{p.worker_in_base:.0f} + {p.worker_in_slope:.0f}·k", "measured", "mid worker calls at positions 0 / 1 / 2: 6,833 / 8,571 / 10,330"),
        ("cheap / frontier tokens per role", "planner 6,996/159 · 6,902/311; worker 6,647/115 · 9,598/196; answer 7,393/245 · 15,227/332", "measured, thin", "5–15 calls each; priced at pace/prices.py list prices"),
        ("verifier P(fail ∣ empty step)", f"{p.verifier_fail_empty:.3f}", "measured", "23 of 24 verifier calls on empty steps"),
        ("verifier P(fail ∣ rows step)", f"{p.verifier_fail_rows['verify-1']['good']:.3f}", "measured", "72 of 124; the same whether the step was good or bad (the measured verifier is uninformative: P(flag∣wrong) 0.92, P(flag∣correct) 0.96)"),
        ("verifier structural fail on error / invalid steps", "1.0 at $0", "measured", "12 of 12"),
        ("answer competence per (agent, category)", "logit of the MAS per-category accuracy (answer on mid) + single-agent (category, tier) offset", "measured, shrunk", f"130 MAS episodes, 150 single-agent episodes; pseudo-count {A.shrink_pseudo}"),
        ("P(good ∣ rows) per data agent", " / ".join(f"{p.good_given_rows[f'data-{i}']:.2f}" for i in (1, 2, 3)), "ASSUMED", "not identifiable from the logs; ordered like the single-agent tier accuracies"),
        ("coverage → answer logit", f"{p.coverage_logit} × (coverage − {p.coverage_center:.3f})", "ASSUMED (slope); centre self-calibrated", "centre = never_verify's expected coverage, so never_verify reproduces the category base rates"),
        ("task-level logit noise", f"sd {p.task_noise_sd}", "ASSUMED", ""),
        ("max role calls / replans / retries", "8 / 1 / 1", "measured (configuration)", "the real orchestrator's defaults"),
        ("worker 'respond' on the prefix call", "disallowed", "ASSUMED", "the first worker call is always data-2 and always queries"),
    ]
    for a in AGENTS:
        if a.stage > 0:
            rows.append((f"{a.agent_id} ({a.tier}, {a.mode}, stage {a.stage})", a.note, "ASSUMED", ""))
    for t in TOOLS:
        rows.append((f"tool {t.tool_id} (stage {t.stage})", t.note, "ASSUMED", f"priced as a verifier-sized cheap call ({A.tool_cost_tokens[0]}/{A.tool_cost_tokens[1]} tokens)"))
    for s in SKILLS:
        rows.append((f"skill {s.skill_id} (stage {s.stage})", s.note, "ASSUMED", "same price as a tool call"))
    rows.append(("planner v2 plan length P(1..4)", str(A.planner_v2_plan_len_p), "ASSUMED", "stage 4"))
    lines = ["| parameter | value | status | source / note |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def table_handbooks(meta: dict) -> str:
    lines = ["| handbook | fitted on day | episodes (learn / tune) | λ_c chosen | SN-IPS quality / cost at λ_c (ESS) | within budget |", "|---|---|---|---|---|---|"]
    for name, hb in meta.get("handbooks", {}).items():
        pick = [r for r in hb["tuning"] if r["lambda_c"] == hb["lambda_c"]][0]
        lines.append(f"| {name} | {hb['day']} | {hb['n_learn']} / {hb['n_tune']} | {hb['lambda_c']} | {pick['quality']:.3f} / ${pick['cost']:.4f} (ESS {pick['ess']:.0f}/{pick['n']}) | "
                     f"{'yes' if hb['within_budget'] else 'no (cheapest taken)'} |")
    return "\n".join(lines)


def table_mix(days: list[dict]) -> str:
    lines = ["| stage | arm | top actions served (share of decisions, last day of the stage) |", "|---|---|---|"]
    last = {}
    for r in days:
        last[(r["generation"], r["arm"])] = r
    for (g, arm), r in last.items():
        if arm in ("continual", "frozen", "so_static", "so_relearn"):
            lines.append(f"| {g} | {SHORT[arm]} | " + ", ".join(f"{a} {100 * v:.0f}%" for a, v in r["served"]["action_mix"].items()) + " |")
    return "\n".join(lines)


def report(run: Path, mild: Path | None, prelude: Path | None) -> str:
    data = load(run)
    cfg = data["meta"]["config"]
    mt = metrics(data, cfg["recover_tol"], cfg["warmup_days"])
    out = ["# Continual MAS orchestrator: results", ""]
    if prelude and prelude.exists():
        out += ["## 0. Interpretation", "", prelude.read_text().strip(), "", "Sections 1–11 below are generated by `analyze.py`; section 0 is authored and passed in with `--prelude`.", ""]
    lam_txt = ", ".join(f"g{k} {v['lambda']:.2f}" for k, v in sorted(data["meta"]["oracle"].items()))
    out += [f"Run: schedule `{cfg['schedule']}`, {cfg['n_days']} days × {cfg['n_tasks']} tasks per arm, budget B = ${cfg['budget']:.3f} per task "
            f"(gate tol ${cfg['tol']:.3f}, ε_q {cfg['eps_q']}), served ε = {cfg['eps']}, held-out {cfg['n_heldout']} tasks per stage, "
            f"seed {cfg['seed']}; {data['meta']['n_actions']} actions, {data['meta']['n_features']} features; runtime {data['meta']['runtime_s']} s on CPU. "
            f"J_g = quality − λ̄_g · cost (USD) with λ̄_g the per-stage oracle's learned price ({lam_txt} per USD): at the price where the budget "
            f"binds, the budgeted optimum maximizes the Lagrangian over every policy, so J_g has a ceiling and regret is well defined.", ""]
    out += ["## 1. Harness stages", "", table_generations(data["meta"]), ""]
    out += ["## 2. Per-stage transfer: hit, FWT, recovery, days to recover, stationary J, BWT", "",
            "Held-out J of the policy in service each day (simulator expected quality on the stage's fixed task set). `hit` = J on the new stage minus J on "
            "the old stage the day before the event (what the event cost the policy in service); `FWT` = that same event-day J minus a uniform policy's; "
            "`recovery` = J at the stage's end minus J on the event day; `BWT` = J on this stage at day 40 minus J on it at the stage's end "
            "(the frozen arm's BWT is 0 by construction, its policy never changes).", "",
            table_events(mt),
            "## 2b. Crossover: the continual arm against never_verify over the stream", "",
            crossover_tables(data["days"], data["gates"], cfg["warmup_days"]), "",
            "## 3. Whole-stream summary", "", table_summary(mt, cfg["warmup_days"]), "",
            "## 4. Stationary periods: mean held-out J over the last 4 days of each stage", "", table_stationary(mt), "",
            "## 5. The continual (cgrpo) arm, day by day", "", "`*` marks an event day. λ̄ is the served policy's learned price of cost (per USD). "
            "`truth` is whether the candidate was acceptable on the simulator (within budget + tol and not more than ε_q below the incumbent's true quality).", "",
            table_days(data["days"], "continual"), ""]
    out += ["## 6. Gate history", "",
            "| arm | candidates | PROMOTE | REVERT | REVERT right (candidate bad) | REVERT false alarm (candidate fine) | PROMOTE right | PROMOTE missed (candidate bad) | correct | mean per-decision ESS / n | mean DR quality CI width | mean DR cost CI width |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    prevented = {}
    for arm in ("continual", "continual_awr"):
        s, p = table_gates(data["gates"], arm)
        out.append(s)
        prevented[arm] = p
    out += ["", "### 6a. Gate estimators against simulator truth (the candidate's true held-out value on the current stage)", "",
            table_estimator_accuracy(data["gates"], cfg["warmup_days"]), "",
            "### 6b. What the gate prevented (candidates that were over budget or regressed quality on the simulator)", ""]
    for arm in ("continual", "continual_awr"):
        out += [f"**{ARM_LABEL[arm]}**", "", prevented[arm], ""]
    out += ["## 7. Cost per request over time (served, every other day and event days)", "", table_cost_over_time(data["days"]), ""]
    out += ["## 8. What each policy serves at the end of each stage", "", table_mix(data["days"]), ""]
    out += ["## 9. SkillOrchestra-style handbooks", "", table_handbooks(data["meta"]), ""]
    out += ["## 10. Simulator calibration", "", "Stage-0 harness, 1,000 simulated tasks per mode, against the real run's 50 tasks per mode:", "",
            table_calibration(data["meta"]), "", "Every parameter, with its status:", "", table_params(), ""]
    if mild and (mild / "days.jsonl").exists():
        dm = load(mild)
        mm = metrics(dm, dm["meta"]["config"]["recover_tol"], dm["meta"]["config"]["warmup_days"])
        out += ["## 11. Sensitivity: the mild schedule", "",
                f"Same days and axes, smaller effects (see `drift.SCHEDULES['mild']`); runtime {dm['meta']['runtime_s']} s.", "",
                table_summary(mm, dm["meta"]["config"]["warmup_days"]), "", table_events(mm), "Stationary J per stage:", "", table_stationary(mm), ""]
    return "\n".join(out)


# ------------------------------------------------------------------ RAO credit and meta-init (RESULTS_rao_metainit.md)
def delegation_by_stage(days: list[dict], arm: str = "continual", full: bool = True) -> dict:
    """{gen: {"share": stage mean of the daily delegation share of served decisions, "by_agent": {agent: mean share},
    "early": mean share over the stage's first 4 days}}. `full=False` recomputes the share from the logged
    top-6 `action_mix` (the only record in runs that predate the `delegate_share` field)."""
    from .rao_reward import delegation_share_from_mix
    per: dict = defaultdict(list)
    for r in days:
        if r["arm"] != arm:
            continue
        sv = r["served"]
        sh = sv["delegate_share"] if (full and "delegate_share" in sv) else delegation_share_from_mix(sv["action_mix"])
        per[r["gen_index"]].append((r["day"], sh, sv.get("delegate_by_agent", {}) if full else {}))
    out = {}
    for g, rows in sorted(per.items()):
        rows.sort()
        agents = sorted({a for _, _, b in rows for a in b})
        out[g] = {"share": float(np.mean([x for _, x, _ in rows])), "early": float(np.mean([x for _, x, _ in rows[:4]])),
                  "by_agent": {a: float(np.mean([b.get(a, 0.0) for _, _, b in rows])) for a in agents},
                  "early_by_agent": {a: float(np.mean([b.get(a, 0.0) for _, _, b in rows[:4]])) for a in agents}}
    return out


def regret_window(data: dict, arm: str, lo: int, hi: int) -> float:
    mt_ranges = stage_ranges(data["meta"], max(r["day"] for r in data["days"]))
    gen_of_day = {d: gi for gi, a, b in mt_ranges for d in range(a, b + 1)}
    oracle = {int(k): v["J"] for k, v in data["meta"]["oracle"].items()}
    H = heldout_table(data["days"])
    return float(sum(oracle[gen_of_day[d]] - H[arm][d][gen_of_day[d]]["J"] for d in H[arm] if lo <= d <= hi))


def first_promotion(gates: list[dict], arm: str = "continual") -> int | None:
    d = [g["day"] for g in gates if g["arm"] == arm and g["decision"] == "PROMOTE"]
    return min(d) if d else None


def run_summary(run: Path, arms: tuple = ("continual", "never_verify", "always_verify")) -> dict:
    data = load(run)
    cfg = data["meta"]["config"]
    mt = metrics(data, cfg["recover_tol"], cfg["warmup_days"])
    out = {}
    for arm in arms:
        if arm not in mt["arms"]:
            continue
        m = mt["arms"][arm]
        out[arm] = {"regret_6_40": m["regret"], "regret_17_40": regret_window(data, arm, 17, cfg["n_days"]),
                    "mean_bwt": m["mean_bwt"], "mean_fwt": m["mean_fwt"],
                    "stationary": {s["gen"]: s["stationary_J"] for s in m["per_stage"]},
                    "bwt": {s["gen"]: s["bwt"] for s in m["per_stage"]}, "fwt": {s["gen"]: s["fwt"] for s in m["per_stage"]},
                    "dtr": {s["gen"]: s["days_to_recover"] for s in m["per_stage"]},
                    "delegation": delegation_by_stage(data["days"], arm),
                    "first_promotion": first_promotion(data["gates"], arm) if arm == "continual" else None,
                    "served_J_1_12": float(np.mean([r["served"]["J"] for r in data["days"] if r["arm"] == arm and r["day"] <= 12]))}
    out["_meta"] = {"runtime_s": data["meta"]["runtime_s"], "meta_init": data["meta"].get("meta_init"), "oracle": mt["oracle"]}
    return out


def _ms(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{np.mean(xs):.3f} ± {np.std(xs, ddof=1) / np.sqrt(len(xs)):.3f}"


def rao_metainit_tables(root: Path, committed: Path | None = None) -> dict:
    """Tables over root/{base,rao,meta}/seed*/ (5 seeds each)."""
    S = {v: [run_summary(p) for p in sorted((root / v).glob("seed*")) if (p / "meta.jsonl").exists()] for v in ("base", "rao", "meta")}
    gens = sorted(S["base"][0]["continual"]["stationary"])
    lines = []
    # ---- table A: delegation share
    lines += ["| row | " + " | ".join(f"g{g}" for g in gens) + " |", "|---|" + "---|" * len(gens)]
    if committed is not None and (committed / "days.jsonl").exists():
        dc = delegation_by_stage(load(committed)["days"], "continual", full=False)
        lines.append("| committed run (seed 0, all arms; top-6 action_mix, stage mean) | " + " | ".join(f"{dc[g]['share']:.2f}" for g in gens) + " |")
    for v, lab in (("base", "continual, credit none"), ("rao", "continual, credit RAO (λ_d 0.5)")):
        lines.append(f"| {lab}: delegation share | " + " | ".join(_ms([s['continual']['delegation'][g]['share'] for s in S[v]]) for g in gens) + " |")
        lines.append(f"| {lab}: share, first 4 days of stage | " + " | ".join(_ms([s['continual']['delegation'][g]['early'] for s in S[v]]) for g in gens) + " |")
        for agent in ("data-1", "data-2", "data-3", "data-4"):
            lines.append(f"| {lab}: → {agent} | " + " | ".join(_ms([s['continual']['delegation'][g]['by_agent'].get(agent, 0.0) for s in S[v]]) for g in gens) + " |")
        lines.append(f"| {lab}: → data-4, first 4 days of stage | " + " | ".join(_ms([s['continual']['delegation'][g]['early_by_agent'].get('data-4', 0.0) for s in S[v]]) for g in gens) + " |")
    for arm in ("never_verify", "always_verify"):
        lines.append(f"| {arm}: delegation share | " + " | ".join(_ms([s[arm]['delegation'][g]['share'] for s in S['base']]) for g in gens) + " |")
    tA = "\n".join(lines)
    # ---- table A2: J / transfer / regret
    lines = ["| row | stationary J " + " | ".join(f"g{g}" for g in gens) + " | mean BWT | mean FWT | regret 6–40 | regret 17–40 |",
             "|---|" + "---|" * (len(gens) + 4)]
    lines.append("| oracle | " + " | ".join(_ms([s["_meta"]["oracle"][g] for s in S["base"]]) for g in gens) + " | | | | |")
    rows = [("base", "continual", "continual, credit none"), ("rao", "continual", "continual, credit RAO"),
            ("base", "never_verify", "never_verify"), ("base", "always_verify", "always_verify")]
    for v, arm, lab in rows:
        ss = [s[arm] for s in S[v]]
        lines.append(f"| {lab} | " + " | ".join(_ms([x["stationary"][g] for x in ss]) for g in gens)
                     + f" | {_ms([x['mean_bwt'] for x in ss])} | {_ms([x['mean_fwt'] for x in ss])} | {_ms([x['regret_6_40'] for x in ss])} | {_ms([x['regret_17_40'] for x in ss])} |")
    paired = {"d_share": {g: [r["continual"]["delegation"][g]["share"] - b["continual"]["delegation"][g]["share"] for r, b in zip(S["rao"], S["base"])] for g in gens},
              "d_J": {g: [r["continual"]["stationary"][g] - b["continual"]["stationary"][g] for r, b in zip(S["rao"], S["base"])] for g in gens},
              "d_regret": [r["continual"]["regret_6_40"] - b["continual"]["regret_6_40"] for r, b in zip(S["rao"], S["base"])]}
    lines.append("| RAO − none (paired by seed): Δ stationary J | " + " | ".join(_ms(paired["d_J"][g]) for g in gens) + f" | | | {_ms(paired['d_regret'])} | |")
    lines.append("| RAO − none (paired by seed): Δ delegation share | " + " | ".join(_ms(paired["d_share"][g]) for g in gens) + " | | | | |")
    tA2 = "\n".join(lines)
    # ---- table B: meta-init
    lines = ["| init | first promotion day | days to recover g1 / g2 / g3 / g4 (never = –; n recovered of 5) | served J days 1–12 | regret 6–40 | regret 17–40 | mean BWT | mean FWT |",
             "|---|---|---|---|---|---|---|---|"]
    for v, lab in (("base", "zero (uniform policy)"), ("meta", "meta (Reptile over g0–g3)")):
        ss = [s["continual"] for s in S[v]]
        fp = [x["first_promotion"] for x in ss]
        dtr = []
        for g in gens[1:]:
            vals = [x["dtr"][g] for x in ss]
            got = [d for d in vals if d is not None]
            dtr.append((f"{np.mean(got):.1f}" if got else "–") + f" ({len(got)})")
        lines.append(f"| {lab} | {_ms([float(x) for x in fp if x is not None])} ({sorted(fp)}) | {' / '.join(dtr)} | {_ms([x['served_J_1_12'] for x in ss])} | "
                     f"{_ms([x['regret_6_40'] for x in ss])} | {_ms([x['regret_17_40'] for x in ss])} | {_ms([x['mean_bwt'] for x in ss])} | {_ms([x['mean_fwt'] for x in ss])} |")
    ss = [s["never_verify"] for s in S["base"]]
    lines.append(f"| never_verify (reference) | – | – | {_ms([x['served_J_1_12'] for x in ss])} | {_ms([x['regret_6_40'] for x in ss])} | {_ms([x['regret_17_40'] for x in ss])} | | |")
    dm = [m["continual"]["regret_6_40"] - b["continual"]["regret_6_40"] for m, b in zip(S["meta"], S["base"])]
    dm2 = [m["continual"]["regret_17_40"] - b["continual"]["regret_17_40"] for m, b in zip(S["meta"], S["base"])]
    lines.append(f"| meta − zero (paired by seed) | | | | {_ms(dm)} | {_ms(dm2)} | | |")
    tB = "\n".join(lines)
    lines = ["| init | " + " | ".join(f"stationary J g{g}" for g in gens) + " |", "|---|" + "---|" * len(gens)]
    for v, lab in (("base", "zero"), ("meta", "meta")):
        lines.append(f"| {lab} | " + " | ".join(_ms([s['continual']['stationary'][g] for s in S[v]]) for g in gens) + " |")
    tB2 = "\n".join(lines)
    mi = [s["_meta"]["meta_init"] for s in S["meta"]]
    return {"A": tA, "A2": tA2, "B": tB, "B2": tB2, "summaries": S, "paired": paired, "meta_runtime": [m["runtime_s"] for m in mi if m],
            "audit": [m["audit"] for m in mi if m], "runtimes": {v: [s["_meta"]["runtime_s"] for s in S[v]] for v in S}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RESULTS.md for a continual MAS orchestrator run.")
    ap.add_argument("--run", default="results/mas_continual")
    ap.add_argument("--mild", default=None)
    ap.add_argument("--prelude", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rao-metainit", default=None, help="root of the {base,rao,meta}/seed* runs: print the RAO / meta-init tables")
    a = ap.parse_args(argv)
    if a.rao_metainit:
        t = rao_metainit_tables(Path(a.rao_metainit), Path(a.run))
        print("\n\n".join(t[k] for k in ("A", "A2", "B", "B2")))
        print({"meta_runtime_s": t["meta_runtime"], "audit": t["audit"], "runtimes": t["runtimes"]})
        return
    run = Path(a.run)
    text = report(run, Path(a.mild) if a.mild else None, Path(a.prelude) if a.prelude else None)
    out = Path(a.out or run / "RESULTS.md")
    out.write_text(text)
    print(f"[analyze] wrote {out}")


if __name__ == "__main__":
    main()
