#!/usr/bin/env python3
"""Regenerate every figure in paper/figures/ from the committed result files.

    .venv-crmarena/bin/python paper/make_figures.py

Inputs (all committed):
  results/crmarena/results.json              CRMArena-Pro CV rows, gate redraw statistics
  results/llm/train_log_phase1{,b,c}.jsonl   GRPO training logs (lambda, batch cost, degeneracy)
  results/ssr/frontier.csv                   SSR policy frontier
  results/mas_continual/{days,gates,meta}.jsonl  and mild/   the continual-orchestrator run
  README.md                                  the simulator gate tables (sim JSON outputs are
                                             gitignored; the values below are copied from the
                                             README's "Results (sim)" section)
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
# Brand-neutral categorical palette (colour-blind safe, ordered by luminance contrast).
C = ["#2F5597", "#C55A11", "#548235", "#7030A0", "#BF9000", "#1F7A8C", "#7F7F7F", "#C00000"]
ARM_LABEL = {
    "continual": "continual (cgrpo, gated)",
    "continual_awr": "continual (AWR+IPS, gated)",
    "frozen": "frozen (days 1-5)",
    "so_static": "static router",
    "so_relearn": "router re-learned per stage",
    "always_verify": "always_verify",
    "never_verify": "never_verify",
}
ARM_COLOR = {a: C[i] for i, a in enumerate(ARM_LABEL)}
ARM_STYLE = {"continual": "-", "continual_awr": "-", "frozen": "--", "so_static": ":",
             "so_relearn": "-.", "always_verify": "--", "never_verify": "-"}


def jsonl(path):
    with open(os.path.join(ROOT, path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------------------
# 1. CRMArena-Pro: quality vs cost per tier, learned policies and the per-task oracle
# --------------------------------------------------------------------------------------
def fig_crm_tiers():
    d = json.load(open(os.path.join(ROOT, "results/crmarena/results.json")))
    rows = d["cv"]["constrained"]["rows"]
    lam2 = d["cv"]["lam2"]["rows"]
    pts = [
        ("fixed cheap", rows["fixed cheap"], "o", C[0]),
        ("fixed mid", rows["fixed mid"], "o", C[1]),
        ("fixed frontier", rows["fixed frontier"], "o", C[2]),
        ("uniform behavior", rows["uniform behavior"], "s", C[6]),
        ("constrained linear (served mix), held-out", rows["constrained cgrpo-lookup linear (served: iterate mix)"], "^", C[3]),
        ("AWR+IPS linear (greedy), held-out", lam2["AWR+IPS linear (greedy)"], "v", C[5]),
        ("per-category rule, held-out", lam2["per-category plug-in rule (train-fold argmax)"], "D", C[4]),
        ("per-task oracle (max q, cheapest)", rows["per-task oracle (max quality, cheapest among ties)"], "*", C[7]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for label, r, m, c in pts:
        ax.scatter(r["c"], r["q"], marker=m, s=90 if m == "*" else 45, color=c, label=label, zorder=3)
    ax.axvline(0.10, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.101, 0.44, "B = $0.10", fontsize=7.5, va="bottom")
    # in-sample point of the constrained policy, for the memorisation gap
    qc = np.array(d["cv"]["constrained"]["in_sample_q_c"])  # per fold (q, c)
    q_in, c_in = qc.mean(axis=0)
    ax.scatter(c_in, q_in, marker="^", s=45, facecolors="none", edgecolors=C[3],
               label="constrained linear, in-sample", zorder=3)
    ax.set_xlabel("cost per task (USD)")
    ax.set_ylabel("quality (held-out, 5-fold mean)")
    ax.set_xlim(0.05, 0.18)
    ax.set_ylim(0.40, 0.80)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("CRMArena-Pro b2b, 50 tasks x 3 tiers")
    save(fig, "fig_crm_tiers")


# --------------------------------------------------------------------------------------
# 2. Gate power: CI width vs the budget margin, and the redraw statistics
# --------------------------------------------------------------------------------------
def fig_gate_power():
    d = json.load(open(os.path.join(ROOT, "results/crmarena/results.json")))
    st = d["gate"]["redraws"]["stats"]
    promote = d["gate"]["redraws"]["promote_rate"]
    tol = d["gate"]["realized_log"]["tol"]
    width50 = st["cost/dr"]["ci_width"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    n = np.linspace(25, 2500, 400)
    a.plot(n, width50 * np.sqrt(50.0 / n), color=C[0], label="DR cost 90% CI width, $1/\\sqrt{n}$ extrapolation\n(back-of-envelope)")
    a.scatter([50], [width50], color=C[0], zorder=3, label=f"measured at n = 50 (${width50:.4f})")
    a.axhline(2 * tol, color=C[1], ls="--", lw=0.9, label=f"2 x budget tolerance (${2*tol:.3f})")
    n_need = 50 * (width50 / (2 * tol)) ** 2
    a.axvline(n_need, color=C[1], ls=":", lw=0.9)
    a.text(n_need * 1.05, width50 * 0.75, f"n ~ {n_need:,.0f}", fontsize=7.5, color=C[1])
    a.set_xscale("log")
    a.set_xlabel("logged episodes n (one arm per task)")
    a.set_ylabel("90% CI width on cost (USD)")
    a.set_title(f"certifying a ${tol:.3f} margin (PROMOTE rate {promote:.0%} at n = 50)", fontsize=8)
    a.legend(frameon=False, loc="upper right", fontsize=6.8)
    ests = ["ips", "snips", "dr"]
    x = np.arange(3)
    w = 0.38
    b.bar(x - w / 2, [st[f"quality/{e}"]["rmse"] for e in ests], w, color=C[0], label="quality RMSE")
    b.bar(x + w / 2, [st[f"quality/{e}"]["ci_width"] for e in ests], w, color=C[0], alpha=0.45, label="quality CI width")
    b2 = b.twinx()
    b2.plot(x, [st[f"quality/{e}"]["coverage"] for e in ests], "k.-", label="quality CI coverage")
    b2.plot(x, [st[f"cost/{e}"]["coverage"] for e in ests], "k.--", label="cost CI coverage")
    b2.set_ylim(0.8, 1.10)
    b2.set_ylabel("coverage of the exact value")
    b2.spines["right"].set_visible(True)
    b.set_xticks(x)
    b.set_xticklabels([e.upper() for e in ests])
    b.set_ylabel("quality error / width")
    b.set_title("200 redraws of the realized log (n = 50)", fontsize=8)
    b.set_ylim(0, 0.55)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=6.8, ncol=2)
    save(fig, "fig_gate_power")


# --------------------------------------------------------------------------------------
# 3. LLM controller: lambda, batch cost and degenerate-group rate per GRPO step
# --------------------------------------------------------------------------------------
def fig_llm_training():
    runs = [("phase1", "results/llm/train_log_phase1.jsonl"),
            ("phase1b", "results/llm/train_log_phase1b.jsonl"),
            ("phase1c (SFT warm start)", "results/llm/train_log_phase1c.jsonl")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharex=False)
    for i, (name, path) in enumerate(runs):
        log = jsonl(path)
        s = [r["step"] for r in log]
        axes[0].plot(s, [r["lambda"] for r in log], color=C[i], lw=1, label=name)
        axes[1].plot(s, [r["mean_cost"] for r in log], color=C[i], lw=0.8, alpha=0.9)
        dg = [r.get("degenerate_group_rate", np.nan) for r in log]
        axes[2].plot(s, dg, color=C[i], lw=0.8, alpha=0.9)
    axes[0].set_title("learned price $\\lambda$")
    axes[1].set_title("batch mean cost (sim units)")
    axes[1].axhline(0.30, color="k", ls="--", lw=0.8)
    axes[1].text(5, 0.31, "B = 0.30", fontsize=7)
    axes[2].set_title("degenerate-group rate")
    axes[2].set_ylim(-0.02, 1.02)
    for a in axes:
        a.set_xlabel("GRPO step")
    axes[0].legend(frameon=False, loc="upper right")
    save(fig, "fig_llm_training")


# --------------------------------------------------------------------------------------
# 4. SSR frontier
# --------------------------------------------------------------------------------------
def fig_ssr_frontier():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results/ssr/frontier.csv"))))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    fam_style = {
        "never": ("s", C[6], "never"),
        "always": ("X", C[7], "always_1 (cheap / mid)"),
        "uncertainty_gated": ("o", C[0], "uncertainty-gated, tau sweep (cheap / mid)"),
        "controller": ("^", C[3], "priced controller (held-out / in-sample)"),
        "tier_arm": ("*", C[1], "buy a tier up front (single agent)"),
        "oracle": ("D", C[4], "per-task oracle over logged answers"),
    }
    seen = set()
    for r in rows:
        fam = r["family"]
        m, c, lab = fam_style[fam]
        q, cst = float(r["quality"]), float(r["cost"])
        filled = not (fam == "controller" and r["split"] == "in-sample")
        if fam == "uncertainty_gated" and r["tier"] == "mid":
            filled = False
        ax.scatter(cst, q, marker=m, s=110 if m == "*" else 40, color=c if filled else "none",
                   edgecolors=c, linewidths=1.0, label=None if fam in seen else lab, zorder=3)
        seen.add(fam)
        if fam == "tier_arm":
            ax.annotate(r["tier"], (cst, q), textcoords="offset points", xytext=(5, -3), fontsize=7)
        if fam == "controller" and r["split"] == "held-out" and "B=" in r["policy"]:
            ax.annotate("B = $0.10, held-out", (cst, q), textcoords="offset points", xytext=(6, -9), fontsize=7)
    ax.axvline(0.10, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("cost per task (USD, incl. the initial cheap episode)")
    ax.set_ylabel("quality (41 tasks)")
    ax.set_title("SSR refinement policies vs buying a better tier")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    save(fig, "fig_ssr_frontier")


# --------------------------------------------------------------------------------------
# 5-8. Continual orchestrator: regret over days, gate history, BWT/FWT, mild sensitivity
# --------------------------------------------------------------------------------------
STAGE_DAYS = {0: 1, 1: 9, 2: 17, 3: 25, 4: 33}
STAGE_LAST = {0: 8, 1: 16, 2: 24, 3: 32, 4: 40}


def stage_of(day):
    return max(g for g, d0 in STAGE_DAYS.items() if day >= d0)


def load_run(sub=""):
    base = os.path.join("results/mas_continual", sub)
    days = jsonl(os.path.join(base, "days.jsonl"))
    gates = jsonl(os.path.join(base, "gates.jsonl"))
    meta = jsonl(os.path.join(base, "meta.jsonl"))[0]
    byarm = defaultdict(dict)
    for r in days:
        byarm[r["arm"]][r["day"]] = r
    return byarm, gates, meta


def served_J(byarm, arm, day):
    return byarm[arm][day]["heldout"][str(stage_of(day))]["J"]


def transfer(byarm, meta, arm):
    d = byarm[arm]
    bwt = [d[40]["heldout"][str(j)]["J"] - d[STAGE_LAST[j]]["heldout"][str(j)]["J"] for j in range(4)]
    fwt = [d[STAGE_DAYS[j]]["heldout"][str(j)]["J"] - meta["uniform"][str(j)]["J"] for j in range(1, 5)]
    return bwt, fwt


def regret_series(byarm, arm):
    days = sorted(byarm[arm])
    reg = []
    cum = 0.0
    for day in days:
        r = byarm[arm][day]
        if day >= 6:
            cum += r["oracle_J"] - served_J(byarm, arm, day)
        reg.append((day, cum))
    return reg


def fig_continual_regret():
    byarm, gates, meta = load_run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for arm in ARM_LABEL:
        days = sorted(byarm[arm])
        a.plot(days, [served_J(byarm, arm, d) for d in days], ARM_STYLE[arm], color=ARM_COLOR[arm], lw=1.1,
               label=ARM_LABEL[arm])
        reg = regret_series(byarm, arm)
        b.plot([d for d, _ in reg], [c for _, c in reg], ARM_STYLE[arm], color=ARM_COLOR[arm], lw=1.1)
    days = sorted(byarm["continual"])
    a.plot(days, [byarm["continual"][d]["oracle_J"] for d in days], color="k", lw=0.9, alpha=0.5, label="per-stage oracle")
    for ax in (a, b):
        for g in (1, 2, 3, 4):
            ax.axvline(STAGE_DAYS[g] - 0.5, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("day")
    for g, name in zip(range(5), ["g0", "g1 price", "g2 agents", "g3 tools", "g4 skills"]):
        a.text(STAGE_DAYS[g] + 0.3, 0.497, name, fontsize=7, alpha=0.8, va="top")
    a.set_ylabel("held-out $J_g$ of the policy in service")
    a.set_ylim(0.22, 0.50)
    a.set_title("held-out priced value per day")
    b.set_ylabel("cumulative regret vs oracle (days 6-40)")
    b.set_title("cumulative regret")
    fig.legend(frameon=False, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
    save(fig, "fig_continual_regret")


def fig_continual_gate():
    byarm, gates, meta = load_run()
    g = [r for r in gates if r["arm"] == "continual"]
    eps_q, tol = meta["config"]["eps_q"], meta["config"]["tol"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    for r in g:
        day = r["day"]
        pt, lo, hi = r["dr_quality"]
        col = C[2] if r["decision"] == "PROMOTE" else C[7]
        a.plot([day, day], [lo, hi], color=col, lw=1.2)
        a.plot(day, pt, "o", color=col, ms=3)
        if not r["truth"]["acceptable"]:
            a.plot(day, hi + 0.03, "v", color="k", ms=3)
        cpt, clo, chi = r["dr_cost"]
        b.plot([day, day], [clo, chi], color=col, lw=1.2)
        b.plot(day, cpt, "o", color=col, ms=3)
    a.plot([r["day"] for r in g], [r["q_prev"] - eps_q for r in g], "k-", lw=0.8, label="bar: incumbent DR quality $-\\,\\epsilon_q$")
    b.plot([r["day"] for r in g], [r["gate_budget"] + tol for r in g], "k-", lw=0.8, label="bar: budget net of prefix + tol")
    for ax in (a, b):
        for s in (1, 2, 3, 4):
            ax.axvline(STAGE_DAYS[s] - 0.5, color="k", lw=0.6, ls=":", alpha=0.6)
    a.set_ylabel("DR quality [90% CI]")
    b.set_ylabel("DR cost [90% CI] (USD)")
    b.set_xlabel("day")
    a.plot([], [], color=C[2], label="PROMOTE")
    a.plot([], [], color=C[7], label="REVERT")
    a.plot([], [], "kv", ms=3, label="candidate truly unacceptable (over budget)")
    a.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
    b.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_title("gate history of the continual arm (40 candidates: 8 PROMOTE, 32 REVERT)")
    save(fig, "fig_continual_gate")



def fig_continual_crossover():
    """Served-J gap (continual - never_verify) per day, and windowed regret."""
    byarm, gates, meta = load_run()
    g = [r for r in gates if r["arm"] == "continual"]
    first_promote = min(r["day"] for r in g if r["decision"] == "PROMOTE")
    days = sorted(byarm["continual"])
    gap = [byarm["continual"][d]["served"]["J"] - byarm["never_verify"][d]["served"]["J"] for d in days]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    a.axhline(0, color="k", lw=0.6)
    a.bar(days, gap, color=[C[2] if v >= 0 else C[7] for v in gap], width=0.8)
    for stage in (1, 2, 3, 4):
        a.axvline(STAGE_DAYS[stage] - 0.5, color="k", lw=0.6, ls=":", alpha=0.6)
    a.axvline(first_promote, color=C[0], lw=1.0, ls="--")
    a.text(first_promote + 0.4, min(gap) * 0.95, f"first PROMOTE (day {first_promote})", fontsize=7, color=C[0])
    for stage, name in zip(range(5), ["g0", "g1 price", "g2 agents", "g3 tools", "g4 skills"]):
        lo, hi = STAGE_DAYS[stage], STAGE_LAST[stage]
        m = np.mean([gap[d - 1] for d in range(lo, hi + 1)])
        a.plot([lo - 0.4, hi + 0.4], [m, m], color="k", lw=1.2)
        a.text(lo + 0.2, max(gap) * 0.97, f"{name}\n{m:+.3f}", fontsize=6.5, va="top")
    a.set_xlabel("day")
    a.set_ylabel("served $J$: continual $-$ never_verify")
    a.set_title("served-J gap per day (bars) and per stage window (lines)", fontsize=8)
    arms = list(ARM_LABEL)
    w = 0.26
    for k, lo in enumerate((6, 17, 25)):
        reg = [sum(byarm[arm][d]["oracle_J"] - served_J(byarm, arm, d) for d in range(lo, 41)) for arm in arms]
        b.bar(np.arange(len(arms)) + (k - 1) * w, reg, w, color=[C[0], C[1], C[2]][k], label=f"days {lo}-40")
    short = ["continual\n(cgrpo)", "continual\n(AWR)", "frozen", "static\nrouter", "router\nre-learned", "always\nverify", "never\nverify"]
    b.set_xticks(range(len(arms)))
    b.set_xticklabels(short, fontsize=6)
    b.set_ylabel("cumulative regret vs oracle")
    b.set_title("regret over the full stream and after the crossover", fontsize=8)
    b.legend(frameon=False, fontsize=7)
    save(fig, "fig_continual_crossover")


# --------------------------------------------------------------------------------------
# 11. Verifier ablation: always_verify - never_verify under four variants, 5 seeds each
# --------------------------------------------------------------------------------------
def fig_ablation_verifier():
    d = jsonl("results/mas_continual/RESULTS_ablation_verifier.summary.jsonl")[0]
    variants = ["A", "B", "C", "D"]
    labels = ["A: verify-1,\ncap 8 (committed)", "B: verify-2\nfor every arm", "C: B + cap 12", "D: C + verifier\nexempt from cap"]
    def stat(key, sub=None):
        out = []
        for v in variants:
            xs = [r[key] if sub is None else r[key][sub] for r in d["runs"] if r["variant"] == v]
            xs = np.array(xs, dtype=float)
            out.append((xs.mean(), xs.std(ddof=1) / np.sqrt(len(xs))))
        return np.array(out)
    gq, gj, gc, cap = stat("gap_quality"), stat("gap_J"), stat("gap_cost"), stat("cap_hits", "always_verify")
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5))
    fig.subplots_adjust(wspace=0.55)
    x = np.arange(4)
    for ax, (m, title, unit) in zip(axes, [(gq, "expected quality", ""), (gj, "priced $J$ (stage $\\bar\\lambda_g$)", ""),
                                            (gc, "cost per task", "USD"), (cap, "cap hits, always_verify", "share")]):
        cols = [C[2] if v >= 0 else C[7] for v in m[:, 0]] if title != "cap hits, always_verify" else [C[6]] * 4
        ax.bar(x, m[:, 0], yerr=m[:, 1], color=cols, capsize=3, width=0.65)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["A", "B", "C", "D"])
        ax.set_title(title, fontsize=8)
        if unit:
            ax.set_ylabel(unit)
    axes[0].set_ylabel("always_verify $-$ never_verify")
    fig.text(0.5, -0.06, "   |   ".join(l.replace("\n", " ") for l in labels), ha="center", fontsize=6.8)
    save(fig, "fig_ablation_verifier")
    for v, a, b, c, h in zip(variants, gq, gj, gc, cap):
        print(f"  {v}: dq {a[0]:+.3f}±{a[1]:.3f}  dJ {b[0]:+.3f}±{b[1]:.3f}  dcost {c[0]:+.4f}±{c[1]:.4f}  cap {h[0]:.2f}")

def fig_continual_bwt():
    byarm, gates, meta = load_run()
    arms = list(ARM_LABEL)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    w = 0.11
    for i, arm in enumerate(arms):
        bwt, fwt = transfer(byarm, meta, arm)
        a.bar(np.arange(4) + (i - 3) * w, bwt, w, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        b.bar(np.arange(4) + (i - 3) * w, fwt, w, color=ARM_COLOR[arm])
    a.set_xticks(range(4))
    a.set_xticklabels(["g0", "g1", "g2", "g3"])
    a.set_ylabel("BWT$_j$ = $R_{T,j} - R_{j,j}$")
    a.set_title("backward transfer (final day vs stage end)")
    a.axhline(0, color="k", lw=0.6)
    b.set_xticks(range(4))
    b.set_xticklabels(["g1 price", "g2 agents", "g3 tools", "g4 skills"])
    b.set_ylabel("FWT$_j$ = $R_{j-1,j} - b_j$")
    b.set_title("forward transfer (event-day policy vs uniform)")
    b.axhline(0, color="k", lw=0.6)
    a.legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "fig_continual_bwt")


def fig_continual_mild():
    runs = {"default schedule": load_run(), "mild schedule": load_run("mild")}
    arms = list(ARM_LABEL)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    w = 0.38
    for k, (name, (byarm, gates, meta)) in enumerate(runs.items()):
        reg = [regret_series(byarm, a)[-1][1] for a in arms]
        ax.bar(np.arange(len(arms)) + (k - 0.5) * w, reg, w, color=C[0] if k == 0 else C[1], label=name)
    ax.set_xticks(range(len(arms)))
    short = {"continual": "continual\n(cgrpo)", "continual_awr": "continual\n(AWR)", "frozen": "frozen",
             "so_static": "static\nrouter", "so_relearn": "router\nre-learned", "always_verify": "always\nverify",
             "never_verify": "never\nverify"}
    ax.set_xticklabels([short[a] for a in arms], fontsize=7)
    ax.set_ylabel("cumulative regret vs oracle, days 6-40")
    ax.legend(frameon=False)
    ax.set_title("schedule sensitivity")
    save(fig, "fig_continual_mild")


# --------------------------------------------------------------------------------------
# 9. Simulator gate: estimator forest plot (values from README.md, "Results (sim)")
# --------------------------------------------------------------------------------------
SIM_GATE = {
    # README.md > The simulator > Results (sim, from scripts/run_v0.sh). (pt, lo, hi)
    "v1": {
        "behavior (log mean)": ((0.754, 0.743, 0.765), (0.293, 0.286, 0.301)),
        "candidate IPS": ((0.742, 0.638, 0.866), (0.267, 0.228, 0.309)),
        "candidate SNIPS": ((0.761, 0.682, 0.832), (0.272, 0.246, 0.297)),
        "candidate DR": ((0.851, 0.815, 0.887), (0.306, 0.302, 0.309)),
        "ground truth (20k tasks)": ((0.880, None, None), (0.303, None, None)),
    },
    "v2": {
        "incumbent frozen (DR)": ((0.860, 0.852, 0.868), (0.408, 0.404, 0.411)),
        "candidate IPS": ((0.869, 0.826, 0.908), (0.312, 0.303, 0.322)),
        "candidate SNIPS": ((0.872, 0.851, 0.891), (0.313, 0.309, 0.317)),
        "candidate DR": ((0.873, 0.854, 0.889), (0.310, 0.309, 0.311)),
        "ground truth (20k tasks)": ((0.871, None, None), (0.312, None, None)),
    },
}


def fig_sim_gate():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for ax, (ver, rows) in zip(axes, SIM_GATE.items()):
        names = list(rows)
        for i, nm in enumerate(names):
            (q, ql, qh), (c, cl, ch) = rows[nm]
            col = C[7] if "truth" in nm else (C[0] if "DR" in nm else C[6])
            if ql is not None:
                ax.plot([ql, qh], [i, i], color=col, lw=1.4)
            ax.plot(q, i, "o" if ql is not None else "*", color=col, ms=5 if ql is not None else 8)
            ax.text(0.585, i + 0.26, f"cost {c:.3f}" + (f" [{cl:.3f}, {ch:.3f}]" if cl is not None else ""),
                    fontsize=6.5, color="0.3")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlim(0.58, 0.95)
        ax.set_xlabel("quality [90% CI]")
        ax.set_title(f"gate {ver}: " + ("candidate vs uniform behavior (3,000 eps.)" if ver == "v1" else "adapted vs frozen (6,000 eps., drift)"))
    save(fig, "fig_sim_gate")


# --------------------------------------------------------------------------------------
# 10. Appendix table: the continual arm day by day, from days.jsonl / gates.jsonl
# --------------------------------------------------------------------------------------
def tab_continual_days():
    byarm, gates, meta = load_run()
    g = {r["day"]: r for r in gates if r["arm"] == "continual"}
    lines = []
    for day in sorted(byarm["continual"]):
        r = byarm["continual"][day]
        gt = g[day]
        s = r["served"]
        ess = f"{gt['per_decision_ess']:.0f}/{gt['per_decision_n']}"
        truth = "ok" if gt["truth"]["acceptable"] else "BAD"
        ev = "$^*$" if r["event"] else ""
        lines.append(
            f"{day}{ev} & {r['generation']} & {s['expected_quality']:.3f} & {s['cost']:.4f} & {s['J']:.3f} & "
            f"{r['lambda']:.2f} & {gt['decision']} & {gt['dr_quality'][0]:.3f} [{gt['dr_quality'][1]:.3f}, {gt['dr_quality'][2]:.3f}] & "
            f"{gt['dr_cost'][2]:.4f} & {ess} & {truth} \\\\")
    head = ("\\begin{tabular}{llcccclccc c}\n\\toprule\n"
            "day & stage & served $q$ & cost & $J$ & $\\bar\\lambda$ & gate & DR quality [90\\% CI] & "
            "DR cost hi & ESS & truth \\\\\n\\midrule\n")
    with open(os.path.join(OUT, "tab_continual_days.tex"), "w") as f:
        f.write(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("wrote tab_continual_days.tex")


if __name__ == "__main__":
    tab_continual_days()
    fig_crm_tiers()
    fig_gate_power()
    fig_llm_training()
    fig_ssr_frontier()
    fig_continual_regret()
    fig_continual_gate()
    fig_continual_crossover()
    fig_ablation_verifier()
    fig_continual_bwt()
    fig_continual_mild()
    fig_sim_gate()
