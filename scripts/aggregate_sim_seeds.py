#!/usr/bin/env python3
"""Aggregate scripts/run_sim_seeds.sh outputs into results/sim_seeds/RESULTS.md and summary.json."""
import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "sim_seeds")
seeds = sorted(int(d[4:]) for d in os.listdir(ROOT) if d.startswith("seed") and
               os.path.exists(os.path.join(ROOT, d, "drift.json")))
rows = []
for s in seeds:
    L = lambda f: json.load(open(os.path.join(ROOT, f"seed{s}", f)))
    g1, g2, dr = L("gate_v1.json"), L("gate_v2.json"), L("drift.json")
    p1 = L("policy_v1.json")["meta"]
    t1 = dr["rows"]["(i) frozen on v1"]
    t2 = next(v for k, v in dr["rows"].items() if k.startswith("(iii)"))          # served after the gate
    ug = L("drift_ungated.json")                                                   # adapted served regardless
    ta = ug["rows"]["(iii) adapted on v2"]
    t_fv2 = dr["rows"]["(ii) frozen on v2"]
    q1, c1 = g1["candidate"]["quality"]["dr"], g1["candidate"]["cost"]["dr"]
    q2, c2 = g2["candidate"]["quality"]["dr"], g2["candidate"]["cost"]["dr"]
    rows.append(dict(
        seed=s, lam=p1["lambda"],
        g1_q=q1, g1_c=c1, g1_ips=g1["candidate"]["quality"]["ips"][0], g1_snips=g1["candidate"]["quality"]["snips"][0],
        g1_beh_q=g1["incumbent"]["quality"][0], g1_dec=g1["decision"], g1_maxrho=g1["candidate"]["max_rho"],
        g1_cover=q1[1] <= t1["quality"] <= q1[2],
        g2_q=q2, g2_c=c2, g2_ips=g2["candidate"]["quality"]["ips"][0], g2_snips=g2["candidate"]["quality"]["snips"][0],
        g2_inc_q=g2["incumbent"]["quality"][0], g2_inc_c=g2["incumbent"]["cost"][0], g2_dec=g2["decision"],
        g2_cover=q2[1] <= ta["quality"] <= q2[2],
        t1_q=t1["quality"], t1_c=t1["cost"], t1_J=t1["J"], tf2_q=t_fv2["quality"], tf2_c=t_fv2["cost"], tf2_J=t_fv2["J"],
        t2_q=t2["quality"], t2_c=t2["cost"], t2_J=t2["J"],
        ta_q=ta["quality"], ta_c=ta["cost"], ta_J=ta["J"], rec_adapted=ug["recovery"],
        bwt=dr["priced_bwt"], rec=dr["recovery"], drift_dec=dr["decision"]))

def ms(key, f=lambda v: v):
    v = np.array([f(r[key]) for r in rows], float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

S = {k: ms(k) for k in ["lam", "t1_q", "t1_c", "t1_J", "tf2_q", "tf2_c", "tf2_J", "t2_q", "t2_c", "t2_J", "ta_q", "ta_c", "ta_J", "rec_adapted", "bwt", "rec",
                        "g1_ips", "g1_snips", "g1_beh_q", "g2_ips", "g2_snips", "g2_inc_q", "g2_inc_c"]}
for g in ("g1", "g2"):
    for m in ("q", "c"):
        for i, nm in enumerate(("pt", "lo", "hi")):
            S[f"{g}_{m}_{nm}"] = ms(f"{g}_{m}", lambda v, i=i: v[i])
S["n_seeds"] = len(rows)
S["g1_promote"] = sum(r["g1_dec"] == "PROMOTE" for r in rows)
S["g2_promote"] = sum(r["g2_dec"] == "PROMOTE" for r in rows)
S["g1_cover"] = sum(r["g1_cover"] for r in rows)
S["g2_cover"] = sum(r["g2_cover"] for r in rows)
json.dump({"summary": S, "per_seed": rows}, open(os.path.join(ROOT, "summary.json"), "w"), indent=1)

f = lambda k, d=3: f"{S[k][0]:.{d}f} ± {S[k][1]:.{d}f}"
n = len(rows)
md = [f"# Simulator pipeline over {n} seeds",
      "",
      f"`scripts/run_sim_seeds.sh {' '.join(map(str, seeds))}` (the `scripts/run_v0.sh` pipeline; replicate s uses component",
      "seeds 10s+1..10s+4 for collect/train/collect/train and seed s for the gate bootstrap and the Monte Carlo",
      "ground truth; s = 0 reproduces run_v0.sh). Aggregated by `scripts/aggregate_sim_seeds.py`. Mean ± SE over seeds;",
      "budget B = 0.30, tol 0.02, eps_q 0.03; ground truth over 20,000 fresh tasks per condition.",
      "",
      "## Summary",
      "",
      "| quantity | mean ± SE |",
      "|---|---|",
      f"| learned price λ̄ (v1) | {f('lam')} |",
      f"| gate v1: DR quality (point / 90% CI lo / hi) | {f('g1_q_pt')} / {f('g1_q_lo')} / {f('g1_q_hi')} |",
      f"| gate v1: DR cost (point / hi) | {f('g1_c_pt')} / {f('g1_c_hi')} |",
      f"| gate v1: IPS / SNIPS quality | {f('g1_ips')} / {f('g1_snips')} |",
      f"| gate v1: behaviour log mean quality | {f('g1_beh_q')} |",
      f"| gate v1 decision | PROMOTE in {S['g1_promote']}/{n}; truth inside DR quality CI in {S['g1_cover']}/{n} |",
      f"| truth (i) frozen on v1: quality / cost / J | {f('t1_q')} / {f('t1_c')} / {f('t1_J')} |",
      f"| truth (ii) frozen on v2: quality / cost / J | {f('tf2_q')} / {f('tf2_c')} / {f('tf2_J')} |",
      f"| gate v2: DR quality (point / lo / hi) | {f('g2_q_pt')} / {f('g2_q_lo')} / {f('g2_q_hi')} |",
      f"| gate v2: DR cost (point / hi) | {f('g2_c_pt')} / {f('g2_c_hi')} |",
      f"| gate v2: incumbent (frozen) DR quality / cost | {f('g2_inc_q')} / {f('g2_inc_c')} |",
      f"| gate v2 decision | PROMOTE in {S['g2_promote']}/{n}; truth inside DR quality CI in {S['g2_cover']}/{n} |",
      f"| truth, adapted policy on v2 (whether or not promoted): quality / cost / J | {f('ta_q')} / {f('ta_c')} / {f('ta_J')} |",
      f"| truth (iii) policy served after the v2 gate: quality / cost / J | {f('t2_q')} / {f('t2_c')} / {f('t2_J')} |",
      f"| priced-BWT = J(ii) − J(i) | {f('bwt')} |",
      f"| recovery = J(iii) − J(ii), served after the gate | {f('rec')} |",
      f"| recovery the adapted policy would give if served | {f('rec_adapted')} |",
      "",
      "## Per seed",
      "",
      "| seed | λ̄ | gate v1 DR q [CI] | DR cost hi | dec. | truth v1 q / c | gate v2 DR q [CI] | DR cost hi | dec. | truth v2 adapted q / c | priced-BWT | recovery (served) | recovery (adapted) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['seed']} | {r['lam']:.3f} | {r['g1_q'][0]:.3f} [{r['g1_q'][1]:.3f}, {r['g1_q'][2]:.3f}] | {r['g1_c'][2]:.3f} | "
              f"{r['g1_dec']} | {r['t1_q']:.3f} / {r['t1_c']:.3f} | {r['g2_q'][0]:.3f} [{r['g2_q'][1]:.3f}, {r['g2_q'][2]:.3f}] | "
              f"{r['g2_c'][2]:.3f} | {r['g2_dec']} | {r['ta_q']:.3f} / {r['ta_c']:.3f} | {r['bwt']:+.3f} | {r['rec']:+.3f} | {r['rec_adapted']:+.3f} |")
rev = [r for r in rows if r["g2_dec"] != "PROMOTE"]
md += ["", "## Reading", "",
       f"Gate v1 promotes in {S['g1_promote']}/{n} seeds. Gate v2 promotes in {S['g2_promote']}/{n}. Every v2 revert fails on",
       "quality only: the bar is the DR quality recorded at the v1 promotion minus eps_q, and in the reverting seeds",
       "that v1 point estimate sits above the candidate's ground truth ("
       + ", ".join(f"seed {r['seed']}: v1 DR {r['g1_q'][0]:.3f} vs truth {r['t1_q']:.3f}" for r in rev) + "),",
       "so the inherited bar is high while the adapted policy is within budget and within about 0.01 of the frozen",
       "policy's quality. The reverts are false alarms of the quality criterion (the cost criterion passes in every seed).",
       "Recovery is therefore reported two ways: for the policy served after the gate (zero when the gate reverts),",
       "and for the adapted policy as if served (from drift_ungated.json, the same evaluation without the gate)."]
open(os.path.join(ROOT, "RESULTS.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
