"""Logit-saturation report for every policy file in the repo (Controller.saturation_stats).

The served policy is z = L tanh(Wx / L); dz/d(Wx) = sech^2(Wx / L) -> 0 as |Wx| / L grows, so bounded
probabilities do not imply steerable weights. For each policy this script evaluates |Wx| / L over the
legal (action, input) pairs at the states of the log the policy was trained / gated on:

  sim (examples/, policies/, results/sim_seeds/)   logged states of the v1 uniform log (policy_v1) or of
                                                   the v2 frozen-policy log (policy_v2); the sim_seeds
                                                   traces are not committed and are regenerated with the
                                                   pipeline's own seeds into --scratch
  CRMArena (results/crmarena*/policies)            step-0 features of the counterfactual log (the task ids
                                                   of the run's folds)
  MAS (results/mas[/dryrun]/policy_awr_orm.json)   every logged decision of the run's log
  SSR (results/ssr/policies)                       every decision state of the logged refinement trees
  MAS continual (results/mas_continual/**)         logs are not saved: states of a uniform-behaviour pass
                                                   (300 tasks) on the harness generation in service on
                                                   the policy's day, built from the run's own config

Two weight sets per policy: "served" pools the iterate-mix snapshots (what is served and gated), "W" is
the final iterate (what a warm start or the next continual update differentiates through).

  python scripts/saturation_report.py [--scratch DIR] [--out results/saturation_report.md]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.collect import collect, load_policy  # noqa: E402
from pace.controller import Controller  # noqa: E402
from pace.harness.sim import make_sim, parse_reprice  # noqa: E402
from pace.trace import read_jsonl, write_jsonl  # noqa: E402

THR = Controller.SAT_THRESHOLD


def tracked_files() -> set[str]:
    try:
        return set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split())
    except Exception:
        return set()


def is_policy(p: Path) -> bool:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return isinstance(d, dict) and "W" in d and "actions" in d


def stats(pol: Controller, rows: list[tuple[list[float], list[str]]]) -> dict:
    pol = pol.copy()
    pol.ensure_actions(sorted({a for _, legal in rows for a in legal}))
    X = np.array([x for x, _ in rows], float)
    M = np.array([pol.mask(legal) for _, legal in rows])
    out = {"served": pol.saturation_stats(X, M, use_mix=True), "W": pol.saturation_stats(X, M, use_mix=False)}
    P = pol.probs_batch(X, M, eps=0.0)
    out["min_legal_prob_served"] = float(P[M].min())
    return out


# ------------------------------------------------------------------ state sources
def ep_rows(episodes, step0_only=False):
    return [(s.features, s.legal or [s.action_name]) for e in episodes
            for s in (e.steps[:1] if step0_only else e.steps)]


_cache: dict = {}


def sim_v1_log(scratch: Path, seed: int) -> list:
    key = ("v1", seed)
    if key not in _cache:
        p = scratch / f"sim_v1_seed{seed}.jsonl"
        if not p.exists():
            h = make_sim("v1", seed=seed)
            write_jsonl(collect(h, load_policy("uniform", h), 3000, seed), p)
        _cache[key] = ep_rows(read_jsonl(p))
    return _cache[key]


def sim_v2_log(scratch: Path, seed: int, frozen: Path, tag: str) -> list:
    key = ("v2", seed, tag)
    if key not in _cache:
        p = scratch / f"sim_v2_{tag}_seed{seed}.jsonl"
        if not p.exists():
            h = make_sim("v2", seed=seed, reprice=parse_reprice(["mid:1.2"]))
            write_jsonl(collect(h, load_policy(str(frozen), h, 0.3), 6000, seed), p)
        _cache[key] = ep_rows(read_jsonl(p))
    return _cache[key]


def crm_rows(log: Path, ids: set[str] | None) -> list:
    key = ("crm", str(log), None if ids is None else tuple(sorted(ids)))
    if key not in _cache:
        from pace.lookup import DECISION_ACTIONS, LookupTable
        T = LookupTable.from_episodes(read_jsonl(log))
        keep = [i for i, t in enumerate(T.task_ids) if ids is None or t in ids]
        _cache[key] = [(list(T.X[i]), list(DECISION_ACTIONS)) for i in keep]
    return _cache[key]


def crm_fold_ids(results_json: Path) -> set[str] | None:
    try:
        r = json.loads(results_json.read_text())
        return {t for f in r["folds"] for t in f["train"] + f["test"]}
    except Exception:
        return None


def ssr_rows(log: Path, pol: Controller) -> list:
    from pace.experiments.ssr.policy import _leaf_paths
    from pace.experiments.ssr.tree import build_trees, read_records
    tiers = pol.meta.get("tiers") or ["cheap", "mid"]
    mr = int(pol.meta.get("max_rounds", 1))
    trees = build_trees(read_records(log))
    trees = [t for t in trees if t.complete(tiers, mr)] or trees
    rows, seen = [], set()
    for t in trees:
        for rws, _, _ in _leaf_paths(t, tiers, mr):
            for x, legal, _ in rws:
                k = (tuple(x), tuple(legal))
                if k not in seen:
                    seen.add(k); rows.append((list(x), list(legal)))
    return rows


def mas_cont_rows(run_dir: Path, gen_index: int, n_tasks: int = 300) -> list:
    from dataclasses import replace  # noqa: F401
    from pace.experiments.mas_continual import drift, simenv
    from pace.experiments.mas_continual.simenv import ACTIONS, MASSimHarness
    cfg = json.loads((run_dir / "meta.jsonl").read_text().splitlines()[0])["config"]
    ck = (cfg.get("schedule", "default"), cfg.get("verifier") or "verify-1", cfg.get("max_calls") or 8,
          bool(cfg.get("verifier_exempt") or False))
    key = ("mascont", ck, gen_index)
    if key not in _cache:
        drift.apply_bonus_scale(ck[0])
        gens = drift.schedule(ck[0], verifier2_from_day1=ck[1] == "verify-2", max_calls=ck[2],
                              verifier_exempt_from_cap=ck[3])
        g = gens[gen_index]
        h = MASSimHarness(g.params, seed=9000 + gen_index, version=g.version)
        uni = Controller(ACTIONS, simenv.N_FEATURES, eps=0.0)
        uni.ensure_actions(h.all_actions)
        eps_ = collect(h, uni, n_tasks, 9100 + gen_index, log_counterfactual=False, log_state=False)
        _cache[key] = ep_rows(eps_)
    return _cache[key]


def mas_cont_gen(run_dir: Path, name: str) -> int:
    from pace.experiments.mas_continual import drift
    m = re.match(r"oracle_g(\d)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"day(\d+)", name)
    if not m:
        return 0                                              # frozen, meta_init: trained on g0
    cfg = json.loads((run_dir / "meta.jsonl").read_text().splitlines()[0])["config"]
    gens = drift.schedule(cfg.get("schedule", "default"))
    return drift.generation_for(int(m.group(1)), gens).index


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", default="traces/saturation_scratch", help="where regenerated sim logs go")
    ap.add_argument("--out", default="results/saturation_report.md")
    ap.add_argument("--json-out", default="results/saturation_report.json")
    ap.add_argument("--from-json", action="store_true", help="only re-render the md from --json-out")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.from_json:
        write_md(json.loads(Path(a.json_out).read_text()), Path(a.out), None)
        return
    scratch = Path(a.scratch); scratch.mkdir(parents=True, exist_ok=True)
    tracked = tracked_files()
    res = []

    def add(path: Path, group: str, source: str, rows):
        pol = Controller.load(path)
        st = stats(pol, rows)
        rel = str(path.relative_to(ROOT))
        res.append({"policy": rel, "tracked": rel in tracked, "group": group, "source": source,
                    "n_states": len(rows), "L": pol.logit_clip, **st})

    # sim
    v1_run_v0 = ROOT / "traces/v1_behavior.jsonl"
    for p in [ROOT / "examples/policy_v1_sim.json", ROOT / "policies/policy_v1.json"]:
        if p.exists():
            add(p, "sim", "v1 uniform log, seed 1 (run_v0.sh)", sim_v1_log(scratch, 1))
    if (ROOT / "policies/policy_v2.json").exists():
        add(ROOT / "policies/policy_v2.json", "sim", "v2 frozen log, seed 3 (run_v0.sh)",
            sim_v2_log(scratch, 3, ROOT / "policies/policy_v1.json", "run_v0"))
    for s in range(5):
        d = ROOT / f"results/sim_seeds/seed{s}"
        if (d / "policy_v1.json").exists():
            add(d / "policy_v1.json", "sim", f"v1 uniform log, seed {10 * s + 1}", sim_v1_log(scratch, 10 * s + 1))
        if (d / "policy_v2.json").exists():
            add(d / "policy_v2.json", "sim", f"v2 frozen log, seed {10 * s + 3}",
                sim_v2_log(scratch, 10 * s + 3, d / "policy_v1.json", f"s{s}"))
    _ = v1_run_v0
    # CRMArena
    for d in sorted(ROOT.glob("results/crmarena*/")):
        log = ROOT / ("traces/crmarena_b2c_counterfactual.jsonl" if "b2c" in d.name else "traces/crmarena_b2b_counterfactual.jsonl")
        ids = crm_fold_ids(d / "results.json")
        for p in sorted((d / "policies").glob("*.json")):
            if is_policy(p):
                add(p, "crmarena", f"step-0 features, {log.name} ({len(ids) if ids else 'all'} tasks)", crm_rows(log, ids))
    # MAS
    for p, log in [(ROOT / "results/mas/policy_awr_orm.json", ROOT / "results/mas/mas_b2b.jsonl"),
                   (ROOT / "results/mas/dryrun/policy_awr_orm.json", ROOT / "results/mas/dryrun/mas_fake_b2b.jsonl")]:
        if p.exists() and log.exists():
            eps_ = [e for e in read_jsonl(log) if (e.meta or {}).get("mode", "").startswith("controller")
                    and any(s.propensity < 1.0 for s in e.steps)]
            add(p, "mas", f"all decisions of the {len(eps_)} exploration episodes, {log.name}", ep_rows(eps_))
    # SSR
    for p, log in [(ROOT / "results/ssr/policies/ssr_controller_B0.10_real.json", ROOT / "traces/ssr/ssr_b2b_tree.jsonl"),
                   (ROOT / "results/ssr/policies/ssr_controller_B0.10_dryrun.json", ROOT / "results/ssr/dryrun/ssr_tree.jsonl")]:
        if p.exists() and log.exists():
            add(p, "ssr", f"decision states of {log.name}", ssr_rows(log, Controller.load(p)))
    # MAS continual
    for meta in sorted(ROOT.glob("results/mas_continual/**/meta.jsonl")):
        run_dir = meta.parent
        for p in sorted((run_dir / "policies").glob("*.json")):
            if is_policy(p):
                g = mas_cont_gen(run_dir, p.stem)
                add(p, "mas_continual", f"uniform pass, generation g{g}", mas_cont_rows(run_dir, g))

    Path(a.json_out).write_text(json.dumps(res, indent=1))
    write_md(res, Path(a.out), time.time() - t0)
    print(f"{len(res)} policies in {time.time() - t0:.0f}s -> {a.out}")


def write_md(res: list[dict], out: Path, wall: float) -> None:
    f3 = lambda v: f"{v:.3f}"
    sat = lambda r, k: r[k]["frac_saturated"] > 0
    L = ["# Logit saturation of the trained policies", "",
         f"Generated by `scripts/saturation_report.py` ({len(res)} policy files"
         + (f", {wall:.0f}s" if wall is not None else "") + "). Policy: z = L·tanh(Wx/L), "
         "L = logit_clip. The chain-rule factor of every score-function gradient is sech²(Wx/L): 0.42 at |Wx|/L = 1, "
         f"0.027 at {THR}, 1.8e-4 at 5, and exactly 0 in float64 beyond about 19. "
         "Bounded probabilities therefore do not imply steerable weights (tests/test_saturation.py: W = [[300], [-300]], "
         "x = [1], L = 3 gives probabilities [0.9975, 0.0025] and an identically zero gradient for both actions).", "",
         f"For each policy: u = |Wx|/L over the legal (action, state) pairs of the states listed. 'served' pools the "
         f"iterate-mix snapshots; 'W' is the final iterate. Saturated = u > {THR}. 'min p' is the smallest "
         "served (ε-free) probability of a legal action at those states.", ""]
    # headline
    tracked = [r for r in res if r["tracked"]]
    any_t = [r for r in tracked if sat(r, "served") or sat(r, "W")]
    worst = max(res, key=lambda r: max(r["served"]["max_abs_u"], r["W"]["max_abs_u"]))
    n_sat = sum(1 for r in res if sat(r, "served") or sat(r, "W"))
    L += ["## Answer", ""]
    L.append(f"- **Committed (git-tracked) policies: {len(tracked)}.** "
             + ("None enters the saturated regime: " if not any_t else f"{len(any_t)} enter{'s' if len(any_t) == 1 else ''} the saturated regime (u > {THR}): ")
             + "max u over them = "
             + f"{max(max(r['served']['max_abs_u'], r['W']['max_abs_u']) for r in tracked):.3f}"
             + (f" ({', '.join(r['policy'] for r in any_t)})." if any_t else "."))
    L.append(f"- **All policy files present on disk (tracked or not): {len(res)}**, of which {n_sat} have at least one "
             f"saturated legal pair. Largest u anywhere: {max(worst['served']['max_abs_u'], worst['W']['max_abs_u']):.3f} "
             f"(`{worst['policy']}`).")
    L.append("")
    # per-policy table, non-continual
    L += ["## Committed and pipeline policies", "",
          "| policy | tracked | states | n | L | served max u | served frac u>2.5 | served mean sech² | W max u | W frac u>2.5 | W mean sech² | min p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        if r["group"] == "mas_continual":
            continue
        s, w = r["served"], r["W"]
        L.append(f"| `{r['policy']}` | {'yes' if r['tracked'] else 'no'} | {r['source']} | {r['n_states']} | {r['L']:g} | "
                 f"{f3(s['max_abs_u'])} | {f3(s['frac_saturated'])} | {f3(s['mean_sech2'])} | {f3(w['max_abs_u'])} | "
                 f"{f3(w['frac_saturated'])} | {f3(w['mean_sech2'])} | {r['min_legal_prob_served']:.4f} |")
    # continual: per run dir
    cont = [r for r in res if r["group"] == "mas_continual"]
    if cont:
        by = defaultdict(list)
        for r in cont:
            by[str(Path(r["policy"]).parent.parent)].append(r)
        L += ["", "## MAS continual runs (not committed; logs not saved, states from a uniform pass on the generation in service)", "",
              "| run directory | policies | max u (served / W) | policies with any u>2.5 (served / W) | mean frac u>2.5 (W) | min mean sech² (W) | worst policy (W) |",
              "|---|---|---|---|---|---|---|"]
        for k in sorted(by):
            rs = by[k]
            wst = max(rs, key=lambda r: r["W"]["max_abs_u"])
            L.append(f"| `{k}` | {len(rs)} | {max(r['served']['max_abs_u'] for r in rs):.3f} / {max(r['W']['max_abs_u'] for r in rs):.3f} | "
                     f"{sum(sat(r, 'served') for r in rs)} / {sum(sat(r, 'W') for r in rs)} | "
                     f"{np.mean([r['W']['frac_saturated'] for r in rs]):.4f} | {min(r['W']['mean_sech2'] for r in rs):.3f} | "
                     f"`{Path(wst['policy']).name}` |")
        kind = lambda r: ("AWR arm (continual_awr_day*)" if "continual_awr" in r["policy"] else
                          "CGRPO arm (continual_day*)" if "continual_day" in r["policy"] else
                          "oracle (oracle_g*)" if "oracle_g" in r["policy"] else "frozen / meta_init")
        byk = defaultdict(list)
        for r in cont:
            byk[kind(r)].append(r)
        L += ["", "By arm, over all continual run directories (final iterate W):", "",
              "| arm | policies | max u | with any u>2.5 | with any sech²<1e-3 | mean frac u>2.5 (low / high side) | max frac sech²<1e-3 |",
              "|---|---|---|---|---|---|---|"]
        for k in sorted(byk):
            W = [r["W"] for r in byk[k]]
            L.append(f"| {k} | {len(W)} | {max(w['max_abs_u'] for w in W):.2f} | {sum(w['frac_saturated'] > 0 for w in W)} | "
                     f"{sum(w['frac_sech2_below_1e-3'] > 0 for w in W)} | {np.mean([w['frac_saturated'] for w in W]):.4f} "
                     f"({np.mean([w['frac_saturated_low'] for w in W]):.4f} / {np.mean([w['frac_saturated_high'] for w in W]):.4f}) | "
                     f"{max(w['frac_sech2_below_1e-3'] for w in W):.3f} |")
    L += ["", "## Reading", "",
          "- **u > 2.5 is attenuation, not death.** sech² = 0.027 at u = 2.5, 0.0067 at u = 3.19 (the largest committed "
          "value), and < 1e-3 only beyond u ≈ 4.1. Every committed policy stays below u = 3.2 and has no legal pair with "
          "sech² < 1e-3. The CRMArena and SSR policies, and the sim v1 policies, never reach u = 2.5.",
          "- **Committed `results/mas/policy_awr_orm.json`** (AWR, 300 full-batch epochs at lr 2, fixed λ = 2) has 2.6% of "
          "its legal (action, decision) pairs above 2.5, all on the low side: actions pushed toward the floor, where the "
          "gradient through the clip is damped 37× to 150×. It is a one-shot policy that is never warm-started, so this "
          "affects only how far further training could move it.",
          "- **The sim policy_v2 files** (warm-started from policy_v1 and retrained on v2) have about 2% of pairs above 2.5, "
          "all on the high side (a preferred action's logit pinned near +L). There, the attenuation slows moving mass "
          "away from the incumbent's favourite action, which is exactly what a drift update must do.",
          "- **The continual AWR arm is the one place where the weights are effectively frozen.** It warm-starts every "
          "day from the previous day's policy (`train_awr_day(..., init=prev)`), so |Wx| accumulates: by day 40, u reaches "
          "about 13, close to half of the legal pairs are above 2.5, and up to 61% have sech² < 1e-3. Almost all of these "
          "are on the low side, so those actions cannot be brought back by any later reward signal through this "
          "parameterisation. The continual CGRPO arm and the oracles stay at u ≤ 4.2, with sech² < 1e-3 in at most 0.7% of "
          "pairs (one policy). Comparisons of AWR against CGRPO in the continual runs therefore conflate the update rule "
          "with the AWR arm's loss of plasticity in the soft clip.",
          "- MAS-continual states are a uniform-behaviour pass, not the logged days (those logs are not saved). A "
          "uniform pass covers more of the state space than the served policy does, so the fractions here are "
          "indicative of, not equal to, the fractions at the states the arms actually visited.",
          "", "Raw per-policy statistics (both weight sets, sign split, min sech², fraction with sech² < 1e-3): "
          "`results/saturation_report.json`."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
