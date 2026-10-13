"""Restart-state ablation for train_cgrpo on the simulator pipeline (scripts/run_v0.sh, v1 stage).

Per replicate seed s (component seeds as in scripts/run_sim_seeds.sh):
  1. collect 3000 uniform-behavior v1 episodes, seed 10s+1 (identical to the run_sim_seeds log)
  2. train cgrpo, budget 0.30, seed 10s+2 with
       A  restart_mix = 0.5, iters 300                 (the committed default; reproduces policy_v1)
       B  restart_mix = 0.0, max_transitions = T_A     (episode starts only, MATCHED simulator budget)
       C  restart_mix = 0.0, iters 300                 (matched iterations, NOT matched transitions)
  3. ground truth on v1: pace.collect.true_eval over 20,000 fresh tasks (seed 4242, expected quality,
     as pace.drift's row (i)), J = q - lam_ref * c with lam_ref = arm A's learned lambda_bar
  4. gate v1 vs behavior on the step-1 log (pace.gate.gate, budget 0.30, bootstrap seed s)

Usage: python scripts/restart_ablation.py --out results/sim_seeds/restart_ablation.json
                                          [--traces-dir <scratch>] [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.collect import collect, load_policy, true_eval  # noqa: E402
from pace.controller import Controller  # noqa: E402
from pace.gate import gate  # noqa: E402
from pace.harness.sim import make_sim  # noqa: E402
from pace.trace import read_jsonl, write_jsonl  # noqa: E402
from pace.train_offline import train_cgrpo  # noqa: E402

BUDGET = 0.30
N_EVAL = 20000
EVAL_SEED = 4242


def run_seed(args) -> dict:
    s, traces_dir, pol_dir = args
    t0 = time.time()
    b = 10 * s
    tr = Path(traces_dir) / f"seed{s}" / "v1_behavior.jsonl"
    if not tr.exists():
        h = make_sim("v1", seed=b + 1)
        write_jsonl(collect(h, load_policy("uniform", h), 3000, b + 1), tr)
    eps_ = read_jsonl(tr)
    out = {"seed": s, "n_log": len(eps_)}
    arms = {}

    def train(name, **kw):
        h = make_sim("v1", seed=b + 2 + 101)                # as pace.train_offline.main: seed + 101
        t = time.time()
        pol, lam = train_cgrpo(eps_, h, BUDGET, seed=b + 2, **kw)
        pol.save(Path(pol_dir) / f"seed{s}" / f"policy_v1_{name}.json")
        arms[name] = {"policy": pol, "lam_bar": lam, "transitions": pol.meta["transitions"],
                      "iters": pol.meta["iters"], "restart_mix": kw.get("restart_mix", 0.5),
                      "train_s": round(time.time() - t, 1)}

    train("A_mix0.5", restart_mix=0.5, iters=300)
    T_A = arms["A_mix0.5"]["transitions"]
    train("B_mix0.0_matched", restart_mix=0.0, iters=10_000, max_transitions=T_A)
    train("C_mix0.0_iters300", restart_mix=0.0, iters=300)
    ref = Path(ROOT / f"results/sim_seeds/seed{s}/policy_v1.json")
    if ref.exists():
        r = Controller.load(ref)
        a = arms["A_mix0.5"]["policy"]
        out["reproduces_committed_policy_v1"] = bool(
            np.allclose(np.round(a.W, 6), r.W) and len(a.mix) == len(r.mix)
            and all(np.allclose(np.round(x, 6), y) for x, y in zip(a.mix, r.mix)))
    lam_ref = arms["A_mix0.5"]["lam_bar"]
    out["lam_ref"] = lam_ref
    for name, d in arms.items():
        pol = d.pop("policy")
        h = make_sim("v1", seed=EVAL_SEED)
        f = pol.copy(); f.ensure_actions(h.all_actions)
        tru = true_eval(h, f, N_EVAL, EVAL_SEED)
        g = gate(eps_, pol, "behavior", BUDGET, seed=s)
        d.update({"truth_quality": tru["quality"], "truth_cost": tru["cost"],
                  "truth_J_lamref": tru["quality"] - lam_ref * tru["cost"],
                  "truth_J_own": tru["quality"] - d["lam_bar"] * tru["cost"],
                  "gate_decision": g["decision"], "gate_dr_quality": list(g["candidate"]["quality"]["dr"]),
                  "gate_dr_cost": list(g["candidate"]["cost"]["dr"]), "gate_q_prev": g["q_prev"],
                  "gate_quality_ok": g["quality_ok"], "gate_cost_ok": g["cost_ok"]})
    out["arms"] = arms
    out["wall_s"] = round(time.time() - t0, 1)
    print(f"[seed {s}] done in {out['wall_s']}s: " + ", ".join(
        f"{k} T={v['transitions']} it={v['iters']} q={v['truth_quality']:.3f} c={v['truth_cost']:.3f} {v['gate_decision']}"
        for k, v in arms.items()), flush=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--traces-dir", default="traces/sim_seeds_restart")
    ap.add_argument("--policies-dir", default="results/sim_seeds/restart_ablation")
    ap.add_argument("--out", default="results/sim_seeds/restart_ablation.json")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args(argv)
    t0 = time.time()
    with Pool(a.procs) as p:
        res = p.map(run_seed, [(s, a.traces_dir, a.policies_dir) for s in a.seeds])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"budget": BUDGET, "n_eval": N_EVAL, "eval_seed": EVAL_SEED,
                                       "seeds": res, "wall_s": round(time.time() - t0, 1)}, indent=1, default=float))
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
