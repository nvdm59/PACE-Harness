"""Drift evaluation: scripted harness events and the priced-BWT table.

Events:
  reprice(tier, factor)  -- a provider changes a tier's price
  swap("v2")             -- the host harness ships a new version (sim: SimHarnessV2 =
                            + CACHED_RETRIEVE, aged verifier on hard tasks, frontier x1.5)

Priced objective J = quality - lam * cost (lam = the price of cost; default:
the frozen policy's learned Lagrange multiplier). priced-BWT is the backward-transfer
metric of continual learning (Lopez-Paz & Ranzato 2017, arXiv:1706.08840) applied to the
priced objective: how much a FROZEN controller lost when the harness changed under it.
Reported, on sim ground truth (Monte Carlo over n_eval fresh tasks, expected quality):
  (i)   frozen policy on v1
  (ii)  frozen policy on v2 (+ reprice events)
  (iii) serving policy on v2 after re-training on fresh v2 traces and the gate:
        the adapted policy if the gate PROMOTEd it, else the frozen one
  priced-BWT = J(ii) - J(i)      (what drift cost a frozen controller)
  recovery   = J(iii) - J(ii)    (what the gated update won back)

CLI:
  python -m pace.drift --frozen policies/policy_v1.json --adapted policies/policy_v2.json \\
      --gate-json results/gate_v2.json --reprice mid:1.2
If --adapted is omitted, drift.py runs collect -> cgrpo -> gate on v2 itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import collect, load_policy, true_eval
from .controller import Controller
from .gate import estimate, format_table, gate
from .harness.sim import SimHarness, make_sim
from .train_offline import train_cgrpo


# ------------------------------------------------------------------ events
def reprice(harness: SimHarness, tier: str, factor: float) -> SimHarness:
    harness.reprice(tier, factor)
    return harness


def swap(harness: SimHarness, version: str = "v2") -> SimHarness:
    """New harness version; carries over any reprice events already applied
    (on top of whatever repricing the new version ships with)."""
    new = make_sim(version, seed=harness.rng.randrange(1 << 30))
    for t, f in harness.price_factor.items():
        if f != 1.0:
            new.reprice(t, f)
    return new


def apply_events(harness: SimHarness, events: list[tuple]) -> SimHarness:
    for ev in events:
        if ev[0] == "reprice":
            harness = reprice(harness, ev[1], float(ev[2]))
        elif ev[0] == "swap":
            harness = swap(harness, ev[1])
        else:
            raise ValueError(f"unknown event {ev!r}")
    return harness


def parse_events(reprice_specs: list[str], version: str = "v2") -> list[tuple]:
    return [("swap", version)] + [("reprice", *s.split(":")) for s in reprice_specs]


# ------------------------------------------------------------------- table
def drift_report(frozen: Controller, serving_v2: Controller, events: list[tuple], lam: float,
                 n_eval: int = 20000, seed: int = 4242, budget: float | None = None,
                 decision: str = "PROMOTE", tol: float = 0.02) -> dict:
    """Ground-truth (sim) table of the three conditions and the two drift metrics. `tol` is the
    same cost tolerance the gate uses for "within budget"."""
    h1 = make_sim("v1", seed=seed)
    h2 = apply_events(make_sim("v1", seed=seed), events)
    f1 = frozen.copy(); f1.ensure_actions(h1.all_actions)
    f2 = frozen.copy(); f2.ensure_actions(h2.all_actions)
    a2 = serving_v2.copy(); a2.ensure_actions(h2.all_actions)
    rows = {"(i) frozen on v1": true_eval(h1, f1, n_eval, seed),
            "(ii) frozen on v2": true_eval(h2, f2, n_eval, seed),
            f"(iii) {'adapted' if decision == 'PROMOTE' else 'frozen (gate REVERTED)'} on v2": true_eval(h2, a2, n_eval, seed)}
    for r in rows.values():
        r["J"] = r["quality"] - lam * r["cost"]
    J = [r["J"] for r in rows.values()]
    return {"rows": rows, "lam": lam, "budget": budget, "tol": tol, "events": events, "decision": decision,
            "priced_bwt": J[1] - J[0], "recovery": J[2] - J[1], "n_eval": n_eval}


def format_report(rep: dict) -> str:
    ev = ", ".join(f"{e[0]}({', '.join(map(str, e[1:]))})" for e in rep["events"])
    b, tol = rep["budget"], rep.get("tol", 0.02)
    lines = [f"### Drift: {ev}   (J = quality - {rep['lam']:.3f} * cost; sim ground truth, n={rep['n_eval']})", "",
             "| condition | quality | cost | within budget | J |", "|---|---|---|---|---|"]
    for name, r in rep["rows"].items():
        ok = "-" if b is None else ("yes" if r["cost"] <= b + tol else "NO")
        lines.append(f"| {name} | {r['quality']:.3f} | {r['cost']:.3f} | {ok} | {r['J']:.3f} |")
    lines += ["", f"priced-BWT = J(ii) - J(i) = **{rep['priced_bwt']:+.3f}**",
              f"recovery   = J(iii) - J(ii) = **{rep['recovery']:+.3f}**   (gate decision: {rep['decision']})"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Drift evaluation: priced-BWT and recovery.")
    ap.add_argument("--frozen", required=True, help="policy trained on v1")
    ap.add_argument("--adapted", default=None, help="policy re-trained on v2 traces (else trained here)")
    ap.add_argument("--gate-json", default=None, help="gate result for the adapted policy")
    ap.add_argument("--q-prev-from", default=None, help="(self-run) gate JSON that promoted the frozen policy")
    ap.add_argument("--reprice", nargs="*", default=[], help="extra reprice events after the swap, tier:factor")
    ap.add_argument("--lam", type=float, default=None, help="price of cost in J (default: frozen's learned lambda)")
    ap.add_argument("--budget", type=float, default=None, help="cost budget (default: the frozen policy's)")
    ap.add_argument("--tol", type=float, default=0.02, help="cost tolerance for 'within budget' (as in the gate)")
    ap.add_argument("--n-eval", type=int, default=20000, help="Monte Carlo tasks per condition")
    ap.add_argument("--n-collect", type=int, default=3000, help="(self-run) v2 episodes to collect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None, help="write the report as JSON")
    a = ap.parse_args(argv)
    frozen = Controller.load(a.frozen)
    lam = a.lam if a.lam is not None else float(frozen.meta.get("lambda", 0.5))
    budget = a.budget if a.budget is not None else frozen.meta.get("budget")
    events = parse_events(a.reprice)
    if a.adapted:
        adapted = Controller.load(a.adapted)
        decision = json.loads(Path(a.gate_json).read_text())["decision"] if a.gate_json else "PROMOTE"
    else:
        h2 = apply_events(make_sim("v1", seed=a.seed), events)
        beh = load_policy(a.frozen, h2, eps=0.3)
        logs = collect(h2, beh, a.n_collect, a.seed)
        adapted, _ = train_cgrpo(logs, h2, budget or 0.30, init=frozen, seed=a.seed)
        inc = frozen.copy(); inc.ensure_actions(h2.all_actions)
        if a.q_prev_from:
            q_prev = json.loads(Path(a.q_prev_from).read_text())["candidate"]["quality"]["dr"][0]
        else:   # the frozen policy's pre-drift quality, measured on a fresh v1 log
            h1 = make_sim("v1", seed=a.seed + 1)
            logs1 = collect(h1, load_policy(a.frozen, h1, eps=0.3), a.n_collect, a.seed + 1)
            q_prev = estimate(logs1, frozen)["quality"]["dr"][0]
        g = gate(logs, adapted, inc, budget or 0.30, q_prev=q_prev)
        print(format_table(g, "Gate on v2 (adapted vs frozen)"), "\n")
        decision = g["decision"]
    serving = adapted if decision == "PROMOTE" else frozen
    rep = drift_report(frozen, serving, events, lam, a.n_eval, budget=budget, decision=decision, tol=a.tol)
    print(format_report(rep))
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
