"""The day loop: a continually trained orchestrator over a frozen multi-agent structure under
scripted harness stages, promoted behind the off-policy gate, against the design-time
alternatives on the same traffic.

Each day = `n_tasks` tasks served by every arm (same task ids and seeds for every arm):

  continual        a LEARNER trains every day: constrained GRPO on replay from the last
                   `replay_window` days of logged states, warm-started from itself (λ too), so
                   training accumulates whatever the gate decides; `pace.gate.gate` then scores the
                   learner against the INCUMBENT on the log served since the last harness event
                   (up to `gate_window` days) and decides PROMOTE (the learner is served from
                   tomorrow) or REVERT (the incumbent keeps serving). Serving uses ε-exploration;
                   propensities include ε. The policy-independent prefix of every episode (planner +
                   first worker call, about $0.022) is not a decision, so the gate's budget is the
                   per-task budget net of the log's mean prefix cost.
  continual_awr    the same loop with AWR + IPS (fixed price λ) as the trainer.
  frozen           the continual policy as it was at the end of day `warmup_days`, never updated;
                   it ignores capabilities shipped after it (EvoHarnessBench's fixed policy).
  so_static        SkillOrchestra-style router: handbook from the warm-up exploration traces, λ_c
                   tuned once on the last warm-up day, never updated.
  so_relearn       the same router re-fitted at design time after every harness stage: `refit_days`
                   days of exploratory serving (ε = `refit_eps`), then a refit and a re-tune; the
                   refit days count against it.
  always_verify / never_verify   the real run's two fixed modes.
  oracle           per generation: constrained GRPO on fresh exploration under that stage, no gate,
                   more iterations; the ceiling the others are measured against. Its learned price
                   λ̄_g is the price of cost in that stage's priced objective J_g = quality − λ̄_g·cost:
                   at the multiplier where the budget binds, the budgeted optimum maximizes the
                   Lagrangian over every policy, so J_g has a true ceiling and regret is well defined.

Held-out evaluation every day: the policy in service on day t is evaluated (expected quality,
`n_heldout` fixed tasks) under every stage g ≤ the current one, so R[i, j] is available for
backward transfer (BWT), forward transfer (FWT), days-to-recover and cumulative regret
(analyze.py). All numbers are simulator ground truth; the gate never sees them. One pre-registered
schedule (`--schedule default`) and a milder sensitivity schedule (`--schedule mild`).

  python -m pace.experiments.mas_continual.run_continual --out results/mas_continual
  python -m pace.experiments.mas_continual.run_continual --schedule mild --out results/mas_continual/mild

Two options of the cgrpo continual arm (RESULTS_rao_metainit.md): `--delegation-credit rao --lambda-d 0.5`
trains with RAO-style per-decision delegation credit (rao_reward.py); `--init meta` starts the arm
from a meta-learned controller instead of the uniform policy (meta_init.py).

Gate review options (results/mas_continual/gate_v2/RESULTS.md; defaults reproduce the committed run):
  --gate v2             pace.gate.gate_v2: joint-bootstrap quality difference, total episode cost
                        (prefix included) against the FULL budget, overlap guard -> UNEVALUABLE
  --gate-holdout 0.3    each day's served episodes are split 70/30 by task id (seeded hash); the
                        trainers use only the 70% slice, the gate only the 30% slice (same window rule:
                        since the last harness event, at most `gate_window` days)
  --gate-shadow 1       also compute the v2 record when deciding with v1 (logged, never acted on)
  --log-new-action-mass 1   log the served policy's probability on the actions introduced by the
                        current stage, on every day of that stage (against the ε exploration floor)
  --arms gate           continual, continual_awr, always_verify, never_verify (+ the oracle)
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...controller import Controller
from ...gate import gate, gate_v2
from ...trace import Episode
from ..mas.offpolicy import per_decision
from . import drift, simenv
from .router import Handbook, HandbookRouter, tune_lambda_c
from .simenv import ACTIONS, FixedPolicy, MASSimHarness, UniformPolicy, serve
from .meta_init import audit, meta_learn
from .rao_reward import delegation_share, train_cgrpo_replay_shaped
from .train import train_awr_day, train_cgrpo_replay, true_eval


@dataclass
class Config:
    schedule: str = "default"
    n_days: int = 40
    n_tasks: int = 96
    eps: float = 0.10                 # exploration of the served continual policies
    budget: float = 0.090             # USD per task (always_verify measured $0.0906)
    tol: float = 0.005
    eps_q: float = 0.05                # at n ≈ 480 binary outcomes the DR quality CI half-width is ≈ 0.04; the core default 0.03 rejects the incumbent against itself
    warmup_days: int = 5
    gate_window: int = 5
    replay_window: int = 5
    cgrpo_iters: int = 60
    cgrpo_batch: int = 16
    cgrpo_group: int = 6
    awr_lam: float = 2.0
    awr_epochs: int = 200
    n_heldout: int = 150
    oracle_iters: int = 300
    oracle_batch: int = 24
    oracle_group: int = 8
    oracle_collect: int = 200
    refit_days: int = 3
    refit_eps: float = 0.30
    boot: int = 200
    seed: int = 0
    recover_tol: float = 0.02         # "recovered" = within this of the oracle's held-out J
    # verifier ablation knobs (RESULTS_ablation_verifier.md); the committed run uses the defaults
    verifier: str = "verify-1"        # the fixed modes' verifier; "verify-2" also makes it available to every arm from day 1
    max_calls: int = 8
    verifier_exempt: bool = False
    arms: str = "all"                 # "all" | "ablation" (continual cgrpo, always_verify, never_verify, oracle only)
    # RAO-style delegation credit and meta-learned init of the cgrpo arm (RESULTS_rao_metainit.md)
    delegation_credit: str = "none"   # "none" | "rao"
    lambda_d: float = 0.5
    init: str = "zero"                # "zero" (uniform policy) | "meta" (meta_init.meta_learn on stages g0..g3)
    meta_outer: int = 3
    meta_inner_days: int = 2
    # gate review options (gate_v2/RESULTS.md); the defaults are the committed behaviour
    gate: str = "v1"                  # "v1" | "v2" (pace.gate.gate_v2)
    gate_holdout: float = 0.0         # fraction of each day's served episodes (by task id) reserved for the gate
    ess_floor: float = 0.2            # v2 overlap guard
    clip_frac_max: float = 0.1        # v2 overlap guard
    gate_shadow: int = 0              # 1: with --gate v1, also log the v2 record (not acted on)
    log_new_action_mass: int = 0      # 1: log served mass on newly introduced actions


CONTINUAL = ("continual", "continual_awr")
ARMS = ("continual", "continual_awr", "frozen", "so_static", "so_relearn", "always_verify", "never_verify")


def _holdout_split(eps_: list[Episode], frac: float, seed: int, day: int) -> tuple[list[Episode], list[Episode]]:
    """(trainer slice, gate slice) of one day's served episodes, split by task id with a seeded hash
    (the same task id goes to the same side for every arm). frac = 0: both slices are the full day."""
    if frac <= 0:
        return eps_, eps_
    side = lambda e: (zlib.crc32(f"{seed}:{day}:{e.task_id}".encode()) % 100000) < frac * 100000
    return [e for e in eps_ if not side(e)], [e for e in eps_ if side(e)]


def _new_action_mass(eps_: list[Episode], pol: Controller, new_actions: set[str], eps: float) -> dict:
    """Served probability on actions introduced by the current stage, over the logged decisions at which
    at least one of them was legal; the floor is what ε-exploration alone gives them (ε·|new ∩ legal|/|legal|)."""
    pol = pol.copy()
    pol.ensure_actions(ACTIONS)
    mass, base, floor, per_max, per_floor = [], [], [], [], []
    by_action: dict[str, list[float]] = {}
    n_dec = 0
    for e in eps_:
        for s in e.steps:
            n_dec += 1
            legal = s.legal or []
            new = [a for a in legal if a in new_actions]
            if not new:
                continue
            p = pol.probs(s.features, legal, eps=eps)
            pb = pol.probs(s.features, legal, eps=0.0)
            ix = [pol.index(a) for a in new]
            mass.append(float(p[ix].sum())); base.append(float(pb[ix].sum()))
            floor.append(eps * len(new) / len(legal))
            per_max.append(float(p[ix].max())); per_floor.append(eps / len(legal))
            for a, i in zip(new, ix):
                by_action.setdefault(a, []).append(float(p[i]))
    if not mass:
        return {"n_states": 0, "n_decisions": n_dec}
    return {"n_states": len(mass), "n_decisions": n_dec, "mass": float(np.mean(mass)), "mass_no_eps": float(np.mean(base)),
            "floor": float(np.mean(floor)), "per_action_max": float(np.mean(per_max)), "per_action_floor": float(np.mean(per_floor)),
            "by_action": {a: float(np.mean(v)) for a, v in sorted(by_action.items())},
            "chosen_share": sum(1 for e in eps_ for s in e.steps if s.action_name in new_actions) / max(n_dec, 1)}


def _snapshots(logs: list[list[Episode]]) -> list[dict]:
    return [s.state for day in logs for e in day for s in e.steps if s.state is not None]


def _served_stats(eps_: list[Episode], lam_j: float) -> dict:
    """Per-day served numbers; J uses the stage's price λ̄_g."""
    n = len(eps_)
    q = sum(e.quality for e in eps_) / n
    eq = sum(e.meta["expected_quality"] for e in eps_) / n
    c = sum(e.total_cost for e in eps_) / n
    n_dec = max(sum(len(e.steps) for e in eps_), 1)
    return {"quality": q, "expected_quality": eq, "cost": c, "J": eq - lam_j * c,
            "calls": sum(e.meta["n_calls"] for e in eps_) / n, "n": n,
            "retries": sum(e.meta["n_retry"] for e in eps_) / n, "verifier_calls": sum(e.meta["n_verify"] for e in eps_) / n,
            "cap_hit_rate": sum(e.meta["cap_hit"] for e in eps_) / n,
            "verify_share": sum(1 for e in eps_ for s in e.steps if s.action_name.startswith("verify@")) / n_dec,
            "prefix_cost": sum(e.total_cost - sum(s.cost for s in e.steps) for e in eps_) / n,
            "action_mix": _mix(eps_), **_delegation(eps_)}


def _delegation(eps_: list[Episode]) -> dict:
    d = delegation_share(eps_)
    return {"delegate_share": d["share"], "delegate_by_agent": d["by_agent"]}


def _mix(eps_: list[Episode]) -> dict:
    cnt: dict[str, int] = {}
    for e in eps_:
        for s in e.steps:
            cnt[s.action_name] = cnt.get(s.action_name, 0) + 1
    tot = max(sum(cnt.values()), 1)
    return {a: round(v / tot, 3) for a, v in sorted(cnt.items(), key=lambda kv: -kv[1])[:6]}


def train_oracle(h: MASSimHarness, cfg: Config, init: Controller | None, seed: int) -> tuple[Controller, float]:
    tasks = h.sample_tasks(cfg.oracle_collect, seed=seed + 11)
    logs = serve(h, tasks, UniformPolicy(), seed=seed + 12, mode="oracle_explore")
    return train_cgrpo_replay(_snapshots([logs]), h, cfg.budget, iters=cfg.oracle_iters, batch=cfg.oracle_batch,
                              group=cfg.oracle_group, init=init, serve_eps=0.0, seed=seed + 13,
                              lam0=float(init.meta.get("lambda", 0.0)) if init is not None else 0.0)


def run(cfg: Config, out: Path, log=print) -> dict:
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "policies").mkdir(exist_ok=True)
    drift.apply_bonus_scale(cfg.schedule)
    gens = drift.schedule(cfg.schedule, verifier2_from_day1=cfg.verifier == "verify-2", max_calls=cfg.max_calls,
                          verifier_exempt_from_cap=cfg.verifier_exempt)
    arms_cont = CONTINUAL if cfg.arms in ("all", "gate") else ("continual",)
    design_time = cfg.arms == "all"
    rng = random.Random(cfg.seed)
    harness: dict[int, MASSimHarness] = {}
    heldout_tasks: dict[int, list] = {}
    oracle: dict[int, dict] = {}
    oracle_pol: Controller | None = None
    stage0_actions = [a for a in ACTIONS if a in _legal_universe(gens[0].params)]

    # arms
    uniform0 = Controller(ACTIONS, simenv.N_FEATURES, eps=cfg.eps, meta={"name": "uniform", "lambda": 0.0})
    incumbent: dict[str, Controller] = {"continual": uniform0.copy(), "continual_awr": uniform0.copy()}
    learner: dict[str, Controller] = {"continual": uniform0.copy(), "continual_awr": uniform0.copy()}
    if cfg.init == "meta":
        init_pol, rep = meta_learn(cfg.budget, cfg.eps, cfg.schedule, outer=cfg.meta_outer, inner_days=cfg.meta_inner_days,
                                   n_tasks=cfg.n_tasks, iters=cfg.cgrpo_iters, batch=cfg.cgrpo_batch, group=cfg.cgrpo_group,
                                   seed=cfg.seed, gens=gens, log=log)
        rep["audit"] = audit(rep, cfg.seed, len(gens))
        if not rep["audit"]["ok"]:
            raise RuntimeError(f"meta-init touched held-out data: {rep['audit']}")
        incumbent["continual"], learner["continual"] = init_pol.copy(), init_pol.copy()
        init_pol.save(out / "policies" / "meta_init.json")
    elif cfg.init != "zero":
        raise ValueError(cfg.init)
    lam_of: dict[int, float] = {}
    frozen: Controller | None = None
    so_static: HandbookRouter | None = None
    so_relearn: HandbookRouter | None = None
    relearn_window: list[list[Episode]] | None = None
    relearn_start_day: int | None = None
    logs: dict[str, list[list[Episode]]] = {a: [] for a in CONTINUAL + ("so_relearn",)}
    logs_train: dict[str, list[list[Episode]]] = {a: [] for a in CONTINUAL}   # = logs when gate_holdout == 0
    logs_gate: dict[str, list[list[Episode]]] = {a: [] for a in CONTINUAL}
    fixed = {"always_verify": FixedPolicy(True, verifier=cfg.verifier), "never_verify": FixedPolicy(False)}
    last_event_day = 1
    days_f = (out / "days.jsonl").open("w")
    gates_f = (out / "gates.jsonl").open("w")
    meta: dict = {"config": asdict(cfg), "generations": [], "oracle": {}, "handbooks": {}, "events": []}
    if cfg.init == "meta":
        meta["meta_init"] = rep
    for i, g in enumerate(gens):
        meta["generations"].append({"index": g.index, "name": g.name, "axis": g.axis, "day": g.day, "date": g.date,
                                    "description": g.description,
                                    "changed": drift.changed_fields(gens[i - 1].params, g.params) if i else []})
    stats_cal = {}
    for day in range(1, cfg.n_days + 1):
        gen = drift.generation_for(day, gens)
        event = day == gen.day and gen.index > 0
        if gen.index not in harness:
            harness[gen.index] = MASSimHarness(gen.params, seed=cfg.seed * 1000 + gen.index, version=gen.version)
            heldout_tasks[gen.index] = harness[gen.index].sample_tasks(cfg.n_heldout, seed=cfg.seed + 500 + gen.index)
            t0 = time.time()
            oracle_pol, lam_o = train_oracle(harness[gen.index], cfg, oracle_pol, cfg.seed + 100 * gen.index)
            lam_of[gen.index] = lam_o
            ev = true_eval(harness[gen.index], oracle_pol, heldout_tasks[gen.index], seed=cfg.seed + 7, lam_j=lam_o)
            ev.update({"lambda": lam_o, "train_s": round(time.time() - t0, 1)})
            oracle[gen.index] = ev
            meta["oracle"][str(gen.index)] = ev
            # the untrained reference b_j of forward transfer: a uniform policy over the legal set
            meta.setdefault("uniform", {})[str(gen.index)] = true_eval(harness[gen.index], UniformPolicy(), heldout_tasks[gen.index],
                                                                        seed=cfg.seed + 7, lam_j=lam_o)
            oracle_pol.save(out / "policies" / f"oracle_{gen.name}.json")
            log(f"[day {day:2d}] {gen.name}: oracle held-out quality {ev['quality']:.3f} cost ${ev['cost']:.4f} J {ev['J']:.3f} "
                f"λ̄ {lam_o:.2f} ({ev['train_s']} s)")
            if gen.index == 0:   # calibration check of the fixed modes on the stage-0 harness
                for m, pol in fixed.items():
                    stats_cal[m] = simenv.mode_stats(serve(harness[0], harness[0].sample_tasks(1000, seed=99), pol, seed=98, with_state=False))
                meta["calibration_check"] = {"simulated": stats_cal, "measured": simenv.M.modes}
        h = harness[gen.index]
        lam_j = lam_of[gen.index]
        if event:
            last_event_day = day
            meta["events"].append({"day": day, "generation": gen.name, "axis": gen.axis})
            if so_relearn is not None:
                relearn_window, relearn_start_day = [], day
        tasks = h.sample_tasks(cfg.n_tasks, seed=cfg.seed * 100000 + day)

        # ---------------------------------------------------------- serve
        served: dict[str, list[Episode]] = {}
        serving: dict[str, object] = {}
        for arm in arms_cont:
            serving[arm] = incumbent[arm]
            served[arm] = serve(h, tasks, incumbent[arm], seed=cfg.seed + day * 10 + 1, eps=cfg.eps, mode=arm)
            logs[arm].append(served[arm])
            tr, gt = _holdout_split(served[arm], cfg.gate_holdout, cfg.seed, day)
            logs_train[arm].append(tr); logs_gate[arm].append(gt)
        if frozen is not None:
            serving["frozen"] = frozen
            served["frozen"] = serve(h, tasks, frozen, seed=cfg.seed + day * 10 + 2, eps=0.0, mode="frozen",
                                     with_state=False, legal_filter=stage0_actions)
        if so_static is not None:
            serving["so_static"] = so_static
            served["so_static"] = serve(h, tasks, so_static, seed=cfg.seed + day * 10 + 3, eps=0.0, mode="so_static", with_state=False)
        if so_relearn is not None:
            serving["so_relearn"] = so_relearn
            exploring = relearn_window is not None
            served["so_relearn"] = serve(h, tasks, so_relearn, seed=cfg.seed + day * 10 + 4, eps=cfg.refit_eps if exploring else 0.0,
                                         mode="so_relearn", with_state=False)
            if exploring:
                relearn_window.append(served["so_relearn"])
        for m, pol in fixed.items():
            serving[m] = pol
            served[m] = serve(h, tasks, pol, seed=cfg.seed + day * 10 + 5, mode=m, with_state=False)

        # ------------------------------------------------- held-out eval of the serving policies
        heldout: dict[str, dict[int, dict]] = {}
        for arm, pol in serving.items():
            heldout[arm] = {}
            for gi in sorted(harness):
                if gi > gen.index:
                    continue
                lf = stage0_actions if arm == "frozen" else None
                heldout[arm][gi] = true_eval(harness[gi], pol, heldout_tasks[gi], seed=cfg.seed + 7, lam_j=lam_of[gi], legal_filter=lf)

        # ---------------------------------------------------------- train + gate
        gate_rows: dict[str, dict] = {}
        for arm in arms_cont:
            window = logs_gate[arm][max(0, last_event_day - 1):][-cfg.gate_window:]
            gate_log = [e for d in window for e in d]
            train_window = logs_train[arm][max(0, last_event_day - 1):][-cfg.gate_window:]
            awr_log = [e for d in train_window for e in d]          # = gate_log when gate_holdout == 0
            replay = _snapshots(logs_train[arm][-cfg.replay_window:])
            inc, prev = incumbent[arm], learner[arm]
            t0 = time.time()
            if arm == "continual" and cfg.delegation_credit != "none":
                cand, lam_bar = train_cgrpo_replay_shaped(replay, h, cfg.budget, iters=cfg.cgrpo_iters, batch=cfg.cgrpo_batch,
                                                          group=cfg.cgrpo_group, init=prev, serve_eps=cfg.eps,
                                                          lam0=float(prev.meta.get("lambda", 0.0)), seed=cfg.seed + day,
                                                          credit=cfg.delegation_credit, lambda_d=cfg.lambda_d)
            elif arm == "continual":
                cand, lam_bar = train_cgrpo_replay(replay, h, cfg.budget, iters=cfg.cgrpo_iters, batch=cfg.cgrpo_batch,
                                                   group=cfg.cgrpo_group, init=prev, serve_eps=cfg.eps,
                                                   lam0=float(prev.meta.get("lambda", 0.0)), seed=cfg.seed + day)
            else:
                cand = train_awr_day(awr_log, cfg.awr_lam, prev, serve_eps=cfg.eps, epochs=cfg.awr_epochs)
                lam_bar = cfg.awr_lam
            cand.meta["name"] = f"{arm}-day{day}"
            learner[arm] = cand
            train_s = time.time() - t0
            t0 = time.time()
            prefix = sum(e.total_cost - sum(s.cost for s in e.steps) for e in gate_log) / len(gate_log)
            v2rec = None
            if cfg.gate == "v2":
                v2rec = gate_v2(gate_log, cand, inc, cfg.budget, cfg.eps_q, cfg.tol, boot=cfg.boot, seed=cfg.seed + day,
                                ess_floor=cfg.ess_floor, clip_frac_max=cfg.clip_frac_max)
                ev2 = v2rec["estimates"]
                g = {"decision": v2rec["decision"], "quality_ok": v2rec["quality_ok"], "cost_ok": v2rec["cost_ok"],
                     "q_prev": ev2["inc_quality"][0],
                     "candidate": {"quality": {"dr": ev2["cand_quality"], "snips": (float("nan"),) * 3, "ips": (float("nan"),) * 3},
                                   "cost": {"dr": ev2["cand_decision_cost"]}, "n": v2rec["n"],
                                   "max_rho": v2rec["diagnostics"]["max_rho"]},
                     "incumbent": {"quality": ev2["inc_quality"], "cost": ev2["inc_total_cost"]}}
            elif cfg.gate == "v1":
                g = gate(gate_log, cand, inc, cfg.budget - prefix, cfg.eps_q, cfg.tol, boot=cfg.boot, seed=cfg.seed + day)
                if cfg.gate_shadow:
                    v2rec = gate_v2(gate_log, cand, inc, cfg.budget, cfg.eps_q, cfg.tol, boot=cfg.boot, seed=cfg.seed + day,
                                    ess_floor=cfg.ess_floor, clip_frac_max=cfg.clip_frac_max)
            else:
                raise ValueError(cfg.gate)
            pd = per_decision(gate_log, cand, reward="quality", boot=50, seed=cfg.seed + day)
            gate_s = time.time() - t0
            ct = true_eval(h, cand, heldout_tasks[gen.index], seed=cfg.seed + 7, lam_j=lam_j)
            it = heldout[arm][gen.index]
            acceptable = ct["cost"] <= cfg.budget + cfg.tol and ct["quality"] >= it["quality"] - cfg.eps_q
            row = {"day": day, "arm": arm, "generation": gen.name, "decision": g["decision"],
                   "quality_ok": g["quality_ok"], "cost_ok": g["cost_ok"], "q_prev": g["q_prev"],
                   "dr_quality": g["candidate"]["quality"]["dr"], "dr_cost": g["candidate"]["cost"]["dr"],
                   "snips_quality": g["candidate"]["quality"]["snips"], "ips_quality": g["candidate"]["quality"]["ips"],
                   "incumbent_dr": g["incumbent"], "n": g["candidate"]["n"], "max_rho": g["candidate"]["max_rho"],
                   "prefix_cost": prefix, "gate_budget": cfg.budget - prefix,
                   "per_decision_ess": pd["ess"], "per_decision_n": pd["n"], "per_decision_snips": pd["snips"], "per_decision_dr": pd["dr"],
                   "lambda_candidate": lam_bar, "lambda_incumbent": float(inc.meta.get("lambda", 0.0)),
                   "truth": {"candidate": ct, "incumbent": it, "acceptable": bool(acceptable),
                             "over_budget": bool(ct["cost"] > cfg.budget + cfg.tol),
                             "regressed": bool(ct["quality"] < it["quality"] - cfg.eps_q)},
                   "gate_correct": bool((g["decision"] == "PROMOTE") == acceptable),
                   "train_s": round(train_s, 2), "gate_s": round(gate_s, 2)}
            if cfg.gate != "v1" or cfg.gate_holdout > 0 or cfg.gate_shadow:
                row["gate_version"] = cfg.gate
                row["gate_budget"] = cfg.budget if cfg.gate == "v2" else cfg.budget - prefix
                row["gate_holdout"] = cfg.gate_holdout
                row["n_train_episodes"] = len(awr_log)
                row["truth"]["diff_quality"] = ct["quality"] - it["quality"]
            if v2rec is not None:
                row["gate_v2" if cfg.gate == "v2" else "gate_v2_shadow"] = v2rec
            gates_f.write(json.dumps(row) + "\n")
            gate_rows[arm] = row
            if g["decision"] == "PROMOTE":
                incumbent[arm] = cand
            if day % 5 == 0 or event:
                incumbent[arm].save(out / "policies" / f"{arm}_day{day}.json")

        # ------------------------------------------------- design-time arms
        if day == cfg.warmup_days and design_time:
            frozen = learner["continual"].copy()
            frozen.eps = 0.0
            frozen.meta = {**frozen.meta, "name": "frozen"}
            frozen.save(out / "policies" / "frozen.json")
            learn = [e for d in logs["continual"][:-1] for e in d]
            tune = logs["continual"][-1]
            hb = Handbook().learn(learn)
            tl = tune_lambda_c(hb, tune, cfg.budget, cfg.tol)
            so_static = HandbookRouter(hb, tl["lambda_c"], name="so_static")
            so_relearn = HandbookRouter(hb, tl["lambda_c"], name="so_relearn")
            meta["handbooks"]["static"] = {"day": day, "n_learn": len(learn), "n_tune": len(tune), "lambda_c": tl["lambda_c"],
                                           "tuning": tl["rows"], "within_budget": tl["within_budget"], "handbook": hb.to_dict()}
            log(f"[day {day:2d}] frozen controller fixed; SkillOrchestra-style handbook learned on {len(learn)} episodes, "
                f"λ_c = {tl['lambda_c']} (tuned on {len(tune)})")
        if relearn_window is not None and len(relearn_window) >= cfg.refit_days:
            learn = [e for d in relearn_window[:-1] for e in d]
            tune = relearn_window[-1]
            hb = Handbook().learn(learn)
            tl = tune_lambda_c(hb, tune, cfg.budget, cfg.tol)
            so_relearn = HandbookRouter(hb, tl["lambda_c"], name="so_relearn")
            meta["handbooks"][f"relearn_{gen.name}"] = {"day": day, "start_day": relearn_start_day, "n_learn": len(learn),
                                                        "n_tune": len(tune), "lambda_c": tl["lambda_c"], "tuning": tl["rows"],
                                                        "within_budget": tl["within_budget"], "handbook": hb.to_dict()}
            log(f"[day {day:2d}] so_relearn refit on days {relearn_start_day}-{day}: λ_c = {tl['lambda_c']}")
            relearn_window = None

        # ---------------------------------------------------------- record
        for arm in served:
            row = {"day": day, "date": drift._date(day), "generation": gen.name, "gen_index": gen.index, "axis": gen.axis,
                   "event": bool(event), "arm": arm, "served": _served_stats(served[arm], lam_j),
                   "eps_served": (cfg.eps if arm in CONTINUAL else cfg.refit_eps if (arm == "so_relearn" and relearn_window is not None) else 0.0),
                   "lambda": float(serving[arm].meta.get("lambda", serving[arm].meta.get("lambda_c", float("nan")))),
                   "heldout": {str(k): v for k, v in heldout[arm].items()},
                   "oracle_J": oracle[gen.index]["J"], "lambda_stage": lam_j}
            if cfg.log_new_action_mass and arm in CONTINUAL and gen.index > 0:
                prev_gen = gens[[x.index for x in gens].index(gen.index) - 1]
                new_acts = _legal_universe(gen.params) - _legal_universe(prev_gen.params)
                if new_acts:
                    row["new_action_mass"] = {"stage_day": day - gen.day + 1, "new_actions": sorted(new_acts),
                                              **_new_action_mass(served[arm], serving[arm], new_acts, cfg.eps)}
            if arm in gate_rows:
                gr = gate_rows[arm]
                row["gate"] = {k: gr[k] for k in ("decision", "quality_ok", "cost_ok", "dr_quality", "dr_cost", "per_decision_ess",
                                                  "per_decision_n", "n", "lambda_candidate", "gate_correct")}
                row["gate"]["truth"] = gr["truth"]["acceptable"]
            days_f.write(json.dumps(row) + "\n")
        c = served["continual"]
        gr = gate_rows["continual"]
        log(f"[day {day:2d}] {gen.name:12s}{' EVENT' if event else '      '} continual: served q {_served_stats(c, lam_j)['expected_quality']:.3f} "
            f"${_served_stats(c, lam_j)['cost']:.4f} | held-out J {heldout['continual'][gen.index]['J']:.3f} vs oracle {oracle[gen.index]['J']:.3f} "
            f"| gate {gr['decision']} (DR q {gr['dr_quality'][0]:.3f} [{gr['dr_quality'][1]:.3f},{gr['dr_quality'][2]:.3f}], "
            f"cost hi {gr['dr_cost'][2]:.4f}, ESS {gr['per_decision_ess']:.0f}/{gr['per_decision_n']}, truth {'ok' if gr['truth']['acceptable'] else 'BAD'}) "
            f"λ̄ {gr['lambda_candidate']:.2f}" + (f" | awr {gate_rows['continual_awr']['decision']}" if 'continual_awr' in gate_rows else ""))
    days_f.close(); gates_f.close()
    meta["runtime_s"] = round(time.time() - t_start, 1)
    meta["n_actions"], meta["n_features"] = len(ACTIONS), simenv.N_FEATURES
    (out / "meta.jsonl").write_text(json.dumps(meta) + "\n")
    log(f"[done] {cfg.n_days} days in {meta['runtime_s']} s -> {out}")
    return meta


def _legal_universe(p: simenv.Params) -> set[str]:
    out = set()
    for a in p.available_agents:
        m = simenv.AGENT[a].mode
        out.add({"data-worker": "delegate", "verifier": "verify", "planner": "replan", "answer": "answer"}[m] + "@" + a)
    out |= {f"call@{t}" for t in p.available_tools} | {f"engage@{s}" for s in p.available_skills}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Continual MAS orchestrator day loop (CPU, simulator; no API calls).")
    for f, v in asdict(Config()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--out", default="results/mas_continual")
    a = ap.parse_args(argv)
    cfg = Config(**{f: getattr(a, f) for f in asdict(Config())})
    run(cfg, Path(a.out))


if __name__ == "__main__":
    main()
