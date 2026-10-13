"""Offline trainers on a logged trace file.

(a) awr    Advantage-Weighted Regression (Peng et al. 2019, arXiv:1910.00177) with an
           inverse-propensity correction, full batch. One record per logged step: return =
           quality - lam * cost-to-go, a per-bucket value baseline, weights
           exp(A / beta) / mu self-normalized over the batch. `--use-counterfactual` adds the
           logged counterfactual terminal arms as extra records (the support fix: offline RL can
           only reweight what the logs contain). lam is a FIXED price of cost here.

(b) cgrpo  Constrained GRPO on replay (group-relative policy optimization, Shao et al. 2024,
           arXiv:2402.03300, with the group mean as the baseline), for simulable harnesses:
           sample logged states, re-roll `group` continuations from each with the current served
           policy, advantage = Lagrangian reward (quality - lam * cost) minus the group mean.
           lam is LEARNED by dual ascent on an EMA of full-episode cost against the budget, with
           a proportional damping term (the PID-Lagrangian of Stooke, Achiam & Abbeel 2020,
           arXiv:2007.03964, without the derivative term; cf. RCPO, Tessler et al. 2018,
           arXiv:1805.11074). The served policy is the iterate average over the second half of
           training (snapshot mix, see pace.controller).

           Rollouts are sampled from the policy as served, i.e. INCLUDING its eps exploration
           mass, so the cost the dual variable tracks is the served cost. The policy gradient
           therefore uses the score of the eps-mixed policy (`Controller.eps_score_weights`),
           not of the bare softmax.

(c) cgrpo on a lookup table  The same trainer when every arm of a first-step contextual
           bandit is logged (pace.lookup.LookupTable, CRMArena counterfactual logs): a re-roll
           is an exact lookup of the sampled tier's logged (quality, cost).

CLI:
  python -m pace.train_offline --traces traces/v1.jsonl --method cgrpo --budget 0.30 --out policies/p.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .controller import Controller
from .features import N_FEATURES, CRMBuckets, bucket_key, crm_context, featurize
from .harness.base import Harness
from .harness.sim import make_sim, parse_action, parse_reprice
from .trace import Episode, read_jsonl

TERMINAL = {"ANSWER", "ESCALATE"}


def _new_policy(actions: list[str], init: Controller | None, serve_eps: float) -> Controller:
    if init is not None:
        pol = init.copy()
        pol.ensure_actions(actions)
        pol.mix = None
    else:
        pol = Controller(actions, N_FEATURES)
    pol.eps = serve_eps
    return pol


def _dual_step(lam_i: float, cost_ema: float, budget: float, lr_lam: float, kp: float) -> tuple[float, float]:
    """One PI-damped dual-ascent step. Returns (integral state, effective lam).
    lam_i accumulates lr_lam * (cost gap) and is projected at 0; the served lam adds a
    proportional term kp * (cost gap), which stops the pure integral form lagging the policy
    and orbiting the budget."""
    lam_i = max(0.0, lam_i + lr_lam * (cost_ema - budget))
    return lam_i, max(0.0, lam_i + kp * (cost_ema - budget))


# --------------------------------------------------------------------- AWR
def train_awr(episodes: list[Episode], actions: list[str], lam: float = 0.5, beta: float = 0.25,
              epochs: int = 300, lr: float = 2.0, use_counterfactual: bool = False,
              init: Controller | None = None, serve_eps: float = 0.05, log: list | None = None,
              first_step_only: bool = False, legal: list[str] | None = None,
              n_features: int | None = None) -> Controller:
    """AWR + IPS on a logged trace. `first_step_only`: one record per episode (its step-0
    decision, return = quality - lam * episode cost), for logs whose later steps are a fixed
    continuation (CRMArena counterfactual). `legal` then overrides the logged legal set (the
    behavior's support), and the value baseline is the CRM state bucket (category -> org
    fallback) instead of the sim bucket."""
    if init is None and n_features is not None:
        pol = Controller(actions, n_features)
        pol.eps = serve_eps
    else:
        pol = _new_policy(actions, init, serve_eps)
    X, M, A, MU, R, KEY = [], [], [], [], [], []
    for ep in episodes:
        if first_step_only:
            s = ep.steps[0]
            X.append(s.features); M.append(pol.mask(legal or s.legal)); A.append(pol.index(s.action_name))
            MU.append(s.propensity); R.append(ep.quality - lam * ep.total_cost); KEY.append(crm_context(s.features))
            continue
        ctg = np.cumsum([s.cost for s in ep.steps][::-1])[::-1]
        for s, c_rem in zip(ep.steps, ctg):
            m = pol.mask(s.legal)
            X.append(s.features); M.append(m); A.append(pol.index(s.action_name))
            MU.append(s.propensity); R.append(ep.quality - lam * c_rem); KEY.append(bucket_key(s.features))
            if use_counterfactual and s.counterfactual:
                for a, (c, q) in s.counterfactual.items():
                    if parse_action(a)[0] in TERMINAL and a != s.action_name:
                        X.append(s.features); M.append(m); A.append(pol.index(a))
                        MU.append(1.0 / m.sum()); R.append(q - lam * c); KEY.append(bucket_key(s.features))
    X, M, A, MU, R = map(np.asarray, (X, M, A, MU, R))
    if first_step_only:
        bk = CRMBuckets().fit([k[0] for k in KEY], [k[1] for k in KEY], [None] * len(KEY), R)
        base = np.array([bk.predict(c, o, None) for c, o in KEY])
    else:
        sums, cnts = defaultdict(float), defaultdict(int)
        for k, r in zip(KEY, R):
            sums[k] += r; cnts[k] += 1
        base = np.array([sums[k] / cnts[k] for k in KEY])
    w = np.exp(np.clip((R - base) / beta, -5, 5)) / MU      # IPS-corrected AWR weight
    w /= w.mean()
    n = len(R)
    for e in range(epochs):                                 # full-batch: IPS weights are high-variance
        pol.W += lr * pol.grad_logits_batch(X, M, A, w) / n
        if log is not None and (e % 50 == 0 or e == epochs - 1):
            log.append({"epoch": e, "w_max": float(w.max()), "n": int(n)})
    pol.meta.update({"method": "awr", "lambda": lam, "beta": beta, "epochs": epochs,
                     "counterfactual": use_counterfactual, "n_records": int(n), "first_step_only": first_step_only})
    return pol


# ------------------------------------------------------ CGRPO, lookup form
def train_cgrpo_lookup(table, budget: float, iters: int = 300, batch: int = 32, group: int = 8,
                       lr: float = 1.5, lr_lam: float = 20.0, kp: float = 40.0, lam0: float = 0.0,
                       ema: float = 0.9, snap_every: int = 5, serve_eps: float = 0.0, seed: int = 0,
                       init: Controller | None = None, log: list | None = None) -> tuple[Controller, float]:
    """train_cgrpo on a counterfactual lookup table (pace.lookup.LookupTable): a re-roll from a
    logged first-step state is a lookup of the logged (quality, cost) of the sampled tier.
    Group-mean baseline on q - lam * c, lam by PI-damped dual ascent on an EMA of the batch
    cost against `budget`, iterate averaging over the second half (served policy = snapshot mix).
    lr_lam / kp are per dollar of cost gap (CRMArena costs are USD, ~0.05-0.2 per episode;
    the sim defaults 0.10 / 2.0 assume costs ~0.1-0.6 in sim units and lam ~0.4).
    Returns (policy, lam_bar) with lam_bar the mean lam over the averaged half."""
    from .lookup import DECISION_ACTIONS
    rng = np.random.default_rng(seed)
    if init is not None:
        pol = _new_policy(DECISION_ACTIONS, init, serve_eps)
    else:
        pol = Controller(DECISION_ACTIONS, table.X.shape[1], eps=serve_eps)
    pol.mix = None
    n = len(table)
    m = np.ones(len(DECISION_ACTIONS), bool)
    lam_i, lam, cost_ema = lam0, lam0, None
    lam_sum, n_avg = 0.0, 0
    snaps: list[np.ndarray] = []
    for it in range(iters):
        ix = rng.choice(n, size=min(batch, n), replace=False)
        X = table.X[ix]
        Mb = np.tile(m, (len(ix), 1))
        P = pol.probs_batch(X, Mb, eps=serve_eps, use_mix=False)        # the policy as served
        u = rng.random((len(ix), group, 1))
        A = (u > np.cumsum(P, 1)[:, None, :]).sum(-1).clip(0, len(DECISION_ACTIONS) - 1)    # (b, group)
        q = table.Q[ix[:, None], A]; c = table.C[ix[:, None], A]
        r = q - lam * c
        adv = (r - r.mean(1, keepdims=True)).ravel()
        Xg, Mg, Ag = np.repeat(X, group, axis=0), np.repeat(Mb, group, axis=0), A.ravel()
        g = pol.grad_logits_batch(Xg, Mg, Ag, adv * pol.eps_score_weights(Xg, Mg, Ag, serve_eps))
        pol.W += lr * g / len(Xg)
        c_batch = float(c.mean())
        cost_ema = c_batch if cost_ema is None else ema * cost_ema + (1 - ema) * c_batch
        lam_i, lam = _dual_step(lam_i, cost_ema, budget, lr_lam, kp)
        if it >= iters // 2:
            lam_sum += lam; n_avg += 1
            if (it - iters // 2) % snap_every == 0:
                snaps.append(pol.W.copy())
        if log is not None:
            log.append({"iter": it, "lambda": round(lam, 4), "cost_ema": round(cost_ema, 5),
                        "cost_batch": round(c_batch, 5), "quality_batch": round(float(q.mean()), 4)})
    pol.mix = snaps
    lam_bar = lam_sum / max(n_avg, 1)
    pol.meta.update({"method": "cgrpo_lookup", "budget": budget, "lambda": lam_bar, "lambda_final": lam,
                     "iters": iters, "group": group, "batch": batch, "n_snapshots": len(snaps),
                     "n_tasks": n, "lr_lam": lr_lam, "kp": kp, "serve_eps": serve_eps})
    return pol, lam_bar


# ------------------------------------------------------------------- CGRPO
def train_cgrpo(episodes: list[Episode], harness: Harness, budget: float, iters: int = 300, batch: int = 32,
                group: int = 8, lr: float = 1.5, lr_lam: float = 0.10, kp: float = 2.0, lam0: float = 0.0,
                ema: float = 0.9, snap_every: int = 5, init: Controller | None = None,
                serve_eps: float = 0.05, reward: str = "realized", seed: int = 0,
                log: list | None = None, restart_mix: float = 0.5,
                max_transitions: int | None = None) -> tuple[Controller, float]:
    """Constrained GRPO on replay from logged state snapshots (sim harness only).
    Each iteration re-rolls `group` continuations from `batch` logged states: floor((1 -
    restart_mix) * batch) episode starts (step 0; their full-episode cost feeds the dual variable)
    and the rest drawn from ALL logged states (interior restarts, which may also land on step 0).
    restart_mix = 0.5 is the historical default; 0.0 trains on episode starts only, i.e. on the
    episode-start objective E_{s0}[V(s0)] rather than a restart-distribution objective (see
    results/sim_seeds/restart_ablation.md). `reward` = "realized" uses the 0/1 outcome,
    "expected" the harness's success probability (lower variance; sim only).

    Every simulator step consumed is counted (meta "transitions"). With `max_transitions`, training
    stops once that many transitions are consumed (`iters` is then only an upper bound) and the
    iterate average covers the iterations after half the transition budget was spent, so two runs
    can be compared at a matched simulator budget.
    Returns (policy, lam_bar) with lam_bar the mean lam over the averaged half."""
    if not 0.0 <= restart_mix <= 1.0:
        raise ValueError(f"restart_mix must be in [0, 1], got {restart_mix}")
    rng = random.Random(seed)
    pol = _new_policy(harness.all_actions, init, serve_eps)
    starts = [s.state for ep in episodes for s in ep.steps if s.state is not None]
    if not starts:
        raise ValueError("cgrpo needs state snapshots in the traces (sim harness only)")
    starts0 = [s for s in starts if s["step"] == 0]
    lam_i, lam, cost_ema = lam0, lam0, None
    lam_sum, n_avg = 0.0, 0
    pol.mix = None
    snaps: list[np.ndarray] = []
    n_start = int(np.floor((1.0 - restart_mix) * batch))            # restart_mix=0.5 -> batch // 2, as before
    transitions, it_done, avg_from = 0, 0, None
    for it in range(iters):
        if max_transitions is not None and transitions >= max_transitions:
            break
        picks = rng.sample(starts0, n_start) + rng.sample(starts, batch - n_start)
        X, M, A, ADV, full_costs, quals = [], [], [], [], [], []
        for snap in picks:
            base = harness.restore(snap)
            rows, rs = [], []
            for _ in range(group):
                s, traj = base, []
                while not s.done:
                    legal = harness.legal_actions(s)
                    x = featurize(s)
                    a, _ = pol.act(x, legal, rng, use_mix=False)          # sampled as served (eps included)
                    traj.append((x, pol.mask(legal), a))
                    s, _, _, _ = harness.step(s, pol.actions[a])
                    transitions += 1
                q = harness.expected_quality(s) if reward == "expected" else harness.quality(s)
                c_rem = s.cost_so_far - base.cost_so_far
                rs.append(q - lam * c_rem)
                rows.append(traj)
                quals.append(q)
                if base.step == 0:
                    full_costs.append(s.cost_so_far)
            b = sum(rs) / group
            for traj, r in zip(rows, rs):
                for x, m, a in traj:
                    X.append(x); M.append(m); A.append(a); ADV.append(r - b)
        X, M, A, ADV = np.asarray(X), np.asarray(M), np.asarray(A), np.asarray(ADV)
        g = pol.grad_logits_batch(X, M, A, ADV * pol.eps_score_weights(X, M, A, serve_eps))
        pol.W += lr * g / (batch * group)
        c_batch = float(np.mean(full_costs)) if full_costs else float("nan")
        if full_costs:                    # restart_mix = 1 can draw no episode start: keep the dual state
            cost_ema = c_batch if cost_ema is None else ema * cost_ema + (1 - ema) * c_batch
            lam_i, lam = _dual_step(lam_i, cost_ema, budget, lr_lam, kp)
        it_done = it + 1
        if max_transitions is None:
            in_avg = it >= iters // 2
            k = it - iters // 2
        else:                                                         # second half of the transition budget
            if avg_from is None and transitions >= max_transitions / 2:
                avg_from = it
            in_avg = avg_from is not None
            k = 0 if avg_from is None else it - avg_from
        if in_avg:                                                    # iterate averaging
            lam_sum += lam; n_avg += 1
            if k % snap_every == 0:
                snaps.append(pol.W.copy())
        if log is not None:
            log.append({"iter": it, "lambda": round(lam, 4),
                        "cost_ema": None if cost_ema is None else round(cost_ema, 4),
                        "cost_batch": None if not full_costs else round(c_batch, 4),
                        "quality_batch": round(float(np.mean(quals)), 4), "transitions": transitions})
    pol.mix = snaps
    lam_bar = lam_sum / max(n_avg, 1)
    pol.meta.update({"method": "cgrpo", "budget": budget, "lambda": lam_bar, "lambda_final": lam,
                     "iters": it_done, "group": group, "batch": batch, "n_snapshots": len(snaps),
                     "restart_mix": restart_mix, "n_start_per_batch": n_start, "transitions": transitions,
                     "max_transitions": max_transitions,
                     "serve_eps": serve_eps, "harness_version": harness.version,
                     "price_factor": dict(harness.price_factor)})
    return pol, lam_bar


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Train a PACE controller offline from a trace file: constrained GRPO on replay "
                    "(cgrpo, learned lambda against --budget) or AWR+IPS (awr, fixed --lam). "
                    "CRMArena logs are trained as a first-step tier bandit (cgrpo: exact lookups).")
    ap.add_argument("--traces", required=True, help="JSONL of Episodes written by pace.collect")
    ap.add_argument("--method", choices=["awr", "cgrpo"], default="cgrpo")
    ap.add_argument("--budget", type=float, default=0.30,
                    help="cgrpo: cost budget per task (sim units; USD on CRMArena logs)")
    ap.add_argument("--lam", type=float, default=0.5, help="awr: fixed price of cost")
    ap.add_argument("--init", default=None, help="warm-start policy JSON")
    ap.add_argument("--reprice", nargs="*", default=[],
                    help="tier:factor events. sim: the repricing the traces were logged under (passed to the "
                         "replay harness); crmarena: applied to the log, with costs recomputed from the spend "
                         "ledger's token counts")
    ap.add_argument("--spend", default="traces/spend.jsonl", help="crmarena: spend ledger for --reprice")
    ap.add_argument("--iters", type=int, default=300, help="cgrpo: optimizer iterations")
    ap.add_argument("--epochs", type=int, default=300, help="awr: full-batch epochs")
    ap.add_argument("--use-counterfactual", action="store_true",
                    help="awr (sim): add the logged counterfactual terminal arms as extra records")
    ap.add_argument("--serve-eps", type=float, default=0.05,
                    help="eps exploration mass of the SERVED policy (part of what is trained and gated)")
    ap.add_argument("--reward", choices=["realized", "expected"], default="realized",
                    help="cgrpo (sim): 0/1 outcome or the harness's expected success probability")
    ap.add_argument("--restart-mix", type=float, default=0.5,
                    help="cgrpo (sim): fraction of each batch restarted from any logged state (rest: episode starts)")
    ap.add_argument("--max-transitions", type=int, default=None,
                    help="cgrpo (sim): stop after this many simulator transitions (matched-budget comparisons)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="policy JSON to write")
    ap.add_argument("--log", default=None, help="training log JSONL (default: <out>.log.jsonl)")
    a = ap.parse_args(argv)
    t0 = time.time()
    eps_ = read_jsonl(a.traces)
    init = Controller.load(a.init) if a.init else None
    version = eps_[0].harness_version
    log: list = []
    if version.startswith("crmarena"):
        from .lookup import DECISION_ACTIONS, LookupTable, reprice_episodes
        if a.reprice:
            eps_ = reprice_episodes(eps_, a.spend, dict((k, float(v)) for k, v in (r.split(":") for r in a.reprice)))
        if a.method == "awr":
            pol = train_awr(eps_, DECISION_ACTIONS, lam=a.lam, epochs=a.epochs, serve_eps=a.serve_eps, log=log,
                            first_step_only=True, legal=DECISION_ACTIONS, n_features=len(eps_[0].steps[0].features))
            msg = f"first-step tier, lambda(fixed)={a.lam:.3f}/USD"
        else:
            table = LookupTable.from_episodes(eps_)
            pol, lam = train_cgrpo_lookup(table, a.budget, iters=a.iters, serve_eps=a.serve_eps, seed=a.seed, log=log)
            msg = (f"lookup over {len(table)} tasks, budget=${a.budget:.3f} learned lambda={lam:.3f}/USD "
                   f"final cost_ema={log[-1]['cost_ema']:.4f}")
    elif a.method == "awr":
        h = make_sim(version, reprice=parse_reprice(a.reprice)) if version in ("v1", "v2") else None
        actions = h.all_actions if h else sorted({s.action_name for e in eps_ for s in e.steps})
        pol = train_awr(eps_, actions, lam=a.lam, epochs=a.epochs, use_counterfactual=a.use_counterfactual,
                        init=init, serve_eps=a.serve_eps, log=log)
        msg = f"lambda(fixed)={a.lam:.3f}"
    else:   # cgrpo on the simulator
        h = make_sim(version, seed=a.seed + 101, reprice=parse_reprice(a.reprice))
        pol, lam = train_cgrpo(eps_, h, a.budget, iters=a.iters, init=init, serve_eps=a.serve_eps,
                               reward=a.reward, seed=a.seed, log=log, restart_mix=a.restart_mix,
                               max_transitions=a.max_transitions)
        msg = f"budget={a.budget:.2f} learned lambda={lam:.3f} final cost_ema={log[-1]['cost_ema']:.3f}"
    pol.meta["traces"] = a.traces
    pol.save(a.out)
    log_path = Path(a.log or (str(a.out) + ".log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(json.dumps(r) + "\n" for r in log))
    print(f"[train] {a.method} on {len(eps_)} episodes ({version}): {msg}  "
          f"({time.time() - t0:.1f}s) -> {a.out}")


if __name__ == "__main__":
    main()
