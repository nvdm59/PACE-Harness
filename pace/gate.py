"""Off-policy promotion gate: score a candidate on LOGGED traffic, then PROMOTE or REVERT.

Three per-decision estimators, for both quality (terminal reward) and cost (per-step reward),
on a log whose every decision carries the behavior propensity mu (including exploration):

  IPS    mean_i sum_t rho_{0:t} r_t           per-decision importance sampling (Precup, Sutton
                                                & Singh 2000); unbiased, high variance
  SNIPS  per-step self-normalized IPS           sum_i rho_{i,0:t} r_{i,t} / sum_i rho_{i,0:t},
                                                summed over t (Swaminathan & Joachims 2015);
                                                biased, lower variance
  DR     V(s0) + sum_t rho_{0:t} (r_t + V(s_{t+1}) - Q(s_t, a_t))
                                                doubly robust for sequential decisions (Jiang &
                                                Li 2016, arXiv:1511.03722; the bandit form is
                                                Dudik, Langford & Li 2011, arXiv:1103.4601)
The direct model Q is a per-feature-bucket mean fitted from the log (buckets pool over the step
index), iterated under the candidate: bucketed fitted-Q evaluation (Le, Voloshin & Yue 2019,
arXiv:1903.08738), T+1 sweeps. Cumulative ratios are clipped at --clip; the clip biases DR only
through the direct-model residual, which is small when the direct model is good. The direct
model is fitted on the same log it corrects (no cross-fitting), the usual choice at this scale.

90% CIs are percentile bootstraps over EPISODES (200 resamples by default); every resample refits
the direct model, so the DR interval includes direct-model variance. Decision:
  PROMOTE iff DR-quality lower CI >= q_prev - eps  AND  DR-cost upper CI <= budget + tol
where q_prev is the incumbent's quality: the on-policy logged mean if the incumbent is the
behavior policy, else the incumbent's own DR estimate, or, with --q-prev-from, the DR quality
recorded when the incumbent was itself promoted (use this after a repricing that pushed the
incumbent over budget: otherwise an over-budget incumbent sets a bar no budget-respecting
candidate can pass).

CRMArena-Pro logs (harness version crmarena-pro-*) are a contextual bandit at the first step:
the tier is chosen with propensity 1/3 and the continuation is fixed (propensity 1).
`estimate_bandit` scores a tier policy pi(tier | x0) there:
  IPS    mean_i w_i r_i,  w_i = pi(a_i|x_i) / mu_i
  SNIPS  sum_i w_i r_i / sum_i w_i
  DR     mean_i [ sum_a pi(a|x_i) Qhat(x_i, a) + w_i (r_i - Qhat(x_i, a_i)) ]
with Qhat the CRM bucket mean (features.CRMBuckets: (category, tier), falling back to
(org, tier) below 3 samples). The bootstrap resamples TASKS (all logged arms of a task
together) and refits Qhat. `estimate` / `gate` dispatch to it automatically.

CLI:
  python -m pace.gate --traces traces/v1.jsonl --candidate policies/p.json --incumbent behavior --budget 0.30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .controller import Controller
from .features import CRMBuckets, bucket_key, crm_context
from .trace import Episode, read_jsonl

T_MAX = 4


def _key_id(k: tuple, pool_steps: bool) -> int:
    hard, ev, ver, step, v2 = k
    st = 0 if pool_steps else step
    return (((hard * 3 + ev) * 2 + ver) * 4 + st) * 2 + v2


N_KEYS = 2 * 3 * 2 * 4 * 2


class _Log:
    """Episodes flattened into padded (n, T) arrays."""

    def __init__(self, episodes: list[Episode], policy: Controller, pool_steps: bool = True):
        policy.ensure_actions(sorted({a for e in episodes for s in e.steps for a in (s.legal or [s.action_name])}))
        n = len(episodes)
        T = max(T_MAX, max(len(e.steps) for e in episodes))
        self.n, self.T, self.nA = n, T, len(policy.actions)
        self.valid = np.zeros((n, T), bool)
        self.mu = np.ones((n, T))
        self.rq = np.zeros((n, T))
        self.rc = np.zeros((n, T))
        self.a = np.zeros((n, T), int)
        self.kid = np.zeros((n, T), int)
        X, M, idx = [], [], []
        for i, e in enumerate(episodes):
            for t, s in enumerate(e.steps):
                self.valid[i, t] = True
                self.mu[i, t] = s.propensity
                self.rc[i, t] = s.cost
                self.a[i, t] = policy.index(s.action_name)
                self.kid[i, t] = _key_id(bucket_key(s.features), pool_steps)
                X.append(s.features); M.append(policy.mask(s.legal)); idx.append((i, t))
            self.rq[i, len(e.steps) - 1] = e.quality
        self.X, self.M = np.asarray(X), np.asarray(M)
        self.idx = np.asarray(idx)
        self.quality = np.array([e.quality for e in episodes])
        self.cost = np.array([e.total_cost for e in episodes])

    def target_probs(self, policy: Controller) -> np.ndarray:
        """(n, T, A) served probabilities of `policy` at every logged decision."""
        P = np.zeros((self.n, self.T, self.nA))
        P[self.idx[:, 0], self.idx[:, 1]] = policy.probs_batch(self.X, self.M)
        return P


def _fit_dm(valid: np.ndarray, kid: np.ndarray, a: np.ndarray, P: np.ndarray, r: np.ndarray,
            nA: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucketed fitted-Q evaluation of the candidate (iterated to the horizon).
    Q(bucket, a) = mean over logged (bucket, a) transitions of r + V(s');
    V(s) = sum_a pi(a|s) Q(bucket(s), a). Unseen (bucket, a): bucket mean, else global mean.
    Returns Qsa (n,T) = Q(s_t,a_t) and V (n,T) = V(s_t)."""
    flat = (kid * nA + a)[valid]
    cnt = np.bincount(flat, minlength=N_KEYS * nA).reshape(N_KEYS, nA)
    table = np.zeros((N_KEYS, nA))
    for _ in range(T + 1):
        V = np.where(valid, (P * table[kid]).sum(-1), 0.0)
        Vnext = np.zeros_like(V)
        Vnext[:, :-1] = V[:, 1:]
        target = (r + Vnext)[valid]
        sums = np.bincount(flat, weights=target, minlength=N_KEYS * nA).reshape(N_KEYS, nA)
        ksum, kcnt = sums.sum(1, keepdims=True), cnt.sum(1, keepdims=True)
        kmean = np.where(kcnt > 0, ksum / np.maximum(kcnt, 1), target.mean())
        table = np.where(cnt > 0, sums / np.maximum(cnt, 1), kmean)
    V = np.where(valid, (P * table[kid]).sum(-1), 0.0)
    Qsa = np.where(valid, np.take_along_axis(table[kid], a[:, :, None], 2)[:, :, 0], 0.0)
    return Qsa, V


def _all_estimates(valid, kid, a, P, rho, r, nA, T) -> tuple[float, float, float]:
    """(IPS, per-decision SNIPS, DR) point estimates on one (possibly resampled) log.
    `rho` is the clipped cumulative ratio, carried forward past an episode's end (an episode
    that ended before step t contributes r_t = 0 with its final ratio, which is the correct
    weight for the per-step self-normalizer)."""
    ips = float((rho * r * valid).sum(1).mean())
    num, den = (rho * r * valid).sum(0), rho.sum(0)
    snips = float(np.sum(np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)))
    Qsa, V = _fit_dm(valid, kid, a, P, r, nA, T)
    Vnext = np.zeros_like(V)
    Vnext[:, :-1] = V[:, 1:]
    dr = float((V[:, 0] + (rho * (r + Vnext - Qsa) * valid).sum(1)).mean())
    return ips, snips, dr


def _ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 5)), float(np.percentile(samples, 95))


def estimate(episodes: list[Episode], policy: Controller, boot: int = 200, clip: float = 50.0,
             seed: int = 0, pool_steps: bool = True, cross_fit: bool = True) -> dict:
    """{'quality': {'ips':(pt,lo,hi), 'snips':..., 'dr':...}, 'cost': {...}, 'n', 'max_rho'}.
    Bootstrap resamples episodes and REFITS the direct model, so DR CIs include DM variance.
    CRMArena logs go to `estimate_bandit` (`cross_fit` applies there only; the sim direct model
    is fitted in-sample, where buckets hold hundreds of transitions and the in-sample bias is
    negligible)."""
    if is_bandit_log(episodes):
        return estimate_bandit(episodes, policy, boot, seed, cross_fit=cross_fit)
    policy = policy.copy()
    log = _Log(episodes, policy, pool_steps)
    P = log.target_probs(policy)
    pi_a = np.take_along_axis(P, log.a[:, :, None], axis=2)[:, :, 0]
    ratio = np.where(log.valid, pi_a / log.mu, 1.0)
    raw_rho = np.cumprod(ratio, axis=1)                     # carried forward after episode end
    rho = np.minimum(raw_rho, clip)
    rng = np.random.default_rng(seed)
    B = rng.integers(0, log.n, size=(boot, log.n))
    out = {}
    for name, r in (("quality", log.rq), ("cost", log.rc)):
        args = lambda ix: (log.valid[ix], log.kid[ix], log.a[ix], P[ix], rho[ix], r[ix], log.nA, log.T)
        pt = _all_estimates(*args(slice(None)))
        bs = np.array([_all_estimates(*args(b)) for b in B])
        out[name] = {est: (pt[j],) + _ci(bs[:, j]) for j, est in enumerate(("ips", "snips", "dr"))}
    out["n"] = log.n
    out["max_rho"] = float(raw_rho.max())
    return out


def is_bandit_log(episodes: list[Episode]) -> bool:
    return bool(episodes) and episodes[0].harness_version.startswith("crmarena")


def _bandit_arrays(episodes: list[Episode], policy: Controller):
    from .lookup import DECISION_ACTIONS
    pol = policy.copy()
    pol.ensure_actions(DECISION_ACTIONS)
    X = np.array([e.steps[0].features for e in episodes], float)
    M = np.tile(pol.mask(DECISION_ACTIONS), (len(episodes), 1))
    P = pol.probs_batch(X, M)[:, [pol.index(a) for a in DECISION_ACTIONS]]
    a = np.array([DECISION_ACTIONS.index(e.steps[0].action_name) for e in episodes])
    # episode propensity: product over steps (1/3 at step 0, 1.0 under the fixed continuation)
    mu = np.array([float(np.prod([s.propensity for s in e.steps])) for e in episodes])
    ctx = [crm_context(list(x)) for x in X]
    return X, P, a, mu, ctx


def _loo_qhat(cats: list[str], orgs: list[str], tiers: list[str], r: np.ndarray, groups: np.ndarray,
              min_n: int) -> np.ndarray:
    """(n, |TIERS|) cross-fitted direct model: row i's Qhat is the CRMBuckets prediction fitted on
    every row whose group (task id) differs from row i's, with the same (category, tier) ->
    (org, tier) -> tier -> global fallback. Vectorized: per-group bucket totals are subtracted
    from the global totals."""
    from .prices import TIERS
    cat_v = sorted(set(cats)); org_v = sorted(set(orgs))
    ci = np.array([cat_v.index(c) for c in cats]); oi = np.array([org_v.index(o) for o in orgs])
    ti = np.array([TIERS.index(t) for t in tiers])
    g_v, gi = np.unique(groups, return_inverse=True)
    n, G, nT = len(r), len(g_v), len(TIERS)

    def totals(*idx, shape):
        S = np.zeros(shape); N = np.zeros(shape)
        np.add.at(S, idx, r); np.add.at(N, idx, 1.0)
        return S, N
    S_ct, N_ct = totals(ci, ti, shape=(len(cat_v), nT))
    S_ot, N_ot = totals(oi, ti, shape=(len(org_v), nT))
    S_t, N_t = totals(ti, shape=(nT,))
    gS_ct, gN_ct = totals(gi, ci, ti, shape=(G, len(cat_v), nT))
    gS_ot, gN_ot = totals(gi, oi, ti, shape=(G, len(org_v), nT))
    gS_t, gN_t = totals(gi, ti, shape=(G, nT))
    gS, gN = totals(gi, shape=(G,))
    # leave-one-group-out tables evaluated at each row's own (category, org): (n, nT)
    s_ct = S_ct[ci] - gS_ct[gi, ci]; n_ct = N_ct[ci] - gN_ct[gi, ci]
    s_ot = S_ot[oi] - gS_ot[gi, oi]; n_ot = N_ot[oi] - gN_ot[gi, oi]
    s_t = S_t[None, :] - gS_t[gi]; n_t = N_t[None, :] - gN_t[gi]
    glob = (r.sum() - gS[gi]) / np.maximum(n - gN[gi], 1.0)
    out = np.where(n_t > 0, s_t / np.maximum(n_t, 1.0), glob[:, None])
    out = np.where(n_ot >= min_n, s_ot / np.maximum(n_ot, 1.0), out)
    out = np.where(n_ct >= min_n, s_ct / np.maximum(n_ct, 1.0), out)
    return out


def _bandit_point(P, a, mu, ctx, r, min_n: int, tids: list[str] | None = None) -> tuple[float, float, float]:
    """One (IPS, SNIPS, DR) triple. With `tids`, the direct model is CROSS-FITTED: the Qhat used
    for a task's rows is fitted on every other task's rows (leave-one-task-out). An in-sample
    direct model shares the logged reward it is meant to correct, which shrinks the DR
    correction term toward zero and biases DR toward the direct model at small n (the
    cross-fitting argument of Chernozhukov et al. 2018, arXiv:1608.00060); with one logged arm
    per task and tier buckets of a few samples each, that bias is not negligible."""
    from .prices import TIERS
    n = len(r)
    w = P[np.arange(n), a] / mu
    ips = float((w * r).mean())
    snips = float((w * r).sum() / max(w.sum(), 1e-12))
    cats, orgs, tiers = [c for c, _ in ctx], [o for _, o in ctx], [TIERS[j] for j in a]
    if tids is None:
        dm = CRMBuckets(min_n).fit(cats, orgs, tiers, r)
        Qh = np.array([[dm.predict(c, o, t) for t in TIERS] for c, o in ctx])
    else:
        Qh = _loo_qhat(cats, orgs, tiers, np.asarray(r, float), np.asarray(tids), min_n)
    dr = float(((P * Qh).sum(1) + w * (r - Qh[np.arange(n), a])).mean())
    return ips, snips, dr


def estimate_bandit(episodes: list[Episode], policy: Controller, boot: int = 200, seed: int = 0,
                    min_n: int = 3, cross_fit: bool = True) -> dict:
    """IPS / SNIPS / DR of a first-step tier policy on a CRMArena log, with 90% bootstrap CIs
    (resampling tasks, refitting the direct model). `cross_fit`: leave-one-task-out direct
    model (default); False fits it in-sample. Same output shape as `estimate`."""
    X, P, a, mu, ctx = _bandit_arrays(episodes, policy)
    tids = [e.task_id for e in episodes]
    uniq = sorted(set(tids))
    rows_of = {t: [i for i, x in enumerate(tids) if x == t] for t in uniq}
    out = {}
    for name in ("quality", "cost"):
        r = np.array([e.quality if name == "quality" else e.total_cost for e in episodes])
        pt = _bandit_point(P, a, mu, ctx, r, min_n, tids if cross_fit else None)
        bs = []
        brng = np.random.default_rng(seed)
        for _ in range(boot):
            draw = brng.integers(0, len(uniq), len(uniq))
            ix = np.concatenate([rows_of[uniq[k]] for k in draw])
            # a resampled task drawn twice is one task: leave out all of its copies together
            bid = np.concatenate([[f"{uniq[k]}"] * len(rows_of[uniq[k]]) for k in draw]) if cross_fit else None
            bs.append(_bandit_point(P[ix], a[ix], mu[ix], [ctx[i] for i in ix], r[ix], min_n, bid))
        bs = np.array(bs)
        out[name] = {est: (pt[j],) + _ci(bs[:, j]) for j, est in enumerate(("ips", "snips", "dr"))}
    out["n"] = len(episodes)
    out["n_tasks"] = len(uniq)
    out["max_rho"] = float((P[np.arange(len(a)), a] / mu).max())
    return out


def on_policy(episodes: list[Episode], boot: int = 200, seed: int = 0) -> dict:
    """On-policy log means of quality and cost with 90% bootstrap CIs (the behavior policy's value)."""
    q = np.array([e.quality for e in episodes]); c = np.array([e.total_cost for e in episodes])
    B = np.random.default_rng(seed).integers(0, len(q), size=(boot, len(q)))
    return {"quality": (float(q.mean()),) + _ci(q[B].mean(axis=1)),
            "cost": (float(c.mean()),) + _ci(c[B].mean(axis=1))}


def gate(episodes: list[Episode], candidate: Controller, incumbent: Controller | str, budget: float,
         eps_q: float = 0.03, tol: float = 0.02, boot: int = 200, clip: float = 50.0, seed: int = 0,
         q_prev: float | None = None) -> dict:
    """The promotion decision. `incumbent` is "behavior" (bar = on-policy log mean) or a Controller
    (bar = its DR quality on this log). `q_prev` overrides the bar (e.g. the incumbent's DR quality
    recorded when IT was promoted, within budget): after a repricing the incumbent may be over
    budget on the new log, and a bar set by an over-budget policy would block every
    budget-respecting candidate."""
    cand = estimate(episodes, candidate, boot, clip, seed)
    if isinstance(incumbent, str) and incumbent == "behavior":
        inc = on_policy(episodes, boot, seed)
        inc_desc = "behavior (on-policy log mean)"
    else:
        inc_full = estimate(episodes, incumbent, boot, clip, seed)
        inc = {"quality": inc_full["quality"]["dr"], "cost": inc_full["cost"]["dr"]}
        inc_desc = "incumbent (DR)"
    q_src = "incumbent on this log" if q_prev is None else "recorded at incumbent's promotion"
    q_prev = inc["quality"][0] if q_prev is None else float(q_prev)
    q_lo, c_hi = cand["quality"]["dr"][1], cand["cost"]["dr"][2]
    ok_q, ok_c = q_lo >= q_prev - eps_q, c_hi <= budget + tol
    return {"candidate": cand, "incumbent": inc, "incumbent_desc": inc_desc, "q_prev": q_prev, "q_prev_source": q_src,
            "budget": budget, "eps_q": eps_q, "tol": tol, "quality_ok": bool(ok_q), "cost_ok": bool(ok_c),
            "decision": "PROMOTE" if (ok_q and ok_c) else "REVERT"}


def format_table(res: dict, title: str = "Promotion gate", cost_digits: int = 3) -> str:
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    d = cost_digits
    fc = lambda t: f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"
    c = res["candidate"]
    lines = [f"### {title}", "",
             "| policy | estimator | quality [90% CI] | cost [90% CI] |", "|---|---|---|---|",
             f"| {res['incumbent_desc']} | - | {f(res['incumbent']['quality'])} | {fc(res['incumbent']['cost'])} |"]
    for est in ("ips", "snips", "dr"):
        lines.append(f"| candidate | {est.upper()} | {f(c['quality'][est])} | {fc(c['cost'][est])} |")
    lines += ["",
              f"criteria: DR-quality lower CI {c['quality']['dr'][1]:.3f} >= q_prev - eps = "
              f"{res['q_prev']:.3f} ({res['q_prev_source']}) - {res['eps_q']:.2f} = {res['q_prev'] - res['eps_q']:.3f} "
              f"[{'ok' if res['quality_ok'] else 'FAIL'}];  DR-cost upper CI {c['cost']['dr'][2]:.{d}f} <= "
              f"budget + tol = {res['budget'] + res['tol']:.{d}f} [{'ok' if res['cost_ok'] else 'FAIL'}]",
              f"(n = {c['n']} logged episodes"
              + (f" over {c['n_tasks']} tasks" if "n_tasks" in c else "")
              + f", max cumulative ratio {c['max_rho']:.1f})", "",
              f"**DECISION: {res['decision']}**"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Off-policy promotion gate: IPS / SNIPS / DR with 90% bootstrap CIs on a logged "
                    "trace; PROMOTE iff DR-quality lower CI >= q_prev - eps and DR-cost upper CI <= budget + tol.")
    ap.add_argument("--traces", required=True, help="JSONL of Episodes with logged propensities")
    ap.add_argument("--candidate", required=True, help="candidate policy JSON")
    ap.add_argument("--incumbent", default="behavior", help="'behavior' (on-policy log mean) or a policy JSON")
    ap.add_argument("--budget", type=float, required=True, help="cost budget per task")
    ap.add_argument("--eps", type=float, default=0.03, help="allowed quality regression below q_prev")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="cost tolerance over budget (sim cost units; on CRMArena logs costs are USD, e.g. 0.005)")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--clip", type=float, default=50.0, help="cumulative importance-ratio clip (sim logs)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--q-prev", type=float, default=None, help="explicit quality bar")
    ap.add_argument("--q-prev-from", default=None,
                    help="gate JSON that promoted the incumbent; its candidate DR quality becomes the bar")
    ap.add_argument("--title", default="Promotion gate")
    ap.add_argument("--json-out", default=None, help="write the full result as JSON")
    a = ap.parse_args(argv)
    eps_ = read_jsonl(a.traces)
    cand = Controller.load(a.candidate)
    inc = "behavior" if a.incumbent == "behavior" else Controller.load(a.incumbent)
    q_prev = a.q_prev
    if a.q_prev_from:
        q_prev = json.loads(Path(a.q_prev_from).read_text())["candidate"]["quality"]["dr"][0]
    res = gate(eps_, cand, inc, a.budget, a.eps, a.tol, a.boot, a.clip, a.seed, q_prev=q_prev)
    print(format_table(res, a.title, cost_digits=4 if is_bandit_log(eps_) else 3))
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
