"""Refinement policies: how much Socratic self-refinement to buy for this request.

Every policy maps an `SSRState` (category, round index, cost so far, the probe's confidence
signals, what the last round did) to a distribution over {STOP, REFINE@cheap, REFINE@mid, ...}.

  never                  stop at the initial answer; pays nothing beyond the initial episode
  always_1[:tier]        one round on `tier` for every task
  always_2[:tier]        two rounds on `tier`
  uncertainty_gated[:tau[:tier]]
                         pay the probe; refine on `tier` iff the minimum self-reported claim
                         confidence is below tau; repeat at the next node while rounds remain
  controller:<json>      a PACE softmax-linear controller (pace.controller.Controller) over the
                         features below, trained on a logged tree by exact policy gradient with
                         a learned price of cost (dual ascent against a per-task budget)

`needs_probe` says whether the policy consults the decomposition before deciding; the evaluator
charges the probe at every visited node for such policies (and at every refined node for all).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...controller import Controller
from ...harness.crmarena import CATEGORIES
from ...prices import TIERS
from .tree import STOP, SSRState, Tree, action_tier, refine_action, state_of

FEATURE_NAMES = (["bias"] + [f"cat={c}" for c in CATEGORIES] + ["round=0", "round=1", "round>=2",
                 "probe_min", "probe_mean", "answer_conf", "prev_conf_min", "has_prev", "changed_last",
                 "cost_so_far*10"] + [f"arm={t}" for t in TIERS])
N_FEATURES = len(FEATURE_NAMES)


def featurize(s: SSRState) -> list[float]:
    x = [0.0] * N_FEATURES
    x[0] = 1.0
    if s.category in CATEGORIES:
        x[1 + CATEGORIES.index(s.category)] = 1.0
    o = 1 + len(CATEGORIES)
    x[o + min(s.round, 2)] = 1.0
    o += 3
    x[o] = 0.5 if s.probe_min is None else float(s.probe_min)
    x[o + 1] = 0.5 if s.probe_mean is None else float(s.probe_mean)
    x[o + 2] = 0.5 if s.answer_conf is None else float(s.answer_conf)
    x[o + 3] = 1.0 if s.prev_conf_min is None else float(s.prev_conf_min)
    x[o + 4] = 0.0 if s.prev_conf_min is None else 1.0
    x[o + 5] = 1.0 if s.changed_last else 0.0
    x[o + 6] = 10.0 * float(s.cost_so_far)
    o += 7
    if s.initial_arm in TIERS:
        x[o + TIERS.index(s.initial_arm)] = 1.0
    return x


# --------------------------------------------------------------- policies
class Policy:
    name = "policy"
    needs_probe = False

    def probs(self, s: SSRState) -> dict[str, float]:
        raise NotImplementedError

    def decide(self, s: SSRState) -> str:
        return max(self.probs(s).items(), key=lambda kv: kv[1])[0]


class Never(Policy):
    name = "never"

    def probs(self, s):
        return {STOP: 1.0}


class Always(Policy):
    def __init__(self, rounds: int, tier: str = "cheap"):
        self.rounds, self.tier = int(rounds), tier
        self.name = f"always_{rounds}:{tier}"

    def probs(self, s):
        a = refine_action(self.tier)
        return {a: 1.0} if (s.round < self.rounds and a in s.legal) else {STOP: 1.0}


class UncertaintyGated(Policy):
    needs_probe = True

    def __init__(self, tau: float, tier: str = "cheap", max_rounds: int = 2):
        self.tau, self.tier, self.max_rounds = float(tau), tier, int(max_rounds)
        self.name = f"uncertainty_gated:{self.tau:g}:{tier}"

    def probs(self, s):
        a = refine_action(self.tier)
        if s.round < self.max_rounds and a in s.legal and s.probe_min is not None and s.probe_min < self.tau:
            return {a: 1.0}
        return {STOP: 1.0}


class ControllerPolicy(Policy):
    needs_probe = True

    def __init__(self, ctrl: Controller, name: str = "controller", greedy: bool = False):
        self.ctrl, self.name, self.greedy = ctrl, name, greedy

    @classmethod
    def load(cls, path: str | Path, greedy: bool = False) -> "ControllerPolicy":
        c = Controller.load(path)
        if c.n_features != N_FEATURES:
            raise ValueError(f"{path}: {c.n_features} features, SSR policies use {N_FEATURES}")
        return cls(c, f"controller:{Path(path).name}", greedy)

    def probs(self, s):
        legal = [a for a in s.legal if a in self.ctrl.actions]
        p = self.ctrl.probs(featurize(s), legal)
        out = {a: float(p[self.ctrl.index(a)]) for a in legal}
        if self.greedy:
            best = max(out.items(), key=lambda kv: kv[1])[0]
            return {best: 1.0}
        return out


def parse_policy(spec: str, max_rounds: int = 2) -> Policy:
    parts = spec.split(":")
    head = parts[0]
    if head == "never":
        return Never()
    if head.startswith("always_"):
        n = int(head[len("always_"):])
        return Always(n, parts[1] if len(parts) > 1 else "cheap")
    if head == "uncertainty_gated":
        tau = float(parts[1]) if len(parts) > 1 else 0.7
        return UncertaintyGated(tau, parts[2] if len(parts) > 2 else "cheap", max_rounds)
    if head == "controller":
        if len(parts) < 2:
            raise ValueError("controller:<policy.json>")
        return ControllerPolicy.load(":".join(parts[1:]))
    raise ValueError(f"unknown policy {spec!r}")


# ------------------------------------------------------- controller fit
def actions_for(tiers: list[str]) -> list[str]:
    return [STOP] + [refine_action(t) for t in tiers]


def _leaf_paths(tree: Tree, tiers: list[str], max_rounds: int):
    """Every decision sequence through the tree for a probing policy: list of
    (rows=[(features, legal mask names, action name)], quality, cost)."""
    out = []

    def rec(node, cost, rows):
        st = state_of(tree, node, cost, tiers, max_rounds)
        if node.round >= max_rounds:
            out.append((rows, node.quality, cost))
            return
        if not node.probed:
            return
        x = featurize(st)
        probe_usd = float(node.probe.get("usd", 0.0))
        out.append((rows + [(x, st.legal, STOP)], node.quality, cost + probe_usd))
        for a in st.legal:
            t = action_tier(a)
            if t is None:
                continue
            edge, child = node.children[t]
            rec(child, cost + probe_usd + float(edge["refine"]["usd"]), rows + [(x, st.legal, a)])

    rec(tree.root, tree.initial_cost, [])
    return out


def fit_controller(trees: list[Tree], tiers: list[str], max_rounds: int = 2, budget: float | None = 0.10,
                   lam: float = 2.0, steps: int = 1500, lr: float = 2.0, lam_lr: float = 2.0, lam_kp: float = 1.0,
                   ema: float = 0.9, logit_clip: float = 5.0, seed: int = 0, l2: float = 1e-4) -> Controller:
    """Exact policy gradient on the logged tree (no sampling: every branch is enumerated).

    J = E_pi[quality - lam * cost]. With a budget, lam is a Lagrange multiplier updated by damped
    dual ascent on an EMA of the policy's exact expected cost (as in pace.train_offline), and the
    served policy is the iterate average of the second half of training (Controller.mix)."""
    acts = actions_for(tiers)
    ctrl = Controller(acts, N_FEATURES, eps=0.0, logit_clip=logit_clip,
                      meta={"name": "ssr_controller", "features": FEATURE_NAMES, "tiers": tiers})
    rng = np.random.default_rng(seed)
    ctrl.W = rng.normal(0, 0.01, ctrl.W.shape)
    paths = [(ti, p) for ti, t in enumerate(trees) for p in _leaf_paths(t, tiers, max_rounds)]
    if not paths:
        raise ValueError("no complete paths to train on (are the inner nodes probed?)")
    # flatten decision rows
    X, M, A, owner = [], [], [], []           # owner = path index of each row
    for pi, (_, (rows, _, _)) in enumerate(paths):
        for x, legal, a in rows:
            X.append(x)
            M.append([a_ in legal for a_ in acts])
            A.append(acts.index(a))
            owner.append(pi)
    X, M, A, owner = np.array(X, float), np.array(M, bool), np.array(A, int), np.array(owner, int)
    Q = np.array([q for _, (_, q, _) in paths])
    C = np.array([c for _, (_, _, c) in paths])
    T = np.array([ti for ti, _ in paths])
    n_trees = len(trees)
    lam_t = float(lam)
    cost_ema = None
    ctrl.start_mix()
    hist = []
    for step in range(steps):
        P = ctrl.probs_batch(X, M, use_mix=False)
        pa = P[np.arange(len(A)), A]
        logp = np.zeros(len(paths))
        np.add.at(logp, owner, np.log(np.maximum(pa, 1e-12)))
        Pp = np.exp(logp)                                  # probability of each leaf path
        R = Q - lam_t * C
        V = np.zeros(n_trees)
        np.add.at(V, T, Pp * R)                            # exact value per tree
        exp_cost = float(np.sum(Pp * C) / n_trees)
        adv = R - V[T]
        coef = (Pp * adv)[owner] / n_trees
        G = ctrl.grad_logits_batch(X, M, A, coef) - l2 * ctrl.W
        ctrl.W += lr * G
        if budget is not None:
            cost_ema = exp_cost if cost_ema is None else ema * cost_ema + (1 - ema) * exp_cost
            lam_t = max(0.0, lam_t + lam_lr * (cost_ema - budget) + lam_kp * (exp_cost - cost_ema))
        if step >= steps // 2:
            ctrl.add_snapshot()
        hist.append((float(np.sum(Pp * Q) / n_trees), exp_cost, lam_t))
    ctrl.meta.update({"budget": budget, "lambda_final": lam_t, "lambda_bar": float(np.mean([h[2] for h in hist[steps // 2:]])),
                      "train_quality": hist[-1][0], "train_cost": hist[-1][1], "steps": steps, "n_trees": n_trees,
                      "max_rounds": max_rounds})
    return ctrl


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney with tie correction). NaN if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def pareto(points: list[tuple[str, float, float]]) -> set[str]:
    """Names of (name, quality, cost) points not dominated (higher quality, lower cost)."""
    keep = set()
    for n, q, c in points:
        if any(math.isnan(v) for v in (q, c)):
            continue
        dominated = any((q2 >= q and c2 <= c and (q2 > q or c2 < c)) for n2, q2, c2 in points if n2 != n)
        if not dominated:
            keep.add(n)
    return keep
