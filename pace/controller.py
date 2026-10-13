"""Softmax-linear controller over state features, with logged propensities.

The policy PACE serves and trains is

    logits = L * tanh((W x) / L)          soft logit clipping: the softmax never saturates,
                                           so a Lagrange multiplier can still steer it, and
                                           importance ratios against it stay bounded
    pi(a | x) = (1 - eps) * softmax(logits restricted to the legal set) + eps / |legal|

The propensity returned by `act` INCLUDES the eps exploration mass, so a log written by a
served controller is a valid behavior log for off-policy evaluation (the logging discipline
of Bottou et al. 2013, arXiv:1209.2355; Li et al. 2011, arXiv:1003.5956).

Iterate averaging: `mix` holds weight snapshots from the second half of primal-dual training;
the served "mix" policy is the per-state average of the snapshots' action probabilities.
Primal-dual iterates orbit the budget; their average meets it (the standard remedy for
saddle-point subgradient methods, Nedic & Ozdaglar 2009).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


class Controller:
    def __init__(self, actions: list[str], n_features: int, eps: float = 0.0, logit_clip: float = 3.0,
                 W: np.ndarray | None = None, mix: list[np.ndarray] | None = None, meta: dict | None = None):
        self.actions = list(actions)
        self.n_features = int(n_features)
        self.eps = float(eps)
        self.logit_clip = float(logit_clip)
        self.W = np.zeros((len(self.actions), self.n_features)) if W is None else np.asarray(W, float).copy()
        self.mix = None if mix is None else [np.asarray(m, float).copy() for m in mix]
        self.meta = dict(meta or {})
        self._index = {a: i for i, a in enumerate(self.actions)}

    # ------------------------------------------------------------ actions
    def index(self, name: str) -> int:
        return self._index[name]

    def ensure_actions(self, names: list[str]) -> None:
        """Add zero-logit rows for actions this policy has never seen (e.g. a tool shipped in a
        new harness version), so a v1-trained policy can be served and scored on v2 logs."""
        new = [a for a in names if a not in self._index]
        if not new:
            return
        pad = np.zeros((len(new), self.n_features))
        self.W = np.vstack([self.W, pad])
        if self.mix is not None:
            self.mix = [np.vstack([m, pad]) for m in self.mix]
        self.actions += new
        self._index = {a: i for i, a in enumerate(self.actions)}

    def mask(self, legal: list[str] | None) -> np.ndarray:
        """Boolean legal-action mask over `self.actions` (`None` = everything legal)."""
        m = np.zeros(len(self.actions), bool)
        if legal is None:
            m[:] = True
        else:
            for a in legal:
                m[self._index[a]] = True
        return m

    # -------------------------------------------------------------- probs
    def _soft(self, W: np.ndarray, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        L = self.logit_clip
        z = L * np.tanh((W @ x) / L)
        z = np.where(m, z, -np.inf)
        z = z - z[m].max()
        e = np.exp(z)
        return e / e.sum()

    def base_probs(self, x, legal: list[str] | None = None, use_mix: bool = True) -> np.ndarray:
        """Action probabilities WITHOUT the eps mass (the iterate mix when present and `use_mix`)."""
        x = np.asarray(x, float)
        m = self.mask(legal)
        if use_mix and self.mix:
            return np.mean([self._soft(Wk, x, m) for Wk in self.mix], axis=0)
        return self._soft(self.W, x, m)

    def probs(self, x, legal: list[str] | None = None, eps: float | None = None,
              use_mix: bool = True) -> np.ndarray:
        """Served action probabilities: base probabilities mixed with eps-uniform over the legal set."""
        eps = self.eps if eps is None else eps
        m = self.mask(legal)
        p = self.base_probs(x, legal, use_mix)
        if eps > 0:
            p = (1 - eps) * p + eps * m / m.sum()
        return p

    def act(self, x, legal: list[str] | None = None, rng: random.Random | None = None,
            eps: float | None = None, use_mix: bool = True) -> tuple[int, float]:
        """Sample an action. Returns (action index, propensity INCLUDING the eps mass)."""
        rng = rng or random
        p = self.probs(x, legal, eps, use_mix)
        u, acc = rng.random(), 0.0
        idx = int(np.flatnonzero(p > 0)[-1])
        for i in np.flatnonzero(p > 0):
            acc += p[i]
            if u < acc:
                idx = int(i)
                break
        return idx, float(p[idx])

    # ----------------------------------------------------- batched (gate)
    def probs_batch(self, X: np.ndarray, M: np.ndarray, eps: float | None = None,
                    use_mix: bool = True) -> np.ndarray:
        """X: (n, d) features, M: (n, A) legal masks -> (n, A) served probabilities."""
        eps = self.eps if eps is None else eps
        Ws = self.mix if (use_mix and self.mix) else [self.W]
        L = self.logit_clip
        P = np.zeros(M.shape)
        for Wk in Ws:
            z = L * np.tanh((X @ Wk.T) / L)
            z = np.where(M, z, -np.inf)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            P += e / e.sum(axis=1, keepdims=True)
        P /= len(Ws)
        if eps > 0:
            P = (1 - eps) * P + eps * M / M.sum(axis=1, keepdims=True)
        return P

    def grad_logits_batch(self, X: np.ndarray, M: np.ndarray, A: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """sum_i coef_i * d log softmax(a_i | x_i) / dW  at the current W (no eps, no mix).

        This is the score function of the eps-FREE softmax. When actions were sampled from the
        eps-mixed served policy, multiply `coef` by `eps_score_weights` first (see there)."""
        L = self.logit_clip
        raw = X @ self.W.T
        t = np.tanh(raw / L)
        z = np.where(M, L * t, -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        P = e / e.sum(axis=1, keepdims=True)
        G = -P
        G[np.arange(len(A)), A] += 1.0
        G *= (1.0 - t ** 2) * M          # chain rule through the soft clip
        G *= coef[:, None]
        return G.T @ X

    def eps_score_weights(self, X: np.ndarray, M: np.ndarray, A: np.ndarray, eps: float) -> np.ndarray:
        """Per-sample factor (1 - eps) * pi(a|x) / pi_eps(a|x) such that
        grad log pi_eps(a|x) = factor * grad log pi(a|x).

        The served policy is the eps-mixture, and its value (and the cost the budget constrains)
        is an expectation under pi_eps. A rollout sampled from pi_eps therefore needs the score
        of pi_eps, not of the bare softmax: the uniform component carries no gradient, so the
        score is damped on actions the softmax itself rarely picks. Returns ones when eps == 0."""
        if eps <= 0:
            return np.ones(len(A))
        P = self.probs_batch(X, M, eps=0.0, use_mix=False)
        p = P[np.arange(len(A)), A]
        p_eps = (1.0 - eps) * p + eps / M.sum(axis=1)
        return (1.0 - eps) * p / p_eps

    # ------------------------------------------------------- saturation
    SAT_THRESHOLD = 2.5      # |Wx| / L above this: sech^2 < 0.027, the soft clip passes < 3% of the gradient

    def saturation_stats(self, X: np.ndarray, M: np.ndarray | None = None, use_mix: bool = True,
                         threshold: float | None = None) -> dict:
        """How far the soft logit clip z = L tanh(Wx / L) is driven into its flat region.

        dz / d(Wx) = sech^2(Wx / L) -> 0 as |Wx| / L grows, so a policy can have bounded, non-degenerate
        probabilities (z is within [-L, L]) while its weights no longer respond to the gradient: a
        low-probability action whose raw logit is deep in the flat region cannot be steered back up by
        a Lagrange multiplier or by new data. Bounded probabilities do NOT imply steerable weights.

        X: (n, d) inputs; M: optional (n, A) legal mask (only legal (action, input) pairs count).
        With `use_mix` and an iterate mix, every snapshot's (action, input) pairs are pooled.
        Returns max |Wx|/L, the fraction of pairs with |Wx|/L > threshold (default 2.5, split by the
        sign of Wx), the mean and min sech^2 factor, and the fraction of pairs whose sech^2 < 1e-3."""
        thr = self.SAT_THRESHOLD if threshold is None else float(threshold)
        X = np.atleast_2d(np.asarray(X, float))
        M = np.ones((len(X), len(self.actions)), bool) if M is None else np.asarray(M, bool)
        Ws = self.mix if (use_mix and self.mix) else [self.W]
        U = np.concatenate([((X @ Wk.T) / self.logit_clip)[M] for Wk in Ws])     # signed Wx / L over legal pairs
        a = np.abs(U)
        sech2 = 1.0 - np.tanh(a) ** 2
        return {"n_pairs": int(a.size), "n_inputs": int(len(X)), "n_weight_sets": len(Ws),
                "max_abs_u": float(a.max()) if a.size else 0.0,
                "frac_saturated": float((a > thr).mean()) if a.size else 0.0,
                "frac_saturated_low": float((U < -thr).mean()) if a.size else 0.0,
                "frac_saturated_high": float((U > thr).mean()) if a.size else 0.0,
                "mean_sech2": float(sech2.mean()) if a.size else 1.0,
                "min_sech2": float(sech2.min()) if a.size else 1.0,
                "frac_sech2_below_1e-3": float((sech2 < 1e-3).mean()) if a.size else 0.0,
                "threshold": thr}

    # ------------------------------------------------------------- mixing
    def start_mix(self) -> None:
        self.mix = []

    def add_snapshot(self) -> None:
        if self.mix is None:
            self.mix = []
        self.mix.append(self.W.copy())

    # ---------------------------------------------------------------- I/O
    def copy(self) -> "Controller":
        return Controller(self.actions, self.n_features, self.eps, self.logit_clip, self.W, self.mix, self.meta)

    def to_dict(self) -> dict:
        r = lambda a: np.round(a, 6).tolist()
        return {"actions": self.actions, "n_features": self.n_features, "eps": self.eps,
                "logit_clip": self.logit_clip, "W": r(self.W),
                "mix": None if self.mix is None else [r(m) for m in self.mix], "meta": self.meta}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> "Controller":
        return cls(d["actions"], d["n_features"], d.get("eps", 0.0), d.get("logit_clip", 3.0),
                   np.array(d["W"]), None if d.get("mix") is None else [np.array(m) for m in d["mix"]],
                   d.get("meta"))

    @classmethod
    def load(cls, path: str | Path) -> "Controller":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def uniform(cls, actions: list[str], n_features: int) -> "Controller":
        return cls(actions, n_features, eps=0.0, meta={"name": "uniform"})
