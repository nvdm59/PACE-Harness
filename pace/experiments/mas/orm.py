"""Orchestration reward model (ORM): process features -> P(correct outcome).

Trained on the logged (process, outcome) pairs of the MAS. Two small models in numpy (no extra
dependency): L2-regularised logistic regression and a one-hidden-layer MLP. Evaluated by K-fold
cross-validation over TASKS (category-stratified, so a task's episodes never straddle folds):
AUROC, Brier score, expected calibration error and a calibration curve.

Features are computed from a PREFIX of the process log (`upto` role calls), so the same model
scores a finished episode (reward for the log) and a partial one (value estimate at a decision
point). The exported `quality_model.json` is consumed by `relabel_episodes`, which replaces each
episode's quality by quality_hat: the relabelled episodes feed `pace.train_offline.train_awr`
(reward quality_hat - lambda x cost) and `pace.gate.estimate` unchanged. That is the
"reward model as the quality signal" slot of the main README's plug-in table.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...harness.crmarena import CATEGORIES
from ...prices import TIERS
from ...trace import Episode
from .process_verify import CORRECT_THRESHOLD, is_correct

FEATURE_NAMES = (["bias", "n_calls/8", "n_worker/4", "n_verify/4", "n_pass/4", "n_fail/4", "pass_rate",
                  "has_verify", "open_fails", "n_replan", "n_retry", "n_rows/3", "n_empty/3", "n_error/3",
                  "cost_so_far", "last_obs=rows", "last_obs=empty", "last_obs=error", "answered"]
                 + [f"worker_tier={t}/4" for t in TIERS] + [f"answer_tier={t}" for t in TIERS]
                 + [f"cat={c}" for c in CATEGORIES])
VERIFIER_FEATURES = ["n_verify/4", "n_pass/4", "n_fail/4", "pass_rate", "has_verify", "open_fails", "n_retry"]
CATEGORY_FEATURES = [f"cat={c}" for c in CATEGORIES]
# Default ORM input: the process features without the task-category one-hot. With 2-3 tasks per
# category the one-hot memorises the training tasks (the same finding as the linear controller in
# results/crmarena/RESULTS.md); it stays available as an ablation row in the CV table.
DEFAULT_DROP = list(CATEGORY_FEATURES)


def episode_features(ep: Episode, upto: int | None = None) -> list[float]:
    calls = list((ep.meta or {}).get("process") or [])
    if upto is not None:
        calls = calls[:upto]
    workers = [c for c in calls if c["role"] == "worker"]
    verifs = [c for c in calls if c["role"] == "verifier"]
    answers = [c for c in calls if c["role"] == "answer"]
    n_pass = sum(1 for c in verifs if c["verdict"] == "pass")
    n_fail = len(verifs) - n_pass
    last_by_sub: dict = {}
    for c in verifs:
        last_by_sub[c.get("sub_query")] = c["verdict"]
    open_fails = sum(1 for v in last_by_sub.values() if v == "fail")
    last_kind = workers[-1]["verdict"] if workers else "none"
    cost = sum(float(c["cost"]) for c in calls)
    wt = [sum(1 for c in workers if c["tier"] == t) / 4.0 for t in TIERS]
    at = [1.0 if (answers and answers[-1]["tier"] == t) else 0.0 for t in TIERS]
    cat = [1.0 if (ep.meta or {}).get("category") == c else 0.0 for c in CATEGORIES]
    x = [1.0, len(calls) / 8.0, len(workers) / 4.0, len(verifs) / 4.0, n_pass / 4.0, n_fail / 4.0,
         n_pass / len(verifs) if verifs else 0.0, 1.0 if verifs else 0.0, float(open_fails),
         float(sum(1 for c in calls if c["role"] == "planner" and c["verdict"] == "replan")),
         float(sum(1 for c in workers if c.get("retry"))),
         sum(1 for c in workers if c["verdict"] == "rows") / 3.0, sum(1 for c in workers if c["verdict"] == "empty") / 3.0,
         sum(1 for c in workers if c["verdict"] in ("error", "invalid")) / 3.0, cost,
         1.0 if last_kind == "rows" else 0.0, 1.0 if last_kind == "empty" else 0.0,
         1.0 if last_kind in ("error", "invalid") else 0.0, 1.0 if answers else 0.0] + wt + at + cat
    assert len(x) == len(FEATURE_NAMES)
    return x


def kept_names(drop: list[str] | None) -> list[str]:
    return [n for n in FEATURE_NAMES if n not in set(drop or [])]


def dataset(episodes: list[Episode], threshold: float = CORRECT_THRESHOLD, drop: list[str] | None = None):
    X = np.array([episode_features(e) for e in episodes], float)
    y = np.array([1.0 if is_correct(e, threshold) else 0.0 for e in episodes])
    if drop:
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in set(drop)]
        X = X[:, keep]
    return X, y


# -------------------------------------------------------------------- models
class _Standardize:
    def fit(self, X):
        self.mean = X.mean(0)
        self.std = np.where(X.std(0) > 1e-9, X.std(0), 1.0)
        self.mean[0], self.std[0] = 0.0, 1.0           # keep the bias column as is
        return self

    def __call__(self, X):
        return (X - self.mean) / self.std


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogReg:
    name = "logreg"

    def __init__(self, l2: float = 5.0, iters: int = 500, lr: float = 0.5):
        self.l2, self.iters, self.lr = float(l2), int(iters), float(lr)

    def fit(self, X, y):
        self.sc = _Standardize().fit(X)
        Z = self.sc(X)
        n, d = Z.shape
        self.w = np.zeros(d)
        reg = np.full(d, self.l2); reg[0] = 0.0
        for _ in range(self.iters):
            p = _sigmoid(Z @ self.w)
            g = Z.T @ (p - y) / n + reg * self.w / n
            self.w -= self.lr * g
        return self

    def predict_proba(self, X):
        return _sigmoid(self.sc(X) @ self.w)

    def to_dict(self):
        return {"type": "logreg", "w": self.w.tolist(), "mean": self.sc.mean.tolist(), "std": self.sc.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(); m.w = np.array(d["w"]); m.sc = _Standardize(); m.sc.mean = np.array(d["mean"]); m.sc.std = np.array(d["std"])
        return m


class MLP:
    name = "mlp"

    def __init__(self, hidden: int = 16, l2: float = 0.1, iters: int = 400, lr: float = 0.02, seed: int = 0):
        self.hidden, self.l2, self.iters, self.lr, self.seed = int(hidden), float(l2), int(iters), float(lr), int(seed)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.sc = _Standardize().fit(X)
        Z = self.sc(X)
        n, d = Z.shape
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, self.hidden)); self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(self.hidden), self.hidden); self.b2 = 0.0
        params = [self.W1, self.b1, self.W2]
        m = [np.zeros_like(p) for p in params] + [0.0]; v = [np.zeros_like(p) for p in params] + [0.0]
        b1m = b1v = 0.0
        for t in range(1, self.iters + 1):
            H = np.tanh(Z @ self.W1 + self.b1)
            p = _sigmoid(H @ self.W2 + self.b2)
            dz = (p - y) / n
            gW2 = H.T @ dz + self.l2 * self.W2; gb2 = dz.sum()
            dH = np.outer(dz, self.W2) * (1 - H ** 2)
            gW1 = Z.T @ dH + self.l2 * self.W1; gb1 = dH.sum(0)
            for i, (prm, g) in enumerate(((self.W1, gW1), (self.b1, gb1), (self.W2, gW2))):
                m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g ** 2
                prm -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            b1m = 0.9 * b1m + 0.1 * gb2; b1v = 0.999 * b1v + 0.001 * gb2 ** 2
            self.b2 -= self.lr * (b1m / (1 - 0.9 ** t)) / (np.sqrt(b1v / (1 - 0.999 ** t)) + 1e-8)
        return self

    def predict_proba(self, X):
        H = np.tanh(self.sc(X) @ self.W1 + self.b1)
        return _sigmoid(H @ self.W2 + self.b2)

    def to_dict(self):
        return {"type": "mlp", "W1": self.W1.tolist(), "b1": self.b1.tolist(), "W2": self.W2.tolist(),
                "b2": float(self.b2), "mean": self.sc.mean.tolist(), "std": self.sc.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(hidden=len(d["b1"]))
        m.W1, m.b1, m.W2, m.b2 = np.array(d["W1"]), np.array(d["b1"]), np.array(d["W2"]), float(d["b2"])
        m.sc = _Standardize(); m.sc.mean = np.array(d["mean"]); m.sc.std = np.array(d["std"])
        return m


MODELS = {"logreg": LogReg, "mlp": MLP}


# ------------------------------------------------------------------- metrics
def auroc(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney with ties counted 1/2
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_curve(y, p, bins: int = 5) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
                    "frac_pos": float(y[m].mean()) if m.any() else float("nan")})
    return out


def ece(y, p, bins: int = 5) -> float:
    cal = calibration_curve(y, p, bins)
    n = sum(c["n"] for c in cal)
    return float(sum(c["n"] / n * abs(c["mean_pred"] - c["frac_pos"]) for c in cal if c["n"] > 0)) if n else float("nan")


def kfold_by_task(episodes: list[Episode], k: int = 5, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Category-stratified K folds over task ids (same dealing scheme as pace.lookup.kfold_tasks);
    returns (train episode indices, test episode indices) per fold."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for e in episodes:
        if e.task_id not in seen:
            seen.add(e.task_id)
            by_cat[(e.meta or {}).get("category", "?")].append(e.task_id)
    rng = random.Random(seed)
    folds: list[set[str]] = [set() for _ in range(k)]
    j = 0
    for c in sorted(by_cat):
        ts = sorted(by_cat[c]); rng.shuffle(ts)
        for t in ts:
            folds[j % k].add(t); j += 1
    out = []
    for f in folds:
        test = [i for i, e in enumerate(episodes) if e.task_id in f]
        train = [i for i, e in enumerate(episodes) if e.task_id not in f]
        if test and train:
            out.append((train, test))
    return out


def cross_validate(episodes: list[Episode], model: str = "logreg", k: int = 5, seed: int = 0,
                   drop: list[str] | None = None, bins: int = 5, **kw) -> dict:
    X, y = dataset(episodes, drop=drop)
    p_oof = np.full(len(y), np.nan)
    for tr, te in kfold_by_task(episodes, k, seed):
        m = MODELS[model](**({"seed": seed} if model == "mlp" else {}), **kw).fit(X[tr], y[tr])
        p_oof[te] = m.predict_proba(X[te])
    ok = ~np.isnan(p_oof)
    base = float(y.mean())
    return {"model": model, "n": int(ok.sum()), "k": k, "base_rate": base,
            "auroc": auroc(y[ok], p_oof[ok]), "brier": brier(y[ok], p_oof[ok]),
            "brier_base_rate": float(np.mean((base - y[ok]) ** 2)), "ece": ece(y[ok], p_oof[ok], bins),
            "calibration": calibration_curve(y[ok], p_oof[ok], bins), "p_oof": p_oof.tolist(),
            "dropped": list(drop or [])}


# -------------------------------------------------------------------- export
class QualityModel:
    """A fitted ORM, serialisable to `quality_model.json` and usable as a quality signal."""

    def __init__(self, model, feature_names: list[str], meta: dict | None = None):
        self.model, self.feature_names, self.meta = model, list(feature_names), dict(meta or {})

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(np.asarray(X, float)))

    def predict_episode(self, ep: Episode, upto: int | None = None) -> float:
        full = episode_features(ep, upto)
        idx = [FEATURE_NAMES.index(n) for n in self.feature_names]
        return float(self.predict([[full[i] for i in idx]])[0])

    def to_dict(self) -> dict:
        return {"kind": "mas_orm", "quality": "P(correct outcome | process)", "feature_names": self.feature_names,
                "model": self.model.to_dict(), "meta": self.meta}

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "QualityModel":
        if d.get("kind") != "mas_orm":
            raise ValueError("not a MAS quality model")
        m = MODELS[d["model"]["type"]].from_dict(d["model"])
        return cls(m, d["feature_names"], d.get("meta"))

    @classmethod
    def load(cls, path: str | Path) -> "QualityModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_quality_model(episodes: list[Episode], model: str = "logreg", seed: int = 0,
                      drop: list[str] | None = None, **kw) -> QualityModel:
    drop = DEFAULT_DROP if drop is None else drop
    X, y = dataset(episodes, drop=drop)
    m = MODELS[model](**({"seed": seed} if model == "mlp" else {}), **kw).fit(X, y)
    p = m.predict_proba(X)
    return QualityModel(m, kept_names(drop), {"n_train": int(len(y)), "base_rate": float(y.mean()),
                                               "train_auroc": auroc(y, p), "train_brier": brier(y, p),
                                               "dropped": list(drop)})


def relabel_episodes(episodes: list[Episode], qm: QualityModel) -> list[Episode]:
    """Episodes whose `quality` is the ORM's P(correct) at the end of the process, with the
    outcome kept in meta["quality_outcome"]. Steps, costs and propensities are unchanged, so
    `pace.train_offline.train_awr(relabelled, ACTIONS, lam, n_features=...)` trains on reward
    quality_hat - lam x cost and `pace.gate.estimate` scores a candidate on quality_hat."""
    out = []
    for e in episodes:
        q = qm.predict_episode(e)
        out.append(replace(e, quality=q, meta={**(e.meta or {}), "quality_outcome": float(e.quality),
                                               "quality_source": "mas_orm"}))
    return out


def priced_reward(ep: Episode, qm: QualityModel, lam: float) -> float:
    return qm.predict_episode(ep) - lam * float(ep.total_cost)
