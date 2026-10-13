"""Exact-gradient trainer and category LP (pace.train_exact) on tiny synthetic tables (CPU, < 1 s)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pace.controller import Controller  # noqa: E402
from pace.lookup import DECISION_ACTIONS  # noqa: E402
from pace.train_exact import category_lp, exact_grad, exact_objective  # noqa: E402


def _tiny(seed=0, n=7, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Q = rng.random((n, 3))
    C = rng.uniform(0.02, 0.2, size=(n, 3))
    return X, Q, C


def test_exact_grad_matches_finite_difference():
    X, Q, C = _tiny()
    for eps in (0.0, 0.1):
        pol = Controller(DECISION_ACTIONS, X.shape[1])
        pol.W = np.random.default_rng(1).normal(scale=2.0, size=pol.W.shape)   # exercise the soft clip
        lam = 3.0
        g = exact_grad(pol, X, Q, C, lam, eps)
        h = 1e-6
        fd = np.zeros_like(pol.W)
        for i in range(pol.W.shape[0]):
            for j in range(pol.W.shape[1]):
                W0 = pol.W.copy()
                pol.W = W0.copy(); pol.W[i, j] += h
                up = exact_objective(pol, X, Q, C, lam, eps)
                pol.W = W0.copy(); pol.W[i, j] -= h
                dn = exact_objective(pol, X, Q, C, lam, eps)
                pol.W = W0
                fd[i, j] = (up - dn) / (2 * h)
        assert np.allclose(g, fd, atol=1e-7, rtol=1e-5), np.abs(g - fd).max()


def test_category_lp_budget_and_optimality():
    rng = np.random.default_rng(3)
    cats = [f"c{k}" for k in rng.integers(0, 5, size=40)]
    Q = rng.random((40, 3))
    C = rng.uniform(0.02, 0.2, size=(40, 3))
    for B in (0.03, 0.08, 0.11, 0.5):
        lp = category_lp(cats, Q, C, B)
        p = lp["p"]
        assert np.allclose(p.sum(1), 1.0) and (p >= -1e-12).all()
        cv = lp["cats"]
        ci = np.array([cv.index(c) for c in cats])
        n_c = np.bincount(ci, minlength=len(cv)).astype(float)
        Qb = np.array([Q[ci == j].mean(0) for j in range(len(cv))])
        Cb = np.array([C[ci == j].mean(0) for j in range(len(cv))])
        cost = float((n_c[:, None] * p * Cb).sum() / n_c.sum())
        val = float((n_c[:, None] * p * Qb).sum() / n_c.sum())
        if not lp["feasible"]:
            assert cost > B
            continue
        assert cost <= B + 1e-12
        # LP duality: the optimum equals min over mu >= 0 of the Lagrangian dual, whose minimum
        # sits at a breakpoint (a hull-segment rate) or at mu = 0
        def dual(mu):
            return float((n_c * (Qb - mu * Cb).max(1)).sum() / n_c.sum() + mu * B)
        mus = [0.0] + [abs((Qb[j, a] - Qb[j, b]) / (Cb[j, a] - Cb[j, b]))
                       for j in range(len(cv)) for a in range(3) for b in range(3) if Cb[j, a] != Cb[j, b]]
        assert abs(min(dual(m) for m in mus) - val) < 1e-9
