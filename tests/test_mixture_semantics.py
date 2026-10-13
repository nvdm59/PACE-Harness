"""Statewise iterate averaging is not an episode-level mixture (the reviewer's counterexample).

`Controller(mix=[W1, W2])` serves the PER-STATE average of the snapshots' action probabilities.
In a sequential environment that is a different policy from the episode-level 50/50 mixture
(flip a coin once per episode, then follow W1 or W2), and it need not inherit the snapshots'
budget compliance: here each snapshot and the episode-level mixture have expected cost 1.792,
while the statewise average has expected cost 5.0.

Env: one-hot states s0 = e0 (first decision), L = e1, R = e2. At s0 action 0 -> L, action 1 -> R,
cost 0. At L / R the second action costs 1 if it matches the branch (action 0 at L, action 1 at
R) and 9 otherwise; the episode then ends.
"""
import random

import numpy as np

from pace.controller import Controller

W1 = np.array([[20.0, 20.0, 20.0], [-20.0, -20.0, -20.0]])
W2 = -W1
S0, L, R = np.eye(3)


def _pol(W=None, mix=None) -> Controller:
    return Controller(["0", "1"], 3, eps=0.1, logit_clip=3.0, W=W if W is not None else np.zeros((2, 3)), mix=mix)


def expected_cost(pol: Controller) -> float:
    p1 = pol.probs(S0)
    cost = 0.0
    for branch, state in ((0, L), (1, R)):
        p2 = pol.probs(state)
        cost += p1[branch] * (p2[branch] * 1.0 + (1.0 - p2[branch]) * 9.0)
    return float(cost)


def mc_cost(pol: Controller, n: int = 20000, seed: int = 0) -> float:
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(n):
        b, _ = pol.act(S0, rng=rng)
        a, _ = pol.act(L if b == 0 else R, rng=rng)
        tot += 1.0 if a == b else 9.0
    return tot / n


def test_snapshots_and_episode_mixture_meet_cost_1792():
    c1, c2 = expected_cost(_pol(W1)), expected_cost(_pol(W2))
    assert abs(c1 - 1.792) < 1e-3 and abs(c2 - 1.792) < 1e-3
    # closed form: logits +-z with z = 3 tanh(20 / 3); p = 0.9 * sigmoid(2 z) + 0.05; cost = 1 + 16 p (1 - p)
    z = 3.0 * np.tanh(20.0 / 3.0)
    p = 0.9 / (1 + np.exp(-2 * z)) + 0.05
    assert abs(c1 - (1 + 16 * p * (1 - p))) < 1e-9
    episode_mix = 0.5 * c1 + 0.5 * c2
    assert abs(episode_mix - 1.792) < 1e-3


def test_statewise_average_costs_5():
    mixed = _pol(mix=[W1, W2])
    assert np.allclose(mixed.probs(S0), [0.5, 0.5]) and np.allclose(mixed.probs(L), [0.5, 0.5])
    assert abs(expected_cost(mixed) - 5.0) < 1e-9
    # the served policy (sampling through Controller.act) agrees
    assert abs(mc_cost(mixed) - 5.0) < 0.1
    assert abs(mc_cost(_pol(W1)) - 1.792) < 0.06


def test_statewise_average_breaks_a_budget_both_snapshots_meet():
    budget = 2.0
    assert expected_cost(_pol(W1)) <= budget and expected_cost(_pol(W2)) <= budget
    assert expected_cost(_pol(mix=[W1, W2])) > budget
