"""Retention under a grown action set is empirical, not guaranteed.

A controller trained on stage-0 actions {a, b} and then updated on stage-1 data in which a new
action c is available changes its stage-0 behaviour: with c masked out, its probabilities over
{a, b} move. Masking the new action at evaluation does not freeze the old actions' weights, because
the softmax score of every logged decision (grad log pi(c|x) = e_c - pi(.|x), times x) moves every
legal row of W, not only the new one. Uses the trainers' own gradient primitive
(`Controller.grad_logits_batch`, shared by cgrpo / AWR).
"""
import numpy as np

from pace.controller import Controller

X_TEST = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.6, 0.4, 1.0]])


def _train(pol: Controller, X, legal, A, coef, steps=200, lr=0.05):
    M = np.tile(pol.mask(legal), (len(X), 1))
    for _ in range(steps):
        pol.W += lr * pol.grad_logits_batch(X, M, A, coef) / len(X)


def _stage0_probs(pol: Controller) -> np.ndarray:
    ix = [pol.index("a"), pol.index("b")]
    return np.array([pol.probs(x, ["a", "b"], eps=0.0)[ix] for x in X_TEST])


def test_stage1_update_moves_stage0_behaviour():
    rng = np.random.default_rng(0)
    pol = Controller(["a", "b"], 3, eps=0.1, logit_clip=3.0)
    # stage 0: prefer a when feature 0 is on, b when feature 1 is on
    X0 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]] * 50)
    A0 = np.array([0, 1] * 50)
    _train(pol, X0, ["a", "b"], A0, np.ones(len(X0)))
    before = _stage0_probs(pol)
    W_old_before = pol.W.copy()
    assert before[0, 0] > 0.75 and before[1, 1] > 0.75           # stage 0 learned

    # stage 1: new action c shipped and rewarded in every context
    pol.ensure_actions(["c"])
    assert np.allclose(_stage0_probs(pol), before)                # adding a zero row changes nothing yet
    X1 = X0[rng.permutation(len(X0))]
    A1 = np.full(len(X1), pol.index("c"))
    _train(pol, X1, ["a", "b", "c"], A1, np.ones(len(X1)), steps=60)
    after = _stage0_probs(pol)

    # old rows moved, and so did stage-0 behaviour with c masked
    assert np.abs(pol.W[:2] - W_old_before).max() > 0.1
    shift = np.abs(after - before).max()
    assert shift > 0.05, shift
    # direction: the previously preferred action loses most (its row is pushed down by pi(a|x) x)
    assert after[0, 0] < before[0, 0] and after[1, 1] < before[1, 1]
