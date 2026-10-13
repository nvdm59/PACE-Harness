"""The soft logit clip z = L tanh(Wx / L) bounds the probabilities but not the weights: once |Wx|/L
is large, dz/d(Wx) = sech^2(Wx/L) is (numerically) zero, so the score-function gradient of a
low-probability action vanishes even though its probability is not 0."""
import numpy as np

from pace.controller import Controller


def _saturated():
    return Controller(["hi", "lo"], 1, eps=0.0, logit_clip=3.0, W=np.array([[300.0], [-300.0]]))


def test_bounded_probs_but_zero_gradient():
    c = _saturated()
    X, M = np.array([[1.0]]), np.ones((1, 2), bool)
    p = c.probs_batch(X, M, eps=0.0)[0]
    assert np.allclose(p, [0.9975, 0.0025], atol=1e-4)            # e^6 / (1 + e^6) = 0.99753
    assert 0.002 < p[1] < 0.003                                     # low-probability action is NOT at 0
    for a in (0, 1):
        g = c.grad_logits_batch(X, M, np.array([a]), np.array([1.0]))
        assert np.all(g == 0.0)                                     # exactly zero: sech^2(100) underflows
    # the same probabilities from an unsaturated W (z = W x exactly at the bound) do have a gradient
    u = Controller(["hi", "lo"], 1, eps=0.0, logit_clip=3.0, W=np.array([[1.5], [-1.5]]))
    g = u.grad_logits_batch(X, M, np.array([1]), np.array([1.0]))
    assert g[1, 0] > 0.1


def test_saturation_stats():
    st = _saturated().saturation_stats(np.array([[1.0]]))
    assert st["max_abs_u"] == 100.0 and st["frac_saturated"] == 1.0
    assert st["frac_saturated_low"] == 0.5 and st["frac_saturated_high"] == 0.5
    assert st["mean_sech2"] == 0.0 and st["frac_sech2_below_1e-3"] == 1.0
    z = Controller(["a", "b"], 2).saturation_stats(np.ones((5, 2)))
    assert z["max_abs_u"] == 0.0 and z["frac_saturated"] == 0.0 and z["mean_sech2"] == 1.0
    # legal mask restricts the pairs; the iterate mix pools snapshots
    c = Controller(["a", "b"], 1, W=np.array([[30.0], [0.0]]), mix=[np.array([[30.0], [0.0]]), np.zeros((2, 1))])
    st = c.saturation_stats(np.array([[1.0]]))
    assert st["n_pairs"] == 4 and st["frac_saturated"] == 0.25 and st["max_abs_u"] == 10.0
    st = c.saturation_stats(np.array([[1.0]]), M=np.array([[False, True]]))
    assert st["n_pairs"] == 2 and st["frac_saturated"] == 0.0
