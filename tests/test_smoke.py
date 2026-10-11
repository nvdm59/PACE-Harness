"""Smoke tests: the pipeline at small scale. Runs under pytest or plain python:
    python -m pytest tests -q      |      python tests/test_smoke.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pace.collect import collect, load_policy, true_eval  # noqa: E402
from pace.gate import estimate, gate  # noqa: E402
from pace.harness import make_sim  # noqa: E402
from pace.trace import read_jsonl, write_jsonl  # noqa: E402
from pace.train_offline import train_awr, train_cgrpo  # noqa: E402

BUDGET = 0.30


@lru_cache(maxsize=None)
def _log():
    h = make_sim("v1", seed=11)
    return collect(h, load_policy("uniform", h), 3000, seed=11)


@lru_cache(maxsize=None)
def _trained(budget: float):
    pol, lam = train_cgrpo(_log(), make_sim("v1", seed=12), budget, seed=12)
    return pol, lam


def test_trace_roundtrip():
    eps = _log()[:20]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        write_jsonl(eps, p)
        back = read_jsonl(p)
    assert len(back) == 20
    assert back[0].steps[0].propensity == eps[0].steps[0].propensity
    assert back[0].steps[0].counterfactual is not None


def test_cgrpo_learns_lambda_and_meets_budget():
    pol, lam = _trained(BUDGET)
    assert lam > 0, lam
    assert pol.mix, "iterate-averaged mix policy missing"
    truth = true_eval(make_sim("v1", seed=13), pol, n=10000)
    assert truth["cost"] <= BUDGET + 0.02, truth


def test_dr_close_to_truth():
    pol, _ = _trained(BUDGET)
    truth = true_eval(make_sim("v1", seed=14), pol, n=20000)
    est = estimate(_log(), pol, boot=50)
    dr_q = est["quality"]["dr"][0]
    assert abs(dr_q - truth["quality"]) <= 0.03, (dr_q, truth)


def test_gate_reverts_overly_cheap_policy():
    incumbent, _ = _trained(BUDGET)
    cheap, _ = _trained(0.03)            # budget far below what quality needs
    res = gate(_log(), cheap, incumbent, BUDGET, boot=50)
    assert res["decision"] == "REVERT", res["candidate"]["quality"]["dr"]
    assert not res["quality_ok"]
    res_ok = gate(_log(), incumbent, "behavior", BUDGET, boot=50)
    assert res_ok["decision"] == "PROMOTE", res_ok["candidate"]


def test_awr_runs_and_beats_uniform():
    h = make_sim("v1", seed=15)
    pol = train_awr(_log(), h.all_actions, lam=0.5, epochs=150)
    uni = true_eval(h, load_policy("uniform", h), n=5000)
    got = true_eval(h, pol, n=5000)
    assert got["quality"] - 0.5 * got["cost"] > uni["quality"] - 0.5 * uni["cost"], (got, uni)


def test_v2_drift_harness():
    h2 = make_sim("v2", seed=16)
    assert "CACHED_RETRIEVE" in h2.all_actions
    assert abs(h2.price("ANSWER", "frontier") - 1.5 * make_sim("v1").price("ANSWER", "frontier")) < 1e-12


def test_react_adapter_imports_without_sdk_call():
    mod = importlib.import_module("pace.harness.react_api")
    assert set(mod.TIERS) == {"cheap", "mid", "frontier"}
    assert "anthropic" not in vars(mod), "the SDK must be imported lazily, not at module top level"
    h = mod.ReactAPIHarness([mod.QATask("t0", "What is the capital of France?", "Paris")])
    assert h.legal_actions(h.reset(h.tasks[0]))
    assert mod.match("It is Paris.", "Paris") == 1.0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print(f"{'all passed' if not fails else f'{fails} failed'}")
    sys.exit(1 if fails else 0)
