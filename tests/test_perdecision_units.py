"""Per-decision estimators: decision-weighted vs task-weighted units (pace.experiments.mas.offpolicy).

Reviewer's example: a short episode (1 decision) and a long episode (10 decisions). Flattening
decision records makes the long one count 10x under unit="decision"; unit="task" weights each
episode's rows by 1/n_decisions so that every task counts once, and lengthening an episode without
changing its outcome leaves the task-unit estimate unchanged."""
import numpy as np

from pace.controller import Controller
from pace.experiments.mas.offpolicy import cost_prefix, per_decision, unit_weights
from pace.features import N_FEATURES
from pace.trace import Episode, TraceStep

ACTS = ["X", "Y"]


def _ep(tid: str, n: int, quality: float, step_cost: float = 0.1, extra_cost: float = 0.0) -> Episode:
    # the short and the long task sit in different direct-model buckets (feature 2 = proxy_hard), so the
    # bucketed DM is exact here and DR equals SNIPS when the target is the behavior policy
    x = [1.0, 0.0, 1.0 if tid == "short" else 0.0] + [0.0] * (N_FEATURES - 3)
    steps = [TraceStep(task_id=tid, harness_version="test", step=t, features=list(x),
                       action=t % 2, action_name=ACTS[t % 2], propensity=0.5, cost=step_cost, latency=0.0,
                       legal=list(ACTS)) for t in range(n)]
    return Episode(task_id=tid, steps=steps, quality=quality, total_cost=extra_cost + n * step_cost)


def _behavior() -> Controller:
    return Controller(ACTS, N_FEATURES, eps=0.0)          # uniform over the 2 legal actions = the logger


def _episode_mass(eps, unit):
    u = unit_weights(eps, unit)
    owner = np.concatenate([[i] * len(e.steps) for i, e in enumerate(eps)])
    return np.bincount(owner, weights=u)


def test_unit_weights_short_vs_long():
    eps = [_ep("short", 1, 1.0), _ep("long", 10, 1.0)]     # same task outcome
    dec, task = _episode_mass(eps, "decision"), _episode_mass(eps, "task")
    assert list(dec) == [1.0, 10.0]                           # long episode weighs 10x
    assert np.allclose(task, [1.0, 1.0])                      # every task counts once
    # same outcome -> both units give the same quality, whatever the weighting
    for unit in ("decision", "task"):
        assert abs(per_decision(eps, _behavior(), "quality", boot=10, unit=unit)["snips"][0] - 1.0) < 1e-12


def test_outcomes_differ_decision_unit_tracks_length_task_unit_does_not():
    beh = _behavior()
    eps10 = [_ep("short", 1, 1.0), _ep("long", 10, 0.0)]
    eps20 = [_ep("short", 1, 1.0), _ep("long", 20, 0.0)]     # same outcomes, the long episode lengthened
    d10 = per_decision(eps10, beh, "quality", boot=10, unit="decision")
    d20 = per_decision(eps20, beh, "quality", boot=10, unit="decision")
    t10 = per_decision(eps10, beh, "quality", boot=10, unit="task")
    t20 = per_decision(eps20, beh, "quality", boot=10, unit="task")
    assert abs(d10["snips"][0] - 1 / 11) < 1e-12 and abs(d20["snips"][0] - 1 / 21) < 1e-12   # moves with length
    assert abs(d10["behavior"] - 1 / 11) < 1e-12
    for t in (t10, t20):
        assert abs(t["snips"][0] - 0.5) < 1e-12 and abs(t["behavior"] - 0.5) < 1e-12        # task mean
        assert abs(t["dr"][0] - 0.5) < 1e-12                                                # DR too (w = 1)
        assert t["n"] == 2 and abs(t["ess"] - 2.0) < 1e-12
    assert abs(t10["snips"][0] - t20["snips"][0]) < 1e-12 and abs(t10["dr"][0] - t20["dr"][0]) < 1e-12


def test_task_unit_cost_is_total_episode_cost():
    beh = _behavior()
    eps = [_ep("short", 1, 1.0, step_cost=0.2, extra_cost=0.05), _ep("long", 10, 0.0, step_cost=0.1, extra_cost=0.05)]
    pre = cost_prefix(eps)
    assert np.allclose(pre[:2], [0.05, 0.05]) and np.isclose(pre[-1], 0.05 + 0.9)
    tot = np.mean([e.total_cost for e in eps])                                        # (0.25 + 1.05) / 2
    t = per_decision(eps, beh, "cost_to_go", boot=10, unit="task")
    assert abs(t["behavior"] - tot) < 1e-12 and abs(t["snips"][0] - tot) < 1e-12 and abs(t["dr"][0] - tot) < 1e-9
    d = per_decision(eps, beh, "cost_to_go", boot=10, unit="decision")
    ctg_mean = np.mean([0.2] + [0.1 * k for k in range(10, 0, -1)])                  # overlapping cost-to-go rows
    assert abs(d["behavior"] - ctg_mean) < 1e-12 and abs(d["behavior"] - tot) > 0.1    # not a per-task cost
