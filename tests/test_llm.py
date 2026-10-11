"""LLM-controller pieces on CPU (no torch, no GPU, no network):
prompt render / strict parse, SimEpisodeReward determinism, LookupReward on a
fake 3-task CRMArena log, DualAscentLambda direction, and the TRL adapter."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pace.harness import make_sim  # noqa: E402
from pace.llm.env_text import (INVALID_PENALTY, format_action, legal_for, lookup_prompt,  # noqa: E402
                               parse_action, sim_rows, sim_task)
from pace.llm.rewards import (DualAscentLambda, LookupReward, PricedRewardFn, SimEpisodeReward,  # noqa: E402
                              lookup_rows)


def test_prompt_render_parse_roundtrip():
    rows = sim_rows(20, seed=3, budget=0.30)
    for r in rows:
        task = json.loads(r["task"])
        assert "latent" not in r["user"]
        assert "0.30" in r["user"] and f"Observed difficulty: {task['proxy']}" in r["user"]
        legal = legal_for(task)
        assert len(legal) == 12
        for tier, op in legal:
            assert f"{tier} {op}" in r["user"]
            assert parse_action(format_action(tier, op), legal) == (tier, op)
            assert parse_action("  " + format_action(tier, op) + "\n", legal) == (tier, op)
    legal = legal_for(json.loads(rows[0]["task"]))
    for bad in ["", "mid ANSWER", "<action>mid answer</action>", "<action>Mid ANSWER</action>",
                "<action>mid ANSWER</action> because", "<action>mid ANSWER</action><action>cheap ANSWER</action>",
                "<action>mid CONTINUE</action>", "<action>mid  ANSWER</action>", None]:
        assert parse_action(bad, legal) is None, bad
    # lookup mode: only CONTINUE is legal (the only op a counterfactual log covers)
    p = lookup_prompt({"task_id": "b2b-1", "org": "b2b", "category": "lead_routing", "reward_metric": "exact_match"},
                      0.1, {"cheap": 0.03, "mid": 0.08, "frontier": 0.17})
    assert "cheap CONTINUE" in p and "$0.100" in p
    lt = legal_for({"env": "lookup", "task_id": "b2b-1"})
    assert parse_action("<action>frontier CONTINUE</action>", lt) == ("frontier", "CONTINUE")
    assert parse_action("<action>frontier ANSWER</action>", lt) is None


def test_sim_episode_reward_consistent():
    task = json.loads(sim_rows(5, seed=9, budget=0.3)[2]["task"])
    R = SimEpisodeReward()
    a = R(task, "<action>mid ESCALATE</action>", 0.5)
    b = SimEpisodeReward()(task, "<action>mid ESCALATE</action>", 0.5)
    assert a == b and a.valid and a.action == "mid ESCALATE"
    h = make_sim("v1")
    cf = h.counterfactual(h.reset(sim_task(task)))
    # terminal first action: exactly the sim's own counterfactual (cost, expected quality)
    assert abs(a.cost - cf["ESCALATE@mid"][0]) < 1e-9 and abs(a.quality - cf["ESCALATE@mid"][1]) < 1e-6
    assert abs(a.reward - (a.quality - 0.5 * a.cost)) < 1e-12
    # non-terminal first action: + ANSWER on the same tier
    v = R(task, "<action>frontier VERIFY</action>", 0.0)
    assert abs(v.cost - cf["VERIFY@frontier"][0]) < 1e-9 and abs(v.quality - cf["VERIFY@frontier"][1]) < 1e-6
    r = R(task, "<action>cheap RETRIEVE</action>", 0.2)
    assert abs(r.cost - (0.5 * 0.02 + 0.005 + 0.02)) < 1e-9 and r.realized in (0.0, 1.0)
    # invalid: fixed penalty, accounted as the cheapest safe action
    bad = R(task, "I would pick mid.", 0.5)
    assert not bad.valid and bad.reward == INVALID_PENALTY and bad.action == "cheap ANSWER"
    assert abs(bad.cost - 0.02) < 1e-12


def _fake_log() -> list[str]:
    rows = []
    vals = {"b2b-1": {"cheap": (0.0, 0.03), "mid": (1.0, 0.08), "frontier": (1.0, 0.17)},
            "b2b-2": {"cheap": (1.0, 0.02), "mid": (1.0, 0.07), "frontier": (1.0, 0.15)},
            "b2b-3": {"cheap": (0.0, 0.04), "mid": (0.5, 0.09), "frontier": (1.0, 0.20)}}
    for tid, arms in vals.items():
        for tier, (q, c) in arms.items():
            rows.append(json.dumps({"task_id": tid, "steps": [], "quality": q, "total_cost": c, "total_latency": 1.0,
                                    "meta": {"arm": tier, "org": "b2b", "category": "lead_routing",
                                             "reward_metric": "exact_match"}}))
    rows.append('{"task_id": "b2b-4", "quality": 1.0, "total_cost": 0.01, "total_latency": 1, "meta": {"arm": "cheap"}}')
    rows.append('{"task_id": "b2b-9", "qual')        # torn final line of a live log
    return rows


def test_lookup_reward_fake_log():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "crm.jsonl")
        open(p, "w").write("\n".join(_fake_log()) + "\n")
        L = LookupReward(p)
    assert L.complete_tasks() == ["b2b-1", "b2b-2", "b2b-3"]        # b2b-4 lacks mid/frontier
    rows = lookup_rows(L, budget=0.10)
    assert len(rows) == 3 and "lead_routing" in rows[0]["user"]
    t1 = json.loads(rows[0]["task"])
    o = L(t1, "<action>mid CONTINUE</action>", 2.0)
    assert o.valid and (o.quality, o.cost) == (1.0, 0.08) and abs(o.reward - (1.0 - 2.0 * 0.08)) < 1e-12
    o = L(json.loads(rows[2]["task"]), "<action>frontier CONTINUE</action>", 1.0)
    assert (o.quality, o.cost) == (1.0, 0.20)
    bad = L(t1, "<action>mid ANSWER</action>", 1.0)                  # op not in the logged arms
    assert not bad.valid and bad.reward == INVALID_PENALTY and (bad.quality, bad.cost) == (0.0, 0.03)
    # the TRL adapter routes lookup tasks to the table
    fn = PricedRewardFn(DualAscentLambda(0.10), lookup=L)
    rs = fn(prompts=["p"] * 3, completions=["<action>cheap CONTINUE</action>"] * 3,
            task=[r["task"] for r in rows], trainer_state=None)
    assert rs == [0.0, 1.0, 0.0]


def test_dual_ascent_lambda_direction():
    lam = DualAscentLambda(budget=0.30)
    assert lam.lam == 0.0
    for _ in range(20):
        lam.update(0.50)                        # over budget -> price of cost rises
    hi = lam.lam
    assert hi > 0.0 and lam.cost_ema > 0.30
    for _ in range(5):
        lam.update(0.10)                        # under budget -> it falls
    assert lam.lam < hi
    for _ in range(200):
        lam.update(0.05)
    assert lam.lam == 0.0                       # projected at 0
    # same constants as train_offline.train_cgrpo
    assert (lam.lr, lam.kp, lam.ema) == (0.10, 2.0, 0.9)


def test_priced_reward_fn_drain_updates_lambda():
    rows = sim_rows(8, seed=1, budget=0.30)
    lam = DualAscentLambda(0.30)
    fn = PricedRewardFn(lam)
    comps = ["<action>frontier ESCALATE</action>"] * 6 + ["oops", "<action>cheap ANSWER</action>"]
    rs = fn(prompts=[r["user"] for r in rows], completions=comps, task=[r["task"] for r in rows])
    assert len(rs) == 8 and rs[6] == INVALID_PENALTY
    st = fn.drain()
    assert st["n"] == 8 and abs(st["invalid_rate"] - 1 / 8) < 1e-12
    assert st["action_histogram"]["frontier ESCALATE"] == 6 and st["action_histogram"]["INVALID"] == 1
    assert abs(st["mean_cost"] - (6 * 0.6 + 0.02 + 0.02) / 8) < 1e-9
    assert st["lambda_used"] == 0.0 and st["lambda"] > 0.0 and fn.drain() is None
    # degenerate groups: prompt groups whose completions are all identical (zero GRPO advantage)
    two = [r["task"] for r in rows[:2]]
    fn(completions=["<action>mid ANSWER</action>"] * 4 + ["<action>mid ANSWER</action>", "x", "x", "x"],
       task=[two[0]] * 4 + [two[1]] * 4)
    assert fn.drain()["degenerate_group_rate"] == 0.5


def test_price_in_prompt_and_sft_oracle_labels():
    from pace.llm.env_text import LAM_PLACEHOLDER, price_line, round_lam, with_price
    from pace.llm.sft import oracle_action, sft_rows
    r = sim_rows(3, seed=4, budget=0.30, price=True)[0]
    assert LAM_PLACEHOLDER in r["user"]
    filled = with_price(r["user"], 0.4321)
    assert "Price of cost: λ=0.4 " in filled and LAM_PLACEHOLDER not in filled
    assert round_lam(0.45) == 0.5 and round_lam(0.05) == 0.1 and price_line(1.0).startswith("Price of cost: λ=1.0")
    # the per-difficulty oracle respects price, and differs by difficulty away from lambda = 0.5
    assert oracle_action("easy", 0.3) == ("mid", "ESCALATE") and oracle_action("hard", 0.3) == ("frontier", "VERIFY")
    assert oracle_action("easy", 0.5) == oracle_action("hard", 0.5) == ("mid", "ESCALATE")
    assert oracle_action("easy", 1.2) == ("cheap", "ESCALATE") and oracle_action("hard", 1.2) == ("mid", "ESCALATE")
    rows = sft_rows(50, seed=1, budget=0.30)
    for x in rows:
        t = json.loads(x["task"])
        assert f"λ={x['lam']:.1f}" in x["user"] and t["lam"] == x["lam"]
        assert parse_action(x["completion"], legal_for(t)) == oracle_action(t["proxy"], x["lam"])
    # a price-conditioned task is charged the lambda it showed, not the live one
    fn = PricedRewardFn(DualAscentLambda(0.30, lam0=2.0))
    t = json.loads(rows[0]["task"])
    o = fn.score(t, "<action>mid ANSWER</action>")
    assert abs(o.reward - (o.quality - t["lam"] * o.cost)) < 1e-12
    fn(completions=["<action>mid ANSWER</action>"], task=[rows[0]["task"]])
    assert fn.drain()["lambda_shown"] == t["lam"]


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
