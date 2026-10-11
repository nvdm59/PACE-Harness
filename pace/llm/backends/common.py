"""Shared pieces of the backend entry points: the task set, cleanup and scoring.

Task set: the phase-1c training pool, `sim_rows(2000, seed=0, budget=0.30, price=True)`
(simulator v1, seeded shuffled legal actions, price-conditioned prompt). In the TRL runs the
price line carried the live learned lambda; in the backend smoke runs lambda is FIXED at
`FIXED_LAM` = 0.4 (phase 1c's lambda_bar rounded to the shown grid), filled into the prompt and
charged in the reward, so no dual-ascent loop has to be ported into each trainer.

Reward: `SimEpisodeReward` (first action + ANSWER on the same tier), r = quality - lambda * cost,
fixed -1 for an invalid completion. Every platform's logged rollouts are re-scored here with the
same function, so the per-step numbers in `results/llm/backends/` come from one code path.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ..env_text import LAM_PLACEHOLDER, SYSTEM, chat_messages, sim_rows
from ..rewards import Outcome, SimEpisodeReward

FIXED_LAM = 0.4
BUDGET = 0.30
N_TRAIN, TRAIN_SEED = 2000, 0          # phase-1c training pool
N_VAL, VAL_SEED = 64, 777              # a small slice of the eval seed (only so the trainers have a val set)
DATA_SOURCE = "pace_sim_priced"
MODEL = "Qwen/Qwen3-1.7B"
# end-of-turn markers a backend may leave on the decoded text (TRL decodes with skip_special_tokens)
_TAIL_TOKENS = ("<|im_end|>", "<|endoftext|>")

_SCORER: SimEpisodeReward | None = None


def price_text(lam: float) -> str:
    """The price line with lambda printed exactly (`:g`). For 0.4 this is byte-identical to
    env_text.price_line(0.4); for 0.44 it shows 0.44 rather than the 0.1-grid rounding."""
    return f"Price of cost: λ={float(lam):g} (your reward is quality − λ·cost)."


def task_rows(n: int = N_TRAIN, seed: int = TRAIN_SEED, lam: float = FIXED_LAM) -> list[dict]:
    """Rows {"user": prompt with the price line filled, "task": task dict incl. "lam", "index"}."""
    out = []
    for i, r in enumerate(sim_rows(n, seed=seed, budget=BUDGET, price=True)):
        task = {**json.loads(r["task"]), "lam": float(lam)}
        out.append({"index": i, "user": r["user"].replace(LAM_PLACEHOLDER, price_text(lam)), "task": task})
    return out


def messages(user: str) -> list[dict]:
    """System + user chat messages (the same as the TRL runs render)."""
    return chat_messages(user)


def clean_completion(text) -> str:
    """Strip end-of-turn markers some stacks leave on the decoded response; nothing else."""
    s = "" if text is None else str(text)
    changed = True
    while changed:
        changed = False
        t = s.rstrip()
        for tok in _TAIL_TOKENS:
            if t.endswith(tok):
                s, changed = t[: -len(tok)], True
    return s


def score(task: dict | str, completion) -> Outcome:
    """Our simulator reward for one completion, charged at the task's own lambda."""
    global _SCORER
    if _SCORER is None:
        _SCORER = SimEpisodeReward()
    t = json.loads(task) if isinstance(task, str) else task
    lam = float(t.get("lam", FIXED_LAM))
    return _SCORER(t, clean_completion(completion), lam)


def step_stats(records: Iterable[dict]) -> dict:
    """records: {"task": dict|json, "completion": str} of ONE optimizer step ->
    the same keys as results/llm/train_log_*.jsonl (re-scored here)."""
    recs = list(records)
    outs, groups, rgroups = [], defaultdict(set), defaultdict(list)
    for r in recs:
        t = json.loads(r["task"]) if isinstance(r["task"], str) else r["task"]
        o = score(t, r["completion"])
        outs.append(o)
        key = json.dumps(t, sort_keys=True)
        groups[key].add(clean_completion(r["completion"]).strip())
        rgroups[key].append(o.reward)
    n = len(outs)
    if not n:
        return {"n": 0}
    return {"n": n,
            "mean_reward": round(sum(o.reward for o in outs) / n, 5),
            "mean_quality": round(sum(o.quality for o in outs) / n, 5),
            "mean_cost": round(sum(o.cost for o in outs) / n, 5),
            "invalid_rate": round(sum(not o.valid for o in outs) / n, 4),
            "action_histogram": dict(Counter(o.action if o.valid else "INVALID" for o in outs).most_common()),
            "degenerate_group_rate": round(sum(len(v) == 1 for v in groups.values()) / len(groups), 4),
            # groups whose rewards are all equal (zero GRPO advantage); >= the identical-text rate
            "identical_reward_group_rate": round(sum(max(v) - min(v) <= 1e-9 for v in rgroups.values())
                                                 / len(rgroups), 4),
            "lambda": float(json.loads(recs[0]["task"])["lam"] if isinstance(recs[0]["task"], str)
                            else recs[0]["task"].get("lam", FIXED_LAM))}


def eval_stats(tasks: list[dict], completions: list[str], budget: float = BUDGET) -> dict:
    """Eval summary of one policy on a task list: quality, cost, J at the tasks' lambda, within budget
    (cost <= B + 0.02, as the phase-1 tables), invalid rate, and the action mix split by the observed
    difficulty proxy (the question is whether the policy acts differently on easy and hard tasks)."""
    outs = [score(t, c) for t, c in zip(tasks, completions)]
    n = len(outs)
    q = sum(o.quality for o in outs) / n
    c = sum(o.cost for o in outs) / n
    lam = float(tasks[0].get("lam", FIXED_LAM))
    split = {}
    for t, o in zip(tasks, outs):
        d = split.setdefault(t["proxy"], {"n": 0, "q": 0.0, "c": 0.0, "mix": Counter()})
        d["n"] += 1
        d["q"] += o.quality
        d["c"] += o.cost
        d["mix"][o.action if o.valid else "INVALID"] += 1
    by = {k: {"n": d["n"], "quality": round(d["q"] / d["n"], 4), "cost": round(d["c"] / d["n"], 4),
              "mix": {a: round(m / d["n"], 3) for a, m in d["mix"].most_common()}} for k, d in sorted(split.items())}
    mixes = [set(a for a, f in v["mix"].items() if f >= 0.05) for v in by.values()]
    return {"n": n, "quality": round(q, 4), "cost": round(c, 4), "J": round(q - lam * c, 4), "lambda": lam,
            "within_budget": c <= budget + 0.02, "invalid_rate": round(sum(not o.valid for o in outs) / n, 4),
            "mix": {a: round(m / n, 3) for a, m in Counter(o.action if o.valid else "INVALID" for o in outs).most_common()},
            "by_proxy": by,
            # top action per proxy bucket differs = the policy acts differently by difficulty
            "acts_by_difficulty": len({max(v["mix"], key=v["mix"].get) for v in by.values()}) > 1,
            "support_by_proxy_differs": len({frozenset(m) for m in mixes}) > 1}


def load_jsonl(path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass                          # torn last line of a killed run
    return out


def first_steps_summary(lines: list[dict], k: int = 20) -> dict:
    """Mean over the first k logged steps + the last of them (for the side-by-side table)."""
    ls = [x for x in lines if "mean_reward" in x][:k]
    if not ls:
        return {}
    keys = ["mean_reward", "mean_quality", "mean_cost", "invalid_rate"]
    out = {"steps": len(ls)}
    for key in keys:
        vals = [x[key] for x in ls if x.get(key) is not None]
        out[f"{key}_mean"] = round(sum(vals) / len(vals), 4) if vals else None
        out[f"{key}_first"] = ls[0].get(key)
        out[f"{key}_last"] = ls[-1].get(key)
    dg = [x["degenerate_group_rate"] for x in ls if x.get("degenerate_group_rate") is not None]
    out["degenerate_group_rate_mean"] = round(sum(dg) / len(dg), 4) if dg else None
    return out


def markdown_table(lines: list[dict], title: str) -> str:
    rows = [f"### {title}", "",
            "| step | reward | quality | cost | invalid | degenerate groups | s/step | top actions |",
            "|---|---|---|---|---|---|---|---|"]
    for x in lines:
        if "mean_reward" not in x:
            continue
        top = ", ".join(f"{a} {c}" for a, c in list(x.get("action_histogram", {}).items())[:3])
        sec = x.get("step_s")
        rows.append(f"| {x['step']} | {x['mean_reward']:.3f} | {x['mean_quality']:.3f} | {x['mean_cost']:.3f} | "
                    f"{x['invalid_rate']:.1%} | {x.get('degenerate_group_rate', 0) or 0:.0%} | "
                    f"{'' if sec is None else f'{sec:.1f}'} | {top} |")
    return "\n".join(rows) + "\n"


__all__ = ["FIXED_LAM", "BUDGET", "N_TRAIN", "TRAIN_SEED", "N_VAL", "VAL_SEED", "DATA_SOURCE", "MODEL", "SYSTEM",
           "task_rows", "messages", "clean_completion", "score", "step_stats", "load_jsonl",
           "first_steps_summary", "markdown_table", "price_text", "eval_stats"]
