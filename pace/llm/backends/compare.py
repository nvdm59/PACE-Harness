"""Tables for results/llm/backends/COMPARISON.md (phase-1b reproduction on three stacks).

    python -m pace.llm.backends.compare      # prints the markdown tables

Inputs: results/llm/train_log_phase1b.jsonl + eval_table_phase1b.md (TRL 1b, as logged),
results/llm/backends/{trl_dapo_1b,verl_dapo_1b,nemo_rl_dapo_1b}.jsonl (+ _meta.json) and
results/llm/backends/eval_phase1b_repro.json (one evaluator for every adapter).
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import load_jsonl

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results" / "llm"
B = RES / "backends"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def first_k(lines: list[dict], k: int = 20, pre_key: str = "identical_reward_group_rate_pre",
            post_key: str = "identical_reward_group_rate_post") -> dict:
    ls = [x for x in lines if "mean_reward" in x][:k]
    return {"steps": len(ls), "reward": _mean([x["mean_reward"] for x in ls]),
            "quality": _mean([x["mean_quality"] for x in ls]), "cost": _mean([x["mean_cost"] for x in ls]),
            "pre": _mean([x.get(pre_key) for x in ls]), "post": _mean([x.get(post_key) for x in ls]),
            "entropy": _mean([x.get("entropy") for x in ls]), "step_s": _mean([x.get("step_s") for x in ls]),
            "curve": [round(x["mean_reward"], 3) for x in ls[::4]]}


def f(x, fmt=".3f"):
    return "–" if x is None else format(x, fmt)


def main() -> str:
    arms = {}
    trl1b = load_jsonl(RES / "train_log_phase1b.jsonl")
    arms["(1) TRL phase 1b, as logged"] = first_k(trl1b, pre_key="degenerate_group_rate",
                                                   post_key="degenerate_group_rate")
    for name, fn in [("(2) TRL + dynamic sampling + clip 0.2/0.28", "trl_dapo_1b"),
                     ("(3) verl, DAPO recipe", "verl_dapo_1b"), ("(4) NeMo-RL, DAPO recipe", "nemo_rl_dapo_1b")]:
        p = B / f"{fn}.jsonl"
        if p.exists():
            arms[name] = first_k(load_jsonl(p))
    out = ["| arm | steps logged | reward (first 20) | quality | cost | identical-reward groups pre → post filter | "
           "entropy | s/step | reward curve (steps 1, 5, 9, 13, 17) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, a in arms.items():
        out.append(f"| {name} | {a['steps']} | {f(a['reward'])} | {f(a['quality'])} | {f(a['cost'])} | "
                   f"{f(a['pre'], '.0%')} → {f(a['post'], '.0%')} | {f(a['entropy'])} | {f(a['step_s'], '.1f')} | "
                   f"{', '.join(map(str, a['curve']))} |")
    ev = B / "eval_phase1b_repro.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        out += ["", "| policy (2,000 tasks, seed 777, λ = 0.44 in the prompt) | mode | quality | cost | J | within B | "
                "invalid | easy: top actions | hard: top actions | acts by difficulty |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in e["results"].items():
            for mode in ("greedy", "sampled"):
                s = r[mode]
                top = {k: ", ".join(f"{a} {v:.0%}" for a, v in list(d["mix"].items())[:2]) for k, d in s["by_proxy"].items()}
                out.append(f"| {name} | {mode} | {s['quality']:.3f} | {s['cost']:.3f} | {s['J']:.3f} | "
                           f"{'yes' if s['within_budget'] else 'NO'} | {s['invalid_rate']:.1%} | {top.get('easy')} | "
                           f"{top.get('hard')} | {'yes' if s['acts_by_difficulty'] else 'no'} |")
    md = "\n".join(out)
    print(md)
    return md


if __name__ == "__main__":
    main()
