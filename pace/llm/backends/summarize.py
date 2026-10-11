"""Write a backend smoke run's results: results/llm/backends/<run>.jsonl (one line per step, the
TRL train-log schema, re-scored with our reward), <run>_meta.json and <run>.md, and a side-by-side
of the first steps against TRL phase 1 (results/llm/train_log_phase1.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path

from .common import FIXED_LAM, first_steps_summary, load_jsonl, markdown_table

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "llm" / "backends"
TRL_LOG = REPO / "results" / "llm" / "train_log_phase1.jsonl"


def write_results(platform: str, run: str, res: dict, out_dir: Path = OUT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = res.get("lines") or []
    (out_dir / f"{run}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    meta = {k: v for k, v in res.items() if k not in ("lines", "log_tail")}
    meta["platform"] = platform
    meta["fixed_lambda"] = lines[0].get("lambda", FIXED_LAM) if lines else FIXED_LAM
    (out_dir / f"{run}_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    lam = lines[0].get("lambda", FIXED_LAM) if lines else FIXED_LAM
    shape = "8 prompts x 8 completions" if "smoke" in run else "4 prompts x 16 completions (after group filtering)"
    md = [f"# {platform} run `{run}`", "",
          f"Status: **{res.get('status')}**; container wall clock {res.get('container_s')} s on {res.get('gpu')}; "
          f"first step logged {res.get('first_step_s')} s after container start; versions {res.get('versions')}.",
          "", f"Fixed price λ = {lam} in the prompt and in the reward; {shape} per step; "
          "every number re-scored from the logged completions with `pace.llm.rewards.SimEpisodeReward`.", ""]
    md.append(markdown_table(lines, f"{platform}: per step") if lines else "_No training step was logged._\n")
    trl = first_steps_summary(load_jsonl(TRL_LOG)) if TRL_LOG.exists() else {}
    mine = first_steps_summary(lines)
    if mine:
        md += ["", side_by_side({platform: mine, "TRL phase 1 (λ learned from 0)": trl}), ""]
    (out_dir / f"{run}.md").write_text("\n".join(md))
    print("\n".join(md))
    return meta


def side_by_side(summaries: dict[str, dict]) -> str:
    rows = ["| platform | steps | reward (mean) | quality (mean) | cost (mean) | invalid (first → last) | degenerate groups |",
            "|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        if not s:
            continue
        rows.append(f"| {name} | {s['steps']} | {s['mean_reward_mean']:.3f} | {s['mean_quality_mean']:.3f} | "
                    f"{s['mean_cost_mean']:.3f} | {s['invalid_rate_first']:.1%} → {s['invalid_rate_last']:.1%} | "
                    f"{_pct(s.get('degenerate_group_rate_mean'))} |")
    return "\n".join(rows)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"
