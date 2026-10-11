"""Trace schema: one TraceStep per controller decision, one Episode per task.

Every step logs the behavior policy's propensity (including exploration),
so any later candidate policy can be scored off-policy (pace.gate).
`state` and `legal` are optional extras: `legal` is the legal-action set the
behavior policy chose from (needed to evaluate a candidate's probability at
that step); `state` is a harness snapshot, present only when the harness is
simulable, enabling re-rolls from logged states (constrained GRPO on replay).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class TraceStep:
    task_id: str
    harness_version: str
    step: int
    features: list[float]
    action: int
    action_name: str
    propensity: float
    cost: float
    latency: float
    counterfactual: dict | None = None
    legal: list[str] | None = None
    state: dict | None = None


@dataclass
class Episode:
    task_id: str
    steps: list[TraceStep] = field(default_factory=list)
    quality: float = 0.0
    total_cost: float = 0.0
    total_latency: float = 0.0
    meta: dict | None = None

    @property
    def harness_version(self) -> str:
        return self.steps[0].harness_version if self.steps else ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        steps = [TraceStep(**s) for s in d.get("steps", [])]
        return cls(task_id=d["task_id"], steps=steps, quality=float(d["quality"]),
                   total_cost=float(d["total_cost"]), total_latency=float(d["total_latency"]),
                   meta=d.get("meta"))


def write_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), separators=(",", ":")) + "\n")
            n += 1
    return n


def append_jsonl(episode: Episode, path: str | Path) -> None:
    """Append one episode and flush (crash-safe incremental logging for paid runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(episode.to_dict(), separators=(",", ":")) + "\n")
        f.flush()


def iter_jsonl(path: str | Path) -> Iterator[Episode]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield Episode.from_dict(json.loads(line))


def read_jsonl(path: str | Path) -> list[Episode]:
    return list(iter_jsonl(path))
