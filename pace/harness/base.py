"""Abstract frozen harness. PACE never changes the harness; it only decides,
per step, which (op, tier) action the harness executes next.

A harness exposes: the action vocabulary, seeded task sampling, an observable per-step
state, a legal-action set, `step`, and a quality score of the terminal state. Optional
capabilities (counterfactual arms, snapshot / restore) are what make replay training
possible; only the simulator has them."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable


@dataclass
class HarnessState:
    """Observable state the controller featurizes, plus harness-private `extra`.

    `difficulty_proxy` is what the harness can OBSERVE about difficulty
    (e.g. a cheap classifier or question heuristics) -- never the latent truth.
    """
    task_id: str
    difficulty_proxy: str = "easy"   # "easy" | "hard"
    evidence: int = 0
    verified: bool = False
    step: int = 0
    cost_so_far: float = 0.0
    latency_so_far: float = 0.0
    done: bool = False
    version: str = "v1"
    extra: dict = field(default_factory=dict)

    def clone(self) -> "HarnessState":
        # extra holds only scalars by convention; a shallow dict copy suffices
        return replace(self, extra=dict(self.extra))


class Harness(ABC):
    version: str = "base"

    @property
    @abstractmethod
    def all_actions(self) -> list[str]:
        """Every action name this harness version can ever expose."""

    @abstractmethod
    def sample_tasks(self, n: int, seed: int = 0) -> Iterable[Any]:
        """Yield n tasks (harness-specific objects)."""

    @abstractmethod
    def reset(self, task: Any) -> HarnessState: ...

    @abstractmethod
    def legal_actions(self, state: HarnessState) -> list[str]: ...

    @abstractmethod
    def step(self, state: HarnessState, action: str) -> tuple[HarnessState, bool, float, float]:
        """Execute `action`; return (new_state, done, cost, latency). Must not mutate `state`."""

    @abstractmethod
    def quality(self, state: HarnessState) -> float:
        """Realized quality in [0,1] of a terminal state."""

    # ---- optional capabilities -------------------------------------------
    def featurize(self, state: HarnessState) -> list[float]:
        """State -> controller features. Default: the sim feature map in pace.features."""
        from ..features import featurize  # lazy: pace.features imports this module
        return featurize(state)

    @property
    def n_features(self) -> int:
        from ..features import N_FEATURES
        return N_FEATURES

    def episode_meta(self, state: HarnessState) -> dict | None:
        """Optional per-episode metadata logged with the Episode (answer, category, ...)."""
        return None

    def counterfactual(self, state: HarnessState) -> dict | None:
        """action -> (cost, expected_quality) when the harness is simulable."""
        return None

    def expected_quality(self, state: HarnessState) -> float:
        """Ground-truth expected quality of a terminal state (sim only)."""
        return self.quality(state)

    def snapshot(self, state: HarnessState) -> dict | None:
        return None

    def restore(self, snap: dict) -> HarnessState:
        raise NotImplementedError(f"{type(self).__name__} cannot restore logged states")

    @staticmethod
    def state_to_dict(state: HarnessState) -> dict:
        return asdict(state)
