"""Serve the handbook to a tier: a "procedural hints" block appended to the system prompt.

Nothing in the core harness changes. `HintedBackend` wraps the backend's prompt builder
(`system_prompt`) and appends the hint block for the task's category, built from the
leave-one-task-out variant (`distill.notes_for_task`), so the agent never sees notes distilled from
its own task. `PrefixedLLM` wraps `harness.llm` the way `ssr.refine.tag_purpose` does, so every call
of a served episode (agent turns and the CRMArena extractor) is billed under `skills:<purpose>`.
"""
from __future__ import annotations

from ...harness.crmarena import CRMArenaHarness
from .distill import notes_for_task

HINT_HEADER = ("\n\nProcedural hints for this category (distilled from other solved questions of the "
               "same category; general guidance, not answers):")


def hint_block(notes: list[str]) -> str:
    if not notes:
        return ""
    return HINT_HEADER + "".join(f"\n- {n}" for n in notes)


class PrefixedLLM:
    """Delegates to a PricedLLM, billing each call under `<prefix><purpose>`."""

    def __init__(self, inner, prefix: str = "skills:"):
        self.inner, self.prefix = inner, prefix

    def complete(self, tier_or_model, messages, max_tokens, task_id, purpose):
        return self.inner.complete(tier_or_model, messages, max_tokens, task_id, self.prefix + purpose)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class HintedBackend:
    """Wraps a CRMArena backend; only `system_prompt` changes (hint block appended)."""

    def __init__(self, inner, handbook: dict):
        self.inner, self.handbook = inner, handbook
        self.served: dict[str, dict] = {}

    def system_prompt(self, task, eval_mode="aided", privacy_aware=False):
        base = self.inner.system_prompt(task, eval_mode, privacy_aware)
        notes, srcs, vk = notes_for_task(self.handbook, task.task_id, task.category)
        self.served[task.task_id] = {"n_notes": len(notes), "source_task_ids": srcs, "variant": vk}
        return base + hint_block(notes)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SkilledCRMArenaHarness(CRMArenaHarness):
    """CRMArenaHarness whose backend carries the hint block; logs which variant was served."""

    def __init__(self, handbook: dict, **kw):
        super().__init__(**kw)
        self.backend = HintedBackend(self.backend, handbook)

    def episode_meta(self, state):
        m = super().episode_meta(state)
        m["skills"] = self.backend.served.get(state.task_id, {"n_notes": 0, "source_task_ids": [], "variant": None})
        return m
