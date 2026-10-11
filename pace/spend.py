"""Spend ledger with a hard cap.

Every priced model call is appended to `<root>/spend.jsonl` as
    {ts, model, in_tokens, out_tokens, usd, task_id, purpose}
and `<root>/spend_total.json` is rewritten with the running totals.

The cap is checked BEFORE a call: the call's worst case is priced from an
input-token estimate plus the call's `max_tokens` (the output upper bound),
and `SpendCapReached` is raised if `spent + worst_case > cap`. Realized spend
therefore never exceeds the cap as long as the input estimate is an upper
bound; `estimate_input_tokens` is deliberately conservative (2 chars/token,
where English prose and Salesforce JSON run about 3-4).

The ledger is cumulative across runs: on start it re-reads spend.jsonl, so a
resumed collection with `--cap 100` after stopping at `--cap 50` has $50 of
headroom, not $100.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .prices import bare_model, call_cost, price_of

STOP_MARKER = "STOPPED_AT_CAP"
CHARS_PER_TOKEN_UPPER = 2.0      # conservative: real text is ~3-4 chars/token
MESSAGE_OVERHEAD_TOKENS = 8      # per-message role/format framing


class SpendCapReached(RuntimeError):
    def __init__(self, spent: float, worst_case: float, cap: float, model: str = "", purpose: str = ""):
        self.spent, self.worst_case, self.cap = spent, worst_case, cap
        self.model, self.purpose = model, purpose
        super().__init__(f"spend cap ${cap:g} reached: spent ${spent:.4f}, next call ({model}, {purpose}) "
                         f"could cost up to ${worst_case:.4f}")


class SpendCap:
    """A hard USD cap on cumulative ledger spend. `None` means uncapped."""

    def __init__(self, cap_usd: float | None):
        self.cap_usd = None if cap_usd is None else float(cap_usd)

    def check(self, spent: float, worst_case: float, model: str = "", purpose: str = "") -> None:
        if self.cap_usd is not None and spent + worst_case > self.cap_usd:
            raise SpendCapReached(spent, worst_case, self.cap_usd, model, purpose)

    def headroom(self, spent: float) -> float:
        return float("inf") if self.cap_usd is None else self.cap_usd - spent


def estimate_input_tokens(messages: Iterable[dict] | str) -> int:
    """Upper-bound-ish input token estimate for a chat request (no network)."""
    if isinstance(messages, str):
        return int(len(messages) / CHARS_PER_TOKEN_UPPER) + MESSAGE_OVERHEAD_TOKENS
    n = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):        # content blocks
            c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        n += int(len(c or "") / CHARS_PER_TOKEN_UPPER) + MESSAGE_OVERHEAD_TOKENS
    return n


class SpendLedger:
    def __init__(self, root: str | Path = "traces", cap: float | SpendCap | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.root / "spend.jsonl"
        self.total_path = self.root / "spend_total.json"
        self.cap = cap if isinstance(cap, SpendCap) else SpendCap(cap)
        self.total = 0.0
        self.n_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.by_model: dict[str, float] = defaultdict(float)
        self.by_purpose: dict[str, float] = defaultdict(float)
        self.tasks: set[str] = set()
        self._load()

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        if not self.jsonl.exists():
            return
        for line in self.jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # a torn final line from a hard kill
                continue
            self._accumulate(r)

    def _accumulate(self, r: dict) -> None:
        usd = float(r["usd"])
        self.total += usd
        self.n_calls += 1
        self.in_tokens += int(r.get("in_tokens", 0))
        self.out_tokens += int(r.get("out_tokens", 0))
        self.by_model[r.get("model", "?")] += usd
        self.by_purpose[r.get("purpose", "?")] += usd
        if r.get("task_id") is not None:
            self.tasks.add(str(r["task_id"]))

    # --------------------------------------------------------------- guard
    @staticmethod
    def worst_case(model: str, in_tokens_est: int, max_tokens: int) -> float:
        return call_cost(model, int(in_tokens_est), int(max_tokens))

    def guard(self, model: str, in_tokens_est: int, max_tokens: int, purpose: str = "") -> float:
        """Raise SpendCapReached if this call could push spend past the cap. Returns the worst case."""
        price_of(model)                      # unknown model -> KeyError before any spend
        wc = self.worst_case(model, in_tokens_est, max_tokens)
        self.cap.check(self.total, wc, bare_model(model), purpose)
        return wc

    # -------------------------------------------------------------- record
    def record(self, model: str, in_tokens: int, out_tokens: int, task_id: str | None = None,
               purpose: str = "", usd: float | None = None, cache_write_tokens: int = 0,
               cache_read_tokens: int = 0, extra: dict | None = None) -> float:
        model = bare_model(model)
        if usd is None:
            usd = call_cost(model, in_tokens, out_tokens, cache_write_tokens, cache_read_tokens)
        r = {"ts": round(time.time(), 3), "model": model, "in_tokens": int(in_tokens),
             "out_tokens": int(out_tokens), "usd": round(float(usd), 8), "task_id": task_id,
             "purpose": purpose}
        if cache_write_tokens or cache_read_tokens:
            r["cache_write_tokens"], r["cache_read_tokens"] = int(cache_write_tokens), int(cache_read_tokens)
        if extra:
            r.update(extra)
        with self.jsonl.open("a") as f:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._accumulate(r)
        self.write_total()
        return float(usd)

    # ------------------------------------------------------------- outputs
    def snapshot(self) -> dict:
        return {"total_usd": round(self.total, 6), "n_calls": self.n_calls, "in_tokens": self.in_tokens,
                "out_tokens": self.out_tokens, "n_tasks": len(self.tasks),
                "cap_usd": self.cap.cap_usd,
                "by_model": {k: round(v, 6) for k, v in sorted(self.by_model.items())},
                "by_purpose": {k: round(v, 6) for k, v in sorted(self.by_purpose.items())},
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def write_total(self) -> None:
        tmp = self.total_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=1))
        os.replace(tmp, self.total_path)

    def summary(self) -> str:
        cap = "uncapped" if self.cap.cap_usd is None else f"cap ${self.cap.cap_usd:g}"
        per = self.total / len(self.tasks) if self.tasks else 0.0
        models = ", ".join(f"{k} ${v:.2f}" for k, v in sorted(self.by_model.items()))
        return (f"[spend] ${self.total:.4f} of {cap} | {self.n_calls} calls, {len(self.tasks)} tasks, "
                f"${per:.4f}/task | in {self.in_tokens:,} out {self.out_tokens:,} tok | {models or '-'}")

    def write_stop_marker(self, exc: SpendCapReached | None = None) -> Path:
        self.write_total()
        p = self.root / STOP_MARKER
        body = {"stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "total_usd": round(self.total, 6),
                "cap_usd": self.cap.cap_usd, "n_calls": self.n_calls, "n_tasks": len(self.tasks)}
        if exc is not None:
            body.update({"next_call_worst_case_usd": round(exc.worst_case, 6), "next_call_model": exc.model,
                         "next_call_purpose": exc.purpose})
        p.write_text(json.dumps(body, indent=1) + "\n")
        return p

    def clear_stop_marker(self) -> None:
        p = self.root / STOP_MARKER
        if p.exists():
            p.unlink()
