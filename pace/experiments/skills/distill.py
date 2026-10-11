"""Distill per-category procedural SOQL notes from successful mid / frontier episodes.

Input: the counterfactual log (one episode per (task, tier arm); `meta.queries` holds the SOQL/SOSL
sequence, `quality` the CRMArena score). For each category, successful episodes (quality >= 0.5)
of the mid and frontier arms become exemplars (question -> query sequence -> answer, score). The
cheap arm is never a source.

The writer (mid tier, Sonnet 5.5) turns a category's exemplars into <= 3 notes of <= 40 words with
no task-specific values. One writer call per distinct SOURCE SET:

  full          all source tasks of the category (the deployable handbook entry)
  leave-one-out the source tasks minus one task t, for every source task t

Serving task t uses the variant whose source set is (category sources) \\ {t}; when t was not
itself a source the full entry already excludes it, so no extra call is made. Variants are cached
by (category, sorted source ids), so a run stopped at the cap resumes without repeating calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ...harness.crmarena import CATEGORIES

SUCCESS_Q = 0.5
SOURCE_ARMS = ("mid", "frontier")
WRITER_TIER = "mid"
PURPOSE = "skills:distill"
MAX_NOTES = 3
MAX_WORDS = 40
DISTILL_MAX_TOKENS = 1500
MAX_QUERIES_PER_EXEMPLAR = 8
QUERY_MAX_CHARS = 500
ANSWER_MAX_CHARS = 200

WRITER_SYSTEM = (
    "You write a procedural handbook for a smaller CRM agent. The agent answers business questions "
    "about a Salesforce org by issuing SOQL/SOSL queries, reading the rows, and replying with a final "
    "answer. You are shown how stronger agents solved several questions of ONE task category. Extract "
    "what transfers to NEW questions of the same category: which objects to query first, how records "
    "join (relationship / lookup fields), which filters and date windows matter, how to aggregate, and "
    "the pitfalls to avoid. Never include task-specific values (record Ids, names, dates, amounts, "
    "answers); describe the procedure, not the result.")

WRITER_USER = """Task category: {category}

Solved examples ({n}):
{exemplars}

Write at most {max_notes} concise procedural notes for this category, each at most {max_words} words, \
in the form "For <category> questions: query X first, join on Y, filter Z, ...". Use real object and \
field names seen in the queries. No task-specific values.
Output JSON only: {{"notes": ["...", "..."]}}"""


@dataclass
class Exemplar:
    task_id: str
    category: str
    arm: str
    question: str
    queries: list
    answer: str
    quality: float

    def render(self, i: int) -> str:
        qs = [str(q)[:QUERY_MAX_CHARS] for q in self.queries[:MAX_QUERIES_PER_EXEMPLAR]]
        body = "\n".join(f"    {j + 1}. {q}" for j, q in enumerate(qs)) or "    (no queries)"
        return (f"[{i}] ({self.arm} tier, score {self.quality:.2f})\n  Question: {self.question.strip()}\n"
                f"  Queries in order:\n{body}\n  Final answer: {str(self.answer)[:ANSWER_MAX_CHARS]}")


# ----------------------------------------------------------------- log
def load_log(path: str | Path, task_ids: Iterable[str] | None = None) -> list[dict]:
    """First episode per (task_id, arm), optionally restricted to `task_ids` (a log that another
    collector is still appending to may hold more tasks or a torn final line)."""
    keep = None if task_ids is None else set(map(str, task_ids))
    seen: set[tuple[str, str]] = set()
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid, arm = str(d["task_id"]), str((d.get("meta") or {}).get("arm", "policy"))
        if (keep is not None and tid not in keep) or (tid, arm) in seen:
            continue
        seen.add((tid, arm))
        out.append(d)
    return out


def _task_sort_key(tid: str):
    m = re.search(r"(\d+)$", tid)
    return (int(m.group(1)) if m else 0, tid)


def extract_exemplars(rows: list[dict], questions: dict[str, str], arms=SOURCE_ARMS,
                      min_quality: float = SUCCESS_Q) -> dict[str, list[Exemplar]]:
    """category -> successful exemplars of the source arms, ordered by (task idx, arm order)."""
    by_cat: dict[str, list[Exemplar]] = {}
    for d in rows:
        m = d.get("meta") or {}
        arm = str(m.get("arm"))
        if arm not in arms or float(d["quality"]) < min_quality:
            continue
        cat = str(m.get("category"))
        by_cat.setdefault(cat, []).append(Exemplar(
            task_id=str(d["task_id"]), category=cat, arm=arm, question=questions.get(str(d["task_id"]), ""),
            queries=list(m.get("queries") or [])[:-1] if m.get("end_reason") else list(m.get("queries") or []),
            answer=str(m.get("answer")), quality=float(d["quality"])))
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: (_task_sort_key(e.task_id), arms.index(e.arm)))
    return {c: by_cat[c] for c in _ordered(by_cat)}


def _ordered(cats) -> list[str]:
    cats = set(cats)
    return [c for c in CATEGORIES if c in cats] + sorted(c for c in cats if c not in CATEGORIES)


# ----------------------------------------------------------- variants
def source_sets(category_tasks: list[str], exemplars: list[Exemplar]) -> dict[str, list[str]]:
    """serving key -> source task ids. 'full' plus one entry per task of the category: the sources
    minus that task (leave-one-task-out)."""
    sources = sorted({e.task_id for e in exemplars}, key=_task_sort_key)
    out = {"full": sources}
    for t in sorted(category_tasks, key=_task_sort_key):
        out[t] = [s for s in sources if s != t]
    return out


def variant_key(category: str, source_ids: list[str]) -> str:
    return f"{category}|{','.join(source_ids)}"


# -------------------------------------------------------------- writer
_ID_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{15}(?:[A-Za-z0-9]{3})?\b")


def sanitize_note(note: str, max_words: int = MAX_WORDS) -> str:
    note = _ID_RE.sub("<Id>", " ".join(str(note).split()))
    words = note.split(" ")
    return " ".join(words[:max_words]).strip()


def parse_notes(text: str, max_notes: int = MAX_NOTES) -> tuple[list[str], bool]:
    notes = None
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and isinstance(obj.get("notes"), list):
                notes = [str(n) for n in obj["notes"]]
        except json.JSONDecodeError:
            notes = None
    ok = notes is not None
    if notes is None:   # fallback: bullet / numbered lines
        notes = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln) for ln in (text or "").splitlines()
                 if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", ln)]
    notes = [sanitize_note(n) for n in notes if str(n).strip()]
    return [n for n in notes if n][:max_notes], ok


def writer_messages(category: str, exemplars: list[Exemplar]) -> list[dict]:
    body = "\n\n".join(e.render(i + 1) for i, e in enumerate(exemplars))
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": WRITER_USER.format(category=category, n=len(exemplars), exemplars=body,
                                                           max_notes=MAX_NOTES, max_words=MAX_WORDS)}]


def write_notes(llm, category: str, exemplars: list[Exemplar]) -> dict:
    r = llm.complete(WRITER_TIER, writer_messages(category, exemplars), DISTILL_MAX_TOKENS,
                     f"category:{category}", PURPOSE)
    notes, ok = parse_notes(r.text)
    return {"notes": notes, "parse_ok": ok, "usd": round(r.usd, 8), "in_tokens": r.in_tokens,
            "out_tokens": r.out_tokens, "model": r.model}


# ------------------------------------------------------------ handbook
def empty_handbook(meta: dict | None = None) -> dict:
    return {"meta": dict(meta or {}), "categories": {}, "variants": {}}


def load_handbook(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else empty_handbook()


def save_handbook(hb: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(hb, indent=1, sort_keys=True) + "\n")
    tmp.replace(p)


def plan(tasks_by_cat: dict[str, list[str]], exemplars: dict[str, list[Exemplar]]) -> dict:
    """category -> {tasks, sources, serve: {key: source ids}} (no calls)."""
    out = {}
    for cat in _ordered(set(tasks_by_cat) | set(exemplars)):
        ex = exemplars.get(cat, [])
        sets = source_sets(tasks_by_cat.get(cat, []), ex)
        out[cat] = {"tasks": sorted(tasks_by_cat.get(cat, []), key=_task_sort_key), "sources": sets["full"],
                    "serve": {k: v for k, v in sets.items()}}
    return out


def distill(llm, tasks_by_cat: dict[str, list[str]], exemplars: dict[str, list[Exemplar]], hb: dict,
            save=None, log=print) -> dict:
    """Fill `hb` with every needed variant, one writer call per distinct non-empty source set; variants
    already present are reused. `save(hb)` is called after each call (crash/cap-safe)."""
    p = plan(tasks_by_cat, exemplars)
    for cat, entry in p.items():
        ex = exemplars.get(cat, [])
        serve = {}
        for key, srcs in entry["serve"].items():
            vk = variant_key(cat, srcs) if srcs else None
            serve[key] = vk
            if vk is None or vk in hb["variants"]:
                continue
            sel = [e for e in ex if e.task_id in set(srcs)]
            res = write_notes(llm, cat, sel)
            hb["variants"][vk] = {"category": cat, "source_task_ids": list(srcs),
                                  "exemplars": [{"task_id": e.task_id, "arm": e.arm, "quality": e.quality}
                                                for e in sel], **res}
            log(f"[distill] {cat:32s} sources={','.join(srcs):40s} notes={len(res['notes'])} ${res['usd']:.4f}")
            hb["categories"][cat] = {**entry, "serve": serve}
            if save:
                save(hb)
        full = serve.get("full")
        hb["categories"][cat] = {"tasks": entry["tasks"], "sources": entry["sources"], "serve": serve,
                                 "notes": hb["variants"][full]["notes"] if full else []}
        if save:
            save(hb)
    return hb


def notes_for_task(hb: dict, task_id: str, category: str) -> tuple[list[str], list[str], str | None]:
    """(notes, source task ids, variant key) to serve on `task_id`: never distilled from that task."""
    entry = hb.get("categories", {}).get(category)
    if not entry:
        return [], [], None
    vk = entry["serve"].get(task_id)
    if vk is None and task_id not in entry["serve"]:
        vk = entry["serve"].get("full")       # a task outside the distilled sample: the full entry
    if vk is None:
        return [], [], None
    v = hb["variants"][vk]
    if task_id in v["source_task_ids"]:
        raise AssertionError(f"leakage: variant {vk} was distilled from {task_id}")
    return list(v["notes"]), list(v["source_task_ids"]), vk


def total_usd(hb: dict) -> float:
    return sum(float(v.get("usd", 0.0)) for v in hb.get("variants", {}).values())


def exemplar_dicts(exemplars: dict[str, list[Exemplar]]) -> dict:
    return {c: [asdict(e) for e in v] for c, v in exemplars.items()}
