"""Skill handbook for the cheap tier (pace/experiments/skills) -- offline, no API, no Salesforce.

    python -m pytest tests/test_skills.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pace.collect import collect_crmarena  # noqa: E402
from pace.experiments.skills import analyze as an  # noqa: E402
from pace.experiments.skills.distill import (distill, empty_handbook, extract_exemplars, load_log,  # noqa: E402
                                             notes_for_task, parse_notes, source_sets)
from pace.experiments.skills.fake import (FakeBackend, FakeCompletion, fake_log_rows, make_fake_tasks,  # noqa: E402
                                          write_fake_log)
from pace.experiments.skills.serve import HINT_HEADER, PrefixedLLM, SkilledCRMArenaHarness  # noqa: E402
from pace.harness.crmarena import CRMArenaHarness, PricedLLM  # noqa: E402
from pace.spend import SpendLedger  # noqa: E402


def _setup(d):
    tasks = make_fake_tasks()
    comp = FakeCompletion(tasks)
    ledger = SpendLedger(os.path.join(d, "distill"), 1.0)
    h = CRMArenaHarness(ledger=ledger, llm=PricedLLM(ledger, completion_fn=comp), backend=FakeBackend(tasks),
                        tasks=tasks)
    qs = {t.task_id: t.query for t in tasks}
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t.task_id)
    return tasks, comp, ledger, h, qs, by_cat


def test_exemplar_extraction_excludes_cheap_and_failures():
    tasks = make_fake_tasks()
    rows = fake_log_rows(tasks)
    ex = extract_exemplars(rows, {t.task_id: t.query for t in tasks})
    assert list(ex) == ["handle_time"]                     # case_routing: no mid/frontier success
    got = [(e.task_id, e.arm) for e in ex["handle_time"]]
    assert got == [("b2b-1", "mid"), ("b2b-1", "frontier"), ("b2b-2", "mid")]
    e = ex["handle_time"][0]
    assert e.queries == ["SELECT Id, OwnerId FROM Case WHERE Subject = 'b2b-1'"]   # final answer stripped
    assert e.question.startswith("[b2b-1]") and e.quality == 1.0
    # load_log keeps the first episode per (task, arm) and restricts to the given ids
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "log.jsonl")
        write_fake_log(tasks, p)
        with open(p, "a") as f:
            f.write(json.dumps(rows[0]) + "\n{torn")
        assert len(load_log(p)) == 9 and len(load_log(p, ["b2b-3"])) == 3


def test_leave_one_task_out_exclusion():
    with tempfile.TemporaryDirectory() as d:
        tasks, comp, ledger, h, qs, by_cat = _setup(d)
        ex = extract_exemplars(fake_log_rows(tasks), qs)
        assert source_sets(by_cat["handle_time"], ex["handle_time"]) == {
            "full": ["b2b-1", "b2b-2"], "b2b-1": ["b2b-2"], "b2b-2": ["b2b-1"]}
        hb = distill(h.llm, by_cat, ex, empty_handbook(), log=lambda *_: None)
        for t in tasks:
            notes, srcs, vk = notes_for_task(hb, t.task_id, t.category)
            assert t.task_id not in srcs
        assert notes_for_task(hb, "b2b-1", "handle_time")[1] == ["b2b-2"]
        assert notes_for_task(hb, "b2b-3", "case_routing") == ([], [], None)
        # the writer for the b2b-1 variant never saw b2b-1's exemplars
        writer_msgs = [c["messages"][1]["content"] for c in comp.calls]
        loo1 = [m for m in writer_msgs if "[b2b-2]" in m and "[b2b-1]" not in m]
        assert len(loo1) == 1 and len(comp.calls) == 3      # full + two leave-one-out variants
        # rerun reuses cached variants: no new calls
        distill(h.llm, by_cat, ex, hb, log=lambda *_: None)
        assert len(comp.calls) == 3
        # a tampered handbook that would leak is refused
        hb["variants"][hb["categories"]["handle_time"]["serve"]["b2b-1"]]["source_task_ids"].append("b2b-1")
        try:
            notes_for_task(hb, "b2b-1", "handle_time")
            raise RuntimeError("leak not detected")
        except AssertionError:
            pass


def test_notes_parse_and_sanitize():
    notes, ok = parse_notes('{"notes": ["a b c", "use Id 005Wt000003NFKoIAO", "x", "too many"]}')
    assert ok and notes == ["a b c", "use Id <Id>", "x"]
    notes, ok = parse_notes("- " + " ".join(["w"] * 60) + "\n2. second")
    assert not ok and len(notes[0].split()) == 40 and notes[1] == "second"
    # object / field names survive the Id filter
    assert parse_notes('{"notes": ["join OpportunityLineItem on Pricebook2Id"]}')[0] == [
        "join OpportunityLineItem on Pricebook2Id"]


def test_hint_injected_exactly_once_and_ledger_purposes():
    with tempfile.TemporaryDirectory() as d:
        tasks, comp, ledger, h, qs, by_cat = _setup(d)
        hb = distill(h.llm, by_cat, extract_exemplars(fake_log_rows(tasks), qs), empty_handbook(),
                     log=lambda *_: None)
        purposes = {json.loads(l)["purpose"] for l in open(os.path.join(d, "distill", "spend.jsonl"))}
        assert purposes == {"skills:distill"}
        sledger = SpendLedger(os.path.join(d, "serve"), 1.0)
        comp2 = FakeCompletion(tasks)
        sh = SkilledCRMArenaHarness(hb, ledger=sledger, llm=PrefixedLLM(PricedLLM(sledger, completion_fn=comp2)),
                                    backend=FakeBackend(tasks), tasks=tasks)
        out = os.path.join(d, "skills.jsonl")
        assert collect_crmarena(sh, out, 3, "counterfactual", ["cheap"], log=lambda *_: None) == 0
        for c in comp2.calls:
            joined = "\n".join(m["content"] for m in c["messages"])
            tid = "b2b-3" if "[b2b-3]" in joined else None
            assert joined.count(HINT_HEADER) == (0 if tid else 1)
            assert c["messages"][0]["content"].count(HINT_HEADER) == (0 if tid else 1)
        recs = [json.loads(l) for l in open(os.path.join(d, "serve", "spend.jsonl"))]
        assert {r["purpose"] for r in recs} == {"skills:agent:cheap"}
        eps = [json.loads(l) for l in open(out)]
        assert {e["meta"]["skills"]["n_notes"] > 0 for e in eps if e["task_id"] != "b2b-3"} == {True}
        assert all(e["task_id"] not in e["meta"]["skills"]["source_task_ids"] for e in eps)


def test_dry_run_end_to_end():
    from pace.experiments.skills import run as R
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            for cmd in (["distill", "--cap", "1"], ["serve", "--cap", "1"], ["analyze"]):
                try:
                    R.main(cmd + ["--dry-run", "--n-tasks", "3"])
                except SystemExit as e:
                    assert e.code == 0, (cmd, e.code)
            txt = open(os.path.join(R.DRYRUN, "RESULTS.md")).read()
            assert "| cheap+skills | 0.667" in txt and "net +2" in txt
            # cap semantics: an exhausted serve ledger stops with exit 3 and writes the marker
            try:
                R.main(["serve", "--cap", "0.000001", "--dry-run", "--n-tasks", "3"])
            except SystemExit as e:
                assert e.code == 3
            assert os.path.exists(os.path.join(R.DRYRUN, "ledger_serve", "STOPPED_AT_CAP"))
        finally:
            os.chdir(cwd)


def test_oracle_tiebreak():
    tab = {"a": {"cheap": (1.0, 0.1), "mid": (1.0, 0.2)}, "b": {"cheap": (0.0, 0.1), "mid": (1.0, 0.2)}}
    o = an.oracle(tab, ["cheap", "mid"])
    assert o["quality"] == 1.0 and abs(o["cost"] - 0.15) < 1e-12 and o["picks"] == {"cheap": 1, "mid": 1}


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("ok")
