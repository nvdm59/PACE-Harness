"""CLI for the priced-SSR experiment.

  python -m pace.experiments.ssr.run run --policy tree --n-tasks 50 --org b2b --tiers cheap,mid \
      --cap 50 [--initial from-log --initial-arm cheap] [--dry-run]
  python -m pace.experiments.ssr.run analyze [--log traces/ssr_b2b_tree.jsonl] \
      [--out results/ssr/RESULTS.md --csv results/ssr/frontier.csv]

`run --policy tree` collects the full refinement tree (every tier at every round, up to
--max-rounds), so `analyze` can score every policy exactly. `run --policy <spec>` serves one
policy live and logs only its branch (cheaper; other policies then are not evaluable).

Real runs make paid API calls and Salesforce queries behind the same hard cap as pace.collect:
--cap is required, the ledger is cumulative (traces/spend.jsonl) and the tree log is resumable.
--dry-run swaps in the fake world (pace/experiments/ssr/fake.py): no key, no network, no spend.
Exit codes as pace.collect: 0 done, 3 stopped at the cap, 4 a model call failed after retries.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from ...collect import EXIT_LLM_FAILED, EXIT_STOPPED_AT_CAP, FixedArmPolicy, parse_tiers, run_episode
from ...harness.crmarena import TURN0_IDX, CRMArenaHarness, CRMTask, LLMCallFailed, PricedLLM
from ...prices import TIERS
from ...spend import SpendCapReached, SpendLedger
from ...trace import read_jsonl
from .policy import parse_policy
from .refine import K_SAMPLES, RESOLVE_MAX_TURNS, Claim, Probe, normalize_answer, probe, refine_step
from .tree import STOP, SSRState, append_record, build_trees, done_keys, read_records, refine_action

DRYRUN_DIR = Path("results/ssr/dryrun")


# ---------------------------------------------------------------- initial
class Initial:
    """The initial answer episode of a (task, tier): from the counterfactual log (free: the
    episode and its CRMArena score are already paid for) or run fresh through the harness."""

    def __init__(self, source: str, log_path: str | None = None):
        self.source = source
        self.by: dict[tuple[str, str], object] = {}
        if source == "from-log":
            if not log_path or not Path(log_path).exists():
                raise SystemExit(f"--initial from-log needs an existing log; got {log_path!r}")
            for ep in read_jsonl(log_path):
                arm = (ep.meta or {}).get("arm") or ep.steps[0].action_name.split("@")[1]
                self.by[(ep.task_id, arm)] = ep

    def has(self, task_id: str, arm: str) -> bool:
        return self.source == "fresh" or (task_id, arm) in self.by

    @staticmethod
    def _queries(m: dict, answer: str) -> list[str]:
        """CRMArena's `actions` trail ends with the answer itself; keep only the queries."""
        return [q for q in (m.get("queries") or []) if normalize_answer(q) != normalize_answer(answer)]

    def get(self, harness, task: CRMTask, arm: str, rng: random.Random) -> dict:
        if self.source == "from-log":
            ep = self.by[(task.task_id, arm)]
            m = ep.meta or {}
            ans = str(m.get("answer") or "None")
            return {"answer": ans, "quality": float(ep.quality),
                    "cost": float(ep.total_cost), "queries": self._queries(m, ans), "source": "log",
                    "eval_usd": float(m.get("eval_cost_usd", 0.0))}
        pol = FixedArmPolicy(arm, harness.all_actions, len(TIERS), TURN0_IDX)
        ep = run_episode(harness, pol, task, rng, log_counterfactual=False, log_state=False)
        m = ep.meta or {}
        ans = str(m.get("answer") or "None")
        return {"answer": ans, "quality": float(ep.quality),
                "cost": float(ep.total_cost), "queries": self._queries(m, ans), "source": "fresh",
                "eval_usd": float(m.get("eval_cost_usd", 0.0))}


# ------------------------------------------------------------ collection
def node_record(task: CRMTask, arm: str, path: list[str], answer: str, quality: float, initial_cost: float,
                pr: Probe | None, backend: str, extra: dict | None = None) -> dict:
    r = {"kind": "node", "task_id": task.task_id, "category": task.category, "reward_metric": task.reward_metric,
         "org": task.org_type, "initial_arm": arm, "path": list(path), "round": len(path), "answer": answer,
         "quality": float(quality), "initial_cost": float(initial_cost),
         "probe": pr.to_dict() if pr is not None else None, "backend": backend}
    if extra:
        r.update(extra)
    return r


def edge_record(task: CRMTask, arm: str, path: list[str], res, backend: str) -> dict:
    return {"kind": "edge", "task_id": task.task_id, "initial_arm": arm, "path": list(path), "round": len(path),
            "tier": res.tier, "refine": res.to_dict(), "backend": backend}


class Collector:
    def __init__(self, harness, out: str | Path, tiers: list[str], max_rounds: int = 2, k: int = K_SAMPLES,
                 resolve_max_turns: int = RESOLVE_MAX_TURNS, decompose_tier: str = "cheap", backend: str = "crmarena",
                 log=print):
        self.h, self.out, self.tiers = harness, Path(out), list(tiers)
        self.max_rounds, self.k, self.turns, self.dtier = int(max_rounds), int(k), int(resolve_max_turns), decompose_tier
        self.backend, self.log = backend, log
        self.done = done_keys(self.out)
        self.n_new = 0

    def _write(self, r: dict) -> None:
        append_record(r, self.out)
        self.done.add((r["kind"], r["task_id"], r["initial_arm"], tuple(r["path"])))
        self.n_new += 1

    def _root_extra(self, path, cost_so_far) -> dict:
        x = {"cost_so_far": cost_so_far}
        if not path:
            init = getattr(self, "_init", None) or {}
            x.update({"queries": init.get("queries") or [], "initial_source": init.get("source"),
                      "initial_eval_usd": init.get("eval_usd", 0.0)})
        return x

    def _probe(self, task, answer, queries) -> Probe:
        return probe(self.h, task, answer, queries, self.dtier)

    def _refine(self, task, arm, path, answer, quality, pr: Probe, tier, queries, cost_so_far):
        res = refine_step(self.h, task, answer, quality, pr.claims, tier, queries, self.k, self.turns, len(path) + 1)
        child_path = path + [tier]
        self._write(edge_record(task, arm, child_path, res, self.backend))
        self.log(f"[ssr] {task.task_id} arm={arm} path={'/'.join(child_path):12s} q {quality:.2f} -> {res.quality_after:.2f}"
                 f" conf_min {res.conf_min:.2f} ${res.usd:.4f} ({res.calls} calls, {res.worker_turns} worker turns)")
        return res, child_path

    # ------------------------------------------------------------ tree
    def expand_tree(self, task, arm, path, answer, quality, initial_cost, queries, cost_so_far) -> None:
        key = ("node", task.task_id, arm, tuple(path))
        if len(path) >= self.max_rounds:
            if key not in self.done:
                self._write(node_record(task, arm, path, answer, quality, initial_cost, None, self.backend))
            return
        if key in self.done:
            return self._expand_existing(task, arm, path, answer, quality, initial_cost, queries, cost_so_far)
        pr = self._probe(task, answer, queries)
        self._write(node_record(task, arm, path, answer, quality, initial_cost, pr, self.backend,
                                self._root_extra(path, cost_so_far)))
        for tier in self.tiers:
            res, cp = self._refine(task, arm, path, answer, quality, pr, tier, queries, cost_so_far)
            self.expand_tree(task, arm, cp, res.answer_after, res.quality_after, initial_cost, queries,
                             cost_so_far + pr.usd + res.usd)

    def _expand_existing(self, task, arm, path, answer, quality, initial_cost, queries, cost_so_far) -> None:
        """Resume: the node is logged; re-run only the missing branches below it."""
        trees = {t.key: t for t in build_trees(read_records(self.out))}
        tree = trees.get((task.task_id, arm))
        node = tree.node_at(path) if tree else None
        if node is None:
            return
        if node.probe is None:
            # logged as a leaf by an earlier, shallower run (--max-rounds): probe it now and append an
            # updated node record (the tree builder keeps the last record per path)
            pr = self._probe(task, answer, queries)
            self._write(node_record(task, arm, path, answer, quality, initial_cost, pr, self.backend,
                                    {**self._root_extra(path, cost_so_far), "upgraded": True}))
            for tier in self.tiers:
                res, cp = self._refine(task, arm, path, answer, quality, pr, tier, queries, cost_so_far)
                self.expand_tree(task, arm, cp, res.answer_after, res.quality_after, initial_cost, queries,
                                 cost_so_far + pr.usd + res.usd)
            return
        pr = Probe([Claim(c["claim"], c["question"], c["expected"], c["self_conf"]) for c in node.probe["claims"]],
                   node.probe["answer_conf"], node.probe["min_conf"], node.probe["mean_conf"], node.probe["usd"],
                   node.probe["calls"], node.probe["parse_ok"])
        for tier in self.tiers:
            if tier in node.children:
                edge, child = node.children[tier]
                self.expand_tree(task, arm, child.path, child.answer, child.quality, initial_cost, queries,
                                 cost_so_far + pr.usd + float(edge["refine"]["usd"]))
            else:
                res, cp = self._refine(task, arm, path, answer, quality, pr, tier, queries, cost_so_far)
                self.expand_tree(task, arm, cp, res.answer_after, res.quality_after, initial_cost, queries,
                                 cost_so_far + pr.usd + res.usd)

    # ---------------------------------------------------------- policy
    def serve_policy(self, policy, task, arm, answer, quality, initial_cost, queries) -> None:
        path: list[str] = []
        cost = initial_cost
        prev = None
        while True:
            legal = [STOP] + ([refine_action(t) for t in self.tiers] if len(path) < self.max_rounds else [])
            pr = None
            if (policy.needs_probe or len(path) < self.max_rounds) and len(path) < self.max_rounds:
                need = policy.needs_probe
                st = SSRState(task.task_id, task.category, arm, len(path), cost, legal,
                              prev_conf_min=None if prev is None else prev.conf_min,
                              changed_last=bool(prev and prev.changed))
                if need:
                    pr = self._probe(task, answer, queries)
                    cost += pr.usd
                    st.probe_min, st.probe_mean, st.answer_conf = pr.min_conf, pr.mean_conf, pr.answer_conf
                action = policy.decide(st)
                if action != STOP and pr is None:
                    pr = self._probe(task, answer, queries)
                    cost += pr.usd
            else:
                action = STOP
            self._write(node_record(task, arm, path, answer, quality, initial_cost, pr, self.backend,
                                    {**self._root_extra(path, cost), "policy": policy.name, "decision": action}))
            if action == STOP:
                return
            tier = action.split("@", 1)[1]
            prev, path = self._refine(task, arm, path, answer, quality, pr, tier, queries, cost)
            answer, quality = prev.answer_after, prev.quality_after
            cost += prev.usd

    # ------------------------------------------------------------ main
    def run(self, tasks: list[CRMTask], arms: list[str], initial: Initial, policy_spec: str, seed: int = 0) -> int:
        ledger = self.h.ledger
        ledger.clear_stop_marker()
        if ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd:
            ledger.write_stop_marker()
            self.log(f"[ssr] ledger already at ${ledger.total:.4f} >= cap ${ledger.cap.cap_usd:g}; nothing run.")
            return EXIT_STOPPED_AT_CAP
        policy = None if policy_spec == "tree" else parse_policy(policy_spec, self.max_rounds)
        rng = random.Random(seed + 7919)
        n_skipped = 0
        for i, task in enumerate(tasks, 1):
            for arm in arms:
                if not initial.has(task.task_id, arm):
                    self.log(f"[ssr] {task.task_id} arm={arm}: no initial episode in the log; skipped")
                    continue
                root_key = ("node", task.task_id, arm, ())
                if policy is not None and root_key in self.done:
                    n_skipped += 1
                    continue
                try:
                    if root_key in self.done:      # resume: the root is logged with the initial episode
                        root = next(r for r in read_records(self.out) if r["kind"] == "node"
                                    and r["task_id"] == task.task_id and r["initial_arm"] == arm and not r["path"])
                        init = {"answer": root["answer"], "quality": root["quality"], "cost": root["initial_cost"],
                                "queries": root.get("queries") or [], "eval_usd": 0.0, "source": "resume"}
                    else:
                        init = initial.get(self.h, task, arm, rng)
                    self._init = init
                    if policy is None:
                        self.expand_tree(task, arm, [], init["answer"], init["quality"], init["cost"],
                                         init["queries"], init["cost"])
                    else:
                        self.serve_policy(policy, task, arm, init["answer"], init["quality"], init["cost"],
                                          init["queries"])
                except SpendCapReached as e:
                    p = ledger.write_stop_marker(e)
                    self.log(f"[ssr] STOPPED AT CAP in task {task.task_id} arm {arm}: {e}")
                    self.log(ledger.summary())
                    self.log(f"[ssr] wrote {p}; {self.n_new} new records this run -> {self.out}. "
                             f"Resume with a higher --cap (the ledger is cumulative; logged branches are skipped).")
                    return EXIT_STOPPED_AT_CAP
                except LLMCallFailed as e:
                    ledger.write_total()
                    self.log(f"[ssr] ABORTED on task {task.task_id} arm {arm}: {e}")
                    self.log(ledger.summary())
                    return EXIT_LLM_FAILED
            if i % 10 == 0:
                self.log(ledger.summary())
        self.log(f"[ssr] done: {self.n_new} new records, {n_skipped} (task, arm) already served -> {self.out}")
        self.log(ledger.summary())
        return 0


# -------------------------------------------------------------- builders
def build_dry_run(a, log=print):
    from .fake import FakeBackend, FakeWorld, make_fake_tasks
    traces_dir = a.traces_dir or str(DRYRUN_DIR / "ledger")
    ledger = SpendLedger(traces_dir, a.cap)
    tasks = make_fake_tasks(a.n_tasks, a.seed)
    world = FakeWorld(tasks, a.seed)
    llm = PricedLLM(ledger, completion_fn=world, timeout=30, effort="low")
    h = CRMArenaHarness(org=a.org, ledger=ledger, llm=llm, backend=FakeBackend(), tasks=tasks,
                        max_turns=a.max_turns)
    out = a.out or str(DRYRUN_DIR / "ssr_tree.jsonl")
    initial = Initial("fresh")
    log(f"[ssr] DRY RUN: fake world, {len(tasks)} fake tasks, ledger at {traces_dir} (no API calls)")
    return h, tasks, initial, out, "fake"


def build_real(a, log=print):
    if a.cap is None:
        raise SystemExit("real runs make paid API calls: pass --cap USD (or --dry-run)")
    try:
        from dotenv import load_dotenv
        load_dotenv(a.env_file)
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"ANTHROPIC_API_KEY is not set (looked in the environment and {a.env_file})")
    ledger = SpendLedger(a.traces_dir or "traces", a.cap)
    llm = PricedLLM(ledger, timeout=a.timeout, effort=None if a.effort == "none" else a.effort)
    h = CRMArenaHarness(org=a.org, ledger=ledger, llm=llm, max_turns=a.max_turns, obs_max_chars=a.obs_max_chars)
    initial = Initial(a.initial, a.log)
    tasks = h.sample_tasks(a.n_tasks, a.seed)      # same seeded order as pace.collect
    out = a.out or str(Path(a.traces_dir or "traces") / f"ssr_{a.org}_{'tree' if a.policy == 'tree' else 'policy'}.jsonl")
    return h, tasks, initial, out, "crmarena"


def cmd_run(a) -> int:
    tiers = parse_tiers(a.tiers)
    arms = parse_tiers(a.initial_arm)
    h, tasks, initial, out, backend = (build_dry_run if a.dry_run else build_real)(a)
    c = Collector(h, out, tiers, a.max_rounds, a.k, a.resolve_max_turns, a.decompose_tier, backend)
    print(f"[ssr] policy={a.policy} tiers={tiers} initial={initial.source}:{arms} n_tasks={len(tasks)} "
          f"max_rounds={a.max_rounds} k={a.k} cap={'none' if h.ledger.cap.cap_usd is None else f'${h.ledger.cap.cap_usd:g}'} "
          f"ledger=${h.ledger.total:.4f} -> {out}")
    return c.run(tasks, arms, initial, a.policy, a.seed)


def cmd_analyze(a) -> int:
    from .analyze import analyze
    log = a.log or str(DRYRUN_DIR / "ssr_tree.jsonl")
    out = a.out or "results/ssr/RESULTS.md"
    csv = a.csv or "results/ssr/frontier.csv"
    prelude = a.prelude if a.prelude is not None else str(Path(out).with_name("INTERPRETATION.md"))
    analyze(log, out, csv, budget=a.budget, folds=a.folds, seed=a.seed, policy_out=a.policy_out,
            prelude=prelude, counterfactual=a.counterfactual)
    print(f"[ssr] wrote {out} and {csv}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Priced Socratic self-refinement on CRMArena-Pro.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="collect a refinement tree or serve one policy")
    r.add_argument("--policy", default="tree", help="tree | never | always_1[:tier] | always_2[:tier] | "
                   "uncertainty_gated[:tau[:tier]] | controller:<json>")
    r.add_argument("--n-tasks", type=int, default=50)
    r.add_argument("--org", default="b2b", choices=["b2b", "b2c"])
    r.add_argument("--tiers", default="cheap,mid", help="refinement tiers (comma list of cheap,mid,frontier)")
    r.add_argument("--cap", type=float, default=None, help="hard USD cap on cumulative ledger spend (required unless --dry-run)")
    r.add_argument("--dry-run", action="store_true", help="fake world: no API, no Salesforce, no spend")
    r.add_argument("--initial", default="from-log", choices=["from-log", "fresh"],
                   help="initial answers: reuse the counterfactual log (free) or run them fresh")
    r.add_argument("--initial-arm", default="cheap", help="which initial tier(s) to refine: comma list or 'all'")
    r.add_argument("--log", default="traces/crmarena_b2b_counterfactual.jsonl", help="counterfactual log for --initial from-log")
    r.add_argument("--max-rounds", type=int, default=2)
    r.add_argument("--k", type=int, default=K_SAMPLES, help="worker samples per claim")
    r.add_argument("--resolve-max-turns", type=int, default=RESOLVE_MAX_TURNS, help="turn cap of a worker mini-episode")
    r.add_argument("--decompose-tier", default="cheap")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="tree log (default traces/ssr_<org>_tree.jsonl; dry-run results/ssr/dryrun/)")
    r.add_argument("--traces-dir", default=None, help="ledger dir (default traces; dry-run results/ssr/dryrun/ledger)")
    r.add_argument("--max-turns", type=int, default=12)
    r.add_argument("--obs-max-chars", type=int, default=20000)
    r.add_argument("--effort", default="low")
    r.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--env-file", default="third_party/CRMArena/.env")
    z = sub.add_parser("analyze", help="exact policy evaluation on a tree log -> RESULTS.md + CSV")
    z.add_argument("--log", default=None)
    z.add_argument("--out", default=None)
    z.add_argument("--csv", default=None)
    z.add_argument("--budget", type=float, default=0.10, help="per-task budget for the constrained controller")
    z.add_argument("--folds", type=int, default=5)
    z.add_argument("--seed", type=int, default=0)
    z.add_argument("--policy-out", default=None, help="where to save the all-tasks controller JSON")
    z.add_argument("--prelude", default=None, help="markdown inserted as §0 (default: INTERPRETATION.md next to --out)")
    z.add_argument("--counterfactual", default="traces/crmarena_b2b_counterfactual.jsonl",
                   help="collection log whose single-agent tier arms become reference rows ('' to skip)")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())
