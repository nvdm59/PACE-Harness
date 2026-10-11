"""The refinement tree log and exact policy evaluation on it.

`run.py --policy tree` logs, per (task, initial arm), every branch of the refinement decision
up to `max_rounds`: the initial answer and its probe (root node), then for every tier an edge
(resolve + rewrite on that tier) into a child node (the refined answer and its probe), and so on.
Each refinement policy is a deterministic or stochastic map from the node's observable state to
{STOP, REFINE@tier}, so once every branch is logged, the value of ANY policy is an exact lookup,
exactly as the first-step tier was in `pace.lookup` for the counterfactual collection. A policy
run live (`--policy uncertainty_gated:0.6:cheap`) logs only its own branch; it can still be
evaluated, and other policies are marked not evaluable on that log.

Records (one JSON line each, appended as they finish):
  node  {kind, task_id, category, reward_metric, initial_arm, path, round, answer, quality,
         initial_cost, probe|null}
  edge  {kind, task_id, initial_arm, path (child's), round, tier, refine: RefineResult}

Costing rules used by the evaluator, per policy:
  cost = initial episode cost
       + probe cost at every visited node where the policy consults the probe or refines
       + resolve + rewrite cost of every refinement taken
The extractor / privacy-judge spend is reported separately and never charged, as in the core
harness. A policy that stops at the root without looking (`never`) pays the initial episode only.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ...prices import TIERS

STOP = "STOP"


def refine_action(tier: str) -> str:
    return f"REFINE@{tier}"


def action_tier(action: str) -> str | None:
    return action.split("@", 1)[1] if action.startswith("REFINE@") else None


@dataclass
class Node:
    path: list[str]
    answer: str
    quality: float
    probe: dict | None = None                  # Probe.to_dict() or None (leaf / not probed)
    children: dict[str, tuple[dict, "Node"]] = field(default_factory=dict)   # tier -> (edge, child)
    record: dict | None = None

    @property
    def round(self) -> int:
        return len(self.path)

    @property
    def probed(self) -> bool:
        return self.probe is not None

    def edge_in(self, tree: "Tree") -> dict | None:
        if not self.path:
            return None
        n = tree.root
        edge = None
        for t in self.path:
            edge, n = n.children[t]
        return edge


@dataclass
class Tree:
    task_id: str
    initial_arm: str
    category: str
    reward_metric: str
    initial_cost: float
    root: Node

    @property
    def key(self) -> tuple[str, str]:
        return (self.task_id, self.initial_arm)

    def node_at(self, path: list[str]) -> Node | None:
        n = self.root
        for t in path:
            if t not in n.children:
                return None
            n = n.children[t][1]
        return n

    def nodes(self) -> Iterable[Node]:
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(c for _, c in n.children.values())

    def edges(self) -> Iterable[tuple[Node, dict, Node]]:
        for n in self.nodes():
            for e, c in n.children.values():
                yield n, e, c

    def complete(self, tiers: list[str], max_rounds: int) -> bool:
        """Every branch over `tiers` logged to depth `max_rounds`, with probes on every inner node."""
        def ok(n: Node) -> bool:
            if n.round >= max_rounds:
                return True
            if not n.probed:
                return False
            return all(t in n.children and ok(n.children[t][1]) for t in tiers)
        return ok(self.root)


# ------------------------------------------------------------------- I/O
def read_records(path: str | Path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:      # torn final line
            continue
    return out


def record_key(r: dict) -> tuple:
    return (r["kind"], str(r["task_id"]), str(r["initial_arm"]), tuple(r.get("path") or []))


def done_keys(path: str | Path) -> set[tuple]:
    return {record_key(r) for r in read_records(path)}


def append_record(r: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(r, separators=(",", ":")) + "\n")
        f.flush()


def build_trees(records: list[dict]) -> list[Tree]:
    """Records -> trees. Edges whose parent node is missing are dropped (cap-interrupted)."""
    nodes: dict[tuple, list[dict]] = defaultdict(list)
    edges: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        k = (str(r["task_id"]), str(r["initial_arm"]))
        (nodes if r["kind"] == "node" else edges)[k].append(r)
    trees: list[Tree] = []
    for k in sorted(set(nodes)):
        by_path = {tuple(r["path"]): r for r in nodes[k]}
        if () not in by_path:
            continue
        rr = by_path[()]
        root = Node([], str(rr.get("answer")), float(rr["quality"]), rr.get("probe"), record=rr)
        tree = Tree(k[0], k[1], rr.get("category", "unknown"), rr.get("reward_metric", "?"),
                    float(rr.get("initial_cost", 0.0)), root)
        for e in sorted(edges.get(k, []), key=lambda e: len(e["path"])):
            path = tuple(e["path"])
            parent = tree.node_at(list(path[:-1]))
            child_rec = by_path.get(path)
            if parent is None or child_rec is None:
                continue
            child = Node(list(path), str(child_rec.get("answer")), float(child_rec["quality"]),
                         child_rec.get("probe"), record=child_rec)
            parent.children[path[-1]] = (e, child)
        trees.append(tree)
    return trees


def load_trees(path: str | Path) -> list[Tree]:
    return build_trees(read_records(path))


# ------------------------------------------------------------------ state
@dataclass
class SSRState:
    """What a refinement policy observes at a node. `probe_*` are None when the node was not probed."""
    task_id: str
    category: str
    initial_arm: str
    round: int
    cost_so_far: float
    legal: list[str]
    probe_min: float | None = None
    probe_mean: float | None = None
    answer_conf: float | None = None
    prev_conf_min: float | None = None      # per-claim conf after the last round's re-solve
    changed_last: bool = False


def state_of(tree: Tree, node: Node, cost_so_far: float, tiers: list[str], max_rounds: int) -> SSRState:
    # legal = every tier the experiment offers, not only the branches that happen to be logged: a policy
    # whose choice was not logged is reported "not evaluable" (None) rather than silently re-routed to STOP
    legal = [STOP] + ([refine_action(t) for t in tiers] if node.round < max_rounds else [])
    e = node.edge_in(tree)
    rf = (e or {}).get("refine") or {}
    p = node.probe or {}
    return SSRState(tree.task_id, tree.category, tree.initial_arm, node.round, cost_so_far, legal,
                    p.get("min_conf"), p.get("mean_conf"), p.get("answer_conf"),
                    rf.get("conf_min"), bool(rf.get("changed", False)))


@dataclass
class Outcome:
    quality: float
    cost: float
    eval_usd: float
    rounds: float
    n_calls: float = 0.0


def evaluate_policy(tree: Tree, policy, tiers: list[str], max_rounds: int) -> Outcome | None:
    """Exact expected (quality, cost, ...) of `policy` on one logged tree, or None if the policy
    needs a branch or a probe that was not logged."""

    def rec(node: Node, cost: float, eval_usd: float, rounds: int) -> Outcome | None:
        st = state_of(tree, node, cost, tiers, max_rounds)
        if node.round >= max_rounds:
            return Outcome(node.quality, cost, eval_usd, rounds)
        if policy.needs_probe and not node.probed:
            return None
        probs = policy.probs(st)
        q = c = ev = r = 0.0
        for a, p in probs.items():
            if p <= 0:
                continue
            if a == STOP:
                extra = (node.probe or {}).get("usd", 0.0) if policy.needs_probe else 0.0
                o = Outcome(node.quality, cost + extra, eval_usd, rounds)
            else:
                t = action_tier(a)
                if t not in node.children or not node.probed:
                    return None
                edge, child = node.children[t]
                rf = edge["refine"]
                o = rec(child, cost + node.probe["usd"] + float(rf["usd"]), eval_usd + float(rf.get("usd_eval", 0.0)),
                        rounds + 1)
                if o is None:
                    return None
            q += p * o.quality
            c += p * o.cost
            ev += p * o.eval_usd
            r += p * o.rounds
        return Outcome(q, c, ev, r)

    return rec(tree.root, tree.initial_cost, 0.0, 0)


def evaluate_policy_on(trees: list[Tree], policy, tiers: list[str], max_rounds: int) -> dict:
    outs = [(t, evaluate_policy(t, policy, tiers, max_rounds)) for t in trees]
    ok = [o for _, o in outs if o is not None]
    n = len(ok)
    if n == 0:
        return {"n": 0, "n_total": len(trees), "quality": float("nan"), "cost": float("nan"),
                "eval_usd": float("nan"), "rounds": float("nan"), "q_se": float("nan"), "per_task": {}}
    qs = [o.quality for o in ok]
    mean = sum(qs) / n
    se = (sum((q - mean) ** 2 for q in qs) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan")
    return {"n": n, "n_total": len(trees),
            "quality": mean, "cost": sum(o.cost for o in ok) / n,
            "eval_usd": sum(o.eval_usd for o in ok) / n, "rounds": sum(o.rounds for o in ok) / n, "q_se": se,
            "per_task": {t.task_id: (o.quality, o.cost) for t, o in outs if o is not None}}


def kfold(trees: list[Tree], k: int = 5, seed: int = 0) -> list[tuple[list[Tree], list[Tree]]]:
    """Category-stratified K folds over trees (shuffle within category, deal round-robin)."""
    rng = random.Random(seed)
    by: dict[str, list[Tree]] = defaultdict(list)
    for t in trees:
        by[t.category].append(t)
    folds: list[list[Tree]] = [[] for _ in range(k)]
    j = 0
    for c in sorted(by):
        ts = sorted(by[c], key=lambda t: t.key)
        rng.shuffle(ts)
        for t in ts:
            folds[j % k].append(t)
            j += 1
    return [([t for i, f in enumerate(folds) if i != fi for t in f], folds[fi]) for fi in range(k)]


def tiers_in(trees: list[Tree]) -> list[str]:
    seen = {e["tier"] for t in trees for _, e, _ in t.edges()}
    return [t for t in TIERS if t in seen]


def max_rounds_in(trees: list[Tree]) -> int:
    return max((n.round for t in trees for n in t.nodes()), default=0)
