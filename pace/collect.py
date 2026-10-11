"""Run a behavior policy on a harness and log Episodes.

Every logged decision carries the behavior policy's propensity INCLUDING its exploration
mass, the legal set it chose from and, when the harness is simulable, the counterfactual
arms and a state snapshot. That is the logging discipline that makes the later off-policy
gate (pace.gate) and replay training (pace.train_offline) possible (Bottou et al. 2013,
arXiv:1209.2355). `true_eval` is the simulator's Monte Carlo ground truth.

CLI (sim):
  python -m pace.collect --harness v1 --policy uniform --n 3000 --out traces/v1.jsonl
  python -m pace.collect --harness v2 --reprice mid:1.2 --policy policies/p.json --eps 0.3 ...
CLI (CRMArena-Pro, paid API calls behind a hard spend cap):
  python -m pace.collect --harness crmarena --org b2b --categories all --n-tasks 150 \
      --tiers all --mode counterfactual --cap 50
Exit codes: 0 done, 3 stopped at the spend cap (traces/STOPPED_AT_CAP written),
4 a model call failed after retries (bad key, rejected parameter, outage).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from .controller import Controller
from .harness.base import Harness
from .harness.sim import make_sim, parse_reprice
from .trace import Episode, TraceStep, append_jsonl, write_jsonl

EXIT_STOPPED_AT_CAP = 3
EXIT_LLM_FAILED = 4


def load_policy(spec: str, harness: Harness, eps: float | None = None) -> Controller:
    """'uniform' or a policy JSON, with its action set extended to the harness's and eps overridden."""
    if spec == "uniform":
        pol = Controller.uniform(harness.all_actions, harness.n_features)
    else:
        pol = Controller.load(spec)
        pol.ensure_actions(harness.all_actions)
    if eps is not None:
        pol.eps = eps
    return pol


def run_episode(harness: Harness, policy, task, rng: random.Random,
                log_counterfactual: bool = True, log_state: bool = True, eps: float | None = None) -> Episode:
    """One episode of `policy` (a Controller or anything with `.act` / `.actions`) on `harness`."""
    s = harness.reset(task)
    steps: list[TraceStep] = []
    while not s.done:
        legal = harness.legal_actions(s)
        x = harness.featurize(s)
        a, prop = policy.act(x, legal, rng, eps=eps)
        name = policy.actions[a]
        cf = harness.counterfactual(s) if log_counterfactual else None
        snap = harness.snapshot(s) if log_state else None
        s, _, cost, lat = harness.step(s, name)
        steps.append(TraceStep(task_id=s.task_id, harness_version=harness.version, step=len(steps),
                               features=[round(v, 6) for v in x], action=a, action_name=name,
                               propensity=prop, cost=cost, latency=lat,
                               counterfactual={k: list(v) for k, v in cf.items()} if cf else None,
                               legal=legal, state=snap))
    q = harness.quality(s)
    return Episode(task_id=s.task_id, steps=steps, quality=q,
                   total_cost=s.cost_so_far, total_latency=s.latency_so_far, meta=harness.episode_meta(s))


def collect(harness: Harness, policy: Controller, n: int, seed: int = 0, **kw) -> list[Episode]:
    """n seeded tasks from the harness, one logged episode each."""
    rng = random.Random(seed + 7919)
    return [run_episode(harness, policy, t, rng, **kw) for t in harness.sample_tasks(n, seed)]


def true_eval(harness: Harness, policy: Controller, n: int = 20000, seed: int = 12345,
              eps: float | None = None) -> dict:
    """Ground-truth (Monte Carlo) quality / cost / latency of a policy on a SIM harness.
    Quality uses the harness's expected success probability (lower variance)."""
    rng = random.Random(seed)
    q = c = l = 0.0
    for t in harness.sample_tasks(n, seed):
        s = harness.reset(t)
        while not s.done:
            legal = harness.legal_actions(s)
            a, _ = policy.act(harness.featurize(s), legal, rng, eps=eps)
            s, _, _, _ = harness.step(s, policy.actions[a])
        q += harness.expected_quality(s)
        c += s.cost_so_far
        l += s.latency_so_far
    return {"quality": q / n, "cost": c / n, "latency": l / n, "n": n}


class FixedArmPolicy:
    """Counterfactual-mode behavior policy for one tier arm.

    The behavior is "pick a tier uniformly at random per task, then CONTINUE on
    it until the agent answers (ANSWER is forced on the last allowed turn)".
    Its episode probability is 1/|tiers|; it is logged per decision as
    propensity 1/|tiers| on the first step and 1.0 on later steps (the tier is
    fixed by then), so per-decision importance products give exactly 1/|tiers|.
    """

    def __init__(self, tier: str, actions: list[str], n_tiers: int, first_step_index: int):
        self.tier, self.actions = tier, list(actions)
        self.p0 = 1.0 / n_tiers
        self.first = first_step_index           # feature index of "turn=0"
        self._idx = {a: i for i, a in enumerate(self.actions)}

    def act(self, x, legal, rng=None, eps=None):
        name = f"CONTINUE@{self.tier}" if f"CONTINUE@{self.tier}" in legal else f"ANSWER@{self.tier}"
        prop = self.p0 if x[self.first] > 0.5 else 1.0
        return self._idx[name], prop


def done_keys(path: str | Path) -> set[tuple[str, str]]:
    """(task_id, arm) pairs already logged; tolerant of a torn final line."""
    out: set[tuple[str, str]] = set()
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.add((str(d["task_id"]), str((d.get("meta") or {}).get("arm", "policy"))))
    return out


def parse_tiers(spec: str) -> list[str]:
    from .prices import TIERS
    tiers = list(TIERS) if spec == "all" else [t.strip() for t in spec.split(",") if t.strip()]
    bad = [t for t in tiers if t not in TIERS]
    if bad or not tiers:
        raise SystemExit(f"--tiers must be 'all' or a comma list of {TIERS}; got {spec!r}")
    return tiers


def collect_crmarena(harness, out: str | Path, n_tasks: int, mode: str = "counterfactual",
                     tiers: list[str] | None = None, policy: Controller | None = None, eps: float | None = None,
                     seed: int = 0, summary_every: int = 10, log=print) -> int:
    """Run CRMArena-Pro episodes, appending each to `out` as it finishes.

    Resumable: (task_id, arm) pairs already in `out` are skipped. Returns 0 when
    done, EXIT_STOPPED_AT_CAP when the ledger's cap stopped the run (the ledger
    is flushed and STOPPED_AT_CAP written; the interrupted episode is dropped and
    is re-run on resume).
    """
    from .harness.crmarena import LLMCallFailed
    from .prices import TIERS
    from .spend import SpendCapReached
    ledger = harness.ledger
    tiers = list(tiers or TIERS)
    if mode == "counterfactual":
        from .harness.crmarena import TURN0_IDX
        arms = [(t, FixedArmPolicy(t, harness.all_actions, len(tiers), TURN0_IDX)) for t in tiers]
    elif mode == "policy":
        if policy is None:
            raise ValueError("policy mode needs a controller")
        arms = [("policy", policy)]
    else:
        raise ValueError(mode)
    done = done_keys(out)
    ledger.clear_stop_marker()
    if ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd:
        p = ledger.write_stop_marker()
        log(f"[collect] ledger already at ${ledger.total:.4f} >= cap ${ledger.cap.cap_usd:g}; "
            f"nothing run. Raise --cap to continue. ({p})")
        return EXIT_STOPPED_AT_CAP
    tasks = harness.sample_tasks(n_tasks, seed)
    rng = random.Random(seed + 7919)
    n_new = n_skipped = 0
    for i, task in enumerate(tasks, 1):
        for arm, pol in arms:
            if (task.task_id, arm) in done:
                n_skipped += 1
                continue
            try:
                ep = run_episode(harness, pol, task, rng, log_counterfactual=False, log_state=False, eps=eps)
            except SpendCapReached as e:
                p = ledger.write_stop_marker(e)
                log(f"[collect] STOPPED AT CAP before task {task.task_id} arm {arm}: {e}")
                log(ledger.summary())
                log(f"[collect] wrote {p}; {n_new} new episodes this run -> {out}. "
                    f"Resume with a higher --cap (the ledger is cumulative).")
                return EXIT_STOPPED_AT_CAP
            except LLMCallFailed as e:
                ledger.write_total()
                log(f"[collect] ABORTED on task {task.task_id} arm {arm}: {e}")
                log(ledger.summary())
                log(f"[collect] {n_new} new episodes this run -> {out}; fix the cause and rerun to resume.")
                return EXIT_LLM_FAILED
            ep.meta = dict(ep.meta or {})
            ep.meta.update({"arm": arm, "mode": mode, "seed": seed,
                            "behavior": ("uniform_tier_continue_until_answer" if mode == "counterfactual"
                                         else (policy.meta.get("name", "controller") if policy else "")),
                            "eps": eps if mode == "policy" else None})
            append_jsonl(ep, out)
            n_new += 1
            log(f"[collect] {task.task_id} {task.category:32s} arm={arm:8s} q={ep.quality:.3f} "
                f"cost=${ep.total_cost:.4f} turns={len(ep.steps)} ({ep.meta.get('end_reason')})")
        if summary_every and i % summary_every == 0:
            log(ledger.summary())
    log(f"[collect] done: {n_new} new episodes, {n_skipped} already logged -> {out}")
    log(ledger.summary())
    return 0


def _main_crmarena(a) -> int:
    from .harness.crmarena import CATEGORIES, CRMArenaHarness, PricedLLM
    from .spend import SpendLedger
    if a.cap is None and not a.no_cap:
        raise SystemExit("--harness crmarena makes paid API calls: pass --cap USD (or --no-cap explicitly)")
    if a.n_tasks is None:
        raise SystemExit("--n-tasks is required for --harness crmarena")
    cats = None if a.categories in (None, "all") else [c.strip() for c in a.categories.split(",") if c.strip()]
    if cats:
        bad = [c for c in cats if c not in CATEGORIES]
        if bad:
            raise SystemExit(f"unknown categories {bad}; choose from {CATEGORIES}")
    tiers = parse_tiers(a.tiers)
    try:
        from dotenv import load_dotenv     # CRMArena's .env holds the org creds + ANTHROPIC_API_KEY
        load_dotenv(a.env_file)
    except ImportError:
        pass
    ledger = SpendLedger(a.traces_dir, None if a.no_cap else a.cap)
    llm = PricedLLM(ledger, timeout=a.timeout, effort=None if a.effort == "none" else a.effort)
    h = CRMArenaHarness(org=a.org, categories=cats, ledger=ledger, llm=llm, max_turns=a.max_turns,
                        max_tokens=a.max_tokens, answer_max_tokens=a.answer_max_tokens,
                        obs_max_chars=a.obs_max_chars, eval_mode=a.eval_mode)
    out = a.out or str(Path(a.traces_dir) / f"crmarena_{a.org}_{a.mode}.jsonl")
    at_cap = ledger.cap.cap_usd is not None and ledger.total >= ledger.cap.cap_usd
    if not os.environ.get("ANTHROPIC_API_KEY") and not at_cap:
        raise SystemExit(f"ANTHROPIC_API_KEY is not set (looked in the environment and {a.env_file})")
    pol = None
    if a.mode == "policy":
        pol = load_policy(a.policy, h, a.eps if a.eps is not None else 0.2)
        if pol.n_features != h.n_features:
            raise SystemExit(f"policy {a.policy} has {pol.n_features} features; crmarena needs {h.n_features}")
        if a.tiers != "all":
            print(f"[collect] note: --tiers restricts counterfactual arms only; policy mode acts over {h.all_actions}")
    print(f"[collect] crmarena org={a.org} mode={a.mode} tiers={tiers} n_tasks={a.n_tasks} "
          f"cap={'none' if a.no_cap else f'${a.cap:g}'} ledger=${ledger.total:.4f} -> {out}")
    return collect_crmarena(h, out, a.n_tasks, a.mode, tiers, pol, pol.eps if pol else None, a.seed)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Collect logged episodes with a behavior policy.")
    ap.add_argument("--harness", default="v1", help="sim version v1 | v2, or 'crmarena'")
    ap.add_argument("--reprice", nargs="*", default=[], help="tier:factor events, e.g. mid:1.2")
    ap.add_argument("--policy", default="uniform", help="'uniform' or a policy JSON")
    ap.add_argument("--eps", type=float, default=None, help="override the policy's exploration eps")
    ap.add_argument("--n", type=int, default=3000, help="sim: number of episodes")
    ap.add_argument("--seed", type=int, default=0, help="task sample and exploration seed")
    ap.add_argument("--out", default=None, help="sim: required; crmarena: default traces/crmarena_<org>_<mode>.jsonl")
    ap.add_argument("--no-counterfactual", action="store_true", help="sim: do not log the counterfactual arms")
    g = ap.add_argument_group("crmarena (real API calls; spend is capped)")
    g.add_argument("--org", default="b2b", choices=["b2b", "b2c"])
    g.add_argument("--categories", default="all", help="'all' or a comma list of CRMArena-Pro task categories")
    g.add_argument("--n-tasks", type=int, default=None, help="tasks to run (seeded, category-stratified sample)")
    g.add_argument("--tiers", default="all", help="'all' or a comma list of cheap,mid,frontier (counterfactual arms)")
    g.add_argument("--mode", default="counterfactual", choices=["counterfactual", "policy"])
    g.add_argument("--cap", type=float, default=None, help="hard USD cap on cumulative ledger spend")
    g.add_argument("--no-cap", action="store_true", help="explicitly run uncapped (not recommended)")
    g.add_argument("--traces-dir", default="traces", help="where spend.jsonl / spend_total.json / STOPPED_AT_CAP live")
    g.add_argument("--max-turns", type=int, default=12)
    g.add_argument("--max-tokens", type=int, default=2048, help="per agent turn (includes thinking)")
    g.add_argument("--answer-max-tokens", type=int, default=1024)
    g.add_argument("--obs-max-chars", type=int, default=20000, help="truncate Salesforce observations; 0 = off")
    g.add_argument("--effort", default="low", help="output_config.effort for mid/frontier, or 'none' to omit")
    g.add_argument("--eval-mode", default="aided", choices=["aided", "default"],
                   help="CRMArena agent_eval_mode ('aided' adds the optional domain metadata, as in their run script)")
    g.add_argument("--timeout", type=float, default=120.0, help="per-call timeout, seconds")
    g.add_argument("--env-file", default="third_party/CRMArena/.env")
    a = ap.parse_args(argv)
    if a.harness == "crmarena":
        sys.exit(_main_crmarena(a))
    if a.out is None:
        ap.error("--out is required for the sim harness")
    h = make_sim(a.harness, seed=a.seed, reprice=parse_reprice(a.reprice))
    pol = load_policy(a.policy, h, a.eps)
    eps_ = collect(h, pol, a.n, a.seed, log_counterfactual=not a.no_counterfactual)
    write_jsonl(eps_, a.out)
    q = np.mean([e.quality for e in eps_]); c = np.mean([e.total_cost for e in eps_])
    L = np.mean([len(e.steps) for e in eps_])
    print(f"[collect] {len(eps_)} episodes on sim {a.harness} (reprice={a.reprice or '-'}) "
          f"policy={a.policy} eps={pol.eps:.2f}: quality {q:.3f}  cost {c:.3f}  steps {L:.2f} -> {a.out}")


if __name__ == "__main__":
    main()
