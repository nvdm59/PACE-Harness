"""SkillOrchestra-style routing (arXiv 2602.19672) as a baseline: a competence handbook learned
from execution traces and a per-decision selection rule

    final_score(a | task) = competence[a][skill(task)] − λ_c · avg_cost[a],   ties broken by lower cost,

with the skill of a task being its CRMArena category (the axis the real run found carries the
signal). SkillOrchestra learns the handbook once (explore → learn → select → test) and routes
statically; the two variants here are

  static     handbook from the warm-up days' exploration traces, λ_c tuned once on a held-out
             slice of those traces, never updated (new agents / tools / skills that arrive later
             are scored at the handbook's prior, the global mean, and their logged cost)
  relearned  after every harness stage, the router serves its current handbook WITH exploration
             for `refit_days` days, then refits the handbook on those traces and re-tunes λ_c on
             the last of them: the design-time alternative to online adaptation, with the lag
             counted against it

The handbook entry for (action, category) is the mean final quality of the logged episodes in
which that action was taken at that category, shrunk toward the action's mean over all
categories and then toward the global mean (pseudo-count `pseudo`), which is the usual
small-sample estimate of a competence profile from traces. `avg_cost[a]` is the mean logged cost
of the calls the action triggered. λ_c is chosen on the grid by the episode-level self-normalised
IPS of the held-out slice (pace.experiments.mas.offpolicy, step-ratio clip 5, ESS reported):
the largest estimated quality whose estimated cost is within the budget, else the cheapest.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from ...trace import Episode
from ..mas.offpolicy import episode_level
from .simenv import ACTIONS, FEATURE_NAMES, split_action

CAT_IX = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("cat=")]
CAT_NAMES = [n[4:] for n in FEATURE_NAMES if n.startswith("cat=")]
LAMBDA_C_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)


def category_of(features) -> str:
    v = [features[i] for i in CAT_IX]
    j = max(range(len(v)), key=lambda i: v[i])
    return CAT_NAMES[j] if v[j] > 0.5 else "unknown"


class Handbook:
    def __init__(self, pseudo: float = 2.0):
        self.pseudo = float(pseudo)
        self.competence: dict[tuple[str, str], float] = {}
        self.action_mean: dict[str, float] = {}
        self.avg_cost: dict[str, float] = {}
        self.glob = 0.5
        self.n_episodes = 0

    def learn(self, episodes: list[Episode]) -> "Handbook":
        q_sum, q_cnt = defaultdict(float), defaultdict(int)
        a_sum, a_cnt = defaultdict(float), defaultdict(int)
        c_sum, c_cnt = defaultdict(float), defaultdict(int)
        qs = []
        for e in episodes:
            qs.append(float(e.quality))
            for s in e.steps:
                cat = category_of(s.features)
                q_sum[(s.action_name, cat)] += e.quality; q_cnt[(s.action_name, cat)] += 1
                a_sum[s.action_name] += e.quality; a_cnt[s.action_name] += 1
                c_sum[s.action_name] += s.cost; c_cnt[s.action_name] += 1
        self.glob = float(np.mean(qs)) if qs else 0.5
        self.action_mean = {a: (a_sum[a] + self.pseudo * self.glob) / (a_cnt[a] + self.pseudo) for a in a_cnt}
        self.competence = {k: (q_sum[k] + self.pseudo * self.action_mean[k[0]]) / (q_cnt[k] + self.pseudo) for k in q_cnt}
        self.avg_cost = {a: c_sum[a] / c_cnt[a] for a in c_cnt}
        self.n_episodes = len(episodes)
        return self

    def score(self, action: str, cat: str, lam_c: float) -> tuple[float, float]:
        comp = self.competence.get((action, cat), self.action_mean.get(action, self.glob))
        cost = self.avg_cost.get(action, self._default_cost(action))
        return comp - lam_c * cost, cost

    def _default_cost(self, action: str) -> float:
        """An action never logged (a capability that arrived after the handbook was learned) is
        priced at the mean logged cost of actions with the same op, else the global mean cost."""
        op = split_action(action)[0]
        same = [c for a, c in self.avg_cost.items() if split_action(a)[0] == op]
        return float(np.mean(same)) if same else (float(np.mean(list(self.avg_cost.values()))) if self.avg_cost else 0.0)

    def to_dict(self) -> dict:
        return {"competence": {f"{a}|{c}": round(v, 4) for (a, c), v in self.competence.items()},
                "action_mean": {a: round(v, 4) for a, v in self.action_mean.items()},
                "avg_cost": {a: round(v, 6) for a, v in self.avg_cost.items()}, "global": self.glob, "n_episodes": self.n_episodes}


class HandbookRouter:
    """Deterministic argmax router with the Controller's duck-typed surface (actions, act, probs,
    copy, ensure_actions, index, meta) so pace.experiments.mas.offpolicy and the day loop treat
    it like any policy. `eps` mixes uniform exploration over the legal set; the logged propensity
    includes it."""

    def __init__(self, handbook: Handbook, lam_c: float, eps: float = 0.0, name: str = "so_router"):
        self.hb, self.lam_c, self.eps = handbook, float(lam_c), float(eps)
        self.actions = list(ACTIONS)
        self._index = {a: i for i, a in enumerate(self.actions)}
        self.meta = {"name": name, "lambda_c": self.lam_c, "handbook_episodes": handbook.n_episodes}
        self.n_features = len(FEATURE_NAMES)

    def index(self, name: str) -> int:
        return self._index[name]

    def ensure_actions(self, names) -> None:
        for a in names:
            if a not in self._index:
                self._index[a] = len(self.actions); self.actions.append(a)

    def copy(self) -> "HandbookRouter":
        return HandbookRouter(self.hb, self.lam_c, self.eps, self.meta["name"])

    def choose(self, x, legal: list[str]) -> str:
        cat = category_of(x)
        best, best_key = None, None
        for a in legal:
            sc, cost = self.hb.score(a, cat, self.lam_c)
            key = (round(sc, 9), -cost)           # cost breaks ties among equally scored actions
            if best_key is None or key > best_key:
                best, best_key = a, key
        return best

    def probs(self, x, legal: list[str] | None = None, eps: float | None = None) -> np.ndarray:
        eps = self.eps if eps is None else eps
        legal = legal or self.actions
        p = np.zeros(len(self.actions))
        p[self._index[self.choose(x, legal)]] = 1.0
        if eps > 0:
            m = np.zeros(len(self.actions)); m[[self._index[a] for a in legal]] = 1.0
            p = (1 - eps) * p + eps * m / m.sum()
        return p

    def act(self, x, legal: list[str], rng: random.Random | None = None, eps: float | None = None, **_) -> tuple[int, float]:
        rng = rng or random
        p = self.probs(x, legal, eps)
        u, acc = rng.random(), 0.0
        idx = int(np.flatnonzero(p > 0)[-1])
        for i in np.flatnonzero(p > 0):
            acc += p[i]
            if u < acc:
                idx = int(i)
                break
        return idx, float(p[idx])


def tune_lambda_c(hb: Handbook, heldout: list[Episode], budget: float, tol: float, grid=LAMBDA_C_GRID,
                  clip: float = 5.0, boot: int = 50) -> dict:
    """Pick λ_c on `grid` by episode-level SN-IPS on the held-out slice: the largest estimated
    quality among the values whose estimated cost is within budget + tol, else the cheapest.
    Returns {'lambda_c', 'rows': [...]} with each row's estimates and ESS."""
    rows = []
    for lc in grid:
        r = HandbookRouter(hb, lc)
        q = episode_level(heldout, r, reward="quality", clip=clip, boot=boot)
        c = episode_level(heldout, r, reward="cost", clip=clip, boot=boot)
        rows.append({"lambda_c": lc, "quality": q["snips"][0], "cost": c["snips"][0], "ess": q["ess"], "n": q["n"]})
    ok = [r for r in rows if r["cost"] <= budget + tol]
    pick = max(ok, key=lambda r: r["quality"]) if ok else min(rows, key=lambda r: r["cost"])
    return {"lambda_c": pick["lambda_c"], "rows": rows, "within_budget": bool(ok)}
