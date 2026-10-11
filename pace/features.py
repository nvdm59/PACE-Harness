"""State -> feature vector, and the discrete buckets the DR direct model averages over.

Simulator (48 dims). Base block b (12 dims): bias, difficulty-proxy one-hot (easy, hard),
evidence count one-hot (0, 1, 2), verified flag, step index one-hot (0..3), cost-so-far.
Harness-version one-hot enters as crosses: x = [b, hard*b, v2*b, v2*hard*b]. v1 is the
reference version, so a v1-trained policy transfers to v2 unchanged (its v2 weights are zero),
which is exactly the "frozen policy under drift" condition. The difficulty used is the
harness's OBSERVED proxy, never the latent.

CRMArena-Pro (43 dims). The layout is owned by pace.harness.crmarena.FEATURE_NAMES; this module
only decodes the (category, org) context from it for the bandit direct model.

Direct models are bucketed means (the "direct method" of Dudik, Langford & Li 2011,
arXiv:1103.4601, with a lookup table as the regressor): `bucket_key` for the sim (iterated as
bucketed fitted-Q evaluation in pace.gate), `CRMBuckets` for CRMArena.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from .harness.base import HarnessState

VERSIONS = ["v1", "v2"]
_BASE_NAMES = ["bias", "proxy_easy", "proxy_hard", "ev0", "ev1", "ev2", "verified",
               "step0", "step1", "step2", "step3", "cost_so_far"]


def _base(state: HarnessState) -> list[float]:
    ev = min(max(int(state.evidence), 0), 2)
    st = min(max(int(state.step), 0), 3)
    hard = 1.0 if state.difficulty_proxy == "hard" else 0.0
    b = [1.0, 1.0 - hard, hard, 0.0, 0.0, 0.0, 1.0 if state.verified else 0.0,
         0.0, 0.0, 0.0, 0.0, float(state.cost_so_far)]
    b[3 + ev] = 1.0
    b[7 + st] = 1.0
    return b


def featurize(state: HarnessState) -> list[float]:
    b = _base(state)
    hard = b[2]
    v2 = 1.0 if state.version == "v2" else 0.0
    return b + [hard * v for v in b] + [v2 * v for v in b] + [v2 * hard * v for v in b]


def feature_names() -> list[str]:
    return (list(_BASE_NAMES) + [f"hard*{n}" for n in _BASE_NAMES]
            + [f"v2*{n}" for n in _BASE_NAMES] + [f"v2*hard*{n}" for n in _BASE_NAMES])


N_FEATURES = len(feature_names())


def bucket_key(features: list[float]) -> tuple[int, int, int, int, int]:
    """Discrete bucket (proxy hard, evidence, verified, step, v2) of a sim feature vector; drops
    cost-so-far. The DR direct model is a mean over logged transitions in the same bucket."""
    b = features[:len(_BASE_NAMES)]
    v2 = features[2 * len(_BASE_NAMES)]   # v2*bias
    ev = int(max(range(3), key=lambda i: b[3 + i]))
    st = int(max(range(4), key=lambda i: b[7 + i]))
    return (int(b[2] > 0.5), ev, int(b[6] > 0.5), st, int(v2 > 0.5))


# ------------------------------------------------------------- CRMArena-Pro
# The CRMArena log (pace.harness.crmarena, 43 dims) is a contextual bandit at the first step:
# the controller picks the tier, then a fixed continuation runs (propensity 1 after step 0).
# Its DR direct model is a bucketed mean: bucket = (task_category, tier), falling back to
# (org, tier) when the category bucket holds fewer than `min_n` logged samples, then to the
# tier mean, then to the global mean.
CRM_MIN_BUCKET = 3


@lru_cache(maxsize=None)
def _crm_layout() -> tuple[list[str], list[int], list[str], list[int]]:
    """(categories, their feature indices, orgs, their feature indices), read from the harness's
    FEATURE_NAMES so there is one definition of the layout."""
    from .harness.crmarena import CATEGORIES, FEATURE_NAMES, ORG_TYPES
    return (list(CATEGORIES), [FEATURE_NAMES.index(f"cat={c}") for c in CATEGORIES],
            list(ORG_TYPES), [FEATURE_NAMES.index(f"org={o}") for o in ORG_TYPES])


def crm_context(features: list[float]) -> tuple[str, str]:
    """(task_category, org) decoded from a CRMArena feature vector's one-hot blocks."""
    cats, cat_ix, orgs, org_ix = _crm_layout()
    cv = [features[i] for i in cat_ix]
    ov = [features[i] for i in org_ix]
    cat = cats[max(range(len(cv)), key=lambda i: cv[i])] if max(cv) > 0.5 else "unknown"
    org = orgs[max(range(len(ov)), key=lambda i: ov[i])] if max(ov) > 0.5 else "unknown"
    return cat, org


class CRMBuckets:
    """Bucketed means over (category, org, tier) samples with the (category, tier) ->
    (org, tier) -> (tier) -> global fallback. `tier=None` everywhere gives a per-state
    (category -> org -> global) mean, used as the AWR baseline."""

    def __init__(self, min_n: int = CRM_MIN_BUCKET):
        self.min_n = int(min_n)
        self.sums: dict[tuple, float] = {}
        self.cnts: dict[tuple, int] = {}
        self.glob = 0.0

    def fit(self, cats: Iterable[str], orgs: Iterable[str], tiers: Iterable[str | None],
            values: Iterable[float]) -> "CRMBuckets":
        self.sums, self.cnts = {}, {}
        vals = [float(v) for v in values]
        for c, o, t, v in zip(cats, orgs, tiers, vals):
            for k in (("cat", c, t), ("org", o, t), ("tier", t)):
                self.sums[k] = self.sums.get(k, 0.0) + v
                self.cnts[k] = self.cnts.get(k, 0) + 1
        self.glob = sum(vals) / len(vals) if vals else 0.0
        return self

    def key(self, cat: str, org: str, tier: str | None) -> tuple:
        """The bucket actually used for (cat, org, tier) under the fallback rule."""
        for k in (("cat", cat, tier), ("org", org, tier)):
            if self.cnts.get(k, 0) >= self.min_n:
                return k
        return ("tier", tier) if self.cnts.get(("tier", tier), 0) > 0 else ("global",)

    def predict(self, cat: str, org: str, tier: str | None) -> float:
        k = self.key(cat, org, tier)
        return self.glob if k == ("global",) else self.sums[k] / self.cnts[k]
