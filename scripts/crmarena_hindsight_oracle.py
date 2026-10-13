"""Hindsight (luck) component of the realised per-task oracle on CRMArena-Pro B2B (offline estimate).

The realised per-task oracle takes, per task, the best of three SINGLE stochastic outcomes, so part of
its edge over any policy is luck over outcome noise, not information. One repeated run exists:
traces/cheap_rerun_b2b.jsonl, a second cheap-tier run on the first 50 B2B tasks of the
counterfactual collection (same Episode schema, same task ids). This script measures the cheap
tier's run-to-run agreement on those 50 tasks and, under a simple Bernoulli model (assumptions in
the output), estimates E[realised best-of-three] vs the best single tier and vs a per-task oracle
that knows each tier's success PROBABILITY but not the draw (no luck).

  python scripts/crmarena_hindsight_oracle.py      # instant, deterministic, no API

Writes results/crmarena/hindsight_oracle.md (new file).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pace.lookup import LookupTable  # noqa: E402
from pace.prices import TIERS  # noqa: E402
from pace.trace import read_jsonl  # noqa: E402

THRESH = 0.5        # correctness = quality >= 0.5 (6 of the 150 cells are fractional text-similarity scores)


def scenario(P: np.ndarray) -> dict:
    """P (n, 3) per-task success probabilities; tiers independent given the task."""
    e_or = float((1.0 - np.prod(1.0 - P, axis=1)).mean())      # E[max_t X_t], X_t ~ Bern(p_t) independent
    p_or = float(P.max(1).mean())                               # knows p, not the draw: no luck
    best = float(P.mean(0).max())
    return {"E_realised_oracle": e_or, "prob_oracle": p_or, "best_single_tier": best,
            "best_tier": TIERS[int(P.mean(0).argmax())], "luck": e_or - p_or, "selection": p_or - best,
            "total_gap": e_or - best}


def main():
    rr = read_jsonl(ROOT / "traces" / "cheap_rerun_b2b.jsonl")
    T = LookupTable.from_episodes(read_jsonl(ROOT / "traces" / "crmarena_b2b_counterfactual.jsonl"))
    pos = {t: i for i, t in enumerate(T.task_ids)}
    assert all(e.task_id in pos for e in rr) and len({e.task_id for e in rr}) == len(rr)
    assert all((e.meta or {}).get("arm") == "cheap" for e in rr)
    ix = np.array([pos[e.task_id] for e in rr])
    Qraw = T.Q[ix]
    X = (Qraw >= THRESH).astype(float)                          # run 1, all tiers
    x2 = np.array([e.quality >= THRESH for e in rr], float)     # run 2, cheap
    x1 = X[:, 0]
    n = len(rr)
    flips = x1 != x2
    F = float(flips.mean())
    up, down = int(((x1 == 0) & (x2 == 1)).sum()), int(((x1 == 1) & (x2 == 0)).sum())
    m = up + down
    sign_p = min(1.0, sum(math.comb(m, j) for j in range(m + 1) if math.comb(m, j) <= math.comb(m, up)) / 2 ** m)
    raw_agree = float((np.isclose(Qraw[:, 0], [e.quality for e in rr])).mean())
    var_hat = float(((x1 - x2) ** 2 / 2).mean())                # unbiased per-task Bernoulli variance, averaged
    rho = float(np.corrcoef(x1, x2)[0, 1])                      # test-retest reliability of one cheap run
    brng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        b = brng.integers(0, n, n)
        if x1[b].std() > 0 and x2[b].std() > 0:
            bs.append(np.corrcoef(x1[b], x2[b])[0, 1])
    rho_lo, rho_hi = (float(v) for v in np.percentile(bs, [5, 95]))
    R_obs = float(X.max(1).mean())                              # the realised oracle (run 1), binarised
    best_obs = float(X.mean(0).max())
    m_cheap = float((x1 + x2).mean() / 2)
    rho2 = 2 * rho / (1 + rho)                                  # Spearman-Brown: reliability of the 2-run mean

    def kelley(rho_mf: float) -> np.ndarray:
        """Kelley true-score estimates of p_it (mean-preserving shrinkage toward the tier mean):
        p = m_t + rel * (xbar - m_t); cheap uses its two-run mean with reliability rho2,
        mid / frontier one run with an assumed reliability rho_mf."""
        P = np.zeros_like(X)
        P[:, 0] = m_cheap + rho2 * ((x1 + x2) / 2 - m_cheap)
        for t in (1, 2):
            P[:, t] = X[:, t].mean() + rho_mf * (X[:, t] - X[:, t].mean())
        return P

    k_rows = []
    for lab, r_mf in ((f"1 (no noise on mid/frontier)", 1.0), (f"{rho_hi:.3f} (cheap ρ, 95th pct)", rho_hi),
                      (f"{rho:.3f} (= cheap ρ)", rho), (f"{rho_lo:.3f} (cheap ρ, 5th pct)", rho_lo)):
        s_ = scenario(kelley(r_mf))
        s_["luck_anchored"] = R_obs - s_["prob_oracle"]
        k_rows.append((lab, r_mf, s_))
    delta_star = (1 - math.sqrt(1 - 2 * F)) / 2
    f_rows = []
    for lab, d in (("0", 0.0), (f"δ* = {delta_star:.4f} (2δ(1−δ) = F)", delta_star), (f"F = {F:.4f}", F)):
        P = (1 - d) * X + d * (1 - X)
        P[:, 0] = (x1 + x2) / 2
        po = float(P.max(1).mean())
        f_rows.append((lab, d, po, R_obs - po))
    anchored = [r[2]["luck_anchored"] for r in k_rows[1:]] + [r[3] for r in f_rows[1:]]
    f = lambda v: f"{v:.4f}"
    L = ["# Hindsight (luck) component of the realised per-task oracle: CRMArena-Pro B2B, first 50 tasks", "",
         "**An estimate under stated assumptions, not a measurement.** Produced by "
         "`scripts/crmarena_hindsight_oracle.py` from `traces/cheap_rerun_b2b.jsonl` (a second cheap-tier run "
         "on the first 50 B2B tasks) and `traces/crmarena_b2b_counterfactual.jsonl` (run 1, all tiers).", "",
         "## 1. Cheap-tier repeatability (measured)", "",
         f"- tasks with two cheap runs: {n} (task ids match the counterfactual log; arm = cheap in every rerun episode)",
         f"- correctness = quality ≥ {THRESH} (the 50 × 3 cells hold {int(((Qraw > 0) & (Qraw < 1)).sum())} fractional "
         "text-similarity scores, all < 0.5)",
         f"- correctness agreement between the two runs: **{1 - F:.3f}** ({n - int(flips.sum())}/{n}); "
         f"flip rate F = {F:.3f} ({int(flips.sum())} tasks: {up} wrong→right, {down} right→wrong; two-sided sign "
         f"test p = {sign_p:.3f}). Exact raw-quality agreement: {raw_agree:.3f}.",
         f"- cheap accuracy: run 1 {x1.mean():.3f}, run 2 {x2.mean():.3f} (rerun better by {x2.mean() - x1.mean():+.3f}; "
         f"not significant at {m} flips, but drift between runs cannot be excluded)",
         f"- estimated per-task outcome variance (mean over tasks of (x1 − x2)²/2, unbiased for p(1 − p)): "
         f"**{var_hat:.4f}** (0.25 would be a coin flip on every task; 0 fully repeatable)",
         f"- test-retest reliability (correlation of the two runs' correctness): ρ = **{rho:.3f}** "
         f"(task bootstrap 90% interval [{rho_lo:.3f}, {rho_hi:.3f}], 2000 resamples, seed 0)", "",
         f"Realised on these 50 tasks (run 1, binarised): per-task oracle **{f(R_obs)}**, best single tier {f(best_obs)} "
         f"(frontier); raw quality {f(float(Qraw.max(1).mean()))} vs {f(float(Qraw.mean(0).max()))}. "
         f"Realised oracle − best single = {R_obs - best_obs:.4f}.", "",
         "## 2. Model and definitions", "",
         "Each (task, tier) outcome X_it is Bernoulli(p_it). Luck (hindsight) = realised oracle − probability oracle, "
         "where the probability oracle picks per task the tier with the highest p_it (it knows each tier's success "
         "probability on the task, not the draw). Selection = probability oracle − best single tier. The realised "
         "oracle is observed (0.76 above), so the anchored luck estimate is R_obs − (estimated probability "
         "oracle); the model column E[realised oracle] = mean_i 1 − Π_t(1 − p_it) additionally assumes tiers "
         "independent given the task.", "",
         "**Model K (primary): Kelley true-score shrinkage**, mean-preserving (each tier's mean accuracy is "
         "unchanged): p̂ = m_t + rel·(x̄ − m_t). Cheap: x̄ = mean of its two runs, rel = 2ρ/(1 + ρ) "
         f"= {rho2:.3f} (Spearman-Brown). Mid / frontier: one run, reliability ρ_mf unknown, swept from 1 (no noise) "
         "over the cheap tier's ρ and its 90% interval.", "",
         "| ρ_mf (mid/frontier reliability) | probability oracle | best single tier | selection | luck, anchored "
         "(R_obs − prob. oracle) | E[realised oracle] (model) | luck (model) |", "|---|---|---|---|---|---|---|"]
    for lab, r_mf, s_ in k_rows:
        L.append(f"| {lab} | {f(s_['prob_oracle'])} | {f(s_['best_single_tier'])} ({s_['best_tier']}) | "
                 f"{f(s_['selection'])} | {f(s_['luck_anchored'])} | {f(s_['E_realised_oracle'])} | {f(s_['luck'])} |")
    L += ["", "**Model F (the reviewer's band): realised outcome ± flip probability.** Cheap p = mean of its two "
          "runs; mid / frontier p = (1 − δ)x + δ(1 − x). This shrinks tier means toward 0.5, so only the anchored "
          "luck is reported.", "",
          "| δ | probability oracle | luck, anchored (R_obs − prob. oracle) |", "|---|---|---|"]
    for lab, d, po, lk in f_rows:
        L.append(f"| {lab} | {f(po)} | {f(lk)} |")
    L += ["", "## 3. Estimate", "",
          f"- **Hindsight (luck) component of the realised per-task oracle on these 50 tasks: about "
          f"{min(anchored):.3f} to {max(anchored):.3f} quality points** (anchored estimates over Model K with ρ_mf in the "
          f"cheap tier's 90% reliability interval, and Model F with δ ∈ {{δ*, F}}), against a realised oracle − best "
          f"single tier gap of {R_obs - best_obs:.3f}. With no noise on mid/frontier (ρ_mf = 1, δ = 0) it is "
          f"{min(k_rows[0][2]['luck_anchored'], f_rows[0][3]):.3f}–{max(k_rows[0][2]['luck_anchored'], f_rows[0][3]):.3f}, "
          "the cheap tier's own contribution.",
          f"- Upper sensitivity: the model-internal luck (E[realised oracle] − probability oracle, tiers independent) "
          f"is {min(r[2]['luck'] for r in k_rows[1:]):.3f}–{max(r[2]['luck'] for r in k_rows[1:]):.3f} for ρ_mf < 1. But "
          f"that model predicts a realised oracle of {min(r[2]['E_realised_oracle'] for r in k_rows[1:]):.3f}–"
          f"{max(r[2]['E_realised_oracle'] for r in k_rows[1:]):.3f}, above the observed {R_obs:.3f}: the data favour "
          "positively correlated outcomes across tiers (hard tasks fail everywhere) and/or more repeatable "
          "mid/frontier tiers, either of which lowers luck, so the anchored figures are the estimate and the "
          "model-internal ones an upper bound under independence.",
          f"- Expected realised best-of-three vs expected best single tier: best single tier stays "
          f"{k_rows[0][2]['best_single_tier']:.3f} (frontier) under Model K (mean-preserving); the expected realised "
          f"oracle is {R_obs:.3f} as observed (anchored), {min(r[2]['E_realised_oracle'] for r in k_rows):.3f}–"
          f"{max(r[2]['E_realised_oracle'] for r in k_rows):.3f} under the independent-tier model.",
          "- Assumptions: (i) mid/frontier noise is not measured; ρ_mf / δ borrow the cheap tier's repeatability, "
          "and stronger tiers may be more (or less) repeatable; (ii) the probability oracle is estimated by a "
          "plug-in max over shrunk p̂, itself an estimate (max of noisy estimates is biased upward, so luck may be "
          "understated); (iii) the model E[realised oracle] assumes independence across tiers given the task; "
          "positively correlated outcomes make it smaller; (iv) 50 tasks, the first 50 of the collection, not a "
          "random sample of the 300; (v) the cheap rerun scored higher than run 1, so part of the flip rate may be "
          "drift rather than noise; (vi) correctness is binarised at 0.5.",
          "- Implication: part of the realised per-task oracle's lead over any outcome-blind policy is luck; a "
          "fairer ceiling is the probability oracle, whose estimation needs repeated runs on every tier.", ""]
    out = ROOT / "results" / "crmarena" / "hindsight_oracle.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
