"""Evaluate the GRPO-trained LLM controller on fresh seeded sim tasks (Modal GPU).

  modal run pace/llm/eval_modal.py                       # all checkpoints of run phase1
  modal run pace/llm/eval_modal.py --runs phase1,phase1b    # several runs, one container

Every row is scored through the same reward path (pace.llm.rewards.SimEpisodeReward):
the policy's first action, then ANSWER on the same tier. Rows: the LLM controller
(greedy, and sampled at T=1), the base model zero-shot (greedy, no adapter), the
linear controller in examples/policy_v1_sim.json (greedy first action), fixed
ANSWER on cheap / mid / frontier, and the best fixed action per difficulty proxy
under lambda_bar (an oracle reference). J = quality - lambda_bar * cost, with
lambda_bar the mean learned lambda over the last 100 steps of the run's train_log.jsonl.
Every saved checkpoint (adapter_step100/200/300) is evaluated; the best one with
cost <= budget + 0.02 is named.
The table is written to <run>/eval_table.md on the volume and to results/llm/ locally.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pace.llm.modal_common import GPUS, VOL_PATH, generate, image, load_model, render, volume  # noqa: E402

app = modal.App("pace-llm-eval", image=image)


def llm_actions(model, tok, rendered: list[str], greedy: bool) -> list[str]:
    """Greedy decoding is deterministic per prompt, so identical prompts are decoded once."""
    if greedy:
        uniq = sorted(set(rendered))
        out = dict(zip(uniq, generate(model, tok, uniq, greedy=True)))
        return [out[p] for p in rendered]
    return generate(model, tok, rendered, greedy=False)


@app.function(gpu=GPUS, timeout=3600, volumes={VOL_PATH: volume})
def evaluate(runs: str = "phase1", adapter: str = "", n: int = 2000, seed: int = 777, budget: float = -1.0,
             lam_window: int = 100, tol: float = 0.02) -> dict:
    """Evaluate one or more runs (comma list) in one container: the base model is loaded
    once and the zero-shot / baseline completions are shared; each run is scored with its own lambda_bar."""
    import numpy as np
    from peft import PeftModel

    from pace.controller import Controller
    from pace.harness.sim import TIERS, make_sim
    from pace.llm.env_text import format_action, sim_rows

    t0 = time.time()
    volume.reload()
    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    metas = {r: json.loads((Path(VOL_PATH) / r / "run_meta.json").read_text()) for r in run_list}
    model_name = metas[run_list[0]]["model_loaded"]
    assert all(m["model_loaded"] == model_name for m in metas.values()), "runs use different base models"
    budget = float(metas[run_list[0]]["budget"] if budget < 0 else budget)
    rows = sim_rows(n, seed=seed, budget=budget)
    tasks = [json.loads(r["task"]) for r in rows]

    # ---------------------------------------------------------------- generation (GPU)
    comps: dict[str, list[str]] = {}
    model, tok, _, _ = load_model(model_name)
    model = model.to("cuda").eval()
    adapters_by_run, lam_bar = {}, {}
    for run in run_list:
        out = Path(VOL_PATH) / run
        meta = metas[run]
        lam_bar[run] = lambda_bar(out, lam_window)[0]
        if meta.get("price_in_prompt"):     # the policy is price-conditioned: show it lambda_bar
            from pace.llm.env_text import round_lam, with_price
            prows = sim_rows(n, seed=seed, budget=budget, price=True)
            rendered = [render(tok, with_price(r["user"], round_lam(lam_bar[run]))) for r in prows]
        else:
            rendered = [render(tok, r["user"]) for r in rows]

        def gen(key, m, greedy):
            t = time.time()
            comps[key] = llm_actions(m, tok, rendered, greedy=greedy)
            print(f"[gen] {key}: {time.time() - t:.0f}s", flush=True)

        gen(f"{run}/zero-shot", model, True)
        if adapter:
            ads = [a.strip() for a in adapter.split(",") if a.strip()]
        else:   # every saved checkpoint; adapter_final duplicates the last step when training finished
            ads = sorted([p.name for p in out.glob("adapter_step*")], key=lambda s: int(s.replace("adapter_step", "")))
            if (out / "adapter_final").exists() and f"adapter_step{meta.get('steps')}" not in ads:
                ads.append("adapter_final")
        base = model
        if meta.get("sft"):
            # two-stage run: the SFT adapter is merged into the base; the GRPO adapters sit on top of it
            m = PeftModel.from_pretrained(model, str(out / "adapter_sft")).eval()
            gen(f"{run}/adapter_sft (greedy)", m, True)
            gen(f"{run}/adapter_sft (sampled T=1)", m, False)
            base = m.merge_and_unload()
            ads = ["adapter_sft"] + ads
        adapters_by_run[run] = ads
        for ad in [a for a in ads if a != "adapter_sft"]:
            m = PeftModel.from_pretrained(base, str(out / ad)).eval()
            gen(f"{run}/{ad} (greedy)", m, True)
            gen(f"{run}/{ad} (sampled T=1)", m, False)
            base = m.unload()        # drop this adapter's LoRA layers before loading the next one
        if meta.get("sft"):          # back to the plain base model for the next run
            del base
            model, tok, _, _ = load_model(model_name)
            model = model.to("cuda").eval()

    # linear controller: greedy first action of the served (iterate-averaged, eps-free) policy
    pol = Controller.load("/root/src/examples/policy_v1_sim.json")
    h = make_sim("v1")
    lin = []
    for t in h.sample_tasks(n, seed):
        s = h.reset(t)
        p = pol.base_probs(h.featurize(s), h.legal_actions(s))
        op, tier = pol.actions[int(np.argmax(p))].split("@")
        lin.append(format_action(tier, op))
    comps["linear controller (policy_v1_sim.json, greedy)"] = lin
    for tier in TIERS:
        comps[f"fixed {tier} ANSWER"] = [format_action(tier, "ANSWER")] * n
    (Path(VOL_PATH) / f"eval_completions_seed{seed}.json").write_text(json.dumps(comps))
    volume.commit()

    # ------------------------------------------------------------------- scoring (CPU)
    tables = {}
    for run in run_list:
        tables[run] = score_run(run, metas[run], comps, adapters_by_run[run], tasks, budget, n, seed,
                                lam_window, tol, time.time() - t0)
    volume.commit()
    return tables


def lambda_bar(out: Path, lam_window: int) -> tuple[float, float, int]:
    """(mean lambda over the last lam_window steps, final lambda, window size) of a run's train_log."""
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines() if l.strip()]
    tail = log[-lam_window:]
    return float(sum(r["lambda"] for r in tail) / len(tail)), float(log[-1]["lambda"]), len(tail)


def score_run(run, meta, comps, adapters, tasks, budget, n, seed, lam_window, tol, gen_s) -> str:
    import numpy as np

    from pace.harness.sim import TIERS
    from pace.llm.env_text import format_action
    from pace.llm.rewards import SimEpisodeReward

    import itertools

    out = Path(VOL_PATH) / run
    # lambda-bar: the mean learned lambda over the last `lam_window` steps (primal-dual iterates
    # orbit; the average is the price the run converged to); the final lambda is reported alongside
    lam, lam_final, n_tail = lambda_bar(out, lam_window)
    R = SimEpisodeReward()

    def score(completions):
        os_ = [R(t, c, lam) for t, c in zip(tasks, completions)]
        q = float(np.mean([o.quality for o in os_])); c = float(np.mean([o.cost for o in os_]))
        hist = {}
        for o in os_:
            k = o.action if o.valid else "INVALID"
            hist[k] = hist.get(k, 0) + 1
        return {"quality": q, "cost": c, "J": q - lam * c, "invalid": float(np.mean([not o.valid for o in os_])),
                "mix": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
                "samples": sorted(set(completions))[:6]}

    results, ckpt = {}, {}
    results["base model zero-shot (greedy, no adapter)"] = score(comps[f"{run}/zero-shot"])
    for ad in adapters:
        for mode in ("greedy", "sampled T=1"):
            ckpt[f"LLM {ad} ({mode})"] = results[f"LLM {ad} ({mode})"] = score(comps[f"{run}/{ad} ({mode})"])
    for k in ["linear controller (policy_v1_sim.json, greedy)"] + [f"fixed {t} ANSWER" for t in TIERS]:
        results[k] = score(comps[k])
    # oracle reference: the best single action per observed difficulty proxy under lambda_bar
    legal = [(t, o) for t in TIERS for o in ("ANSWER", "RETRIEVE", "VERIFY", "ESCALATE")]
    best_a = {}
    for proxy in ("easy", "hard"):
        idx = [i for i, t in enumerate(tasks) if t["proxy"] == proxy]
        best_a[proxy] = max(legal, key=lambda a: np.mean([R(tasks[i], format_action(*a), lam).reward for i in idx]))
    results["best fixed action per proxy under lambda_bar (oracle ref.)"] = score(
        [format_action(*best_a[t["proxy"]]) for t in tasks])
    # budget references on these tasks: the best deterministic per-difficulty rule with cost <= B, and the
    # best randomized per-difficulty policy (a mixture of two such rules) at cost <= B
    tot = {}
    for proxy in ("easy", "hard"):
        idx = [i for i, t in enumerate(tasks) if t["proxy"] == proxy]
        for a in legal:
            os_ = [R(tasks[i], format_action(*a), 0.0) for i in idx]
            tot[(proxy, a)] = (sum(o.quality for o in os_) / n, sum(o.cost for o in os_) / n)
    rules = [((ae, ah), tot[("easy", ae)][0] + tot[("hard", ah)][0], tot[("easy", ae)][1] + tot[("hard", ah)][1])
             for ae, ah in itertools.product(legal, legal)]
    det = max((r for r in rules if r[2] <= budget), key=lambda r: r[1])
    results[f"best deterministic per-difficulty rule, cost <= B (easy {' '.join(det[0][0])}, hard "
            f"{' '.join(det[0][1])})"] = score([format_action(*(det[0][0] if t["proxy"] == "easy" else det[0][1]))
                                                 for t in tasks])
    lo = [r for r in rules if r[2] <= budget]; hi = [r for r in rules if r[2] > budget]
    best_mix = (det[1], det[2], 1.0, det[0], det[0])
    for a_, b_ in itertools.product(lo, hi):           # mixture w*a + (1-w)*b with cost exactly B
        w = (b_[2] - budget) / (b_[2] - a_[2])
        q = w * a_[1] + (1 - w) * b_[1]
        if q > best_mix[0]:
            best_mix = (q, budget, w, a_[0], b_[0])
    q_m, c_m = best_mix[0], best_mix[1]
    results[f"best randomized per-difficulty policy at cost B (LP; {best_mix[2]:.2f} x easy "
            f"{' '.join(best_mix[3][0])}/hard {' '.join(best_mix[3][1])} + rest easy {' '.join(best_mix[4][0])}/hard "
            f"{' '.join(best_mix[4][1])})"] = {"quality": q_m, "cost": c_m, "J": q_m - lam * c_m, "invalid": 0.0,
                                              "mix": {}, "samples": []}

    hdr = (f"Run `{run}` (model {meta['model_loaded']}, {meta.get('last_saved_step', '?')} steps; dual step "
           f"lr {meta['lambda']['lr']}, kp {meta['lambda']['kp']}, ema {meta['lambda']['ema']}; beta {meta['beta']}; "
           f"entropy_coef {meta.get('entropy_coef', 0.0)}; num_generations {meta.get('num_generations', 8)}"
           + (f"; SFT warm start {meta['sft']['steps']} steps" if meta.get("sft") else "")
           + (f"; events {meta['events']}" if meta.get("events") else "") + f"). Sim v1, {n} fresh tasks (seed {seed}; training used seed "
           f"{meta['seed']}), budget B = {budget:.2f}. J = quality - lambda_bar * cost with lambda_bar = {lam:.3f}, "
           f"the mean learned lambda over the last {n_tail} training steps (final lambda {lam_final:.3f}). "
           "First action from the policy, then ANSWER on the same tier. Quality = expected success probability. "
           f"'within budget' = cost <= B + {tol:.2f}.\n\n")
    lines = ["| policy | quality | cost | J | within budget | invalid | first-action mix |",
             "|---|---|---|---|---|---|---|"]
    if meta.get("price_in_prompt"):
        hdr = hdr.replace("\n\n", f" The LLM prompts show the price λ={round(lam + 1e-9, 1):.1f} (lambda_bar rounded).\n\n", 1)
    for name, r in results.items():
        mix = ", ".join(f"{k} {v / n:.0%}" for k, v in list(r["mix"].items())[:4])
        lines.append(f"| {name} | {r['quality']:.3f} | {r['cost']:.3f} | {r['J']:.3f} | "
                     f"{'yes' if r['cost'] <= budget + tol + 1e-9 else 'NO'} | {r['invalid']:.1%} | {mix} |")
    ok = {k: v for k, v in ckpt.items() if v["cost"] <= budget + tol + 1e-9}
    best = max(ok, key=lambda k: ok[k]["J"]) if ok else None
    table = hdr + "\n".join(lines) + "\n"
    table += ("\nBest LLM checkpoint within budget (by J): "
              + (f"**{best}**: quality {ok[best]['quality']:.3f}, cost {ok[best]['cost']:.3f}, J {ok[best]['J']:.3f}"
                 if best else "**none** of the LLM checkpoints is within budget") + "\n")
    zs = results["base model zero-shot (greedy, no adapter)"]["samples"]
    table += "\nBase model zero-shot raw outputs (distinct, up to 6): " + json.dumps(zs) + "\n"
    table += f"\nEval (all runs, generation + scoring) {gen_s:.0f}s on {meta.get('gpu')}.\n"
    (out / "eval_table.md").write_text(table)
    (out / "eval_results.json").write_text(json.dumps({"lambda_bar": lam, "lambda_final": lam_final,
                                                       "adapters": adapters, "best_within_budget": best,
                                                       "results": results}, indent=1))
    print(table, flush=True)
    return table


@app.local_entrypoint()
def main(runs: str = "phase1", adapter: str = "", n: int = 2000, seed: int = 777):
    tables = evaluate.remote(runs=runs, adapter=adapter, n=n, seed=seed)
    dst = Path(__file__).resolve().parents[2] / "results" / "llm"
    dst.mkdir(parents=True, exist_ok=True)
    for run, table in tables.items():
        (dst / f"eval_table_{run}.md").write_text(table)
        print(f"wrote {dst / f'eval_table_{run}.md'}")
