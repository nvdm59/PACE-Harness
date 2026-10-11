"""Shared Modal pieces for the LLM controller apps (image, volume, model loading).
Imports only `modal` at top level, so the local entrypoint (any Python) can import it."""
from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]   # locally: the repo root
VOL_PATH = "/vol"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GPUS = ["A100-40GB", "A10G"]          # Modal tries them in order

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", "transformers==5.19.0", "trl==1.15.0", "peft==0.21.2",
                 "accelerate==1.15.0", "datasets==5.1.0", "numpy")
    .env({"PYTHONPATH": "/root/src", "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO / "pace"), "/root/src/pace", ignore=["**/__pycache__/**"])
    .add_local_file(str(REPO / "examples" / "policy_v1_sim.json"), "/root/src/examples/policy_v1_sim.json")
)
volume = modal.Volume.from_name("pace-llm", create_if_missing=True)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(name: str):
    """-> (model, tokenizer, name actually loaded, note). Falls back loudly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    note = None
    for cand in [name] + ([FALLBACK_MODEL] if name != FALLBACK_MODEL else []):
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            model = AutoModelForCausalLM.from_pretrained(cand, dtype=torch.bfloat16)
            if cand != name:
                print(f"[model] FALLBACK: {name} failed to load ({note}); using {cand}", flush=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return model, tok, cand, note
        except Exception as e:  # noqa: BLE001
            note = f"{type(e).__name__}: {e}"[:500]
            print(f"[model] could not load {cand}: {note}", flush=True)
    raise RuntimeError(f"no model could be loaded: {note}")


def render(tok, user: str) -> str:
    from pace.llm.env_text import chat_messages
    kw = {"tokenize": False, "add_generation_prompt": True}
    try:   # Qwen3: switch the thinking block off (16-token completions)
        return tok.apply_chat_template(chat_messages(user), enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(chat_messages(user), **kw)


def generate(model, tok, prompts: list[str], greedy: bool, bs: int = 256, seed: int = 0,
             temperature: float = 1.0) -> list[str]:
    """Batched decoding of up to 16 new tokens (left padding)."""
    import torch
    tok.padding_side = "left"
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=16, do_sample=not greedy,
                               temperature=temperature if not greedy else None, top_p=1.0 if not greedy else None,
                               top_k=0 if not greedy else None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def run_sft(model, tok, train_rows: list[dict], steps: int, lr: float, bs: int = 16, warmup: int = 10, log=print):
    """Plain cross-entropy LoRA fine-tune on (prompt -> completion) pairs; loss on the completion
    tokens (+ end-of-turn) only. Returns the PeftModel."""
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS,
                                          task_type="CAUSAL_LM"))
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # fp32 LoRA weights, bf16 base under autocast
    pm.to("cuda").train()
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    ex = []
    for r in train_rows:
        p_ids = tok(render(tok, r["user"]), add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ex.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    rng = random.Random(0)
    order, k = [], 0
    hist = []
    for step in range(steps):
        if k + bs > len(order):
            order = list(range(len(ex))); rng.shuffle(order); k = 0
        batch = [ex[i] for i in order[k:k + bs]]; k += bs
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = pm(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        hist.append(float(loss))
        if step % 25 == 0 or step == steps - 1:
            log(f"[sft {step}] loss {sum(hist[-25:]) / len(hist[-25:]):.4f}")
    pm.eval()
    return pm, hist
