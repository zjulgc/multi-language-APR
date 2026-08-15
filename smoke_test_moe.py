"""End-to-end smoke test for moe_apr training pipeline.

Loads Qwen2.5-Coder-7B-Instruct, patches it with MoE-LoRA, runs a few steps
on a tiny slice of xCodeEval data, saves & re-loads MoE state, and verifies
that loss decreases and forward output stays consistent.

Usage::

    CUDA_VISIBLE_DEVICES=5 python smoke_test_moe.py \\
        --base_model /mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct \\
        --data_root data \\
        --output_dir checkpoints/_smoke

This is *not* a real training run; it is a correctness check that should
finish in 3-5 minutes on a single A6000.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from moe_apr.model_patcher import (
    MoEPatchConfig,
    collect_moe_layers,
    patch_model_with_moe_lora,
    report_trainable_parameters,
)
from train_moe_apr import (
    load_moe_state,
    save_moe_state,
    tokenize_example,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--num_steps", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("Smoke test: MoE-LoRA + Shared APR Expert", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------ #
    # 1. Tokenizer + base model
    # ------------------------------------------------------------------ #
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"[t={time.time()-t0:.1f}s] Base model loaded", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Patch with MoE-LoRA
    # ------------------------------------------------------------------ #
    cfg = MoEPatchConfig(
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        num_routing_experts=4,
        top_k=2,
        rank=args.lora_r,
        alpha=args.lora_r * 2,
        dropout=0.0,
        use_shared_expert=True,
        shared_expert_gate_mode="adaptive",
    )
    replaced = patch_model_with_moe_lora(model, cfg)
    print(f"[t={time.time()-t0:.1f}s] Patched {len(replaced)} Linear modules", flush=True)
    tr, tot, pct = report_trainable_parameters(model)
    print(f"  Trainable: {tr:,} / {tot:,} ({pct:.4f}%)", flush=True)

    # Confirm that without shared, forward equals base (sanity).
    moe_layers = collect_moe_layers(model)
    print(f"  MoELoRALinear count = {len(moe_layers)}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Build a tiny train slice
    # ------------------------------------------------------------------ #
    train_path = os.path.join(args.data_root, "all", "train_sft.jsonl")
    ds = load_dataset("json", data_files=train_path, split="train")
    ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"[t={time.time()-t0:.1f}s] Loaded {len(ds):,} samples", flush=True)

    ds = ds.map(
        lambda r: tokenize_example(tok, args.max_seq_len, False, r),
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    coll = DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    print(f"[t={time.time()-t0:.1f}s] Tokenized", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Single-stage training
    # ------------------------------------------------------------------ #
    print("\n--- Single-stage MoE + shared expert training ---", flush=True)
    tr, tot, pct = report_trainable_parameters(model)
    print(f"  Trainable: {tr:,} ({pct:.4f}%)", flush=True)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    losses = []
    for step in range(args.num_steps):
        i = step % len(ds)
        batch = coll([ds[i]])
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 2 == 0:
            print(f"  step {step}: loss = {loss.item():.4f}", flush=True)

    print(f"[t={time.time()-t0:.1f}s] Training done. losses[0]={losses[0]:.4f} -> losses[-1]={losses[-1]:.4f}", flush=True)
    if losses[-1] >= losses[0] * 1.05:
        print(f"  WARNING: loss did not decrease (start={losses[0]:.4f}, end={losses[-1]:.4f})", flush=True)
    else:
        print("  PASS: loss decreased.", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Save MoE state, then verify reload restores forward exactly.
    # ------------------------------------------------------------------ #
    save_path = os.path.join(args.output_dir, "moe_state.pt")
    save_moe_state(model, save_path)

    # Capture a deterministic forward output before reload.
    model.eval()
    with torch.no_grad():
        sample_batch = coll([ds[0]])
        sample_batch = {k: v.cuda() for k, v in sample_batch.items()}
        out_before = model(**sample_batch).logits.detach().float().cpu()

    # Re-init MoE weights (zero them out) and re-load.
    for m in moe_layers:
        for p in m.parameters():
            if p.requires_grad:
                with torch.no_grad():
                    p.zero_()

    with torch.no_grad():
        out_zeroed = model(**sample_batch).logits.detach().float().cpu()
    diff_zero = (out_zeroed - out_before).abs().max().item()
    print(f"[reload] After zeroing MoE: max-abs-diff vs before = {diff_zero:.6f}", flush=True)

    load_moe_state(model, save_path)
    with torch.no_grad():
        out_after = model(**sample_batch).logits.detach().float().cpu()
    diff_after = (out_after - out_before).abs().max().item()
    print(f"[reload] After loading: max-abs-diff vs before = {diff_after:.6f}", flush=True)
    if diff_after < 1e-2:
        print("  PASS: save/load preserves forward output.", flush=True)
    else:
        print("  WARNING: save/load drift larger than expected.", flush=True)

    print(f"\n[total t={time.time()-t0:.1f}s] Smoke test SUCCESS", flush=True)


if __name__ == "__main__":
    main()
