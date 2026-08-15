"""Quick evaluation script for in-flight training monitoring.

Computes two cheap metrics on a held-out validation subset:

1. **LM loss** (next-token cross-entropy on `output` part only) — same loss
   used during training. Aggregated overall and per `lang_cluster`.

2. **Greedy generation accuracy**:
   - `exact_match`: % of samples whose greedy decode equals oracle fix verbatim.
   - `prefix_match_50`: % of samples where the first 50 tokens of the greedy
     decode match the oracle fix's first 50 tokens (a softer match).

Greedy generation is slow on 7B; default ``--num_gen_samples 200`` keeps it
to a few minutes. LM loss runs on the full dev split.

This is **NOT** a substitute for ExecEval pass@1; treat it as a proxy signal
during training (e.g., to confirm loss is improving in a meaningful way).

Two model modes:

    --mode base              : evaluate the base Qwen model directly (zero-shot baseline).
    --mode peft   --adapter X: evaluate base + standard PEFT LoRA adapter (HF peft format).
    --mode moe    --moe_state X --patch_config Y: evaluate base + our MoE-LoRA.

Example::

    PYTHONPATH=. python quick_eval.py \\
        --base_model /mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct \\
        --val_file data/all/validation_sft.jsonl \\
        --mode moe --moe_state checkpoints/single_stage_adaptive_shared/moe_state.pt \\
        --num_loss_samples 1000 --num_gen_samples 100 \\
        --output_json quick_eval_single_stage.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_utils import render_chat_prompt, tokenize_example
from train_moe_apr import load_moe_state
from moe_apr.model_patcher import (
    MoEPatchConfig,
    patch_model_with_moe_lora,
)


def load_model(args) -> tuple:
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    print("Loading base model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if args.mode == "peft":
        from peft import PeftModel

        print(f"Loading PEFT adapter: {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.bfloat16)
    elif args.mode == "moe":
        if not args.patch_config:
            args.patch_config = os.path.join(os.path.dirname(args.moe_state), "patch_config.json")
        with open(args.patch_config) as f:
            cfg = json.load(f)
        patch_cfg = MoEPatchConfig.from_dict(cfg)
        replaced = patch_model_with_moe_lora(model, patch_cfg)
        print(f"Patched {len(replaced)} modules", flush=True)
        load_moe_state(model, args.moe_state)
    elif args.mode == "base":
        pass
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    model.eval()
    model.config.use_cache = True
    return model, tok


@torch.no_grad()
def compute_lm_loss(model, tok, samples: List[Dict[str, Any]], max_len: int, debug_first: int = 3) -> Dict[str, float]:
    per_lang_loss: Dict[str, List[float]] = defaultdict(list)
    overall_losses: List[float] = []
    n_skipped_empty = 0
    n_skipped_truncated = 0
    n_skipped_no_target = 0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    device = next(model.parameters()).device

    for idx, s in enumerate(samples):
        oracle = (s.get("output") or "").strip()
        if not oracle:
            n_skipped_empty += 1
            continue

        encoded = tokenize_example(tok, max_len, train_on_inputs=False, row=s)
        full_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
        target_tokens = (labels != -100).sum().item()

        if idx < debug_first:
            print(
                f"[lm_loss debug] sample={idx} target_tokens={target_tokens} "
                f"full_len={full_ids.shape[1]} oracle_chars={len(oracle)}",
                flush=True,
            )

        if target_tokens <= 1:
            n_skipped_truncated += 1
            continue

        logits = model(full_ids).logits  # (1, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        n_tokens = (shift_labels != -100).sum().item()
        if n_tokens == 0:
            n_skipped_no_target += 1
            continue
        loss_sum = loss_fn(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
        loss = loss_sum / n_tokens

        overall_losses.append(loss)
        per_lang_loss[s.get("lang_cluster", "UNK")].append(loss)

    n_total = len(samples)
    n_used = len(overall_losses)
    if n_used == 0:
        print(f"[lm_loss WARN] 0 samples used out of {n_total}: empty_output={n_skipped_empty} truncated={n_skipped_truncated} no_target={n_skipped_no_target}", flush=True)
        print("[lm_loss WARN] HINT: are you running on validation_sft.jsonl (oracles hidden)? Try intrain_validation_sft.jsonl.", flush=True)
    else:
        print(f"[lm_loss] used {n_used}/{n_total} samples (skipped: empty={n_skipped_empty}, truncated={n_skipped_truncated}, no_target={n_skipped_no_target})", flush=True)

    out = {
        "overall_lm_loss": sum(overall_losses) / max(1, n_used),
        "n_samples_loss": n_used,
        "n_samples_total": n_total,
        "n_skipped_empty_output": n_skipped_empty,
        "n_skipped_truncated": n_skipped_truncated,
        "n_skipped_no_target": n_skipped_no_target,
        "per_lang_lm_loss": {k: sum(v) / len(v) for k, v in per_lang_loss.items()},
        "per_lang_n": {k: len(v) for k, v in per_lang_loss.items()},
    }
    return out


@torch.no_grad()
def compute_greedy_match(model, tok, samples: List[Dict[str, Any]], max_input_len: int, max_new_tokens: int) -> Dict[str, Any]:
    exact_per_lang: Dict[str, List[int]] = defaultdict(list)
    prefix50_per_lang: Dict[str, List[int]] = defaultdict(list)
    overall_exact: List[int] = []
    overall_prefix50: List[int] = []

    for s in samples:
        prompt = render_chat_prompt(tok, s["instruction"], s["input"])
        oracle = s["output"].strip()
        ids = tok(prompt, truncation=True, max_length=max_input_len, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

        is_exact = int(gen == oracle)
        oracle_pref = tok(oracle, return_tensors="pt").input_ids[0, :50].tolist()
        gen_pref = tok(gen, return_tensors="pt").input_ids[0, :50].tolist()
        is_prefix50 = int(oracle_pref == gen_pref)

        lang = s.get("lang_cluster", "UNK")
        exact_per_lang[lang].append(is_exact)
        prefix50_per_lang[lang].append(is_prefix50)
        overall_exact.append(is_exact)
        overall_prefix50.append(is_prefix50)

    return {
        "overall_exact_match": sum(overall_exact) / max(1, len(overall_exact)),
        "overall_prefix50_match": sum(overall_prefix50) / max(1, len(overall_prefix50)),
        "n_samples_gen": len(overall_exact),
        "per_lang_exact_match": {k: sum(v) / len(v) for k, v in exact_per_lang.items()},
        "per_lang_prefix50_match": {k: sum(v) / len(v) for k, v in prefix50_per_lang.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--mode", choices=["base", "peft", "moe"], required=True)
    ap.add_argument("--adapter", default="", help="PEFT adapter dir (mode=peft)")
    ap.add_argument("--moe_state", default="", help="moe_state.pt path (mode=moe)")
    ap.add_argument("--patch_config", default="", help="auto-derived from --moe_state dir if empty")
    ap.add_argument("--num_loss_samples", type=int, default=500)
    ap.add_argument("--num_gen_samples", type=int, default=100)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_input_len", type=int, default=1536)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--balanced_per_lang", action="store_true",
                    help="Sample roughly equal samples per lang_cluster for both loss and gen evals")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(args)
    print(f"[t={time.time()-t0:.1f}s] Model ready", flush=True)

    ds = load_dataset("json", data_files=args.val_file, split="train")
    print(f"Loaded {len(ds):,} validation samples", flush=True)

    if args.balanced_per_lang:
        per_lang: Dict[str, List[int]] = defaultdict(list)
        for i, lc in enumerate(ds["lang_cluster"]):
            per_lang[lc].append(i)
        rng = torch.Generator().manual_seed(args.seed)
        loss_idx, gen_idx = [], []
        per_lang_loss_n = max(1, args.num_loss_samples // max(1, len(per_lang)))
        per_lang_gen_n = max(1, args.num_gen_samples // max(1, len(per_lang)))
        for lc, idxs in per_lang.items():
            perm = torch.randperm(len(idxs), generator=rng).tolist()
            loss_idx += [idxs[i] for i in perm[:per_lang_loss_n]]
            gen_idx += [idxs[i] for i in perm[:per_lang_gen_n]]
        print(f"Balanced sampling: {len(loss_idx)} for loss, {len(gen_idx)} for gen", flush=True)
    else:
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(ds), generator=rng).tolist()
        loss_idx = perm[:args.num_loss_samples]
        gen_idx = perm[:args.num_gen_samples]

    loss_samples = [ds[i] for i in loss_idx]
    gen_samples = [ds[i] for i in gen_idx]

    print("Computing LM loss ...", flush=True)
    loss_metrics = compute_lm_loss(model, tok, loss_samples, args.max_seq_len)
    print(f"[t={time.time()-t0:.1f}s] LM loss: {loss_metrics['overall_lm_loss']:.4f}", flush=True)

    print(f"Computing greedy generation match on {len(gen_samples)} samples ...", flush=True)
    gen_metrics = compute_greedy_match(model, tok, gen_samples, args.max_input_len, args.max_new_tokens)
    print(
        f"[t={time.time()-t0:.1f}s] greedy exact={gen_metrics['overall_exact_match']:.4f}, "
        f"prefix50={gen_metrics['overall_prefix50_match']:.4f}",
        flush=True,
    )

    report = {
        "mode": args.mode,
        "adapter": args.adapter,
        "moe_state": args.moe_state,
        "val_file": args.val_file,
        "num_loss_samples": args.num_loss_samples,
        "num_gen_samples": args.num_gen_samples,
        "balanced_per_lang": args.balanced_per_lang,
        **loss_metrics,
        **gen_metrics,
        "elapsed_seconds": time.time() - t0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
