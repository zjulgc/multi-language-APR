#!/usr/bin/env python3
"""Baseline B1 -- MergeRepair-style merging of the 11 per-language LoRA adapters.

MergeRepair (Dehghan et al., ICSME 2024) asks whether task-specific adapters can
be *merged* into one adapter instead of trained jointly. That is exactly the
multilingual-interference question this paper studies, from the opposite
direction: our multiLoRA trains one adapter on all 11 languages (and pays the
interference cost), while merging trains 11 clean per-language adapters and
combines them post hoc, with no joint training at all.

We already have the 11 mono adapters, so this baseline costs zero GPU training.

Merging methods (all from PEFT's reference implementations, which operate in
LoRA parameter space -- i.e. A and B are averaged separately, matching
MergeRepair's "equal-weight averaging applied on parameters"):
  linear      -- equal-weight parameter averaging (task arithmetic)
  ties        -- TIES: trim to top-`density` magnitude, elect sign, disjoint mean
  dare_linear -- DARE: random drop with rescaling, then linear merge

Usage:
  python scripts/merge_mono_adapters.py --out_root checkpoints/merged_mono
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# The 11 xCodeEval languages, named as the mono checkpoints are on disk.
LANGS = ["C", "Cpp", "Csharp", "Go", "Java", "Javascript", "Kotlin", "PHP",
         "Python", "Ruby", "Rust"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--ckpt_root", default="checkpoints")
    ap.add_argument("--prefix", default="mono_vanilla_",
                    help="Checkpoint dir prefix; '_llama3' suffix selects the second base model.")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--out_root", default="checkpoints/merged_mono")
    ap.add_argument("--methods", nargs="+", default=["linear", "ties", "dare_linear"])
    ap.add_argument("--density", type=float, default=0.2,
                    help="Fraction of parameters kept by TIES/DARE (PEFT default convention).")
    args = ap.parse_args()

    adapters = [(lang, os.path.join(args.ckpt_root, f"{args.prefix}{lang}{args.suffix}"))
                for lang in args.langs]
    missing = [p for _, p in adapters if not os.path.exists(os.path.join(p, "adapter_config.json"))]
    if missing:
        raise FileNotFoundError(f"missing adapter dirs: {missing}")

    t0 = time.time()
    print(f"[merge] loading base on CPU (fp32 for merge precision) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True,
    )

    first_lang, first_path = adapters[0]
    peft_model = PeftModel.from_pretrained(model, first_path, adapter_name=first_lang)
    for lang, path in adapters[1:]:
        peft_model.load_adapter(path, adapter_name=lang)
    names = [lang for lang, _ in adapters]
    print(f"[merge] {len(names)} adapters loaded ({time.time()-t0:.0f}s): {names}", flush=True)

    weights = [1.0 / len(names)] * len(names)  # equal weight: no language is privileged
    os.makedirs(args.out_root, exist_ok=True)

    for method in args.methods:
        out_name = f"merged_{method}"
        kwargs = {}
        if method in ("ties", "dare_linear", "dare_ties", "magnitude_prune"):
            kwargs["density"] = args.density
        print(f"[merge] combining with '{method}' {kwargs} ...", flush=True)
        peft_model.add_weighted_adapter(
            adapters=names, weights=weights, adapter_name=out_name,
            combination_type=method, **kwargs,
        )
        out_dir = os.path.join(args.out_root, out_name)
        peft_model.save_pretrained(args.out_root, selected_adapters=[out_name])
        meta = {
            "baseline": "MergeRepair-style adapter merging (Dehghan et al., ICSME 2024)",
            "combination_type": method,
            "source_adapters": {lang: path for lang, path in adapters},
            "weights": "equal (1/N)",
            "density": kwargs.get("density"),
            "peft_impl": "PeftModel.add_weighted_adapter",
        }
        with open(os.path.join(out_dir, "merge_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[merge] saved -> {out_dir}", flush=True)

    print(f"[merge] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
