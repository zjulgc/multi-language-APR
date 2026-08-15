"""Generate APR candidates with a MoE-LoRA + Shared APR Expert checkpoint.

Output format is identical to ``generate_apr_local.py`` so it works with
``instruction_dataset/xCodeEval_repo/evaluation/apr/eval_apr.py``.

Example::

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python generate_moe_apr.py \\
        --base_model /mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct \\
        --moe_state checkpoints/single_stage_adaptive_shared/moe_state.pt \\
        --patch_config checkpoints/single_stage_adaptive_shared/patch_config.json \\
        --local_apr_dir instruction_dataset/xCodeEval/apr \\
        --split validation \\
        --output_dir dumped/oai/apr_n_sample_20 \\
        --num_samples 1 \\
        --temperature 0.2 \\
        --max_items 500
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_apr.model_patcher import (
    MoEPatchConfig,
    patch_model_with_moe_lora,
    apply_inference_bypass_from_env,
)
from prompt_utils import render_chat_prompt
from train_moe_apr import load_moe_state

# Re-use generate_apr_local helpers via copy (avoid hard dep).


def _to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)]


def build_instruction(sample: Dict[str, Any]) -> str:
    sample_inputs = _to_list(sample.get("prob_desc_sample_inputs", []))
    sample_outputs = _to_list(sample.get("prob_desc_sample_outputs", []))
    io_examples = []
    for idx, (inp, out) in enumerate(zip(sample_inputs, sample_outputs), start=1):
        io_examples.append(f"Example {idx} Input:\n{inp}\nExample {idx} Output:\n{out}")
    examples_block = "\n\n".join(io_examples) if io_examples else "No examples provided."
    return (
        f"Fix the buggy {sample.get('lang_cluster', 'programming')} code.\n"
        f"Problem Description:\n{sample.get('prob_desc_description', '')}\n\n"
        f"Input Specification:\n{sample.get('prob_desc_input_spec', '')}\n\n"
        f"Output Specification:\n{sample.get('prob_desc_output_spec', '')}\n\n"
        f"{examples_block}\n\n"
        f"Notes:\n{sample.get('prob_desc_notes', '')}\n\n"
        f"Input from: {sample.get('prob_desc_input_from', '')}\n"
        f"Output to: {sample.get('prob_desc_output_to', '')}\n\n"
        f"Bug outcome: {sample.get('bug_exec_outcome', '')}\n\n"
        "Return only the fixed code without extra explanation."
    )


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def load_unittest_db(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def attach_hidden_unit_tests(sample: Dict[str, Any], unittest_db: Dict[str, Any]) -> Dict[str, Any]:
    sample = dict(sample)
    if "hidden_unit_tests" not in sample and unittest_db and sample.get("src_uid") in unittest_db:
        sample["hidden_unit_tests"] = json.dumps(unittest_db[sample["src_uid"]], ensure_ascii=False)
    return sample


def is_valid_output(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        choices = payload.get("oai_response", {}).get("choices", [])
        return bool(choices) and "source_data" in payload
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--moe_state", required=True, help="Path to moe_state.pt produced by train_moe_apr.py")
    ap.add_argument("--patch_config", default="", help="Optional path to patch_config.json (auto-derived from moe_state dir)")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1, help="Number of prompts to generate per forward pass")
    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help="Split dataset by global index modulo this value")
    ap.add_argument("--shard_index", type=int, default=0, help="Generate samples where index %% num_shards equals this")
    ap.add_argument("--max_input_len", type=int, default=2048)
    ap.add_argument("--local_apr_dir", default="")
    ap.add_argument("--subset_jsonl", default="", help="Optional preselected jsonl subset to generate on")
    ap.add_argument("--skip_existing", action="store_true", help="Skip samples whose output JSON already exists")
    ap.add_argument(
        "--unittest_db",
        default="",
        help="Optional xCodeEval unittest_db.json to inject hidden_unit_tests into source_data",
    )
    args = ap.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.patch_config:
        candidate = os.path.join(os.path.dirname(args.moe_state), "patch_config.json")
        if os.path.exists(candidate):
            args.patch_config = candidate
    if not args.patch_config:
        raise FileNotFoundError("Pass --patch_config or place patch_config.json next to moe_state.pt")

    with open(args.patch_config) as f:
        cfg_dict = json.load(f)
    patch_cfg = MoEPatchConfig.from_dict(cfg_dict)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    replaced = patch_model_with_moe_lora(model, patch_cfg)
    print(f"Patched {len(replaced)} modules.", flush=True)
    load_moe_state(model, args.moe_state)
    apply_inference_bypass_from_env(model)
    model.eval()
    model.config.use_cache = True

    if args.subset_jsonl:
        ds = load_dataset("json", data_files=args.subset_jsonl, split="train")
    elif args.local_apr_dir:
        split_files = sorted(glob(os.path.join(args.local_apr_dir, args.split, "*.jsonl")))
        ds = load_dataset("json", data_files={args.split: split_files})[args.split]
    else:
        ds = load_dataset("NTU-NLP-sg/xCodeEval", "apr", trust_remote_code=True)[args.split]
    if args.max_items > 0:
        ds = ds.select(range(min(args.max_items, len(ds))))

    unittest_db = load_unittest_db(args.unittest_db)

    print(f"Generating on {len(ds)} samples ...", flush=True)
    pending = []
    for idx, sample in enumerate(ds):
        if idx % args.num_shards != args.shard_index:
            continue
        sample = attach_hidden_unit_tests(sample, unittest_db)
        file_idx = args.start_index + idx
        out_file = os.path.join(
            args.output_dir, f"{file_idx}_{args.temperature}_{sample.get('lang_cluster', 'UNK')}.json"
        )
        if args.skip_existing and os.path.exists(out_file) and is_valid_output(out_file):
            if (idx + 1) % 50 == 0:
                print(f"  skipped/generated {idx + 1}/{len(ds)}", flush=True)
            continue

        instruction = build_instruction(sample)
        prompt = render_chat_prompt(tok, instruction, sample.get("bug_source_code", ""))
        pending.append((idx, prompt, sample, out_file))

        if len(pending) >= args.batch_size:
            write_batch_outputs(model, tok, pending, args)
            pending = []

        if (idx + 1) % 50 == 0:
            print(f"  generated {idx + 1}/{len(ds)}", flush=True)

    if pending:
        write_batch_outputs(model, tok, pending, args)

    print(f"Done. Wrote files to {args.output_dir}", flush=True)


def write_batch_outputs(model, tok, pending, args) -> None:
    prompts = [item[1] for item in pending]
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=args.num_samples > 1 or args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    for batch_idx, (_, _, sample, out_file) in enumerate(pending):
        choices = []
        start = batch_idx * args.num_samples
        end = start + args.num_samples
        for one in generated[start:end]:
            text = tok.decode(one[prompt_len:], skip_special_tokens=True).strip()
            choices.append({"message": {"content": text}})

        out = {
            "oai_response": {"choices": choices},
            "source_data": to_serializable(sample),
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
