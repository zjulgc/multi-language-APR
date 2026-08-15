import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import icl_retrieval
from prompt_utils import cot_instruction, render_chat_prompt, render_icl_prompt


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1, help="Number of prompts to generate per forward pass")
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1, help="Split dataset by global index modulo this value")
    parser.add_argument("--shard_index", type=int, default=0, help="Generate samples where index %% num_shards equals this")
    parser.add_argument("--subset_jsonl", type=str, default="", help="Optional preselected jsonl subset to generate on")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose output JSON already exists")
    parser.add_argument(
        "--unittest_db",
        type=str,
        default="",
        help="Optional xCodeEval unittest_db.json to inject hidden_unit_tests into source_data",
    )
    parser.add_argument(
        "--local_apr_dir",
        type=str,
        default="",
        help="Local APR dir containing train/ validation/ test jsonl files",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="plain",
        choices=["plain", "icl", "cot"],
        help="plain: SFT-matching prompt; icl: RING-style few-shot with retrieved "
        "same-language exemplars; cot: analyze-then-fix zero-shot CoT",
    )
    parser.add_argument("--icl_bank", type=str, default="data/eval/icl_bank_perlang3k.jsonl")
    parser.add_argument("--icl_shots", type=int, default=2)
    parser.add_argument(
        "--prompt_max_len",
        type=int,
        default=2048,
        help="Prompt-side tokenizer truncation; raise to 4096 for --prompt_style icl",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.bfloat16)
    model.eval()

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

    icl_bank = icl_retrieval.load_bank(args.icl_bank) if args.prompt_style == "icl" else None

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
                print(f"Generated/skipped {idx + 1} samples...")
            continue

        instruction = build_instruction(sample)
        buggy = sample.get("bug_source_code", "")
        if args.prompt_style == "cot":
            prompt = render_chat_prompt(tokenizer, cot_instruction(instruction), buggy)
        elif args.prompt_style == "icl":
            exemplars = icl_retrieval.retrieve(
                icl_bank, sample.get("lang_cluster", ""), buggy, args.icl_shots
            )
            prompt = render_icl_prompt(tokenizer, instruction, buggy, exemplars)
        else:
            prompt = render_chat_prompt(tokenizer, instruction, buggy)
        pending.append((idx, prompt, sample, out_file))

        if len(pending) >= args.batch_size:
            write_batch_outputs(model, tokenizer, pending, args)
            pending = []

        if (idx + 1) % 50 == 0:
            print(f"Generated {idx + 1} samples...")

    if pending:
        write_batch_outputs(model, tokenizer, pending, args)

    print(f"Done. Wrote files to {args.output_dir}")


def write_batch_outputs(model, tokenizer, pending, args) -> None:
    prompts = [item[1] for item in pending]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.prompt_max_len
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=args.num_samples > 1,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    for batch_idx, (_, _, sample, out_file) in enumerate(pending):
        choices = []
        start = batch_idx * args.num_samples
        end = start + args.num_samples
        for one in generated[start:end]:
            text = tokenizer.decode(one[prompt_len:], skip_special_tokens=True).strip()
            choices.append({"message": {"content": text}})

        out = {
            "oai_response": {"choices": choices},
            "source_data": to_serializable(sample),
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
