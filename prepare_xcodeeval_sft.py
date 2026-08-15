import argparse
import json
import os
from glob import glob
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


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

    io_examples: List[str] = []
    for idx, (inp, out) in enumerate(zip(sample_inputs, sample_outputs), start=1):
        io_examples.append(
            f"Example {idx} Input:\n{inp}\nExample {idx} Output:\n{out}"
        )

    examples_block = "\n\n".join(io_examples)
    if not examples_block:
        examples_block = "No examples provided."

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


def convert_split(rows: Iterable[Dict[str, Any]], include_output: bool) -> List[Dict[str, Any]]:
    converted = []
    for row in rows:
        item = {
            "instruction": build_instruction(row),
            "input": row.get("bug_source_code", ""),
            "lang_cluster": row.get("lang_cluster", ""),
            "src_uid": row.get("src_uid", ""),
            "bug_exec_outcome": row.get("bug_exec_outcome", ""),
        }
        if include_output:
            item["output"] = row.get("fix_source_code", "")
        converted.append(item)
    return converted


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_local_jsonl_with_limit(files: List[str], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit > 0 and len(rows) >= limit:
                        return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--val_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument(
        "--local_apr_dir",
        type=str,
        default="",
        help="Local APR dir containing train/ validation/ test jsonl files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_apr_dir:
        train_files = sorted(glob(os.path.join(args.local_apr_dir, "train", "*.jsonl")))
        val_files = sorted(glob(os.path.join(args.local_apr_dir, "validation", "*.jsonl")))
        test_files = sorted(glob(os.path.join(args.local_apr_dir, "test", "*.jsonl")))
        if args.train_limit > 0 or args.val_limit > 0 or args.test_limit > 0:
            train_rows = read_local_jsonl_with_limit(train_files, args.train_limit)
            val_rows = read_local_jsonl_with_limit(val_files, args.val_limit)
            test_rows = read_local_jsonl_with_limit(test_files, args.test_limit)
        else:
            ds = load_dataset(
                "json",
                data_files={
                    "train": train_files,
                    "validation": val_files,
                    "test": test_files,
                },
            )
            train_rows = ds["train"]
            val_rows = ds["validation"]
            test_rows = ds["test"]
    else:
        ds = load_dataset("NTU-NLP-sg/xCodeEval", "apr", trust_remote_code=True)
        train_rows = ds["train"]
        val_rows = ds["validation"]
        test_rows = ds["test"]

    if args.train_limit > 0 and hasattr(train_rows, "select"):
        train_rows = train_rows.select(range(min(args.train_limit, len(train_rows))))
    if args.val_limit > 0 and hasattr(val_rows, "select"):
        val_rows = val_rows.select(range(min(args.val_limit, len(val_rows))))
    if args.test_limit > 0 and hasattr(test_rows, "select"):
        test_rows = test_rows.select(range(min(args.test_limit, len(test_rows))))

    train_out = convert_split(train_rows, include_output=True)
    val_out = convert_split(val_rows, include_output=True)
    test_out = convert_split(test_rows, include_output=False)

    write_jsonl(os.path.join(args.output_dir, "train_sft.jsonl"), train_out)
    write_jsonl(os.path.join(args.output_dir, "validation_sft.jsonl"), val_out)
    write_jsonl(os.path.join(args.output_dir, "test_infer.jsonl"), test_out)

    print(f"Train rows: {len(train_out)}")
    print(f"Validation rows: {len(val_out)}")
    print(f"Test rows: {len(test_out)}")


if __name__ == "__main__":
    main()
