"""Prepare xCodeEval APR data for MoE-LoRA training.

Outputs (relative to --output_dir):

    all/                                   <- full mixed data
        train_sft.jsonl
        validation_sft.jsonl
        test_infer.jsonl
    by_family/
        c_family/                          <- C / C++ / C#
            train_sft.jsonl
            validation_sft.jsonl
        jvm_family/                        <- Java / Kotlin
            train_sft.jsonl
            validation_sft.jsonl
        dynamic_typed/                     <- Python / JavaScript / Ruby / PHP
            train_sft.jsonl
            validation_sft.jsonl
        systems/                           <- Rust / Go
            train_sft.jsonl
            validation_sft.jsonl
    distribution_report.json               <- per-split / per-family / per-language counts

The script joins problem descriptions from problem_descriptions.jsonl by `src_uid`
to enrich each sample's instruction. Test split is kept un-split-by-family because
evaluation aggregates per-language metrics across the whole 17k test set.

Usage:

    python prepare_xcodeeval_by_family.py \
        --apr_dir instruction_dataset/xCodeEval/apr \
        --problem_desc instruction_dataset/xCodeEval/problem_descriptions.jsonl \
        --output_dir data
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from glob import glob
from typing import Any, Dict, Iterable, List, Optional


LANG_FAMILY: Dict[str, str] = {
    "C": "c_family",
    "C++": "c_family",
    "C#": "c_family",
    "Java": "jvm_family",
    "Kotlin": "jvm_family",
    "Python": "dynamic_typed",
    "Javascript": "dynamic_typed",
    "JavaScript": "dynamic_typed",
    "Ruby": "dynamic_typed",
    "PHP": "dynamic_typed",
    "Rust": "systems",
    "Go": "systems",
}

FAMILIES = ["c_family", "jvm_family", "dynamic_typed", "systems"]


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


def load_problem_descriptions(path: str) -> Dict[str, Dict[str, Any]]:
    """Map src_uid -> problem description record."""
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("src_uid")
            if uid:
                out[uid] = row
    return out


def build_instruction(sample: Dict[str, Any], problem: Optional[Dict[str, Any]] = None) -> str:
    """Build an Alpaca-style instruction; merges problem description when available."""
    p = problem or {}

    description = p.get("description", "") or sample.get("prob_desc_description", "")
    input_spec = p.get("input_spec", "") or sample.get("prob_desc_input_spec", "")
    output_spec = p.get("output_spec", "") or sample.get("prob_desc_output_spec", "")
    notes = p.get("notes", "") or sample.get("prob_desc_notes", "")
    input_from = p.get("input_from", "") or sample.get("prob_desc_input_from", "")
    output_to = p.get("output_to", "") or sample.get("prob_desc_output_to", "")

    sample_inputs = _to_list(p.get("sample_inputs", []) or sample.get("prob_desc_sample_inputs", []))
    sample_outputs = _to_list(p.get("sample_outputs", []) or sample.get("prob_desc_sample_outputs", []))

    io_examples = []
    for idx, (inp, out) in enumerate(zip(sample_inputs, sample_outputs), start=1):
        io_examples.append(f"Example {idx} Input:\n{inp}\nExample {idx} Output:\n{out}")
    examples_block = "\n\n".join(io_examples) if io_examples else "No examples provided."

    lang = sample.get("lang_cluster", "programming")

    return (
        f"Fix the buggy {lang} code.\n"
        f"Problem Description:\n{description}\n\n"
        f"Input Specification:\n{input_spec}\n\n"
        f"Output Specification:\n{output_spec}\n\n"
        f"{examples_block}\n\n"
        f"Notes:\n{notes}\n\n"
        f"Input from: {input_from}\n"
        f"Output to: {output_to}\n\n"
        f"Bug outcome: {sample.get('bug_exec_outcome', '')}\n\n"
        "Return only the fixed code without extra explanation."
    )


def to_sft_record(sample: Dict[str, Any], problem: Optional[Dict[str, Any]], include_output: bool) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "instruction": build_instruction(sample, problem),
        "input": sample.get("bug_source_code", ""),
        "lang_cluster": sample.get("lang_cluster", ""),
        "src_uid": sample.get("src_uid", ""),
        "bug_exec_outcome": sample.get("bug_exec_outcome", ""),
        "apr_id": sample.get("apr_id", ""),
    }
    if include_output:
        item["output"] = sample.get("fix_source_code", "")
    return item


def iter_split(split_dir: str) -> Iterable[Dict[str, Any]]:
    files = sorted(glob(os.path.join(split_dir, "*.jsonl")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def process_split(
    split_dir: str,
    split_name: str,
    output_dir: str,
    problems: Dict[str, Dict[str, Any]],
    include_output: bool,
    write_per_family: bool,
    sample_limit: int = -1,
    log_every: int = 100_000,
) -> Dict[str, Any]:
    """Process one split (train / validation / test).

    Writes:
      - {output_dir}/all/{split_filename}
      - {output_dir}/by_family/{family}/{split_filename}  (when write_per_family)
    Returns a stats dict {language: count, family_counts: {family: count}, total: int}.
    """
    if split_name == "test":
        out_filename = "test_infer.jsonl"
    elif split_name == "validation":
        out_filename = "validation_sft.jsonl"
    else:
        out_filename = "train_sft.jsonl"

    all_path = os.path.join(output_dir, "all", out_filename)
    os.makedirs(os.path.dirname(all_path), exist_ok=True)
    all_writer = open(all_path, "w", encoding="utf-8")

    family_writers: Dict[str, Any] = {}
    if write_per_family:
        for fam in FAMILIES:
            fam_path = os.path.join(output_dir, "by_family", fam, out_filename)
            os.makedirs(os.path.dirname(fam_path), exist_ok=True)
            family_writers[fam] = open(fam_path, "w", encoding="utf-8")

    lang_counts: Counter = Counter()
    family_counts: Counter = Counter()
    skipped_unknown_lang: Counter = Counter()
    n = 0

    try:
        for sample in iter_split(split_dir):
            lang = sample.get("lang_cluster", "")
            family = LANG_FAMILY.get(lang)
            if family is None:
                skipped_unknown_lang[lang or "<MISSING>"] += 1
                continue

            problem = problems.get(sample.get("src_uid", "")) if problems else None
            record = to_sft_record(sample, problem, include_output=include_output)

            line = json.dumps(record, ensure_ascii=False) + "\n"
            all_writer.write(line)
            if write_per_family:
                family_writers[family].write(line)

            lang_counts[lang] += 1
            family_counts[family] += 1
            n += 1
            if log_every > 0 and n % log_every == 0:
                print(f"    [{split_name}] processed {n:,} samples ...", flush=True)
            if sample_limit > 0 and n >= sample_limit:
                break
    finally:
        all_writer.close()
        for w in family_writers.values():
            w.close()

    return {
        "total": n,
        "language_counts": dict(lang_counts),
        "family_counts": dict(family_counts),
        "skipped_unknown_lang": dict(skipped_unknown_lang),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apr_dir", required=True, help="xCodeEval APR root containing train/validation/test")
    parser.add_argument(
        "--problem_desc",
        default="",
        help="Optional path to problem_descriptions.jsonl for instruction enrichment",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--val_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Loading problem descriptions from {args.problem_desc} ...", flush=True)
    problems = load_problem_descriptions(args.problem_desc)
    print(f"      Loaded {len(problems)} problem descriptions", flush=True)

    report: Dict[str, Any] = {
        "language_family_map": LANG_FAMILY,
        "splits": {},
    }

    for split_name, limit in [("train", args.train_limit), ("validation", args.val_limit), ("test", args.test_limit)]:
        split_dir = os.path.join(args.apr_dir, split_name)
        if not os.path.isdir(split_dir):
            print(f"[SKIP] {split_dir} does not exist")
            continue

        include_output = split_name != "test"
        write_per_family = split_name in ("train", "validation")

        print(f"[*] Processing split: {split_name} (write_per_family={write_per_family}, limit={limit})", flush=True)
        stats = process_split(
            split_dir=split_dir,
            split_name=split_name,
            output_dir=args.output_dir,
            problems=problems,
            include_output=include_output,
            write_per_family=write_per_family,
            sample_limit=limit,
        )
        report["splits"][split_name] = stats

        print(f"    total: {stats['total']:,}", flush=True)
        for lang, c in sorted(stats["language_counts"].items(), key=lambda kv: -kv[1]):
            print(f"      {lang:12s} {c:>10,}", flush=True)
        print(f"    by family:", flush=True)
        for fam, c in stats["family_counts"].items():
            print(f"      {fam:14s} {c:>10,}", flush=True)
        if stats["skipped_unknown_lang"]:
            print(f"    skipped unknown langs: {stats['skipped_unknown_lang']}", flush=True)

        report_path = os.path.join(args.output_dir, "distribution_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"    distribution report updated -> {report_path}", flush=True)

    print(f"\nAll done. Final report: {os.path.join(args.output_dir, 'distribution_report.json')}", flush=True)


if __name__ == "__main__":
    main()
