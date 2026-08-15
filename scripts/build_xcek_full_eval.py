"""Build the FULL xCodeEval APR test evaluation file (17699 rows).

The raw apr/test/*.jsonl rows carry only bug fields; every prob_desc_* field
is None because xCodeEval ships problem statements separately in
problem_descriptions.jsonl (keyed by src_uid). Generating from the raw rows
produces "blind repair" prompts (Problem Description: None) — this invalidated
the 2026-08-11 first full-set run. This script performs the same join the
100-per-lang subset file was built with, so full-set prompts are byte-identical
to the subset pipeline for overlapping items.
"""
import json
import os
from glob import glob

ROOT = "instruction_dataset/xCodeEval"
OUT = "data/eval/xcodeeval_test_full.jsonl"

DESC_FIELD_MAP = {
    "prob_desc_description": "description",
    "prob_desc_input_spec": "input_spec",
    "prob_desc_output_spec": "output_spec",
    "prob_desc_sample_inputs": "sample_inputs",
    "prob_desc_sample_outputs": "sample_outputs",
    "prob_desc_notes": "notes",
    "prob_desc_input_from": "input_from",
    "prob_desc_output_to": "output_to",
    "prob_desc_time_limit": "time_limit",
    "prob_desc_memory_limit": "memory_limit",
}


def main() -> None:
    desc_by_uid = {}
    with open(os.path.join(ROOT, "problem_descriptions.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            desc_by_uid[d["src_uid"]] = d

    n_rows = 0
    n_missing_desc = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for path in sorted(glob(os.path.join(ROOT, "apr", "test", "*.jsonl"))):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    desc = desc_by_uid.get(row["src_uid"])
                    if desc is None:
                        n_missing_desc += 1
                    else:
                        for dst, src in DESC_FIELD_MAP.items():
                            if row.get(dst) is None:
                                row[dst] = desc.get(src)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_rows += 1
    print(f"wrote {n_rows} rows -> {OUT}; rows without description: {n_missing_desc}")


if __name__ == "__main__":
    main()
