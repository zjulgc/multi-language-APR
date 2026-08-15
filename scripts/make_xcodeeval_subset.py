"""Build a per-LANGUAGE balanced xCodeEval APR eval subset for generation + ExecEval.

Raw ``instruction_dataset/xCodeEval/apr/<split>/<Lang>.jsonl`` rows do not contain
problem statements or hidden unit tests. This script joins:

  - ``problem_descriptions.jsonl`` by ``src_uid``
  - ``unittest_db.json`` by ``src_uid``

and writes a deterministic jsonl subset with N samples PER LANGUAGE, keeping only
samples that HAVE hidden unit tests (so ExecEval can judge them). Consumed by the
generators via ``--subset_jsonl`` and scored per lang_cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from glob import glob
from typing import Any, Dict, Iterable, List

# Canonical 11 xCodeEval languages (lang_cluster spelling in the data).
LANGUAGES = [
    "C", "C++", "C#", "Java", "Kotlin",
    "Python", "Javascript", "Ruby", "PHP", "Rust", "Go",
]


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_problem_descriptions(path: str) -> Dict[str, Dict[str, Any]]:
    return {row["src_uid"]: row for row in iter_jsonl(path)}


def enrich_sample(
    sample: Dict[str, Any],
    problems: Dict[str, Dict[str, Any]],
    unittest_db: Dict[str, Any],
    require_unit_tests: bool = True,
) -> Dict[str, Any] | None:
    src_uid = sample.get("src_uid", "")
    tests = unittest_db.get(src_uid, [])
    if require_unit_tests and not tests:
        return None

    problem = problems.get(src_uid, {})
    out = dict(sample)
    out.update(
        {
            "prob_desc_description": problem.get("description", ""),
            "prob_desc_input_spec": problem.get("input_spec", ""),
            "prob_desc_output_spec": problem.get("output_spec", ""),
            "prob_desc_notes": problem.get("notes", ""),
            "prob_desc_input_from": problem.get("input_from", ""),
            "prob_desc_output_to": problem.get("output_to", ""),
            "prob_desc_time_limit": problem.get("time_limit", ""),
            "prob_desc_memory_limit": problem.get("memory_limit", ""),
            "prob_desc_sample_inputs": problem.get("sample_inputs", []),
            "prob_desc_sample_outputs": problem.get("sample_outputs", []),
            "hidden_unit_tests": json.dumps(tests, ensure_ascii=False),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apr_dir", required=True, help="xCodeEval APR root containing <split>/<Lang>.jsonl")
    ap.add_argument("--problem_desc", required=True)
    ap.add_argument("--unittest_db", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument(
        "--samples_per_language",
        type=int,
        default=100,
        help="Samples per language (only samples with hidden unit tests). -1 = all eligible.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_empty_unit_tests", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    problems = load_problem_descriptions(args.problem_desc)
    with open(args.unittest_db, "r", encoding="utf-8") as f:
        unittest_db = json.load(f)

    by_lang: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_no_tests = 0
    skipped_unknown_lang = Counter()

    for path in sorted(glob(os.path.join(args.apr_dir, args.split, "*.jsonl"))):
        for sample in iter_jsonl(path):
            lang = sample.get("lang_cluster", "")
            if lang not in LANGUAGES:
                skipped_unknown_lang[lang or "<MISSING>"] += 1
                continue
            enriched = enrich_sample(
                sample,
                problems=problems,
                unittest_db=unittest_db,
                require_unit_tests=not args.allow_empty_unit_tests,
            )
            if enriched is None:
                skipped_no_tests += 1
                continue
            by_lang[lang].append(enriched)

    selected: List[Dict[str, Any]] = []
    short: Dict[str, int] = {}
    for lang in LANGUAGES:
        pool = by_lang.get(lang, [])
        if args.samples_per_language < 0 or len(pool) <= args.samples_per_language:
            take = list(pool)
            if 0 <= args.samples_per_language > len(pool):
                short[lang] = len(pool)  # fewer eligible than requested
        else:
            take = rng.sample(pool, args.samples_per_language)
        selected.extend(take)

    rng.shuffle(selected)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lang_counts = Counter(row.get("lang_cluster", "") for row in selected)
    eligible = {lang: len(by_lang.get(lang, [])) for lang in LANGUAGES}
    print(f"wrote {len(selected)} samples -> {args.output_jsonl}")
    print(f"per-language selected: {dict(sorted(lang_counts.items()))}")
    print(f"eligible-with-tests per language: {eligible}")
    print(f"skipped_no_tests={skipped_no_tests}")
    if short:
        print(f"!! SHORT (fewer eligible than --samples_per_language): {short}")
    if skipped_unknown_lang:
        print(f"skipped_unknown_lang={dict(skipped_unknown_lang)}")


if __name__ == "__main__":
    main()
