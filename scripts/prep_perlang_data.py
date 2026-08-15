"""Per-language (grouping-free) training data: sample N rows per programming
language and drop the 4-family concept entirely.

Motivation: the by_family split (c_family/jvm_family/dynamic_typed/systems) both
(a) rests on a debatable grouping (C# with C/C++, Go with Rust) and (b) hides
extreme intra-family imbalance (JS gets only 349 rows inside dynamic_typed's 15K).
This builds a per-LANGUAGE balanced set: exactly N rows for each of the 11
xCodeEval languages, so every language is a first-class routing target.

All 11 languages have >= N=3000 available (smallest = PHP 3,595), so N=3000 is
exactly balanced across languages with no upsampling.

Layout (per-language dirs, each holding ONE language; the stock
LanguageDataPaths / load_language_dataset / BalancedLanguageSampler consume this
and set the `language` column == language name):

  data/variants/perlang<N>k/by_language/<Lang>/train_sft.jsonl   (N rows each)
  data/variants/perlang<N>k/distribution_report.json

Validation is NOT carved here (PHP only has 3,595 total, so 3K train + 500 val
would not fit for every language); training reuses the shared
data/all/intrain_validation_sft.jsonl via --val_file.

Run:
  python scripts/prep_perlang_data.py --n 3000 --seed 0
"""
import argparse
import json
import os
import random

ROOT = "/mnt/backup1/lgc/multi-language-APR"

# Canonical 11 languages, spelled exactly as lang_cluster appears in the data.
# Order matches moe_apr.data_utils.LANGS_CANONICAL (expert index 0..10).
LANGS = [
    "C", "C++", "C#", "Java", "Kotlin",
    "Python", "Javascript", "Ruby", "PHP", "Rust", "Go",
]


def reservoir_sample_all_langs(src_path, langs, k, seed=0):
    """Single streaming pass over src_path, keeping a size-k uniform reservoir
    per language (statistically equivalent to shuffle+head, but O(1) memory per
    language). Returns (rows_per_lang, avail_per_lang)."""
    rngs = {lang: random.Random(seed + i) for i, lang in enumerate(langs)}
    res = {lang: [] for lang in langs}
    seen = {lang: 0 for lang in langs}
    wanted = set(langs)
    with open(src_path) as f:
        for line in f:
            try:
                lang = json.loads(line).get("lang_cluster")
            except Exception:
                continue
            if lang not in wanted:
                continue
            seen[lang] += 1
            pool = res[lang]
            if len(pool) < k:
                pool.append(line)
            else:
                j = rngs[lang].randrange(seen[lang])
                if j < k:
                    pool[j] = line
    for lang in langs:
        rngs[lang].shuffle(res[lang])
    return res, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="rows per language")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=os.path.join(ROOT, "data/all/train_sft.jsonl"))
    ap.add_argument("--out_root", default=None,
                    help="default: data/variants/perlang<N>k")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(ROOT, f"data/variants/perlang{args.n // 1000}k")
    print(f"[prep] source = {args.src}")
    print(f"[prep] out    = {out_root}")
    print(f"[prep] N={args.n} per language, seed={args.seed}")

    rows, avail = reservoir_sample_all_langs(args.src, LANGS, args.n, seed=args.seed)

    report = {"n_per_language": args.n, "seed": args.seed, "source": args.src,
              "languages": {}, "total_written": 0}
    for lang in LANGS:
        got = rows[lang]
        if len(got) < args.n:
            print(f"  !! {lang}: only {len(got)} available (< {args.n}); wrote all")
        d = os.path.join(out_root, "by_language", lang)
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "train_sft.jsonl")
        with open(out_path, "w") as w:
            for line in got:
                w.write(line if line.endswith("\n") else line + "\n")
        report["languages"][lang] = {"available": avail[lang], "written": len(got)}
        report["total_written"] += len(got)
        print(f"  {lang:12s} avail={avail[lang]:>9,}  wrote={len(got):>5,}")

    # Also emit a single merged + shuffled file for the dense (vanilla LoRA)
    # baseline, which loads one --train_file instead of per-language dirs. Same
    # 33K rows, so all three trainers (MoE variants + vanilla) see identical data.
    merged = [ln for lang in LANGS for ln in rows[lang]]
    random.Random(args.seed + 999).shuffle(merged)
    all_dir = os.path.join(out_root, "all")
    os.makedirs(all_dir, exist_ok=True)
    merged_path = os.path.join(all_dir, "train_sft.jsonl")
    with open(merged_path, "w") as w:
        for line in merged:
            w.write(line if line.endswith("\n") else line + "\n")
    report["merged_train"] = {"path": merged_path, "rows": len(merged)}
    print(f"[prep] merged (shuffled) train -> {merged_path} ({len(merged):,} rows)")

    with open(os.path.join(out_root, "distribution_report.json"), "w") as w:
        json.dump(report, w, indent=2)
    print(f"[prep] total written = {report['total_written']:,} rows "
          f"({len(LANGS)} languages x {args.n})")
    print(f"[prep] report -> {os.path.join(out_root, 'distribution_report.json')}")


if __name__ == "__main__":
    main()
