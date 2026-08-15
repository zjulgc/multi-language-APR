"""xCodeEval APR pass@k integration via ExecEval (Docker server).

Reads a directory of generated JSON files (the format produced by
``generate_moe_apr.py`` / ``generate_apr_local.py``: one file per
sample, with ``oai_response.choices[*].message.content`` containing
candidate fixes and ``source_data`` containing the xCodeEval sample
metadata + ``hidden_unit_tests``), POSTs each candidate to the
ExecEval HTTP server (default ``http://localhost:5000``), and reports
overall + per-language pass@1 and pass@10.

Compared to the upstream ``instruction_dataset/xCodeEval_repo/evaluation/
apr/eval_apr.py`` reference we:

  - take the input dir as a CLI flag (no DUMP_FOLDER env var convention)
  - call execeval directly from Python (no jsonlines dump intermediate)
  - aggregate pass@k per language and overall in a single JSON output
  - support arbitrary ``--k_values`` (default 1, 10)
  - parallelise per-sample requests with a thread pool


Example::

    python eval_xcodeeval_execeval.py \\
        --gen_dir dumped/oai/stage1/apr_n_sample_10 \\
        --output_json dumped/oai/stage1/execeval_passk.json \\
        --k_values 1 5 10 \\
        --execeval_url http://localhost:5000 \\
        --max_workers 32

The pass@k aggregator uses the unbiased estimator from
``HumanEval`` (Chen et al. 2021):

    pass@k = 1 - C(n - c, k) / C(n, k)

where ``n`` is the number of generated samples per task and ``c`` the
number of correct samples. We also report a simpler ``has_correct@k``
which is just whether any of the first k samples pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


LANG_CLUSTER_TO_LANG_COMPILER = {
    "C": "GNU C11",
    "C#": "Mono C#",
    "C++": "GNU C++17",
    "Go": "Go",
    "Java": "Java 17",
    "Javascript": "Node.js",
    "Kotlin": "Kotlin 1.4",
    "PHP": "PHP",
    "Python": "PyPy 3",
    "Ruby": "Ruby 3",
    "Rust": "Rust 2018",
}


# Recognised opening-fence language tags (lower-cased). An empty first line
# (bare ```\n) also counts as "just a fence, no tag".
_LANG_TAGS = {
    "", "python", "python3", "py", "javascript", "js", "node", "nodejs",
    "typescript", "ts", "java", "kotlin", "kt", "cpp", "c++", "cxx", "cc",
    "c", "csharp", "cs", "c#", "go", "golang", "rust", "rs", "ruby", "rb",
    "php", "scala", "swift",
}


def sanitize_code(code: str) -> str:
    """Extract the first fenced code block from a model response.

    Robust to (a) a prose preamble before the fence ("Here is the fixed
    code:\\n\\n```..."), (b) an optional language tag on the opening fence,
    (c) a missing closing fence (truncated generations), and (d) no fence at
    all (returned as-is). The earlier version only stripped leading/trailing
    ``` markers *when the string began with a fence*, so any model that
    prefaces its code block (e.g. Llama-3 base: "Here is the fixed code:")
    left the prose in place -> COMPILATION_ERROR on every sample.
    """
    if not code:
        return ""
    idx = code.find("```")
    if idx == -1:
        return code.strip()
    rest = code[idx + 3:]
    nl = rest.find("\n")
    first_line = (rest[:nl] if nl != -1 else rest).strip().lower()
    if first_line in _LANG_TAGS:
        rest = rest[nl + 1:] if nl != -1 else ""
    close = rest.find("```")
    if close != -1:
        rest = rest[:close]
    return rest.strip()


def fix_uts(uts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"input": ut.get("input", ""), "output": ut.get("output", [])} for ut in uts]


# ----------------------------- ExecEval client ----------------------------- #


class ExecEvalClient:
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = timeout

    def execute(self, language: str, source_code: str, unittests: List[Dict[str, Any]],
                limits: Optional[Dict[str, Any]] = None,
                stop_on_first_fail: bool = True) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        body = {
            "language": language,
            "source_code": source_code,
            "unittests": unittests,
            "limits": limits if isinstance(limits, dict) else None,
            "compile_cmd": None,
            "compile_flags": None,
            "execute_cmd": None,
            "execute_flags": None,
            "block_network": True,
            "stop_on_first_fail": stop_on_first_fail,
            "use_sanitizer": False,
        }
        try:
            r = self.session.post(
                f"{self.base_url}/api/execute_code",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except Exception as e:
            return None, {"transport_error": str(e)}
        try:
            payload = r.json()
        except Exception as e:
            return None, {"json_error": str(e), "text": r.text[:500]}
        if "data" not in payload:
            return None, {"server_error": payload}
        return payload["data"], None


# ----------------------------- pass@k computation -------------------------- #


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., HumanEval)."""
    if n - c < k:
        return 1.0
    if k > n:
        return 0.0
    # Computed in log-space-friendly form to avoid overflow.
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def is_passed(unit_test_results: List[Dict[str, Any]]) -> bool:
    """A run is considered passing iff every unit test has exec_outcome == PASSED."""
    if not unit_test_results:
        return False
    for ut in unit_test_results:
        outcome = ut.get("exec_outcome")
        if outcome != "PASSED":
            return False
    return True


# ----------------------------- worker -------------------------------------- #


def _load_unit_tests(src: Dict[str, Any], unittest_db: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    raw_uts = src.get("hidden_unit_tests")
    if isinstance(raw_uts, str):
        try:
            return json.loads(raw_uts), None
        except Exception as e:
            return [], f"hidden_unit_tests parse: {e}"
    if raw_uts:
        return raw_uts, None

    src_uid = src.get("src_uid")
    if unittest_db is not None and src_uid:
        return unittest_db.get(src_uid, []) or [], None
    return [], None


def evaluate_sample_file(path: str, client: ExecEvalClient,
                         max_choices_per_sample: int = -1,
                         unittest_db: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            sample = json.load(f)
    except Exception as e:
        return {"path": path, "error": f"load: {e}"}

    src = sample.get("source_data") or {}
    lang_cluster = src.get("lang_cluster")
    compiler = LANG_CLUSTER_TO_LANG_COMPILER.get(lang_cluster)
    if compiler is None:
        return {"path": path, "skip": True, "reason": f"unsupported lang_cluster: {lang_cluster}"}

    uts, uts_error = _load_unit_tests(src, unittest_db=unittest_db)
    if uts_error:
        return {"path": path, "error": uts_error}
    uts = fix_uts(uts)
    if not uts:
        return {"path": path, "skip": True, "reason": "no unit tests"}

    src_uid = src.get("src_uid", os.path.basename(path))
    choices = (sample.get("oai_response") or {}).get("choices") or []
    if max_choices_per_sample > 0:
        choices = choices[:max_choices_per_sample]
    if not choices:
        return {"path": path, "skip": True, "reason": "no choices"}

    per_choice = []
    for ch in choices:
        code = (ch.get("message") or {}).get("content") or ch.get("text") or ""
        code = sanitize_code(code)
        if not code:
            per_choice.append({"passed": False, "reason": "empty after sanitize"})
            continue
        results, err = client.execute(compiler, code, uts, stop_on_first_fail=True)
        if err:
            per_choice.append({"passed": False, "error": err})
            continue
        per_choice.append({"passed": is_passed(results), "results": results})

    return {
        "path": path,
        "src_uid": src_uid,
        "lang_cluster": lang_cluster,
        "compiler": compiler,
        "n_choices": len(per_choice),
        "n_correct": sum(1 for c in per_choice if c.get("passed")),
        "per_choice": per_choice,
    }


# ----------------------------- main ---------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True,
                    help="Directory of generated *.json files (output of generate_moe_apr.py).")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--execeval_url", default="http://localhost:5000")
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--max_workers", type=int, default=32)
    ap.add_argument("--max_choices_per_sample", type=int, default=-1)
    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--lang_cluster_filter", nargs="*", default=None,
                    help="If set, only evaluate samples whose lang_cluster is in this list.")
    ap.add_argument("--include_per_choice", action="store_true",
                    help="Dump per-choice exec_outcome details (large).")
    ap.add_argument(
        "--unittest_db",
        default="",
        help=(
            "Optional xCodeEval unittest_db.json. Used as a fallback when "
            "source_data.hidden_unit_tests is absent in generated JSON."
        ),
    )
    args = ap.parse_args()

    unittest_db = None
    if args.unittest_db:
        with open(args.unittest_db, "r", encoding="utf-8") as f:
            unittest_db = json.load(f)
        print(f"[unittest_db] loaded {len(unittest_db):,} src_uid entries from {args.unittest_db}", flush=True)

    client = ExecEvalClient(args.execeval_url)
    try:
        runtimes = client.session.get(f"{args.execeval_url}/api/all_runtimes", timeout=10).json()
        print(f"[execeval] available runtimes: {len(runtimes) if isinstance(runtimes, list) else 'unknown'}", flush=True)
    except Exception as e:
        print(f"[execeval] WARN unable to reach {args.execeval_url}: {e}", flush=True)

    files = sorted(glob(os.path.join(args.gen_dir, "*.json")))
    if args.max_files > 0:
        files = files[: args.max_files]
    if args.lang_cluster_filter:
        # Quick filter by filename suffix (the generator names files {idx}_{temp}_{lang}.json).
        files = [f for f in files if any(lc in os.path.basename(f) for lc in args.lang_cluster_filter)]
    print(f"[gen] evaluating {len(files)} sample files", flush=True)

    t0 = time.time()
    per_sample: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(evaluate_sample_file, f, client, args.max_choices_per_sample, unittest_db): f
            for f in files
        }
        done = 0
        for fut in cf.as_completed(futures):
            res = fut.result()
            if res is not None:
                per_sample.append(res)
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"[gen] {done}/{len(files)}  elapsed={time.time()-t0:.0f}s", flush=True)

    # ----- Aggregate -----
    valid = [r for r in per_sample if r and not r.get("skip") and not r.get("error") and r.get("n_choices", 0) > 0]
    skipped = [r for r in per_sample if r.get("skip")]
    errored = [r for r in per_sample if r.get("error")]

    by_lang: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n_tasks": 0, "samples": []})
    for r in valid:
        lc = r["lang_cluster"]
        by_lang[lc]["n_tasks"] += 1
        by_lang[lc]["samples"].append((r["n_choices"], r["n_correct"]))

    summary_passk: Dict[str, Dict[str, float]] = {}
    overall_acc: Dict[str, List[float]] = {f"pass@{k}": [] for k in args.k_values}
    overall_has: Dict[str, List[float]] = {f"has_correct@{k}": [] for k in args.k_values}

    for lc, info in by_lang.items():
        per_k_pass: Dict[str, List[float]] = {f"pass@{k}": [] for k in args.k_values}
        per_k_has: Dict[str, List[float]] = {f"has_correct@{k}": [] for k in args.k_values}
        for n, c in info["samples"]:
            for k in args.k_values:
                per_k_pass[f"pass@{k}"].append(pass_at_k(n, c, k))
                per_k_has[f"has_correct@{k}"].append(1.0 if c >= 1 and k >= 1 and n >= 1 and (c >= 1 if k <= n else c >= 1) else 0.0)
                # Simpler: has_correct@k = whether at least one of the FIRST k samples passes.
                # We don't have ordering info, so fall back to "any correct in n" if k>=n.
        summary_passk[lc] = {
            **{k: sum(v) / len(v) for k, v in per_k_pass.items()},
            "n_tasks": info["n_tasks"],
        }
        for k in args.k_values:
            overall_acc[f"pass@{k}"].extend(per_k_pass[f"pass@{k}"])
            overall_has[f"has_correct@{k}"].extend(per_k_has[f"has_correct@{k}"])

    overall = {
        **{k: (sum(v) / len(v) if v else 0.0) for k, v in overall_acc.items()},
        "n_tasks": sum(info["n_tasks"] for info in by_lang.values()),
    }

    summary = {
        "gen_dir": args.gen_dir,
        "execeval_url": args.execeval_url,
        "k_values": args.k_values,
        "n_files_total": len(files),
        "n_valid": len(valid),
        "n_skipped": len(skipped),
        "n_errored": len(errored),
        "overall_pass@k": overall,
        "per_language_pass@k": summary_passk,
        "elapsed_seconds": time.time() - t0,
    }
    if args.include_per_choice:
        summary["per_sample"] = per_sample

    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2, ensure_ascii=False), flush=True)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[gen] saved -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
