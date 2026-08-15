"""HumanEvalFix evaluation for the MoE-LoRA APR model.

Loads the BigCode HumanEvalPack ``humanevalfix*`` configs (6 languages:
python, js, java, go, cpp, rust), generates a fix for each buggy_solution
using the same chat template used during training, executes the
hidden test suite in a per-language subprocess, and reports pass@1
overall and per language.

Supports three model modes mirroring quick_eval.py:

    --mode base                            (zero-shot Qwen)
    --mode peft  --adapter <dir>           (vanilla PEFT LoRA)
    --mode moe   --moe_state <path>        (our MoE-LoRA + Shared)
                  [--patch_config <json>]  (auto-derived if next to moe_state)

The execution sandbox is a plain ``subprocess.run`` with timeout. We do NOT
use Docker here -- the language binaries (python3, node, java, go, g++,
rustc) must be available in PATH. For untrusted code in production,
prefer the ExecEval Docker server; this script is for local dev signal.

Dataset reference:
  https://huggingface.co/datasets/bigcode/humanevalpack
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import icl_retrieval
from prompt_utils import cot_instruction, render_chat_prompt, render_icl_prompt

# HEF language keys -> xcek lang_cluster names used by the ICL exemplar bank
HEF_TO_XCEK_LANG = {"python": "Python", "js": "Javascript", "java": "Java",
                    "cpp": "C++", "rust": "Rust", "go": "Go"}
ICL_BANK = None  # loaded in main() when --prompt_style icl
from train_moe_apr import load_moe_state
from moe_apr.model_patcher import (
    MoEPatchConfig,
    patch_model_with_moe_lora,
)
from moe_apr.moe_layer import get_moe_ablation

LANGS = ("python", "js", "java", "go", "cpp", "rust")

# ----- per-language compile + run drivers ---------------------------------- #

# Each driver returns ``(passed: bool, log: str)``.

DEFAULT_TIMEOUT = 30  # seconds per sample compile+run combined


def _run(cmd: List[str], cwd: str, stdin: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", (e.stderr or "") + "\n[TIMEOUT]"
    except FileNotFoundError as e:
        return 127, "", f"[NOT FOUND] {e}"


def _python_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "solution.py"
        src.write_text(fix_code + "\n\n" + test_code + "\ncheck(" + _python_entry(test_code) + ")\n")
        rc, out, err = _run([sys.executable, str(src)], cwd=td)
        return rc == 0, (out + err)[:4000]


def _python_entry(test_code: str) -> str:
    # HumanEvalPack tests define ``def check(candidate):``. Try to extract the
    # candidate name from the canonical signature in the prompt; fall back to
    # the most-recently-defined top-level function in ``fix_code`` (handled
    # below where we wrap).
    m = re.search(r"check\(([A-Za-z_][A-Za-z0-9_]*)\)", test_code)
    if m:
        return m.group(1)
    return "candidate"  # not used because we override below


def _js_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "solution.js"
        assert_override = (
            "console.assert = (cond, ...args) => { "
            "if (!cond) { throw new Error(args.join(' ') || 'Assertion failed'); } "
            "};\n"
        )
        src.write_text(assert_override + fix_code + "\n\n" + test_code + "\n")
        rc, out, err = _run(["node", str(src)], cwd=td)
        return rc == 0, (out + err)[:4000]


def _java_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        # HumanEvalPack java samples expect the class to be ``Solution`` or
        # similar; the test uses ``Main.main``. We place fix and test in two
        # files but keep a single Main if present.
        full = fix_code + "\n\n" + test_code + "\n"
        # Try to extract first public class name; default to Main.
        m = re.search(r"public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", full)
        cls = m.group(1) if m else "Main"
        src = Path(td) / f"{cls}.java"
        src.write_text(full)
        rc, out, err = _run(["javac", "--release", "11", str(src)], cwd=td, timeout=DEFAULT_TIMEOUT)
        if rc != 0:
            return False, ("[compile]\n" + out + err)[:4000]
        rc, out, err = _run(["java", "-cp", td, cls], cwd=td, timeout=DEFAULT_TIMEOUT)
        return rc == 0, (out + err)[:4000]


# HumanEvalPack Go tests import github.com/stretchr/testify/assert, which isn't
# fetchable offline (GOPROXY blocked). The tests only ever call assert.New +
# assert.Equal, so we vendor a ~20-line local shim (faithful to testify's
# ObjectsAreEqual) and point the module at it via a go.mod ``replace``. Imports
# are recomputed from actual usage because the dataset's own import blocks are
# inconsistent (duplicate / unused / missing across samples).
_GO_TESTIFY_SHIM = '''package assert

import (
    "bytes"
    "reflect"
    "testing"
)

type Assertions struct{ t *testing.T }

func New(t *testing.T) *Assertions { return &Assertions{t} }

func ObjectsAreEqual(e, a interface{}) bool {
    if e == nil || a == nil {
        return e == a
    }
    if x, ok := e.([]byte); ok {
        y, ok := a.([]byte)
        if !ok {
            return false
        }
        if x == nil || y == nil {
            return x == nil && y == nil
        }
        return bytes.Equal(x, y)
    }
    return reflect.DeepEqual(e, a)
}

func (s *Assertions) Equal(e, a interface{}, m ...interface{}) bool {
    if ObjectsAreEqual(e, a) {
        return true
    }
    s.t.Errorf("Not equal: expected=%#v actual=%#v", e, a)
    return false
}
'''

_GO_STD = {
    "math": "math", "sort": "sort", "strings": "strings", "strconv": "strconv",
    "fmt": "fmt", "unicode": "unicode", "rand": "math/rand", "time": "time",
    "regexp": "regexp", "bytes": "bytes", "errors": "errors", "os": "os",
    "utf8": "unicode/utf8", "md5": "crypto/md5", "bufio": "bufio", "io": "io",
    "reflect": "reflect", "bits": "math/bits",
}


def _go_imports(code: str) -> str:
    """Emit an import block for exactly the stdlib packages the code references
    (plus testing + the testify shim), sidestepping the dataset's inconsistent
    import blocks."""
    used = set(re.findall(r"\b([a-z][a-z0-9]*)\.", code))
    paths = {"testing", "github.com/stretchr/testify/assert"}
    for tok, path in _GO_STD.items():
        if tok in used:
            paths.add(path)
    return "import (\n" + "\n".join(f'    "{p}"' for p in sorted(paths)) + "\n)\n"


def _go_strip_scaffold(code: str) -> str:
    """Drop package decl + import blocks the model emitted. Instruct/xCodeEval
    models return a whole Go file (``package main`` + imports, sometimes a
    ``func main``); we re-emit package + imports ourselves, so keeping the
    model's would duplicate ``package main`` ("expected declaration, found
    'package'") and re-add unused imports."""
    code = re.sub(r"(?m)^\s*package\s+\w+\s*$", "", code)
    code = re.sub(r"(?ms)^\s*import\s*\(.*?\)\s*$", "", code)
    code = re.sub(r'(?m)^\s*import\s+"[^"]+"\s*$', "", code)
    return code


def _go_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        shim = Path(td) / "shim" / "testify" / "assert"
        shim.mkdir(parents=True)
        (Path(td) / "go.mod").write_text(
            "module humanevalfix\ngo 1.20\n"
            "require github.com/stretchr/testify v1.8.4\n"
            "replace github.com/stretchr/testify => ./shim/testify\n")
        (Path(td) / "shim" / "testify" / "go.mod").write_text(
            "module github.com/stretchr/testify\ngo 1.20\n")
        (shim / "assert.go").write_text(_GO_TESTIFY_SHIM)
        body = _go_strip_scaffold(fix_code) + "\n\n" + _go_strip_scaffold(test_code) + "\n"
        (Path(td) / "solution_test.go").write_text(
            "package main\n\n" + _go_imports(body) + "\n" + body)
        env = dict(os.environ)
        env["GOFLAGS"] = "-mod=mod"
        env["GOPROXY"] = "off"
        env["GOSUMDB"] = "off"
        env.setdefault("GOCACHE", "/tmp/lgc-gocache")
        env.setdefault("GOPATH", "/mnt/backup1/lgc/gopath")
        env["PATH"] = "/mnt/backup1/lgc/goroot/bin:" + env.get("PATH", "")
        try:
            proc = subprocess.run(["go", "test", "./..."], cwd=td, capture_output=True,
                                  text=True, timeout=90, env=env)
        except subprocess.TimeoutExpired as e:
            return False, ((e.stdout or "") + (e.stderr or "") + "\n[TIMEOUT]")[:4000]
        return proc.returncode == 0, (proc.stdout + proc.stderr)[:4000]


def _cpp_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "solution.cpp"
        src.write_text(fix_code + "\n\n" + test_code + "\n")
        bin_path = Path(td) / "solution.out"
        rc, out, err = _run(["g++", "-O2", "-std=c++17", "-o", str(bin_path), str(src)], cwd=td)
        if rc != 0:
            return False, ("[compile]\n" + out + err)[:4000]
        rc, out, err = _run([str(bin_path)], cwd=td)
        return rc == 0, (out + err)[:4000]


def _rust_runner(fix_code: str, test_code: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "solution.rs"
        full = re.sub(r"^\s*use\s+(rand|regex|md5)\b.*;\s*$", "", fix_code + "\n\n" + test_code + "\n", flags=re.MULTILINE)
        src.write_text(full)
        bin_path = Path(td) / "solution.out"
        rc, out, err = _run(["rustc", "--edition=2018", "--test", "-O", "-o", str(bin_path), str(src)], cwd=td, timeout=60)
        if rc != 0:
            return False, ("[compile]\n" + out + err)[:4000]
        rc, out, err = _run([str(bin_path)], cwd=td)
        return rc == 0, (out + err)[:4000]


RUNNERS = {
    "python": _python_runner,
    "js": _js_runner,
    "java": _java_runner,
    "go": _go_runner,
    "cpp": _cpp_runner,
    "rust": _rust_runner,
}


# --------------------------- Prompt + extraction --------------------------- #


def make_instruction(language: str) -> str:
    pretty = {"python": "Python", "js": "JavaScript", "java": "Java",
              "go": "Go", "cpp": "C++", "rust": "Rust"}.get(language, language)
    return f"Fix the buggy {pretty} code so that all unit tests pass."


CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z+\-]*)\n(.*?)```", re.DOTALL)


def extract_fix(generated: str) -> str:
    """Extract code from a model generation. Strip code fences if present."""
    m = CODE_FENCE_RE.search(generated)
    if m:
        return m.group(1).strip()
    # Stop at next "@@" in case the model echoed the template.
    cut = generated.split("\n@@")[0]
    return cut.strip()


def compose_candidate(row: Dict[str, Any], lang: str, fix: str) -> str:
    """Compose HumanEvalPack declaration + generated body when needed.

    HumanEvalPack ``buggy_solution``/``canonical_solution`` are commonly only
    function-body fragments for Java/C++/Rust/JS/Python. The APR model is
    prompted with that same fragment, so generations often need the dataset
    ``declaration`` prepended before local execution.
    """
    declaration = (row.get("declaration") or "").rstrip()
    if not declaration:
        return fix

    stripped = fix.lstrip("\n")
    entry = row.get("entry_point") or ""
    signature = row.get("signature") or ""

    if lang == "python":
        if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", stripped, re.MULTILINE):
            return fix
        return declaration + "\n" + stripped

    if lang == "js":
        has_function = (
            re.search(rf"\bfunction\s+{re.escape(entry)}\s*\(", stripped)
            or re.search(rf"\b(const|let|var)\s+{re.escape(entry)}\b", stripped)
        )
        return fix if has_function else declaration + "\n" + stripped

    if lang == "java":
        if re.search(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*", stripped):
            return fix
        if signature and signature in stripped:
            imports = declaration.split("class Solution", 1)[0].rstrip()
            return imports + "\nclass Solution {\n    " + stripped
        return declaration + "\n" + stripped

    if lang == "cpp":
        if signature and signature in stripped:
            return fix
        return declaration + "\n" + stripped

    if lang == "rust":
        if entry and re.search(rf"\bfn\s+{re.escape(entry)}\s*\(", stripped):
            return fix
        return declaration + "\n" + stripped

    if lang == "go":
        # Go entry_point is CamelCase; take the func name from the declaration.
        gm = re.search(r"func\s+([A-Za-z_]\w*)\s*\(", declaration)
        fname = gm.group(1) if gm else entry
        if fname and re.search(rf"\bfunc\s+{re.escape(fname)}\s*\(", stripped):
            return fix
        return declaration + "\n" + stripped

    return fix


# ------------------------------- Model loading ----------------------------- #


def load_model(args) -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    print("[load] base model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if args.mode == "peft":
        from peft import PeftModel
        print(f"[load] PEFT adapter {args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.bfloat16)
    elif args.mode == "moe":
        cfg_path = args.patch_config or os.path.join(os.path.dirname(args.moe_state), "patch_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        patch_cfg = MoEPatchConfig.from_dict(cfg)
        replaced = patch_model_with_moe_lora(model, patch_cfg)
        print(f"[load] patched {len(replaced)} modules", flush=True)
        load_moe_state(model, args.moe_state)
        ab_mode, ab_norm = get_moe_ablation()
        print(f"[load] branch ablation: MOE_ABLATE={ab_mode} MOE_ABLATE_NORM={ab_norm}", flush=True)
    elif args.mode == "base":
        pass
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    model.eval()
    model.config.use_cache = True
    return model, tok


# ------------------------------- Generation -------------------------------- #


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021 (Codex), eq. 1.

    n = candidates generated, c = candidates that pass. Returns the probability
    that at least one of k candidates drawn without replacement passes.
    """
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k), computed stably as a running product.
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


@torch.no_grad()
def generate_fix(
    model, tok, instruction: str, buggy_code: str,
    max_input_len: int = 1536, max_new_tokens: int = 512,
    num_samples: int = 1, temperature: float = 0.0, top_p: float = 0.95,
    exemplars=None,
) -> List[str]:
    """Return ``num_samples`` candidate fixes.

    Greedy (the historical behaviour, and what every pass@1 number in the paper
    was produced with) when num_samples == 1 and temperature == 0.
    """
    if exemplars:
        prompt = render_icl_prompt(tok, instruction, buggy_code, exemplars)
    else:
        prompt = render_chat_prompt(tok, instruction, buggy_code)
    enc = tok(prompt, truncation=True, max_length=max_input_len, return_tensors="pt").to(model.device)
    do_sample = num_samples > 1 or temperature > 0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        num_return_sequences=num_samples,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**enc, **gen_kwargs)
    prompt_len = enc.input_ids.shape[1]
    return [tok.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]


# ------------------------------- Main eval --------------------------------- #


def _evaluate_language_batched(model, tok, lang: str, ds, args, runner, k_values) -> Dict[str, Any]:
    """Fast path for greedy n=1: batched generation + threaded test execution.

    Bookkeeping (per_sample order, pass@1 = n_pass/n) matches the sequential
    path exactly; greedy-in-batch left-padding jitter is the same order as the
    documented xcek batch jitter (~1%).
    """
    rows = list(ds)
    n_total = len(rows)

    prompts: List[str] = []
    for row in rows:
        instruction = row.get("instruction") or make_instruction(lang)
        buggy_for_prompt = compose_candidate(row, lang, row["buggy_solution"])
        exemplars = None
        if args.prompt_style == "cot":
            instruction = cot_instruction(instruction)
        elif args.prompt_style == "icl":
            exemplars = icl_retrieval.retrieve(
                ICL_BANK, HEF_TO_XCEK_LANG[lang], buggy_for_prompt, args.icl_shots)
        if exemplars:
            prompts.append(render_icl_prompt(tok, instruction, buggy_for_prompt, exemplars))
        else:
            prompts.append(render_chat_prompt(tok, instruction, buggy_for_prompt))

    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    gens: List[str] = []
    bs = args.gen_batch_size
    for s in range(0, n_total, bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_input_len).to(model.device)
        out = model.generate(**enc, do_sample=False,
                             max_new_tokens=args.max_new_tokens,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
        plen = enc.input_ids.shape[1]
        gens.extend(tok.decode(o[plen:], skip_special_tokens=True) for o in out)
        if (s // bs) % 4 == 0 or s + bs >= n_total:
            print(f"[hef][{lang}] gen {min(s + bs, n_total)}/{n_total}", flush=True)
    tok.padding_side = old_side

    def _judge(idx: int):
        fix = compose_candidate(rows[idx], lang, extract_fix(gens[idx]))
        try:
            ok, log = runner(fix, rows[idx]["test"])
        except Exception as e:
            ok, log = False, f"[runner-error] {e}"
        return idx, ok, log

    verdicts: List[Any] = [None] * n_total
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.test_workers) as pool:
        for idx, ok, log in pool.map(_judge, range(n_total)):
            verdicts[idx] = (ok, log)

    n_pass = 0
    per_sample: List[Dict[str, Any]] = []
    passk_acc: Dict[int, List[float]] = {k: [] for k in k_values}
    for i, row in enumerate(rows):
        ok, log = verdicts[i]
        n_pass += int(ok)
        for k in k_values:
            passk_acc[k].append(pass_at_k(1, int(ok), k))
        per_sample.append({
            "task_id": row.get("task_id", f"{lang}/{i}"),
            "pass": ok,
            "n_choices": 1,
            "n_correct": int(ok),
            "log_excerpt": log[:300] if not ok else "",
        })
    print(f"[hef][{lang}] {n_total}/{n_total}  pass@1={n_pass / max(1, n_total):.3f}", flush=True)

    out: Dict[str, Any] = {
        "pass@1": n_pass / max(1, n_total),
        "n_pass": n_pass,
        "n": n_total,
        "n_samples": 1,
        "per_sample": per_sample if args.dump_per_sample else None,
    }
    for k in k_values:
        out[f"pass@{k}"] = sum(passk_acc[k]) / max(1, n_total)
    return out


def evaluate_one_language(model, tok, lang: str, args) -> Dict[str, Any]:
    # Prefer local JSONL files if --data_dir is set (HF Hub is often firewalled).
    if args.data_dir:
        local_jsonl = os.path.join(args.data_dir, f"{lang}_humanevalpack.jsonl")
        if os.path.exists(local_jsonl):
            print(f"[hef] loading from local jsonl: {local_jsonl}", flush=True)
            ds = load_dataset("json", data_files=local_jsonl, split="train")
        else:
            print(f"[hef] local jsonl not found ({local_jsonl}), trying HF hub ...", flush=True)
            try:
                ds = load_dataset("bigcode/humanevalpack", lang, split="test", trust_remote_code=True)
            except Exception as e:
                return {"pass@1": None, "n": 0, "error": f"missing local file and hub unreachable: {e}"}
    else:
        try:
            ds = load_dataset("bigcode/humanevalpack", lang, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"[hef] FAILED to load {lang} from hub: {e}", flush=True)
            return {"pass@1": None, "n": 0, "error": str(e)}

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    n_total = len(ds)
    n_pass = 0
    per_sample: List[Dict[str, Any]] = []
    runner = RUNNERS[lang]
    n_samples = max(1, args.num_samples)
    k_values = [k for k in args.k_values if k <= n_samples]
    passk_acc: Dict[int, List[float]] = {k: [] for k in k_values}

    if n_samples == 1 and getattr(args, "gen_batch_size", 1) > 1:
        return _evaluate_language_batched(model, tok, lang, ds, args, runner, k_values)

    for i, row in enumerate(ds):
        buggy = row["buggy_solution"]
        test_code = row["test"]
        instruction = row.get("instruction") or make_instruction(lang)
        buggy_for_prompt = compose_candidate(row, lang, buggy)

        exemplars = None
        if args.prompt_style == "cot":
            instruction = cot_instruction(instruction)
        elif args.prompt_style == "icl":
            exemplars = icl_retrieval.retrieve(
                ICL_BANK, HEF_TO_XCEK_LANG[lang], buggy_for_prompt, args.icl_shots
            )

        gens = generate_fix(model, tok, instruction, buggy_for_prompt,
                            max_input_len=args.max_input_len, max_new_tokens=args.max_new_tokens,
                            num_samples=n_samples, temperature=args.temperature, top_p=args.top_p,
                            exemplars=exemplars)

        n_correct = 0
        first_log = ""
        for gen in gens:
            fix = extract_fix(gen)
            fix = compose_candidate(row, lang, fix)
            try:
                ok, log = runner(fix, test_code)
            except Exception as e:
                ok, log = False, f"[runner-error] {e}"
            n_correct += int(ok)
            if not ok and not first_log:
                first_log = log

        ok_any = n_correct >= 1
        n_pass += int(ok_any)
        for k in k_values:
            passk_acc[k].append(pass_at_k(n_samples, n_correct, k))
        per_sample.append({
            "task_id": row.get("task_id", f"{lang}/{i}"),
            "pass": ok_any,
            "n_choices": n_samples,
            "n_correct": n_correct,
            "log_excerpt": first_log[:300] if not ok_any else "",
        })

        if (i + 1) % 5 == 0 or i == n_total - 1:
            running = sum(passk_acc[k_values[0]]) / (i + 1) if k_values else n_pass / (i + 1)
            print(f"[hef][{lang}] {i+1}/{n_total}  pass@{k_values[0] if k_values else 1}={running:.3f}", flush=True)

    out: Dict[str, Any] = {
        # pass@1 stays the unbiased estimator, which for greedy (n=1) is exactly
        # the old n_pass/n -- so historical numbers remain comparable.
        "pass@1": (sum(passk_acc[1]) / max(1, n_total)) if 1 in passk_acc else n_pass / max(1, n_total),
        "n_pass": n_pass,
        "n": n_total,
        "n_samples": n_samples,
        "per_sample": per_sample if args.dump_per_sample else None,
    }
    for k in k_values:
        out[f"pass@{k}"] = sum(passk_acc[k]) / max(1, n_total)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--mode", choices=["base", "peft", "moe"], required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--moe_state", default="")
    ap.add_argument("--patch_config", default="")
    ap.add_argument("--languages", nargs="+", default=list(LANGS))
    ap.add_argument("--data_dir", default="",
                    help="Optional local dir with {lang}_humanevalpack.jsonl files (preferred when HF Hub is firewalled).")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_input_len", type=int, default=1536)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--dump_per_sample", action="store_true")
    ap.add_argument("--num_samples", type=int, default=1,
                    help="Candidates per task. >1 switches on sampling (see --temperature).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 with --num_samples 1 = greedy (how every pass@1 in the paper was produced).")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--k_values", type=int, nargs="+", default=[1],
                    help="k values for the unbiased pass@k estimator; k > --num_samples is dropped.")
    ap.add_argument("--prompt_style", default="plain", choices=["plain", "icl", "cot"],
                    help="icl: RING-style few-shot with retrieved same-language exemplars "
                         "(raise --max_input_len to 4096); cot: analyze-then-fix zero-shot CoT "
                         "(raise --max_new_tokens to 1024).")
    ap.add_argument("--icl_bank", default="data/eval/icl_bank_perlang3k.jsonl")
    ap.add_argument("--icl_shots", type=int, default=2)
    ap.add_argument("--gen_batch_size", type=int, default=1,
                    help=">1 enables the batched-greedy fast path (n=1 only): "
                         "batched generation + threaded local tests.")
    ap.add_argument("--test_workers", type=int, default=4,
                    help="Thread pool size for local compile+test in the fast path.")
    args = ap.parse_args()

    if args.prompt_style == "icl":
        global ICL_BANK
        ICL_BANK = icl_retrieval.load_bank(args.icl_bank)

    t0 = time.time()
    model, tok = load_model(args)
    print(f"[t={time.time()-t0:.1f}s] model ready", flush=True)

    results: Dict[str, Any] = {}
    for lang in args.languages:
        if lang not in RUNNERS:
            print(f"[hef] unknown language '{lang}', skipping", flush=True)
            continue
        results[lang] = evaluate_one_language(model, tok, lang, args)

    valid = [r for r in results.values() if r.get("pass@1") is not None]
    n_all = max(1, sum(r["n"] for r in valid))
    # Sample-weighted over languages (every HumanEvalFix language has 164 tasks,
    # so this equals the macro average as long as no language is missing).
    overall = sum(r["pass@1"] * r["n"] for r in valid) / n_all
    overall_passk = {
        f"pass@{k}": sum(r[f"pass@{k}"] * r["n"] for r in valid if f"pass@{k}" in r) / n_all
        for k in args.k_values
        if any(f"pass@{k}" in r for r in valid)
    }

    ab_mode, ab_norm = get_moe_ablation()
    summary = {
        "mode": args.mode,
        "adapter": args.adapter,
        "moe_state": args.moe_state,
        "moe_ablate": ab_mode,
        "moe_ablate_norm": ab_norm,
        "max_samples": args.max_samples,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "overall_pass@1": overall,
        "overall_pass@k": overall_passk,
        "per_language": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"} for k, v in results.items()},
        "elapsed_seconds": time.time() - t0,
    }
    if args.dump_per_sample:
        summary["per_sample"] = {k: v.get("per_sample") for k, v in results.items()}

    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"}, indent=2, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[hef] saved -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
