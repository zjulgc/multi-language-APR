#!/bin/bash
# Score the pass@10 xCodeEval generations with ExecEval (CPU/docker, no GPU).
# eval_xcodeeval_execeval.py already implements the unbiased pass@k estimator,
# so we just ask for --k_values 1 10 over the n=10 generation dirs.
#
# Runs a health probe FIRST: this box has twice produced silently-wrong scores
# from a sick ExecEval (full disk -> ENOSPC read as COMPILATION_ERROR; and a
# post-reboot window where JVM/Go/.NET were mass-misjudged MLE, which cost
# multiLoRA ~1.2pp on xcek). Compiled languages are the canary.
#
# Usage: bash scripts/score_pass10_xcek.sh [tag ...]     (default: base multi s3)
set -u
cd /mnt/backup1/lgc/multi-language-APR
export PYTHONPATH=. TMPDIR=/mnt/backup1/lgc/tmp
PY=/data/lgc/.conda/envs/peft/bin/python
UDB=instruction_dataset/xCodeEval/unittest_db.json
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(base multi s3)
LOG=logs/score_pass10_xcek.log
ts(){ date '+%F %T'; }

echo "[$(ts)] ExecEval health probe (Rust/Java/Go canaries) ..." | tee -a "$LOG"
$PY - <<'EOF' | tee -a "$LOG"
import json, requests
# Compiler names MUST match LANG_CLUSTER_TO_LANG_COMPILER in
# eval_xcodeeval_execeval.py -- ExecEval 400s with KeyError on anything else.
CASES = {
    "Rust": ("fn main(){ let mut s=String::new(); std::io::stdin().read_line(&mut s).unwrap(); "
             "println!(\"{}\", s.trim()); }", "Rust 2018"),
    "Java": ("import java.util.*; public class Main{public static void main(String[] a){"
             "Scanner s=new Scanner(System.in); System.out.println(s.nextLine());}}", "Java 17"),
    "Go":   ("package main\nimport (\"bufio\";\"fmt\";\"os\";\"strings\")\n"
             "func main(){r:=bufio.NewReader(os.Stdin);l,_:=r.ReadString('\\n');"
             "fmt.Println(strings.TrimSpace(l))}", "Go"),
}
ut = [{"input": "hello\n", "output": ["hello"]}]
bad = []
for name, (src, compiler) in CASES.items():
    body = {"language": compiler, "source_code": src, "unittests": ut, "limits": None,
            "compile_cmd": None, "compile_flags": None, "execute_cmd": None,
            "execute_flags": None, "block_network": True, "stop_on_first_fail": True,
            "use_sanitizer": False}
    outcome = None
    try:
        r = requests.post("http://localhost:5000/api/execute_code", json=body, timeout=180)
        payload = r.json()
        if "data" not in payload:          # server-side error, e.g. unknown language
            outcome = f"SERVER_ERROR {payload.get('error')}"
        else:
            outcome = (payload["data"] or [{}])[0].get("exec_outcome")
    except Exception as e:
        outcome = f"ERROR {e}"
    ok = outcome == "PASSED"
    print(f"  {name:5s} -> {outcome} {'OK' if ok else '<-- UNHEALTHY'}")
    if not ok:
        bad.append(name)
print("HEALTH: " + ("OK" if not bad else f"FAILED for {bad} -- do NOT trust scores"))
EOF

if ! grep -q "^HEALTH: OK" "$LOG"; then
  echo "[$(ts)] ExecEval unhealthy -- aborting before it writes misleading numbers." | tee -a "$LOG"
  exit 1
fi

for tag in "${TAGS[@]}"; do
  GEN="eval_outputs/xcek100_${tag}_n10"
  OUT="results/xcek100_${tag}_n10.json"
  n=$(ls "$GEN"/*.json 2>/dev/null | wc -l)
  if [ "$n" -lt 1100 ]; then
    echo "[$(ts)] $tag: only $n/1100 generations -- skipping (rerun when complete)" | tee -a "$LOG"; continue
  fi
  echo "[$(ts)] scoring $tag ($n files) -> $OUT" | tee -a "$LOG"
  $PY -u eval_xcodeeval_execeval.py --gen_dir "$GEN" --unittest_db "$UDB" \
    --k_values 1 10 --include_per_choice --output_json "$OUT" >> "$LOG" 2>&1
  $PY -c "
import json; d=json.load(open('$OUT')); o=d['overall_pass@k']
print(f\"[$tag] pass@1={o.get('pass@1',0)*100:.1f}  pass@10={o.get('pass@10',0)*100:.1f}  n={o.get('n_tasks')}\")" | tee -a "$LOG"
done
echo "[$(ts)] done" | tee -a "$LOG"
