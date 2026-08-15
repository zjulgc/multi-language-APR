#!/bin/bash
# A/B check for the MoE inference fast path (moe_apr/moe_layer.py).
#
# Generates the SAME xCodeEval items twice with the same greedy settings --
# once forcing the per-expert loop (MOE_FAST_INFER=0), once with the
# concatenated fast path -- and reports wall-clock for each. The texts are
# diffed afterwards; they must be identical (or near-identical: bf16 changes
# float summation order, which can flip a token late in a long generation).
set -u
cd /mnt/backup1/lgc/multi-language-APR
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/mnt/backup1/lgc/tmp
PY=/data/lgc/.conda/envs/peft/bin/python
BASE=/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct
MS=checkpoints/single_stage_perlang3k_e11r4_s3r4/moe_state.pt
SUB=data/eval/xcodeeval_test_100perlang.jsonl
GPU=${GPU:-0}
N=${N:-4}
OUTROOT=/tmp/claude-1006/-mnt-backup1-lgc-multi-language-APR/0a6138b4-502a-41ae-a6aa-0da86d300e0c/scratchpad/fastpath_verify

run_one() {  # $1 = MOE_FAST_INFER value, $2 = out subdir
  local out="$OUTROOT/$2"
  rm -rf "$out"; mkdir -p "$out"
  local t0=$(date +%s)
  MOE_FAST_INFER="$1" CUDA_VISIBLE_DEVICES="$GPU" $PY -u generate_moe_apr.py \
    --base_model "$BASE" --moe_state "$MS" --subset_jsonl "$SUB" \
    --output_dir "$out" --num_samples 1 --temperature 0.0 \
    --max_new_tokens 1024 --batch_size 1 --max_items "$N" > "$OUTROOT/$2.log" 2>&1
  local t1=$(date +%s)
  echo "[verify] MOE_FAST_INFER=$1 -> $((t1 - t0))s for $N items (incl. ~60s model load)"
}

mkdir -p "$OUTROOT"
run_one 0 loop
run_one 1 fast

$PY - "$OUTROOT" <<'EOF'
import json, os, sys, glob
root = sys.argv[1]
loop = {os.path.basename(p): p for p in glob.glob(f"{root}/loop/*.json")}
fast = {os.path.basename(p): p for p in glob.glob(f"{root}/fast/*.json")}
same = diff = 0
for name in sorted(loop):
    if name not in fast:
        print(f"[verify] MISSING in fast: {name}"); continue
    a = json.load(open(loop[name]))["oai_response"]["choices"][0]["message"]["content"]
    b = json.load(open(fast[name]))["oai_response"]["choices"][0]["message"]["content"]
    if a == b:
        same += 1
    else:
        diff += 1
        pref = os.path.commonprefix([a, b])
        print(f"[verify] DIFFER: {name}  identical prefix {len(pref)}/{len(a)} chars")
print(f"[verify] identical: {same}/{same + diff}")
EOF
