#!/usr/bin/env bash
# Fresh unified re-score of all xcek greedy generations on the current healthy
# ExecEval, so every per-language cell comes from one scoring regime.
# CPU-only (ExecEval :5000). Writes to results/xcek_fresh/ — never clobbers archived JSONs.
set -u
cd /mnt/backup1/lgc/multi-language-APR
OUT=results/xcek_fresh
mkdir -p "$OUT"
EXE=http://localhost:5000
GENROOT=eval_outputs

score () {  # name  gen_dir  [filter_token]
  local name="$1" gd="$2" filt="${3:-}"
  local out="$OUT/${name}.json"
  if [[ ! -d "$gd" ]]; then echo "[skip] missing gen_dir $gd"; return; fi
  if [[ -s "$out" ]]; then echo "[have] $out"; return; fi
  echo "===== $(date +%H:%M:%S) scoring $name  ($gd)  filter='${filt}' ====="
  # w8: higher concurrency starves ExecEval and times out slow JVM/compiled jobs,
  # silently depressing Java/Kotlin. w8 reproduces archived base within +-1.
  local args=(--gen_dir "$gd" --output_json "$out" --execeval_url "$EXE"
              --k_values 1 --max_workers 8 --include_per_choice)
  [[ -n "$filt" ]] && args+=(--lang_cluster_filter "$filt")
  python3 eval_xcodeeval_execeval.py "${args[@]}" 2>&1 | tail -3
}

MONO=(C:_C.json Cpp:_C++.json Csharp:_C#.json Go:_Go.json Java:_Java.json \
      Javascript:_Javascript.json Kotlin:_Kotlin.json PHP:_PHP.json \
      Python:_Python.json Ruby:_Ruby.json Rust:_Rust.json)

# ---------- Qwen ----------
score base            "$GENROOT/xcek100_base"
score multi           "$GENROOT/xcek100_vanilla_lora_r64_perlang3k"
score hydralora       "$GENROOT/xcek100_hydralora"
score merged_linear   "$GENROOT/xcek100_merged_linear"
score lora_r128       "$GENROOT/xcek100_lora_r128"
score s3              "$GENROOT/xcek100_single_stage_perlang3k_e11r4_s3r4"
score A1              "$GENROOT/xcek100_ablation_A1_e11_noshared_top1"
score A2              "$GENROOT/xcek100_ablation_A2_e14_noshared_top4"
score dense           "$GENROOT/xcek100_ablation_dense_4e_top4"
for m in "${MONO[@]}"; do
  L="${m%%:*}"; F="${m##*:}"
  score "mono_${L}" "$GENROOT/xcek100_mono_vanilla_${L}" "$F"
done

# ---------- Llama-3 ----------
score base_llama3     "$GENROOT/xcek100_base_llama3"
score multi_llama3    "$GENROOT/xcek100_vanilla_lora_r64_perlang3k_llama3"
score s3_llama3       "$GENROOT/xcek100_single_stage_perlang3k_e11r4_s3r4_llama3"
for m in "${MONO[@]}"; do
  L="${m%%:*}"; F="${m##*:}"
  score "mono_${L}_llama3" "$GENROOT/xcek100_mono_vanilla_${L}_llama3" "$F"
done

echo "===== $(date +%H:%M:%S) ALL DONE ====="
