#!/bin/bash
# Master orchestrator: run the full train->generate->ExecEval pipeline for all 14
# models (3 multilingual + 11 per-language mono), packing them onto free GPUs
# (one model per GPU; the GPU is held through that model's whole chain). Detached
# per-model chains (setsid) survive disconnection.
#
# Launch this itself detached so it keeps scheduling the queued models:
#   setsid bash scripts/run_all_pipeline.sh </dev/null >logs/master_pipeline.log 2>&1 &
set -u
cd /mnt/backup1/lgc/multi-language-APR
mkdir -p logs results eval_outputs

GPUS="${GPUS:-0 1 2 3 4 5}"
FREE_MEM_MB="${FREE_MEM_MB:-2000}"

# "NAME|TYPE|TRAIN_CMD"  -- multilingual first (they hold GPUs ~27h), then mono.
JOBS=(
  "single_stage_perlang3k_e11r4_s3r4|moe|bash scripts/launch/_launch_perlang3k_e11r4_s3r4.sh"
  "single_stage_perlang3k_e11r4_s1r16|moe|bash scripts/launch/_launch_perlang3k_e11r4_s1r16.sh"
  "vanilla_lora_r64_perlang3k|peft|bash scripts/launch/_launch_perlang3k_vanilla_r64.sh"
  "mono_vanilla_Python|peft|MONO_LANG=Python bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Java|peft|MONO_LANG=Java bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Cpp|peft|MONO_LANG=C++ bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Go|peft|MONO_LANG=Go bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Javascript|peft|MONO_LANG=Javascript bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Rust|peft|MONO_LANG=Rust bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_C|peft|MONO_LANG=C bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Csharp|peft|MONO_LANG=C# bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Kotlin|peft|MONO_LANG=Kotlin bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_Ruby|peft|MONO_LANG=Ruby bash scripts/launch/_launch_perlang_mono_vanilla.sh"
  "mono_vanilla_PHP|peft|MONO_LANG=PHP bash scripts/launch/_launch_perlang_mono_vanilla.sh"
)

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 999999; }
find_free_gpu(){ for g in $GPUS; do [ "$(gpu_used "$g")" -lt "$FREE_MEM_MB" ] && { echo "$g"; return 0; }; done; return 1; }

echo "[master] $(date '+%F %T') scheduling ${#JOBS[@]} models on GPUs: $GPUS"
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME TYPE TRAIN_CMD <<< "$job"
  g=""
  while [ -z "$g" ]; do g=$(find_free_gpu) || { sleep 30; g=""; }; done
  echo "[master] $(date '+%F %T') $NAME ($TYPE) -> GPU $g"
  NAME="$NAME" TYPE="$TYPE" GPU="$g" TRAIN_CMD="$TRAIN_CMD" \
    setsid bash scripts/_pipeline_one.sh </dev/null >/dev/null 2>&1 &
  # wait until this model claims its GPU so the next one doesn't double-book it
  for _ in $(seq 1 40); do [ "$(gpu_used "$g")" -ge "$FREE_MEM_MB" ] && break; sleep 6; done
done
echo "[master] $(date '+%F %T') all ${#JOBS[@]} models launched (chains run detached)"
