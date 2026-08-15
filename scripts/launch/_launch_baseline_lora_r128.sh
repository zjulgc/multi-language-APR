#!/bin/bash
# [Baseline B3 · capacity-scaled plain LoRA] Single dense LoRA at r=128, alpha=256.
#
# WHY r=128 and not "parameter-matched": multiLoRA r64 (161.5M) is ALREADY
# parameter-matched to s3 (157.2M) -- the equal-capacity comparison is the one
# the paper has been reporting all along. So the open question a reviewer asks
# after RQ3 ("MoE organisation doesn't matter") is the *scaling* one:
#
#     if you simply give a plain LoRA twice the capacity, does it reach s3?
#
# r=128 -> 323.0M trainable = 2.06x s3, 2.00x multiLoRA r64.
#   * plain r128 ~ s3  -> nothing about MoE structure matters, only capacity;
#   * plain r128 < s3  -> s3's organisation buys something capacity alone cannot.
# Either way the paper gets a clean answer, which the discarded language-tag
# baseline could not have given (prompts already state the language in NL).
#
# Same data (perlang3k 33K), same 1 epoch, same effective batch 16
# (per_device 2 x grad_accum 2 x 4 GPUs), same LR -- only the rank differs.
#
# Launch:  setsid nohup bash scripts/launch/_launch_baseline_lora_r128.sh </dev/null >/dev/null 2>&1 &
# Smoke:   SMOKE=1 NPROC=2 CUDA_VISIBLE_DEVICES=4,5 bash scripts/launch/_launch_baseline_lora_r128.sh
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR

export CUDA_DEVICE_ORDER=PCI_BUS_ID   # PCI 6 is the RTX 6000D (sm_120): unusable, never schedule there
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY="${PY:-/data/lgc/.conda/envs/peft/bin/python}"
BASE="${BASE:-/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct}"
NAME="${NAME:-baseline_lora_r128_perlang3k}"
OUT="checkpoints/${NAME}"
TRAIN_FILE="${TRAIN_FILE:-data/variants/perlang3k/all/train_sft.jsonl}"
VAL_FILE="${VAL_FILE:-data/all/intrain_validation_sft.jsonl}"
NPROC="${NPROC:-4}"
PORT="${PORT:-29519}"

COMMON=(
  train_lora_xcodeeval.py
  --base_model "$BASE"
  --train_file "$TRAIN_FILE"
  --val_file "$VAL_FILE"
  --output_dir "$OUT"
  --lora_r 128  --lora_alpha 256  --lora_dropout 0.05
  --max_seq_len 2048
  --num_epochs 1
  --train_batch_size 2  --eval_batch_size 4  --grad_accum 2
  --learning_rate 2e-4
)

TORCHRUN=("$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT")

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[lora_r128] smoke (nproc=${NPROC}) ..."; rm -rf "$OUT"
  "${TORCHRUN[@]}" "${COMMON[@]}" --max_train_samples 32 --max_val_samples 8 \
    > "${OUT}.smoke.log" 2>&1 || { echo "[lora_r128] smoke FAILED"; tail -30 "${OUT}.smoke.log"; exit 1; }
  grep -m1 "trainable params" "${OUT}.smoke.log" || true
  [[ -f "${OUT}/adapter_model.safetensors" ]] && echo "[lora_r128] smoke OK (adapter written)" \
    || { echo "[lora_r128] smoke INCOMPLETE"; tail -30 "${OUT}.smoke.log"; exit 1; }
  exit 0
fi

mkdir -p "$OUT"
echo "[lora_r128] full nproc=${NPROC} on GPUs ${CUDA_VISIBLE_DEVICES} -> $OUT"
exec "${TORCHRUN[@]}" "${COMMON[@]}" > "${OUT}/train.log" 2>&1
