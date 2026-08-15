#!/bin/bash
# [Config C] Vanilla dense LoRA baseline on the 3w3 set (perlang3k, 33K merged).
#   single dense adapter r=64, alpha=128 (no MoE, no routing, no shared expert)
# Trainable ~161.5M -- the param-matched reference the two MoE variants bracket
# (A 3xr4-shared 157.2M, B 1xr16-shared 165.0M). Same 33K rows, same recipe.
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"

PY="${PY:-/data/lgc/.conda/envs/peft/bin/python}"
BASE="${BASE:-/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct}"
NAME="${NAME:-vanilla_lora_r64_perlang3k}"
OUT="checkpoints/${NAME}"
TRAIN_FILE="${TRAIN_FILE:-data/variants/perlang3k/all/train_sft.jsonl}"
VAL_FILE="${VAL_FILE:-data/all/intrain_validation_sft.jsonl}"

COMMON=(
  train_lora_xcodeeval.py
  --base_model "$BASE"
  --train_file "$TRAIN_FILE"
  --val_file "$VAL_FILE"
  --output_dir "$OUT"
  --lora_r 64  --lora_alpha 128  --lora_dropout 0.05
  --max_seq_len 2048
  --num_epochs 1
  --train_batch_size 4  --eval_batch_size 4  --grad_accum 4
  --learning_rate 2e-4
  --swanlab  --swanlab_project multi-language-APR
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[C vanilla_r64] smoke ..."; rm -rf "$OUT"
  "$PY" "${COMMON[@]}" --max_train_samples 32 --max_val_samples 8 \
    > "${OUT}.smoke.log" 2>&1 || true
  tail -3 "${OUT}.smoke.log"; echo "[C] smoke done"; exit 0
fi

mkdir -p "$OUT"
echo "[C vanilla_r64] full -> $OUT on GPU $CUDA_VISIBLE_DEVICES"
exec "$PY" -u "${COMMON[@]}" > "${OUT}/train.log" 2>&1
