#!/bin/bash
# Llama-3 counterpart of the r128 baseline (cross-base fairness with the paper's
# Qwen r128). Same recipe as _launch_baseline_lora_r128.sh — r=128, alpha=256,
# perlang3k 33K, 1 epoch, LR 2e-4, max_seq 2048 — but base=Meta-Llama-3-8B and
# grad_accum is auto-set so the EFFECTIVE BATCH stays 16 for whatever NPROC is
# free (2 tbs * GA * NPROC = 16). Foreground (exec) so a caller can chain eval.
#   NPROC=2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/launch/_launch_baseline_lora_r128_llama3.sh
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # PCI 6 = RTX 6000D (sm_120): never schedule there
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set freed A6000 ids, e.g. 2,3}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PY=/data/lgc/.conda/envs/peft/bin/python
BASE=/mnt/backup1/lgc/models/Meta-Llama-3-8B-Instruct
NAME=baseline_lora_r128_perlang3k_llama3
OUT="checkpoints/${NAME}"
NPROC="${NPROC:-2}"
GA=$(( 8 / NPROC )); [ "$GA" -lt 1 ] && GA=1   # 2 * GA * NPROC = 16
PORT="${PORT:-29527}"

COMMON=(
  train_lora_xcodeeval.py
  --base_model "$BASE"
  --train_file data/variants/perlang3k/all/train_sft.jsonl
  --val_file data/all/intrain_validation_sft.jsonl
  --output_dir "$OUT"
  --lora_r 128 --lora_alpha 256 --lora_dropout 0.05
  --max_seq_len 2048 --num_epochs 1
  --train_batch_size 2 --eval_batch_size 4 --grad_accum "$GA"
  --learning_rate 2e-4
)
mkdir -p "$OUT"
echo "[r128-llama3] nproc=$NPROC grad_accum=$GA (eff batch $((2*GA*NPROC))) gpus=$CUDA_VISIBLE_DEVICES -> $OUT"
exec "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT" \
  "${COMMON[@]}" > "${OUT}/train.log" 2>&1
