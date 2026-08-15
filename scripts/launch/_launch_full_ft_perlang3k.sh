#!/bin/bash
# Full-parameter fine-tune of Qwen2.5-Coder-7B-Instruct on the perlang3k 33K set.
# Same data/seq-len/effective-batch recipe as vanilla_lora_r64_perlang3k, but all
# 7.6B params train (LR dropped to full-FT scale 1e-5). 4-GPU FSDP full_shard:
# params+grads+Adam states sharded across ranks (pure-bf16, ~15GB states/GPU).
#   SMOKE=1 bash scripts/launch/_launch_full_ft_perlang3k.sh   # 48-sample dry run
#   setsid nohup bash scripts/launch/_launch_full_ft_perlang3k.sh &
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"
export HF_DATASETS_CACHE=/mnt/backup1/lgc/hf_datasets_cache HF_HOME=/mnt/backup1/lgc/hf_home
export TOKENIZERS_PARALLELISM=false

PY=/data/lgc/.conda/envs/peft/bin/python
TORCHRUN=/data/lgc/.conda/envs/peft/bin/torchrun
BASE="${BASE:-/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct}"
NAME="${NAME:-full_ft_perlang3k}"
OUT="checkpoints/${NAME}"
TRAIN_FILE="${TRAIN_FILE:-data/variants/perlang3k/all/train_sft.jsonl}"
VAL_FILE="${VAL_FILE:-data/all/intrain_validation_sft.jsonl}"
NPROC="${NPROC:-4}"

COMMON=(
  train_full_xcodeeval.py
  --base_model "$BASE"
  --train_file "$TRAIN_FILE"
  --val_file "$VAL_FILE"
  --output_dir "$OUT"
  --max_seq_len 2048
  --num_epochs 1
  --train_batch_size 2  --eval_batch_size 2  --grad_accum 2
  --learning_rate 1e-5
  --swanlab  --swanlab_project multi-language-APR
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[full_ft] smoke ..."; rm -rf "${OUT}_smoke"
  "$TORCHRUN" --nproc_per_node="$NPROC" --master_port 29617 \
    train_full_xcodeeval.py \
    --base_model "$BASE" --train_file "$TRAIN_FILE" --val_file "$VAL_FILE" \
    --output_dir "${OUT}_smoke" --max_seq_len 2048 --num_epochs 1 \
    --train_batch_size 2 --eval_batch_size 2 --grad_accum 2 \
    --learning_rate 1e-5 --max_train_samples 48 --max_val_samples 8 \
    > "logs/${NAME}_smoke.log" 2>&1 || true
  tail -5 "logs/${NAME}_smoke.log"; echo "[full_ft] smoke done"; exit 0
fi

mkdir -p "$OUT" logs
echo "[full_ft] full -> $OUT on GPUs $CUDA_VISIBLE_DEVICES ($NPROC ranks)"
exec "$TORCHRUN" --nproc_per_node="$NPROC" --master_port 29617 \
  "${COMMON[@]}" > "logs/${NAME}_train.log" 2>&1
