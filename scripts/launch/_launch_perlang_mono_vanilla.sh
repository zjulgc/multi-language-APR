#!/bin/bash
# Per-language MONO vanilla dense LoRA baseline: one r=64 LoRA trained on a SINGLE
# language's 3K subset (data/variants/perlang3k/by_language/<MONO_LANG>/), using the
# same recipe as the multilingual vanilla run (native chat template, seq 2048,
# gradient checkpointing, batch 4 x grad_accum 4 = eff 16, 1 epoch). This is the
# "mono dense" arm compared against the multilingual MoE.
#
# NOTE: uses MONO_LANG (not LANG) so it never clobbers the shell locale variable.
#   MONO_LANG in: C C++ C# Java Kotlin Python Javascript Ruby PHP Rust Go
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"
if [[ -f .swanlab.env ]]; then set -a; source .swanlab.env; set +a; fi

PY="${PY:-/data/lgc/.conda/envs/peft/bin/python}"
BASE="${BASE:-/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct}"
MONO_LANG="${MONO_LANG:?set MONO_LANG to one of: C C++ C# Java Kotlin Python Javascript Ruby PHP Rust Go}"
DATA_ROOT="${DATA_ROOT:-data/variants/perlang3k}"
TRAIN_FILE="${DATA_ROOT}/by_language/${MONO_LANG}/train_sft.jsonl"
VAL_FILE="${VAL_FILE:-data/all/intrain_validation_sft.jsonl}"
# checkpoint dir: sanitize + / # so the name is filesystem/URL friendly
SAFE=$(echo "$MONO_LANG" | sed 's/+/p/g; s/#/sharp/g')
NAME="${NAME:-mono_vanilla_${SAFE}}"
OUT="checkpoints/${NAME}"

[[ -f "$TRAIN_FILE" ]] || { echo "missing train file: $TRAIN_FILE" >&2; exit 1; }

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
  --swanlab  --swanlab_project multi-language-APR  --swanlab_experiment "$NAME"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[mono $MONO_LANG] smoke ..."; rm -rf "$OUT"
  "$PY" "${COMMON[@]}" --max_train_samples 32 --max_val_samples 8 > "${OUT}.smoke.log" 2>&1 || true
  tail -3 "${OUT}.smoke.log"; echo "[mono $MONO_LANG] smoke done"; exit 0
fi

mkdir -p "$OUT"
echo "[mono $MONO_LANG] full ($(wc -l < "$TRAIN_FILE") rows) -> $OUT on GPU $CUDA_VISIBLE_DEVICES"
exec "$PY" -u "${COMMON[@]}" > "${OUT}/train.log" 2>&1
