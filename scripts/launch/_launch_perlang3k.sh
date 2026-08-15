#!/bin/bash
# Grouping-free per-LANGUAGE MoE: 11 routing experts (one per language) + 1 shared
# expert, trained on a per-language BALANCED set (3K rows each, 33K total).
# Drops the 4-family concept: sampling AND routing are both by individual language.
#   - data: data/variants/perlang3k/by_language/<Lang>/ (each dir == one language)
#   - BalancedLanguageSampler balances 11-way (language column == language name)
#   - --route_by language -> routing target = LANG_TO_EXPERT[lang_cluster] (0..10)
# top_k=1 on routing side => joint softmax pool [shared, 1 routing] (effective top-2).
# Free router + Switch load-balance (no oracle / no route supervision).
set -euo pipefail
cd /mnt/backup1/lgc/multi-language-APR

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/mnt/backup1/lgc/tmp}"

if [[ -f .swanlab.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .swanlab.env
  set +a
fi

PY="${PY:-/data/lgc/.conda/envs/peft/bin/python}"
BASE="${BASE:-/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct}"
NAME="${NAME:-single_stage_perlang3k_shared_top1}"
OUT="checkpoints/${NAME}"
DATA_ROOT="${DATA_ROOT:-data/variants/perlang3k}"

# 11 languages == 11 routing experts. Names must match lang_cluster spelling and
# moe_apr.data_utils.LANGS_CANONICAL order (expert index 0..10).
LANGS="C,C++,C#,Java,Kotlin,Python,Javascript,Ruby,PHP,Rust,Go"

COMMON=(
  train_moe_apr.py
  --base_model "$BASE"
  --data_root "$DATA_ROOT"
  --languages "$LANGS"
  --val_file data/all/intrain_validation_sft.jsonl
  --output_dir "$OUT"
  --num_routing_experts 11
  --route_by language
  --top_k 1
  --lora_r 6
  --lora_alpha 12
  --shared_lora_r 16
  --shared_lora_alpha 32
  --num_shared_experts 1
  --shared_expert_gate_mode adaptive
  --max_seq_len 1024
  --num_epochs 1.0
  --train_batch_size 2
  --eval_batch_size 4
  --grad_accum 8
  --learning_rate 2e-4
  --warmup_ratio 0.05
  --router_aux_loss_coef 0.01
  --logging_steps 1
  --eval_steps 500
  --save_steps 500
  --eval_strategy no
  --max_samples_per_language 3000
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --swanlab
  --swanlab_project multi-language-APR
)

SMOKE="${SMOKE:-0}"
if [[ "$SMOKE" == "1" ]]; then
  echo "[perlang3k] smoke run ..."
  rm -rf "$OUT"
  "$PY" "${COMMON[@]}" \
    --max_samples_per_language 4 \
    --max_val_samples 8 \
    --grad_accum 1 \
    --save_steps 100000 \
    --logging_steps 1 \
    > "${OUT}.smoke.log" 2>&1
  grep -q "^Done\.$" "${OUT}.smoke.log"
  echo "[perlang3k] smoke OK"
  exit 0
fi

mkdir -p "$OUT"
echo "[perlang3k] full training -> $OUT on GPU $CUDA_VISIBLE_DEVICES"
exec "$PY" -u "${COMMON[@]}" > "${OUT}/train.log" 2>&1
