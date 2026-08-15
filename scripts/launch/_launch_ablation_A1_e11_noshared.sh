#!/bin/bash
# [Ablation A1] s0-11e-top1: 11 routing experts rank 4 (top_k=1), NO shared expert.
# Identical to Config B (_launch_perlang3k_e11r4_s1r16.sh) minus the shared expert.
# Answers: is the always-active shared APR expert necessary at all?
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
NAME="${NAME:-ablation_A1_e11_noshared_top1}"
OUT="checkpoints/${NAME}"
DATA_ROOT="${DATA_ROOT:-data/variants/perlang3k}"
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
  --lora_r 4  --lora_alpha 8
  --shared_expert_gate_mode none
  --max_seq_len 2048
  --num_epochs 1.0
  --train_batch_size 4  --eval_batch_size 4  --grad_accum 4
  --learning_rate 2e-4  --warmup_ratio 0.05
  --router_aux_loss_coef 0.01
  --logging_steps 1  --eval_steps 500  --save_steps 500  --eval_strategy no
  --max_samples_per_language 3000
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --swanlab  --swanlab_project multi-language-APR
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "[A1 e11_noshared] smoke ..."; rm -rf "$OUT"
  "$PY" "${COMMON[@]}" --max_samples_per_language 4 --max_val_samples 8 \
    --grad_accum 1 --save_steps 100000 > "${OUT}.smoke.log" 2>&1
  grep -q "^Done\.$" "${OUT}.smoke.log" && echo "[A1] smoke OK"; exit 0
fi

mkdir -p "$OUT"
if [[ -n "${RESUME:-}" ]]; then
  COMMON+=(--resume_from_checkpoint "$RESUME")
  echo "[A1 e11_noshared] RESUME from $RESUME -> $OUT on GPU $CUDA_VISIBLE_DEVICES"
  exec "$PY" -u "${COMMON[@]}" >> "${OUT}/train.log" 2>&1
fi
echo "[A1 e11_noshared] full -> $OUT on GPU $CUDA_VISIBLE_DEVICES"
exec "$PY" -u "${COMMON[@]}" > "${OUT}/train.log" 2>&1
