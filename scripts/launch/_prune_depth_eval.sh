#!/bin/bash
# RQ3 深度剖面的推理期剪枝验证（2026-08-15）。
#
# 观察：中段层 L9-18 的路由接近均匀（熵 0.761）、语言可分性最低（eta2 0.105），
# 首尾两端则高度集中。若"路由是否分化"能指示模块是否可剪，则剪掉中段应比剪掉
# 等量的两端模块代价更小。
#
# 两个条件各剪 10 层 x 7 投影 = 70/196 个模块（36% 的适配器参数）：
#   mid  : L9-18            低语言信号（假设可剪）
#   ends : L0-4 + L23-27    高语言信号（等量对照）
# 完整 MultiRepair 的参照值：xcek 282/1100、HEF 667/984。
#
# 剪枝仅作用于推理（moe_layer.py 中 bypass_at_inference 在 training 时为 no-op）。
set -u
cd /mnt/backup1/lgc/multi-language-APR
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/backup1/lgc/tmp HF_DATASETS_CACHE=/mnt/backup1/lgc/hf_datasets_cache HF_HOME=/mnt/backup1/lgc/hf_home
export PATH=/mnt/backup1/lgc/goroot/bin:$PATH
export GOCACHE=/tmp/lgc-gocache GOPATH=/mnt/backup1/lgc/gopath

PY=/data/lgc/.conda/envs/peft/bin/python
BASE=/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct
MS=checkpoints/single_stage_perlang3k_e11r4_s3r4/moe_state.pt
SUBSET=data/eval/xcodeeval_test_100perlang.jsonl
UDB=instruction_dataset/xCodeEval/unittest_db.json
LOG=logs/prune_depth.log
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $*" >> "$LOG"; }

gen_xcek(){  # $1 gpu  $2 tag  $3 layer-spec
  local gdir=eval_outputs/xcek100_prune_$2
  local out=results/xcek_fresh/prune_$2.json
  [ -s "$out" ] && { log "$2 xcek exists, skip"; return; }
  log "$2 xcek gen -> gpu$1 (bypass $3)"
  MOE_BYPASS_LAYERS="$3" CUDA_VISIBLE_DEVICES=$1 $PY -u generate_moe_apr.py \
      --base_model "$BASE" --moe_state "$MS" --subset_jsonl "$SUBSET" \
      --output_dir "$gdir" --num_samples 1 --temperature 0.0 \
      --max_new_tokens 1024 --batch_size 8 --skip_existing \
      >> "logs/prune_gen_$2.log" 2>&1
  log "$2 xcek gen done -> scoring"
  (
    flock 9
    $PY -u eval_xcodeeval_execeval.py --gen_dir "$gdir" --unittest_db "$UDB" \
        --k_values 1 --max_workers 8 --include_per_choice --output_json "$out" \
        >> "logs/prune_score_$2.log" 2>&1
  ) 9>eval_outputs/.xcekfull_score.global.lock
  log "$2 xcek scored -> $out"
}

run_hef(){  # $1 gpu  $2 tag  $3 layer-spec
  local out=results/hef_prune_$2.json
  [ -s "$out" ] && { log "$2 hef exists, skip"; return; }
  log "$2 hef -> gpu$1 (bypass $3)"
  MOE_BYPASS_LAYERS="$3" CUDA_VISIBLE_DEVICES=$1 $PY -u eval_humanevalfix.py \
      --mode moe --base_model "$BASE" --moe_state "$MS" \
      --languages python js java cpp rust go --data_dir data/variants/humanevalfix \
      --dump_per_sample --gen_batch_size 8 --test_workers 4 \
      --output_json "$out" >> "logs/prune_hef_$2.log" 2>&1
  log "$2 hef done"
}

MID="9-18"
ENDS="0-4,23-27"

log "prune depth eval up (pid $$)"
( gen_xcek 0 mid  "$MID"  ) &
( gen_xcek 2 ends "$ENDS" ) &
( run_hef  3 mid  "$MID"; run_hef 3 ends "$ENDS" ) &
wait
log "PRUNE_DEPTH_ALL_DONE"
