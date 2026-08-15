#!/bin/bash
# Prompting-family baselines (training-free): Few-shot ICL (RING-style, 2-shot
# same-language retrieval) + Zero-shot CoT, on Qwen2.5-Coder-7B-Instruct and
# Meta-Llama-3-8B-Instruct, on xcek100 (1100) + HEF (984). 8 jobs on 6 lanes:
#   GPU0 qwen-icl-xcek | GPU1 qwen-cot-xcek | GPU2 qwen-icl-hef | GPU3 qwen-cot-hef
#   GPU4 llama-icl-xcek -> llama-icl-hef    | GPU5 llama-cot-xcek -> llama-cot-hef
# xcek scoring runs in background under the global ExecEval lock (w8) so the
# GPU proceeds to the chained HEF job while the CPU scores.
#   setsid nohup bash scripts/launch/_prompting_baselines.sh &
set -u
cd /mnt/backup1/lgc/multi-language-APR
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/mnt/backup1/lgc/tmp HF_DATASETS_CACHE=/mnt/backup1/lgc/hf_datasets_cache HF_HOME=/mnt/backup1/lgc/hf_home
export PATH=/mnt/backup1/lgc/goroot/bin:$PATH
export GOCACHE=/tmp/lgc-gocache GOPATH=/mnt/backup1/lgc/gopath
mkdir -p logs results/xcek_fresh eval_outputs "$GOCACHE" "$GOPATH"

PY=/data/lgc/.conda/envs/peft/bin/python
QWEN=/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct
LLAMA=/mnt/backup1/lgc/models/Meta-Llama-3-8B-Instruct
UDB=instruction_dataset/xCodeEval/unittest_db.json
SUBSET=data/eval/xcodeeval_test_100perlang.jsonl
HEFDIR=data/variants/humanevalfix
LOG=logs/prompting_baselines.log
ts(){ date '+%F %T'; }
log(){ echo "[$(ts)] $*" >> "$LOG"; }

wait_gpu(){  # block until GPU $1 is free (<4GB); it should be free already
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
    [ "$used" -lt 4096 ] && return
    log "gpu$1 busy (${used}MiB), waiting"; sleep 300
  done
}

grab_any(){  # echo first free GPU among $@ (double-checked 60s apart)
  while :; do
    for g in "$@"; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
      if [ "$u" -lt 4096 ]; then
        sleep 60
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
        [ "$u" -lt 4096 ] && { echo "$g"; return; }
      fi
    done
    sleep 120
  done
}

gen_xcek(){  # $1 gpu $2 model $3 style $4 tag
  local extra="" bs=8
  # icl: 4k+ prompts -> fp32-logits prefill OOMs at BS 8 (~19GB spike); use 4
  if [ "$3" = icl ]; then extra="--prompt_style icl --prompt_max_len 4096 --max_new_tokens 1024"; bs=4
  else extra="--prompt_style cot --max_new_tokens 1536"; fi
  local gdir=eval_outputs/xcek100_$4
  local out=results/xcek_fresh/$4.json
  [ -s "$out" ] && { log "xcek $4 already scored, skip"; return; }
  until CUDA_VISIBLE_DEVICES=$1 $PY -u generate_apr_local.py \
      --base_model "$2" --subset_jsonl "$SUBSET" --unittest_db "$UDB" \
      --output_dir "$gdir" --num_samples 1 --temperature 0.0 \
      --batch_size $bs --skip_existing $extra >> "logs/gen_$4.log" 2>&1 \
      && [ "$(ls "$gdir" | grep -c '\.json$')" -ge 1100 ]; do
    log "xcek $4 gen incomplete ($(ls "$gdir" 2>/dev/null | grep -c '\.json$')/1100), retrying"; sleep 120
  done
  log "xcek $4 gen done -> scoring (bg, global lock)"
  (
    flock 9
    $PY -u eval_xcodeeval_execeval.py --gen_dir "$gdir" --unittest_db "$UDB" \
      --k_values 1 --max_workers 8 --include_per_choice --output_json "$out" \
      >> "logs/score_$4.log" 2>&1
    log "xcek $4 scored -> $out"
  ) 9>eval_outputs/.xcekfull_score.global.lock &
}

run_hef(){  # $1 gpu $2 model $3 style $4 out_json  (batched fast path, 08-15 用户要求提速)
  local extra=""
  if [ "$3" = icl ]; then extra="--prompt_style icl --max_input_len 4096 --gen_batch_size 4 --test_workers 4"
  else extra="--prompt_style cot --max_new_tokens 1024 --gen_batch_size 8 --test_workers 4"; fi
  [ -s "$4" ] && { log "hef $4 exists, skip"; return; }
  until CUDA_VISIBLE_DEVICES=$1 $PY -u eval_humanevalfix.py --mode base \
      --base_model "$2" --languages python js java cpp rust go \
      --data_dir "$HEFDIR" --dump_per_sample $extra \
      --output_json "$4" >> "logs/hef_$(basename "$4" .json).log" 2>&1 \
      && [ -s "$4" ]; do
    log "hef $4 failed, retrying"; sleep 120
  done
  log "hef $4 done"
}

log "prompting baselines fleet up (pid $$)"

# 用户指令(08-15 04:30): 有空卡就上,Llama HEF 不再等 Qwen HEF(取消此前的串行门).
# GPU2/3 上有游离的 Qwen HEF python(重启时特意留下的),对应泳道等卡空后经
# exists-guard 正常跳过,仅在其崩溃时补跑.
( wait_gpu 0; gen_xcek 0 "$QWEN" icl fewshot_icl2 ) &
( wait_gpu 1; gen_xcek 1 "$QWEN" cot zeroshot_cot ) &
( wait_gpu 2; run_hef 2 "$QWEN" icl results/hef_fewshot_icl2.json ) &
( wait_gpu 3; run_hef 3 "$QWEN" cot results/hef_zeroshot_cot.json ) &
# GPU4 被外部 Ollama runner 占用(04:05 起,勿动别人进程);llama-icl-hef 改为在
# 4/0/1 中抢首个空卡(0/1 生成完即释放)
( G=$(grab_any 4 0 1); log "llama icl hef -> gpu$G"; \
  run_hef "$G" "$LLAMA" icl results/hef_fewshot_icl2_llama3.json ) &
( wait_gpu 5; run_hef 5 "$LLAMA" cot results/hef_zeroshot_cot_llama3.json ) &
wait
log "PROMPTING_BASELINES_ALL_DONE"
