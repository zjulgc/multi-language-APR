#!/bin/bash
# RQ4 branch ablation: which branch of s3 (MoE-LoRA with 3 always-on shared
# experts) actually carries the cross-lingual prior?
#
# At inference we zero one branch's gate weights and re-measure HumanEvalFix
# pass@1 (greedy, 6 langs x 164 tasks). The switch lives in moe_apr/moe_layer.py
# and is inference-only -- training numbers cannot change.
#
#   condition          MOE_ABLATE  MOE_ABLATE_NORM  what survives
#   -----------------  ----------  ---------------  -------------------------------------
#   full               none        (n/a)            everything (control)
#   no_shared_renorm   shared      drop_renorm      11 routed experts, weights renormalized to 1
#   no_routing_renorm  routing     drop_renorm      3 shared experts, weights renormalized to 1
#   no_shared_drop     shared      drop             11 routed experts, original weights (sum < 1)
#   no_routing_drop    routing     drop             3 shared experts, original weights (sum < 1)
#
# Work is sharded per (condition, language): one HEF language is ~1.5-2 h, one
# full 6-language sweep is ~5-11 h on this shared box. Sharding lets the fleet
# use whatever cards free up, and -- because jobs are ordered LANGUAGE-MAJOR --
# a partial run still yields a complete 3-condition comparison for the languages
# it got through, instead of one finished condition and nothing to compare it to.
#
# Usage:
#   scripts/run_rq4_branch_ablation.sh one <condition> <lang> <gpu>
#   scripts/run_rq4_branch_ablation.sh plan                       # print job order
#   scripts/run_rq4_branch_ablation.sh fleet                      # auto-schedule on free GPUs
#
# GPU rules on this box: PCI_BUS_ID ordering; GPU6 (RTX 6000D, sm_120) is
# unusable under torch 2.2.1; GPU0-5 are shared with other people's jobs, so the
# fleet only claims a card with >= $NEED_MB MiB free and never kills anything.
set -u
cd /mnt/backup1/lgc/multi-language-APR || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=.
export TMPDIR=/mnt/backup1/lgc/tmp
export HF_DATASETS_CACHE=/mnt/backup1/lgc/hf_datasets_cache
export HF_HOME=/mnt/backup1/lgc/hf_home
export PATH=/mnt/backup1/lgc/goroot/bin:$PATH
export GOCACHE=/tmp/lgc-gocache
export GOPATH=/mnt/backup1/lgc/gopath

PY=/data/lgc/.conda/envs/peft/bin/python
BASE=/mnt/backup1/lgc/models/Qwen2.5-Coder-7B-Instruct
MOE_STATE=checkpoints/single_stage_perlang3k_e11r4_s3r4/moe_state.pt

# Main result first (full + the two renormalized ablations), then the "drop"
# variants as a supplement. Within each block, language-major.
PRIMARY_CONDS="${PRIMARY_CONDS:-full no_shared_renorm no_routing_renorm}"
SECONDARY_CONDS="${SECONDARY_CONDS:-no_shared_drop no_routing_drop}"
LANG_ORDER="${LANG_ORDER:-python js java cpp rust go}"

ALLOWED_GPUS="${ALLOWED_GPUS:-0 1 2 3 4 5}"   # never 6: sm_120, no torch 2.2.1 kernels
# A 7B bf16 inference process peaks ~17 GB, but a 48 GB card only fits two of
# them (usable capacity is 47.4 GiB, not 48). 20000 was too tight: other people's
# schedulers poll the same cards, so two claimants can pass the check on the same
# free block and the loser OOMs mid-generation. Extra headroom + retry (below).
NEED_MB="${NEED_MB:-22000}"
POLL_S="${POLL_S:-120}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"             # per shard, across OOM/transient failures

ts() { date '+%F %T'; }

env_for() {  # -> "MOE_ABLATE MOE_ABLATE_NORM"
  case "$1" in
    full)              echo "none drop_renorm" ;;
    no_shared_renorm)  echo "shared drop_renorm" ;;
    no_routing_renorm) echo "routing drop_renorm" ;;
    no_shared_drop)    echo "shared drop" ;;
    no_routing_drop)   echo "routing drop" ;;
    *) return 1 ;;
  esac
}

out_for() { echo "results/hef_s3_ablate_$1_$2.json"; }
rc_for()  { echo "logs/rq4_ablate_$1_$2.rc"; }

run_one() {  # $1 = condition, $2 = lang, $3 = gpu
  local cond="$1" lang="$2" gpu="$3" ab norm out log rc
  read -r ab norm <<<"$(env_for "$cond")" || { echo "unknown condition '$cond'"; return 2; }
  out=$(out_for "$cond" "$lang")
  log="logs/rq4_ablate_${cond}_${lang}.log"
  mkdir -p results logs
  rm -f "$(rc_for "$cond" "$lang")"
  echo "[$(ts)] gpu$gpu <- $cond/$lang (MOE_ABLATE=$ab MOE_ABLATE_NORM=$norm)"
  CUDA_VISIBLE_DEVICES="$gpu" MOE_ABLATE="$ab" MOE_ABLATE_NORM="$norm" \
    "$PY" -u eval_humanevalfix.py \
      --base_model "$BASE" \
      --mode moe --moe_state "$MOE_STATE" \
      --languages "$lang" --dump_per_sample \
      --output_json "$out" >"$log" 2>&1
  rc=$?
  # The scheduler reads this instead of `wait`-ing: a shard that dies (OOM from a
  # card another scheduler grabbed at the same instant, say) must go back in the
  # queue rather than be silently dropped.
  echo "$rc" >"$(rc_for "$cond" "$lang")"
  echo "[$(ts)] DONE rc=$rc $cond/$lang -> $out"
  return $rc
}

plan() {  # language-major within each priority block; skips finished shards
  local lang cond
  for cond_block in "$PRIMARY_CONDS" "$SECONDARY_CONDS"; do
    for lang in $LANG_ORDER; do
      for cond in $cond_block; do
        [ -s "$(out_for "$cond" "$lang")" ] || echo "$cond $lang"
      done
    done
  done
}

free_mb() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>/dev/null | awk -F, '{print $2-$1}'; }

# Claim the first GPU with enough headroom, sampled twice so we neither race a
# process that is mid-teardown nor double-book a card we just launched onto.
claim_gpu() {
  local g f1 f2
  for g in $ALLOWED_GPUS; do
    f1=$(free_mb "$g"); [ -n "$f1" ] || continue
    if [ "$f1" -ge "$NEED_MB" ]; then
      sleep 20; f2=$(free_mb "$g")
      if [ -n "$f2" ] && [ "$f2" -ge "$NEED_MB" ]; then echo "$g"; return 0; fi
    fi
  done
  return 1
}

# Shards already running -- possibly started by an earlier scheduler process that
# we replaced. Reading this from `ps` (rather than from in-memory state) is what
# makes the fleet safe to restart: it never double-launches a live shard.
inflight() {
  ps -eo args= 2>/dev/null \
    | grep -o 'results/hef_s3_ablate_[a-z_]*\.json' \
    | sed 's|results/hef_s3_ablate_||; s|\.json$||' \
    | while IFS= read -r stem; do
        for L in $LANG_ORDER; do
          case "$stem" in *_"$L") echo "${stem%_$L} $L" ;; esac
        done
      done
}

fleet() {
  mkdir -p logs results
  local pids=() job gpu n p key
  declare -A launched=()   # a shard writes its JSON only at the very end, so
                           # `plan` alone cannot tell "not started" from "running"
  declare -A attempts=()
  declare -A adopted=()   # started by a previous scheduler: no .rc file will appear,
                          # so liveness has to come from `ps`, not from run_one

  while IFS= read -r j; do
    [ -n "$j" ] && { launched[$j]=1; adopted[$j]=1; echo "[$(ts)] adopting in-flight shard: $j"; }
  done < <(inflight)

  while true; do
    local alive=()
    for p in "${pids[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]:-}")
    n=0; for p in "${pids[@]:-}"; do [ -n "$p" ] && n=$((n+1)); done

    # Requeue anything that stopped running without producing its JSON.
    local live; live=$(inflight)
    for key in "${!launched[@]}"; do
      # shellcheck disable=SC2086
      set -- $key
      [ -s "$(out_for "$1" "$2")" ] && continue
      local rcf why; rcf=$(rc_for "$1" "$2")
      if [ -n "${adopted[$key]:-}" ]; then
        printf '%s\n' "$live" | grep -qxF "$key" && continue   # still running
        why="vanished (adopted from a previous scheduler)"
      else
        [ -s "$rcf" ] || continue                              # still running
        why="rc=$(cat "$rcf")"
      fi
      if [ "${attempts[$key]:-0}" -ge "$MAX_ATTEMPTS" ]; then
        echo "[$(ts)] GIVING UP on $key after ${attempts[$key]} attempts ($why)"
        continue
      fi
      echo "[$(ts)] REQUEUE $key ($why, ${attempts[$key]:-0}/$MAX_ATTEMPTS attempts so far)"
      rm -f "$rcf"
      unset "launched[$key]" "adopted[$key]"
    done

    job=""
    while IFS= read -r cand; do
      [ -n "${launched[$cand]:-}" ] || { job="$cand"; break; }
    done < <(plan)

    if [ -z "$job" ]; then
      [ "$n" -eq 0 ] && break
      sleep "$POLL_S"; continue
    fi
    if [ "$n" -ge "$MAX_PARALLEL" ]; then sleep "$POLL_S"; continue; fi

    if gpu=$(claim_gpu); then
      # shellcheck disable=SC2086
      run_one $job "$gpu" &
      pids+=($!)
      launched[$job]=1
      attempts[$job]=$(( ${attempts[$job]:-0} + 1 ))
      sleep 180   # let it allocate so the next claim_gpu poll sees the card as busy
    else
      echo "[$(ts)] no GPU with >= ${NEED_MB}MiB free; next job '$job'; waiting ${POLL_S}s"
      sleep "$POLL_S"
    fi
  done
  wait
  echo "[$(ts)] fleet done"
}

case "${1:-fleet}" in
  one)   shift; run_one "$1" "$2" "$3" ;;
  plan)  plan ;;
  fleet) fleet ;;
  *) echo "usage: $0 {one <condition> <lang> <gpu> | plan | fleet}"; exit 2 ;;
esac
