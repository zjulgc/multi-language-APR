#!/bin/bash
# Driver: train a per-language MONO vanilla LoRA for each language, packing runs
# onto free GPUs (one run per GPU, queued). Each run is detached (setsid) and
# logs to checkpoints/mono_vanilla_<lang>/train.log + SwanLab.
#
# Run this itself detached so it survives disconnection while it queues:
#   setsid bash scripts/run_perlang_mono_all.sh </dev/null >logs/mono_driver.log 2>&1 &
#
# Override the set / GPUs:
#   LANGS="Python Java C++ Go Javascript Rust" GPUS="0 1 2 3 4 5" bash scripts/run_perlang_mono_all.sh
set -uo pipefail
cd /mnt/backup1/lgc/multi-language-APR

LANGS="${LANGS:-C C++ C# Java Kotlin Python Javascript Ruby PHP Rust Go}"
GPUS="${GPUS:-0 1 2 3 4 5}"
FREE_MEM_MB="${FREE_MEM_MB:-2000}"   # a GPU counts as free if used mem < this

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 999999; }
find_free_gpu() {
  for g in $GPUS; do
    [ "$(gpu_used "$g")" -lt "$FREE_MEM_MB" ] && { echo "$g"; return 0; }
  done
  return 1
}

for L in $LANGS; do
  g=""
  while [ -z "$g" ]; do g=$(find_free_gpu) || { sleep 20; g=""; }; done
  echo "[driver] launching mono $L on GPU $g"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" MONO_LANG="$L" \
    setsid bash scripts/launch/_launch_perlang_mono_vanilla.sh </dev/null >/dev/null 2>&1 &
  # wait until this run actually claims the GPU, so the next lang doesn't double-book it
  for _ in $(seq 1 40); do
    [ "$(gpu_used "$g")" -ge "$FREE_MEM_MB" ] && break
    sleep 6
  done
done
echo "[driver] all ${LANGS} launched; watch checkpoints/mono_vanilla_*/train.log"
