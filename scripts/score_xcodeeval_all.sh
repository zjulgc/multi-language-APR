#!/bin/bash
# Sequentially (re)score every xCodeEval gen dir with ExecEval, now that /data
# has space and Rust/Go compile again. Overwrites the old results whose Rust/Go
# were corrupted by the full-disk ENOSPC. Sequential because one eval already
# uses --max_workers 32 == the ExecEval server's worker count. A clean-scored
# marker prevents re-scoring on restart. Waits for s1r16/s3r4 gen to finish.
set -u
cd /mnt/backup1/lgc/multi-language-APR
PY=/data/lgc/.conda/envs/peft/bin/python
UDB=instruction_dataset/xCodeEval/unittest_db.json
LOG=logs/score_xcodeeval.log
EXPECT=1100
ts(){ date '+%F %T'; }
NAMES=(base
  mono_vanilla_C mono_vanilla_Cpp mono_vanilla_Csharp mono_vanilla_Go mono_vanilla_Java
  mono_vanilla_Javascript mono_vanilla_Kotlin mono_vanilla_PHP mono_vanilla_Python
  mono_vanilla_Ruby mono_vanilla_Rust
  vanilla_lora_r64_perlang3k
  single_stage_perlang3k_e11r4_s1r16 single_stage_perlang3k_e11r4_s3r4)

echo "[$(ts)] xcodeeval scorer up (pid $$); ${#NAMES[@]} models, sequential" >> "$LOG"
while :; do
  for NAME in "${NAMES[@]}"; do
    MARK=results/.clean_scored_$NAME
    [ -e "$MARK" ] && continue
    GEN=eval_outputs/xcek100_$NAME; RES=results/xcek100_$NAME.json
    n=$(ls "$GEN"/*.json 2>/dev/null | grep -vc gen_done)
    if [ "$n" -lt "$EXPECT" ] && [ ! -e "$GEN/.gen_done" ]; then continue; fi   # gen not complete yet
    echo "[$(ts)] SCORE $NAME ($n files) -> $RES" >> "$LOG"
    $PY -u eval_xcodeeval_execeval.py --gen_dir "$GEN" --execeval_url http://localhost:5000 \
      --unittest_db "$UDB" --output_json "$RES" --max_workers 32 >> "$LOG" 2>&1
    if [ -e "$RES" ]; then touch "$MARK"; echo "[$(ts)] DONE  $NAME" >> "$LOG"
    else echo "[$(ts)] FAIL  $NAME (see log)" >> "$LOG"; fi
  done
  alldone=1; for NAME in "${NAMES[@]}"; do [ -e "results/.clean_scored_$NAME" ] || alldone=0; done
  [ "$alldone" -eq 1 ] && { echo "[$(ts)] all ${#NAMES[@]} scored; exiting" >> "$LOG"; break; }
  sleep 60
done
