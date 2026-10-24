#!/usr/bin/env bash
# Chạy base + v2 cho nhiều ngôn ngữ x model x seed, rồi tổng hợp AUC theo ngôn ngữ.
# Chạy lại đúng lệnh này nếu bị ngắt: phần nào đã xong (có log hoàn tất) sẽ được bỏ qua.
# Đổi dataset:  DATASET=triviaqa bash run_all.sh
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${DATASET:-halueval}
LANGS=(en fr vi bn)
MODELS=(mistral-7b llama2-7b)
SEEDS=(1 11 29 37 43)
BS=1
LOGDIR=logs/multi
mkdir -p "$LOGDIR"

done_ok() { [ -f "$1" ] && grep -q "$2" "$1"; }

for LG in "${LANGS[@]}"; do
for M in "${MODELS[@]}"; do
  echo "################ $DATASET | LANG: $LG | MODEL: $M ################"
  B="python run_pipeline.py --dataset $DATASET --lang $LG --model $M"
  PRE="$LOGDIR/${DATASET}_${LG}_${M}"

  # ---- Trích feature: chỉ 1 lần / (lang, model); feature không phụ thuộc seed ----
  if [ ! -f "results/features/X_seq_${DATASET}_${LG}_${M}.npy" ]; then
    echo "[extract v1] $LG $M"
    $B --seed 1 --stage extract --extract_batch_size $BS 2>&1 | tee "${PRE}_extract_v1.log"
  else
    echo "[extract v1] $LG $M: đã có, bỏ qua"
  fi

  if ! done_ok "${PRE}_extract_v2.log" "Stage 1 (v2) completed"; then
    echo "[extract v2] $LG $M (bị ngắt thì chạy lại để resume)"
    $B --seed 1 --feat v2 --stage extract --prompt_mode verify --views last,amean \
       --extract_batch_size $BS 2>&1 | tee "${PRE}_extract_v2.log"
  else
    echo "[extract v2] $LG $M: đã xong, bỏ qua"
  fi

  for S in "${SEEDS[@]}"; do
    P="$B --seed $S"

    # ---- A. BASE ----
    LOG="${PRE}_s${S}_base.log"
    if done_ok "$LOG" "Saved results to"; then
      echo "[base] $LG $M seed=$S: đã xong, bỏ qua"
    else
      echo "[base] $LG $M seed=$S"
      {
        $P --stage train_oof --tag base_s$S
        $P --stage ensemble  --tag base_s$S
      } 2>&1 | tee "$LOG"
    fi

    # ---- B. V2 ----
    LOG="${PRE}_s${S}_v2.log"
    if done_ok "$LOG" "Saved results to"; then
      echo "[v2] $LG $M seed=$S: đã xong, bỏ qua"
    else
      echo "[v2] $LG $M seed=$S"
      {
        $P --feat v2 --stage probe     --views last,amean
        $P --feat v2 --stage train_oof --improved --tag v2_s$S
        $P --feat v2 --stage ensemble  --improved --stacker lite --tag v2_s$S
      } 2>&1 | tee "$LOG"
    fi
  done
done
done

#echo "################ TỔNG HỢP ################"
#python summarize_results.py --logdir "$LOGDIR" --dataset "$DATASET" --out results/summary_${DATASET}
