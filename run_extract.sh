#!/usr/bin/env bash
# Chỉ chạy bước trích feature (extract v1 + v2) cho nhiều ngôn ngữ x model.
# Feature không phụ thuộc seed nên chỉ cần chạy 1 lần / (lang, model).
# Chạy lại đúng lệnh này nếu bị ngắt: phần nào đã xong sẽ được bỏ qua / resume.
# Đổi dataset:  DATASET=triviaqa bash run_extract.sh
# Sau đó chạy run_all.sh như bình thường: nó sẽ tự bỏ qua phần extract đã xong.
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${DATASET:-triviaqa}
LANGS=(en fr vi bn)
MODELS=(mistral-7b llama2-7b)
BS=2
LOGDIR=logs/multi
mkdir -p "$LOGDIR"

done_ok() { [ -f "$1" ] && grep -q "$2" "$1"; }

for LG in "${LANGS[@]}"; do
for M in "${MODELS[@]}"; do
  echo "################ $DATASET | LANG: $LG | MODEL: $M ################"
  B="python run_pipeline.py --dataset $DATASET --lang $LG --model $M"
  PRE="$LOGDIR/${DATASET}_${LG}_${M}"

  # ---- Trích feature v1: chỉ 1 lần / (lang, model) ----
  if [ ! -f "results/features/X_seq_${DATASET}_${LG}_${M}.npy" ]; then
    echo "[extract v1] $LG $M"
    $B --seed 1 --stage extract --extract_batch_size $BS 2>&1 | tee "${PRE}_extract_v1.log"
  else
    echo "[extract v1] $LG $M: đã có, bỏ qua"
  fi

  # ---- Trích feature v2 ----
  if ! done_ok "${PRE}_extract_v2.log" "Stage 1 (v2) completed"; then
    echo "[extract v2] $LG $M (bị ngắt thì chạy lại để resume)"
    $B --seed 1 --feat v2 --stage extract --prompt_mode verify --views last,amean \
       --extract_batch_size $BS 2>&1 | tee "${PRE}_extract_v2.log"
  else
    echo "[extract v2] $LG $M: đã xong, bỏ qua"
  fi
done
done

echo "################ EXTRACT XONG ################"
