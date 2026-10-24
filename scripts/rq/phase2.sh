#!/usr/bin/env bash
# RQ1 phase 2, GPU instance only: experiment G (answer form) together with analysis B.
# For each model and each answer form (as is, strip, stop): stage 1 once, then per seed and
# variant stages 2 to 4 and the RQ1 analysis (--stage rq, one row per run in results/rq.csv).
# "as is" is the baseline: same analysis without G.
#
# Usage: scripts/rq/phase2.sh [lang]         (default en)
#        MODELS="mistral-7b" SEEDS="1 2" VARIANTS="base" scripts/rq/phase2.sh en
# Time (float16 reference, NF4 not measured): stage 1 about 5.5 min per model and form in
# English, then about 5 min per run: 2 models x 3 forms x 5 seeds x 2 variants = 60 runs, ~5 h.
set -euo pipefail
cd "$(dirname "$0")/../.."

lang=${1:-en}
for model in ${MODELS:-mistral-7b llama2-7b}; do
    for form in as_is strip stop; do
        form_opt=() suffix=""
        if [ "$form" != as_is ]; then form_opt=(--answer_form "$form") suffix="_$form"; fi
        common=(--dataset halueval --lang "$lang" --model "$model" "${form_opt[@]}")
        if [ ! -f "results/features/y_halueval_${lang}_${model}${suffix}.npy" ]; then
            ./run.sh "${common[@]}" --stage extract
        fi
        for variant in ${VARIANTS:-base improved}; do
            variant_opt=()
            if [ "$variant" = improved ]; then variant_opt=(--improved); fi
            for seed in ${SEEDS:-1 2 3 4 5}; do
                run=("${common[@]}" "${variant_opt[@]}" --seed "$seed" --tag "${variant}_s${seed}")
                ./run.sh "${run[@]}" --stage train_oof
                ./run.sh "${run[@]}" --stage ensemble
                ./run.sh "${run[@]}" --stage rq
            done
        done
    done
done
echo "Done. Read results/rq.csv: columns B_strat_*, C_*, verdict; compare the forms per model."
