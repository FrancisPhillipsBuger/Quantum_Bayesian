#!/bin/bash
# Train every BayesianFramework variant in turn on OC22 IS2RE-Total; failures do not stop the run.
# Usage: ./run_train_all_variants.sh
# Env:   CONFIG_DIR, DATA_ROOT, RESULT_BASE, RESULT_SUFFIX, VARIANTS="baseline quantum" (subset)
set -o pipefail

cd "$(dirname "$0")"
source ./run_common.sh

init_env

NAME="is2re-total"
BASE_DIR="${DATA_ROOT}/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total"
require_dir "Dataset" "${BASE_DIR}"

if [ -n "${VARIANTS}" ]; then
    read -r -a VARIANTS <<< "${VARIANTS}"
else
    VARIANTS=("${VARIANT_NAMES[@]}")
fi
for v in "${VARIANTS[@]}"; do result_root "${v}" > /dev/null || exit 1; done

mkdir -p "${RESULT_BASE}"
LOG_FILE="${RESULT_BASE}/run_train_all_variants_${NAME}_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

start=$(date +'%s')
print_header "all variants" "${VARIANTS[*]}" "${NAME}"

i=0
for VARIANT in "${VARIANTS[@]}"; do
    i=$((i + 1))
    RESULT_DIR="$(result_root "${VARIANT}")/${NAME}"
    mkdir -p "${RESULT_DIR}"
    run_job "${i}" "${#VARIANTS[@]}" "${VARIANT}" "${NAME}" "${BASE_DIR}" "${RESULT_DIR}" || true
done

print_footer "${start}"
[ "${N_FAIL}" -eq 0 ]
