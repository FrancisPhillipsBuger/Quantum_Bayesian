#!/bin/bash
# Train one BayesianFramework variant over the dataset job list; stops at the first failure.
# Usage: ./run_train.sh [baseline | quantum | quantum_kernel | quantum_reupload]
# Env:   CONFIG_DIR, DATA_ROOT, RESULT_BASE, RESULT_SUFFIX
set -e

cd "$(dirname "$0")"
source ./run_common.sh

init_env

VARIANT="${1:-baseline}"
RESULT_ROOT="$(result_root "${VARIANT}")" || exit 1

mkdir -p "${RESULT_ROOT}"
LOG_FILE="${RESULT_ROOT}/run_train_${VARIANT}_$(date +'%Y%m%d_%H%M%S').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Jobs "name|base_dir", run in order; results go to RESULT_ROOT/name
JOBS=(
  "adsorbed|${DATA_ROOT}/is2res_by_adsorbate/adsorbed"
  "clean|${DATA_ROOT}/is2res_by_adsorbate/clean"
  "is2re-total|${DATA_ROOT}/is2res_total_train_val_test_lmdbs/data/oc22/is2re-total"
)

NAMES=()
for job in "${JOBS[@]}"; do NAMES+=("${job%%|*}"); done

start=$(date +'%s')
print_header "single variant" "${VARIANT}" "${NAMES[*]}"

i=0
for job in "${JOBS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r NAME BASE_DIR <<< "${job}"
    require_dir "Dataset" "${BASE_DIR}"
    run_job "${i}" "${#JOBS[@]}" "${VARIANT}" "${NAME}" "${BASE_DIR}" "${RESULT_ROOT}/${NAME}" \
        || { print_footer "${start}"; exit 1; }
done

print_footer "${start}"
