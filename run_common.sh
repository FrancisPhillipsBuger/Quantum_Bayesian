#!/bin/bash
# Shared setup sourced by run_train.sh and run_train_all_variants.sh.

VARIANT_NAMES=(baseline quantum quantum_kernel reupload)
RULE="========================================================================"
N_OK=0
N_FAIL=0

# BLAS pinning, file-handle limit, PYTHONPATH, CONFIG_DIR, DATA_ROOT and RESULT_BASE
init_env() {
    ulimit -n 65535 2>/dev/null || true
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
    REPO_ROOT="$(pwd)"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
    CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/config_file}"
    DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset}"
    RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results}"
    require_dir "Config directory" "${CONFIG_DIR}"
}

# result_root <variant>: per-variant result root; RESULT_SUFFIX keeps runs apart
result_root() {
    local base="${RESULT_BASE:-$(pwd)/results}"
    case "$1" in
        baseline)       echo "${base}/result${RESULT_SUFFIX}" ;;
        quantum)        echo "${base}/result_quantum${RESULT_SUFFIX}" ;;
        quantum_kernel) echo "${base}/result_quantum_kernel${RESULT_SUFFIX}" ;;
        reupload)       echo "${base}/result_quantum_reupload${RESULT_SUFFIX}" ;;
        *) echo "[Error]      Unknown variant: $1 (expected: ${VARIANT_NAMES[*]})" >&2; return 1 ;;
    esac
}

# require_dir <label> <path>
require_dir() {
    if [ ! -d "$2" ]; then
        echo "[Error]      $1 not found: $2" >&2
        exit 1
    fi
}

# hms <seconds> -> HH:MM:SS
hms() {
    printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60))
}

# print_header <title> <variants> <datasets>
print_header() {
    echo "${RULE}"
    echo "BayesianFramework training: $1"
    echo "  Variants : $2"
    echo "  Datasets : $3"
    echo "  Config   : ${CONFIG_DIR}"
    echo "  Log file : ${LOG_FILE}"
    echo "  Started  : $(date +'%Y-%m-%d %H:%M:%S')"
    echo "${RULE}"
}

# run_job <i> <n> <variant> <dataset> <base_dir> <result_dir>; returns the Python exit code
run_job() {
    local t0 rc dt
    t0=$(date +'%s')
    echo
    echo "[$1/$2] $3 | $4"
    echo "  Data   : $5"
    echo "  Output : $6"
    python3 -m main_train train --variant "$3" --config_dir "${CONFIG_DIR}" \
        --base_dir "$5" --result_dir "$6"
    rc=$?
    dt=$(( $(date +'%s') - t0 ))
    if [ ${rc} -eq 0 ]; then
        N_OK=$((N_OK + 1))
        echo "[$1/$2] $3 | $4: completed in $(hms ${dt})"
    else
        N_FAIL=$((N_FAIL + 1))
        echo "[$1/$2] $3 | $4: FAILED (exit ${rc}) after $(hms ${dt})"
    fi
    return ${rc}
}

# print_footer <start epoch seconds>
print_footer() {
    echo
    echo "${RULE}"
    echo "Finished: ${N_OK} completed, ${N_FAIL} failed | Total time $(hms $(( $(date +'%s') - $1 )))"
    echo "${RULE}"
}
