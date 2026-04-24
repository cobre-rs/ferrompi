#!/bin/bash
# ==========================================================================
# FerroMPI Integration Test Runner
#
# Builds all examples and runs each test example with mpiexec.
# Reports pass/fail for each and returns nonzero if any test fails.
#
# Usage:
#   ./tests/run_mpi_tests.sh               # Run default tests
#   ./tests/run_mpi_tests.sh rma           # Run with rma feature (includes shared_memory)
#   ./tests/run_mpi_tests.sh numa          # Run with numa feature (implies rma)
#   MPI_NP=8 ./tests/run_mpi_tests.sh      # Run with 8 processes
#
# Environment:
#   MPI_NP      — Number of MPI processes (default: 4)
#   MPIEXEC     — Path to mpiexec (default: mpiexec)
#   BUILD_MODE  — "debug" or "release" (default: debug)
#
# Prerequisites:
#   - MPICH 4.0+ or OpenMPI 5.0+ installed
#   - Rust toolchain installed
# ==========================================================================
set -euo pipefail

# Configuration
FEATURES="${1:-}"
NP="${MPI_NP:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"
BUILD_MODE="${BUILD_MODE:-debug}"

# Auto-detect OpenMPI and add --oversubscribe to avoid binding errors in CI
MPIEXEC_ARGS=""
if "$MPIEXEC" --version 2>&1 | grep -q "Open MPI"; then
    MPIEXEC_ARGS="--oversubscribe"
fi

PASSED=0
FAILED=0
SKIPPED=0
FAILED_TESTS=()

# Color codes (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    BOLD=''
    RESET=''
fi

# ========================================================================
# Helper functions
# ========================================================================

print_header() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}║       FerroMPI Integration Test Runner            ║${RESET}"
    echo -e "${BOLD}╚════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "  Processes:   ${CYAN}${NP}${RESET}"
    echo -e "  Features:    ${CYAN}${FEATURES:-default}${RESET}"
    echo -e "  Build mode:  ${CYAN}${BUILD_MODE}${RESET}"
    echo -e "  mpiexec:     ${CYAN}${MPIEXEC}${RESET}"
    echo -e "  mpiexec args:${CYAN}${MPIEXEC_ARGS:- (none)}${RESET}"
    echo ""
}

check_prerequisites() {
    if ! command -v "$MPIEXEC" &> /dev/null; then
        echo -e "${RED}ERROR: ${MPIEXEC} not found in PATH${RESET}"
        echo "Please install MPICH or OpenMPI first:"
        echo "  Ubuntu: sudo apt install mpich libmpich-dev"
        echo "  macOS:  brew install mpich"
        exit 1
    fi

    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}ERROR: cargo not found in PATH${RESET}"
        echo "Please install the Rust toolchain first."
        exit 1
    fi
}

build_examples() {
    echo -e "${BOLD}Building examples...${RESET}"
    local build_args=(--examples)

    if [ "$BUILD_MODE" = "release" ]; then
        build_args+=(--release)
    fi

    if [ -n "$FEATURES" ]; then
        build_args+=(--features "$FEATURES")
    fi

    if ! cargo build "${build_args[@]}" 2>&1; then
        echo -e "${RED}ERROR: Build failed${RESET}"
        exit 1
    fi
    echo -e "${GREEN}Build successful!${RESET}"
    echo ""
}

# Run a single test example
# Arguments: $1 = test name, $2 = number of processes (optional, defaults to NP)
run_test() {
    local name="$1"
    local procs="${2:-$NP}"
    local binary="./target/${BUILD_MODE}/examples/${name}"

    if [ ! -f "$binary" ]; then
        echo -e "  ${YELLOW}SKIP${RESET}  ${name} (binary not found)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo -n -e "  Running ${BOLD}${name}${RESET} (n=${procs})... "

    local output
    local exit_code=0
    # shellcheck disable=SC2086
    output=$("$MPIEXEC" $MPIEXEC_ARGS -n "$procs" "$binary" 2>&1) || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}PASS${RESET}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${RESET} (exit code: ${exit_code})"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$name")
        # Print output for failed tests to aid debugging
        echo "    --- output ---"
        echo "$output" | sed 's/^/    /'
        echo "    --- end ---"
    fi
}

# ========================================================================
# Main
# ========================================================================

print_header
check_prerequisites
build_examples

echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${BOLD}Running integration tests${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"

# Smoke test first (to verify MPI works)
run_test hello_world

# Core test examples
run_test test_lifecycle 2
run_test test_collectives
run_test test_blocking_extra
run_test test_nonblocking
run_test test_p2p_extra
run_test test_nonblocking_collectives
run_test test_waitany 4
run_test test_cancel 2
run_test test_persistent
run_test test_info 2
run_test test_comm_split 4
run_test test_errhandler_returns
run_test test_gather_inplace 4
run_test test_allgather_inplace 4
run_test test_scatter_inplace 4
run_test test_alltoall_inplace 4
run_test test_igather_inplace 4
run_test test_iallgather_inplace 4
run_test test_iscatter_inplace 4
run_test test_ialltoall_inplace 4
run_test test_gather_init_inplace 4
run_test test_allgather_init_inplace 4
run_test test_scatter_init_inplace 4
run_test test_alltoall_init_inplace 4
run_test test_allreduce_bytes 4
run_test test_error_context 2
run_test test_request_table_concurrency 2

echo ""
echo -e "${BOLD}────────────────────────────────────────${RESET}"
echo -e "${BOLD}Running existing examples as smoke tests${RESET}"
echo -e "${BOLD}────────────────────────────────────────${RESET}"

# Existing examples that serve as additional smoke tests
run_test ring
run_test allreduce
run_test nonblocking
run_test comm_split
run_test reduce_op_bitwise 2
run_test reduce_op_maxloc 4

# persistent_bcast may fail on MPI < 4.0 — run but don't count failure as fatal
echo ""
echo -n -e "  Running ${BOLD}persistent_bcast${RESET} (n=${NP}, MPI 4.0+)... "
# shellcheck disable=SC2086
PERSIST_OUTPUT=$("$MPIEXEC" $MPIEXEC_ARGS -n "$NP" "./target/${BUILD_MODE}/examples/persistent_bcast" 2>&1) || true
echo -e "${YELLOW}DONE${RESET} (MPI 4.0+ required; may show 'not available')"

# hybrid_openmp — run as smoke test
run_test hybrid_openmp 2

# RMA / shared memory tests (only if rma or numa feature is enabled)
if [[ "$FEATURES" == *"rma"* ]] || [[ "$FEATURES" == *"numa"* ]]; then
    echo ""
    echo -e "${BOLD}────────────────────────────────────────${RESET}"
    echo -e "${BOLD}Running RMA/shared memory tests${RESET}"
    echo -e "${BOLD}────────────────────────────────────────${RESET}"
    run_test test_rma_window
    run_test shared_memory
fi

# ========================================================================
# Summary
# ========================================================================
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════╗${RESET}"
TOTAL=$((PASSED + FAILED + SKIPPED))
if [ "$FAILED" -eq 0 ]; then
    echo -e "${BOLD}║${RESET}  ${GREEN}Results: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped (${TOTAL} total)${RESET}"
    echo -e "${BOLD}║${RESET}  ${GREEN}All tests passed!${RESET}"
else
    echo -e "${BOLD}║${RESET}  ${RED}Results: ${PASSED} passed, ${FAILED} failed, ${SKIPPED} skipped (${TOTAL} total)${RESET}"
    echo -e "${BOLD}║${RESET}  ${RED}Failed tests: ${FAILED_TESTS[*]}${RESET}"
fi
echo -e "${BOLD}╚════════════════════════════════════════════════════╝${RESET}"

exit "$FAILED"
