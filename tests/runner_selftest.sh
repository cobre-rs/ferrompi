#!/usr/bin/env bash
# ==========================================================================
# Self-test for tests/run_mpi_tests.sh's decision logic.
#
# Sources the runner (its main guard keeps this side-effect free) and checks
# parse_directive, expand_np, impl_id, feature_closure, classify,
# lib_debug_assertions, lib_opt_level, artifact_outcome, discover and the
# mode-flag refusal of main against fixed fixtures. Needs no MPI, no cargo and
# no network.
#
# Prints "ok <case>" or "not ok <case>: <got> != <want>" per case and exits
# 1 if any case failed.
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/run_mpi_tests.sh
source "$SCRIPT_DIR/run_mpi_tests.sh"
set -euo pipefail

SELFTEST_TMP=$(mktemp -d)
trap 'rm -rf "$SELFTEST_TMP"' EXIT

FAILED=0

check() {
  local name="$1" got="$2" want="$3"
  if [[ "$got" == "$want" ]]; then
    echo "ok $name"
  else
    echo "not ok $name: $got != $want"
    FAILED=1
  fi
}

contains() {
  if [[ "$1" == *"$2"* ]]; then
    echo yes
  else
    echo no
  fi
}

mk_outfile() {
  local f
  f=$(mktemp -p "$SELFTEST_TMP")
  printf '%s' "$1" >"$f"
  printf '%s' "$f"
}

mk_dir() {
  mktemp -d -p "$SELFTEST_TMP"
}

# check_discover_rejects <case> <dir>
# Checks that discover exits 2 on <dir>.
check_discover_rejects() {
  local rc=0
  (discover "$2") >/dev/null 2>&1 || rc=$?
  check "$1" "$rc" "2"
}

# --- parse_directive -----------------------------------------------------

check "parse_directive: full directive" \
  "$(parse_directive "np=2.. timeout=30 skip-ok=mpich-4.2,openmpi")" \
  "2..|30|mpich-4.2,openmpi|||"

rc=0
parse_directive "np=2 frobnicate=1" >/dev/null 2>&1 || rc=$?
check "parse_directive: unknown key rejected" "$rc" "1"

rc=0
parse_directive "np=1 expect=never" >/dev/null 2>&1 || rc=$?
check "parse_directive: invalid expect rejected" "$rc" "1"

check "parse_directive: valgrind flag" \
  "$(parse_directive "np=1 valgrind")" \
  "1||||1|"

check "parse_directive: release flag" \
  "$(parse_directive "np=2 release")" \
  "2|||||1"

check "parse_directive: release with valgrind" \
  "$(parse_directive "np=1 valgrind release")" \
  "1||||1|1"

# --- discover: directive-level rejections ---------------------------------

d=$(mk_dir)
cat >"$d/example_abort_no_stderr.rs" <<'EOF'
// mpi-test: np=1 expect=abort
fn main() {}
EOF
check_discover_rejects "discover: expect=abort without stderr line rejected" "$d"

d=$(mk_dir)
cat >"$d/example_unfin_no_stderr.rs" <<'EOF'
// mpi-test: np=1 expect=unfinalized
fn main() {}
EOF
check_discover_rejects "discover: expect=unfinalized np=1 without stderr line rejected" "$d"

d=$(mk_dir)
cat >"$d/example_unfin_np2.rs" <<'EOF'
// mpi-test: np=2 expect=unfinalized
// mpi-test-stderr: done
fn main() {}
EOF
rc=0
out=$( (discover "$d"; echo "${DIRECTIVE_NP[example_unfin_np2]}|${DIRECTIVE_EXPECT[example_unfin_np2]}") 2>&1 ) || rc=$?
check "discover: expect=unfinalized accepted at np=2" "$rc|$out" "0|2|unfinalized"

d=$(mk_dir)
cat >"$d/example_release.rs" <<'EOF'
// mpi-test: np=2.. timeout=30 release
fn main() {}
EOF
cat >"$d/example_plain.rs" <<'EOF'
// mpi-test: np=2
fn main() {}
EOF
rc=0
out=$( (discover "$d"; echo "${DIRECTIVE_RELEASE[example_release]}|${DIRECTIVE_RELEASE[example_plain]}") 2>&1 ) || rc=$?
check "discover: release flag stored per example" "$rc|$out" "0|1|"

# --- discover: file-grammar rejections ------------------------------------

d=$(mk_dir)
cat >"$d/test_x.rs" <<'EOF'
fn main() {}
EOF
rc=0
out=$( (discover "$d") 2>&1 ) || rc=$?
check "discover: missing directive exit code" "$rc" "2"
check "discover: missing directive names test_x.rs" "$(contains "$out" "test_x.rs")" "yes"

d=$(mk_dir)
cat >"$d/example_two_stderr.rs" <<'EOF'
// mpi-test: np=1
// mpi-test-stderr: foo
// mpi-test-stderr: bar
fn main() {}
EOF
check_discover_rejects "discover: two stderr lines rejected" "$d"

d=$(mk_dir)
cat >"$d/example_orphan_stderr.rs" <<'EOF'
// mpi-test-stderr: foo
fn main() {}
EOF
check_discover_rejects "discover: stderr line without directive rejected" "$d"

d=$(mk_dir)
cat >"$d/example_two_directives.rs" <<'EOF'
// mpi-test: np=1
// mpi-test: np=2
fn main() {}
EOF
check_discover_rejects "discover: two mpi-test lines rejected" "$d"

# --- expand_np -------------------------------------------------------------

check "expand_np: 2.. matches available" "$(expand_np "2.." 2 3 4)" "2 3 4"
check "expand_np: 3.. falls back to base" "$(expand_np "3.." 2)" "3"
check "expand_np: fixed np ignores list" "$(expand_np "4" 2 3)" "4"

# --- impl_id -----------------------------------------------------------

hydra_sample=$'HYDRA build details:\n    Version:                                 4.2.3\n    Release date:                            unreleased'
check "impl_id: mpich hydra sample" "$(impl_id "$hydra_sample")" "mpich-4.2.3"
check "impl_id: open mpi" "$(impl_id "mpiexec (Open MPI) 5.0.7")" "openmpi-5.0.7"
check "impl_id: openrte" "$(impl_id "mpiexec (OpenRTE) 4.1.6")" "openmpi-4.1.6"

# --- classify ------------------------------------------------------------

f=$(mk_outfile $'normal output\n')
check "classify: exit 0, no SKIP -> PASS" "$(classify 0 "$f" "" "" "" "unknown")" "PASS"

f=$(mk_outfile $'SKIP: x\n')
check "classify: unregistered SKIP, no skip-ok" \
  "$(classify 0 "$f" "" "" "" "unknown")" "FAIL unregistered SKIP: x"

f=$(mk_outfile $'SKIP: x\n')
check "classify: SKIP registered by impl prefix" \
  "$(classify 0 "$f" "openmpi" "" "" "openmpi-5.0.7")" "SKIP x"

f=$(mk_outfile $'SKIP: x\n')
check "classify: SKIP unregistered, impl prefix mismatch" \
  "$(classify 0 "$f" "openmpi-4" "" "" "openmpi-5.0.7")" "FAIL unregistered SKIP: x"

f=$(mk_outfile "")
check "classify: exit 124 -> FAIL timeout" "$(classify 124 "$f" "" "" "" "unknown")" "FAIL timeout"
check "classify: exit 1 -> FAIL exit 1" "$(classify 1 "$f" "" "" "" "unknown")" "FAIL exit 1"

f=$(mk_outfile $'some output\n')
check "classify: exit 99 -> FAIL valgrind errors" \
  "$(classify 99 "$f" "" "" "" "unknown")" "FAIL valgrind errors"

check "classify: valgrind runs with the exit code classify reports" \
  "$(contains " ${VALGRIND_ARGS[*]} " " --error-exitcode=99 ")" "yes"

f=$(mk_outfile $'shutdown complete\n')
check "classify: exit 99 with expect=unfinalized and literal present still FAIL valgrind errors" \
  "$(classify 99 "$f" "" "unfinalized" "shutdown" "unknown")" "FAIL valgrind errors"

f=$(mk_outfile $'some output\n')
check "classify: expect=abort, exit 0" \
  "$(classify 0 "$f" "" "abort" "boom" "unknown")" "FAIL exited 0, abort expected"

f=$(mk_outfile $'thread \'main\' panicked: boom\n')
check "classify: expect=abort, exit 134, literal present" \
  "$(classify 134 "$f" "" "abort" "boom" "unknown")" "PASS"

f=$(mk_outfile $'no marker here\n')
check "classify: expect=abort, exit 134, literal absent" \
  "$(classify 134 "$f" "" "abort" "boom" "unknown")" "FAIL abort marker missing"

f=$(mk_outfile $'boom\nFAIL: something returned instead of aborting\n')
check "classify: expect=abort, exit 134, literal present, FAIL: line present" \
  "$(classify 134 "$f" "" "abort" "boom" "unknown")" "FAIL output contains a FAIL: line"

f=$(mk_outfile $'SKIP: x\n')
check "classify: expect=abort, exit 0, registered SKIP" \
  "$(classify 0 "$f" "openmpi-4" "abort" "boom" "openmpi-4.1.6")" "SKIP x"

f=$(mk_outfile $'SKIP: x\n')
check "classify: expect=abort, exit 0, unregistered SKIP" \
  "$(classify 0 "$f" "" "abort" "boom" "mpich-4.2.3")" "FAIL unregistered SKIP: x"

f=$(mk_outfile $'please warn user\n')
check "classify: exit 0, no SKIP, literal present" \
  "$(classify 0 "$f" "" "" "warn" "unknown")" "PASS"

f=$(mk_outfile $'nothing to see\n')
check "classify: exit 0, no SKIP, literal absent" \
  "$(classify 0 "$f" "" "" "warn" "unknown")" "FAIL stderr marker missing"

f=$(mk_outfile $'SKIP: x\n')
check "classify: SKIP short-circuits literal check" \
  "$(classify 0 "$f" "openmpi" "" "warn" "openmpi-5.0.7")" "SKIP x"

f=$(mk_outfile $'shutdown complete\n')
check "classify: expect=unfinalized, exit 0, literal present" \
  "$(classify 0 "$f" "" "unfinalized" "shutdown" "unknown")" "PASS"
check "classify: expect=unfinalized, exit 1, literal present" \
  "$(classify 1 "$f" "" "unfinalized" "shutdown" "unknown")" "PASS"

f=$(mk_outfile $'nothing\n')
check "classify: expect=unfinalized, exit 1, literal absent" \
  "$(classify 1 "$f" "" "unfinalized" "shutdown" "unknown")" "FAIL stderr marker missing"

f=$(mk_outfile $'shutdown complete\n')
check "classify: expect=unfinalized, exit 101" \
  "$(classify 101 "$f" "" "unfinalized" "shutdown" "unknown")" "FAIL exit 101"

f=$(mk_outfile $'ferrompi: MPI_Finalize skipped: 1 window(s) still alive\n')
check "classify: exit 0, finalize skipped without expect=unfinalized" \
  "$(classify 0 "$f" "" "" "" "unknown")" "FAIL MPI_Finalize skipped without expect=unfinalized"

f=$(mk_outfile $'SKIP: x\nferrompi: MPI_Finalize skipped: 1 window(s) still alive\n')
check "classify: finalize skipped fails a registered SKIP" \
  "$(classify 0 "$f" "openmpi" "" "" "openmpi-5.0.7")" "FAIL MPI_Finalize skipped without expect=unfinalized"

f=$(mk_outfile $'ferrompi: MPI_Finalize skipped: 1 window(s) still alive\n')
check "classify: expect=unfinalized, finalize skipped line present, literal matches" \
  "$(classify 1 "$f" "" "unfinalized" "MPI_Finalize skipped" "unknown")" "PASS"

check "finalize-skipped marker matches the crate's warning" \
  "$(grep -qF -- "$FINALIZE_SKIPPED_MARKER" "$SCRIPT_DIR/../src/lib.rs" && echo yes || echo no)" \
  "yes"

# --- classify: a FAIL line fails whatever the exit status ---------------------
# mpiexec on some MPICH stacks returns 0 although every rank exited nonzero.

fail_line_outcome="FAIL output contains a FAIL: line"

f=$(mk_outfile $'FAIL: some check\n')
check "classify: normal, anchored FAIL: line, exit 0" \
  "$(classify 0 "$f" "" "" "" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'rank 1: FAIL: Win::put failed\n')
check "classify: normal, rank N: FAIL: line, exit 0" \
  "$(classify 0 "$f" "" "" "" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'rank 1: FAIL Test 2: expected 4, got 3\n')
check "classify: normal, rank N: FAIL Test N line (no colon), exit 0" \
  "$(classify 0 "$f" "" "" "" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'PASS: first\nFAIL: second\nPASS: third\n')
check "classify: normal, PASS lines do not mask a FAIL: line" \
  "$(classify 0 "$f" "" "" "" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'warn: noted\nFAIL: some check\n')
check "classify: normal, stderr literal present does not mask a FAIL: line" \
  "$(classify 0 "$f" "" "" "warn" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'SKIP: x\nFAIL: some check\n')
check "classify: registered SKIP does not mask a FAIL: line" \
  "$(classify 0 "$f" "openmpi" "" "" "openmpi-5.0.7")" "$fail_line_outcome"

f=$(mk_outfile $'rank 0: FAIL: some check\nSKIP: x\n')
check "classify: registered SKIP after a rank N: FAIL: line" \
  "$(classify 0 "$f" "openmpi" "" "" "openmpi-5.0.7")" "$fail_line_outcome"

f=$(mk_outfile $'shutdown complete\nFAIL: some check\n')
check "classify: expect=unfinalized, FAIL: line, exit 0" \
  "$(classify 0 "$f" "" "unfinalized" "shutdown" "unknown")" "$fail_line_outcome"
check "classify: expect=unfinalized, FAIL: line, exit 1" \
  "$(classify 1 "$f" "" "unfinalized" "shutdown" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'shutdown complete\nrank 0: FAIL: some check\n')
check "classify: expect=unfinalized, rank N: FAIL: line, exit 0" \
  "$(classify 0 "$f" "" "unfinalized" "shutdown" "unknown")" "$fail_line_outcome"

# Two ranks that skip MPI_Finalize each print the marker. Open MPI exits 1 for
# that, and Open MPI and MPICH can turn a failing rank's exit 2 into 0 or 1, so
# the FAIL line is what fails a failing rank; a status of 2 or more fails on its
# own.
f=$(mk_outfile $'skipped\nskipped\n')
check "classify: expect=unfinalized, two ranks, exit 0, literal present" \
  "$(classify 0 "$f" "" "unfinalized" "skipped" "unknown")" "PASS"
check "classify: expect=unfinalized, two ranks, exit 1, literal present" \
  "$(classify 1 "$f" "" "unfinalized" "skipped" "unknown")" "PASS"
check "classify: expect=unfinalized, two ranks, a rank's exit 2" \
  "$(classify 2 "$f" "" "unfinalized" "skipped" "unknown")" "FAIL exit 2"

f=$(mk_outfile $'skipped\nskipped\nFAIL: rank 1 check\n')
check "classify: expect=unfinalized, two ranks, FAIL: line, exit 1" \
  "$(classify 1 "$f" "" "unfinalized" "skipped" "unknown")" "$fail_line_outcome"
check "classify: expect=unfinalized, two ranks, FAIL: line, exit 0" \
  "$(classify 0 "$f" "" "unfinalized" "skipped" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'boom\nrank 1: FAIL: returned instead of aborting\n')
check "classify: expect=abort, exit 134, literal present, rank N: FAIL: line" \
  "$(classify 134 "$f" "" "abort" "boom" "unknown")" "$fail_line_outcome"

f=$(mk_outfile $'SKIP: x\nFAIL: some check\n')
check "classify: expect=abort, exit 0, registered SKIP and a FAIL: line" \
  "$(classify 0 "$f" "openmpi-4" "abort" "boom" "openmpi-4.1.6")" "$fail_line_outcome"

f=$(mk_outfile $'FAIL: some check\n')
check "classify: exit 124 with a FAIL: line stays FAIL timeout" \
  "$(classify 124 "$f" "" "" "" "unknown")" "FAIL timeout"

f=$(mk_outfile $'ferrompi: MPI_ERR_PROC_FAILED_PENDING is not supported\nPASS: FAILURES=0 FAILED=0\n')
check "classify: FAIL inside a longer word is not a FAIL line" \
  "$(classify 0 "$f" "" "" "" "unknown")" "PASS"

# --- feature_closure -------------------------------------------------------

check "feature_closure: numa pulls in rma" \
  "$(feature_closure "numa" '{"numa":["rma"],"rma":[]}')" "numa,rma"

# --- lib_debug_assertions ---------------------------------------------------

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"ferrompi","kind":["lib"]},"profile":{"debug_assertions":true}}')
check "lib_debug_assertions: debug assertions on" \
  "$(lib_debug_assertions "$f")" "true"

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"ferrompi","kind":["lib"]},"profile":{"debug_assertions":false}}')
check "lib_debug_assertions: debug assertions off" \
  "$(lib_debug_assertions "$f")" "false"

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"hello_world","kind":["example"]},"profile":{"debug_assertions":true}}')
check "lib_debug_assertions: no library artifact" \
  "$(lib_debug_assertions "$f")" ""

# --- lib_opt_level ---------------------------------------------------------

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"ferrompi","kind":["lib"]},"profile":{"opt_level":"3"}}')
check "lib_opt_level: optimised" \
  "$(lib_opt_level "$f")" "3"

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"ferrompi","kind":["lib"]},"profile":{"opt_level":"0"}}')
check "lib_opt_level: unoptimised" \
  "$(lib_opt_level "$f")" "0"

f=$(mk_outfile '{"reason":"compiler-artifact","target":{"name":"hello_world","kind":["example"]},"profile":{"opt_level":"3"}}')
check "lib_opt_level: no library artifact" \
  "$(lib_opt_level "$f")" ""

# --- main: the mode flags are exclusive ------------------------------------

rc=0
out=$( (main --release --valgrind rma) 2>&1 >/dev/null ) || rc=$?
check "main: --release --valgrind exits 2" "$rc|$out" "2|ERROR: --release and --valgrind cannot be combined"

rc=0
out=$( (main --valgrind --release rma) 2>&1 >/dev/null ) || rc=$?
check "main: --valgrind --release exits 2" "$rc|$out" "2|ERROR: --release and --valgrind cannot be combined"

# --- artifact_outcome (the missing-artifact replay) -----------------------

check "artifact_outcome: required feature enabled -> missing binary" \
  "$(artifact_outcome "rma" "rma,numa")" "FAIL missing binary"
check "artifact_outcome: required feature outside closure -> skip" \
  "$(artifact_outcome "numa" "rma")" "SKIP(feature)"

exit "$FAILED"
