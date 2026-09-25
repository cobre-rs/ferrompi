#!/usr/bin/env bash
# ==========================================================================
# Self-test for tests/run_mpi_tests.sh's decision logic.
#
# Sources the runner (its main guard keeps this side-effect free) and checks
# parse_directive, expand_np, impl_id, feature_closure, classify,
# artifact_outcome and discover against fixed fixtures. Needs no MPI, no
# cargo and no network.
#
# Prints "ok <case>" or "not ok <case>: <got> != <want>" per case and exits
# 1 if any case failed.
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/run_mpi_tests.sh
source "$SCRIPT_DIR/run_mpi_tests.sh"
set +e
set -uo pipefail

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

# --- parse_directive -----------------------------------------------------

check "parse_directive: full directive" \
  "$(parse_directive "np=2.. timeout=30 skip-ok=mpich-4.2,openmpi")" \
  "2..|30|mpich-4.2,openmpi||"

rc=0
parse_directive "np=2 frobnicate=1" >/dev/null 2>&1 || rc=$?
check "parse_directive: unknown key rejected" "$rc" "1"

rc=0
parse_directive "np=1 expect=never" >/dev/null 2>&1 || rc=$?
check "parse_directive: invalid expect rejected" "$rc" "1"

check "parse_directive: valgrind flag" \
  "$(parse_directive "np=1 valgrind")" \
  "1||||1"

# --- discover: directive-level rejections ---------------------------------

d=$(mk_dir)
cat >"$d/example_abort_no_stderr.rs" <<'EOF'
// mpi-test: np=1 expect=abort
fn main() {}
EOF
rc=0
(discover "$d") >/dev/null 2>&1 || rc=$?
check "discover: expect=abort without stderr line rejected" "$rc" "2"

d=$(mk_dir)
cat >"$d/example_unfin_no_stderr.rs" <<'EOF'
// mpi-test: np=1 expect=unfinalized
fn main() {}
EOF
rc=0
(discover "$d") >/dev/null 2>&1 || rc=$?
check "discover: expect=unfinalized np=1 without stderr line rejected" "$rc" "2"

d=$(mk_dir)
cat >"$d/example_unfin_np2.rs" <<'EOF'
// mpi-test: np=2 expect=unfinalized
// mpi-test-stderr: done
fn main() {}
EOF
rc=0
(discover "$d") >/dev/null 2>&1 || rc=$?
check "discover: expect=unfinalized requires np=1 rejected" "$rc" "2"

# --- discover: file-grammar rejections ------------------------------------

d=$(mk_dir)
cat >"$d/test_x.rs" <<'EOF'
fn main() {}
EOF
out=$( (discover "$d") 2>&1 )
rc=$?
check "discover: missing directive exit code" "$rc" "2"
check "discover: missing directive names test_x.rs" "$(contains "$out" "test_x.rs")" "yes"

d=$(mk_dir)
cat >"$d/example_two_stderr.rs" <<'EOF'
// mpi-test: np=1
// mpi-test-stderr: foo
// mpi-test-stderr: bar
fn main() {}
EOF
rc=0
(discover "$d") >/dev/null 2>&1 || rc=$?
check "discover: two stderr lines rejected" "$rc" "2"

d=$(mk_dir)
cat >"$d/example_orphan_stderr.rs" <<'EOF'
// mpi-test-stderr: foo
fn main() {}
EOF
rc=0
(discover "$d") >/dev/null 2>&1 || rc=$?
check "discover: stderr line without directive rejected" "$rc" "2"

# --- expand_np -------------------------------------------------------------

check "expand_np: 2.. matches available" "$(expand_np "2.." 2 3 4)" "2 3 4"
check "expand_np: 3.. falls back to base" "$(expand_np "3.." 2)" "3"
check "expand_np: fixed np ignores list" "$(expand_np "4" 2 3)" "4"

# --- valgrind_np -------------------------------------------------------

check "valgrind_np: np=N.. gives minimal N" "$(valgrind_np "2..")" "2"
check "valgrind_np: fixed np unchanged" "$(valgrind_np "4")" "4"

# --- build_cmd -----------------------------------------------------------

check "build_cmd: exact prefix" \
  "$(build_cmd "/repo/tests/valgrind/mpich.supp")" \
  "valgrind -q --error-exitcode=99 --track-origins=yes --leak-check=no --suppressions=/repo/tests/valgrind/mpich.supp"

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

# --- feature_closure -------------------------------------------------------

check "feature_closure: numa pulls in rma" \
  "$(feature_closure "numa" '{"numa":["rma"],"rma":[]}')" "numa,rma"

# --- artifact_outcome (the missing-artifact replay) -----------------------

check "artifact_outcome: required feature enabled -> missing binary" \
  "$(artifact_outcome "rma" "rma,numa")" "FAIL missing binary"
check "artifact_outcome: required feature outside closure -> skip" \
  "$(artifact_outcome "numa" "rma")" "SKIP(feature)"

if ((FAILED)); then
  exit 1
fi
exit 0
