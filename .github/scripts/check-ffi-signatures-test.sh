#!/usr/bin/env bash
# Fixture test for check-ffi-signatures.py: each case copies the repository's
# csrc/ferrompi.h and src/ffi.rs into a scratch directory, applies one sed
# edit to one of them, and checks whether the check flags the pair and names
# the function. A case whose edit no longer applies fails. No network.
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly CHECK="${SCRIPT_DIR}/check-ffi-signatures.py"
readonly TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FAIL_COUNT=0

# run_check [UNBOUND_NAME]: the check on the scratch copies; with a name, run
# as if UNBOUND_PROTOTYPES listed it.
run_check() {
  if [[ $# -eq 0 ]]; then
    python3 "$CHECK" "$TMP/ferrompi.h" "$TMP/ffi.rs"
  else
    python3 - "$CHECK" "$1" "$TMP/ferrompi.h" "$TMP/ffi.rs" <<'EOF'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("check_ffi_signatures", sys.argv[1])
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
check.UNBOUND_PROTOTYPES = {sys.argv[2]: "fixture"}
sys.exit(check.main(sys.argv[3:]))
EOF
  fi
}

# check <flagged|clean> <h|rs|-> <sed script> <output fragment> [UNBOUND_NAME]
check() {
  local want="$1" which="$2" edit="$3" fragment="$4" got rc=0 out target
  shift 4
  cp "$REPO_ROOT/csrc/ferrompi.h" "$TMP/ferrompi.h"
  cp "$REPO_ROOT/src/ffi.rs" "$TMP/ffi.rs"
  if [[ "$which" != - ]]; then
    target="$TMP/ferrompi.h"
    [[ "$which" == rs ]] && target="$TMP/ffi.rs"
    sed -E -e "$edit" "$target" > "$TMP/edited"
    if cmp -s "$target" "$TMP/edited"; then
      echo "FAIL: the edit no longer applies to $which: $edit" >&2
      FAIL_COUNT=$((FAIL_COUNT + 1))
      return
    fi
    mv "$TMP/edited" "$target"
  fi
  out=$(run_check "$@" 2>&1) || rc=$?
  case "$rc" in
    0) got=clean ;;
    1) got=flagged ;;
    *) got="exit $rc" ;;
  esac
  if [[ "$got" == "$want" && "$out" == *"$fragment"* ]]; then
    if [[ "$want" == clean ]]; then
      echo "PASS: clean: ${edit:-the files as they are}${1:+ (unbound: $1)}"
    else
      echo "PASS: flagged: $fragment${1:+ (unbound: $1)}"
    fi
  else
    echo "FAIL: expected $want naming '$fragment', got $got: $edit" >&2
    printf '%s\n' "$out" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

check clean - '' 'OK: '

# The deliberately changed declaration: a communicator handle widened in Rust.
check flagged rs 's/fn ferrompi_barrier\(comm: int32_t\)/fn ferrompi_barrier(comm: int64_t)/' \
  'ferrompi_barrier: parameter 1: '
check flagged rs 's/fn ferrompi_ibarrier\(comm: int32_t, request: \*mut int64_t\)/fn ferrompi_ibarrier(comm: int32_t)/' \
  'ferrompi_ibarrier: 1 parameter, '
check flagged rs 's/(fn ferrompi_startall\(count: int64_t, requests: )\*const/\1*mut/' \
  'ferrompi_startall: parameter 2: '
check flagged rs 's/(fn ferrompi_buffer_detach\(buffer: )\*mut \*mut c_void/\1*mut c_void/' \
  'ferrompi_buffer_detach: parameter 1: '
check flagged rs 's/(fn ferrompi_wait\(request: int64_t, status: \*mut )FerrompiStatus/\1int32_t/' \
  'ferrompi_wait: parameter 2: '
check flagged rs 's/(fn ferrompi_max_stringtag_len\(\) -> )int64_t/\1c_int/' \
  'ferrompi_max_stringtag_len: returns `c_int`'
check flagged rs 's/(fn ferrompi_win_fence_mode_values\(out: \*mut int32_t\));/\1 -> c_int;/' \
  'ferrompi_win_fence_mode_values: returns `c_int`'
# int is c_int only, and int32_t never c_int, although both are 32 bits wide.
check flagged rs 's/(fn ferrompi_error_class_index\(error_class: )c_int/\1int32_t/' \
  'ferrompi_error_class_index: parameter 1: '
check flagged rs 's/(fn ferrompi_comm_free\(comm: )int32_t/\1c_int/' \
  'ferrompi_comm_free: parameter 1: '
# The #[link_name] extern, a feature-gated extern block, a feature-gated and a
# completion entry of guarded_extern!.
check flagged rs '/link_name = "ferrompi_error_info"/,/;/ s/code: c_int/code: int64_t/' \
  'ferrompi_error_info: parameter 1: '
check flagged rs 's/(fn ferrompi_win_free\(win: )int32_t/\1int64_t/' \
  'ferrompi_win_free: parameter 1: '
check flagged rs 's/(fn ferrompi_win_flush\(rank: int32_t, win: )int32_t/\1int64_t/' \
  'ferrompi_win_flush: parameter 2: '
check flagged rs 's/(fn ferrompi_win_unlock_all\(win: )int32_t/\1int64_t/' \
  'ferrompi_win_unlock_all: parameter 1: '
check flagged rs 's/fn ferrompi_barrier\(comm: int32_t\)/fn ferrompi_barrier(comm: usize)/' \
  'ferrompi_barrier: cannot read its types'
check flagged rs '$a\
extern "C" { pub fn ferrompi_new_call(comm: int32_t) -> c_int; }' \
  'ferrompi_new_call: no prototype in '
check clean rs 's/fn ferrompi_barrier\(comm: int32_t\)/fn ferrompi_barrier(communicator: int32_t)/' \
  'OK: '

# Header edits: a prototype over several lines, one with no extern, and ones
# the check cannot read.
check flagged h '/^int ferrompi_send\(/,/\);/ s/int64_t count/int32_t count/' \
  'ferrompi_send: parameter 2: '
check flagged h '$a\
int ferrompi_new_call(int32_t comm);' \
  'ferrompi_new_call: no extern declaration in '
check flagged h 's/^int ferrompi_barrier\(int32_t comm\);/int ferrompi_barrier(size_t comm);/' \
  'ferrompi_barrier: cannot read its types'
check flagged h 's/^int ferrompi_barrier\(int32_t comm\);/int ferrompi_barrier(int (*cb)(int));/' \
  'ferrompi_barrier: cannot read this declaration'
check clean h 's/^(int ferrompi_group_incl\(int32_t group_handle, int32_t n, )const int32_t\* ranks/\1int32_t const* ranks/' \
  'OK: '
check clean h 's/^int ferrompi_barrier\(int32_t comm\);/int ferrompi_barrier(int32_t \/* handle *\/ comm);/' \
  'OK: '

# UNBOUND_PROTOTYPES: a listed prototype needs no extern, and a stale entry fails.
check clean h '$a\
int ferrompi_new_call(int32_t comm);' \
  'OK: ' ferrompi_new_call
check flagged - '' 'ferrompi_barrier: has an extern in ' ferrompi_barrier
check flagged - '' 'ferrompi_gone: in UNBOUND_PROTOTYPES, but ' ferrompi_gone

if [[ "$FAIL_COUNT" -ne 0 ]]; then
  echo "$FAIL_COUNT case(s) failed" >&2
  exit 1
fi
