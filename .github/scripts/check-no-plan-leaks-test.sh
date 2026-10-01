#!/usr/bin/env bash
# Fixture test for check-no-plan-leaks.sh: each case writes one line to a
# scratch file and checks whether the gate flags it. No network.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/docs/adr"

FAIL_COUNT=0

# check <flagged|clean> <relative path> <line>
check() {
  local want="$1" path="$2" line="$3" got
  printf '%s\n' "$line" > "$TMP/$path"
  if (cd "$TMP" && bash "$SCRIPT_DIR/check-no-plan-leaks.sh" "$path") > /dev/null; then
    got=clean
  else
    got=flagged
  fi
  if [[ "$got" == "$want" ]]; then
    echo "PASS: $want: $line"
  else
    echo "FAIL: expected $want, got $got: $line" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

check flagged src.rs '// no precedent in prior epics'
check flagged src.rs '// fixed by ticket-042'
check flagged src.rs 'untouched by this Ticket'
check flagged src.rs '// see Requirement 3a'
check flagged src.rs '// decision D-11 applies'
check flagged src.rs '// Fix 1: reorder the barrier'
check flagged src.rs '// AC: exits 0'
check flagged src.rs '// closes SND-16'
check flagged src.rs '// per D-7'
check flagged src.rs '// see ADR-042'
check flagged src.rs '// implements R76'
check flagged src.rs '// feature F2-002'
check flagged src.rs '// see plans/ferrompi-0.5.x-hardening'
check flagged src.rs '// see MEMORY.md'
check flagged src.rs '// Option A keeps the table'
check flagged src.rs '// the ordering rule in §5 applies'
check clean src.rs 'fn regression_ticket_042() {}'
check clean src.rs '/// See [ADR-0006](crate::doc::adr_0006_mpi5_abi_direction).'
check clean src.rs '// MPI-4.1 §7.13 orders persistent collective initialisation.'
check clean src.rs '// per MPI 4.0 §10.2 and C11 §6.7.4'
check clean src.rs 'let r: Option<i32> = None; // an MPI epoch'
check clean docs/adr/0009-x.md '### Option A (chosen): Hand-written wrapper'

if [[ "$FAIL_COUNT" -ne 0 ]]; then
  echo "$FAIL_COUNT case(s) failed" >&2
  exit 1
fi
