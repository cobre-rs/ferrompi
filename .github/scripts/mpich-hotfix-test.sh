#!/usr/bin/env bash
# Fixture test for verify() in mpich-hotfix.sh. Sources the script (its main
# is guarded, so sourcing only defines the functions) and never calls
# download() or install(): no network, no root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/mpich-hotfix.sh"

FAIL_COUNT=0

# assert_verify_fails <description> <dir>
assert_verify_fails() {
  local description="$1" dir="$2"
  if verify "$dir" >/dev/null 2>&1; then
    echo "FAIL: $description: verify exited 0, expected non-zero" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    echo "PASS: $description"
  fi
}

main() {
  local mismatch_dir empty_dir
  mismatch_dir=$(mktemp -d)
  empty_dir=$(mktemp -d)
  trap 'rm -rf "$mismatch_dir" "$empty_dir"' EXIT

  echo "not the real package" >"${mismatch_dir}/${LIBUCX_DEB}"
  echo "not the real package" >"${mismatch_dir}/${LIBMPICH_DEB}"
  assert_verify_fails "checksum mismatch" "$mismatch_dir"

  assert_verify_fails "missing files (empty dir)" "$empty_dir"

  if ((FAIL_COUNT > 0)); then
    exit 1
  fi
  exit 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
