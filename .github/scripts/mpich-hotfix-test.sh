#!/usr/bin/env bash
# Fixture test for mpich-hotfix.sh. Sources the script (its main is guarded,
# so sourcing only defines the functions) and never calls the real download()
# or install(): no network, no root.
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

# assert_main_cleanup_survives_set_u
# Regression test: main()'s EXIT trap must not reference its own `local dir`
# after main() has returned, or it fails with "unbound variable" under
# `set -u` even when download/verify/install all succeeded. Runs main() in
# an isolated subshell with dpkg and the other three functions stubbed out,
# so it needs no network, no root and no real dpkg.
assert_main_cleanup_survives_set_u() {
  local description="main() cleanup survives set -u after normal return"
  local output rc=0
  output=$(
    bash -c "
      set -euo pipefail
      source '${SCRIPT_DIR}/mpich-hotfix.sh'
      dpkg() { echo amd64; }
      download() { :; }
      verify() { :; }
      install() { :; }
      main
    " 2>&1
  ) || rc=$?

  if [[ "$rc" != 0 ]]; then
    echo "FAIL: $description: exited $rc: $output" >&2
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

  assert_main_cleanup_survives_set_u

  if ((FAIL_COUNT > 0)); then
    exit 1
  fi
  exit 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
