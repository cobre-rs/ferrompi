#!/usr/bin/env bash
# Fixture test for release-notes.sh. Builds a scratch CHANGELOG.md under
# mktemp -d and checks the extracted section for a handful of version
# shapes, then re-runs the script against every released version heading of
# the repository's own CHANGELOG.md. No network.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

FAIL_COUNT=0

# assert_notes <description> <dir> <version> <expected>
assert_notes() {
  local description="$1" dir="$2" version="$3" expected="$4"
  local actual rc=0
  actual=$(cd "$dir" && bash "${SCRIPT_DIR}/release-notes.sh" "$version") || rc=$?
  if [[ "$rc" != 0 || "$actual" != "$expected" ]]; then
    echo "FAIL: $description: got '$actual'" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    echo "PASS: $description"
  fi
}

# assert_fails <description> <dir> <version>
assert_fails() {
  local description="$1" dir="$2" version="$3"
  if (cd "$dir" && bash "${SCRIPT_DIR}/release-notes.sh" "$version") >/dev/null 2>&1; then
    echo "FAIL: $description: exited 0" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    echo "PASS: $description"
  fi
}

# assert_real_changelog_notes <version>
assert_real_changelog_notes() {
  local version="$1"
  local description="CHANGELOG.md $version has notes"
  local actual rc=0
  actual=$(cd "$REPO_ROOT" && bash "${SCRIPT_DIR}/release-notes.sh" "$version") || rc=$?
  if [[ "$rc" != 0 || -z "$actual" ]]; then
    echo "FAIL: $description: exited $rc" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    echo "PASS: $description"
  fi
}

main() {
  local fixture_dir
  fixture_dir=$(mktemp -d)
  trap 'rm -rf "$fixture_dir"' EXIT

  cat >"${fixture_dir}/CHANGELOG.md" <<'EOF'
## [Unreleased]

## [1.10.0] - 2026-02-01

### Added

- ten

## [1.1.0] - 2026-01-15

### Fixed
- one

- one, second paragraph


## [1.0.0] - 2026-01-01

## [0.9.0] - 2025-12-01

- last

[Unreleased]: https://example.com/compare/v1.10.0...HEAD
[1.10.0]: https://example.com/compare/v1.1.0...v1.10.0
EOF

  assert_notes "1.1.0 ends at next heading, not matched by 1.10.0" "$fixture_dir" "1.1.0" \
    "$(printf '### Fixed\n- one\n\n- one, second paragraph')"

  assert_notes "0.9.0 ends at link references" "$fixture_dir" "0.9.0" "- last"

  assert_fails "empty section" "$fixture_dir" "1.0.0"
  assert_fails "missing section" "$fixture_dir" "2.0.0"

  local version
  while IFS= read -r version; do
    assert_real_changelog_notes "$version"
  done < <(sed -n 's/^## \[\([0-9][0-9.]*\)\].*/\1/p' "${REPO_ROOT}/CHANGELOG.md")

  if ((FAIL_COUNT > 0)); then
    exit 1
  fi
  exit 0
}

main "$@"
