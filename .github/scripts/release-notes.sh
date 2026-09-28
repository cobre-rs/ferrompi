#!/usr/bin/env bash
# Prints the release notes for one version from ./CHANGELOG.md: the lines
# under its "## [<version>]" heading, up to the next "## [" heading or the
# first link-reference line, without leading or trailing blank lines.
# Exits 1 if the section is missing or has no non-blank line.
set -euo pipefail

version="${1:?usage: release-notes.sh <version>}"

notes=$(awk -v heading="## [$version]" '
  found && (/^## \[/ || /^\[[^]]+\]: /) { exit }
  found && (started || NF) { started = 1; print }
  index($0, heading) == 1 { found = 1 }
' CHANGELOG.md)

if [[ -z "$notes" ]]; then
  echo "release-notes.sh: CHANGELOG.md has no notes for $version" >&2
  exit 1
fi
printf '%s\n' "$notes"
