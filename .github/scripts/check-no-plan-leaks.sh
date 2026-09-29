#!/usr/bin/env bash
#
# check-no-plan-leaks.sh — plan-structure leak gate.
#
# Shipped files describe behaviour, not how the work was organized. Plans live
# in plans/ (gitignored) and the audit register in audits/; their identifiers
# mean nothing to a reader of the crate, its docs or its changelog. Commit
# messages may reference plan structure; shipped files may not.
#
# Usage: check-no-plan-leaks.sh [FILE...]
#   With no arguments, scans every tracked file under the shipped paths below.
#   With arguments, scans exactly those files (used by the fixture test).
#
# Forbidden in every scanned file:
#   epic, ticket, sprint (bare words, any case of the first letter, plural too:
#     a number is not what makes them leaks — "prior epics" names the plan);
#   T0NN task ids; F<n>-NNN feature ids; W-<n> workstream ids; "the campaign";
#   "Requirement 3" / "Requirements 3"; "decision D-11"; "Fix 1"; AC tags
#     ("AC:", "AC1", "AC-1");
#   register ids (SND-01, COR-12, ...), decision ids (D-7), PERF-<n>, plan ADR
#     ids (ADR-0NN; the public ADRs are ADR-00NN and stay allowed) and plan
#     requirement ids (R12);
#   plans/ paths, MEMORY.md, and .claude/ paths.
# Forbidden outside docs/adr/ only:
#   "Option A" — a plan-alternative label. An ADR's own Options section is the
#     one place where lettered alternatives are the durable record.
# Section references:
#   "§<n>" is allowed only when the same line names its source: a standard
#   ("MPI-4.1 §7.13", "MPI 4.0 §10.2", "C11 §6.7.4"), a spec, or an .md file.
#   A bare "§5" points into a plan document the reader cannot open.
#
# Exit codes: 0 no leaks; 1 leaks found (printed as FILE:LINE:TEXT).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

readonly PATTERN='\b[Ee]pics?\b|\b[Tt]ickets?\b|\b[Ss]prints?\b|\bT0[0-9][0-9]\b|\bF[0-9]?-[0-9]{2,}\b|\bW-[0-9]+\b|\b[Tt]he campaign\b|\b[Rr]equirements? [0-9]+[a-z]?\b|\b[Dd]ecision [A-Z]-?[0-9]+\b|\bFix [0-9]\b|\bAC: |\bAC[0-9]\b|\bAC-[0-9]|\b(SND|COR|ARC|ABI|PRF|BLT|DOC|INF|VER)-[0-9]{2}\b|\bD-[0-9]{1,2}\b|\bPERF-[0-9]+\b|\bADR-0[0-9]{2}\b|\bR[0-9]{1,3}\b|plans/|MEMORY\.md|\.claude/'
readonly OPTION_LABEL='\bOption [A-Z]\b'
readonly SECTION_REF='§ ?[0-9]'
readonly SECTION_REF_ALLOW='(MPI[- ]?[0-9][0-9.]*|C[0-9]{2}|[Ss]pec[a-z]*|\.md)[] )]*[ (]?§'

if [[ $# -gt 0 ]]; then
    files=("$@")
else
    mapfile -t files < <(cd "$REPO_ROOT" && git ls-files -- \
        src csrc build.rs examples benches tests docs \
        README.md CHANGELOG.md CONTRIBUTING.md Cargo.toml)
    cd "$REPO_ROOT"
fi

violations=""
for file in "${files[@]}"; do
    [[ -f "$file" ]] || continue
    hits=$(grep -nE "$PATTERN" "$file" || true)
    if [[ "$file" != docs/adr/* && "$file" != */docs/adr/* ]]; then
        hits+=$'\n'$(grep -nE "$OPTION_LABEL" "$file" || true)
    fi
    hits+=$'\n'$(grep -nE "$SECTION_REF" "$file" | grep -vE "$SECTION_REF_ALLOW" || true)
    while IFS= read -r hit; do
        [[ -n "$hit" ]] && violations+="${file}:${hit}"$'\n'
    done <<< "$hits"
done

if [[ -n "$violations" ]]; then
    echo "FAIL: plan-structure references in shipped files:"
    echo
    printf '%s' "$violations"
    echo
    echo "Rewrite each in behavioural terms: name the invariant or the design, not the"
    echo "plan item, register row or decision that produced it. Keep the rest of the"
    echo "sentence. A section reference needs its source on the same line (MPI-4.1 §7.13)."
    exit 1
fi

echo "OK: no plan-structure references in shipped files."
