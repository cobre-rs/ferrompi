#!/usr/bin/env bash
#
# check-tutorial.sh — run the tutorial's programs, and check its excerpts and links.
#
# The tutorial in docs/tutorial/ is HTML, so rustdoc never compiles its code;
# this script does. A Rust block of a page,
#
#   <pre data-lang="rust" data-run="NP" id="NAME"><code>...</code></pre>
#
# is a complete program. The script builds each one as an example of a scratch
# crate that depends on this checkout of ferrompi, runs it with
# `mpiexec -n NP`, and fails if it exits non-zero or outlives the timeout. A
# block
#
#   <pre class="output" data-output-of="NAME"><code>...</code></pre>
#
# holds the lines that program prints. Ranks print in any order, so the script
# compares the sorted lines of the block with the sorted lines of the run.
#
# A block of any language that quotes a file of this repository instead,
#
#   <pre data-lang="rust" data-source="src/comm/blocking.rs"><code>...</code></pre>
#
# is an excerpt, not a program. Its file is named from the repository root.
# Each run of lines between elision lines must appear in that file verbatim, on
# consecutive lines, and the runs in the order the block gives them. An elision
# line is a comment that starts with an ellipsis: `// …` or `/* ... */`.
#
# A checked block opens `<pre ...><code>` on one line and closes
# `</code></pre>` on its last line, with the code between them as escaped
# text: no markup inside.
#
# It then copies the tutorial into the built rustdoc at <target>/doc/tutorial,
# where the published site has it, and checks every relative href and src of
# every page: each must name a file there, and an id in that file when it
# carries a #fragment.
#
# Usage: check-tutorial.sh [PAGE.html...]
#   With no arguments, checks every page in docs/tutorial/. Run
#   `cargo doc --no-deps --all-features` first: the published site documents
#   every feature, and the pages link to all of it.
#
# Environment:
#   MPIEXEC           launcher command, split on spaces (default: mpiexec)
#   TUTORIAL_TIMEOUT  per-run wall-clock cap in seconds (default: 60)
#   CARGO_TARGET_DIR  honoured; the scratch crate lives in <target>/tutorial-check
#
# Exit codes: 0 every check passed; 1 a check failed; 2 setup error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

FAIL_COUNT=0
PASS_COUNT=0
EXCERPT_COUNT=0

# fail <message>
fail() {
    echo "FAIL $1" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

# extract_programs <page> <examples-dir> <expected-dir> <manifest> <excerpts-dir>
# Writes each program of <page> to <examples-dir>/<name>.rs and appends
# "<name> <np>" to <manifest>; writes each output block to
# <expected-dir>/<name>.txt. <name> is the page's file stem and the block's id,
# joined by "__", with dashes turned into underscores. Writes each excerpt to
# <excerpts-dir>/<page-stem>__<line>.txt, and appends its path, the file it
# quotes and "<page>:<line>", tab-separated, to <excerpts-dir>/sources.
# Returns non-zero, after naming the offending line, when a block breaks the
# format above.
extract_programs() {
    local page="$1" examples="$2" expected="$3" manifest="$4" excerpts="$5"
    local prefix
    prefix="$(basename "$page" .html)__"
    awk -v examples="$examples" -v expected="$expected" -v manifest="$manifest" \
        -v excerpts="$excerpts" -v prefix="$prefix" -v page="$(basename "$page")" '
        function attr(tag, name,    pattern) {
            pattern = " " name "=\"[^\"]*\""
            if (!match(tag, pattern)) return ""
            return substr(tag, RSTART + length(name) + 3, RLENGTH - length(name) - 4)
        }
        function complain(message) {
            printf "%s:%d: %s\n", FILENAME, FNR, message > "/dev/stderr"
            bad = 1
        }
        function unescape(text) {
            gsub(/&lt;/, "<", text)
            gsub(/&gt;/, ">", text)
            gsub(/&quot;/, "\"", text)
            gsub(/&#39;/, "\047", text)
            gsub(/&amp;/, "\\&", text)
            return text
        }
        function emit(text) {
            if (text ~ /</) complain("markup inside a checked block; write the code as escaped text")
            print unescape(text) > file
        }
        function block_name(id,    name) {
            name = prefix id
            gsub(/-/, "_", name)
            return name
        }
        {
            if (file == "") {
                if (!match($0, /<pre[ >][^>]*>/)) next
                tag = substr($0, RSTART, RLENGTH)
                line = substr($0, RSTART + RLENGTH)
                if (tag ~ / data-source="/) {
                    source = attr(tag, "data-source")
                    if (source == "") {
                        complain("an excerpt needs data-source=\"<file>\"")
                        next
                    }
                    file = excerpts "/" prefix FNR ".txt"
                    printf "%s\t%s\t%s:%d\n", file, source, page, FNR >> (excerpts "/sources")
                } else if (tag ~ /data-lang="rust"/) {
                    np = attr(tag, "data-run")
                    id = attr(tag, "id")
                    if (np !~ /^[1-9][0-9]*$/ || id == "") {
                        complain("a Rust block needs data-run=\"<processes>\" and an id, or data-source")
                        next
                    }
                    name = block_name(id)
                    if (name in programs) {
                        complain("a second Rust block with id " id)
                        next
                    }
                    programs[name] = 1
                    file = examples "/" name ".rs"
                    print name, np >> manifest
                } else if (tag ~ / data-output-of="/) {
                    file = expected "/" block_name(attr(tag, "data-output-of")) ".txt"
                } else {
                    next
                }
                if (!sub(/^<code>/, "", line)) {
                    complain("a checked block opens <code> right after <pre ...>")
                    file = ""
                    next
                }
            } else {
                line = $0
            }
            end = index(line, "</code></pre>")
            if (end > 0) {
                if (end > 1) emit(substr(line, 1, end - 1))
                close(file)
                file = ""
            } else if (line != "" || started) {
                emit(line)
            }
            started = (file != "")
        }
        END {
            if (file != "") complain("a checked block never closes </code></pre>")
            exit bad
        }
    ' "$page"
}

# same_lines <expected> <actual>
# Succeeds when both files hold the same lines in any order; otherwise prints
# a diff of the sorted lines.
same_lines() {
    diff <(sort "$1") <(sort "$2") >&2
}

# quotes <excerpt> <source>
# Succeeds when <source> holds every run of lines of <excerpt> between its
# elision lines, verbatim, on consecutive lines, and the runs in order;
# otherwise prints the first line of the first run it cannot find. Each run
# takes its earliest match after the previous one, which leaves every later run
# the most room.
quotes() {
    awk '
        function elided(line) {
            return line ~ /^[ \t]*(\/\/|\/\*)[ \t]*(…|\.\.\.)/
        }
        # Concatenating "" compares the lines as strings, never as numbers.
        NR == FNR { source[++n] = $0 ""; next }
        { excerpt[++m] = $0 "" }
        END {
            from = 1
            i = 1
            while (i <= m) {
                if (elided(excerpt[i])) { i++; continue }
                first = i
                while (i <= m && !elided(excerpt[i])) i++
                len = i - first
                found = 0
                for (at = from; at + len - 1 <= n && !found; at++) {
                    k = 0
                    while (k < len && source[at + k] == excerpt[first + k]) k++
                    found = (k == len)
                }
                if (!found) {
                    printf "not in the file: %s\n", excerpt[first] > "/dev/stderr"
                    exit 1
                }
                from = at - 1 + len
                runs++
            }
            if (runs == 0) {
                print "the excerpt quotes no lines" > "/dev/stderr"
                exit 1
            }
        }
    ' "$2" "$1"
}

# check_excerpts <root> <excerpts-dir>
# Checks each excerpt that <excerpts-dir>/sources lists against the file it
# quotes, named from <root>. Calls fail once per excerpt that does not match.
check_excerpts() {
    local root="$1" excerpts="$2"
    local excerpt source where
    while IFS=$'\t' read -r excerpt source where; do
        if [[ ! -f "$root/$source" ]]; then
            fail "$where: the excerpt quotes $source, which does not exist"
        elif quotes "$excerpt" "$root/$source"; then
            EXCERPT_COUNT=$((EXCERPT_COUNT + 1))
        else
            fail "$where: the excerpt no longer matches $source"
        fi
    done < "$excerpts/sources"
}

# check_links <site-dir> <page>...
# Checks the relative href and src values of each page, which sits in
# <site-dir>/tutorial/. Calls fail once per broken link.
check_links() {
    local site="$1"
    shift
    local page staged link path fragment target
    for page in "$@"; do
        staged="$site/tutorial/$(basename "$page")"
        while IFS= read -r link; do
            case "$link" in
                http://* | https://* | mailto:*) continue ;;
            esac
            path="${link%%#*}"
            fragment=""
            [[ "$link" == *"#"* ]] && fragment="${link#*#}"
            if [[ -z "$path" ]]; then
                target="$staged"
            else
                target="$(dirname "$staged")/$path"
                [[ -d "$target" ]] && target="$target/index.html"
            fi
            if [[ ! -f "$target" ]]; then
                fail "$(basename "$page"): $link: no such file"
            elif [[ -n "$fragment" ]] && ! grep -qF "id=\"$fragment\"" "$target"; then
                fail "$(basename "$page"): $link: no id \"$fragment\" in the target"
            fi
        done < <(grep -oE '(href|src)="[^"]*"' "$staged" | sed -E 's/^(href|src)="//; s/"$//')
    done
}

# build_scratch_crate <crate-dir>
# Writes a crate whose one dependency is this checkout of ferrompi, with the
# `rma` feature that the one-sided chapter needs and the `slurm` feature that
# the building-and-running chapter needs, pinned to the repository's
# Cargo.lock, and empties its examples/ directory.
build_scratch_crate() {
    local crate="$1"
    rm -rf "$crate/examples"
    mkdir -p "$crate/examples" "$crate/src"
    cat > "$crate/Cargo.toml" << EOF
[package]
name = "tutorial-check"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
ferrompi = { path = "$REPO_ROOT", features = ["rma", "slurm"] }

[workspace]
EOF
    : > "$crate/src/lib.rs"
    cp "$REPO_ROOT/Cargo.lock" "$crate/Cargo.lock"
}

# run_programs <target-dir> <page>...
# Extracts the checked blocks of every page, checks the excerpts, then builds
# and runs the programs.
run_programs() {
    local target="$1"
    shift
    local crate="$target/tutorial-check"
    local expected="$crate/expected" outputs="$crate/outputs" manifest="$crate/manifest"
    local excerpts="$crate/excerpts"
    local timeout_s="${TUTORIAL_TIMEOUT:-60}"
    local -a mpiexec
    read -ra mpiexec <<< "${MPIEXEC:-mpiexec}"

    build_scratch_crate "$crate"
    rm -rf "$expected" "$outputs" "$excerpts"
    mkdir -p "$expected" "$outputs" "$excerpts"
    : > "$manifest"
    : > "$excerpts/sources"

    local page
    for page in "$@"; do
        extract_programs "$page" "$crate/examples" "$expected" "$manifest" "$excerpts" \
            || fail "$(basename "$page"): malformed checked block"
    done
    check_excerpts "$REPO_ROOT" "$excerpts"

    local file name
    for file in "$expected"/*.txt; do
        [[ -e "$file" ]] || continue
        name="$(basename "$file" .txt)"
        grep -q "^$name " "$manifest" || fail "$name: an output block names no program"
    done

    if [[ ! -s "$manifest" ]]; then
        echo "No programs to run."
        return
    fi

    if ! cargo build --manifest-path "$crate/Cargo.toml" --target-dir "$target" --examples; then
        fail "the tutorial's programs do not build"
        return
    fi

    local np bin out status
    while read -r name np; do
        bin="$target/debug/examples/$name"
        out="$outputs/$name.out"
        status=0
        timeout --kill-after=10 "$timeout_s" "${mpiexec[@]}" -n "$np" "$bin" \
            < /dev/null > "$out" 2> "$outputs/$name.err" || status=$?
        if [[ "$status" -eq 124 || "$status" -eq 137 ]]; then
            fail "$name (np=$np): still running after ${timeout_s}s"
        elif [[ "$status" -ne 0 ]]; then
            tail -n 20 "$outputs/$name.err" >&2
            fail "$name (np=$np): exit status $status"
        elif [[ -f "$expected/$name.txt" ]] && ! same_lines "$expected/$name.txt" "$out"; then
            fail "$name (np=$np): output differs from the page"
        else
            echo "PASS $name (np=$np)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    done < "$manifest"
}

main() {
    local target="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
    local -a pages
    if [[ $# -gt 0 ]]; then
        pages=("$@")
    else
        pages=("$REPO_ROOT"/docs/tutorial/*.html)
    fi

    if [[ ! -f "$target/doc/ferrompi/index.html" ]]; then
        echo "No rustdoc in $target/doc: run \`cargo doc --no-deps --all-features\` first." >&2
        exit 2
    fi
    rm -rf "$target/doc/tutorial"
    cp -r "$REPO_ROOT/docs/tutorial" "$target/doc/tutorial"
    check_links "$target/doc" "${pages[@]}"

    run_programs "$target" "${pages[@]}"

    echo
    echo "Tutorial check: $PASS_COUNT programs and $EXCERPT_COUNT excerpts passed, $FAIL_COUNT failures."
    [[ "$FAIL_COUNT" -eq 0 ]] || exit 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
