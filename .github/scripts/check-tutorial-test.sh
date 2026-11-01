#!/usr/bin/env bash
# Fixture test for check-tutorial.sh. Sources the script (its main is guarded,
# so sourcing only defines the functions) and runs its block extraction, output
# comparison and link check on fixture pages: no MPI, no cargo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/check-tutorial.sh"

TEST_FAILURES=0
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# expect <description> <command>...
expect() {
    local description="$1"
    shift
    if "$@"; then
        echo "PASS: $description"
    else
        echo "FAIL: $description" >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
}

# expect_not <description> <command>...
expect_not() {
    local description="$1"
    shift
    if "$@" 2> /dev/null; then
        echo "FAIL: $description: succeeded, expected a failure" >&2
        TEST_FAILURES=$((TEST_FAILURES + 1))
    else
        echo "PASS: $description"
    fi
}

# extract_into <name> <page-body>
# Writes <page-body> to <name>.html in a fresh directory and extracts it.
extract_into() {
    local dir="$WORK/$1"
    mkdir -p "$dir/examples" "$dir/expected" "$dir/excerpts"
    printf '%s\n' "$2" > "$dir/$1.html"
    : > "$dir/manifest"
    : > "$dir/excerpts/sources"
    extract_programs "$dir/$1.html" "$dir/examples" "$dir/expected" "$dir/manifest" \
        "$dir/excerpts"
}

# --- extraction ------------------------------------------------------------

extract_into page '<p>Intro</p>
<pre data-lang="bash"><code>cargo build</code></pre>
<pre data-lang="rust" data-run="3" id="hello-world"><code>use ferrompi::Mpi;

fn main() -&gt; ferrompi::Result&lt;()&gt; {
    let s = &amp;"it&#39;s";
    println!(&quot;{s}&quot;);
    Ok(())
}</code></pre>
<pre class="output" data-output-of="hello-world"><code>line one
line two</code></pre>'

expect "a program is unescaped into its own example" \
    diff "$WORK/page/examples/page__hello_world.rs" - << 'EOF'
use ferrompi::Mpi;

fn main() -> ferrompi::Result<()> {
    let s = &"it's";
    println!("{s}");
    Ok(())
}
EOF
expect "the manifest names the program and its process count" \
    diff "$WORK/page/manifest" - <<< "page__hello_world 3"
expect "an output block is saved under its program's name" \
    diff "$WORK/page/expected/page__hello_world.txt" - << 'EOF'
line one
line two
EOF
expect "a bash block is not extracted" \
    test "$(ls "$WORK/page/examples")" = "page__hello_world.rs"

expect_not "a Rust block without data-run is rejected" \
    extract_into no_run '<pre data-lang="rust" id="x"><code>fn main() {}</code></pre>'
expect_not "a Rust block without an id is rejected" \
    extract_into no_id '<pre data-lang="rust" data-run="2"><code>fn main() {}</code></pre>'
expect_not "markup inside a Rust block is rejected" \
    extract_into markup '<pre data-lang="rust" data-run="2" id="x"><code><span>fn</span> main() {}</code></pre>'
expect_not "two Rust blocks with one id are rejected" \
    extract_into twice '<pre data-lang="rust" data-run="2" id="x"><code>fn main() {}</code></pre>
<pre data-lang="rust" data-run="2" id="x"><code>fn main() {}</code></pre>'
expect_not "an unterminated block is rejected" \
    extract_into open '<pre data-lang="rust" data-run="2" id="x"><code>fn main() {}'

# --- excerpts --------------------------------------------------------------

ROOT="$WORK/root"
mkdir -p "$ROOT/src"
cat > "$ROOT/src/lib.rs" << 'EOF'
pub fn first() -> i32 {
    1
}

pub fn second() -> i32 {
    2
}
EOF

extract_into excerpt '<pre data-lang="rust" data-source="src/lib.rs"><code>pub fn first() -&gt; i32 {
// …
pub fn second() -&gt; i32 {
    2
}</code></pre>'

expect "an excerpt is not extracted as a program" \
    test -z "$(ls "$WORK/excerpt/examples")"
expect "an excerpt is listed with the file it quotes and where it is" \
    diff "$WORK/excerpt/excerpts/sources" - \
    <<< "$WORK/excerpt/excerpts/excerpt__1.txt	src/lib.rs	excerpt.html:1"
expect "an excerpt is unescaped" \
    grep -qx 'pub fn second() -> i32 {' "$WORK/excerpt/excerpts/excerpt__1.txt"
expect_not "an excerpt without a file is rejected" \
    extract_into no_source '<pre data-lang="rust" data-source=""><code>fn f() {}</code></pre>'

# quotes_text <excerpt-text>
# Checks <excerpt-text> against the fixture's src/lib.rs.
quotes_text() {
    printf '%s\n' "$1" > "$WORK/excerpt.txt"
    quotes "$WORK/excerpt.txt" "$ROOT/src/lib.rs"
}

expect "an excerpt with an elision matches" \
    quotes "$WORK/excerpt/excerpts/excerpt__1.txt" "$ROOT/src/lib.rs"
expect "a C-style elision line matches" \
    quotes_text 'pub fn first() -> i32 {
    /* ... */
}'
expect "a run may start on any line" quotes_text '    2'
expect_not "a changed line does not match" quotes_text 'pub fn first() -> i64 {'
expect_not "lines the file does not hold on consecutive lines do not match" \
    quotes_text 'pub fn first() -> i32 {
}'
expect_not "runs out of order do not match" \
    quotes_text 'pub fn second() -> i32 {
// …
pub fn first() -> i32 {'
expect_not "one line of the file does not match two runs" \
    quotes_text '    1
// …
    1'
expect_not "an excerpt of elisions alone does not match" quotes_text '// …'
expect_not "a number does not match its other spelling" quotes_text '    2.0'

mkdir -p "$WORK/listed"
printf '%s\t%s\t%s\n' "$WORK/excerpt.txt" src/gone.rs page.html:3 > "$WORK/listed/sources"
printf '%s\t%s\t%s\n' "$WORK/excerpt/excerpts/excerpt__1.txt" src/lib.rs page.html:9 \
    >> "$WORK/listed/sources"
FAIL_COUNT=0
EXCERPT_COUNT=0
EXCERPT_ERRORS="$(check_excerpts "$ROOT" "$WORK/listed" 2>&1 \
    && echo "count=$FAIL_COUNT passed=$EXCERPT_COUNT")"
expect "a missing file is one failure, and the matching excerpt passes" \
    grep -q "count=1 passed=1" <<< "$EXCERPT_ERRORS"
expect "a missing file is named with the excerpt's place" \
    grep -q "page.html:3: the excerpt quotes src/gone.rs, which does not exist" \
    <<< "$EXCERPT_ERRORS"

# --- output comparison -----------------------------------------------------

printf 'rank 0\nrank 1\n' > "$WORK/expected.txt"
printf 'rank 1\nrank 0\n' > "$WORK/reordered.out"
printf 'rank 0\nrank 0\n' > "$WORK/wrong.out"
expect "lines in another order match" same_lines "$WORK/expected.txt" "$WORK/reordered.out"
expect_not "different lines do not match" same_lines "$WORK/expected.txt" "$WORK/wrong.out"

# --- links -----------------------------------------------------------------

SITE="$WORK/site"
mkdir -p "$SITE/tutorial" "$SITE/api"
echo '<h2 id="here">Here</h2>' > "$SITE/tutorial/other.html"
echo '<h1>API</h1>' > "$SITE/api/index.html"
cat > "$SITE/tutorial/page.html" << 'EOF'
<h2 id="local">Local</h2>
<link rel="stylesheet" href="other.html">
<a href="other.html#here">ok</a>
<a href="#local">ok</a>
<a href="../api/">ok</a>
<a href="https://example.com/missing.html">external</a>
<a href="missing.html">broken</a>
<a href="other.html#nowhere">broken</a>
<script src="gone.js"></script>
EOF

FAIL_COUNT=0
LINK_ERRORS="$(check_links "$SITE" "$WORK/page.html" 2>&1 && echo "count=$FAIL_COUNT")"
expect "three broken links are reported" grep -q "count=3" <<< "$LINK_ERRORS"
expect "a missing file is named" grep -q "missing.html: no such file" <<< "$LINK_ERRORS"
expect "a missing id is named" grep -q 'no id "nowhere"' <<< "$LINK_ERRORS"
expect "a missing script is named" grep -q "gone.js: no such file" <<< "$LINK_ERRORS"

echo
if [[ "$TEST_FAILURES" -gt 0 ]]; then
    echo "$TEST_FAILURES check-tutorial fixture tests failed." >&2
    exit 1
fi
echo "All check-tutorial fixture tests passed."
