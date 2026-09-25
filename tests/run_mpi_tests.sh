#!/usr/bin/env bash
# ==========================================================================
# FerroMPI MPI integration test runner.
#
# Discovers directive-bearing examples/*.rs files, builds them through
# `cargo build --examples --message-format=json-render-diagnostics`, and
# runs each one under mpiexec. Executable paths are resolved from the cargo
# JSON artifact stream, so the runner works with a relocated
# CARGO_TARGET_DIR.
#
# Usage:
#   tests/run_mpi_tests.sh [features]
#   tests/run_mpi_tests.sh --valgrind [features]
#
# --valgrind selects only examples whose directive has the `valgrind` flag,
# runs each once at its minimal np under `valgrind --error-exitcode=99`
# with tests/valgrind/mpich.supp, and multiplies the per-run timeout by 10.
#
# Environment:
#   MPI_NP_LIST       Space-separated process counts (default: 4)
#   MPIEXEC           Path to mpiexec (default: mpiexec)
#   MPI_TEST_TIMEOUT  Per-run wall-clock cap in seconds (default: 90)
# ==========================================================================
set -euo pipefail

MPI_NP_LIST="${MPI_NP_LIST:-4}"
MPIEXEC="${MPIEXEC:-mpiexec}"
MPI_TEST_TIMEOUT="${MPI_TEST_TIMEOUT:-90}"

FEATURES=""
IMPL_ID=""
FEATURE_CLOSURE=""
TMPDIR_RUN=""
VALGRIND_MODE=0
VALGRIND_SUPP=""
declare -a MPI_NP_LIST_ARR=()
declare -a MPIEXEC_ARGS=()

declare -A DIRECTIVE_NP=()
declare -A DIRECTIVE_TIMEOUT=()
declare -A DIRECTIVE_SKIPOK=()
declare -A DIRECTIVE_EXPECT=()
declare -A DIRECTIVE_STDERR=()
declare -A DIRECTIVE_VALGRIND=()
declare -a DIRECTIVE_ORDER=()

declare -A ARTIFACTS=()
declare -A REQUIRED_FEATURES=()

declare -a SKIP_REASONS=()
declare -a FAILED_RUNS=()
PASS_COUNT=0
SKIP_COUNT=0
SKIP_FEATURE_COUNT=0
FAIL_COUNT=0
RUNS_EXECUTED=0

# ==========================================================================
# Pure functions
# ==========================================================================

# parse_directive <line>
# Parses the content of a `// mpi-test: ` line into
# np/timeout/skip-ok/expect/valgrind fields. Prints
# "<np>|<timeout>|<skip-ok>|<expect>|<valgrind>" (empty string for an absent
# field, "1" for the valgrind flag) and returns 0, or prints a reason on
# stderr and returns 1.
parse_directive() {
  local line="$1"
  local np="" tmo="" skip_ok="" expect="" valgrind=""
  local -a kvs
  read -ra kvs <<<"$line"

  local kv
  for kv in "${kvs[@]}"; do
    case "$kv" in
      np=*)
        np="${kv#np=}"
        if [[ ! "$np" =~ ^[0-9]+(\.\.)?$ ]]; then
          echo "invalid np value: $kv" >&2
          return 1
        fi
        ;;
      timeout=*)
        tmo="${kv#timeout=}"
        if [[ ! "$tmo" =~ ^[0-9]+$ ]]; then
          echo "invalid timeout value: $kv" >&2
          return 1
        fi
        ;;
      skip-ok=*)
        skip_ok="${kv#skip-ok=}"
        if [[ -z "$skip_ok" ]]; then
          echo "invalid skip-ok value: $kv" >&2
          return 1
        fi
        ;;
      expect=*)
        expect="${kv#expect=}"
        case "$expect" in
          abort | unfinalized) ;;
          *)
            echo "invalid expect value: $kv" >&2
            return 1
            ;;
        esac
        ;;
      valgrind)
        valgrind=1
        ;;
      *)
        echo "unknown directive key: $kv" >&2
        return 1
        ;;
    esac
  done

  if [[ -z "$np" ]]; then
    echo "missing np key" >&2
    return 1
  fi

  printf '%s|%s|%s|%s|%s\n' "$np" "$tmo" "$skip_ok" "$expect" "$valgrind"
}

# expand_np <spec> <np-list...>
# np=N runs once at N. np=N.. runs at every np-list value >= N, or once at N
# if none qualifies. Prints the resulting np values space-separated.
expand_np() {
  local spec="$1"
  shift
  local -a np_list=("$@")

  if [[ "$spec" != *".." ]]; then
    echo "$spec"
    return 0
  fi

  local base="${spec%..}"
  local -a matched=()
  local n
  for n in "${np_list[@]}"; do
    if ((n >= base)); then
      matched+=("$n")
    fi
  done

  if ((${#matched[@]} == 0)); then
    echo "$base"
  else
    echo "${matched[*]}"
  fi
}

# valgrind_np <spec>
# Prints the minimal np for valgrind mode: np=N and np=N.. both give N.
valgrind_np() {
  local spec="$1"
  echo "${spec%..}"
}

# build_cmd <suppressions-path>
# Prints the valgrind invocation prefix inserted between `mpiexec -n <np>`
# and the executable.
build_cmd() {
  local supp="$1"
  echo "valgrind -q --error-exitcode=99 --track-origins=yes --leak-check=no --suppressions=$supp"
}

# impl_id <mpiexec --version output>
# HYDRA + a Version: X.Y.Z line becomes mpich-X.Y.Z; Open MPI/OpenRTE +
# X.Y.Z becomes openmpi-X.Y.Z; anything else becomes unknown.
impl_id() {
  local output="$1"

  if [[ "$output" == *HYDRA* ]]; then
    local v
    v=$(printf '%s\n' "$output" | sed -n 's/^[[:space:]]*Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -1)
    if [[ -n "$v" ]]; then
      echo "mpich-$v"
      return 0
    fi
  fi

  if [[ "$output" == *"Open MPI"* || "$output" == *"OpenRTE"* ]]; then
    local v
    v=$(printf '%s\n' "$output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [[ -n "$v" ]]; then
      echo "openmpi-$v"
      return 0
    fi
  fi

  echo "unknown"
}

# feature_closure <requested-csv> <features-json>
# <features-json> is cargo metadata's `packages[0].features` object (feature
# name -> implied feature list). Prints the transitive closure of
# <requested-csv> as a sorted comma-separated list.
feature_closure() {
  local requested_csv="$1" features_json="$2"

  local -A imp_of=()
  local feat implied
  while IFS='|' read -r feat implied; do
    imp_of["$feat"]="$implied"
  done < <(jq -r 'to_entries[] | "\(.key)|\(.value | join(","))"' <<<"$features_json")

  local -A closure=()
  local -a reqs
  IFS=',' read -ra reqs <<<"$requested_csv"
  local f
  for f in "${reqs[@]}"; do
    closure["$f"]=1
  done

  local changed=1
  while ((changed)); do
    changed=0
    for f in "${!closure[@]}"; do
      local -a implies
      IFS=',' read -ra implies <<<"${imp_of[$f]:-}"
      local imp
      for imp in "${implies[@]}"; do
        if [[ -z "${closure[$imp]+x}" ]] && [[ -n "${imp_of[$imp]+x}" ]]; then
          closure["$imp"]=1
          changed=1
        fi
      done
    done
  done

  local -a sorted
  mapfile -t sorted < <(printf '%s\n' "${!closure[@]}" | sort)
  (
    IFS=,
    echo "${sorted[*]}"
  )
}

# classify <exit> <output-file> <skip-ok> <expect> <stderr-literal> <impl-id>
# Prints one outcome line: "PASS", "SKIP <reasons>" or "FAIL <detail>".
classify() {
  local exit_code="$1" outfile="$2" skip_ok="$3" expect="$4" literal="$5" impl="$6"

  if [[ "$exit_code" == "124" || "$exit_code" == "137" ]]; then
    echo "FAIL timeout"
    return 0
  fi

  if [[ "$exit_code" == "99" ]]; then
    echo "FAIL valgrind errors"
    return 0
  fi

  if [[ "$expect" == "abort" ]]; then
    if [[ "$exit_code" == "0" ]]; then
      echo "FAIL exited 0, abort expected"
    elif grep -qF -- "$literal" "$outfile"; then
      echo "PASS"
    else
      echo "FAIL abort marker missing"
    fi
    return 0
  fi

  if [[ "$expect" == "unfinalized" ]]; then
    if [[ "$exit_code" != "0" && "$exit_code" != "1" ]]; then
      echo "FAIL exit $exit_code"
      return 0
    fi
  elif [[ "$exit_code" != "0" ]]; then
    echo "FAIL exit $exit_code"
    return 0
  fi

  local skip_lines
  skip_lines=$(grep '^SKIP: ' "$outfile" || true)

  if [[ -n "$skip_lines" ]]; then
    local registered=1 ok
    local -a skip_oks
    IFS=',' read -ra skip_oks <<<"$skip_ok"
    for ok in "${skip_oks[@]}"; do
      if [[ "$impl" == "$ok"* ]]; then
        registered=0
      fi
    done

    local reasons="" first_reason="" r
    while IFS= read -r r; do
      r="${r#SKIP: }"
      if [[ -z "$first_reason" ]]; then
        first_reason="$r"
      fi
      if [[ -z "$reasons" ]]; then
        reasons="$r"
      else
        reasons="$reasons; $r"
      fi
    done <<<"$skip_lines"

    if [[ "$registered" == 0 ]]; then
      echo "SKIP $reasons"
    else
      echo "FAIL unregistered SKIP: $first_reason"
    fi
    return 0
  fi

  if [[ -n "$literal" ]] && ! grep -qF -- "$literal" "$outfile"; then
    echo "FAIL stderr marker missing"
    return 0
  fi

  echo "PASS"
}

# artifact_outcome <required-features-csv> <closure-csv>
# Decides the outcome for an example whose build artifact is missing.
# Prints "SKIP(feature)" if any required feature falls outside the closure,
# or "FAIL missing binary" otherwise.
artifact_outcome() {
  local required_csv="$1" closure_csv="$2"
  local -a reqs
  IFS=',' read -ra reqs <<<"$required_csv"
  local req
  for req in "${reqs[@]}"; do
    if [[ ",$closure_csv," != *",$req,"* ]]; then
      echo "SKIP(feature)"
      return 0
    fi
  done
  echo "FAIL missing binary"
}

# ==========================================================================
# Discovery, build and execution
# ==========================================================================

# discover <examples-dir>
# Reads every top-level <examples-dir>/*.rs, validates its directive grammar
# and populates the DIRECTIVE_* globals. Exits 2 on any grammar error, before
# building anything.
discover() {
  local examples_dir="$1"
  local f base line stderr_count directive_count
  for f in $(printf '%s\n' "$examples_dir"/*.rs | sort); do
    base=$(basename "$f" .rs)
    line=$(sed -n 's#^// mpi-test: ##p' "$f" | head -1)
    stderr_count=$(grep -c '^// mpi-test-stderr: ' "$f" || true)
    directive_count=$(grep -c '^// mpi-test: ' "$f" || true)

    if ((directive_count > 1)); then
      echo "ERROR: $f: more than one // mpi-test: line" >&2
      exit 2
    fi

    if [[ -z "$line" ]]; then
      if ((stderr_count > 0)); then
        echo "ERROR: $f: // mpi-test-stderr line without a // mpi-test: line" >&2
        exit 2
      fi
      if [[ "$base" == test_* ]]; then
        echo "ERROR: $f: missing // mpi-test: directive" >&2
        exit 2
      fi
      continue
    fi

    local parsed
    if ! parsed=$(parse_directive "$line"); then
      echo "ERROR: $f: invalid // mpi-test: directive: $line" >&2
      exit 2
    fi
    local np tmo skip_ok expect valgrind
    IFS='|' read -r np tmo skip_ok expect valgrind <<<"$parsed"

    if ((stderr_count > 1)); then
      echo "ERROR: $f: more than one // mpi-test-stderr line" >&2
      exit 2
    fi

    local stderr_literal=""
    if ((stderr_count == 1)); then
      stderr_literal=$(sed -n 's#^// mpi-test-stderr: ##p' "$f" | head -1)
    fi

    if [[ -n "$expect" && "$stderr_count" -eq 0 ]]; then
      echo "ERROR: $f: expect=$expect requires a // mpi-test-stderr line" >&2
      exit 2
    fi

    if [[ "$expect" == "unfinalized" && "$np" != "1" ]]; then
      echo "ERROR: $f: expect=unfinalized requires np=1" >&2
      exit 2
    fi

    DIRECTIVE_ORDER+=("$base")
    DIRECTIVE_NP["$base"]="$np"
    DIRECTIVE_TIMEOUT["$base"]="${tmo:-$MPI_TEST_TIMEOUT}"
    DIRECTIVE_SKIPOK["$base"]="$skip_ok"
    DIRECTIVE_EXPECT["$base"]="$expect"
    DIRECTIVE_STDERR["$base"]="$stderr_literal"
    DIRECTIVE_VALGRIND["$base"]="$valgrind"
  done
}

# build <features>
# Builds every example (dev profile only) and populates ARTIFACTS[name] with
# each built target's absolute executable path, resolved from cargo's JSON
# artifact stream so a relocated CARGO_TARGET_DIR still works. Exits 2 on a
# build failure.
build() {
  local features="$1"
  local -a build_args=(build --examples --message-format=json-render-diagnostics)
  if [[ -n "$features" ]]; then
    build_args+=(--features "$features")
  fi

  local json_file="$TMPDIR_RUN/cargo-build.json"
  if ! cargo "${build_args[@]}" >"$json_file"; then
    echo "ERROR: cargo build --examples failed" >&2
    exit 2
  fi

  local name exe
  while IFS='|' read -r name exe; do
    ARTIFACTS["$name"]="$exe"
  done < <(jq -r 'select(.reason=="compiler-artifact" and (.target.kind|index("example")) and .executable!=null) | "\(.target.name)|\(.executable)"' "$json_file")
}

# run_all
# Runs every discovered example (expanding np) through mpiexec, prints one
# report line per run, and tallies PASS/SKIP/SKIP(feature)/FAIL. In
# VALGRIND_MODE, only valgrind-tagged examples run, each once at its minimal
# np under the valgrind prefix, with the timeout multiplied by 10.
run_all() {
  local name
  for name in "${DIRECTIVE_ORDER[@]}"; do
    if ((VALGRIND_MODE)) && [[ -z "${DIRECTIVE_VALGRIND[$name]}" ]]; then
      continue
    fi

    local np_spec="${DIRECTIVE_NP[$name]}"
    local run_timeout="${DIRECTIVE_TIMEOUT[$name]}"
    local skip_ok="${DIRECTIVE_SKIPOK[$name]}"
    local expect="${DIRECTIVE_EXPECT[$name]}"
    local literal="${DIRECTIVE_STDERR[$name]}"
    local required="${REQUIRED_FEATURES[$name]:-}"
    local exe="${ARTIFACTS[$name]:-}"

    if ((VALGRIND_MODE)); then
      run_timeout=$((run_timeout * 10))
    fi

    if [[ -z "$exe" ]]; then
      local outcome
      outcome=$(artifact_outcome "$required" "$FEATURE_CLOSURE")

      if [[ "$outcome" == "SKIP(feature)" ]]; then
        echo "SKIP(feature) $name (np=$np_spec) required-features=$required"
        SKIP_FEATURE_COUNT=$((SKIP_FEATURE_COUNT + 1))
      else
        echo "FAIL $name (np=$np_spec) missing binary"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_RUNS+=("$name (np=$np_spec): missing binary")
      fi
      continue
    fi

    local -a nps
    if ((VALGRIND_MODE)); then
      nps=("$(valgrind_np "$np_spec")")
    else
      read -ra nps <<<"$(expand_np "$np_spec" "${MPI_NP_LIST_ARR[@]}")"
    fi

    local np
    for np in "${nps[@]}"; do
      local outfile="$TMPDIR_RUN/${name}.np${np}.out"
      local rc=0
      local -a exec_cmd=("$MPIEXEC" "${MPIEXEC_ARGS[@]}" -n "$np")
      if ((VALGRIND_MODE)); then
        local -a vg_prefix
        read -ra vg_prefix <<<"$(build_cmd "$VALGRIND_SUPP")"
        exec_cmd+=("${vg_prefix[@]}")
      fi
      exec_cmd+=("$exe")
      timeout --kill-after=10 "$run_timeout" "${exec_cmd[@]}" >"$outfile" 2>&1 || rc=$?

      local outcome
      outcome=$(classify "$rc" "$outfile" "$skip_ok" "$expect" "$literal" "$IMPL_ID")
      local status="${outcome%% *}"
      local reason=""
      if [[ "$status" != "$outcome" ]]; then
        reason="${outcome#* }"
      fi

      if [[ -n "$reason" ]]; then
        echo "$status $name (np=$np) $reason"
      else
        echo "$status $name (np=$np)"
      fi

      RUNS_EXECUTED=$((RUNS_EXECUTED + 1))
      case "$status" in
        PASS)
          PASS_COUNT=$((PASS_COUNT + 1))
          ;;
        SKIP)
          SKIP_COUNT=$((SKIP_COUNT + 1))
          SKIP_REASONS+=("$name (np=$np): $reason")
          ;;
        FAIL)
          FAIL_COUNT=$((FAIL_COUNT + 1))
          FAILED_RUNS+=("$name (np=$np): $reason")
          tail -n 40 "$outfile" | sed 's/^/    /'
          ;;
      esac
    done
  done
}

# ==========================================================================
# Main
# ==========================================================================

main() {
  if [[ "${1:-}" == "--valgrind" ]]; then
    VALGRIND_MODE=1
    shift
  fi
  FEATURES="${1:-}"

  local -a missing=()
  command -v "$MPIEXEC" >/dev/null 2>&1 || missing+=("$MPIEXEC")
  command -v cargo >/dev/null 2>&1 || missing+=("cargo")
  command -v jq >/dev/null 2>&1 || missing+=("jq")
  if ((VALGRIND_MODE)); then
    command -v valgrind >/dev/null 2>&1 || missing+=("valgrind")
  fi
  if ((${#missing[@]} > 0)); then
    echo "ERROR: missing required tool(s): ${missing[*]}" >&2
    exit 2
  fi

  if ((VALGRIND_MODE)); then
    VALGRIND_SUPP="$(git rev-parse --show-toplevel)/tests/valgrind/mpich.supp"
  fi

  read -ra MPI_NP_LIST_ARR <<<"$MPI_NP_LIST"

  TMPDIR_RUN=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$TMPDIR_RUN'" EXIT

  local version_output
  version_output=$("$MPIEXEC" --version 2>&1 || true)
  IMPL_ID=$(impl_id "$version_output")

  if [[ "$IMPL_ID" == openmpi-* ]]; then
    MPIEXEC_ARGS+=(--oversubscribe)
  fi

  echo "ferrompi MPI test runner"
  echo "  mpiexec:      $MPIEXEC ($IMPL_ID)"
  echo "  mpiexec args: ${MPIEXEC_ARGS[*]:-(none)}"
  echo "  np list:      ${MPI_NP_LIST_ARR[*]}"
  echo "  features:     ${FEATURES:-default}"
  echo "  timeout:      ${MPI_TEST_TIMEOUT}s"
  if ((VALGRIND_MODE)); then
    echo "  mode:         valgrind (suppressions=$VALGRIND_SUPP)"
  fi
  echo

  discover examples
  build "$FEATURES"

  local metadata
  metadata=$(cargo metadata --no-deps --format-version 1)
  local features_json
  features_json=$(jq -c '.packages[0].features' <<<"$metadata")

  local name req_csv
  while IFS='|' read -r name req_csv; do
    REQUIRED_FEATURES["$name"]="$req_csv"
  done < <(jq -r '.packages[0].targets[] | select(.kind==["example"]) | "\(.name)|\((."required-features" // []) | join(","))"' <<<"$metadata")

  local default_csv
  default_csv=$(jq -r '(.packages[0].features.default // []) | join(",")' <<<"$metadata")
  local requested_csv="$default_csv"
  if [[ -n "$FEATURES" ]]; then
    local -a explicit
    IFS=', ' read -ra explicit <<<"$FEATURES"
    local f
    for f in "${explicit[@]}"; do
      [[ -n "$f" ]] && requested_csv="${requested_csv:+$requested_csv,}$f"
    done
  fi
  FEATURE_CLOSURE=$(feature_closure "$requested_csv" "$features_json")

  run_all

  echo
  echo "======================================================================"
  echo "Summary"
  echo "======================================================================"
  echo "PASS: $PASS_COUNT"
  echo "SKIP: $SKIP_COUNT"
  echo "SKIP(feature): $SKIP_FEATURE_COUNT"
  echo "FAIL: $FAIL_COUNT"

  if ((${#SKIP_REASONS[@]} > 0)); then
    echo
    echo "SKIP reasons:"
    local r
    for r in "${SKIP_REASONS[@]}"; do
      echo "  $r"
    done
  fi

  if ((${#FAILED_RUNS[@]} > 0)); then
    echo
    echo "Failed runs:"
    local r
    for r in "${FAILED_RUNS[@]}"; do
      echo "  $r"
    done
  fi

  if ((FAIL_COUNT > 0 || RUNS_EXECUTED == 0)); then
    exit 1
  fi
  exit 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
