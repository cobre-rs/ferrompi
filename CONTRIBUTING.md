# Contributing to FerroMPI

This file lists the commands and policies FerroMPI's CI actually enforces.

## Getting started

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ferrompi.git`.
3. Create a branch: `git checkout -b feature/my-feature`.
4. Make your changes.
5. Run the commands in [Before opening a PR](#before-opening-a-pr).
6. Push and open a pull request.

## Setup

- Rust 1.85 or newer (the MSRV: `Cargo.toml`'s `rust-version`).
- An MPI library with its development headers, and `pkg-config`. See
  [`docs/mpi-compatibility.md`](docs/mpi-compatibility.md) to choose an
  implementation and select or override it at build time.

Ubuntu/Debian:

```bash
sudo apt-get install mpich libmpich-dev pkg-config
```

macOS (not CI-tested): `brew install mpich pkg-config`

## Before opening a PR

Run every command CI runs, with its exact flags:

```bash
cargo fmt --all -- --check
bash .github/scripts/check-no-plan-leaks.sh

cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features rma -- -D warnings
cargo clippy --all-targets --features numa -- -D warnings

cargo test --lib --features numa
cargo test --doc
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

bash tests/runner_selftest.sh
bash tests/build_probe.sh

MPI_NP_LIST="2 3 4" ./tests/run_mpi_tests.sh
MPI_NP_LIST="2 3 4" ./tests/run_mpi_tests.sh rma
./tests/run_mpi_tests.sh --valgrind rma   # optional, needs valgrind

cargo +1.85 build --locked --all-features --lib --tests --examples
```

Lint policy: `clippy::all` at `-D warnings`, not pedantic strictness. Every `unsafe`
block needs a `// SAFETY:` comment (`clippy::undocumented_unsafe_blocks` is denied).
Every public item needs a doc comment (`missing_docs`, fatal under `-D warnings`).

Shipped files (source, examples, benches, tests, docs, README, CHANGELOG) describe
behaviour, not how the work was planned: no work-item, requirement or audit-register
identifiers and no paths into planning notes. `check-no-plan-leaks.sh` enforces this;
commit messages are exempt. To run it with `cargo fmt --check` before every commit:
`ln -sf ../../.github/scripts/pre-commit .git/hooks/pre-commit`.

## Writing an MPI test

An MPI-exercising test is an `examples/test_*.rs` binary, discovered and run by
`tests/run_mpi_tests.sh`. It carries one directive comment:

```
// mpi-test: np=<N>|<N>.. [timeout=<s>] [skip-ok=<impl>[,<impl>]] [expect=abort|expect=unfinalized] [valgrind]
```

and at most one:

```
// mpi-test-stderr: <literal>
```

`mpi-test-stderr` is required when `expect=` is set; `expect=unfinalized` additionally
requires `np=1`.

Outcome rules:

- A run that times out, or a `--valgrind` run whose process exits 99, fails.
- A `SKIP: <reason>` line is reported as SKIP only when the running implementation
  matches one of `skip-ok`'s prefixes; otherwise it fails.
- When present, the `mpi-test-stderr` literal must appear in the run's output on
  every non-SKIP run, or the run fails.
- A run whose output contains `MPI_Finalize skipped` fails unless the example
  declares `expect=unfinalized`.
- An `expect=abort` run passes when it exits non-zero with the `mpi-test-stderr`
  literal in its output. One that exits 0 after a `SKIP:` line follows the SKIP
  rule above.

`examples/common/mod.rs` has the shared helpers: `check` (aggregate a per-rank
verdict via `allreduce(Min)` and report `FAIL: <name>` from rank 0), `skip` (print
`SKIP: <reason>` from rank 0), `mpi_major` (the running library's major version)
and `is_count` (match a `Count`-class MPI error).

## Benchmarks

Benchmarks run by hand; no CI job runs them. Build without running, then launch
the compiled binary under `mpiexec` yourself — running `mpiexec cargo bench`
starts one `cargo` process per rank instead of one coordinated run:

```bash
cargo bench --no-run
mpiexec -n 2 target/release/deps/<bench-binary>   # allreduce_roundtrip, persistent_vs_iallreduce
mpiexec -n 1 target/release/deps/<bench-binary>   # ffi_overhead runs as a singleton
```

See [`benches/README.md`](benches/README.md) for the binary names and output layout.

## CI

Pull requests to `main` and `develop` run:

- the test matrix (MPICH 4.2.1 and Open MPI 4.1.6, default and `rma` features),
  Open MPI 5.0.7 (`rma` only), the MSRV (1.85) build, the MPI Forum ABI-stubs
  build, the large-count example, code coverage, the documentation and package
  build, and a Valgrind memcheck run (`.github/workflows/test.yml`);
- `cargo audit` and a Dependency Review (`.github/workflows/security.yml`), which
  also run monthly on a schedule;
- automatic PR labeling by changed files (`.github/workflows/pr-labels.yml`).

## Changelog and releases

A user-visible change adds an entry under `## [Unreleased]` in `CHANGELOG.md`; do
not reference internal tracking IDs. The release commit renames that
heading to `## [X.Y.Z] - YYYY-MM-DD`. `.github/scripts/release-notes.sh <version>`
extracts that section; `publish.yml` runs it before publishing and fails the tag
if the section is missing or empty, and a CI fixture test requires notes for
every released heading.

## Adding features

For a large or architecture-changing feature, open an issue to discuss the
approach before implementing.

## Reporting issues

Include:

- Rust version (`rustc --version`)
- MPI version (`mpiexec --version`)
- Operating system
- A minimal reproduction
- Error messages and backtraces

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project (MIT OR Apache-2.0).
