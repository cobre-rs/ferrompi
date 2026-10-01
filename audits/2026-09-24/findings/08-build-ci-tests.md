# 08 — Build, packaging, CI and test infrastructure

## Command results at audit time (v0.5.0, MPICH 4.2.3, rustc 1.95)

| Command | Result |
|---|---|
| `cargo build --all-features` | pass (MPI via `mpicc` at /opt/mpich) |
| `cargo clippy --all-targets --all-features -- -D warnings` | pass |
| `cargo test --doc --all-features` | pass — 174 passed, 19 ignored; only 3 compiled doctests come from `docs/*.md` |
| `cargo doc --no-deps --all-features` | pass, 0 warnings |
| `cargo doc --no-deps` (default features = what docs.rs builds) | **13 warnings** (unresolved intra-doc links to `rma` items) |
| `cargo test --lib --all-features` | pass, 181 tests |
| `cargo fmt --check` | pass |
| `cargo +1.74 check --lib --all-features` | **FAIL** — 4 errors at `src/op.rs:160,233` (`#[unsafe(no_mangle)]`); passes with that attribute patched |
| CONTRIBUTING's `cargo clippy --lib -- -D warnings -W clippy::pedantic` | **FAIL** at `build.rs:60`; library has 331 pedantic warnings |
| `tests/run_mpi_tests.sh` unmodified, with `CARGO_TARGET_DIR` set | **"0 passed, 0 failed, 57 skipped … All tests passed!", exit 0** |
| runner patched to honour `CARGO_TARGET_DIR`: default np=4 / rma np=4 / rma np=8 | 57/57, 73/73, 73/73 |
| patched runner, rma np=3 and np=5 | 71/73 — `test_nonblocking`, `test_p2p_extra` time out (deadlock) |
| patched runner, rma np=1 | 65/73 — 8 tests assert size ≥ 2 |
| never-run examples (`gatherv`, `scan`, `pi_monte_carlo`, `topology`, `persistent_bcast`) at np=4 | all exit 0 |

Logs: [`../repros/evidence/`](../repros/README.md); patched runner: `../repros/tools/run_mpi_tests_patched.sh`.

---

### INF-01 — The MPI test runner passes when it ran nothing

- **Severity:** major · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `tests/run_mpi_tests.sh:118` (binary path hard-coded `./target/${BUILD_MODE}/examples/…`, ignores `CARGO_TARGET_DIR`), `:125-129` (missing binary → SKIP, `return 0`), `:264-269` (SKIP never fails); `tests/run_mpi_coverage.sh:111` same pattern.
- **Fix:** resolve binaries via `cargo build --message-format=json` (or honour `CARGO_TARGET_DIR`); missing binary = FAIL; print SKIP count and fail if an unexpected SKIP occurs.

### INF-02 — Declared MSRV is false; no MSRV job

- **Severity:** major · **Verified:** repro (`cargo +1.74`) · **Target:** 0.5.x
- **Locations:** `Cargo.toml:5` (`rust-version = "1.74"`), `src/op.rs:160, 233` (`#[unsafe(no_mangle)]` needs 1.82; introduced in b51378d, shipped in 0.4.1 and 0.5.0); `README.md:90`, `CONTRIBUTING.md:18`. Dev-dep `rand 0.10.1` declares `rust-version = "1.85"` (edition 2024) → examples/benches cannot build on the declared MSRV either.
- **Impact:** Cargo's MSRV-aware resolver selects 0.5.0 for users on 1.74–1.81, who then fail to compile.
- **Fix — decided (D-12, 2026-09-24):** declare `rust-version = "1.85"` (dev-deps such as `rand 0.10`/edition 2024 then also build on the MSRV); add one CI job on 1.85 covering lib, tests and examples; update `README.md:90`, `CONTRIBUTING.md:18`.
- **Library deps fit 1.74:** thiserror 1.68, cc 1.63, pkg-config 1.31, syn 1.71.

### INF-03 — `build.rs` never re-runs when the MPI selection changes

- **Severity:** major · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `build.rs:12-15` (only `rerun-if-changed=csrc/*`), `:83` (`MPI_PKG_CONFIG`), `:105` (`CRAY_MPICH_DIR`), `:152` (`MPICC`); `PATH` not tracked. The pkg-config crate does cover `PKG_CONFIG_*`.
- **Evidence:** rebuild with `MPI_PKG_CONFIG=bogus CRAY_MPICH_DIR=/bogus` → `Fresh ferrompi`; with `/opt/mpich/bin` removed from `PATH` → `Fresh`; `MPICC=/nonexistent/mpicc` → lib recompiles but the build script does not re-run and still links `-L native=/opt/mpich/lib -l mpi`.
- **Impact:** `module swap` on HPC systems silently keeps the old MPI.
- **Fix:** `cargo:rerun-if-env-changed=` for `MPICC`, `MPI_PKG_CONFIG`, `CRAY_MPICH_DIR`, `PATH`.

### INF-04 — `build.rs` probe precedence contradicts the docs; explicit overrides fail silently

- **Severity:** major · **Verified:** repro (decoy `.pc`, `repros/tools/decoy-mpich.pc.in`) · **Target:** 0.5.x
- **Locations:** `build.rs:81-113` — order is `MPI_PKG_CONFIG` → pkg-config `mpich/ompi/mpi` → `MPICC` → `CRAY_MPICH_DIR` → fixed prefixes; a failed `MPI_PKG_CONFIG` probe falls through silently. Docs: `docs/mpi-compatibility.md:395-399, 421, 428-436`, `README.md:476-490` (DOC-07).
- **Evidence:** decoy `mpich.pc` + `MPICC=/opt/mpich/bin/mpicc` → "Found MPI via pkg-config: mpich"; the C compile then picked a stray `/usr/local/include/mpi.h` and failed (`fatal error: mpi_proto.h`) — the shim is built with the system `cc` using whatever include path the probe found, so headers from one MPI can be mixed with another's libs.
- **Also:** `mpicc -show` is MPICH syntax; Open MPI's wrapper uses `--showme` (probe falls back to pkg-config `ompi`); Cray `cc` has no `-show`.
- **Fix:** explicit env vars take precedence (`CRAY_MPICH_DIR`, `MPICC`, `MPI_PKG_CONFIG` — order to be fixed in the plan) and a failing explicit variable is a hard error; try `-show` then `--showme`; correct the docs.

### INF-05 — `build.rs` dead or misleading output

- **Severity:** minor · **Target:** 0.5.x
- `:59-60` comment advertises `MPI_SKIP_LIBS` — never read (`SKIP_LIBS` hard-coded).
- `:67-70` `cargo:rustc-env=MPI_VERSION` — no consumer (`env!`/`option_env!` nowhere), set only on the pkg-config path, and holds the *package* version, not the MPI standard version.
- `:1-6` header says "Finds the MPICH installation".
- `:115-128` fixed-prefix fallback probes only `{prefix}/lib` (no `lib64`/multiarch) and always links `-lmpi`.
- `:130-135` panic message omits `MPICC`.
- `:33-35` `opt_level(3)` duplicates what `cc` derives from the profile; `:173-174` `#[allow(clippy::unnecessary_wraps)]` on a fn that never fails.
- `:40-47` no-rpath choice is fine but undocumented for users (DOC-08/DOC-14).

### INF-06 — No `links` manifest key

- **Severity:** minor · **Verified:** reasoning · **Target:** 0.5.x
- ferrompi exports global C symbols (`ferrompi_*`) and `#[no_mangle]` Rust symbols (`rust_user_op_invoke`…). Two semver-incompatible ferrompi versions in one graph would clash at link time or silently resolve to one version's symbols instead of Cargo's clear "links conflict". Fix: `links = "ferrompi"` (or `"mpi"`).

### INF-07 — docs.rs builds default features only

- **Severity:** major (public docs miss the RMA API) · **Verified:** live check · **Target:** 0.5.x
- No `[package.metadata.docs.rs]`; docs.rs `struct.Win.html`, `struct.SharedWindow.html`, `slurm/index.html` → 404; default-feature `cargo doc` gives 13 unresolved-link warnings (`src/lib.rs:76-87, 131-136`); CI doc builds use only `--features rma`/`--all-features`, so these are never caught.
- **Fix:** `[package.metadata.docs.rs] all-features = true` + `rustdoc-args = ["--cfg", "docsrs"]` with `doc(cfg)` badges; CI `cargo doc` at default features with `-D warnings`.

### INF-08 — Doctests never run on pull requests

- **Severity:** major · **Target:** 0.5.x
- `.github/workflows/test.yml:117` runs `cargo test --lib --features numa` only; doctests (174, incl. `compile_fail` seal/feature checks and `docs/*.md` snippets) run only in `publish.yml:79` at tag time, after merge.
- **Fix:** `cargo test --doc --all-features` and one default-feature doctest run in `test.yml`.

### INF-09 — MPI-4 capability probes turn regressions into silent passes

- **Severity:** major · **Target:** 0.5.x
- **Locations:** `examples/test_persistent.rs:34-42` and the same probe in `test_{gather,allgather,alltoall,scatter}_init_inplace.rs`: `bcast_init` → `Err(_) => { SKIP; return }` (exit 0); the runner shows output only on failure, so SKIPs are invisible in CI.
- **Consequences:** any regression making `bcast_init` fail on MPICH 4 passes five suites; the Open MPI leg (4.x, MPI 3.1) runs no MPI-4 code; `scatter_init_inplace` runs **nowhere** in CI (skipped on MPICH "4.2." — MPICH bug workaround — and on Open MPI 4).
- **Fix:** probe with `Mpi::version()`; fail on unexpected errors; runner counts and prints SKIPs with reasons.

### INF-10 — Tests deadlock at odd process counts; CI uses np=4 only

- **Severity:** minor · **Verified:** repro · **Target:** 0.5.x
- `examples/test_nonblocking.rs:155-199`, `examples/test_p2p_extra.rs:118-184`: even/odd pairing; with size = 3, rank 2 blocks in `recv` from rank 0, which never sends to it (rank 0 still prints "PASS: blocking send/recv" first). np=1: 8 tests abort on `size >= 2` asserts. `README.md:465` advertises `MPI_NP` as a free choice.
- **Fix:** skip unpaired ranks or pair cyclically; add np=3 (and np=2) to CI.

### INF-11 — No large-count integration test

- **Severity:** major (test gap for a headline feature) · **Target:** 0.5.x
- Large-count `_c` dispatch (`count > INT_MAX`, e.g. `csrc/ferrompi.c:1004-1008`) is advertised ("ideal for >2GB transfers", `README.md:16,34,44`) but never exercised; the two `*_count_overflow` examples only check the `MPI_ERR_COUNT` rejection via a fake count through raw FFI.
- **Fix:** opt-in (env-gated, memory-heavy) example transferring `u8 × (2³¹ + 16)` for send/recv, bcast, allreduce and one RMA op; verifies COR-05 too.

### INF-12 — Public APIs with no integration test; examples never run

- **Severity:** minor · **Target:** 0.5.x
- Untested: `Communicator::topology` (`gather_topology`, reworked in 0.5.0 PERF-04) — only `examples/topology.rs` with 0 asserts, never run; `abort`; `WinFenceAssert::no_check/no_precede/no_succeed`; `TopologyInfo::num_hosts`; `standard_version`; `slurm::*` (unit tests only). Never run by any runner: `gatherv`, `scan`, `pi_monte_carlo`, `topology`; `persistent_bcast` runs without timeout and its result is discarded (`:245-247`).

### INF-13 — Error-class assertions accept any class

- **Severity:** major (masked COR-01) · **Target:** 0.5.x
- `examples/test_errhandler_returns.rs:43-54`, `:95-104`; `examples/test_get_group_invalid_handle.rs:77-84` — catch-all arms print "PASS … class is implementation-defined". Fix together with COR-01: assert the exact class on both CI implementations.

### INF-14 — Release notes extraction yields empty bodies

- **Severity:** minor · **Verified:** repro + `gh release view` (v0.5.0 = "", v0.4.1 and v0.4.0 = 0 chars; re-verified by main session) · **Target:** 0.5.x
- `.github/workflows/publish.yml:165` `awk "/^## \[$VERSION\]/,/^## \[/" CHANGELOG.md | sed '$d'` — the header line matches both range patterns, so the range is that single line, which `sed '$d'` deletes.
- **Fix:** `awk -v v="$VERSION" 'f && /^## \[/ {exit} $0 ~ "^## \\[" v "\\]" {f=1} f' CHANGELOG.md`; backfill the three releases' notes.

### INF-15 — CI coverage gaps

- **Severity:** minor · **Target:** 0.5.x (except where noted)
- Clippy and integration runs cover default + `rma`; `numa` never linted with `-D warnings` in PR CI; PR doc build uses `rma` only; no MSRV job (INF-02); no sanitizer/Miri job; coverage (`tests/run_mpi_coverage.sh:185-205`) runs 11 of 73 examples with no timeout (group, custom datatype, UserOp, persistent p2p, 13 of 15 RMA tests excluded) → codecov understates; `fail_ci_if_error: false` (`test.yml:163`).

### INF-16 — MPICH hotfix step: unpinned downloads without checksum, copy-pasted ×4

- **Severity:** minor (supply-chain hygiene) · **Target:** 0.5.x
- `.github/workflows/test.yml:62-92, 159-181`, `publish.yml:29-52, 119-138`: `curl` of pinned `.deb`s from `archive.ubuntu.com/pool` with no checksum on unpinned `ubuntu-latest`; `publish.yml` uses `cargo package --allow-dirty` in a clean tag checkout.
- **Fix:** one composite action / script with `sha256sum -c`; pin the runner image; drop `--allow-dirty`.

### INF-17 — `security.yml` hygiene

- **Severity:** nit · **Target:** 0.5.x
- `:27` `cargo install cargo-audit --locked || true` hides install failures (audit silently skipped); `:15` `security-events: write` granted but unused; runs only for PRs to `main` (DOC-11).

### INF-18 — Packaging ships repo internals (and would ship `audits/`)

- **Severity:** minor · **Target:** 0.5.x
- No `include`/`exclude` in `Cargo.toml`: the published crate contains `.github/**`, `test.sh`, `.gitignore`, 78 examples, benches — and, since 2026-09-24, `audits/**` files outside the nested repro crates (`evidence/`, `tools/`, READMEs, findings). `docs/` **must stay** (`src/doc.rs` `include_str!`s it; a scratch package without it failed to build).
- **Fix:** an `include` allow-list (`src/**`, `csrc/**`, `docs/**`, `build.rs`, `README.md`, `CHANGELOG.md`, `LICENSE-*`, `Cargo.toml`) — decide examples/benches in the plan; verify with `cargo package --list`.

### INF-19 — ADR-0002's mandated TSan step does not exist

- **Severity:** minor · **Target:** 0.5.x (or amend the ADR; moot after ARC-01)
- `docs/adr/0002-handle-tables.md:365-377` requires a TSan CI step or a mandatory manual pre-release step in `CONTRIBUTING.md`/`docs/testing.md`; neither exists; only a comment in `examples/test_request_table_concurrency.rs:13`.

### INF-20 — CI covers MPICH 4.2 + Open MPI 4.x only

- **Severity:** minor · **Target:** 0.5.x (add Open MPI 5 leg) / later (ABI stub job ABI-07)
- No MPI-4 implementation other than MPICH is tested; COR-06 (`MPI_COMM_SELF` errhandler) and all MPI-4 paths are untested on a second implementation. Open MPI 5 (MPI 4.0) is the obvious second leg.

### INF-21 — Unit tests mutate a process-global static (latent)

- **Severity:** nit · **Verified:** stress-tested 300× with `--test-threads=2`, 0 failures · **Target:** none now
- `src/lib.rs:252` (`ATTACHED_BUFFER`), tests `:900`, `:928`. The `Mutex` + seed/clean pattern holds today; a future real-MPI attach test would need a dedicated test lock.

### INF-22 — Third-party GitHub Actions pinned by tag, not commit SHA (added 2026-09-24)

- **Severity:** minor (supply chain) · **Target:** later · **Source:** design council (CI brief).
- Tag pins are mutable; commit-SHA pins (with a comment naming the version) are the hardened practice. This reverses the tag style kept in commit `9175dbf`; dependabot can keep SHA pins current.

### INF-23 — Publishing uses a long-lived crates.io token (added 2026-09-24)

- **Severity:** minor (supply chain) · **Target:** later · **Source:** design council (CI brief).
- crates.io Trusted Publishing (OIDC from GitHub Actions) removes the stored `CARGO_REGISTRY_TOKEN` secret.
