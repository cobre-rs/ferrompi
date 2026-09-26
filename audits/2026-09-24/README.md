# ferrompi assessment — 2026-09-24 (v0.5.0, commit `755497b`)

Full-codebase assessment: soundness, MPI correctness, architecture, MPI-5 ABI
readiness, performance, AI bloat / overengineering, documentation, and
build/CI/test infrastructure. This directory is the **system of record** for the
findings; update the **Status** column below as work lands.

## How to use this directory

| Path | Content |
|---|---|
| `README.md` (this file) | method, decision log, roadmap, **finding register with status** |
| [`findings/01-soundness.md`](findings/01-soundness.md) | SND — safe Rust → UB |
| [`findings/02-correctness.md`](findings/02-correctness.md) | COR — MPI semantics, C shim, lifecycle bugs |
| [`findings/03-architecture-api.md`](findings/03-architecture-api.md) | ARC — structure and public API (with reversibility) |
| [`findings/04-mpi5-abi.md`](findings/04-mpi5-abi.md) | ABI — verified MPI-5 ABI facts, coupling points, migration path |
| [`findings/05-performance.md`](findings/05-performance.md) | PRF — measured overhead; paths verified efficient |
| [`findings/06-bloat-overengineering.md`](findings/06-bloat-overengineering.md) | BLT — removable/collapsible text, speculative machinery |
| [`findings/07-docs.md`](findings/07-docs.md) | DOC — documentation that is false or missing |
| [`findings/08-build-ci-tests.md`](findings/08-build-ci-tests.md) | INF — build.rs, packaging, CI, test runner/coverage |
| [`findings/09-verified-sound.md`](findings/09-verified-sound.md) | VER — checked and found correct (do not re-flag) |
| [`repros/`](repros/README.md) | runnable evidence: 5 standalone Cargo crates, C probes, logs, tools; each row maps to finding IDs with pre-fix output and expected post-fix behaviour |

Each finding records: severity, how it was verified, target release, exact
locations (`file:line` at commit `755497b` — line numbers drift as code changes;
search by symbol if a line no longer matches), defect, evidence, fix direction,
and — where applicable — an **acceptance** check that proves it fixed.

**Updating status:** when a finding is fixed, set its Status to `fixed (<commit or PR>)`;
when a plan takes it, `planned (<plan name>)`; `wont-fix (<reason>)` or
`superseded (<ID>)` otherwise. Delete the matching repro once a regression test
exists in `examples/`/`src/` (see `repros/README.md`). Do not edit finding text to
match a fix — the text records what was true at `755497b`.

## Method

- Seven independent reviewers ran in parallel (C/MPI correctness, Rust soundness,
  architecture, AI bloat + overengineering, docs/build/CI, performance, MPI-5 ABI);
  the main session re-verified the headline findings (COR-01, COR-02, SND-06,
  INF-01, INF-02, INF-14, BLT-15) and spot-checked ABI facts against the MPI Forum
  reference header.
- Environment: Fedora x86_64, i7-12700KF, MPICH 4.2.3 (`/opt/mpich`, ch4:ofi),
  rustc 1.95; MSRV checks with rustc 1.74.
- Verification levels used in the findings: **repro** (program in `repros/`,
  observed output recorded), **measured**, **reading** (confirmed in code/standard),
  **plausible** (not confirmed).
- Severity: **critical** = safe code → UB/memory corruption/data race, or silent
  data corruption; **major**; **minor**; **nit**; **info**.
- No repository code was modified by the assessment.

## Bottom line

The crate has a solid core (sealed datatype traits, sound UserOp trampolines,
`MPI_ERRORS_RETURN` everywhere, ~1 ns blocking-path overhead, clean clippy/fmt), but:

1. **~13 ways for safe code to cause UB** (SND-01…13), all but one reproduced —
   heap overflow from unvalidated collective arguments, use-after-free from
   buffer lifetimes, a miscompiled shared-window reader, data races under
   `ThreadLevel::Single`.
2. **Errors misreport on MPICH-family MPIs** (COR-01) and **request handles alias
   unrelated requests** after reuse (COR-02/03).
3. **The C request table is the architectural pivot** (ARC-01): it causes the
   handle ABA, the main per-call overhead (+58% on small nonblocking p2p), fixed
   caps, and it is exactly what the MPI-5 ABI makes unnecessary.
4. **~6,600 lines (~14%) of bloat** (BLT), much of it test theater and comment
   restatement.
5. **CI can pass without testing anything** (INF-01), MSRV is false (INF-02),
   docs.rs lacks the RMA API (INF-07), and several docs are false (DOC).

## Decision log

| ID | Decision | Status | Where it applies |
|---|---|---|---|
| D-1 | **Buffer-safety model:** `PersistentRequest` and `Win::create` take **ownership** of their buffers (access only while inactive; `into_*` returns them). Nonblocking `Request`s and RMA epochs use **closure scopes**. Rejected: lifetime-only handles (defeated by `mem::forget`, SND-05), `unsafe fn` constructors. | **accepted 2026-09-24** | 0.6 — SND-01…05, ARC-06 |
| D-2 | **Handle tables → by-value handles:** store MPI handles in Rust as opaque 8-byte values; stateless C shim; global finalized flag for late drops. | **accepted 2026-09-24** | 0.7, before the ABI backend — ARC-01, COR-02, PRF-01/02 |
| D-3 | **Thread level:** runtime check now (record init thread + provided level; reject non-init-thread calls below `Serialized`); type-level encoding deferred. | **accepted 2026-09-24** | 0.5.x — SND-11, ARC-05 |
| D-4 | **`Info`:** wire `Option<&Info>` into window constructors and `split_type` **if** the SDDP stack needs `alloc_shared_noncontig` or hardware-guided splits; otherwise delete. | rule accepted; **open:** SDDP requirement answer | 0.6 — ARC-12 |
| D-5 | **`SharedWindow` merged into `Win`** with one deprecation cycle. | **accepted 2026-09-24** | 0.6 — ARC-13, SND-13 |
| D-6 | **0.5.x scope:** every finding targeted 0.5.x **plus** the non-breaking bloat removals (BLT items marked 0.5.x†). | **accepted 2026-09-24** | 0.5.x plan scope |
| D-7 | **Bound for `*_custom` `T`** (SND-08): new public `unsafe trait PlainData: Copy + 'static` that users implement for their `#[repr(C)]` types (blanket-implemented for all `MpiDatatype` types), plus a size/extent check. Rejected: `unsafe fn` receivers, byte-slice API. | **accepted 2026-09-24** | 0.5.x |
| D-8 | **Window memory at finalize** (SND-10): finalize does **not** free windows that are still alive — memory leaked + stderr warning; **extended 2026-09-24 (council A1):** because Open MPI frees live windows inside `MPI_Finalize` itself, `MPI_Finalize` is **skipped** (stderr warning) whenever an MPI-allocated window (`Win::allocate`, `SharedWindow`) is still alive. Rejected: accessors panic after finalize; refcount-deferred finalize; `MPI_Abort`; pulling `Win<'mpi>` forward. | **accepted 2026-09-24** | 0.5.x |
| D-9 | **`Serialized` overlap detection:** atomic in-call flag compiled only under `debug_assertions` (returns `Err` on overlap); release builds pay nothing. The below-`Serialized` init-thread check is always on. | **accepted 2026-09-24** | 0.5.x — SND-11 |
| D-10 | **MPI-5 ABI path:** 0.5.x build.rs fixes + reject draft ABI; 0.6 protect one-way doors; 0.7 shim compiled against the ABI header (auto-detected) after D-2; Rust-native backend only on a concrete requirement. | **accepted 2026-09-24** | ABI-01…07 |
| D-11 | **RMA remote bounds** (SND-09): window creation (already collective) allgathers each rank's exposed length once (8·P bytes per window); every put/get/accumulate is bounds-checked against the target rank. | **accepted 2026-09-24** | 0.5.x |
| D-12 | **MSRV** (INF-02): declare `rust-version = "1.85"` (dev-deps such as `rand 0.10`/edition 2024 then build on the MSRV too); one CI job on 1.85 covers lib, tests, examples. Rejected: keep 1.74 via `#[no_mangle]`; 1.82. | **accepted 2026-09-24** | 0.5.x |
| D-13 | **Error shape for the hardening release:** add `Error::ThreadLevelViolation` and `Error::Finalized`; `cancel` on non-p2p requests → existing `NotSupported`; mark `Error` `#[non_exhaustive]` now (rest of ARC-02 stays in the API milestone). | **accepted 2026-09-24** | 0.5.x — SND-11, COR-07, COR-10 |
| D-14 | **Release numbering:** the hardening milestone (labelled `0.5.x` in this register) ships as **0.6.0** (it contains semver-breaking changes); the register's `0.6` milestone ships as 0.7.0 and `0.7` as 0.8.0. Labels in this file are milestone names, kept unchanged. | **accepted 2026-09-24** | all |
| D-15 | **`Drop` on a non-init thread below `Serialized`:** print a diagnostic (type + thread) and abort the process — never call MPI from the wrong thread, never leak an active request. | **accepted 2026-09-24** | 0.5.x — SND-11 |
| D-16 | **Ratified MPI-5 ABI detected at build time:** build normally, document as runtime-untested until the ABI milestone; add the ABI-07 stub compile+link CI job now. No ABI-specific cfg or code path in 0.5.x. | **accepted 2026-09-24** | 0.5.x — ABI-02, ABI-07 |
| D-17 | **Sanitizers:** amend ADR-0002 to drop the TSan mandate; add a valgrind memcheck CI job (MPICH leg) over the soundness regression examples with an MPICH suppression file. | **accepted 2026-09-24** | 0.5.x — INF-15, INF-19 |
| D-18 | **Package contents:** allow-list (`src/`, `csrc/`, `docs/`, `build.rs`, README, CHANGELOG, licenses); `examples/`, `benches/`, `audits/`, `.github/`, `tests/`, `test.sh` excluded. | **accepted 2026-09-24** | 0.5.x — INF-18 |
| D-19 | **`LongDoubleInt`/`LongInt`:** cfg-gated to verified targets (Linux x86_64/aarch64/ppc64le); Windows unsupported; per-target layouts deferred to the API milestone. | **accepted 2026-09-24** | 0.5.x — COR-11 |
| D-20 | **Discriminant "semver contract"** retracted now in docs only (architecture.md + ADR-0003 amendment note). | **accepted 2026-09-24** | 0.5.x — DOC-01, ARC-03 |

## Roadmap (accepted 2026-09-24)

Release numbers per D-14: milestone `0.5.x` → **0.6.0**, `0.6` → **0.7.0**, `0.7` → **0.8.0**.


- **0.5.x — non-breaking fixes** (soundness fixes may tighten behaviour: new `Err`s where UB used to occur): error classes in C; null sentinel + generation counter for completed requests and write-back on error; all missing size/range validation; `MPI_ERR_COUNT` instead of truncation; `COMM_SELF` errhandler; finalized flag (re-init → `Err`, op-table sweep, window-accessor guard); zero window memory; thread-level runtime check; MSRV; build.rs rerun tracking, `-D` pass-through, precedence; runner skips fail; doctests in PR CI; release-notes `awk`; docs.rs all-features; stale-docs purge. Scope (D-6): every register row targeted `0.5.x` or `0.5.x†`.
- **0.6 — breaking API work:** D-1 buffer-safety model; `#[non_exhaustive]` + additive `rma`; deprecate `raw_handle`, retract discriminant contract; ops/datatypes as parameters; structured errors; `Status` from waits; typed `PROC_NULL`/`ANY`; `SharedWindow` → `Win`; D-4 `Info`; `numa` → `slurm`; remaining (breaking) bloat.
- **0.7 — internals + ABI:** D-2 by-value handles (removes tables, ABA, caps, per-request cost); build against the ABI `mpi.h` when detected.

## Finding register

Status values: `open` · `planned (<plan>)` · `fixed (<ref>)` · `wont-fix (<reason>)` · `superseded (<ID>)`.
Target: release the fix belongs to (`0.5.x†` = non-breaking bloat, in 0.5.x scope per D-6).

### Soundness — [01](findings/01-soundness.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| SND-01 | Nonblocking `Request` not tied to its buffer | critical | repro | 0.6 (D-1) | open |
| SND-02 | `PersistentRequest` not tied to its buffer | critical | repro | 0.6 (D-1) | open |
| SND-03 | RMA origin buffers not tied to the epoch; 4 rustdoc examples are UB | critical | reading | 0.6 (D-1) | open |
| SND-04 | `PendingFetchResult` dropped before epoch close → write into freed heap | critical | repro | 0.6 (D-1) | open |
| SND-05 | `mem::forget(Win::create)` leaves MPI aliasing a released buffer | critical | repro | 0.6 (D-1) | open |
| SND-06 | gather/allgather/scatter never validate buffer sizes (9 methods) | critical | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| SND-07 | V-collectives don't validate counts/displs vs size and buffer | critical | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| SND-08 | `*_custom` p2p: unbounded `T`, unchecked extent | critical | repro | 0.5.x (D-7) | planned (ferrompi-0.5.x-hardening) |
| SND-09 | RMA target range / origin count unvalidated (remote OOB write) | critical | repro | 0.5.x (D-11) | planned (ferrompi-0.5.x-hardening) |
| SND-10 | Window memory used after finalize | critical | repro | 0.5.x (D-8) | planned (ferrompi-0.5.x-hardening) |
| SND-11 | `Communicator` Send+Sync regardless of thread level | critical | repro | 0.5.x (D-3, D-9) | fixed (49cb30c) |
| SND-12 | Uninitialised window memory exposed as `&[T]` | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| SND-13 | `SharedWindow` slices over concurrently-written memory (observed miscompile) | critical | repro | 0.6 (D-5); 0.5.x doc warning | planned (ferrompi-0.5.x-hardening: doc warning); API fix open (0.6) |
| SND-14 | `UserOp` fat-pointer transmute relies on unspecified layout | minor | reading | 0.5.x† | fixed (2d53b18) |
| SND-15 | `fetch_and_op`/`compare_and_swap` result pointer derived from a shared borrow | minor | reading | 0.5.x | fixed (9777b12) |
| SND-16 | `Mpi` drop racing a concurrent guarded call at `Serialized`/`Multiple` reaches MPI after finalize | major | reading | 0.6 | open (0.6: thread-safety API redesign) |

### Correctness — [02](findings/02-correctness.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| COR-01 | Error classes decoded with Open MPI numbering (wrong on MPICH) | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-02 | Stale request handles act on unrelated requests (ABA) | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-03 | No request write-back on error in wait/test-many | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-04 | `MPI_STATUSES_IGNORE` loses per-request error | minor | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-05 | Counts > `INT_MAX` silently truncated | critical | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-06 | `MPI_ERRORS_RETURN` not on `MPI_COMM_SELF` | major | reading | 0.5.x | fixed (19daebb) |
| COR-07 | Finalize/re-init lifecycle aborts; `UserOp` drop after finalize | major | repro | 0.5.x | fixed (e1c123c) |
| COR-08 | Finalize sweep frees active requests / collectively frees windows | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-09 | `-1` = PROC_NULL (MPICH) vs ANY_SOURCE (Open MPI) | minor | repro | 0.6 | open |
| COR-10 | `cancel()` allowed on collective/RMA requests | minor | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-11 | `LongDoubleInt`/`LongInt` layout wrong on macOS arm64 / Windows | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening: target gating); per-target layouts open (0.6) |
| COR-12 | Pre-MPI-4 stubs never yield `NotSupported`; 3 contradicting docs | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-13 | `Win::sync` rustdoc wrong; example fails on MPICH | minor | repro | 0.5.x docs / 0.6 API | planned (ferrompi-0.5.x-hardening: docs); sync on lock guards open (0.6) |
| COR-14 | `MPI_UNDEFINED` from `MPI_Get_count` leaks; rc ignored | nit | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| COR-15 | `op_set_closure` no bounds check; `op_create_user` no `op_used` check | nit | reading | 0.5.x | fixed (2d53b18) |
| COR-16 | `type_create_struct` maybe-uninitialised arrays at count 0 | nit | compiler | 0.5.x | planned (ferrompi-0.5.x-hardening) |

### Architecture / API — [03](findings/03-architecture-api.md)

| ID | Title | Sev | Reversibility | Target | Status |
|---|---|---|---|---|---|
| ARC-01 | C handle tables are the root cause of ABA, per-request cost, caps, sweep | major | two-way (large) | 0.7 (D-2) | open |
| ARC-02 | Non-additive `rma` feature; no `#[non_exhaustive]` anywhere | major | one-way | 0.6 | planned (ferrompi-0.5.x-hardening: #[non_exhaustive] on Error); rest open (0.6) |
| ARC-03 | Shim representation in public API (`raw_handle`, discriminant contract, pub `from_code`) | major | one-way | 0.6 | open |
| ARC-04 | Handles not tied to `Mpi` lifetime | major | mixed | 0.5.x guards / 0.6 | planned (ferrompi-0.5.x-hardening: runtime guards); lifetime parameter open (0.6) |
| ARC-05 | Thread-safety model inconsistent | major | mixed | 0.5.x (D-3) | fixed (49cb30c) |
| ARC-06 | Buffer-safety model (umbrella SND-01…05) | critical | one-way | 0.6 (D-1) | open |
| ARC-07 | Ops and datatypes not parameters | major | one-way | 0.6 | open |
| ARC-08 | Error model misleads and loses context | major | one-way | 0.6 | open |
| ARC-09 | `Status` always discarded | major | one-way | 0.6 | open |
| ARC-10 | Copy-pasted families drifted (root of SND-06) | major | two-way | 0.5.x (validators) | planned (ferrompi-0.5.x-hardening) |
| ARC-11 | Capability gaps (in-place nonblocking, mprobe, HW_GUIDED…) | minor | additive | 0.6+ | open |
| ARC-12 | `Info` public but unused | minor | one-way | 0.6 (D-4) | open |
| ARC-13 | `SharedWindow` duplicates `Win`; `WinKind` dead | minor | one-way | 0.6 (D-5) | open |
| ARC-14 | `numa` feature has no NUMA code, implies `rma` | minor | one-way | 0.6 | open |
| ARC-15 | Naming/coverage asymmetries | minor | one-way | 0.6 | open |
| ARC-16 | Magic `-1` sentinels; no typed source/tag | minor | one-way | 0.6 | open |
| ARC-17 | RMA redundant `target_count`, duplicate tags | minor | one-way | 0.6 | open |
| ARC-18 | Open MPI 5 capabilities unused (persistent collectives, `create_from_group` gated on `MPI_VERSION >= 4`; OMPI 5 reports 3.1) | minor | two-way | 0.6 | open |

### MPI-5 ABI — [04](findings/04-mpi5-abi.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| ABI-01 | build.rs drops `-D` flags (`MPI_ABI` lost → silent ABI mismatch) | critical (latent) | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| ABI-02 | No ABI detection probe; MPICH 4.3 draft ABI accepted | major | reading | 0.5.x reject / 0.7 build | planned (ferrompi-0.5.x-hardening: reject draft ABI); ABI build open (0.7) |
| ABI-03 | Build-selection env vars not tracked (= INF-03) | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| ABI-04 | Public-API one-way doors (= ARC-02/03) | major | reading | 0.6 | open |
| ABI-05 | ADR-0001 misstates what the ABI adds; ADR-0006 needed | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| ABI-06 | No interop API (deliberately deferred) | info | — | after ABI backend | open |
| ABI-07 | Optional CI compile+link job against Forum ABI stubs | info | built locally | 0.5.x or 0.7 | planned (ferrompi-0.5.x-hardening) |

### Performance — [05](findings/05-performance.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| PRF-01 | Request table: ~14 ns / 2 locked RMWs per request (+58%/+33% small nonblocking p2p) | major | measured | 0.7 (D-2); no 0.5.x stop-gap | open |
| PRF-02 | Bitmap concentrates contention; ADR/comment claim the opposite | minor | measured | 0.5.x docs / 0.7 | planned (ferrompi-0.5.x-hardening: docs); table change open (0.7) |
| PRF-03 | `ffi_overhead` bench cannot measure FFI overhead | minor | measured | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| PRF-04 | "Persistent 10–30% faster" refuted as stated; bench at 1 MiB only | minor | measured | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| PRF-05 | `start_all`/`wait_all` zero 512 B scratch per call | nit | measured | — | wont-fix (adds `unsafe` for ~8 ns; rejected in 0.5.x planning) |
| PRF-06 | Dead per-callback tag lookup; `Vec` per `wait_some`; topology 256·P | nit | reading | 0.5.x | fixed (2d53b18) for the callback tag lookup; Vec return and topology 256·P deferred |
| PRF-07 | `[profile.release]` doesn't reach downstream; comment says it does | minor | Cargo semantics | 0.5.x | planned (ferrompi-0.5.x-hardening) |

### Bloat / overengineering — [06](findings/06-bloat-overengineering.md)

| ID | Title | ~Lines | Target | Status |
|---|---|---:|---|---|
| BLT-01 | 76/181 unit tests catch no regression; 16 compile witnesses | 1,000 | 0.5.x† | fixed (99f3f9d) |
| BLT-02 | Private C header restates signatures in Doxygen | 800 | 0.5.x† | fixed (f09d912) |
| BLT-03 | `MPI_VERSION < 3` branches unreachable | 455 | 0.5.x† | fixed (7c6daa0) |
| BLT-04 | 12 near-identical in-place example binaries | 450 | 0.5.x† | fixed (25b1b5d) |
| BLT-06 | Example scaffolding duplicated in 26 files | 390 | 0.5.x† | fixed (f503dbf) |
| BLT-08 | Boilerplate SAFETY / marshalling; 59 `unsafe` blocks without SAFETY | 300 | 0.5.x† | fixed (0647d69) |
| BLT-09 | Three drifted test runners + deprecated `test.sh` | 200 | 0.5.x† | fixed (a0501aa) |
| BLT-10 | 42 redundant `[[example]]` entries | 185 | 0.5.x† | planned (ferrompi-0.5.x-hardening) |
| BLT-11 | Tables/indexes repeated across 3–5 docs; marketing tone | 250 | 0.5.x† | planned (ferrompi-0.5.x-hardening) |
| BLT-14 | 15 in-place C shims differ only by `MPI_IN_PLACE`; dead `is_root` | 300 | 0.5.x† | fixed (38c789e) |
| BLT-15 | `UserOp` double registry + dead per-callback lookup | 120 | 0.5.x† | fixed (2d53b18) |
| BLT-16 | Benches measuring nothing; duplicated bench protocol | 210 | 0.5.x† | planned (ferrompi-0.5.x-hardening) |
| BLT-20 | Six copies of the slot-claim loop | 70 | superseded by ARC-01 | open |
| BLT-21 | Process artifacts, stale line refs, expired promises in comments | 45 | 0.5.x† | fixed (3d206a8) |
| BLT-23 | `Group::undefined()` FFI call returning literal −1 | 35 | 0.5.x† | fixed (e874118) |
| BLT-25 | Dead C branches (errhandler install) | 30 | 0.5.x† | fixed (7389231) |
| BLT-26 | `const _` Send/Sync asserts beside `unsafe impl` | 27 | 0.5.x† | fixed (c074ed1) |
| BLT-27 | Dead `ferrompi_init`; module-wide `allow(dead_code)` | 25 | 0.5.x† | fixed (3cde221) |
| BLT-29 | `with_handles` implemented twice | 20 | 0.5.x† | fixed (2b1237e) |
| BLT-32 | `ReduceOp` compile_fail doctest via 14 `cfg_attr` | 16 | 0.6 (with ARC-02) | open |
| BLT-34 | `use super::*` in 11 test modules | — | 0.5.x† | fixed (99f3f9d) |

### Documentation — [07](findings/07-docs.md)

| ID | Title | Sev | Target | Status |
|---|---|---|---|---|
| DOC-01 | `docs/architecture.md` false statements | major | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-02 | ADR-0001 driver 1 false (= ABI-05) | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-03 | ADR-0002 claims that do not hold | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-04 | ADR-0004 lifetime rejection mis-argued; nonexistent variant | minor | 0.5.x note / 0.6 new ADR | planned (ferrompi-0.5.x-hardening: note); new ADR open (0.6) |
| DOC-05 | ADR-0005 Decision 7 describes the rejected design; plan sections | major | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-06 | Migration guide: nonexistent APIs, false safety claim | major | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-07 | `docs/mpi-compatibility.md` inaccuracies | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-08 | `README.md` inaccuracies (badge, version, reqs, `LD_LIBRARY_PATH`) | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-09 | Crate rustdoc (`src/lib.rs`) inaccuracies | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-10 | Stale/wrong item-level rustdoc | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-11 | CONTRIBUTING template filler + false policies | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-12 | CHANGELOG internal IDs / restated rustdoc | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-13 | `benches/README.md` claims | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-14 | Missing docs: lifecycle, error reporting, runtime lib path, Open MPI build | minor | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| DOC-15 | Wrong C comments | nit | 0.5.x | planned (ferrompi-0.5.x-hardening) |

### Build / CI / tests — [08](findings/08-build-ci-tests.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| INF-01 | MPI test runner passes when it ran nothing | major | repro | 0.5.x | fixed (87ac8d9) |
| INF-02 | Declared MSRV false; no MSRV job | major | repro | 0.5.x (D-12: 1.85) | planned (ferrompi-0.5.x-hardening) |
| INF-03 | build.rs never re-runs on MPI selection change | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-04 | build.rs precedence contradicts docs; overrides fail silently | major | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-05 | build.rs dead/misleading output | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-06 | No `links` manifest key | minor | reasoning | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-07 | docs.rs builds default features only (no RMA docs) | major | live check | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-08 | Doctests never run on PRs | major | reading | 0.5.x | fixed (acd809f) |
| INF-09 | MPI-4 probes turn regressions into silent passes | major | reading | 0.5.x | fixed (87ac8d9) |
| INF-10 | Odd-np deadlocks; CI np=4 only | minor | repro | 0.5.x | fixed (6ee2c4e) |
| INF-11 | No large-count integration test | major | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-12 | Public APIs untested; examples never run | minor | reading | 0.5.x | fixed (da87ce5) |
| INF-13 | Error-class assertions accept any class | major | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-14 | Release notes extraction yields empty bodies | minor | repro | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-15 | CI coverage gaps (numa clippy, sanitizers, coverage 11/73) | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-16 | MPICH hotfix: unchecked downloads, copy-pasted ×4 | minor | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-17 | `security.yml` hygiene | nit | reading | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-18 | Package ships repo internals (incl. `audits/`) | minor | `cargo package` | 0.5.x | planned (ferrompi-0.5.x-hardening) |
| INF-19 | ADR-0002 mandated TSan step missing | minor | reading | 0.5.x / moot after ARC-01 | planned (ferrompi-0.5.x-hardening) |
| INF-20 | CI covers MPICH 4.2 + Open MPI 4.x only (no Open MPI 5) | minor | reading | 0.5.x | fixed (3f8ec50) |
| INF-21 | Unit tests mutate a global static (latent) | nit | stress test | — | open |
| INF-22 | Third-party GitHub Actions pinned by tag, not commit SHA | minor | reading | later | open |
| INF-23 | Publishing uses a long-lived crates.io token (no Trusted Publishing) | minor | reading | later | open |
