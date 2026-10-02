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
| [`repros/`](repros/README.md) | runnable evidence: 3 standalone Cargo crates, C probes, logs, tools; each row maps to finding IDs with pre-fix output and expected post-fix behaviour |

Each finding records: severity, how it was verified, target release, exact
locations (`file:line` at commit `755497b` — line numbers drift as code changes;
search by symbol if a line no longer matches), defect, evidence, fix direction,
and — where applicable — an **acceptance** check that proves it fixed.

**Updating status:** when a finding is fixed, set its Status to `fixed (<commit or PR>)`;
when a plan takes it, `planned (<plan name>)`; `wont-fix (<reason>)` or
`superseded (<ID>)` otherwise. Delete the matching repro once a regression test
exists in `examples/`/`src/` (see `repros/README.md`). Do not edit finding text to
match a fix — the text records what was true at `755497b`. Repros deleted after
their fix stay in git history: `git show e7e2e90:audits/2026-09-24/repros/<path>`
prints one.

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
| D-4 | **`Info`:** wire `Option<&Info>` into window constructors and `split_type` **if** the SDDP stack needs `alloc_shared_noncontig` or hardware-guided splits; otherwise delete. | rule accepted; **decided 2026-10-01: delete (D-25)** | 0.6 — ARC-12 |
| D-5 | **`SharedWindow` merged into `Win`** with one deprecation cycle. | **accepted 2026-09-24**; deprecation cycle superseded (D-26) | 0.6 — ARC-13, SND-13 |
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
| D-21 | **0.6.0 close-out:** COR-17, COR-19, COR-20, SND-17, INF-22, ARC-18 and the `WinKind` part of ARC-13 move to the `0.5.x` milestone and ship in 0.6.0; for COR-19 this supersedes the 2026-09-28 ruling that kept the gap as a tracked open row; for SND-17 the fix contract becomes an abort of the process, replacing the finding's still-pending fix direction. | **accepted 2026-09-29** | 0.5.x — COR-17, COR-19, COR-20, SND-17, INF-22, ARC-18, ARC-13, COR-21 |
| D-22 | **0.6.0 follow-ups:** SND-18, COR-22 and COR-23 are opened for gaps the close-out reviews found and ship in 0.6.0; for COR-22, a rank-local failure after MPI created a communicator or window on every rank leaks that object instead of freeing it on one rank (the COR-19 leak policy generalized). | **accepted 2026-09-29** | 0.5.x — SND-18, COR-22, COR-23 |
| D-23 | **0.6.0 correctness fixes:** COR-24, COR-25, SND-19 and SND-20 are opened for pre-existing gaps the follow-up reviews found and ship in 0.6.0; `Communicator::topology` returns `Err` on every rank when a local query fails on any rank; `Mpi::wtime` takes `&self` (breaking); `allreduce_with_op` returns `Err(Count)` above `INT_MAX` elements on every MPI; the fixed string buffers are checked against the `MPI_MAX_*` constants at build time. | **accepted 2026-09-30** | 0.5.x — COR-24, COR-25, SND-19, SND-20, COR-26 |
| D-24 | **Release numbers for open work; 0.6.0 final fixes:** open register rows name the release their fix ships in, not a milestone label: milestone `0.6` becomes `0.7.0` and `0.7` becomes `0.8.0`, in Target cells and in the open remainders of Status cells. This supersedes, for open rows, D-14's rule that labels are milestone names kept unchanged; fixed rows keep the `0.5.x` milestone they were fixed under (shipped as 0.6.0), and decision rows D-1 to D-23 and `findings/` keep D-14's milestone names. SND-21, DOC-16, INF-24 and BLT-35 move into 0.6.0; INF-26 is opened for the two `SharedWindow` examples INF-24 leaves, targeted 0.7.0 with ARC-13. | **accepted 2026-09-30** | all — SND-21, DOC-16, INF-24, BLT-35, INF-26, COR-27, INF-27 |
| D-25 | **`Info` deleted (D-4 resolved):** the SDDP stack needs neither `alloc_shared_noncontig` nor hardware-guided splits (operator, 2026-10-01), so D-4's rule resolves to delete: the public `Info` type, its C table and FFI declarations, `ResourceKind::Info`, its error plumbing and `examples/test_info.rs` are removed. | **accepted 2026-10-01** | 0.7.0 — ARC-12, D-4 |
| D-26 | **`SharedWindow` removed in 0.7.0:** supersedes D-5's one deprecation cycle. A deprecated `SharedWindow` would ship the critical slice aliasing (SND-13) for a release, and every `Win` signature breaks in 0.7.0 anyway; `Win::allocate_shared` with per-rank shared views replaces it, and the 0.6 → 0.7 migration guide documents the move. | **accepted 2026-10-01** | 0.7.0 — ARC-13, SND-13, INF-26, D-5 |
| D-27 | **Window memory access model, every window:** no safe `&[T]`/`&mut [T]` over window memory that another process or an RMA call may write while the borrow lives. A safe view with Relaxed atomic element access and element-wise bulk copies is the default on every window, and a build fails at compile time on a target where an element type does not fit its atomic; `unsafe` zero-copy slice accessors for local and remote regions carry a written synchronization contract; remote regions of shared windows are cached from `MPI_Win_shared_query`. Issue #28's per-region `RefCell` borrow tracking is not adopted: it coordinates borrows within one process only and cannot exclude peer writes. The fix covers `Win::local_slice`/`local_slice_mut` too (SND-22). | **accepted 2026-10-01** | 0.7.0 — SND-13, SND-22, ARC-13 |
| D-28 | **Publish-then-read-only window deferred:** a collective `freeze` that turns a shared window into a read-only window with safe zero-copy `&[T]` reads is additive future work, for when the SDDP stack adopts node-shared windows; 0.7.0 ships the safe view and the `unsafe` slice accessors only. | **accepted 2026-10-01** | later — SND-13 |
| D-29 | **Typed reduction-op classes:** collective ops (predefined plus `UserOp`), accumulate ops (adding `Replace`) and fetch ops (adding `NoOp`) are separate types, and `MaxLoc`/`MinLoc` leave `ReduceOp` and exist only as pair-gated constants of the collective op type, so MPI's op rules become compile errors; enum shapes are identical with and without `rma`. This deviates from ARC-02's "`Replace`/`NoOp` unconditional in `ReduceOp`"; BLT-32's doctest is deleted either way. | **accepted 2026-10-01** | 0.7.0 — ARC-02, ARC-07, BLT-32 |
| D-30 | **`ThreadLevel` stays exhaustive:** MPI closes the set of thread levels, the reason ARC-02 keeps `LockType` and `GroupComparison` exhaustive; this departs from ARC-02's list. | **accepted 2026-10-01** | 0.7.0 — ARC-02 |
| D-31 | **No `HW_GUIDED` split:** `MPI_COMM_TYPE_HW_GUIDED` needs an `mpi_hw_resource_type` info hint (without one, MPICH 4.2.3 returns `MPI_COMM_NULL` on every rank) and `Info` is deleted (D-25); no consumer needs it, and `SplitType` becomes `#[non_exhaustive]`, so it can be added later without a break. | **accepted 2026-10-01** | 0.7.0 — ARC-11 (wont-fix part) |
| D-32 | **`LongInt`/`LongDoubleInt` per-target layouts not pursued:** D-19's Linux x86_64/aarch64/ppc64le gating stays; the SDDP stack is Linux-only and no CI leg exists for other targets. | **accepted 2026-10-01** | COR-11 (remainder wont-fix) |
| D-33 | **No datatype parameter on collectives in 0.7.0:** custom datatypes stay point-to-point only (`*_custom`); the SDDP stack exchanges primitive arrays and variable-length byte buffers that no derived datatype describes; ARC-07's op parameter stays in scope. | **accepted 2026-10-01** | later — ARC-07 (datatype part) |
| D-34 | **Handles stay `'static`; finalize safety at runtime:** no `'mpi` lifetime on any handle, since an owning struct holding `Mpi` and its communicators, as the SDDP stack uses, would become self-referential; only the buffer-safety scopes carry lifetimes. ARC-04's lifetime parameter is superseded by runtime accounting. At `Serialized`/`Multiple` a per-thread sharded in-flight counter in the FFI guard counts each call (SND-16). At every thread level an atomic count of extent tokens, never touched on the blocking-collective path, records live nonblocking scopes, active persistent requests and matched but unreceived messages. `Mpi::drop` first enters a `Finalizing` state that refuses new calls, extent opens and frees but admits open extents' completion and closing calls; it then waits only for the calls in flight and for lexically bound extents (nonblocking scopes) running on other threads. It skips `MPI_Finalize` with a warning, as D-8 does, when a live window exists, when the init thread holds a scope's token, or when a movable handle (a persistent request or a message) holds a token: such a token always selects the skip path, whatever thread holds the handle, because waiting on it could deadlock `Mpi::drop`. An open RMA epoch holds no token: it borrows its window, so the live window check already covers it. After a skip MPI stays initialized, so scope and epoch closes still run, while frees are leaked. | **accepted 2026-10-01** | 0.7.0 — ARC-04, SND-16 |
| D-35 | **0.7.0 deprecation and public-surface rules:** an old item is deprecated only when it forwards in one line with unchanged semantics, so the `numa` feature stays one release as `numa = ["slurm", "rma"]` and is removed in 0.8.0; every changed signature breaks without a deprecated twin; the eight `raw_handle` methods are deleted, their example uses moving into crate unit tests; `DatatypeTag` and `MpiDatatype::TAG` leave the public API (generic `StructField::new::<T>`, `CustomDatatype::contiguous::<T>`). | **accepted 2026-10-01** | 0.7.0 — ARC-03, ARC-14, ABI-04 |
| D-36 | **Planner deviations from register fix directions, approved at the design sign-off:** (a) no `NotRoot` error variant: ARC-08's list is narrowed, because the only root-only misuse goes with (b); (b) the in-place gathers (`gather_inplace`, `igather_inplace`, `gather_init_inplace`) accept non-root ranks, where the data is the rank's send block, resolving that ARC-15 asymmetry instead of naming it with an error; (c) send destinations take the same `Source` type as receive sources, with `Source::Any` rejected at run time as `InvalidArgument`, instead of a separate destination type. | **accepted 2026-10-01** | 0.7.0 — ARC-08, ARC-15, ARC-16 |

## Roadmap (accepted 2026-09-24)

Release numbers (D-14, D-24): milestone `0.5.x` ships as **0.6.0**; the milestones once labelled `0.6` and `0.7` are **0.7.0** and **0.8.0**, the labels open rows use.


- **0.6.0 (milestone `0.5.x`) — non-breaking fixes** (soundness fixes may tighten behaviour: new `Err`s where UB used to occur): error classes in C; null sentinel + generation counter for completed requests and write-back on error; all missing size/range validation; `MPI_ERR_COUNT` instead of truncation; `COMM_SELF` errhandler; finalized flag (re-init → `Err`, op-table sweep, window-accessor guard); zero window memory; thread-level runtime check; MSRV; build.rs rerun tracking, `-D` pass-through, precedence; runner skips fail; doctests in PR CI; release-notes `awk`; docs.rs all-features; stale-docs purge. Scope (D-6): every register row targeted `0.5.x`, `0.5.x†` or `0.6.0`.
- **0.7.0 (formerly `0.6`) — breaking API work:** D-1 buffer-safety model; `#[non_exhaustive]` + additive `rma`; deprecate `raw_handle`, retract discriminant contract; ops/datatypes as parameters; structured errors; `Status` from waits; typed `PROC_NULL`/`ANY`; `SharedWindow` → `Win`; D-4 `Info`; `numa` → `slurm`; remaining (breaking) bloat.
- **0.8.0 (formerly `0.7`) — internals + ABI:** D-2 by-value handles (removes tables, ABA, caps, per-request cost); build against the ABI `mpi.h` when detected.

## Finding register

Status values: `open` · `planned (<plan>)` · `fixed (<ref>)` · `wont-fix (<reason>)` · `superseded (<ID>)`.
Target: the release the fix belongs to. Open rows name the release (`0.6.0`, `0.7.0`, `0.8.0`; D-24); fixed rows keep the milestone they were fixed under (`0.5.x`, shipped as 0.6.0; `0.5.x†` = non-breaking bloat, in 0.5.x scope per D-6).

### Soundness — [01](findings/01-soundness.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| SND-01 | Nonblocking `Request` not tied to its buffer | critical | repro | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| SND-02 | `PersistentRequest` not tied to its buffer | critical | repro | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| SND-03 | RMA origin buffers not tied to the epoch; 4 rustdoc examples are UB | critical | reading | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| SND-04 | `PendingFetchResult` dropped before epoch close → write into freed heap | critical | repro | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| SND-05 | `mem::forget(Win::create)` leaves MPI aliasing a released buffer | critical | repro | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| SND-06 | gather/allgather/scatter never validate buffer sizes (9 methods) | critical | repro | 0.5.x | fixed (3637ae5) |
| SND-07 | V-collectives don't validate counts/displs vs size and buffer | critical | repro | 0.5.x | fixed (9c23425) |
| SND-08 | `*_custom` p2p: unbounded `T`, unchecked extent | critical | repro | 0.5.x (D-7) | fixed (ceada55) |
| SND-09 | RMA target range / origin count unvalidated (remote OOB write) | critical | repro | 0.5.x (D-11) | fixed (91bb508) |
| SND-10 | Window memory used after finalize | critical | repro | 0.5.x (D-8) | fixed (90d9544) |
| SND-11 | `Communicator` Send+Sync regardless of thread level | critical | repro | 0.5.x (D-3, D-9) | fixed (49cb30c) |
| SND-12 | Uninitialised window memory exposed as `&[T]` | major | repro | 0.5.x | fixed (4747cd1) |
| SND-13 | `SharedWindow` slices over concurrently-written memory (observed miscompile) | critical | repro | 0.7.0 (D-5); 0.5.x doc warning | fixed (31e6514): doc warning; API fix planned (ferrompi-0.7.0; design input: issue #28, per-region slice guards) |
| SND-14 | `UserOp` fat-pointer transmute relies on unspecified layout | minor | reading | 0.5.x† | fixed (2d53b18) |
| SND-15 | `fetch_and_op`/`compare_and_swap` result pointer derived from a shared borrow | minor | reading | 0.5.x | fixed (9777b12) |
| SND-16 | `Mpi` drop racing a concurrent guarded call at `Serialized`/`Multiple` reaches MPI after finalize | major | reading | 0.7.0 | planned (ferrompi-0.7.0) |
| SND-17 | Under fault-tolerant MPI, a receive failing with MPIX_ERR_PROC_FAILED_PENDING is treated as complete while MPI still owns its buffer | major | reading | 0.5.x | fixed (20db67d) |
| SND-18 | Under fault-tolerant MPI, a wildcard receive started while the request table is full returns `ResourceExhausted` after its internal wait fails with MPI_ERR_PROC_FAILED_PENDING, while MPI still owns its buffer | major | repro | 0.5.x | fixed (105eb34) |
| SND-19 | `allreduce_with_op` above `INT_MAX` elements on MPI 4.0 calls `MPI_Allreduce_c` with a classic user function; MPICH narrows the count to `int` without splitting (its assertion compiles out under NDEBUG), so the Rust callback can receive a negative or wrapped length and build slices past the buffers | major | reading | 0.5.x | fixed (b96d7eb) |
| SND-20 | The processor-name (256 B), library-version (8192 B) and error-string (512 B) Rust buffers are not checked against `MPI_MAX_PROCESSOR_NAME`, `MPI_MAX_LIBRARY_VERSION_STRING` and `MPI_MAX_ERROR_STRING`; a library with a larger constant would write past them | minor | reading | 0.5.x | fixed (631fa89) |
| SND-21 | `Request::wait_any`/`test_any` mark `done[idx]` and report the index `MPI_Waitany`/`MPI_Testany` returned without checking `0 <= idx < count`, and `wait_some`/`test_some` index `done` and `indices` by the returned `outcount` and indices the same way; the C shim relies on the library writing valid values (or `MPI_UNDEFINED`) even when the call fails, so a library that writes an out-of-range value makes it write past the Rust-owned buffers | minor | reading | 0.6.0 | fixed (58c20be) |
| SND-22 | `Win::local_slice(&self) -> &[T]` and `local_slice_mut` hand out plain slices over window memory that a peer's passive-target `put`/`accumulate`, or a fence that lands remote writes, can change while the borrow lives (`Win::fence` also takes `&self`); the rustdoc asks only for proper MPI epoch synchronization. This is SND-13's aliasing class on every window, not only `SharedWindow` | critical | reading | 0.7.0 | planned (ferrompi-0.7.0) |

### Correctness — [02](findings/02-correctness.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| COR-01 | Error classes decoded with Open MPI numbering (wrong on MPICH) | major | repro | 0.5.x | fixed (ea3cff2) |
| COR-02 | Stale request handles act on unrelated requests (ABA) | major | repro | 0.5.x | fixed (a33d65b) |
| COR-03 | No request write-back on error in wait/test-many | major | repro | 0.5.x | fixed (f7859d4) |
| COR-04 | `MPI_STATUSES_IGNORE` loses per-request error | minor | repro | 0.5.x | fixed (35422e7) |
| COR-05 | Counts > `INT_MAX` silently truncated | critical | repro | 0.5.x | fixed (0554ed9, 0427a97, 36c7116) |
| COR-06 | `MPI_ERRORS_RETURN` not on `MPI_COMM_SELF` | major | reading | 0.5.x | fixed (19daebb) |
| COR-07 | Finalize/re-init lifecycle aborts; `UserOp` drop after finalize | major | repro | 0.5.x | fixed (e1c123c) |
| COR-08 | Finalize sweep frees active requests / collectively frees windows | minor | reading | 0.5.x | fixed (8c8bf37, 459d6ea) |
| COR-09 | `-1` = PROC_NULL (MPICH) vs ANY_SOURCE (Open MPI) | minor | repro | 0.7.0 | planned (ferrompi-0.7.0) |
| COR-10 | `cancel()` allowed on collective/RMA requests | minor | repro | 0.5.x | fixed (4947bc4) |
| COR-11 | `LongDoubleInt`/`LongInt` layout wrong on macOS arm64 / Windows | minor | reading | 0.5.x | fixed (3a01966): target gating; per-target layouts wont-fix (D-32) |
| COR-12 | Pre-MPI-4 stubs never yield `NotSupported`; 3 contradicting docs | minor | reading | 0.5.x | fixed (58f8fdc, 134bfed, 98236f8): stub mapping, ADR-0004, docs/mpi-compatibility.md |
| COR-13 | `Win::sync` rustdoc wrong; example fails on MPICH | minor | repro | 0.5.x docs / 0.7.0 API | fixed (3f50248, 98236f8): rustdoc, docs/mpi-compatibility.md; sync on lock guards planned (ferrompi-0.7.0) |
| COR-14 | `MPI_UNDEFINED` from `MPI_Get_count` leaks; rc ignored | nit | reading | 0.5.x | fixed (e918274) |
| COR-15 | `op_set_closure` no bounds check; `op_create_user` no `op_used` check | nit | reading | 0.5.x | fixed (2d53b18) |
| COR-16 | `type_create_struct` maybe-uninitialised arrays at count 0 | nit | compiler | 0.5.x | fixed (34a748c) |
| COR-17 | Open MPI frees a persistent request that errors in `MPI_Wait`, `MPI_Test` or `MPI_Waitall`; `PersistentRequest` then keeps `active` set after a single-request wait/test (further `wait` fails, `start` reports already-active), and `MPI_Waitall` over already-finished requests can return success and lose the truncation error | minor | repro | 0.5.x | fixed (50c4791, c733945, 8ff3ce1) |
| COR-18 | `SharedWindow::allocate` with a zero count returns `Err(Internal)` and leaks the registered window, which `Mpi::drop` does not count | minor | repro | 0.5.x | fixed (4747cd1) |
| COR-19 | `Win::allocate`/`SharedWindow::allocate` leave the window live but uncounted in `LIVE_WINDOWS` when zeroing fails or MPI returns a null base for a non-zero count, so `Mpi::drop` can call `MPI_Finalize` with it alive | minor | reading | 0.5.x | fixed (4263c80) |
| COR-20 | `Mpi::init_thread` resets to uninitialized when installing `MPI_ERRORS_RETURN` on `MPI_COMM_WORLD`/`MPI_COMM_SELF` fails after `MPI_Init_thread` succeeded, so a retry calls `MPI_Init_thread` twice | minor | reading | 0.5.x | fixed (762e841) |
| COR-21 | `PersistentRequest::start_all` marks no request active when `MPI_Startall` fails, although MPI may have started some; `Drop` then frees a started request without waiting | minor | reading | 0.5.x | fixed (388dc16, f42fe0a, fc7bc7f) |
| COR-22 | When a rank's communicator or window table is full, or installing `MPI_ERRORS_RETURN` on a new communicator fails, after MPI created the object on every rank, that rank frees it alone with the collective `MPI_Comm_free`/`MPI_Win_free` | minor | reading | 0.5.x | fixed (3247cea, 3527040, 6344c40) |
| COR-23 | In debug builds at `ThreadLevel::Serialized`, `Request::wait` and `PersistentRequest::wait` rejected by the overlap check mark the request completed or inactive although MPI never saw the call | minor | repro | 0.5.x | fixed (63c6629, 6305cf2) |
| COR-24 | `Communicator::topology` returns early on a rank whose processor-name query fails, before the hostname `MPI_Allgather`, leaving the other ranks blocked in it | minor | reading | 0.5.x | fixed (38082f6) |
| COR-25 | `Mpi::wtime` calls `MPI_Wtime` before `Mpi::init` and after the `Mpi` handle is dropped; MPICH aborts the process ("Attempting to use an MPI routine (internal_Wtime) before initializing or after finalizing MPICH") | minor | repro | 0.5.x | fixed (2dd6a0b) |
| COR-26 | `Mpi::library_version` ends in a NUL character on Open MPI 4.1.6 and 5.0.7, whose `MPI_Get_library_version` reports a length that counts the terminator (MPI-4.1 §9.1.1 stores it at `version[resultlen]`, outside the length); the processor-name and error-string readers also keep every byte up to the reported length, and the error-string reader does not clamp that length to its 512-byte buffer | minor | repro | 0.5.x | fixed (cc2ec21) |
| COR-27 | `Request::wait_some` returns `Ok(vec![])`, which its rustdoc gives as "no requests were active", when `MPI_Waitsome` reports an `outcount` of 0, which MPI-4.1 §3.7.5 does not allow (Waitsome waits until at least one operation completes); a non-conforming library can silently end a caller's completion loop | minor | reading | 0.6.0 | fixed (40cdcb5) |

### Architecture / API — [03](findings/03-architecture-api.md)

| ID | Title | Sev | Reversibility | Target | Status |
|---|---|---|---|---|---|
| ARC-01 | C handle tables are the root cause of ABA, per-request cost, caps, sweep | major | two-way (large) | 0.8.0 (D-2) | open |
| ARC-02 | Non-additive `rma` feature; no `#[non_exhaustive]` anywhere | major | one-way | 0.7.0 | fixed (1bbac44): #[non_exhaustive] on Error; rest planned (ferrompi-0.7.0) |
| ARC-03 | Shim representation in public API (`raw_handle`, discriminant contract, pub `from_code`) | major | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-04 | Handles not tied to `Mpi` lifetime | major | mixed | 0.5.x guards / 0.7.0 | fixed (90d9544): runtime guards; lifetime parameter superseded (D-34) |
| ARC-05 | Thread-safety model inconsistent | major | mixed | 0.5.x (D-3) | fixed (49cb30c) |
| ARC-06 | Buffer-safety model (umbrella SND-01…05) | critical | one-way | 0.7.0 (D-1) | planned (ferrompi-0.7.0) |
| ARC-07 | Ops and datatypes not parameters | major | one-way | 0.7.0 | planned (ferrompi-0.7.0): op parameter; datatype parameter deferred (D-33) |
| ARC-08 | Error model misleads and loses context | major | one-way | 0.7.0 | fixed (f371c7a, b36ca61, 5710b2b, 9c534c8, 529957e, 3b4e20d, 87416c3, db2a5d4) |
| ARC-09 | `Status` always discarded | major | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-10 | Copy-pasted families drifted (root of SND-06) | major | two-way | 0.5.x (validators) | fixed (c402d59) |
| ARC-11 | Capability gaps (in-place nonblocking, mprobe, HW_GUIDED…) | minor | additive | 0.7.0 | planned (ferrompi-0.7.0); HW_GUIDED wont-fix (D-31) |
| ARC-12 | `Info` public but unused | minor | one-way | 0.7.0 (D-4) | planned (ferrompi-0.7.0) |
| ARC-13 | `SharedWindow` duplicates `Win`; `WinKind` dead | minor | one-way | 0.5.x WinKind / 0.7.0 (D-5) | fixed (b39adfb): WinKind removed; SharedWindow merge planned (ferrompi-0.7.0; design input: issue #28) |
| ARC-14 | `numa` feature has no NUMA code, implies `rma` | minor | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-15 | Naming/coverage asymmetries | minor | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-16 | Magic `-1` sentinels; no typed source/tag | minor | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-17 | RMA redundant `target_count`, duplicate tags | minor | one-way | 0.7.0 | planned (ferrompi-0.7.0) |
| ARC-18 | Open MPI 5 capabilities unused (persistent collectives, `create_from_group` gated on `MPI_VERSION >= 4`; OMPI 5 reports 3.1) | minor | two-way | 0.5.x | fixed (213cfd0, d555791, c58cd05) |

### MPI-5 ABI — [04](findings/04-mpi5-abi.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| ABI-01 | build.rs drops `-D` flags (`MPI_ABI` lost → silent ABI mismatch) | critical (latent) | reading | 0.5.x | fixed (8e5e108) |
| ABI-02 | No ABI detection probe; MPICH 4.3 draft ABI accepted | major | reading | 0.5.x reject / 0.8.0 build | fixed (67fbcd4): draft ABI rejected; ABI build open (0.8.0) |
| ABI-03 | Build-selection env vars not tracked (= INF-03) | major | repro | 0.5.x | fixed (8f9fbfd) |
| ABI-04 | Public-API one-way doors (= ARC-02/03) | major | reading | 0.7.0 | planned (ferrompi-0.7.0) |
| ABI-05 | ADR-0001 misstates what the ABI adds; ADR-0006 needed | minor | reading | 0.5.x | fixed (03f01e8, fb99656) |
| ABI-06 | No interop API (deliberately deferred) | info | — | after ABI backend | open |
| ABI-07 | Optional CI compile+link job against Forum ABI stubs | info | built locally | 0.5.x or 0.7 | fixed (5d62df7) |

### Performance — [05](findings/05-performance.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| PRF-01 | Request table: ~14 ns / 2 locked RMWs per request (+58%/+33% small nonblocking p2p) | major | measured | 0.8.0 (D-2); no 0.6.0 stop-gap | open; 0.5.x hardening raised it to ~23 ns per request (MPICH bisect: ~3.6 ns of it from the `RequestKind` field, 4947bc4) |
| PRF-02 | Bitmap concentrates contention; ADR/comment claim the opposite | minor | measured | 0.5.x docs / 0.8.0 | fixed (01866f1, 605fd2d): shim comment, ADR-0002; table change open (0.8.0) |
| PRF-03 | `ffi_overhead` bench cannot measure FFI overhead | minor | measured | 0.5.x | fixed (f74665a, 15d570c) |
| PRF-04 | "Persistent 10–30% faster" refuted as stated; bench at 1 MiB only | minor | measured | 0.5.x | fixed (e73c3d0, 134bfed, eb63b5d, 76eded0, 15d570c) |
| PRF-05 | `start_all`/`wait_all` zero 512 B scratch per call | nit | measured | — | wont-fix (adds `unsafe` for ~8 ns; rejected in 0.5.x planning) |
| PRF-06 | Dead per-callback tag lookup; `Vec` per `wait_some`; topology 256·P | nit | reading | 0.5.x | fixed (2d53b18) for the callback tag lookup; Vec return and topology 256·P deferred |
| PRF-07 | `[profile.release]` doesn't reach downstream; comment says it does | minor | Cargo semantics | 0.5.x | fixed (affb75c, 0c7b942) |

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
| BLT-10 | 42 redundant `[[example]]` entries | 185 | 0.5.x† | fixed (07c5241) |
| BLT-11 | Tables/indexes repeated across 3–5 docs; marketing tone | 250 | 0.5.x† | fixed (713cb4f, 76eded0, a6262a9) |
| BLT-14 | 15 in-place C shims differ only by `MPI_IN_PLACE`; dead `is_root` | 300 | 0.5.x† | fixed (38c789e) |
| BLT-15 | `UserOp` double registry + dead per-callback lookup | 120 | 0.5.x† | fixed (2d53b18) |
| BLT-16 | Benches measuring nothing; duplicated bench protocol | 210 | 0.5.x† | fixed (aa460f1) |
| BLT-20 | Six copies of the slot-claim loop | 70 | superseded by ARC-01 | open; also covers the ~38 request-registration tails (`FERROMPI_ERR_REQUESTS_FULL`) in `csrc/ferrompi.c`, which D-2 removes with the tables |
| BLT-21 | Process artifacts, stale line refs, expired promises in comments | 45 | 0.5.x† | fixed (3d206a8) |
| BLT-23 | `Group::undefined()` FFI call returning literal −1 | 35 | 0.5.x† | fixed (e874118) |
| BLT-25 | Dead C branches (errhandler install) | 30 | 0.5.x† | fixed (7389231) |
| BLT-26 | `const _` Send/Sync asserts beside `unsafe impl` | 27 | 0.5.x† | fixed (c074ed1) |
| BLT-27 | Dead `ferrompi_init`; module-wide `allow(dead_code)` | 25 | 0.5.x† | fixed (3cde221) |
| BLT-29 | `with_handles` implemented twice | 20 | 0.5.x† | fixed (2b1237e) |
| BLT-32 | `ReduceOp` compile_fail doctest via 14 `cfg_attr` | 16 | 0.7.0 (with ARC-02) | planned (ferrompi-0.7.0) |
| BLT-34 | `use super::*` in 11 test modules | — | 0.5.x† | fixed (99f3f9d) |
| BLT-35 | Five tidy-ups left after 0.6.0: the always-taken first-error guard in `zero_own_segment`; `docs/architecture.md` and `docs/mpi-compatibility.md` each restate a paragraph given earlier in the file; single-use locals in `examples/pi_monte_carlo.rs`; a `src/lib.rs` test comment restating `stub_mpi()` | ~20 | 0.6.0 | fixed (96bef9b) |

### Documentation — [07](findings/07-docs.md)

| ID | Title | Sev | Target | Status |
|---|---|---|---|---|
| DOC-01 | `docs/architecture.md` false statements | major | 0.5.x | fixed (713cb4f) |
| DOC-02 | ADR-0001 driver 1 false (= ABI-05) | minor | 0.5.x | fixed (03f01e8) |
| DOC-03 | ADR-0002 claims that do not hold | minor | 0.5.x | fixed (605fd2d) |
| DOC-04 | ADR-0004 lifetime rejection mis-argued; nonexistent variant | minor | 0.5.x note / 0.7.0 new ADR | fixed (134bfed): note; new ADR planned (ferrompi-0.7.0) |
| DOC-05 | ADR-0005 Decision 7 describes the rejected design; plan sections | major | 0.5.x | fixed (134bfed) |
| DOC-06 | Migration guide: nonexistent APIs, false safety claim | major | 0.5.x | fixed (eb63b5d) |
| DOC-07 | `docs/mpi-compatibility.md` inaccuracies | minor | 0.5.x | fixed (98236f8) |
| DOC-08 | `README.md` inaccuracies (badge, version, reqs, `LD_LIBRARY_PATH`) | minor | 0.5.x | fixed (76eded0) |
| DOC-09 | Crate rustdoc (`src/lib.rs`) inaccuracies | minor | 0.5.x | fixed (a6262a9) |
| DOC-10 | Stale/wrong item-level rustdoc | minor | 0.5.x | fixed (e5034a8) |
| DOC-11 | CONTRIBUTING template filler + false policies | minor | 0.5.x | fixed (067749d) |
| DOC-12 | CHANGELOG internal IDs / restated rustdoc | minor | 0.5.x | fixed (0c7b942) |
| DOC-13 | `benches/README.md` claims | minor | 0.5.x | fixed (15d570c) |
| DOC-14 | Missing docs: lifecycle, error reporting, runtime lib path, Open MPI build | minor | 0.5.x | fixed (98236f8, 90c95c6) |
| DOC-15 | Wrong C comments | nit | 0.5.x | fixed (01866f1) |
| DOC-16 | No doc says an error from a collective can reach only some ranks; the private `exchange_window_words` comment claims every rank sees the same error, so after a rank-local `MPI_Allgather` failure `Win::create`/`Win::allocate` can leave the other ranks blocked in `MPI_Win_create`/`MPI_Win_allocate` (`Communicator` docs cover local validation failures only) | minor | 0.6.0 | fixed (fa071ed) |

### Build / CI / tests — [08](findings/08-build-ci-tests.md)

| ID | Title | Sev | Verified | Target | Status |
|---|---|---|---|---|---|
| INF-01 | MPI test runner passes when it ran nothing | major | repro | 0.5.x | fixed (87ac8d9) |
| INF-02 | Declared MSRV false; no MSRV job | major | repro | 0.5.x (D-12: 1.85) | fixed (39b9fbb, bd3b98c, 76eded0, 067749d): rust-version 1.85, MSRV CI job, README and CONTRIBUTING docs |
| INF-03 | build.rs never re-runs on MPI selection change | major | repro | 0.5.x | fixed (8f9fbfd) |
| INF-04 | build.rs precedence contradicts docs; overrides fail silently | major | repro | 0.5.x | fixed (0c8e405, 8e5e108) |
| INF-05 | build.rs dead/misleading output | minor | reading | 0.5.x | fixed (0c8e405, 8e5e108, 50e7d63) |
| INF-06 | No `links` manifest key | minor | reasoning | 0.5.x | fixed (39b9fbb) |
| INF-07 | docs.rs builds default features only (no RMA docs) | major | live check | 0.5.x | fixed (844ec07, b0a1a41) |
| INF-08 | Doctests never run on PRs | major | reading | 0.5.x | fixed (acd809f) |
| INF-09 | MPI-4 probes turn regressions into silent passes | major | reading | 0.5.x | fixed (87ac8d9) |
| INF-10 | Odd-np deadlocks; CI np=4 only | minor | repro | 0.5.x | fixed (6ee2c4e) |
| INF-11 | No large-count integration test | major | reading | 0.5.x | fixed (033e16a, db52738) |
| INF-12 | Public APIs untested; examples never run | minor | reading | 0.5.x | fixed (da87ce5) |
| INF-13 | Error-class assertions accept any class | major | reading | 0.5.x | fixed (ea3cff2) |
| INF-14 | Release notes extraction yields empty bodies | minor | repro | 0.5.x | fixed (700cd57, 0c7b942) |
| INF-15 | CI coverage gaps (numa clippy, sanitizers, coverage 11/73) | minor | reading | 0.5.x | fixed (a0501aa, 1569e4c, bd3b98c, b0a1a41, 2e9dd20) |
| INF-16 | MPICH hotfix: unchecked downloads, copy-pasted ×4 | minor | reading | 0.5.x | fixed (cd55e0b, bbd98f6) |
| INF-17 | `security.yml` hygiene | nit | reading | 0.5.x | fixed (d42b069) |
| INF-18 | Package ships repo internals (incl. `audits/`) | minor | `cargo package` | 0.5.x | fixed (07c5241, bbd98f6) |
| INF-19 | ADR-0002 mandated TSan step missing | minor | reading | 0.5.x / moot after ARC-01 | fixed (605fd2d) |
| INF-20 | CI covers MPICH 4.2 + Open MPI 4.x only (no Open MPI 5) | minor | reading | 0.5.x | fixed (3f8ec50) |
| INF-21 | Unit tests mutate a global static (latent) | nit | stress test | 0.7.0 | planned (ferrompi-0.7.0) |
| INF-22 | Third-party GitHub Actions pinned by tag, not commit SHA | minor | reading | 0.5.x | fixed (1dfc61f) |
| INF-23 | Publishing uses a long-lived crates.io token (no Trusted Publishing) | minor | reading | 0.7.0 | planned (ferrompi-0.7.0) |
| INF-24 | `examples/test_rma_rget.rs`, `test_rma_raccumulate.rs`, `test_rma_win_lock.rs` and `test_rma_win_flush_sync.rs` open a passive-target lock right after a `Win::fence` with no assert (no `MPI_MODE_NOSUCCEED`, no barrier), unlike the `Win::raccumulate` rustdoc pattern; works on MPICH and Open MPI | nit | reading | 0.6.0 | fixed (cfb67b4) |
| INF-25 | The cargo-registry cache step is repeated in 8 `test.yml` jobs (per-job keys, so a shared composite action would need an input) | nit | reading | 0.7.0 | planned (ferrompi-0.7.0) |
| INF-26 | `examples/test_rma_window.rs` and `examples/shared_memory.rs` take a passive-target lock after a `SharedWindow::fence()` and before the next one, with no `MPI_MODE_NOSUCCEED` and no barrier between the fence and the lock (in `shared_memory.rs` a read of a peer's segment comes between them), unlike the `Win::raccumulate` rustdoc pattern; `SharedWindow::fence` takes no assert argument, so the pattern cannot be expressed until `SharedWindow` merges into `Win`; works on MPICH and Open MPI | nit | reading | 0.7.0 | planned (ferrompi-0.7.0) |
| INF-27 | `examples/pi_monte_carlo.rs` reduces the maximum elapsed time into a temporary, then reads `recv[0]`, which still holds the global inside count, so "Max time" prints a sample count (e.g. `Max time: 78541885.0000s`) | nit | repro | 0.6.0 | fixed (2ee62a7) |
