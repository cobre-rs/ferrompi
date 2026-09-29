# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-18

This release resolves all 11 findings from a 2026 architecture & performance
assessment (soundness, FFI overhead, lock-free allocator, typed errors), plus
dependency refreshes and documentation work.

### Breaking Changes

- **`Request::wait_all` now takes `&mut [Request]`** instead of consuming a
  `Vec<Request>`, so one backing buffer can be reused across a drain loop
  (mirroring `PersistentRequest::wait_all`). Migration: pass `&mut [r1, r2, r3]`
  or `&mut my_vec` instead of `vec![r1, r2, r3]`. On success each request is
  marked completed in place; on error they are left active so `Drop` re-waits.
- **New `Error::ResourceExhausted { resource: ResourceKind }` variant.** The
  `Error` enum is not `#[non_exhaustive]`, so consumers matching it without a
  trailing `_` arm will fail to compile. Adds the public `ResourceKind` enum
  (`Request`, `Communicator`, `Datatype`, `Operation`, `Window`, `Group`,
  `Info`).

### Added

- **Typed handle-table-exhaustion errors.** Each internal handle table
  now returns a distinct sentinel on overflow, surfaced as
  `Error::ResourceExhausted { resource }` instead of an opaque
  `Error::Mpi { class: Other, .. }`, so a long-running job can distinguish an
  internal cap from a genuine MPI fault and back off.
- **`[profile.release]`** with `lto = "thin"` and `codegen-units = 1` for
  this repository's own release builds and benches. Cargo ignores a
  dependency's profiles, so crates depending on ferrompi were unaffected.

### Changed

- **Request handle table is now a lock-free 64-bit occupancy bitmap**,
  replacing the dense `atomic_int` slot array. Removes the cache-line false
  sharing and O(N) scan that affected concurrent posting under
  `MPI_THREAD_MULTIPLE`; allocation uses an `acq_rel` `fetch_or` claim with a
  hardware find-first-zero. The six small fixed tables keep the simpler
  CAS-scan design.
- **Hot paths are now inlinable:** `#[inline]` on `Error::check`/
  `check_with_op` and the non-generic `Request` / `PersistentRequest`
  completion methods; `#[cold]` + `#[inline(never)]` on the cold
  error-construction helpers.
- **Per-call allocations removed from completion paths:** `start_all` /
  `wait_all` / `wait_any` / `wait_some` / `test_any` / `test_some` use stack
  scratch buffers (both the Rust and C sides) instead of allocating on every
  call.
- **`gather_topology` host de-duplication is now O(size)**, down from
  O(size × distinct_hosts), allocating one `String` per distinct host.
- **Dependencies refreshed:** `rand` dev-dependency 0.9 → 0.10 (uses the new
  `RngExt` trait), `cargo update` to latest compatible (cc 1.2.64,
  pkg-config 0.3.33, …), and CI actions bumped (codecov-action v6,
  action-gh-release v3, size-label-action v0.5.7).
- **README repositioned** around FerroMPI's own capabilities rather than a
  scorecard against other crates; ADR-0002 amended for the bitmap design.

### Fixed

- **Use-after-free window on request-table exhaustion closed.** When a
  nonblocking/RMA initiator had already started a transfer but the request
  table was full, the C shim called `MPI_Request_free` — which does not
  cancel an active operation — and returned an error, letting the caller
  drop a buffer MPI was still using. It now drives the orphaned request to
  completion with `MPI_Wait` before returning. Persistent `*_init` shims,
  whose requests are inactive, still use `MPI_Request_free`.
- **ADR-0002 memory-ordering claim corrected:** the alloc-path rationale
  wrongly stated the `acq_rel` claim publishes the subsequent table write;
  the real (external-happens-before) contract is now documented.
- **Broken intra-doc link** in `Error::from_code` that failed the `-D warnings`
  documentation build.
- **CHANGELOG comparison links** repaired: the `[0.4.1]` link was missing and
  `[Unreleased]` still compared from `v0.4.0`.

## [0.4.1] - 2026-05-18

### Added

- **Groups.** `Group` (from `Communicator::group()`, freed on drop) with
  `size`, `rank`, the set operations (`include`, `exclude`, `union`,
  `intersection`, `difference`, `range_include`, `range_exclude` over
  `RankRange` progressions), `compare` (returning `GroupComparison`) and
  `translate_ranks`. `Mpi::create_from_group` builds a communicator from a
  group on MPI 4.0 and later.
- **Custom datatypes.** `CustomDatatype`, committed on construction and
  freed on drop, built with `contiguous`, `vector`, `create_struct` (from
  `StructField`s) and `resized`. `send_custom`, `recv_custom`,
  `isend_custom` and `irecv_custom` send and receive with it; the sealed
  `BytePermutable` trait marks the types they accept.
- **User-defined reductions.** `UserOp<T>` registers a
  `Fn(&[T], &mut [T]) + Send + Sync + 'static` closure with `MPI_Op_create`
  (commutative or not, at most 16 live per process), used by
  `Communicator::allreduce_with_op`.
- **RMA windows (feature `rma`).** `Win<T>` over caller-owned or
  MPI-allocated memory: fence, post/start/complete/wait and passive-target
  lock/lock-all epochs (`WinFenceAssert`, `WinPscwAssert`, `LockType`,
  `WinLockGuard`, `WinLockAllGuard`), flush and sync, blocking and
  request-based put, get and accumulate, and the atomic `get_accumulate`,
  `fetch_and_op` and `compare_and_swap` (`PendingFetchResult<T>`; integer
  types only, via `AtomicMpiDatatype`).
- **Buffered and persistent point-to-point.** `Mpi::buffer_attach` (buffers
  above `i32::MAX` bytes return `Err(InvalidBuffer)`) and `buffer_detach`,
  and the persistent `send_init`, `bsend_init`, `rsend_init`, `ssend_init`
  and `recv_init`.
- **Documentation.** `docs/architecture.md`, `docs/migrating-from-rsmpi.md`,
  `docs/mpi-compatibility.md`, ADR-0001, ADR-0003, ADR-0004 and ADR-0005,
  published in rustdoc under `ferrompi::doc`.
- **V-collective length validation.** `gatherv`, `scatterv`,
  `allgatherv`, `alltoallv`, and their nonblocking variants now return
  `Err(Error::InvalidBuffer)` when `counts.len() != displs.len()`. The
  persistent `*_init` variants already had this guard; the new guards
  bring blocking and nonblocking into parity.
- **Persistent collectives reject oversized counts.** All persistent
  `*_init` shims now return `MPI_ERR_COUNT` when `count > INT_MAX`
  instead of silently truncating the cast to `int`. Full `_c` dispatch
  for persistent operations is deferred.
- **5 new direct-FFI integration tests** for boundary conditions:
  `test_comm_table_concurrency` (4-thread duplicate stress under
  `MPI_THREAD_MULTIPLE`), `test_persistent_count_overflow`,
  `test_waitall_count_overflow`, `test_get_group_invalid_handle`,
  `test_create_from_group_null_handle`. Also new tests in
  `test_user_op.rs` (non-commutative reduction) and `test_waitany.rs`
  (`test_any` / `test_some` polling loops).
- **Send/Sync status table** in the crate-level rustdoc enumerating
  every public type's auto-trait status and rationale (see lib.rs
  "Send/Sync Status of Public Types").
- **37 new SAFETY comments** across `src/comm/{blocking,persistent,
  v_collective}.rs` documenting pointer validity, type-tag mapping,
  and handle ownership at each `unsafe` FFI block.

### Changed

- **`Error::from_code(0)` no longer panics.** Previously
  `assert!(code != 0, ...)`; now returns
  `Error::Internal("from_code called with success code 0")`. Library
  code must not panic — `check_with_op` remains the canonical
  success-vs-error idiom.
- **`Communicator::allreduce_indexed` error tag corrected.** Errors
  now carry `operation: Some("allreduce_indexed")` instead of the
  generic `"allreduce"`. Defensive audit also corrected
  `allreduce_bytes` (was `"allreduce"`, now `"allreduce_bytes"`).
- **`supports_create_from_group` no longer caches transient
  failures.** If `Mpi::version()` fails (e.g., called pre-init), the
  result is not cached; a subsequent call can re-probe.
- **`src/error.rs` migrated to `thiserror v2`.** Both `MpiErrorClass`
  and `Error` now derive `thiserror::Error`. All existing `Display`
  strings preserved byte-for-byte (cobre parsers are safe).
- **Request::Drop documented as blocking.** Added loud `# Drop
  Behavior` rustdoc sections to `Request` and `PersistentRequest`
  explaining the `MPI_Wait`-in-Drop semantics and deadlock risk.
  ADR-0004 gained a Drop-behavior subsection. Cancel-then-wait is
  deferred to v0.5.

### Fixed

- **`Win::fetch_and_op` and `Win::compare_and_swap` are now sound under
  non-blocking RMA semantics.** Previously these methods passed
  stack-local pointers (`addr_of!(origin)`, `result.as_mut_ptr()`) to
  `MPI_Fetch_and_op` / `MPI_Compare_and_swap`, but MPI may not actually
  read/write those buffers until the closing fence/complete/unlock —
  by which time the stack frames have been reused. OpenMPI tolerated
  it; MPICH 4.2.x exhibited the UB as zeroed returns. The fix boxes
  origin (and `compare`, for CAS) and result on the heap; the returned
  `PendingFetchResult<T>` owns the `Box`es so MPI's pointers remain
  valid until `resolve()` is called.
- **`install_errors_return_win` is now best-effort.** Per MPI-3 §9.4.4,
  newly-created windows default to `MPI_ERRORS_ARE_FATAL`. The C shim
  attempts to upgrade to `MPI_ERRORS_RETURN`, but OpenMPI 4.x has been
  observed to reject `MPI_Win_set_errhandler` with `MPI_ERR_WIN` on
  freshly-created windows in some configurations (e.g., `--btl=self,tcp`
  in CI). The shim now logs the failure to `stderr` and keeps the
  window with its MPI default error handler, instead of failing
  `Win::create`/`Win::allocate` outright.
- **Persistent P2P integration tests now add a post-init barrier**
  before issuing `start()`. Without this barrier, OpenMPI 4.x can
  deadlock under `--btl=self,tcp` when rank 0's `start()` issues before
  rank 1's `recv_init` has completed. `test_persistent_p2p`,
  `test_persistent_ssend`, and `test_persistent_bsend` were updated;
  the existing `test_persistent_rsend` already had this barrier.
- **`test_persistent_count_overflow` now skips on MPI < 4.0.**
  Persistent collectives are stubbed (return `MPI_ERR_OTHER`) on
  pre-MPI-4 runtimes like OpenMPI 4.1.x (which reports
  `MPI_VERSION = 3`), so the count guard is never reached.
- **`test_rma_win_create` now gracefully skips when `MPI_Win_create`
  is unsupported.** OpenMPI 4.x with `--btl=self,tcp` (the CI fallback
  when UCX/rdma is unavailable) returns `MPI_ERR_WIN` from
  `MPI_Win_create` over caller-owned memory. `Win::allocate` (which
  uses MPI-managed memory) still tests cleanly.
- **`comm_table` is now thread-safe under `MPI_THREAD_MULTIPLE`.** The
  communicator handle table joined the other 6 handle tables in
  using C11 atomic-CAS slot allocation. Two threads calling
  `world.duplicate()` or `world.split()` concurrently no longer race
  on slot indices.
- **`op_table` reads are now atomic.** The user-defined op table
  joined the others; `op_table[handle]` is `_Atomic(MPI_Op)` and
  accessed via `atomic_store/load_explicit` everywhere.
- **`tables_initialized` is now atomic.** The init guard uses a
  CAS so concurrent first-time initialization is safe (the previous
  plain-int flag was UB-prone under `MPI_THREAD_MULTIPLE`).
- **3 non-atomic acquire-loads in `info_free`/`group_free`/
  `type_free` upgraded to `memory_order_acquire`** for parity with
  the get-side acquire loads.
- **`ferrompi_op_free` preserves the live op on `MPI_Op_free` error**
  instead of unconditionally dropping the Rust closure. If MPI
  rejects the free, the slot remains valid so the trampoline cannot
  invoke a null closure pointer.
- **Window error handlers' return code is now propagated.** All 3
  Win creators (`win_allocate_shared`, `win_create`, `win_allocate`)
  now check `MPI_Win_set_errhandler`'s return code; failures free
  the partial window and surface as `Err`.
- **Overflow guards moved above `malloc` in `waitall`/`startall`.**
  The `count > INT_MAX` check now runs before the allocation
  attempt, preventing wasted allocations on overflow paths.
- **`get_group` sentinel mismatch fixed.** Invalid group handles
  now return `MPI_GROUP_NULL` (was incorrectly returning
  `MPI_GROUP_EMPTY`); 7+ callers' `MPI_GROUP_NULL` guards now fire
  correctly. Slot 0 still returns `MPI_GROUP_EMPTY` as intended.
- **MPI-4 `comm_create_from_group` NULL guard added.** Returns
  `MPI_ERR_OTHER` if the call yields `MPI_COMM_NULL` instead of
  allowing a NULL communicator into the handle table.
- **`reduce_inplace` non-root passes `buf` as `recvbuf`** instead
  of `NULL`, satisfying strict MPI implementations that reject NULL
  for the not-significant-but-must-be-valid `recvbuf` argument.
- **`lib.rs` Capabilities list and Send/Sync table** brought into
  alignment with the actual public API (Groups, CustomDatatype,
  UserOp, distributed Win RMA, Info, persistent P2P).
- **6 doc-precision fixes**: ADR-0005 function rename
  (`ferrompi_call_rust_closure` → `rust_user_op_invoke`), 6
  malformed rustdoc link suffixes in `src/datatype.rs`, 3 stale
  "4561 LOC" citations updated to 4629, migration guide
  `Request::Drop` paragraph corrected, ADR-0004 Drop-behavior
  subsection added.

## [0.4.0] - 2026-04-24

### Breaking Changes

- **`Error::Mpi` gains an `operation` field.** A new field
  `operation: Option<&'static str>` is appended to the `Error::Mpi`
  variant. Consumers pattern-matching `Error::Mpi { class, code, message }`
  without a trailing `..` will fail to compile. Recommended migrations:
  `Error::Mpi { class, code, message, .. }` if the operation tag is
  not needed, or `Error::Mpi { class, code, message, operation }` to
  read it. This bundles the breaking-change boundary that motivates
  the v0.4.0 minor bump.
- **`Display` format change.** When `operation.is_some()`, the error
  message now reads `"MPI error in {op}: ..."` (e.g.
  `"MPI error in bcast: invalid rank (class=ERR_RANK, code=6)"`)
  instead of `"MPI error: ..."`. Consumers parsing the human-readable
  `Display` output (not recommended, but extant) must account for the
  new prefix.

### Added

- **`Error::from_code_with_op(code, op)`** -- Constructs `Error::Mpi`
  with the operation tag pre-populated. Replaces the pattern of
  constructing `Error::from_code` and patching `operation` separately.
- **`Error::check_with_op(code, op)`** -- Mirror of `check` that
  propagates the operation tag on the error path.
- **`examples/test_error_context.rs`** -- Integration example that
  triggers an out-of-range broadcast root and asserts the new
  operation-tagged error format end-to-end.
- **`examples/test_request_table_concurrency.rs`** -- Multi-threaded
  isend/irecv stress test (4 threads × 100 iterations) that exercises
  the request table under `MPI_THREAD_MULTIPLE`.
- **`docs/adr/0002-handle-tables.md`** -- Architecture Decision Record
  explaining the C11-atomics-with-CAS strategy chosen for the request
  handle table and the rejected alternatives (pthread mutex, Treiber
  stack).

### Changed

- **All 101 internal FFI call sites now tag errors with the underlying
  C function name** (e.g., `"bcast"`, `"allreduce"`,
  `"allreduce_init"`, `"wait"`, `"isend"`). When an MPI call fails,
  the resulting `Error::Mpi.operation` is populated with the tag,
  giving downstream code (notably cobre) structured error context
  without needing to invent sentinel values.

### Fixed

- **Request handle table is now safe under `MPI_THREAD_MULTIPLE`.**
  Previously, concurrent `alloc_request` calls could observe the same
  `request_used[i] == 0` slot and both write `1`, with the second
  thread's `request_table[i]` write clobbering the first -- a silent
  lost-request data race that TSan would flag. The C wrapper now uses
  `atomic_compare_exchange_strong_explicit` on `request_used[i]` with
  `memory_order_acq_rel` semantics, paired with an
  `atomic_store_explicit(..., memory_order_release)` on free and an
  `atomic_load_explicit(..., memory_order_acquire)` on read. The
  comm/win/info tables retain their existing implementations and are
  scheduled for hardening in a later release.

## [0.3.0] - 2026-04-10

### Added

- **Topology reporting** -- New `Communicator::topology(&mpi)` collective that
  gathers rank-to-host mapping across all processes and returns a `TopologyInfo`
  struct. The `Display` implementation produces a human-readable report showing
  MPI library version, standard version, thread level, process distribution
  across nodes, and (with the `numa` feature) SLURM job metadata.
- **`Mpi::library_version()`** -- Returns the MPI implementation version string
  (e.g. "Open MPI v4.1.6") by wrapping `MPI_Get_library_version`.
- **`TopologyInfo`**, **`HostEntry`**, and **`SlurmInfo`** public types with
  accessors for programmatic inspection of job topology.
- **`topology` example** -- Demonstrates one-liner topology reporting and
  programmatic access to host/rank mapping.

### Fixed

- 34 findings from security/correctness assessment addressed.

## [0.2.2] - 2026-03-27

### Fixed

- **aarch64 compatibility** -- All `c_char` casts now use `std::ffi::c_char`
  instead of hardcoded `i8`. On aarch64 (ARM), `c_char` is `u8` (unsigned),
  while on x86_64 it is `i8` (signed). The previous `.cast::<i8>()` calls
  caused type mismatches on ARM targets. Affected: `get_version`,
  `get_processor_name`, `error_info`, `info_get`.

## [0.2.1] - 2026-03-27

### Fixed

- **Remove RPATH from linked binaries** -- Build script no longer embeds
  `-Wl,-rpath` with the build machine's library paths. Pre-built release
  binaries were failing on HPC clusters and containers where MPI is installed
  in a different path. Users must ensure `libmpi` is discoverable at runtime
  via `LD_LIBRARY_PATH`, `ldconfig`, or their cluster's module system.

### Changed

- **Repository URLs** -- Updated all URLs after migration to `cobre-rs` org.

## [0.2.0] - 2026-02-13

### Breaking Changes

- Removed all type-specific methods (`_f64`, `_i32`, etc.) in favor of generic `MpiDatatype` API
- `Communicator` is now `Send + Sync` for hybrid MPI+threads programs
- Error type restructured: `Error::MpiError(i32)` → `Error::Mpi { class, code, message }` with `MpiErrorClass` enum providing rich error categorization
- Removed `InvalidRank`, `InvalidCommunicator`, etc. (now covered by `MpiErrorClass` variants)

### Added

- Generic `MpiDatatype` trait for `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`
- Communicator management: `split()`, `split_type()`, `split_shared()`, `duplicate()`
- MPI_Info object support with RAII lifecycle (`Info` type)
- Complete nonblocking point-to-point: `isend`, `irecv`, `sendrecv`
- Probe/Iprobe: `probe<T>`, `iprobe<T>` with `Status` struct (source, tag, count)
- Scalar reduce variants: `reduce_scalar`, `allreduce_scalar`
- In-place reduce variants: `reduce_inplace`, `allreduce_inplace`
- Scan/Exscan: `scan`, `exscan`, `scan_scalar`, `exscan_scalar` (blocking, nonblocking, persistent)
- V-collectives: `gatherv`, `scatterv`, `allgatherv`, `alltoallv` (blocking, nonblocking, persistent)
- Alltoall: `alltoall` (blocking, nonblocking, persistent)
- Reduce-scatter-block: `reduce_scatter_block` (blocking, nonblocking, persistent)
- All 15 nonblocking collective variants (`i`-prefixed)
- All 15 persistent collective variants (`*_init`, MPI 4.0+)
- Shared memory windows: `SharedWindow<T>` with RAII lock guards (`LockGuard`, `LockAllGuard`) (feature: `rma`)
- Window synchronization: `fence`, `lock`, `lock_all`, `flush`, `flush_all`
- SLURM environment helpers: `is_slurm_job`, `job_id`, `local_rank`, `local_size`, `num_nodes`, `cpus_per_task`, `node_name`, `node_list` (feature: `numa`)
- Comprehensive test suite: unit tests for `slurm` module, MPI integration test runner (`tests/run_mpi_tests.sh`)
- CI matrix: MPICH × OpenMPI × default/rma feature combinations
- New examples: `comm_split`, `scan`, `gatherv`, `shared_memory`, `hybrid_openmp`

### Changed

- C handle table limits expanded: 256 communicators, 16384 requests, 256 windows, 64 infos
- Rich error messages via `MPI_Error_class` + `MPI_Error_string`
- C wrapper layer significantly expanded (~2400 lines, up from ~700)

## [0.1.0] - 2026-01-13

### Added

- Initial release
- MPI 4.0+ support with persistent collectives
- Safe Rust API for MPI operations
- Examples: hello_world, ring, allreduce, nonblocking, persistent_bcast, pi_monte_carlo
- Comprehensive documentation
- Initial CI/CD setup with GitHub Actions

[Unreleased]: https://github.com/cobre-rs/ferrompi/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/cobre-rs/ferrompi/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/cobre-rs/ferrompi/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/cobre-rs/ferrompi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cobre-rs/ferrompi/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/cobre-rs/ferrompi/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cobre-rs/ferrompi/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cobre-rs/ferrompi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cobre-rs/ferrompi/releases/tag/v0.1.0
