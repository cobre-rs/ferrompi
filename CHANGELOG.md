# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- All 15 nonblocking collective variants: `ibroadcast`, `iallreduce`, `ireduce`, `igather`, `iallgather`, `iscatter`, `ibarrier`, `iscan`, `iexscan`, `ialltoall`, `igatherv`, `iscatterv`, `iallgatherv`, `ialltoallv`, `ireduce_scatter_block`
- All 15 persistent collective variants (MPI 4.0+): `bcast_init`, `allreduce_init`, `allreduce_init_inplace`, `reduce_init`, `gather_init`, `scatter_init`, `allgather_init`, `scan_init`, `exscan_init`, `alltoall_init`, `gatherv_init`, `scatterv_init`, `allgatherv_init`, `alltoallv_init`, `reduce_scatter_block_init`
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

[Unreleased]: https://github.com/cobre-rs/ferrompi/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/cobre-rs/ferrompi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cobre-rs/ferrompi/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/cobre-rs/ferrompi/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cobre-rs/ferrompi/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cobre-rs/ferrompi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cobre-rs/ferrompi/releases/tag/v0.1.0
