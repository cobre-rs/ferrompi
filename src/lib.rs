//! # ferrompi
//!
//! Safe, generic Rust bindings for MPI (Message Passing Interface).
//!
//! This crate wraps MPI functionality through a thin C layer, providing:
//! - Type-safe generic API for all MPI datatypes
//! - Blocking, nonblocking, and persistent (MPI 4.0+) collectives
//! - Communicator management (split, duplicate)
//! - RMA shared memory windows (with `rma` feature)
//! - SLURM environment helpers (with `numa` feature)
//! - Large count support (MPI 4.0+ `_c` variants for blocking/nonblocking
//!   collectives; persistent collectives currently reject `count > INT_MAX`
//!   with `MPI_ERR_COUNT` — full `_c` dispatch for persistent ops is deferred)
//!
//! ## Supported Types
//!
//! All communication operations are generic over [`MpiDatatype`]:
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`
//!
//! ## Quick Start
//!
//! ```no_run
//! use ferrompi::{Mpi, ReduceOp};
//!
//! fn main() -> Result<(), ferrompi::Error> {
//!     let mpi = Mpi::init()?;
//!     let world = mpi.world();
//!
//!     let rank = world.rank();
//!     let size = world.size();
//!     println!("Hello from rank {} of {}", rank, size);
//!
//!     // Generic broadcast — works with any MpiDatatype
//!     let mut data = vec![0.0f64; 100];
//!     if rank == 0 {
//!         data.fill(42.0);
//!     }
//!     world.broadcast(&mut data, 0)?;
//!
//!     // Generic all-reduce
//!     let sum = world.allreduce_scalar(rank as f64, ReduceOp::Sum)?;
//!     println!("Rank {rank}: sum of all ranks = {sum}");
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Dependencies |
//! |---------|-------------|--------------|
//! | `rma`   | RMA shared memory window operations | — |
//! | `numa`  | NUMA-aware windows and SLURM helpers | `rma` |
//!
//! ## Capabilities
//!
//! - **Generic API**: All operations work with any [`MpiDatatype`] (`f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`)
//! - **Blocking collectives**: barrier, broadcast, reduce, allreduce, gather, scatter, allgather,
//!   alltoall, scan, exscan, reduce\_scatter\_block, plus V-variants (gatherv, scatterv, allgatherv, alltoallv)
//! - **Nonblocking collectives**: All 15 `i`-prefixed variants with [`Request`] handles
//! - **Persistent collectives** (MPI 4.0+): All 15 `_init` variants with [`PersistentRequest`] handles
//! - **Scalar and in-place variants**: `reduce_scalar`, `allreduce_scalar`, `reduce_inplace`,
//!   `allreduce_inplace`, `scan_scalar`, `exscan_scalar`
//! - **Point-to-point**: `send`, `recv`, `isend`, `irecv`, `sendrecv`, `probe`, `iprobe`
//! - **Communicator management**: `split`, `split_type`, `split_shared`, `duplicate`
//! - **Group operations**: [`Group`] with incl/excl/union/intersection/difference,
//!   [`RankRange`] for range constructors, [`GroupComparison`].
//!
//!   Note: [`Mpi::create_from_group`] requires MPI 4.0+.
//!   Support is probed once and cached; see the function rustdoc for the cache invariant.
//! - **Custom datatypes**: [`CustomDatatype`]
//!   (contiguous/vector/struct/resized) and [`StructField`]
//!   for struct-type builders.
//! - **User-defined reduction operations**: [`UserOp`] wraps `MPI_Op_create`
//!   with safe closure storage and trampoline.
//! - **Distributed RMA windows** (feature `rma`): [`Win<T>`](crate::Win) with
//!   [`WinFenceAssert`], [`WinPscwAssert`],
//!   [`WinLockGuard`], and [`WinLockAllGuard`]
//!   RAII guards.
//! - **Info objects**: [`Info`] for runtime hint passing to communicator,
//!   window, and operation constructors.
//! - **Persistent point-to-point**: `send_init`, `bsend_init`, `rsend_init`, `ssend_init`,
//!   `recv_init` methods on [`Communicator`], each returning a
//!   [`PersistentRequest`].
//! - **Shared memory windows** (feature `rma`): [`SharedWindow<T>`](crate::SharedWindow)
//!   with RAII lock guards for NUMA-aware intra-node shared memory (distinct from the
//!   distributed [`Win<T>`](crate::Win) windows above).
//! - **SLURM helpers** (feature `numa`): Job topology queries via `slurm` module
//! - **Rich error handling**: [`MpiErrorClass`] categorization with messages from the MPI runtime
//!
//! ## Thread Safety
//!
//! [`Communicator`] is `Send + Sync` to support hybrid MPI + threads programs
//! (e.g., MPI between nodes, `std::thread::scope` within a node).
//!
//! The actual thread-safety guarantees depend on the thread level requested
//! at initialization:
//!
//! | Thread Level | Who can call MPI | Synchronization |
//! |--------------|------------------|-----------------|
//! | [`ThreadLevel::Single`] | Main thread only | N/A |
//! | [`ThreadLevel::Funneled`] | Main thread only | N/A |
//! | [`ThreadLevel::Serialized`] | Any thread | User must serialize |
//! | [`ThreadLevel::Multiple`] | Any thread | None needed |
//!
//! ```no_run
//! use ferrompi::{Mpi, ThreadLevel};
//!
//! // Request serialized thread support for hybrid MPI + threads
//! let mpi = Mpi::init_thread(ThreadLevel::Funneled).unwrap();
//! assert!(mpi.thread_level() >= ThreadLevel::Funneled);
//! ```
//!
//! [`Mpi`] itself is `!Send + !Sync` — MPI initialization and finalization
//! must occur on the same thread. Only [`Communicator`] handles (and the
//! operations on them) may cross thread boundaries.
//!
//! ### Send/Sync Status of Public Types
//!
//! | Type | Send/Sync | Notes |
//! |------|-----------|-------|
//! | [`Communicator`] | `Send + Sync` | Explicit `unsafe impl` in `src/comm/mod.rs`; cross-thread use is the primary hybrid MPI use case. |
//! | [`Mpi`] | `!Send + !Sync` | `PhantomData<*const ()>` field; init and finalize must occur on the same thread. |
//! | [`Group`] | `Send + Sync` | Explicit `unsafe impl` in `src/group.rs`; handles are opaque integers, MPI-thread-safe under `MPI_THREAD_MULTIPLE`. |
//! | [`Request`] | `Send + Sync` | Auto-derived; `i64` + `bool` fields. Cross-thread use requires `MPI_THREAD_MULTIPLE`. Buffer-lifetime invariant still applies. |
//! | [`PersistentRequest`] | `Send + Sync` | Auto-derived; same shape as `Request`. ADR-0004 §"Drop behavior" applies across thread boundaries. |
//! | [`Status`] | `Send + Sync` | POD wrapper; all fields are `Copy`. |
//! | [`CustomDatatype`] | `Send + Sync` | Explicit `unsafe impl` in `src/datatype_builder.rs`; handle is an opaque integer. |
//! | [`Info`] | `Send + Sync` | Auto-derived from `i32` + `bool` fields; MPI info objects are thread-safe under `MPI_THREAD_MULTIPLE`. |
//! | [`UserOp<T>`] | `Send + Sync` (for `T: MpiDatatype`) | Auto-derived: fields are `i32` + `PhantomData<T>`. The trait bound `MpiDatatype: Copy + Send + 'static` and the fact that all concrete `MpiDatatype` impls are also `Sync` give `Send + Sync` for `UserOp<T>`. The global closure registry uses internal `unsafe impl Send/Sync` on its slots; that is a separate object from `UserOp<T>` itself. |
//! | [`Win<T>`](crate::Win) (feature `rma`) | `!Send + !Sync` | `NonNull<T>` field suppresses auto-traits; RMA window's local memory pointer is not safe to share across threads. |
//! | [`SharedWindow<T>`](crate::SharedWindow) (feature `rma`) | `!Send + !Sync` | `NonNull<T>` field; same rationale as `Win<T>`. |
//! | [`LockGuard<'a, T>`] (feature `rma`) | `!Send + !Sync` | Borrows `SharedWindow<T>`; inherits non-Send/Sync. |
//! | [`LockAllGuard<'a, T>`] (feature `rma`) | `!Send + !Sync` | Borrows `SharedWindow<T>`; inherits non-Send/Sync. |
//! | [`WinLockGuard<'g, 'a, T>`](crate::WinLockGuard) (feature `rma`) | `!Send + !Sync` | Borrows `Win<T>`; inherits non-Send/Sync. |
//! | [`WinLockAllGuard<'g, 'a, T>`](crate::WinLockAllGuard) (feature `rma`) | `!Send + !Sync` | Borrows `Win<T>`; inherits non-Send/Sync. |
//!
//! ## Hybrid MPI+OpenMP
//!
//! For hybrid parallelism, use [`Mpi::init_thread()`] with the appropriate level:
//!
//! - **[`Funneled`](ThreadLevel::Funneled)** (recommended): Only the main thread makes MPI calls.
//!   OpenMP threads handle computation between MPI calls.
//! - **[`Serialized`](ThreadLevel::Serialized)**: Any thread can make MPI calls, but only one at a time.
//! - **[`Multiple`](ThreadLevel::Multiple)**: Full concurrent MPI from any thread (highest overhead).
//!
//! ```no_run
//! use ferrompi::{Mpi, ThreadLevel, ReduceOp};
//!
//! let mpi = Mpi::init_thread(ThreadLevel::Funneled).unwrap();
//! assert!(mpi.thread_level() >= ThreadLevel::Funneled);
//!
//! let world = mpi.world();
//! // Worker threads compute locally, main thread calls MPI
//! let local = 42.0_f64;
//! let global = world.allreduce_scalar(local, ReduceOp::Sum).unwrap();
//! ```
//!
//! ### SLURM Configuration
//!
//! ```bash
//! #SBATCH --ntasks-per-node=4        # MPI ranks per node
//! #SBATCH --cpus-per-task=8          # OpenMP threads per rank
//! #SBATCH --bind-to core             # Pin MPI ranks
//! export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
//! srun ./my_program
//! ```
//!
//! Use the `slurm` module (with `numa` feature) to read these values at runtime.
//! See `examples/hybrid_openmp.rs` for the full pattern.
//!
//! ## Extended documentation
//!
//! Long-form documentation artifacts are embedded in the [`doc`] module and
//! render as individual pages in this rustdoc. The same content is available
//! as plain Markdown in the `docs/` directory.
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`doc::architecture`] | Six-layer stack, handle tables, thread-safety model, FFI/ABI invariants, and generic `MpiDatatype` design |
//! | [`doc::migrating_from_rsmpi`] | Function-for-function API mapping and migration cookbook from rsmpi |
//! | [`doc::mpi_compatibility`] | Compatibility matrix for MPICH, Open MPI, Intel MPI, and Cray MPI |
//! | [`doc::adr_0001_why_c_wrapper`] | ADR-0001: why a hand-written C wrapper is used instead of `bindgen` |
//! | [`doc::adr_0002_handle_tables`] | ADR-0002: C11 atomic CAS strategy for the request-table under `MPI_THREAD_MULTIPLE` |
//! | [`doc::adr_0003_generic_mpi_datatype`] | ADR-0003: sealed `MpiDatatype` trait family and `DatatypeTag` ABI contract |
//! | [`doc::adr_0004_persistent_collective_approach`] | ADR-0004: `PersistentRequest` lifecycle and buffer-lifetime invariants |
//! | [`doc::adr_0005_mpi_op_create`] | ADR-0005: `MPI_Op_create` closure storage, trampoline safety, and drop ordering |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(clippy::undocumented_unsafe_blocks)]
// Clippy suppressions live at the call site (`#[allow(clippy::NAME)]`
// with a justification comment) rather than crate-wide.

use std::ffi::{c_char, CString};
#[cfg(feature = "rma")]
use std::io::Write;

mod comm;
mod datatype;
mod datatype_builder;
pub mod doc;
mod error;
mod ffi;
mod group;
mod info;
mod op;
mod persistent;
mod request;
mod rt;
#[cfg(feature = "numa")]
pub mod slurm;
mod status;
mod topology;
#[cfg(feature = "rma")]
mod window;

pub use comm::{Communicator, SplitType};
#[cfg(feature = "rma")]
pub use datatype::AtomicMpiDatatype;
pub use datatype::{
    BytePermutable, DatatypeTag, DoubleInt, FloatInt, Int2, LongDoubleInt, LongInt, MpiDatatype,
    MpiIndexedDatatype, PlainData, ShortInt,
};
pub use datatype_builder::{CustomDatatype, StructField};
pub use error::{Error, MpiErrorClass, ResourceKind, Result};
pub use group::{Group, GroupComparison, RankRange};
pub use info::Info;
pub use op::UserOp;
pub use persistent::PersistentRequest;
pub use request::Request;
pub use status::Status;
#[cfg(feature = "numa")]
pub use topology::SlurmInfo;
pub use topology::{HostEntry, TopologyInfo};
#[cfg(feature = "rma")]
pub use window::{
    LockAllGuard, LockGuard, LockType, PendingFetchResult, SharedWindow, Win, WinFenceAssert,
    WinKind, WinLockAllGuard, WinLockGuard, WinPscwAssert,
};

use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

/// Process-wide buffer attached for buffered sends (`MPI_Buffer_attach`).
///
/// MPI allows at most one attached buffer per process. This static holds the
/// `Box<[u8]>` so that the allocation remains valid for the duration of the
/// attachment. The `Mutex` is held only briefly during `buffer_attach` and
/// `buffer_detach` transitions; MPI itself manages the buffer between those
/// two calls.
static ATTACHED_BUFFER: Mutex<Option<Box<[u8]>>> = Mutex::new(None);

/// MPI thread support levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum ThreadLevel {
    /// Only single-threaded execution
    Single = 0,
    /// Multi-threaded, but MPI calls only from main thread
    Funneled = 1,
    /// Multi-threaded, but MPI calls serialized by user
    Serialized = 2,
    /// Full multi-threaded support
    Multiple = 3,
}

/// Reduction operations
///
/// The `Replace` and `NoOp` variants are only available with the `rma` feature.
///
/// # Feature-gated variants
///
/// Without `--features rma`, referencing `ReduceOp::Replace` is a compile error:
///
#[cfg_attr(not(feature = "rma"), doc = "```compile_fail")]
#[cfg_attr(
    not(feature = "rma"),
    doc = "// This must not compile without --features rma."
)]
#[cfg_attr(not(feature = "rma"), doc = "let _ = ferrompi::ReduceOp::Replace;")]
#[cfg_attr(not(feature = "rma"), doc = "```")]
#[cfg_attr(feature = "rma", doc = "```no_run")]
#[cfg_attr(
    feature = "rma",
    doc = "// With --features rma, ReduceOp::Replace is available."
)]
#[cfg_attr(feature = "rma", doc = "let _ = ferrompi::ReduceOp::Replace;")]
#[cfg_attr(feature = "rma", doc = "```")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    /// Sum of values
    Sum = 0,
    /// Maximum value
    Max = 1,
    /// Minimum value
    Min = 2,
    /// Product of values
    Prod = 3,
    /// Bitwise OR (`MPI_BOR`). Valid only for integer types; MPI returns
    /// `MPI_ERR_OP` when used with floating-point types.
    BitwiseOr = 4,
    /// Bitwise AND (`MPI_BAND`). Valid only for integer types; MPI returns
    /// `MPI_ERR_OP` when used with floating-point types.
    BitwiseAnd = 5,
    /// Bitwise XOR (`MPI_BXOR`). Valid only for integer types; MPI returns
    /// `MPI_ERR_OP` when used with floating-point types.
    BitwiseXor = 6,
    /// Logical OR (`MPI_LOR`). Interprets nonzero as `true`. Valid for
    /// integer types.
    LogicalOr = 7,
    /// Logical AND (`MPI_LAND`). Interprets nonzero as `true`. Valid for
    /// integer types.
    LogicalAnd = 8,
    /// Logical XOR (`MPI_LXOR`). Interprets nonzero as `true`. Valid for
    /// integer types.
    LogicalXor = 9,
    /// Maximum value with location (`MPI_MAXLOC`). Returns the maximum value
    /// and the rank (index) where it occurred. Only valid with
    /// [`MpiIndexedDatatype`] via
    /// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
    MaxLoc = 10,
    /// Minimum value with location (`MPI_MINLOC`). Returns the minimum value
    /// and the rank (index) where it occurred. Only valid with
    /// [`MpiIndexedDatatype`] via
    /// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
    MinLoc = 11,
    /// Replace the target buffer with the source value (`MPI_REPLACE`).
    ///
    /// Only valid for `MPI_Accumulate`-family operations. Passing to
    /// `allreduce`, `reduce`, `scan`, etc. returns `MPI_ERR_OP` from MPI.
    ///
    /// This variant is only present when the `rma` feature is enabled.
    #[cfg(feature = "rma")]
    Replace = 12,
    /// No-op: leaves the target buffer unchanged (`MPI_NO_OP`).
    ///
    /// Only valid for `MPI_Accumulate`-family operations. Passing to
    /// `allreduce`, `reduce`, `scan`, etc. returns `MPI_ERR_OP` from MPI.
    ///
    /// # Compile-time availability
    ///
    /// This variant is only present when the `rma` feature is enabled.
    #[cfg(feature = "rma")]
    NoOp = 13,
}

/// MPI environment handle.
///
/// This type represents an initialized MPI environment. There can only be one
/// instance of this type at a time. When dropped, it finalizes MPI. After the
/// handle is dropped, every method that calls MPI through a communicator,
/// group, request, window, datatype, info or user op returns
/// `Err(`[`Error::Finalized`]`)` without calling MPI. The queries the MPI
/// standard allows at any time ([`Mpi::version`], [`Mpi::library_version`],
/// [`Mpi::wtime`], [`Mpi::is_initialized`], [`Mpi::is_finalized`]) still
/// answer.
///
/// If any window (feature `rma`) is still alive when this handle is
/// dropped, `MPI_Finalize` is skipped instead — with a stderr warning —
/// because some MPI implementations free MPI-allocated window memory inside
/// `MPI_Finalize` itself, and some abort while tearing down internal state
/// that still tracks a live window.
///
/// At [`ThreadLevel::Serialized`]/[`ThreadLevel::Multiple`], dropping this
/// handle while another thread is still inside an MPI call through this
/// crate is a program error that `ferrompi` does not detect.
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// let mpi = Mpi::init().expect("Failed to initialize MPI");
/// let world = mpi.world();
/// println!("Running on {} processes", world.size());
/// // MPI is finalized when `mpi` goes out of scope
/// ```
pub struct Mpi {
    /// The thread level that was provided
    thread_level: ThreadLevel,
    /// Marker to make Mpi !Send and !Sync
    _marker: PhantomData<*const ()>,
}

impl Mpi {
    /// Initialize MPI with single-threaded support.
    ///
    /// # Errors
    ///
    /// Returns `Err(`[`Error::AlreadyInitialized`]`)` while an `Mpi` handle
    /// exists or another thread is initializing, `Err(`[`Error::Finalized`]`)`
    /// once MPI has been finalized, or `Err(`[`Error::Mpi`]`)` if
    /// `MPI_Init_thread` itself fails.
    pub fn init() -> Result<Self> {
        Self::init_thread(ThreadLevel::Single)
    }

    /// Initialize MPI with the specified thread support level.
    ///
    /// # Arguments
    ///
    /// * `required` - The minimum thread support level required
    ///
    /// # Returns
    ///
    /// Returns the MPI handle. The actual thread support level provided can be
    /// queried with [`thread_level()`](Self::thread_level).
    ///
    /// # Errors
    ///
    /// Returns `Err(`[`Error::AlreadyInitialized`]`)` while an `Mpi` handle
    /// exists or another thread is initializing, `Err(`[`Error::Finalized`]`)`
    /// once MPI has been finalized, or `Err(`[`Error::Mpi`]`)` if
    /// `MPI_Init_thread` itself fails.
    pub fn init_thread(required: ThreadLevel) -> Result<Self> {
        rt::begin_init()?;

        let mut already_finalized: i32 = 0;
        // SAFETY: already_finalized is a local out-parameter that
        // ferrompi_finalized writes before this reads it below;
        // ferrompi_finalized is legal to call at any point in the process
        // lifecycle, including before MPI_Init_thread.
        unsafe { ffi::ferrompi_finalized(&mut already_finalized) };
        if already_finalized != 0 {
            rt::abandon_init();
            return Err(Error::Finalized);
        }

        let mut provided: i32 = 0;
        // SAFETY: provided is a local out-parameter written by MPI_Init_thread;
        // rt::begin_init's compare-exchange above guarantees this is the only
        // MPI_Init_thread call in the process at a time, and the finalized
        // check above guarantees MPI has not already been finalized.
        let ret = unsafe { ffi::ferrompi_init_thread(required as i32, &mut provided) };

        if ret != 0 {
            rt::abandon_init();
            return Err(Error::Mpi {
                class: MpiErrorClass::Raw(ret),
                code: ret,
                message: format!("MPI_Init_thread failed with code {ret}"),
                operation: Some("init_thread"),
            });
        }

        let thread_level = match provided {
            0 => ThreadLevel::Single,
            1 => ThreadLevel::Funneled,
            2 => ThreadLevel::Serialized,
            _ => ThreadLevel::Multiple,
        };

        rt::activate(thread_level);

        Ok(Mpi {
            thread_level,
            _marker: PhantomData,
        })
    }

    /// Get the thread support level that was provided.
    pub fn thread_level(&self) -> ThreadLevel {
        self.thread_level
    }

    /// Get a handle to `MPI_COMM_WORLD`.
    pub fn world(&self) -> Communicator {
        Communicator::world()
    }

    /// Get the current wall-clock time.
    ///
    /// This is a high-resolution timer suitable for benchmarking.
    pub fn wtime() -> f64 {
        // SAFETY: ferrompi_wtime takes no pointer arguments and touches no
        // Rust-owned memory; it is a pure read of MPI_Wtime().
        unsafe { ffi::ferrompi_wtime() }
    }

    /// Get the MPI library version string (implementation-specific).
    ///
    /// Returns a string such as `"Open MPI v4.1.6"` or `"Intel(R) MPI Library 2021.7"`.
    /// This wraps `MPI_Get_library_version`.
    pub fn library_version() -> Result<String> {
        // MPI_MAX_LIBRARY_VERSION_STRING is 8192 in most implementations.
        let mut buf = [0u8; 8192];
        let mut len: i32 = 0;
        // SAFETY: buf is a local 8192-byte buffer sized to
        // MPI_MAX_LIBRARY_VERSION_STRING; the C layer writes at most buf.len()
        // bytes into it and reports the written length through the local `len`
        // out-parameter.
        let ret = unsafe {
            ffi::ferrompi_get_library_version(buf.as_mut_ptr().cast::<c_char>(), &mut len)
        };
        Error::check_with_op(ret, "get_library_version")?;
        let len = (len.max(0) as usize).min(buf.len());
        // Trim trailing whitespace/newlines that some implementations append.
        let s = std::str::from_utf8(&buf[..len])
            .map_err(|_| Error::Internal("Invalid UTF-8 in library version string".into()))?;
        Ok(s.trim_end().to_string())
    }

    /// Get the MPI standard version string (e.g., "MPI 4.0").
    pub fn version() -> Result<String> {
        let mut buf = [0u8; 256];
        let mut len: i32 = 0;
        // SAFETY: buf is a local 256-byte buffer, large enough for an MPI
        // version string ("MPI x.y"); the C layer writes at most buf.len()
        // bytes into it and reports the written length through `len`.
        let ret = unsafe { ffi::ferrompi_get_version(buf.as_mut_ptr().cast::<c_char>(), &mut len) };
        Error::check_with_op(ret, "get_version")?;

        let len = (len.max(0) as usize).min(buf.len());
        let s = std::str::from_utf8(&buf[..len])
            .map_err(|_| Error::Internal("Invalid UTF-8 in version string".into()))?;
        Ok(s.to_string())
    }

    /// Check if MPI has been initialized.
    pub fn is_initialized() -> bool {
        let mut flag: i32 = 0;
        // SAFETY: flag is a local out-parameter that ferrompi_initialized writes
        // before this function reads it below.
        unsafe { ffi::ferrompi_initialized(&mut flag) };
        flag != 0
    }

    /// Check if MPI has been finalized.
    ///
    /// Returns `true` once the `Mpi` handle has been dropped, including when
    /// `MPI_Finalize` itself was skipped because a window was still alive.
    pub fn is_finalized() -> bool {
        if rt::is_finalized() {
            return true;
        }
        let mut flag: i32 = 0;
        // SAFETY: flag is a local out-parameter that ferrompi_finalized writes
        // before this function reads it below.
        unsafe { ffi::ferrompi_finalized(&mut flag) };
        flag != 0
    }

    /// Returns `true` if the runtime MPI version is 4.0 or later.
    ///
    /// Caching semantics:
    /// - An `Err` from `Mpi::version()` (e.g., called before `Mpi::init`)
    ///   is NOT cached; a subsequent call after init can re-probe and
    ///   observe support correctly.
    /// - A successful `version()` whose string parses to a major version
    ///   `≥ 4` caches `true`.
    /// - A successful `version()` whose string parses to a major version
    ///   `< 4` (including unrecognized formats that yield major `= 0` via
    ///   the `unwrap_or(0)` fallback) caches `false`. Re-probing is not
    ///   possible once a successful `version()` has been seen, so a
    ///   non-standard version string format from an unusual MPI build
    ///   permanently disables `Mpi::create_from_group` for this process.
    fn supports_create_from_group() -> bool {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        if let Some(&cached) = SUPPORTED.get() {
            return cached;
        }
        // Probe. If `version()` fails, return `false` without caching;
        // a future call can re-probe successfully.
        let Ok(v) = Mpi::version() else {
            return false;
        };
        // Mpi::version() returns a string like "MPI 4.0" or "MPI 3.1".
        // Extract the major version number from the second whitespace-delimited
        // token, then its first dot-delimited component.
        let major: u32 = v
            .split_whitespace()
            .nth(1)
            .and_then(|tok| tok.split('.').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let supported = major >= 4;
        // Race: if another thread already won set(), discard our value;
        // both threads agree on the result anyway because the probe is
        // deterministic given a successful version() call.
        let _ = SUPPORTED.set(supported);
        supported
    }

    /// Create a communicator from a group without requiring a parent
    /// communicator (MPI 4.0+).
    ///
    /// `stringtag` must be identical across all ranks that participate
    /// in the call; ranks with different tags or in different groups
    /// produce separate communicators.
    ///
    /// # Errors
    ///
    /// - Returns `Err(Error::Internal(_))` if `stringtag` contains a null byte
    ///   (the FFI call is never invoked in this case).
    /// - Returns `Err(Error::NotSupported("MPI_Comm_create_from_group"))` on
    ///   MPI < 4.0 installations.
    /// - Returns `Err(Error::Mpi { .. })` if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let g = world.group().unwrap();
    /// // All ranks participate with the same tag.
    /// let comm = mpi.create_from_group(&g, "my-tag").unwrap();
    /// assert_eq!(comm.size(), world.size());
    /// ```
    pub fn create_from_group(&self, group: &group::Group, stringtag: &str) -> Result<Communicator> {
        let c_tag = CString::new(stringtag)
            .map_err(|_| Error::Internal("stringtag contains null byte".into()))?;
        if !Self::supports_create_from_group() {
            return Err(Error::NotSupported(
                "MPI_Comm_create_from_group".to_string(),
            ));
        }
        let mut new_handle: i32 = -1;
        // SAFETY: c_tag.as_ptr() is a valid, null-terminated C string that
        // lives for the duration of this call. group.handle is a valid group
        // handle obtained from ferrompi_comm_group or a group-constructor shim.
        // &mut new_handle is a pointer to a stack-allocated i32.
        let ret = unsafe {
            ffi::ferrompi_comm_create_from_group(group.handle, c_tag.as_ptr(), &mut new_handle)
        };
        Error::check_with_op(ret, "comm_create_from_group")?;
        Communicator::from_handle(new_handle)
    }

    /// Attach a user-provided buffer for use by buffered sends.
    ///
    /// This wraps `MPI_Buffer_attach`. Once attached, the buffer is owned by
    /// MPI until [`buffer_detach`](Self::buffer_detach) is called. Only one
    /// buffer may be attached per process at a time; attempting to attach a
    /// second buffer without first detaching returns
    /// `Err(`[`Error::InvalidOp`]`)`.
    ///
    /// The `buffer` is stored in a process-wide static so its allocation
    /// remains valid for the lifetime of the attachment. You must not access
    /// the raw bytes of `buffer` between `buffer_attach` and `buffer_detach` —
    /// MPI owns the contents during that window.
    ///
    /// # Buffer Sizing
    ///
    /// The recommended buffer size for `N` buffered sends of `count` elements
    /// of type `T` is:
    ///
    /// ```text
    /// N * (MPI_BSEND_OVERHEAD + count * size_of::<T>())
    /// ```
    ///
    /// `MPI_BSEND_OVERHEAD` is implementation-specific; use at least a few
    /// hundred extra bytes per buffered send. For safety, use a generous margin.
    ///
    /// Buffers larger than `i32::MAX` bytes are rejected with
    /// `Err(`[`Error::InvalidBuffer`]`)` before the FFI call; the
    /// underlying `MPI_Buffer_attach` takes an `int` count and cannot
    /// address larger buffers.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if `buffer.len() > i32::MAX as usize`.
    /// - [`Error::InvalidOp`] if a buffer is already attached.
    /// - [`Error::Mpi`] if `MPI_Buffer_attach` fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// let mpi = Mpi::init().unwrap();
    /// mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice()).unwrap();
    /// // ... buffered sends here ...
    /// let _ = mpi.buffer_detach().unwrap();
    /// ```
    pub fn buffer_attach(&self, buffer: Box<[u8]>) -> Result<()> {
        let mut guard = ATTACHED_BUFFER
            .lock()
            .map_err(|_| Error::Internal("ATTACHED_BUFFER mutex poisoned".into()))?;
        if guard.is_some() {
            return Err(Error::InvalidOp);
        }
        if buffer.len() > i32::MAX as usize {
            return Err(Error::InvalidBuffer);
        }
        let ptr = buffer.as_ptr() as *mut std::ffi::c_void;
        let size = buffer.len() as i64;
        // Store the box in the static BEFORE calling MPI so the memory is
        // guaranteed alive when MPI begins using the buffer.
        *guard = Some(buffer);
        // SAFETY: ptr points to the boxed slice we just stored in the static;
        // the allocation remains valid until buffer_detach drops it. size is
        // the exact byte length of that allocation.
        let ret = unsafe { ffi::ferrompi_buffer_attach(ptr, size) };
        if ret != 0 {
            // Roll back: reclaim the box so the caller can retry.
            guard.take();
            return Err(Error::from_code_with_op(ret, "buffer_attach"));
        }
        Ok(())
    }

    /// Detach the previously attached buffer and return it to the caller.
    ///
    /// This wraps `MPI_Buffer_detach`. The call **blocks** until all buffered
    /// sends that are currently using the buffer have completed. Once this
    /// returns, the returned `Box<[u8]>` is owned by the caller again and
    /// may be dropped or reused.
    ///
    /// Do **not** call this from a `Drop` implementation (e.g., on a wrapper
    /// around `Mpi`). `MPI_Buffer_detach` blocks until pending sends drain;
    /// blocking inside `Drop` can produce hard-to-diagnose hangs.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOp`] if no buffer is currently attached.
    /// - [`Error::Mpi`] if `MPI_Buffer_detach` fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// let mpi = Mpi::init().unwrap();
    /// mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice()).unwrap();
    /// // ... buffered sends ...
    /// let _buf = mpi.buffer_detach().unwrap(); // dropped here
    /// ```
    pub fn buffer_detach(&self) -> Result<Box<[u8]>> {
        let mut guard = ATTACHED_BUFFER
            .lock()
            .map_err(|_| Error::Internal("ATTACHED_BUFFER mutex poisoned".into()))?;
        if guard.is_none() {
            return Err(Error::InvalidOp);
        }
        let mut out_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut out_size: i64 = 0;
        // SAFETY: out_ptr and out_size are valid stack-allocated output parameters.
        // MPI_Buffer_detach writes the buffer pointer and its size into them.
        let ret = unsafe { ffi::ferrompi_buffer_detach(&mut out_ptr, &mut out_size) };
        if ret != 0 {
            return Err(Error::from_code_with_op(ret, "buffer_detach"));
        }
        // Reclaim the Box from the static; this is the same allocation MPI just
        // released. We take() here so the static is cleared atomically.
        let buf = guard.take().expect("guard was Some; take() must succeed");
        Ok(buf)
    }
}

impl Drop for Mpi {
    fn drop(&mut self) {
        // rt::finalize() moves the lifecycle state from Active to Finalized
        // before ferrompi_finalize runs, so a handle whose drop is nested
        // inside the finalize sweep (a closure captured by a request/op that
        // the sweep drops) observes Finalized and makes no MPI call. It also
        // returns false for a stub Mpi built without init (Uninit) and for a
        // second call on an already-finalized state, so this branch runs
        // ferrompi_finalize at most once.
        if rt::finalize() {
            #[cfg(feature = "rma")]
            {
                let live = window::live_windows();
                if live > 0 {
                    let _ = writeln!(
                        std::io::stderr(),
                        "ferrompi: MPI_Finalize skipped: {live} window(s) still alive; their memory stays valid until the process exits"
                    );
                    return;
                }
            }
            // SAFETY: ferrompi_finalize takes no arguments. rt::finalize()
            // just returned true, so state was Active — MPI_Init(_thread)
            // succeeded and MPI_Finalize has not yet been called for this
            // process.
            unsafe {
                ffi::ferrompi_finalize();
            }
        }
    }
}

// Mpi is not Send or Sync - MPI must be used from the thread that initialized it
// (unless thread level is Multiple)
// This is enforced by PhantomData<*const ()> in the struct

#[cfg(test)]
mod tests {
    // Note: MPI tests must be run with mpiexec
    // cargo build --examples && mpiexec -n 4 ./target/debug/examples/hello_world

    use super::{group, Error, Mpi, ReduceOp, ThreadLevel, ATTACHED_BUFFER};
    use std::marker::PhantomData;

    /// Minimal stub `Mpi` handle for tests that never call `Mpi::init`
    /// because their early-return path fires before any MPI call.
    fn stub_mpi() -> Mpi {
        Mpi {
            thread_level: ThreadLevel::Single,
            _marker: PhantomData,
        }
    }

    // ── ThreadLevel tests ──────────────────────────────────────────────

    #[test]
    fn thread_level_repr_values() {
        assert_eq!(ThreadLevel::Single as i32, 0);
        assert_eq!(ThreadLevel::Funneled as i32, 1);
        assert_eq!(ThreadLevel::Serialized as i32, 2);
        assert_eq!(ThreadLevel::Multiple as i32, 3);
    }

    // ── ReduceOp tests ─────────────────────────────────────────────────

    #[test]
    fn reduce_op_repr_values() {
        let ops = [
            (ReduceOp::Sum, 0),
            (ReduceOp::Max, 1),
            (ReduceOp::Min, 2),
            (ReduceOp::Prod, 3),
            (ReduceOp::BitwiseOr, 4),
            (ReduceOp::BitwiseAnd, 5),
            (ReduceOp::BitwiseXor, 6),
            (ReduceOp::LogicalOr, 7),
            (ReduceOp::LogicalAnd, 8),
            (ReduceOp::LogicalXor, 9),
            (ReduceOp::MaxLoc, 10),
            (ReduceOp::MinLoc, 11),
        ];
        for (op, expected) in ops {
            assert_eq!(op as i32, expected);
        }
        #[cfg(feature = "rma")]
        {
            assert_eq!(ReduceOp::Replace as i32, 12);
            assert_eq!(ReduceOp::NoOp as i32, 13);
        }
    }

    // ── Mpi::buffer_attach / buffer_detach unit tests ─────────────────────

    /// Both guard paths of the attached-buffer static in one test, so no other
    /// test can interleave with the shared state: detach with nothing attached
    /// and attach while a buffer is attached both return `Err(InvalidOp)`
    /// without reaching MPI (the static is seeded directly).
    #[test]
    fn buffer_attach_detach_guards_return_invalid_op() {
        let mpi = stub_mpi();

        *ATTACHED_BUFFER.lock().unwrap() = None;
        let result = mpi.buffer_detach();
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "expected Err(InvalidOp) on detach without attach, got: {result:?}"
        );

        *ATTACHED_BUFFER.lock().unwrap() = Some(vec![0u8; 4].into_boxed_slice());
        let result = mpi.buffer_attach(vec![0u8; 8].into_boxed_slice());
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "expected Err(InvalidOp) on double attach, got: {result:?}"
        );

        ATTACHED_BUFFER.lock().unwrap().take();
    }

    // ── Mpi::create_from_group unit tests ─────────────────────────────────

    /// Verify that a `stringtag` containing a null byte is rejected before
    /// the FFI call is ever invoked.  We test the null-byte path directly
    /// by calling the public method on a `Group` stub with handle 0
    /// (MPI_GROUP_EMPTY); the early-return on bad tag means MPI is never
    /// touched, so no running MPI environment is needed.
    #[test]
    fn create_from_group_null_byte_in_tag() {
        // Construct a minimal stub Mpi to call the method (no MPI calls made
        // because the null-byte check fires before the version probe or FFI).
        // We bypass init by constructing the struct directly — this is valid
        // inside the crate's own test module where the fields are accessible.
        let mpi = stub_mpi();
        // Group with handle 0 (MPI_GROUP_EMPTY sentinel) — never dereferenced
        // because the null-byte check fires first.
        let g = group::Group { handle: 0 };
        let result = mpi.create_from_group(&g, "bad\0tag");
        match result {
            Err(Error::Internal(msg)) => {
                assert!(
                    msg.contains("null byte"),
                    "expected 'null byte' in error message, got: {msg}"
                );
            }
            Ok(_) => panic!("expected Err(Error::Internal(_)), got Ok(_)"),
            Err(e) => panic!("expected Err(Error::Internal(_)), got Err({e})"),
        }
    }
}
