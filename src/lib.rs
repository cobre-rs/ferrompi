//! # ferrompi
//!
//! Safe, generic Rust bindings for MPI (Message Passing Interface).
//!
//! This crate wraps MPI functionality through a thin C layer.
//!
//! ## Capabilities
//!
//! - **Generic API**: All operations work with any [`MpiDatatype`] (`f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`)
//! - **Blocking collectives**: barrier, broadcast, reduce, allreduce, gather, scatter, allgather,
//!   alltoall, scan, exscan, reduce\_scatter\_block, plus V-variants (gatherv, scatterv, allgatherv, alltoallv)
//! - **Nonblocking collectives**: All 15 `i`-prefixed variants with [`Request`] handles
//! - **Persistent collectives** (MPI 4.0, or Open MPI 5): `*_init` forms of every blocking collective except
//!   `barrier`, plus five in-place forms, with [`PersistentRequest`] handles
//! - **Large counts**: with an MPI 4.0 library, a count above `i32::MAX` uses MPI's `_c` call
//!   (point-to-point, collectives including persistent ones, RMA); below
//!   MPI 4.0 it returns [`Error::Mpi`] with class [`MpiErrorClass::Count`]. The V-collectives
//!   (`gatherv`, `scatterv`, `allgatherv`, `alltoallv`, and their persistent forms), which take
//!   counts as `i32` arrays, and `allreduce_with_op`, whose user function takes an `int` length,
//!   return that error on every MPI version.
//! - **Scalar and in-place variants**: `reduce_scalar`, `allreduce_scalar`, `reduce_inplace`,
//!   `allreduce_inplace`, `scan_scalar`, `exscan_scalar`
//! - **Point-to-point**: `send`, `recv`, `isend`, `irecv`, `sendrecv`, `probe`, `iprobe`
//! - **Persistent point-to-point**: `send_init`, `bsend_init`, `rsend_init`, `ssend_init`,
//!   `recv_init` methods on [`Communicator`], each returning a [`PersistentRequest`]
//! - **Communicator management**: `split`, `split_type`, `split_shared`, `duplicate`
//! - **Group operations**: [`Group`] with incl/excl/union/intersection/difference,
//!   [`RankRange`] for range constructors, [`GroupComparison`].
//!
//!   Note: [`Mpi::create_from_group`] needs MPI 4.0, or Open MPI 5.
//! - **Custom datatypes**: [`CustomDatatype`]
//!   (contiguous/vector/struct/resized) and [`StructField`]
//!   for struct-type builders.
//! - **User-defined reduction operations**: [`UserOp`] wraps `MPI_Op_create`
//!   with safe closure storage and trampoline.
//! - **Distributed RMA windows** (feature `rma`): `Win<T>` with
//!   `WinFenceAssert`, `WinPscwAssert`,
//!   `WinLockGuard`, and `WinLockAllGuard`
//!   RAII guards.
//! - **Shared memory windows** (feature `rma`): `SharedWindow<T>`
//!   with RAII lock guards for intra-node shared memory (distinct from the
//!   distributed `Win<T>` windows above).
//! - **Info objects**: [`Info`] creates and queries MPI info objects;
//!   no ferrompi call takes one yet.
//! - **SLURM helpers** (feature `numa`): Job topology queries via `slurm` module
//! - **Rich error handling**: [`MpiErrorClass`] categorization with messages from the MPI runtime
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
//! | `rma`   | RMA and shared-memory windows (`Win`, `SharedWindow`, lock guards, RMA operations, `ReduceOp::Replace`/`NoOp`) | — |
//! | `numa`  | The `slurm` module and `SlurmInfo` | `rma` |
//!
//! `numa` needs no system library: it implies `rma` and adds only the SLURM
//! job-topology helpers; it adds no NUMA code.
//!
//! ## Thread Safety
//!
//! [`Communicator`], [`Group`], [`Request`], [`PersistentRequest`], [`Status`],
//! [`CustomDatatype`], [`Info`], and [`UserOp`] are `Send + Sync`; [`Mpi`],
//! `Win<T>`, `SharedWindow<T>` (feature `rma`), and their RAII lock guards are not.
//!
//! Which thread may call MPI is set by the provided [`ThreadLevel`]; see its
//! documentation for the per-level rules.
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
//! // Request funneled support for hybrid MPI + threads
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
//! export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
//! srun --cpu-bind=cores ./my_program
//! ```
//!
//! Use the `slurm` module (with `numa` feature) to read these values at runtime.
//! See `examples/hybrid_openmp.rs` for the full pattern.
//!
//! ## Lifecycle
//!
//! Dropping the [`Mpi`] handle finalizes MPI. See [`Mpi`] for what happens to
//! other handles afterwards, and [`Error`] for which failures return `Err`
//! and which abort the process.
//!
//! ## Extended documentation
//!
//! The [`doc`] module embeds the long-form guides and the architecture
//! decision records.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
// Clippy suppressions live at the call site (`#[allow(clippy::NAME)]`
// with a justification comment) rather than crate-wide.

use std::ffi::{c_char, CString};
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
    BytePermutable, DatatypeTag, DoubleInt, FloatInt, Int2, MpiDatatype, MpiIndexedDatatype,
    PlainData, ShortInt,
};
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
pub use datatype::{LongDoubleInt, LongInt};
pub use datatype_builder::{CustomDatatype, StructField};
pub use error::{Error, MpiErrorClass, ResourceKind, Result};
pub use group::{Group, GroupComparison, RankRange};
pub use info::Info;
pub use op::{ReduceOp, UserOp};
pub use persistent::PersistentRequest;
pub use request::Request;
pub use status::Status;
#[cfg(feature = "numa")]
pub use topology::SlurmInfo;
pub use topology::{HostEntry, TopologyInfo};
#[cfg(feature = "rma")]
pub use window::{
    LockAllGuard, LockGuard, LockType, PendingFetchResult, SharedWindow, Win, WinFenceAssert,
    WinLockAllGuard, WinLockGuard, WinPscwAssert,
};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

use std::marker::PhantomData;
use std::sync::Mutex;

/// Process-wide buffer attached for buffered sends (`MPI_Buffer_attach`).
///
/// MPI allows at most one attached buffer per process. This static holds the
/// `Box<[u8]>` so that the allocation remains valid for the duration of the
/// attachment. The `Mutex` is held only briefly during `buffer_attach` and
/// `buffer_detach` transitions; MPI itself manages the buffer between those
/// two calls.
static ATTACHED_BUFFER: Mutex<Option<Box<[u8]>>> = Mutex::new(None);

/// MPI thread support levels.
///
/// | Level | Who may call MPI | Synchronization |
/// |-------|-------------------|------------------|
/// | [`Single`](ThreadLevel::Single) | Only the thread that called [`Mpi::init_thread`] | N/A |
/// | [`Funneled`](ThreadLevel::Funneled) | Only the thread that called [`Mpi::init_thread`] | N/A |
/// | [`Serialized`](ThreadLevel::Serialized) | Any thread | Caller serializes; debug builds detect an overlap |
/// | [`Multiple`](ThreadLevel::Multiple) | Any thread | None needed |
///
/// A call from any other thread at [`Single`](ThreadLevel::Single) or
/// [`Funneled`](ThreadLevel::Funneled) returns
/// `Err(`[`Error::ThreadLevelViolation`]`)` without calling MPI. At
/// [`Serialized`](ThreadLevel::Serialized), the caller must serialize its own
/// calls; a debug build that detects two calls overlapping also returns that
/// error (a consuming [`Request::wait`](crate::Request::wait) aborts the
/// process instead), but a release build does not check.
///
/// [`Mpi::thread_level()`] reports the level MPI actually granted, which may
/// be lower than the level requested to [`Mpi::init_thread`]; it is the
/// granted level, not the requested one, that these rules apply to.
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

/// MPI environment handle.
///
/// # Lifecycle
///
/// There is one `Mpi` per process. While a handle is alive,
/// [`Mpi::init`]/[`Mpi::init_thread`] return
/// `Err(`[`Error::AlreadyInitialized`]`)`; once it is dropped, they return
/// `Err(`[`Error::Finalized`]`)`. MPI cannot be re-initialized.
///
/// Dropping this handle finalizes MPI. Afterward, every method that calls
/// MPI through a communicator, group, request, window, datatype, info or
/// user op returns `Err(`[`Error::Finalized`]`)` without calling MPI. The
/// queries the MPI standard allows at any time ([`Mpi::version`],
/// [`Mpi::library_version`], [`Mpi::is_initialized`],
/// [`Mpi::is_finalized`]) still answer. A communicator, request, window or
/// window lock guard, datatype, group, info object or user op that outlives
/// this handle makes no MPI call when it is dropped.
///
/// If any window (feature `rma`) is still alive when this handle is
/// dropped — whatever constructed it — `MPI_Finalize` is skipped instead,
/// with a stderr warning, because some MPI implementations free
/// MPI-allocated window memory inside `MPI_Finalize` itself, and some abort
/// while tearing down internal state that still tracks a live window;
/// window memory stays valid until the process exits. Launchers can report
/// such an exit differently from a clean one; see
/// [`doc::mpi_compatibility`]. Debug builds print a note to stderr when
/// finalize still leaves MPI requests unfreed.
///
/// At [`ThreadLevel::Single`]/[`ThreadLevel::Funneled`], dropping a handle
/// whose `Drop` calls MPI — an uncompleted [`Request`], a
/// [`PersistentRequest`], a communicator other than the world, a window or
/// window lock guard, a datatype, group, info object or user op — on a
/// thread other than the one that called [`Mpi::init`]/[`Mpi::init_thread`]
/// prints
/// `ferrompi: <Type> dropped on thread <name or id>` to stderr and aborts
/// the process.
///
/// At [`ThreadLevel::Serialized`]/[`ThreadLevel::Multiple`], dropping this
/// handle while another thread is still inside an MPI call through this
/// crate is a program error that `ferrompi` does not detect.
///
/// A call that would fail one of the checks above (finalized, wrong
/// thread) and is also given invalid arguments may return the argument
/// error (for example [`Error::BufferSize`]) instead of
/// `Err(`[`Error::Finalized`]`)`/`Err(`[`Error::ThreadLevelViolation`]`)`;
/// the order in which these checks run is not part of the API.
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
    /// `MPI_Init_thread` itself fails. An error inside `MPI_Init_thread`
    /// itself usually aborts the process before `Err(`[`Error::Mpi`]`)` can
    /// be returned; see [`Error`].
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
    /// `MPI_Init_thread` itself fails. An error inside `MPI_Init_thread`
    /// itself usually aborts the process before `Err(`[`Error::Mpi`]`)` can
    /// be returned; see [`Error`]. That error's class is
    /// [`MpiErrorClass::Unknown`], because MPI cannot be asked for the class
    /// of an error from its own initialization.
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
                class: MpiErrorClass::Unknown,
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

    /// Get the current wall-clock time (`MPI_Wtime`), in seconds.
    ///
    /// This is a high-resolution timer suitable for benchmarking. It takes
    /// `&self` because MPI must be initialized and not yet finalized; code
    /// without the handle can use [`std::time::Instant`].
    pub fn wtime(&self) -> f64 {
        // SAFETY: ferrompi_wtime takes no pointer arguments and touches no
        // Rust-owned memory; `&self` proves MPI is initialized and not
        // finalized, and `Mpi` is neither `Send` nor `Sync`, so this runs on
        // the thread that initialized MPI.
        unsafe { ffi::ferrompi_wtime() }
    }

    /// Get the MPI library version string (implementation-specific).
    ///
    /// Returns a string such as `"Open MPI v4.1.6"` or `"Intel(R) MPI Library 2021.7"`.
    /// This wraps `MPI_Get_library_version`.
    /// The string ends before the first NUL character, with trailing whitespace removed.
    pub fn library_version() -> Result<String> {
        let mut buf = [0u8; 8192];
        let mut len: i32 = 0;
        // SAFETY: buf is a local 8192-byte buffer; MPI writes at most
        // MPI_MAX_LIBRARY_VERSION_STRING bytes, which the C layer asserts at build time
        // is at most 8192, and reports the length through the local `len`.
        let ret = unsafe {
            ffi::ferrompi_get_library_version(buf.as_mut_ptr().cast::<c_char>(), &mut len)
        };
        Error::check_with_op(ret, "get_library_version")?;
        let len = (len.max(0) as usize).min(buf.len());
        // Open MPI counts the terminating NUL in `len`; some implementations
        // append trailing whitespace.
        let len = buf[..len].iter().position(|&b| b == 0).unwrap_or(len);
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

    /// Create a communicator from a group without requiring a parent
    /// communicator (MPI 4.0, or Open MPI 5).
    ///
    /// `stringtag` must be identical across all ranks that participate
    /// in the call; ranks with different tags or in different groups
    /// produce separate communicators.
    ///
    /// # Errors
    ///
    /// - Returns `Err(`[`Error::InvalidArgument`]`)` if `stringtag` contains a
    ///   NUL byte (the FFI call is never invoked in this case).
    /// - Returns `Err(Error::NotSupported(_))` when ferrompi was built
    ///   against an MPI older than 4.0 other than Open MPI 5.
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
        let c_tag = CString::new(stringtag).map_err(|_| Error::InvalidArgument {
            arg: "stringtag",
            reason: "contains a NUL byte",
        })?;
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
    /// `Err(`[`Error::InvalidState`]`)`.
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
    /// `Err(`[`Error::InvalidArgument`]`)` before the FFI call; the
    /// underlying `MPI_Buffer_attach` takes an `int` count and cannot
    /// address larger buffers.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `buffer.len() > i32::MAX as usize`.
    /// - [`Error::InvalidState`] if a buffer is already attached.
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
            return Err(Error::InvalidState {
                reason: "a buffer is already attached",
            });
        }
        if buffer.len() > i32::MAX as usize {
            return Err(Error::InvalidArgument {
                arg: "buffer",
                reason: "longer than i32::MAX bytes",
            });
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
    /// - [`Error::InvalidState`] if no buffer is currently attached.
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
            return Err(Error::InvalidState {
                reason: "no buffer is attached",
            });
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
        Ok(guard.take().expect("guard was Some; take() must succeed"))
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
                    // A single write so concurrent ranks' output cannot interleave mid-line.
                    let msg = format!(
                        "ferrompi: MPI_Finalize skipped: {live} window(s) still alive; their memory stays valid until the process exits\n"
                    );
                    let _ = std::io::stderr().write_all(msg.as_bytes());
                    return;
                }
            }
            let mut active: i32 = 0;
            // SAFETY: `active` is a valid local out-parameter that
            // ferrompi_finalize writes the unfreed-active-request count
            // into. rt::finalize() just returned true, so state was Active —
            // MPI_Init(_thread) succeeded and MPI_Finalize has not yet been
            // called for this process.
            unsafe {
                ffi::ferrompi_finalize(&mut active);
            }
            if cfg!(debug_assertions) && active > 0 {
                // A single write so concurrent ranks' output cannot interleave mid-line.
                let msg =
                    format!("ferrompi: MPI_Finalize leaves {active} active request(s) unfreed\n");
                let _ = std::io::stderr().write_all(msg.as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: MPI tests must be run with mpiexec
    // cargo build --examples && mpiexec -n 4 ./target/debug/examples/hello_world

    use super::{group, Error, Mpi, ThreadLevel, ATTACHED_BUFFER};
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

    // ── Mpi::buffer_attach / buffer_detach unit tests ─────────────────────

    /// Both guard paths of the attached-buffer static in one test, so no other
    /// test can interleave with the shared state: detach with nothing attached
    /// and attach while a buffer is attached both return `Err(InvalidState)`
    /// without reaching MPI (the static is seeded directly).
    #[test]
    fn buffer_attach_detach_guards_return_invalid_state() {
        let mpi = stub_mpi();

        *ATTACHED_BUFFER.lock().unwrap() = None;
        let result = mpi.buffer_detach();
        assert!(
            matches!(
                result,
                Err(Error::InvalidState {
                    reason: "no buffer is attached"
                })
            ),
            "expected Err(InvalidState) on detach without attach, got: {result:?}"
        );

        *ATTACHED_BUFFER.lock().unwrap() = Some(vec![0u8; 4].into_boxed_slice());
        let result = mpi.buffer_attach(vec![0u8; 8].into_boxed_slice());
        assert!(
            matches!(
                result,
                Err(Error::InvalidState {
                    reason: "a buffer is already attached"
                })
            ),
            "expected Err(InvalidState) on double attach, got: {result:?}"
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
        let mpi = stub_mpi();
        // Group with handle 0 (MPI_GROUP_EMPTY sentinel) — never dereferenced
        // because the null-byte check fires first.
        let g = group::Group { handle: 0 };
        let result = mpi.create_from_group(&g, "bad\0tag");
        assert!(
            matches!(
                result,
                Err(Error::InvalidArgument {
                    arg: "stringtag",
                    reason: "contains a NUL byte"
                })
            ),
            "expected Err(InvalidArgument) for a NUL byte in the tag"
        );
    }
}
