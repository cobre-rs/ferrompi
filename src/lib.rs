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
//! - Large count support (MPI 4.0+ `_c` variants)
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
//! - **Shared memory windows** (feature `rma`): [`SharedWindow<T>`] with RAII lock guards
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

#![warn(missing_docs)]
#![warn(clippy::all)]
// Allow certain pedantic lints for existing code
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]

use std::ffi::c_char;

mod comm;
mod datatype;
mod error;
mod ffi;
mod info;
mod persistent;
mod request;
#[cfg(feature = "numa")]
pub mod slurm;
mod status;
mod topology;
#[cfg(feature = "rma")]
mod window;

pub use comm::{Communicator, SplitType};
pub use datatype::{DatatypeTag, MpiDatatype};
pub use error::{Error, MpiErrorClass, Result};
pub use info::Info;
pub use persistent::PersistentRequest;
pub use request::Request;
pub use status::Status;
#[cfg(feature = "numa")]
pub use topology::SlurmInfo;
pub use topology::{HostEntry, TopologyInfo};
#[cfg(feature = "rma")]
pub use window::{LockAllGuard, LockGuard, LockType, SharedWindow};

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag tracking whether MPI has been initialized
static MPI_INITIALIZED: AtomicBool = AtomicBool::new(false);

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
}

/// MPI environment handle.
///
/// This type represents an initialized MPI environment. There can only be one
/// instance of this type at a time. When dropped, it finalizes MPI.
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
    /// Returns an error if MPI is already initialized or if initialization fails.
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
    /// Returns an error if MPI is already initialized or if initialization fails.
    pub fn init_thread(required: ThreadLevel) -> Result<Self> {
        // Check if already initialized
        if MPI_INITIALIZED.swap(true, Ordering::SeqCst) {
            return Err(Error::AlreadyInitialized);
        }

        let mut provided: i32 = 0;
        let ret = unsafe { ffi::ferrompi_init_thread(required as i32, &mut provided) };

        if ret != 0 {
            MPI_INITIALIZED.store(false, Ordering::SeqCst);
            // Cannot call Error::from_code here because MPI runtime is not
            // initialized — MPI_Error_class/MPI_Error_string would be UB.
            return Err(Error::Mpi {
                class: MpiErrorClass::Raw(ret),
                code: ret,
                message: format!("MPI_Init_thread failed with code {ret}"),
            });
        }

        let thread_level = match provided {
            0 => ThreadLevel::Single,
            1 => ThreadLevel::Funneled,
            2 => ThreadLevel::Serialized,
            _ => ThreadLevel::Multiple,
        };

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
        let ret = unsafe {
            ffi::ferrompi_get_library_version(buf.as_mut_ptr().cast::<c_char>(), &mut len)
        };
        Error::check(ret)?;
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
        let ret = unsafe { ffi::ferrompi_get_version(buf.as_mut_ptr().cast::<c_char>(), &mut len) };

        if ret != 0 {
            return Err(Error::from_code(ret));
        }

        let len = (len.max(0) as usize).min(buf.len());
        let s = std::str::from_utf8(&buf[..len])
            .map_err(|_| Error::Internal("Invalid UTF-8 in version string".into()))?;
        Ok(s.to_string())
    }

    /// Check if MPI has been initialized.
    pub fn is_initialized() -> bool {
        let mut flag: i32 = 0;
        unsafe { ffi::ferrompi_initialized(&mut flag) };
        flag != 0
    }

    /// Check if MPI has been finalized.
    pub fn is_finalized() -> bool {
        let mut flag: i32 = 0;
        unsafe { ffi::ferrompi_finalized(&mut flag) };
        flag != 0
    }
}

impl Drop for Mpi {
    fn drop(&mut self) {
        // Only finalize if we successfully initialized
        if MPI_INITIALIZED.load(Ordering::SeqCst) {
            unsafe {
                ffi::ferrompi_finalize();
            }
            MPI_INITIALIZED.store(false, Ordering::SeqCst);
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

    use super::*;

    // ── ThreadLevel tests ──────────────────────────────────────────────

    #[test]
    fn thread_level_ordering() {
        assert!(ThreadLevel::Single < ThreadLevel::Funneled);
        assert!(ThreadLevel::Funneled < ThreadLevel::Serialized);
        assert!(ThreadLevel::Serialized < ThreadLevel::Multiple);
        // Transitive: smallest < largest
        assert!(ThreadLevel::Single < ThreadLevel::Multiple);
    }

    #[test]
    fn thread_level_equality() {
        assert_eq!(ThreadLevel::Single, ThreadLevel::Single);
        assert_eq!(ThreadLevel::Funneled, ThreadLevel::Funneled);
        assert_eq!(ThreadLevel::Serialized, ThreadLevel::Serialized);
        assert_eq!(ThreadLevel::Multiple, ThreadLevel::Multiple);
        assert_ne!(ThreadLevel::Single, ThreadLevel::Multiple);
        assert_ne!(ThreadLevel::Funneled, ThreadLevel::Serialized);
    }

    #[test]
    fn thread_level_repr_values() {
        assert_eq!(ThreadLevel::Single as i32, 0);
        assert_eq!(ThreadLevel::Funneled as i32, 1);
        assert_eq!(ThreadLevel::Serialized as i32, 2);
        assert_eq!(ThreadLevel::Multiple as i32, 3);
    }

    #[test]
    fn thread_level_debug_clone() {
        let level = ThreadLevel::Funneled;
        let cloned = level;
        assert_eq!(format!("{cloned:?}"), "Funneled");

        assert_eq!(format!("{:?}", ThreadLevel::Single), "Single");
        assert_eq!(format!("{:?}", ThreadLevel::Serialized), "Serialized");
        assert_eq!(format!("{:?}", ThreadLevel::Multiple), "Multiple");
    }

    // ── ReduceOp tests ─────────────────────────────────────────────────

    #[test]
    fn reduce_op_repr_values() {
        assert_eq!(ReduceOp::Sum as i32, 0);
        assert_eq!(ReduceOp::Max as i32, 1);
        assert_eq!(ReduceOp::Min as i32, 2);
        assert_eq!(ReduceOp::Prod as i32, 3);
    }

    #[test]
    fn reduce_op_equality() {
        assert_eq!(ReduceOp::Sum, ReduceOp::Sum);
        assert_eq!(ReduceOp::Max, ReduceOp::Max);
        assert_eq!(ReduceOp::Min, ReduceOp::Min);
        assert_eq!(ReduceOp::Prod, ReduceOp::Prod);
        assert_ne!(ReduceOp::Sum, ReduceOp::Max);
        assert_ne!(ReduceOp::Min, ReduceOp::Prod);
        assert_ne!(ReduceOp::Sum, ReduceOp::Prod);
    }

    #[test]
    fn reduce_op_debug_clone() {
        let op = ReduceOp::Sum;
        let cloned = op;
        assert_eq!(format!("{cloned:?}"), "Sum");

        assert_eq!(format!("{:?}", ReduceOp::Max), "Max");
        assert_eq!(format!("{:?}", ReduceOp::Min), "Min");
        assert_eq!(format!("{:?}", ReduceOp::Prod), "Prod");
    }
}
