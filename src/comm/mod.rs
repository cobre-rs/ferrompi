//! Safe wrappers for MPI communicator operations.
//!
//! All communication methods are generic over [`MpiDatatype`], supporting
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.

use crate::error::{Error, Result};
use crate::ffi;

mod blocking;
mod mgmt;
mod nonblocking;
mod p2p;
mod persistent;
mod v_collective;

/// Split types for [`Communicator::split_type`].
///
/// These constants map to MPI communicator split type values. Currently only
/// shared-memory splits are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SplitType {
    /// Split by shared memory domain (same physical node).
    /// Maps to `MPI_COMM_TYPE_SHARED`.
    Shared = 0,
}

/// An MPI communicator.
///
/// This type wraps an MPI communicator handle and provides safe methods for
/// collective and point-to-point communication operations.
///
/// # Thread Safety
///
/// `Communicator` is `Send + Sync`, matching the thread-safety model of
/// C/Fortran MPI. The actual thread-safety guarantees depend on the
/// thread level provided by [`Mpi::init_thread()`](crate::Mpi::init_thread):
///
/// - [`ThreadLevel::Single`](crate::ThreadLevel::Single) /
///   [`Funneled`](crate::ThreadLevel::Funneled): MPI calls only from main thread
/// - [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized): MPI calls
///   from any thread, but serialized by the user (e.g., via a `Mutex`)
/// - [`ThreadLevel::Multiple`](crate::ThreadLevel::Multiple): MPI calls from
///   any thread concurrently without external synchronization
///
/// For hybrid MPI + threads programs, request at least
/// [`ThreadLevel::Funneled`](crate::ThreadLevel::Funneled) (master thread
/// makes MPI calls) or [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized)
/// (any thread, but only one at a time).
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// println!("I am rank {} of {}", world.rank(), world.size());
/// ```
pub struct Communicator {
    pub(crate) handle: i32,
    pub(crate) rank: i32,
    pub(crate) size: i32,
}

// SAFETY: Communicator handles are integer indices into a C-side table.
// The C MPI library manages its own thread safety based on the thread level
// requested via MPI_Init_thread. Sending a Communicator to another thread is
// safe because MPI_Comm operations are defined to be callable from any thread
// when the appropriate thread level (Serialized or Multiple) was requested.
// Users must ensure they requested sufficient thread support and serialize
// access themselves when using ThreadLevel::Serialized.
//
// All seven C-layer handle tables (comm_table, request_table, win_table,
// info_table, group_table, datatype_table, op_table) use the C11 atomic-CAS
// pattern, eliminating data races under MPI_THREAD_MULTIPLE.
// See docs/adr/0002-handle-tables.md for the full rationale and design.
unsafe impl Send for Communicator {}
// SAFETY: &Communicator exposes only reads of immutable fields and FFI calls
// whose concurrent use MPI governs by the initialized thread level, and this
// type does not check that level.
unsafe impl Sync for Communicator {}

impl Communicator {
    /// Constant for opting out of a communicator split.
    ///
    /// Processes passing this as the `color` to [`split()`](Self::split) will not be
    /// included in any resulting communicator.
    pub const UNDEFINED: i32 = -1;

    /// Get a handle to `MPI_COMM_WORLD`.
    pub(crate) fn world() -> Self {
        // SAFETY: ferrompi_comm_world() returns the well-known COMM_WORLD handle.
        // MPI must be initialized before this is called (enforced by Mpi::world()).
        let handle = unsafe { ffi::ferrompi_comm_world() };
        Self::from_handle(handle).expect("COMM_WORLD must be valid post-init")
    }

    /// Construct a `Communicator` by querying rank and size once from MPI.
    ///
    /// This is the canonical internal constructor. All construction sites use it
    /// so that `rank` and `size` are cached at creation time and subsequent calls
    /// to [`rank()`](Self::rank) / [`size()`](Self::size) require no FFI round-trip.
    ///
    /// Returns `Err` if either `MPI_Comm_rank` or `MPI_Comm_size` fails.
    pub(crate) fn from_handle(handle: i32) -> Result<Self> {
        let mut rank: i32 = 0;
        // SAFETY: `handle` is a valid MPI communicator handle obtained from MPI
        // functions. `rank` is a local variable so the pointer is valid for writes.
        let ret = unsafe { ffi::ferrompi_comm_rank(handle, &mut rank) };
        Error::check_with_op(ret, "comm_rank")?;
        let mut size: i32 = 0;
        // SAFETY: same as above — `size` is a local variable, `handle` is valid.
        let ret = unsafe { ffi::ferrompi_comm_size(handle, &mut size) };
        Error::check_with_op(ret, "comm_size")?;
        Ok(Communicator { handle, rank, size })
    }

    /// Get the raw communicator handle (for advanced use).
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }

    /// Get the rank of the calling process in this communicator.
    ///
    /// Returns the cached value stored at construction time. No FFI call is made.
    #[inline]
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get the number of processes in this communicator.
    ///
    /// Returns the cached value stored at construction time. No FFI call is made.
    #[inline]
    pub fn size(&self) -> i32 {
        self.size
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        // Don't free COMM_WORLD (handle 0)
        if self.handle != 0 {
            // SAFETY: self.handle is a valid, non-zero communicator handle
            // registered in the C-side comm table (checked above); Drop takes
            // &mut self and runs at most once per value, so this cannot
            // double-free the handle.
            unsafe { ffi::ferrompi_comm_free(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::SplitType;

    #[test]
    fn split_type_repr_value() {
        // SplitType::Shared has repr value 0
        assert_eq!(SplitType::Shared as i32, 0);
    }
}
