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
// NOTE: The C-layer handle tables (comm_table, request_table, etc.) are not
// internally synchronized. At ThreadLevel::Single and Funneled, only the main
// thread calls MPI so no data race occurs. At Serialized, the user is required
// to serialize all MPI calls. At Multiple, concurrent MPI calls are safe per
// the MPI standard, but our handle table allocation/deallocation is not
// protected by mutexes. This is acceptable because MPI_THREAD_MULTIPLE only
// guarantees that MPI calls are thread-safe — the handle tables are modified
// only inside MPI entry/exit points which the user serializes or the MPI
// implementation serializes internally.
unsafe impl Send for Communicator {}
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
            unsafe { ffi::ferrompi_comm_free(self.handle) };
        }
    }
}

// Compile-time assertions: Communicator must be Send + Sync
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<Communicator>();
    }
};

#[cfg(test)]
mod tests {
    use crate::comm::{Communicator, SplitType};

    /// Helper: create a Communicator with handle 0 (COMM_WORLD).
    /// Drop for handle 0 is a no-op, so this is safe without MPI.
    /// `size: 1` prevents divide-by-zero in tests that do `buf.len() / comm.size()`.
    fn dummy_comm() -> Communicator {
        Communicator {
            handle: 0,
            rank: 0,
            size: 1,
        }
    }

    #[test]
    fn cached_rank_size_returns_field_values() {
        // Construct directly with arbitrary rank/size values to prove that
        // rank() and size() return the cached fields without any FFI call.
        let comm = Communicator {
            handle: 0,
            rank: 7,
            size: 42,
        };
        assert_eq!(comm.rank(), 7);
        assert_eq!(comm.size(), 42);
    }

    #[test]
    fn split_type_shared_repr_value_and_traits() {
        // SplitType::Shared has repr value 0
        assert_eq!(SplitType::Shared as i32, 0);

        // Clone works (Copy implies Clone)
        let st = SplitType::Shared;
        let cloned = st;
        assert_eq!(cloned, SplitType::Shared);

        // Debug works
        assert_eq!(format!("{:?}", st), "Shared");
    }

    #[test]
    fn communicator_undefined_is_negative_one() {
        assert_eq!(Communicator::UNDEFINED, -1);
    }

    #[test]
    fn communicator_raw_handle_returns_correct_value() {
        let comm = dummy_comm();
        assert_eq!(comm.raw_handle(), 0);
    }
}
