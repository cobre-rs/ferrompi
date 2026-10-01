//! Safe wrappers for MPI communicator operations.
//!
//! All communication methods are generic over [`MpiDatatype`], supporting
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.

use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::rt;
use std::ffi::c_void;

mod blocking;
mod mgmt;
mod nonblocking;
mod p2p;
mod persistent;
mod v_collective;

/// Checks that a buffer of `whole` elements holds one `block`-element slot per
/// rank of a `size`-rank communicator (`whole >= block * size`, with the product
/// computed by checked arithmetic).
fn check_rank_slots(whole: usize, block: usize, size: i32) -> Result<()> {
    let needed = block
        .checked_mul(size as usize)
        .ok_or(Error::InvalidBuffer)?;
    if whole < needed {
        return Err(Error::InvalidBuffer);
    }
    Ok(())
}

/// Checks that two buffers hold the same number of elements.
fn check_same_len(a: usize, b: usize) -> Result<()> {
    if a != b {
        return Err(Error::InvalidBuffer);
    }
    Ok(())
}

/// Returns the per-rank block count of a buffer of `whole` elements split
/// evenly across a `size`-rank communicator; Err when `size <= 0` or
/// `whole % size != 0`.
fn rank_block(whole: usize, size: i32) -> Result<usize> {
    if size <= 0 {
        return Err(Error::InvalidBuffer);
    }
    let size = size as usize;
    if whole % size != 0 {
        return Err(Error::InvalidBuffer);
    }
    Ok(whole / size)
}

/// Arguments of an in-place scatter as `(sendbuf, sendcount, recvbuf, recvcount, tag)`:
/// at root, `data` is the send buffer split in `size` blocks and the NULL
/// receive buffer is the in-place marker; elsewhere, `data` is the receive
/// buffer and the send side is NULL and ignored.
fn scatter_inplace_args<T: MpiDatatype>(
    data: &mut [T],
    is_root: bool,
    size: i32,
) -> Result<(*const c_void, i64, *mut c_void, i64, i32)> {
    if is_root {
        let per = rank_block(data.len(), size)? as i64;
        let (sp, _, dt) = buf(data);
        Ok((sp, per, std::ptr::null_mut::<std::ffi::c_void>(), 0i64, dt))
    } else {
        let (rp, rn, dt) = buf_mut(data);
        Ok((std::ptr::null::<std::ffi::c_void>(), 0i64, rp, rn, dt))
    }
}

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
/// # Argument validation
///
/// Collective methods that document an [`Error::InvalidBuffer`] condition check it
/// on the calling rank before any MPI call. The check is local: ranks that already
/// entered the collective are not told about the failure and may block.
///
/// # Errors in collective calls
///
/// MPI may return an error from a collective call on some ranks only. The
/// ranks that returned `Ok` go on to their next collective call and can
/// block there, waiting for a rank that returned `Err`. Methods that
/// document an error "on every rank" guarantee that for their own checks
/// once their collective calls have succeeded on every rank.
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
// Every MPI call goes through the lifecycle guard (`rt::enter`): below
// `ThreadLevel::Serialized` (Single or Funneled), it rejects a call from any
// thread other than the one that called `Mpi::init`/`init_thread`, with
// `Err(Error::ThreadLevelViolation)`, without touching MPI. A `Drop` cannot
// return `Err`, so the equivalent guard (`rt::drop_guard`) instead aborts the
// process on a wrong-thread drop below `Serialized`. At `Serialized`, the
// caller must serialize its own calls (debug builds detect two overlapping
// calls). At `Multiple`, calls from any thread may run concurrently.
//
// The C layer's handle tables claim slots lock-free: six with a C11
// atomic compare-exchange, the request table with an atomic bitmap
// (`fetch_or`), so there are no data races under MPI_THREAD_MULTIPLE.
// See docs/adr/0002-handle-tables.md for the full rationale and design.
unsafe impl Send for Communicator {}
// SAFETY: &Communicator exposes only reads of immutable fields and FFI calls
// gated by the same lifecycle guard described above, so its thread safety
// follows the requested `ThreadLevel` the same way.
unsafe impl Sync for Communicator {}

/// Handle of `MPI_COMM_WORLD`: slot 0 of the C communicator table, set at
/// init and never freed.
const WORLD_HANDLE: i32 = 0;

impl Communicator {
    /// Constant for opting out of a communicator split.
    ///
    /// Processes passing this as the `color` to [`split()`](Self::split) will not be
    /// included in any resulting communicator.
    pub const UNDEFINED: i32 = -1;

    /// Get a handle to `MPI_COMM_WORLD`.
    pub(crate) fn world() -> Self {
        Self::from_handle(WORLD_HANDLE).expect("COMM_WORLD must be valid post-init")
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
        // Don't free COMM_WORLD
        if self.handle != WORLD_HANDLE {
            if !rt::drop_guard("Communicator") {
                return;
            }
            // SAFETY: self.handle is a valid, non-zero communicator handle
            // registered in the C-side comm table (checked above); Drop takes
            // &mut self and runs at most once per value, so this cannot
            // double-free the handle.
            unsafe { ffi::ferrompi_comm_free(self.handle) };
        }
    }
}

/// A `Communicator` for unit tests: wraps `WORLD_HANDLE` so `Drop` never
/// reaches MPI, with the given rank and size.
#[cfg(test)]
fn test_comm(rank: i32, size: i32) -> Communicator {
    Communicator {
        handle: WORLD_HANDLE,
        rank,
        size,
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::{
        check_rank_slots, check_same_len, rank_block, scatter_inplace_args, SplitType,
    };
    use crate::datatype::DatatypeTag;
    use crate::error::Error;

    #[test]
    fn split_type_repr_value() {
        // SplitType::Shared has repr value 0
        assert_eq!(SplitType::Shared as i32, 0);
    }

    #[test]
    fn check_rank_slots_boundaries() {
        assert!(check_rank_slots(8, 2, 4).is_ok());
        assert!(matches!(
            check_rank_slots(7, 2, 4),
            Err(Error::InvalidBuffer)
        ));
        assert!(check_rank_slots(9, 2, 4).is_ok());
        assert!(check_rank_slots(0, 0, 4).is_ok());
        assert!(matches!(
            check_rank_slots(usize::MAX, usize::MAX / 2 + 1, 2),
            Err(Error::InvalidBuffer)
        ));
    }

    #[test]
    fn check_same_len_boundaries() {
        assert!(check_same_len(5, 5).is_ok());
        assert!(matches!(check_same_len(5, 4), Err(Error::InvalidBuffer)));
        assert!(check_same_len(0, 0).is_ok());
    }

    #[test]
    fn rank_block_boundaries() {
        assert_eq!(rank_block(8, 4).unwrap(), 2);
        assert!(matches!(rank_block(7, 4), Err(Error::InvalidBuffer)));
        assert_eq!(rank_block(0, 4).unwrap(), 0);
        assert!(matches!(rank_block(8, 0), Err(Error::InvalidBuffer)));
        assert!(matches!(rank_block(8, -1), Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scatter_inplace_args_shapes() {
        let mut root_data = [0i32, 1, 2, 3];
        let (sendbuf, sendcount, recvbuf, recvcount, tag) =
            scatter_inplace_args(&mut root_data, true, 2).unwrap();
        assert!(!sendbuf.is_null());
        assert_eq!(sendcount, 2);
        assert!(recvbuf.is_null());
        assert_eq!(recvcount, 0);
        assert_eq!(tag, DatatypeTag::I32 as i32);

        let mut non_root_data = [0i32];
        let (sendbuf, sendcount, recvbuf, recvcount, tag) =
            scatter_inplace_args(&mut non_root_data, false, 2).unwrap();
        assert!(sendbuf.is_null());
        assert_eq!(sendcount, 0);
        assert!(!recvbuf.is_null());
        assert_eq!(recvcount, 1);
        assert_eq!(tag, DatatypeTag::I32 as i32);

        let mut indivisible_data = [0i32, 1, 2];
        assert!(matches!(
            scatter_inplace_args(&mut indivisible_data, true, 2),
            Err(Error::InvalidBuffer)
        ));
    }
}
