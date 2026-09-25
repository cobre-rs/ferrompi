//! Shared memory windows for intra-node communication.
//!
//! This module provides [`SharedWindow<T>`], a safe Rust wrapper around
//! `MPI_Win_allocate_shared` with RAII lifecycle management. Shared memory
//! windows allow processes on the same node to directly access each other's
//! memory without explicit message passing, enabling high-performance
//! intra-node communication.
//!
//! # Synchronization
//!
//! MPI shared memory windows require explicit synchronization:
//!
//! - **Active target**: Use [`SharedWindow::fence()`] for bulk-synchronous access
//!   patterns where all processes participate in synchronization.
//! - **Passive target**: Use [`SharedWindow::lock()`] or [`SharedWindow::lock_all()`]
//!   for fine-grained, one-sided access patterns. These return RAII guards
//!   ([`LockGuard`] / [`LockAllGuard`]) that automatically unlock on drop.
//!
//! # Feature Gate
//!
//! This module is only available when the `rma` feature is enabled.
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::{Mpi, SharedWindow, LockType};
//!
//! let mpi = Mpi::init().unwrap();
//! let world = mpi.world();
//! let node = world.split_shared().unwrap();
//!
//! // Each process allocates 100 f64s in shared memory
//! let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
//!
//! // Write to local portion
//! {
//!     let local = win.local_slice_mut();
//!     for (i, x) in local.iter_mut().enumerate() {
//!         *x = (node.rank() * 100 + i as i32) as f64;
//!     }
//! }
//!
//! // Fence synchronization
//! win.fence().unwrap();
//!
//! // Read from any rank's memory
//! let remote = win.remote_slice(0).unwrap();
//! println!("Rank 0's first value: {}", remote[0]);
//! ```

use std::ops::{BitOr, BitOrAssign};
use std::ptr::NonNull;
use std::sync::OnceLock;

use crate::datatype::{buf, buf_mut};
use crate::error::{Error, Result};
use crate::ffi;
use crate::group::Group;
use crate::request::Request;
use crate::Communicator;
use crate::MpiDatatype;
use crate::ReduceOp;

// ============================================================================
// WinFenceAssert — bitflags for MPI_Win_fence assert hints
// ============================================================================

/// Assertion bitmask for [`Win::fence`] (active-target synchronization).
///
/// MPI allows callers to pass assertion hints to `MPI_Win_fence` to enable
/// implementation-side optimizations. The integer values of these constants
/// are **implementation-defined** (MPICH and Open MPI differ), so they are
/// queried from the C layer at first use via a `OnceLock<[i32; 4]>` and the
/// `ferrompi_win_fence_mode_values` shim.
///
/// # Composition
///
/// Flags can be combined with `|` and `|=`:
///
/// ```
/// use ferrompi::WinFenceAssert;
///
/// let flags = WinFenceAssert::no_store() | WinFenceAssert::no_put();
/// assert!(flags.bits() != 0);
/// ```
///
/// Use `WinFenceAssert::default()` (equivalent to `WinFenceAssert::none()`)
/// when no assertion is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct WinFenceAssert(i32);

/// Private cache for the four MPI fence mode constants.
///
/// Initialized once on the first call to [`fence_mode_values()`].
/// The array layout is `[NOSTORE, NOPUT, NOPRECEDE, NOSUCCEED]`.
static FENCE_MODE_VALUES: OnceLock<[i32; 4]> = OnceLock::new();

/// Query the four MPI fence mode constants, caching the result.
fn fence_mode_values() -> [i32; 4] {
    *FENCE_MODE_VALUES.get_or_init(|| {
        let mut out = [0i32; 4];
        // SAFETY: `out` is a stack-allocated 4-element array; we pass a valid
        // pointer to its first element. The C shim writes exactly 4 i32 values.
        unsafe { ffi::ferrompi_win_fence_mode_values(out.as_mut_ptr()) };
        out
    })
}

impl WinFenceAssert {
    /// No assertion — always valid; equivalent to passing `0` to
    /// `MPI_Win_fence`.
    #[inline]
    pub fn none() -> Self {
        WinFenceAssert(0)
    }

    /// Asserts that no stores to the window will occur in the following epoch
    /// (`MPI_MODE_NOSTORE`).
    #[inline]
    pub fn no_store() -> Self {
        WinFenceAssert(fence_mode_values()[0])
    }

    /// Asserts that no put operations will occur in the following epoch
    /// (`MPI_MODE_NOPUT`).
    #[inline]
    pub fn no_put() -> Self {
        WinFenceAssert(fence_mode_values()[1])
    }

    /// Asserts that no RMA operations precede this fence (`MPI_MODE_NOPRECEDE`).
    ///
    /// This hints to the implementation that no RMA access epoch was open
    /// before this fence, so no preceding operations need to be completed.
    #[inline]
    pub fn no_precede() -> Self {
        WinFenceAssert(fence_mode_values()[2])
    }

    /// Asserts that no RMA operations will follow this fence
    /// (`MPI_MODE_NOSUCCEED`).
    ///
    /// This hints that the fence closes the last access epoch, so the
    /// implementation need not prepare for subsequent RMA operations.
    #[inline]
    pub fn no_succeed() -> Self {
        WinFenceAssert(fence_mode_values()[3])
    }

    /// Returns the raw integer bitmask.
    #[inline]
    pub fn bits(self) -> i32 {
        self.0
    }

    /// Construct from a raw bit value.
    ///
    /// Intended for unit tests only — production code should use the named
    /// constructors so that the correct MPI constant values are used.
    #[cfg(test)]
    pub(crate) fn from_bits_for_test(v: i32) -> Self {
        WinFenceAssert(v)
    }
}

impl BitOr for WinFenceAssert {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        WinFenceAssert(self.0 | rhs.0)
    }
}

impl BitOrAssign for WinFenceAssert {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// ============================================================================
// WinPscwAssert — bitflags for MPI_Win_post / MPI_Win_start assert hints
// ============================================================================

/// Assertion bitmask for [`Win::post`] and [`Win::start`] (PSCW active-target
/// synchronization).
///
/// MPI allows callers to pass assertion hints to `MPI_Win_post` and
/// `MPI_Win_start` to enable implementation-side optimizations. The integer
/// values of these constants are **implementation-defined** (MPICH and Open MPI
/// differ), so they are queried from the C layer at first use via a
/// `OnceLock<[i32; 3]>` and the `ferrompi_win_pscw_mode_values` shim.
///
/// # Composition
///
/// Flags can be combined with `|` and `|=`:
///
/// ```
/// use ferrompi::WinPscwAssert;
///
/// let flags = WinPscwAssert::no_store() | WinPscwAssert::no_put();
/// assert!(flags.bits() != 0);
/// ```
///
/// Use `WinPscwAssert::default()` (equivalent to `WinPscwAssert::none()`)
/// when no assertion is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct WinPscwAssert(i32);

/// Private cache for the three MPI PSCW mode constants.
///
/// Initialized once on the first call to [`pscw_mode_values()`].
/// The array layout is `[NOCHECK, NOSTORE, NOPUT]`.
static PSCW_MODE_VALUES: OnceLock<[i32; 3]> = OnceLock::new();

/// Query the three MPI PSCW mode constants, caching the result.
fn pscw_mode_values() -> [i32; 3] {
    *PSCW_MODE_VALUES.get_or_init(|| {
        let mut out = [0i32; 3];
        // SAFETY: `out` is a stack-allocated 3-element array; we pass a valid
        // pointer to its first element. The C shim writes exactly 3 i32 values.
        unsafe { ffi::ferrompi_win_pscw_mode_values(out.as_mut_ptr()) };
        out
    })
}

impl WinPscwAssert {
    /// No assertion — always valid; equivalent to passing `0` to
    /// `MPI_Win_post` or `MPI_Win_start`.
    #[inline]
    pub fn none() -> Self {
        WinPscwAssert(0)
    }

    /// Asserts that no RMA synchronization is needed before accessing the
    /// window (`MPI_MODE_NOCHECK`).
    ///
    /// This is valid when the caller is certain that the matching post/start
    /// pair will not overlap with another exposure/access epoch.
    #[inline]
    pub fn no_check() -> Self {
        WinPscwAssert(pscw_mode_values()[0])
    }

    /// Asserts that no stores to the window will occur in the following epoch
    /// (`MPI_MODE_NOSTORE`).
    #[inline]
    pub fn no_store() -> Self {
        WinPscwAssert(pscw_mode_values()[1])
    }

    /// Asserts that no put operations will occur in the following epoch
    /// (`MPI_MODE_NOPUT`).
    #[inline]
    pub fn no_put() -> Self {
        WinPscwAssert(pscw_mode_values()[2])
    }

    /// Returns the raw integer bitmask.
    #[inline]
    pub fn bits(self) -> i32 {
        self.0
    }

    /// Construct from a raw bit value.
    ///
    /// Intended for unit tests only — production code should use the named
    /// constructors so that the correct MPI constant values are used.
    #[cfg(test)]
    pub(crate) fn from_bits_for_test(v: i32) -> Self {
        WinPscwAssert(v)
    }
}

impl BitOr for WinPscwAssert {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        WinPscwAssert(self.0 | rhs.0)
    }
}

impl BitOrAssign for WinPscwAssert {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Lock type for window synchronization.
///
/// Controls the level of concurrency allowed when locking an MPI window
/// for passive target access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    /// Exclusive access (no other locks allowed).
    ///
    /// Only one process may hold an exclusive lock on a given rank's
    /// window at a time. This is suitable for read-write access.
    Exclusive,
    /// Shared access (multiple shared locks allowed).
    ///
    /// Multiple processes may hold shared locks on a given rank's window
    /// simultaneously. This is suitable for read-only access patterns.
    Shared,
}

/// A shared memory window allocated across processes on the same node.
///
/// Created via [`SharedWindow::allocate()`] on a shared-memory communicator
/// (typically obtained from [`Communicator::split_shared()`]).
///
/// Each process in the communicator contributes a local segment of `local_count`
/// elements of type `T` to the shared memory region. All processes can then
/// access any rank's segment via [`remote_slice()`](Self::remote_slice), provided
/// proper synchronization is used.
///
/// # RAII Lifecycle
///
/// The underlying MPI window is freed automatically when the `SharedWindow`
/// is dropped. The shared memory region becomes invalid after the window is freed.
///
/// # Thread Safety
///
/// `SharedWindow` is intentionally **not** `Send` or `Sync`. MPI windows have
/// specific thread-safety rules that depend on the MPI thread level, and incorrect
/// cross-thread usage can cause data corruption or MPI errors.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, SharedWindow, LockType};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let node = world.split_shared().unwrap();
///
/// // Each process allocates 100 f64s in shared memory
/// let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
///
/// // Write to local portion
/// {
///     let local = win.local_slice_mut();
///     for (i, x) in local.iter_mut().enumerate() {
///         *x = (node.rank() * 100 + i as i32) as f64;
///     }
/// }
///
/// // Fence synchronization
/// win.fence().unwrap();
///
/// // Read from any rank's memory
/// let remote = win.remote_slice(0).unwrap();
/// println!("Rank 0's first value: {}", remote[0]);
/// ```
pub struct SharedWindow<T: MpiDatatype> {
    /// The MPI window handle (index into the C-layer window table).
    win_handle: i32,
    /// Non-null pointer to this process's local shared memory segment.
    local_ptr: NonNull<T>,
    /// Number of `T` elements in the local segment.
    local_len: usize,
    /// Number of processes in the communicator that created this window.
    comm_size: i32,
}

impl<T: MpiDatatype> SharedWindow<T> {
    /// Allocate a shared memory window.
    ///
    /// Each process in `comm` allocates `local_count` elements of type `T`
    /// in a shared memory segment accessible by all processes in the communicator.
    /// The communicator should be a shared-memory communicator (e.g., from
    /// [`Communicator::split_shared()`]).
    ///
    /// # Arguments
    ///
    /// * `comm` - A shared-memory communicator (all processes must be on the same node)
    /// * `local_count` - Number of elements of type `T` to allocate locally
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The MPI window allocation fails (e.g., insufficient shared memory)
    /// - The MPI implementation returns a null base pointer
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, SharedWindow};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let node = mpi.world().split_shared().unwrap();
    /// let win = SharedWindow::<f64>::allocate(&node, 1024).unwrap();
    /// ```
    pub fn allocate(comm: &Communicator, local_count: usize) -> Result<Self> {
        let byte_size = local_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::InvalidBuffer)?;
        let size = i64::try_from(byte_size).map_err(|_| Error::InvalidBuffer)?;
        let disp_unit = std::mem::size_of::<T>() as i32;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut win_handle: i32 = 0;

        // SAFETY: We pass valid pointers for out-parameters. The C layer
        // allocates shared memory and returns a window handle + base pointer.
        let ret = unsafe {
            ffi::ferrompi_win_allocate_shared(
                size,
                disp_unit,
                -1, // MPI_INFO_NULL
                comm.raw_handle(),
                &mut baseptr,
                &mut win_handle,
            )
        };
        Error::check_with_op(ret, "win_allocate_shared")?;

        let local_ptr = NonNull::new(baseptr.cast::<T>())
            .ok_or_else(|| Error::Internal("Win_allocate_shared returned null".into()))?;

        Ok(SharedWindow {
            win_handle,
            local_ptr,
            local_len: local_count,
            comm_size: comm.size(),
        })
    }

    /// Get a slice of this process's local shared memory segment.
    ///
    /// The returned slice provides read access to the `local_count` elements
    /// that were allocated by this process in [`allocate()`](Self::allocate).
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before reading data that may have been written by other processes.
    pub fn local_slice(&self) -> &[T] {
        // SAFETY: `local_ptr` was returned by MPI_Win_allocate_shared and is
        // guaranteed valid for `local_len` elements of type T for the lifetime
        // of the window. The pointer is non-null (checked in allocate()).
        unsafe { std::slice::from_raw_parts(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get a mutable slice of this process's local shared memory segment.
    ///
    /// The returned slice provides read-write access to the `local_count` elements
    /// that were allocated by this process in [`allocate()`](Self::allocate).
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before writing data that other processes may read.
    pub fn local_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `local_ptr` was returned by MPI_Win_allocate_shared and is
        // guaranteed valid for `local_len` elements of type T for the lifetime
        // of the window. We have `&mut self` ensuring exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Query another rank's shared memory region.
    ///
    /// Returns a slice of the shared memory segment allocated by the specified
    /// rank. This allows direct read access to remote memory without any
    /// message passing overhead.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank whose shared memory to query (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The rank is out of range
    /// - The MPI shared query call fails
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before reading data that was written by the remote process.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// // Read rank 0's shared memory
    /// let remote = win.remote_slice(0).unwrap();
    /// println!("Rank 0 has {} elements, first = {}", remote.len(), remote[0]);
    /// ```
    pub fn remote_slice(&self, rank: i32) -> Result<&[T]> {
        let mut size: i64 = 0;
        let mut disp_unit: i32 = 0;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();

        // SAFETY: We pass valid pointers for out-parameters. The C layer queries
        // the MPI window for the base pointer and size of the specified rank's
        // shared memory segment.
        let ret = unsafe {
            ffi::ferrompi_win_shared_query(
                self.win_handle,
                rank,
                &mut size,
                &mut disp_unit,
                &mut baseptr,
            )
        };
        Error::check_with_op(ret, "win_shared_query")?;

        let count = size as usize / std::mem::size_of::<T>();
        if baseptr.is_null() {
            if count == 0 {
                // Null with zero count: return an empty slice using a dangling pointer
                // SAFETY: NonNull::dangling() is a valid, aligned pointer for zero-length slices
                return Ok(unsafe {
                    std::slice::from_raw_parts(NonNull::<T>::dangling().as_ptr(), 0)
                });
            }
            return Err(Error::Internal(
                "MPI_Win_shared_query returned null for non-zero size".into(),
            ));
        }
        // SAFETY: MPI_Win_shared_query returns a valid pointer to the shared
        // memory segment of the specified rank. The pointer is valid for `count`
        // elements of type T for the lifetime of the window.
        Ok(unsafe { std::slice::from_raw_parts(baseptr.cast::<T>(), count) })
    }

    /// Fence synchronization (active target).
    ///
    /// A collective operation that synchronizes all accesses to the window.
    /// All processes in the window's communicator must call this function.
    /// After `fence()` returns, all preceding local stores are visible to
    /// remote processes, and all remote stores are visible locally.
    ///
    /// This is the simplest synchronization mode, suitable for bulk-synchronous
    /// access patterns (e.g., write phase → fence → read phase → fence).
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI fence operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// // Write phase
    /// win.local_slice_mut()[0] = 42.0;
    ///
    /// // Synchronize
    /// win.fence().unwrap();
    ///
    /// // Read phase — all writes from all ranks are now visible
    /// let remote = win.remote_slice(0).unwrap();
    /// ```
    pub fn fence(&self) -> Result<()> {
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_fence(0, self.win_handle) };
        Error::check_with_op(ret, "win_fence")
    }

    /// Lock a specific rank's window (passive target synchronization).
    ///
    /// Acquires a lock on the specified rank's window, allowing one-sided
    /// access to that rank's shared memory. Returns a [`LockGuard`] that
    /// automatically unlocks the window when dropped.
    ///
    /// # Arguments
    ///
    /// * `lock_type` - [`LockType::Exclusive`] for read-write access, or
    ///   [`LockType::Shared`] for read-only access
    /// * `rank` - The rank to lock (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI lock operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow, LockType};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// {
    ///     let guard = win.lock(LockType::Shared, 0).unwrap();
    ///     let remote = win.remote_slice(0).unwrap();
    ///     println!("Value: {}", remote[0]);
    ///     guard.flush().unwrap();
    ///     // Lock is released when `guard` is dropped
    /// }
    /// ```
    pub fn lock(&self, lock_type: LockType, rank: i32) -> Result<LockGuard<'_, T>> {
        let lt = match lock_type {
            LockType::Exclusive => ffi::FERROMPI_LOCK_EXCLUSIVE,
            LockType::Shared => ffi::FERROMPI_LOCK_SHARED,
        };
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_lock(lt, rank, 0, self.win_handle) };
        Error::check_with_op(ret, "win_lock")?;
        Ok(LockGuard { window: self, rank })
    }

    /// Lock all ranks' windows (passive target synchronization).
    ///
    /// Acquires shared locks on all ranks in the window's communicator.
    /// Returns a [`LockAllGuard`] that automatically unlocks all windows
    /// when dropped.
    ///
    /// This is useful for algorithms that need to access multiple ranks'
    /// memory in a single epoch without individual lock/unlock overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI lock_all operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// {
    ///     let guard = win.lock_all().unwrap();
    ///     // Access any rank's memory
    ///     for rank in 0..node.size() {
    ///         let remote = win.remote_slice(rank).unwrap();
    ///         println!("Rank {rank} first value: {}", remote[0]);
    ///     }
    ///     guard.flush_all().unwrap();
    ///     // All locks released when `guard` is dropped
    /// }
    /// ```
    pub fn lock_all(&self) -> Result<LockAllGuard<'_, T>> {
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_lock_all(0, self.win_handle) };
        Error::check_with_op(ret, "win_lock_all")?;
        Ok(LockAllGuard { window: self })
    }

    /// Get the raw MPI window handle.
    ///
    /// This is provided for advanced use cases where direct access to the
    /// underlying MPI window handle is needed (e.g., custom FFI calls).
    pub fn raw_handle(&self) -> i32 {
        self.win_handle
    }

    /// Get the number of processes in the window's communicator.
    ///
    /// This equals the size of the communicator that was used to create
    /// the window, and determines the valid range of ranks for
    /// [`remote_slice()`](Self::remote_slice) and [`lock()`](Self::lock).
    pub fn comm_size(&self) -> i32 {
        self.comm_size
    }
}

impl<T: MpiDatatype> Drop for SharedWindow<T> {
    fn drop(&mut self) {
        // SAFETY: win_handle is a valid MPI window handle that was allocated
        // by ferrompi_win_allocate_shared. It has not been freed yet because
        // Drop is only called once, and we don't expose a manual free method.
        unsafe { ffi::ferrompi_win_free(self.win_handle) };
    }
}

// SharedWindow is not Send/Sync by default due to NonNull<T>.
// This is the correct behavior — MPI windows have specific thread-safety
// rules and should not be shared across threads without careful coordination.

// ============================================================================
// Win<'a, T> — general-purpose distributed RMA window
// ============================================================================

/// Opaque holder for the buffers of a pending RMA fetch operation.
///
/// `MPI_Fetch_and_op` and `MPI_Compare_and_swap` only *initiate* the
/// operation. Both the origin (and `compare`, for CAS) buffer and the
/// result buffer must remain at stable memory addresses until the
/// surrounding epoch closes (via `fence`, `complete`/`unlock`, or `flush`)
/// — MPI's network layer may not actually read/write them until then.
///
/// The Rust language's stack-frame and move semantics make naïve
/// `&origin` / `&result` arguments unsound: the function returns before
/// the epoch closes, and the stack locations are reused. To guarantee
/// stable addresses, this type owns heap allocations for the origin (and
/// compare) operands and the result destination; each is freed exactly
/// once when the struct is dropped, whether directly or via [`resolve`].
///
/// The type also carries an `Option<Box<T>>` for `compare_and_swap`'s
/// extra `compare` argument; it is `None` for `fetch_and_op`.
///
/// [`resolve`]: PendingFetchResult::resolve
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let mut win = Win::<i32>::allocate(&world, 1).unwrap();
///
/// win.fence(WinFenceAssert::default()).unwrap();
///
/// let pending = win.fetch_and_op(1, 1, 0, ReduceOp::Sum).unwrap();
///
/// // Close epoch — MPI_Fetch_and_op completes here
/// win.fence(WinFenceAssert::default()).unwrap();
///
/// // SAFETY: epoch is closed; result is now populated.
/// let old = unsafe { pending.resolve() };
/// assert_eq!(old, 100);
/// ```
pub struct PendingFetchResult<T> {
    /// Origin buffer — kept alive so MPI's pending RMA read sees stable
    /// memory across the epoch.  Not used after resolve(); the Box is
    /// dropped when PendingFetchResult is dropped.
    _origin: Box<T>,
    /// Optional `compare` operand for `MPI_Compare_and_swap`.  `None` for
    /// `MPI_Fetch_and_op`.  Same lifetime invariant as `_origin`.
    _compare: Option<Box<T>>,
    /// Result destination — kept alive so MPI's pending RMA write reaches a
    /// stable destination.  Owns its allocation via a raw pointer obtained
    /// from `Box::into_raw` (never a cast from a shared borrow), freed
    /// exactly once by `Drop`.  resolve() reads the value once the epoch
    /// has closed.
    result: NonNull<std::mem::MaybeUninit<T>>,
}

// SAFETY: `result` is a `NonNull` pointer to a heap allocation exclusively owned by
// this struct (freed exactly once, by `Drop`), functionally equivalent to
// `Box<MaybeUninit<T>>`'s ownership semantics. No other code holds a copy of this
// pointer, so sending the struct across threads is sound whenever T: Send.
unsafe impl<T: Send> Send for PendingFetchResult<T> {}
// SAFETY: `result` is a `NonNull` pointer to a heap allocation exclusively owned by
// this struct, reachable only through `self`-consuming methods (`resolve`, `Drop`); no
// interior mutability is exposed through `&PendingFetchResult<T>`, so sharing a shared
// reference across threads is sound whenever T: Sync.
unsafe impl<T: Sync> Sync for PendingFetchResult<T> {}

impl<T: Copy> PendingFetchResult<T> {
    /// Consume the pending result and return the fetched value.
    ///
    /// # Safety
    ///
    /// The caller **must** ensure that the surrounding RMA epoch has been
    /// closed before calling this method. Specifically:
    ///
    /// - For fence-mode windows: [`Win::fence`] must have been called to close
    ///   the epoch.
    /// - For PSCW windows: `MPI_Win_complete` / `MPI_Win_wait` must have been
    ///   called.
    /// - For lock-mode windows: `MPI_Win_unlock` or `MPI_Win_flush` must have
    ///   been called.
    ///
    /// Reading the result before the epoch closes yields an unspecified,
    /// possibly uninitialised value (UB in Rust).
    #[inline]
    pub unsafe fn resolve(self) -> T {
        // SAFETY: Caller guarantees the epoch has closed, at which point MPI has
        // initialised the allocation at `self.result` with the pre-update remote value
        // per MPI-3 §11.6.  All T: Copy types (enforced by the MpiDatatype +
        // AtomicMpiDatatype bounds at the call site) are valid for any bit pattern, so
        // assuming init is sound.  `read()` copies the `MaybeUninit<T>` out of the
        // allocation without freeing it; `self`'s `Drop` impl frees it exactly once,
        // along with the `_origin` and `_compare` boxes, when this function returns.
        unsafe { self.result.as_ptr().read().assume_init() }
    }
}

impl<T> Drop for PendingFetchResult<T> {
    fn drop(&mut self) {
        // SAFETY: `result` was obtained from `Box::into_raw` in `fetch_and_op` or
        // `compare_and_swap` and has not been freed since (the only other free site, the
        // FFI error path in those constructors, returns before a `PendingFetchResult` is
        // ever built). `Drop::drop` runs at most once per value, so reconstructing the
        // `Box` here and letting it drop frees the allocation exactly once.
        unsafe { drop(Box::from_raw(self.result.as_ptr())) };
    }
}

/// Whether a window's local memory is caller-supplied (`Win::create`) or
/// MPI-allocated (`Win::allocate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinKind {
    /// Buffer was supplied by the caller (`Win::create`). The user owns the
    /// memory; `Drop` calls `MPI_Win_free` but does **not** free the buffer.
    Created,
    /// Buffer was allocated by MPI (`Win::allocate`). `Drop` calls
    /// `MPI_Win_free` which also releases the MPI-managed buffer.
    Allocated,
}

/// A general-purpose distributed RMA window.
///
/// `Win<'a, T>` wraps either:
///
/// - `MPI_Win_create` (`Win::create`) — the caller supplies the buffer and
///   retains ownership; the lifetime `'a` ensures the buffer outlives the
///   window.
/// - `MPI_Win_allocate` (`Win::allocate`) — MPI allocates the buffer; the
///   window uses `'static` because there is no caller buffer to track.
///
/// Unlike [`SharedWindow`], which uses `MPI_Win_allocate_shared` and is
/// restricted to intra-node communicators, `Win` works across nodes and is
/// the foundation for all distributed RMA operations (Put, Get, Accumulate,
/// etc.) in this library.
///
/// # RAII Lifecycle
///
/// `MPI_Win_free` is called automatically when the `Win` is dropped. For
/// `Win::allocate`, this also releases the MPI-managed buffer.
///
/// # Thread Safety
///
/// `Win` is intentionally **not** `Send` or `Sync`. MPI windows have
/// specific thread-safety rules that depend on the MPI thread level.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, Win};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Create a window backed by a caller-supplied buffer
/// let mut buf = vec![0i32; 16];
/// let win = Win::create(&world, &mut buf).unwrap();
/// assert!(win.raw_handle() >= 0);
/// assert_eq!(win.comm_size(), world.size());
/// drop(win); // MPI_Win_free called here
/// ```
pub struct Win<'a, T: MpiDatatype> {
    /// The MPI window handle (index into the C-layer window table).
    win_handle: i32,
    /// Non-null pointer to this process's local window buffer.
    ///
    /// For `Win::create` this points into the caller-supplied slice.
    /// For `Win::allocate` this points into the MPI-managed allocation.
    local_ptr: NonNull<T>,
    /// Number of `T` elements in the local buffer.
    local_len: usize,
    /// Number of processes in the window's communicator.
    comm_size: i32,
    /// Captures the `'a` lifetime so the borrow checker enforces that a
    /// caller-supplied buffer outlives the `Win`. For `Win::allocate` (which
    /// uses `'static`) this is a zero-sized phantom that imposes no constraint.
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T: MpiDatatype> Win<'a, T> {
    /// Create an RMA window backed by a caller-supplied buffer.
    ///
    /// Wraps `MPI_Win_create`. The buffer `buf` is mutably borrowed for the
    /// lifetime `'a` of the returned `Win`, ensuring it cannot be dropped or
    /// moved while the window is alive.
    ///
    /// `disp_unit` is set to `size_of::<T>()` and `size` to
    /// `buf.len() * size_of::<T>()`.
    ///
    /// # Arguments
    ///
    /// * `comm` - Communicator; all processes must call this collectively.
    /// * `buf`  - Mutable slice that forms this process's local window memory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The byte size overflows `i64` (`Error::InvalidBuffer`).
    /// - The MPI call fails (`Error::Mpi` with `operation: Some("win_create")`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut buf = vec![0f64; 64];
    /// let win = Win::create(&world, &mut buf).unwrap();
    /// ```
    pub fn create(comm: &Communicator, buf: &'a mut [T]) -> Result<Self> {
        let byte_size = buf
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::InvalidBuffer)?;
        let size = i64::try_from(byte_size).map_err(|_| Error::InvalidBuffer)?;
        let disp_unit = std::mem::size_of::<T>() as i32;
        let mut win_handle: i32 = 0;

        // SAFETY: `buf` is a valid, aligned mutable slice borrowed for `'a`.
        // We pass its raw pointer and byte length to MPI_Win_create. The C
        // shim validates the comm handle before calling MPI. The pointer
        // remains valid for the lifetime of the window because `'a` ensures
        // `buf` outlives `Self`.
        let ret = unsafe {
            ffi::ferrompi_win_create(
                buf.as_mut_ptr().cast::<std::ffi::c_void>(),
                size,
                disp_unit,
                -1, // MPI_INFO_NULL
                comm.raw_handle(),
                &mut win_handle,
            )
        };
        Error::check_with_op(ret, "win_create")?;

        // SAFETY: `buf` is a non-empty or zero-length caller slice. For
        // non-zero length, `buf.as_mut_ptr()` is guaranteed non-null. For
        // zero length we use `NonNull::dangling()` as a valid aligned sentinel.
        let local_ptr = if buf.is_empty() {
            NonNull::<T>::dangling()
        } else {
            // SAFETY: buf.as_mut_ptr() is non-null when buf is non-empty.
            unsafe { NonNull::new_unchecked(buf.as_mut_ptr()) }
        };

        Ok(Win {
            win_handle,
            local_ptr,
            local_len: buf.len(),
            comm_size: comm.size(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: MpiDatatype> Win<'static, T> {
    /// Allocate an RMA window with MPI-managed memory.
    ///
    /// Wraps `MPI_Win_allocate`. MPI allocates `local_count` elements of type
    /// `T` for this process. The allocation is freed automatically when the
    /// window is dropped (via `MPI_Win_free`). The returned `Win` is
    /// `'static` because there is no caller buffer to track.
    ///
    /// `disp_unit` is set to `size_of::<T>()` and `size` to
    /// `local_count * size_of::<T>()`.
    ///
    /// # Arguments
    ///
    /// * `comm`        - Communicator; all processes must call this collectively.
    /// * `local_count` - Number of `T` elements to allocate for this process.
    ///   Zero is allowed; a valid window with a dangling base pointer is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The byte size overflows `i64` (`Error::InvalidBuffer`).
    /// - The MPI call fails (`Error::Mpi` with `operation: Some("win_allocate")`).
    /// - MPI returns a null pointer for a non-zero count (`Error::Internal`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 32).unwrap();
    /// ```
    pub fn allocate(comm: &Communicator, local_count: usize) -> Result<Self> {
        let byte_size = local_count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::InvalidBuffer)?;
        let size = i64::try_from(byte_size).map_err(|_| Error::InvalidBuffer)?;
        let disp_unit = std::mem::size_of::<T>() as i32;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut win_handle: i32 = 0;

        // SAFETY: We pass valid out-parameter pointers to the C shim, which
        // calls MPI_Win_allocate. The C shim validates the comm handle. The
        // returned baseptr is either a valid MPI-allocated pointer or null
        // (only valid if local_count == 0).
        let ret = unsafe {
            ffi::ferrompi_win_allocate(
                size,
                disp_unit,
                -1, // MPI_INFO_NULL
                comm.raw_handle(),
                &mut baseptr,
                &mut win_handle,
            )
        };
        Error::check_with_op(ret, "win_allocate")?;

        let local_ptr = if local_count == 0 {
            // Zero-count: use a dangling aligned pointer (same trick as
            // SharedWindow::remote_slice for zero-size slices).
            NonNull::<T>::dangling()
        } else {
            NonNull::new(baseptr.cast::<T>()).ok_or_else(|| {
                Error::Internal("Win_allocate returned null base pointer for non-zero count".into())
            })?
        };

        Ok(Win {
            win_handle,
            local_ptr,
            local_len: local_count,
            comm_size: comm.size(),
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: MpiDatatype> Win<'_, T> {
    /// Get a shared slice of this process's local window buffer.
    ///
    /// For `Win::create`, this returns a view into the caller-supplied buffer.
    /// For `Win::allocate`, this returns a view into the MPI-managed buffer.
    ///
    /// The borrow lifetime is tied to `&self`, so the slice cannot outlive the
    /// window. Proper MPI epoch synchronization (fence, lock, or PSCW) must be
    /// used before reading data written by remote processes.
    pub fn local_slice(&self) -> &[T] {
        // SAFETY: `local_ptr` was set in the constructor from either a valid
        // caller slice (`Win::create`) or a valid MPI allocation
        // (`Win::allocate`). Both are guaranteed valid for `local_len`
        // elements of type T for the lifetime of the window. The pointer is
        // non-null (zero-length windows use NonNull::dangling()). We hold
        // `&self` so no mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get a mutable slice of this process's local window buffer.
    ///
    /// For `Win::create`, this returns a mutable view into the caller-supplied
    /// buffer. For `Win::allocate`, this returns a mutable view into the
    /// MPI-managed buffer.
    ///
    /// The borrow lifetime is tied to `&mut self`, ensuring exclusive access.
    /// Proper MPI epoch synchronization must be used before writing data that
    /// remote processes may read.
    pub fn local_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `local_ptr` was set in the constructor from either a valid
        // caller slice (`Win::create`) or a valid MPI allocation
        // (`Win::allocate`). Both are guaranteed valid for `local_len`
        // elements of type T. We hold `&mut self`, ensuring exclusive access
        // and no concurrent shared references to the same memory.
        unsafe { std::slice::from_raw_parts_mut(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get the raw MPI window handle.
    ///
    /// Provided for advanced use cases where direct access to the underlying
    /// MPI window handle is needed (e.g., custom FFI calls).
    pub fn raw_handle(&self) -> i32 {
        self.win_handle
    }

    /// Get the number of processes in the window's communicator.
    ///
    /// Equals the size of the communicator used to create the window and
    /// determines the valid rank range for RMA operations.
    pub fn comm_size(&self) -> i32 {
        self.comm_size
    }

    /// Fence synchronization (active-target epoch boundary).
    ///
    /// A collective operation over all processes in this window's communicator.
    /// Every rank must call `fence` at the same logical program point. A pair of
    /// `fence` calls delimits an access/exposure epoch:
    ///
    /// ```text
    /// win.fence(WinFenceAssert::none())   // open epoch
    /// // ... RMA operations (put / get / accumulate) ...
    /// win.fence(WinFenceAssert::none())   // close epoch
    /// ```
    ///
    /// After the closing `fence` returns, all preceding RMA operations are
    /// complete and their effects are visible to all processes.
    ///
    /// # Arguments
    ///
    /// * `assert` — Assertion hint bitmask. Use [`WinFenceAssert::default()`]
    ///   (or [`WinFenceAssert::none()`]) when no optimization hints are needed.
    ///   Compose multiple hints with `|`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_fence"), .. }` if MPI
    /// reports an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 16).unwrap();
    ///
    /// // Open epoch
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// // ... RMA operations would go here ...
    /// // Close epoch
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// ```
    pub fn fence(&self, assert: WinFenceAssert) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // `ferrompi_win_create` or `ferrompi_win_allocate`. The assert value
        // is a valid bitmask obtained from `WinFenceAssert`, which is either
        // 0 (none) or a combination of values returned by the MPI implementation.
        let ret = unsafe { ffi::ferrompi_win_fence(assert.bits(), self.win_handle) };
        Error::check_with_op(ret, "win_fence")
    }

    /// Start an exposure epoch (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_post`. This rank exposes its window memory to the
    /// ranks in `group`, which are expected to call [`Win::start`] and then
    /// [`Win::complete`]. Close the exposure epoch with [`Win::wait_exposure`]
    /// or [`Win::test_exposure`].
    ///
    /// # Arguments
    ///
    /// * `group`  — The group of ranks that will issue RMA operations against
    ///   this rank's window.
    /// * `assert` — Assertion hint bitmask. Use [`WinPscwAssert::default()`]
    ///   (or [`WinPscwAssert::none()`]) when no optimization hints are needed.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_post"), .. }` if MPI reports
    /// an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinPscwAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// let access_group = world.group().unwrap().include(&[1]).unwrap();
    ///
    /// // Rank 0 exposes its window to rank 1
    /// win.post(&access_group, WinPscwAssert::default()).unwrap();
    /// win.wait_exposure().unwrap();
    /// ```
    pub fn post(&self, group: &Group, assert: WinPscwAssert) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. `group.raw_handle()`
        // returns a valid group handle from the C-layer group table.
        // `assert.bits()` is either 0 or a combination of MPI_MODE_* constants
        // returned by the MPI implementation.
        let ret =
            unsafe { ffi::ferrompi_win_post(group.raw_handle(), assert.bits(), self.win_handle) };
        Error::check_with_op(ret, "win_post")
    }

    /// Start an access epoch (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_start`. This rank will issue RMA operations against the
    /// ranks in `group`, which must have called (or will call) [`Win::post`].
    /// Close the access epoch with [`Win::complete`].
    ///
    /// # Arguments
    ///
    /// * `group`  — The group of ranks that have called or will call
    ///   [`Win::post`].
    /// * `assert` — Assertion hint bitmask. Use [`WinPscwAssert::default()`]
    ///   when no optimization hints are needed.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_start"), .. }` if MPI reports
    /// an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinPscwAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// let exposure_group = world.group().unwrap().include(&[0]).unwrap();
    ///
    /// // Rank 1 opens an access epoch to rank 0's window
    /// win.start(&exposure_group, WinPscwAssert::default()).unwrap();
    /// win.complete().unwrap();
    /// ```
    pub fn start(&self, group: &Group, assert: WinPscwAssert) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. `group.raw_handle()`
        // returns a valid group handle from the C-layer group table.
        // `assert.bits()` is either 0 or a combination of MPI_MODE_* constants.
        let ret =
            unsafe { ffi::ferrompi_win_start(group.raw_handle(), assert.bits(), self.win_handle) };
        Error::check_with_op(ret, "win_start")
    }

    /// Complete an access epoch (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_complete`. Closes the access epoch opened by
    /// [`Win::start`]. All RMA operations issued since the matching
    /// `Win::start` are guaranteed to be complete after this returns.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_complete"), .. }` if MPI
    /// reports an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinPscwAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// # let g = world.group().unwrap().include(&[0]).unwrap();
    /// win.start(&g, WinPscwAssert::default()).unwrap();
    /// // ... RMA operations ...
    /// win.complete().unwrap();
    /// ```
    pub fn complete(&self) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. An active access
        // epoch was opened by a prior call to `ferrompi_win_start`.
        let ret = unsafe { ffi::ferrompi_win_complete(self.win_handle) };
        Error::check_with_op(ret, "win_complete")
    }

    /// Wait for the exposure epoch to close (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_wait`. Blocks until all RMA operations issued by the
    /// access group (those that called [`Win::start`] / [`Win::complete`])
    /// against this rank's window are complete. Closes the exposure epoch
    /// opened by [`Win::post`].
    ///
    /// Named `wait_exposure` rather than `wait` to avoid a name collision with
    /// `Request::wait` when both are in scope via `use ferrompi::*`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_wait"), .. }` if MPI reports
    /// an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinPscwAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// # let g = world.group().unwrap().include(&[1]).unwrap();
    /// win.post(&g, WinPscwAssert::default()).unwrap();
    /// // ... wait for remote access to complete ...
    /// win.wait_exposure().unwrap();
    /// ```
    pub fn wait_exposure(&self) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. An active exposure
        // epoch was opened by a prior call to `ferrompi_win_post`.
        let ret = unsafe { ffi::ferrompi_win_wait(self.win_handle) };
        Error::check_with_op(ret, "win_wait")
    }

    /// Nonblocking test for exposure epoch completion (active-target PSCW).
    ///
    /// Wraps `MPI_Win_test`. Returns `true` if the exposure epoch started by
    /// [`Win::post`] has completed (i.e., all access-side ranks have called
    /// [`Win::complete`]), `false` otherwise. Does not block.
    ///
    /// Named `test_exposure` rather than `test` to avoid a name collision when
    /// both `Win` and `Request` are in scope via `use ferrompi::*`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_test"), .. }` if MPI reports
    /// an error.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinPscwAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// # let g = world.group().unwrap().include(&[1]).unwrap();
    /// win.post(&g, WinPscwAssert::default()).unwrap();
    /// // Polling loop (in practice, do useful work between tests)
    /// while !win.test_exposure().unwrap() {
    ///     // do other work
    /// }
    /// ```
    pub fn test_exposure(&self) -> Result<bool> {
        let mut flag: i32 = 0;
        // SAFETY: `win_handle` is a valid MPI window handle. `flag` is a valid
        // stack-allocated i32; the C shim writes exactly one i32 to it via the
        // `*flag` output parameter.
        let ret = unsafe { ffi::ferrompi_win_test(self.win_handle, &mut flag) };
        Error::check_with_op(ret, "win_test")?;
        Ok(flag != 0)
    }
}

impl<T: MpiDatatype> Win<'_, T> {
    /// Lock a specific rank's window (passive-target synchronization).
    ///
    /// Acquires a lock on the specified rank's window, allowing one-sided
    /// access to that rank's memory without requiring the target rank's active
    /// participation. Returns a [`WinLockGuard`] that automatically unlocks
    /// the window when dropped.
    ///
    /// # Arguments
    ///
    /// * `lock_type` - [`LockType::Exclusive`] for read-write access, or
    ///   [`LockType::Shared`] for concurrent read-only access with other
    ///   shared lockers.
    /// * `rank` - The target rank to lock (0-based, within the window's
    ///   communicator).
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_lock"), .. }` if the MPI
    /// lock call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, LockType};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// {
    ///     let guard = win.lock(LockType::Shared, 0).unwrap();
    ///     guard.flush().unwrap();
    ///     // Lock released when `guard` is dropped
    /// }
    /// ```
    pub fn lock(&self, lock_type: LockType, rank: i32) -> Result<WinLockGuard<'_, '_, T>> {
        let lt = match lock_type {
            LockType::Exclusive => ffi::FERROMPI_LOCK_EXCLUSIVE,
            LockType::Shared => ffi::FERROMPI_LOCK_SHARED,
        };
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // `ferrompi_win_create` or `ferrompi_win_allocate`. `lt` is one of
        // the two valid MPI lock-type constants obtained from the C layer.
        // `rank` is caller-supplied; MPI validates it against the window's
        // communicator.
        let ret = unsafe { ffi::ferrompi_win_lock(lt, rank, 0, self.win_handle) };
        Error::check_with_op(ret, "win_lock")?;
        Ok(WinLockGuard { window: self, rank })
    }

    /// Lock all ranks' windows (passive-target synchronization).
    ///
    /// Acquires shared locks on all ranks in the window's communicator.
    /// Returns a [`WinLockAllGuard`] that automatically releases all locks
    /// when dropped.
    ///
    /// Useful for algorithms that need concurrent access to every rank's
    /// window memory in a single epoch without the overhead of individual
    /// `lock` / `unlock` calls per rank.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_lock_all"), .. }` if the
    /// MPI lock_all call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// {
    ///     let guard = win.lock_all().unwrap();
    ///     guard.flush_all().unwrap();
    ///     // All locks released when `guard` is dropped
    /// }
    /// ```
    pub fn lock_all(&self) -> Result<WinLockAllGuard<'_, '_, T>> {
        // SAFETY: `win_handle` is a valid MPI window handle. Passing `0` for
        // the assert value is always valid (no optimization hints).
        let ret = unsafe { ffi::ferrompi_win_lock_all(0, self.win_handle) };
        Error::check_with_op(ret, "win_lock_all")?;
        Ok(WinLockAllGuard { window: self })
    }
}

impl<T: MpiDatatype> Win<'_, T> {
    /// Locally complete pending RMA operations to a specific target rank.
    ///
    /// Ensures that all preceding RMA operations issued to `rank` have
    /// completed locally: the origin buffer is safe to reuse, but the remote
    /// process may not yet have observed the writes.
    ///
    /// # Safety (MPI contract)
    ///
    /// This function **must** be called inside an active passive-target epoch
    /// (i.e., between [`Win::lock`] / [`Win::lock_all`] and the corresponding
    /// unlock). Calling it outside a passive-target epoch is undefined behavior
    /// per the MPI standard.
    ///
    /// # Arguments
    ///
    /// * `rank` - The target rank whose pending operations are to be locally
    ///   completed (0-based, within the window's communicator).
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_flush_local"), .. }` if the
    /// MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, LockType};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// {
    ///     let _guard = win.lock(LockType::Shared, 0).unwrap();
    ///     win.flush_local(0).unwrap();
    ///     // Lock released when `_guard` is dropped
    /// }
    /// ```
    pub fn flush_local(&self, rank: i32) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. `rank` is
        // caller-supplied; MPI validates it against the window's communicator.
        // The caller is responsible for ensuring this is called within a
        // passive-target epoch.
        let ret = unsafe { ffi::ferrompi_win_flush_local(rank, self.win_handle) };
        Error::check_with_op(ret, "win_flush_local")
    }

    /// Locally complete pending RMA operations to all target ranks.
    ///
    /// Ensures that all preceding RMA operations issued to any target have
    /// completed locally. The origin buffers are safe to reuse, but remote
    /// processes may not yet have observed the writes.
    ///
    /// # Safety (MPI contract)
    ///
    /// This function **must** be called inside an active passive-target epoch.
    /// Calling it outside a passive-target epoch is undefined behavior per the
    /// MPI standard.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_flush_local_all"), .. }` if
    /// the MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// {
    ///     let _guard = win.lock_all().unwrap();
    ///     win.flush_local_all().unwrap();
    ///     // Locks released when `_guard` is dropped
    /// }
    /// ```
    pub fn flush_local_all(&self) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. The caller is
        // responsible for ensuring this is called within a passive-target epoch.
        let ret = unsafe { ffi::ferrompi_win_flush_local_all(self.win_handle) };
        Error::check_with_op(ret, "win_flush_local_all")
    }

    /// Synchronize the local public window copy with the local private copy.
    ///
    /// Issues a memory barrier that ensures consistency between the public and
    /// private copies of the window memory on the local process. This is a
    /// purely local operation — it does not require a surrounding lock epoch
    /// and does not communicate with any remote process.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_sync"), .. }` if the MPI
    /// call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// // sync is valid outside any epoch
    /// win.sync().unwrap();
    /// ```
    pub fn sync(&self) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. `MPI_Win_sync`
        // is a local operation and is valid at any point after window creation.
        let ret = unsafe { ffi::ferrompi_win_sync(self.win_handle) };
        Error::check_with_op(ret, "win_sync")
    }
}

impl<T: MpiDatatype> Win<'_, T> {
    /// One-sided write: copy `origin` into the remote rank's window memory.
    ///
    /// Wraps `MPI_Put`. The operation is posted to the network immediately, but
    /// **the write does not complete until the surrounding epoch closes**. You
    /// must call [`Win::fence`] (or a matching `complete`/`unlock`) after the
    /// put to guarantee the data is visible at the target.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local buffer whose contents are written to the target.
    /// * `target_rank` — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to write at the target.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] — if `MPI_Put` fails (e.g., `MPI_ERR_RANK` for an
    ///   invalid `target_rank`).
    ///
    /// # Safety Contract
    ///
    /// This method takes `&self` (not `&mut self`), so the borrow checker
    /// allows multiple puts in the same epoch. However, MPI requires that:
    ///
    /// 1. `Win::put` is called **inside** an active access epoch (fence, start,
    ///    or lock). Calling it outside an epoch is undefined per the MPI
    ///    standard.
    /// 2. The `origin` buffer must remain valid and unmodified until the epoch
    ///    closes. The Rust borrow (`&[T]`) ends when the method returns, but
    ///    the underlying data is read by the network after the call. **Do not
    ///    mutate or drop the buffer before calling the closing `fence`/`unlock`.**
    ///
    /// The borrow checker cannot enforce the epoch-completion lifetime — this
    /// contract is a documented invariant.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 4).unwrap();
    ///
    /// // Open fence epoch on all ranks
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0 puts four elements into rank 1's window at displacement 0
    /// if world.rank() == 0 {
    ///     let buf = [1.0f64, 2.0, 3.0, 4.0];
    ///     win.put(&buf, 1, 0, buf.len() as i64).unwrap();
    /// }
    ///
    /// // Close epoch — put completes here
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// ```
    pub fn put(
        &self,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
    ) -> Result<()> {
        let (p, n, dt) = buf(origin);
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is read by MPI until the epoch closes; the signature does not
        // tie the buffer to the epoch, so keeping it alive and unmodified until then is
        // the caller's documented obligation. `target_rank`, `target_disp`, and
        // `target_count` are passed through unchecked; a value outside the target window
        // is an MPI-level error or an out-of-bounds remote access, and this function does
        // not check it.
        let ret = unsafe {
            ffi::ferrompi_put(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "put")
    }

    /// Request-returning one-sided write: post a put and return a [`Request`]
    /// that completes when the **local** origin buffer is safe to reuse.
    ///
    /// Wraps `MPI_Rput`. This is the request-returning variant of [`Win::put`].
    /// The returned `Request` signals **local** buffer completion — once
    /// `request.wait()` returns, the `origin` slice can be safely overwritten
    /// or dropped. **Remote completion is not guaranteed by the request**: the
    /// remote rank does not observe the written data until the surrounding epoch
    /// closes (fence / complete / unlock).
    ///
    /// # Local vs. Remote completion
    ///
    /// - `request.wait()` → local buffer is free to reuse.
    /// - Epoch close (fence / unlock / complete) → remote rank observes the
    ///   write.
    ///
    /// Both are required for the full operation to be visible end-to-end.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local source buffer (read-only during the operation).
    ///   The buffer must remain valid and unmodified until the returned
    ///   `Request` is waited on.
    /// * `target_rank`  — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to write at the target.
    ///
    /// # Returns
    ///
    /// A [`Request`] handle that completes when the local origin buffer is safe
    /// to reuse. Call [`Request::wait`] to block until local completion.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] — if `MPI_Rput` fails (e.g., `MPI_ERR_RANK` for an
    ///   invalid `target_rank`).
    /// * [`Error::Mpi { class: MpiErrorClass::Other }`] — if the internal
    ///   request table is exhausted.
    ///
    /// # Epoch Requirement
    ///
    /// `rput` must be called inside an active access epoch (passive-target lock,
    /// fence, or PSCW). Calling it outside an epoch is undefined per the MPI
    /// standard.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<i32>::allocate(&world, 4).unwrap();
    ///
    /// if world.rank() == 0 {
    ///     // Open a passive-target epoch on rank 1
    ///     let _guard = win.lock(LockType::Exclusive, 1).unwrap();
    ///     let buf = [10i32, 20, 30, 40];
    ///     // Post the put; local buf is safe to reuse once req.wait() returns
    ///     let req = win.rput(&buf, 1, 0, buf.len() as i64).unwrap();
    ///     req.wait().unwrap(); // local buffer safe to reuse after this
    ///     // Remote write completes when _guard drops (MPI_Win_unlock)
    /// }
    /// // Rank 1 can read its local memory after a barrier here
    /// ```
    pub fn rput(
        &self,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
    ) -> Result<Request> {
        let (p, n, dt) = buf(origin);
        let mut request_handle: i64 = 0;
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is read by MPI until the returned `Request` completes; the
        // signature does not tie the buffer to the `Request`, so keeping it alive and
        // unmodified until then is the caller's documented obligation. `target_rank`,
        // `target_disp`, and `target_count` are passed through unchecked; a value outside
        // the target window is an MPI-level error or an out-of-bounds remote access, and
        // this function does not check it.
        let ret = unsafe {
            ffi::ferrompi_rput(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "rput")?;
        Ok(Request::new(request_handle))
    }

    /// One-sided read: copy data from a remote rank's window memory into
    /// `origin`.
    ///
    /// Wraps `MPI_Get`. The operation is posted to the network immediately, but
    /// **the data in `origin` is not valid until the surrounding epoch closes**.
    /// You must call [`Win::fence`] (or a matching `complete`/`unlock`) after
    /// the get to guarantee the read is complete.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local destination buffer filled by the remote read.
    /// * `target_rank`  — Source rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to read from the target.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] — if `MPI_Get` fails (e.g., `MPI_ERR_RANK` for an
    ///   invalid `target_rank`).
    ///
    /// # Safety Contract
    ///
    /// This method takes `&self` (not `&mut self`), so the borrow checker
    /// allows multiple gets in the same epoch. However, MPI requires that:
    ///
    /// 1. `Win::get` is called **inside** an active access epoch (fence, start,
    ///    or lock). Calling it outside an epoch is undefined per the MPI
    ///    standard.
    /// 2. The `origin` buffer must remain valid and not be read until the epoch
    ///    closes. The Rust borrow (`&mut [T]`) ends when the method returns, but
    ///    the underlying data is written by the network after the call. **Do not
    ///    read or drop the buffer before calling the closing `fence`/`unlock`.**
    ///
    /// The borrow checker cannot enforce the epoch-completion lifetime — this
    /// contract is a documented invariant.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 4).unwrap();
    ///
    /// // Open fence epoch on all ranks
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0 reads four elements from rank 1's window at displacement 0
    /// if world.rank() == 0 {
    ///     let mut buf = [0.0f64; 4];
    ///     win.get(&mut buf, 1, 0, 4).unwrap();
    ///     // buf is NOT yet valid here — read after the closing fence
    /// }
    ///
    /// // Close epoch — get completes here, buf is now valid
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// ```
    pub fn get(
        &self,
        origin: &mut [T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
    ) -> Result<()> {
        let (p, n, dt) = buf_mut(origin);
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is written by MPI until the epoch closes; the signature does
        // not tie the buffer to the epoch, so keeping it alive and untouched until then is
        // the caller's documented obligation. `target_rank`, `target_disp`, and
        // `target_count` are passed through unchecked; a value outside the target window
        // is an MPI-level error or an out-of-bounds remote access, and this function does
        // not check it.
        let ret = unsafe {
            ffi::ferrompi_get(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "get")
    }

    /// Request-returning one-sided read: copy data from a remote rank's window
    /// memory into `origin` and return a [`Request`] that completes when the
    /// local buffer is ready to be read.
    ///
    /// Wraps `MPI_Rget`. Unlike [`Win::get`], the caller does **not** need to
    /// close the epoch to observe the data locally — when `request.wait()`
    /// returns, `origin` already contains the fetched values. Remote-side
    /// completion is implicit because this is a read operation.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local destination buffer filled by the remote read.
    ///   The buffer must remain valid until the returned `Request` is waited on.
    /// * `target_rank`  — Source rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to read from the target.
    ///
    /// # Returns
    ///
    /// A [`Request`] handle. When [`Request::wait`] returns, `origin` contains
    /// the data fetched from the remote rank. This is the key difference from
    /// [`Win::rput`]: local completion here means the *data is already in
    /// `origin`*, not merely that the local buffer is safe to reuse.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] — if `MPI_Rget` fails (e.g., `MPI_ERR_RANK` for an
    ///   invalid `target_rank`).
    /// * [`Error::Mpi { class: MpiErrorClass::Other }`] — if the internal
    ///   request table is exhausted.
    ///
    /// # Epoch Requirement
    ///
    /// `rget` must be called inside an active access epoch (passive-target lock,
    /// fence, or PSCW). Calling it outside an epoch is undefined per the MPI
    /// standard.
    ///
    /// # Cancellation
    ///
    /// Calling `request.cancel()` on an `Rget` request is undefined behavior
    /// in MPI. The MPI standard discourages cancelling RMA requests.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<i32>::allocate(&world, 4).unwrap();
    ///
    /// if world.rank() == 0 {
    ///     // Open a shared passive-target epoch on rank 1
    ///     let _guard = win.lock(LockType::Shared, 1).unwrap();
    ///     let mut buf = [0i32; 4];
    ///     // Post the get; buf contains the data once req.wait() returns
    ///     let count = buf.len() as i64;
    ///     let req = win.rget(&mut buf, 1, 0, count).unwrap();
    ///     req.wait().unwrap(); // buf is valid and ready to read after this
    ///     assert_eq!(buf, [0i32; 4]);
    ///     // _guard drops here (MPI_Win_unlock)
    /// }
    /// ```
    pub fn rget(
        &self,
        origin: &mut [T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
    ) -> Result<Request> {
        let (p, n, dt) = buf_mut(origin);
        let mut request_handle: i64 = 0;
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is written by MPI until the returned `Request` completes; the
        // signature does not tie the buffer to the `Request`, so keeping it alive and
        // untouched until then is the caller's documented obligation. `target_rank`,
        // `target_disp`, and `target_count` are passed through unchecked; a value outside
        // the target window is an MPI-level error or an out-of-bounds remote access, and
        // this function does not check it.
        let ret = unsafe {
            ffi::ferrompi_rget(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "rget")?;
        Ok(Request::new(request_handle))
    }

    /// One-sided reduce-accumulate: combine data from a local origin buffer
    /// with a remote rank's window memory using a reduction operation.
    ///
    /// Wraps `MPI_Accumulate`. The operation is posted immediately, but
    /// **the effect at the target is not visible until the surrounding epoch
    /// closes**. You must call [`Win::fence`] (or a matching `complete`/`unlock`)
    /// after the accumulate to guarantee the operation is complete.
    ///
    /// Any [`ReduceOp`] variant is accepted, including
    /// [`ReduceOp::Replace`] (semantically equivalent to `MPI_Put`) and
    /// [`ReduceOp::NoOp`] (leaves the target unchanged but still participates
    /// in epoch synchronization). Both require the `rma` feature.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local source buffer (read-only).
    /// * `target_rank`  — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to accumulate at the target.
    /// * `op`           — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] with class `MpiErrorClass::Op` — if `op` is not valid
    ///   for the element type (e.g., `BitwiseOr` on `f64`).
    /// * [`Error::Mpi`] — if `MPI_Accumulate` fails for any other reason.
    ///
    /// # Safety Contract
    ///
    /// This method takes `&self` (not `&mut self`), so the borrow checker
    /// allows multiple accumulates in the same epoch. However, MPI requires:
    ///
    /// 1. `Win::accumulate` is called **inside** an active access epoch (fence,
    ///    start, or lock). Calling it outside an epoch is undefined per the
    ///    MPI standard.
    /// 2. The `origin` buffer must remain valid and unmodified until the epoch
    ///    closes.
    ///
    /// The borrow checker cannot enforce the epoch-completion lifetime — this
    /// contract is a documented invariant.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<f64>::allocate(&world, 4).unwrap();
    ///
    /// // Rank 1 initialises its window
    /// if world.rank() == 1 {
    ///     win.local_slice_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// }
    ///
    /// // Open fence epoch on all ranks
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0 adds [10, 20, 30, 40] to rank 1's window
    /// if world.rank() == 0 {
    ///     let buf = [10.0f64, 20.0, 30.0, 40.0];
    ///     win.accumulate(&buf, 1, 0, buf.len() as i64, ReduceOp::Sum).unwrap();
    /// }
    ///
    /// // Close epoch — accumulate completes here
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// // Rank 1's window is now [11.0, 22.0, 33.0, 44.0]
    /// ```
    pub fn accumulate(
        &self,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
        op: ReduceOp,
    ) -> Result<()> {
        let (p, n, dt) = buf(origin);
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is read by MPI until the epoch closes; the signature does not
        // tie the buffer to the epoch, so keeping it alive and unmodified until then is
        // the caller's documented obligation. `target_rank`, `target_disp`, and
        // `target_count` are passed through unchecked; a value outside the target window
        // is an MPI-level error or an out-of-bounds remote access, and this function does
        // not check it. `op as i32` is a valid `ReduceOp` discriminant that the shim maps
        // to an `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_accumulate(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                op as i32,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "accumulate")
    }

    /// Request-returning one-sided reduce: combine data from a local origin
    /// buffer with a remote rank's window memory using a reduction operation,
    /// returning a [`Request`] that completes when the local `origin` buffer is
    /// safe to reuse.
    ///
    /// Wraps `MPI_Raccumulate`. Unlike [`Win::accumulate`], local completion
    /// (i.e., the `origin` buffer being safe to reuse) is signaled by the
    /// returned [`Request`] rather than the epoch boundary. **Remote-side
    /// completion** (visibility at the target) still requires the surrounding
    /// epoch to close (fence, `complete`, or `unlock`). This is the same
    /// local-completion contract as [`Win::rput`].
    ///
    /// Any [`ReduceOp`] variant is accepted, including [`ReduceOp::Replace`]
    /// (semantically equivalent to `MPI_Rput` with request semantics) and
    /// [`ReduceOp::NoOp`] (leaves the target unchanged but still participates
    /// in epoch synchronization). Both require the `rma` feature.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local source buffer (read-only). The buffer must
    ///   remain valid until the returned [`Request`] is waited on.
    /// * `target_rank`  — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to accumulate at the target.
    /// * `op`           — Reduction operation to apply.
    ///
    /// # Returns
    ///
    /// A [`Request`] handle. When [`Request::wait`] returns, the local
    /// `origin` buffer is safe to reuse or drop. The accumulation may not
    /// yet be visible at the target — close the epoch to guarantee remote
    /// visibility.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` does not fit in `i64`.
    /// * [`Error::Mpi`] with class `MpiErrorClass::Op` — if `op` is not valid
    ///   for the element type (e.g., `BitwiseOr` on `f64`).
    /// * [`Error::Mpi { class: MpiErrorClass::Other }`] — if the internal
    ///   request table is exhausted.
    /// * [`Error::Mpi`] — if `MPI_Raccumulate` fails for any other reason.
    ///
    /// # Epoch Requirement
    ///
    /// `raccumulate` must be called inside an active access epoch
    /// (passive-target lock, fence, or PSCW). Calling it outside an epoch is
    /// undefined per the MPI standard.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, ReduceOp, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 4).unwrap();
    ///
    /// if world.rank() == 1 {
    ///     win.local_slice_mut().copy_from_slice(&[1, 2, 3, 4]);
    /// }
    ///
    /// win.fence(ferrompi::WinFenceAssert::default()).unwrap();
    ///
    /// if world.rank() == 0 {
    ///     let _guard = win.lock(LockType::Exclusive, 1).unwrap();
    ///     let buf = [10i32, 20, 30, 40];
    ///     let req = win.raccumulate(&buf, 1, 0, buf.len() as i64, ReduceOp::Sum).unwrap();
    ///     req.wait().unwrap(); // origin buffer safe to reuse after this
    ///     // _guard drops here (MPI_Win_unlock) — remote visibility guaranteed
    /// }
    /// // Rank 1's window is now [11, 22, 33, 44]
    /// ```
    pub fn raccumulate(
        &self,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
        op: ReduceOp,
    ) -> Result<Request> {
        let (p, n, dt) = buf(origin);
        let mut request_handle: i64 = 0;
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is read by MPI until the returned `Request` completes; the
        // signature does not tie the buffer to the `Request`, so keeping it alive and
        // unmodified until then is the caller's documented obligation. `target_rank`,
        // `target_disp`, and `target_count` are passed through unchecked; a value outside
        // the target window is an MPI-level error or an out-of-bounds remote access, and
        // this function does not check it. `op as i32` is a valid `ReduceOp` discriminant
        // that the shim maps to an `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_raccumulate(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                op as i32,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "raccumulate")?;
        Ok(Request::new(request_handle))
    }

    /// Atomic read-modify-write: fetch the pre-update remote value into
    /// `result`, then apply `op(origin, target_old)` at the target — all
    /// atomically with respect to other RMA operations in the same epoch.
    ///
    /// Wraps `MPI_Get_accumulate`. The operation is posted immediately, but
    /// **both the fetch and the accumulation complete only when the surrounding
    /// epoch closes**. You must call [`Win::fence`] (or a matching
    /// `complete`/`unlock`) before reading `result`.
    ///
    /// `ReduceOp::NoOp` is the canonical "atomic get" pattern: `result` receives
    /// the target's current value and the target memory is left unchanged.
    /// `ReduceOp::Replace` replaces the target value and also returns the
    /// pre-update value in `result`.
    ///
    /// # Arguments
    ///
    /// * `origin`       — Local input buffer (read-only during the epoch).
    /// * `result`       — Local destination for the pre-update remote value
    ///   (written by MPI when the epoch closes).
    /// * `target_rank`  — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp`  — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `target_count` — Number of `T` elements to read-modify-write at the
    ///   target.
    /// * `op`           — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidBuffer`] — if `origin.len()` or `result.len()` does
    ///   not fit in `i64`.
    /// * [`Error::Mpi`] with class `MpiErrorClass::Op` — if `op` is not valid
    ///   for the element type (e.g., `BitwiseOr` on `f64`).
    /// * [`Error::Mpi`] — if `MPI_Get_accumulate` fails for any other reason.
    ///
    /// # Safety Contract
    ///
    /// This method takes `&self` (not `&mut self`), so the borrow checker allows
    /// multiple RMA operations in the same epoch. However, MPI requires:
    ///
    /// 1. `Win::get_accumulate` is called **inside** an active access epoch
    ///    (fence, start, or lock). Calling it outside an epoch is undefined per
    ///    the MPI standard.
    /// 2. The `origin` buffer must remain valid and unmodified until the epoch
    ///    closes.
    /// 3. The `result` buffer must remain valid and not be read until the epoch
    ///    closes (its contents are undefined before then).
    /// 4. `origin` and `result` **must not alias** in memory. Although
    ///    `MPI_Get_accumulate` populates the local buffers in a well-defined
    ///    order, Rust's `&[T]` / `&mut [T]` enforce non-aliasing at the type
    ///    level — passing overlapping slices would violate the borrow rules.
    ///
    /// The borrow checker cannot enforce the epoch-completion lifetime — these
    /// contracts are documented invariants.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 4).unwrap();
    ///
    /// // Rank 1 initialises its window
    /// if world.rank() == 1 {
    ///     win.local_slice_mut().copy_from_slice(&[10, 20, 30, 40]);
    /// }
    ///
    /// // Open fence epoch on all ranks
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0 atomically fetches rank 1's values and adds [1, 2, 3, 4]
    /// if world.rank() == 0 {
    ///     let origin = [1i32, 2, 3, 4];
    ///     let mut result = [0i32; 4];
    ///     win.get_accumulate(&origin, &mut result, 1, 0, 4, ReduceOp::Sum).unwrap();
    /// }
    ///
    /// // Close epoch — get_accumulate completes here
    /// win.fence(WinFenceAssert::default()).unwrap();
    /// // Rank 0's result == [10, 20, 30, 40] (pre-update)
    /// // Rank 1's window == [11, 22, 33, 44] (post-update)
    /// ```
    pub fn get_accumulate(
        &self,
        origin: &[T],
        result: &mut [T],
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
        op: ReduceOp,
    ) -> Result<()> {
        let (o_ptr, o_n, dt) = buf(origin);
        let (r_ptr, r_n, _) = buf_mut(result);
        // SAFETY: the call must be inside an active access epoch, which the caller must
        // ensure. `origin` is read by MPI and `result` is written by MPI until the epoch
        // closes; the signature does not tie either buffer to the epoch, so keeping
        // `origin` alive and unmodified, and `result` alive and unread, until then is the
        // caller's documented obligation. `origin` (`&[T]`) and `result` (`&mut [T]`)
        // cannot alias. `target_rank`, `target_disp`, and `target_count` are passed
        // through unchecked; a value outside the target window is an MPI-level error or
        // an out-of-bounds remote access, and this function does not check it.
        // `op as i32` is a valid `ReduceOp` discriminant that the shim maps to an
        // `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_get_accumulate(
                o_ptr,
                o_n,
                dt,
                r_ptr,
                r_n,
                dt,
                target_rank,
                target_disp,
                target_count,
                dt,
                op as i32,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "get_accumulate")
    }

    /// Single-element atomic fetch-and-update: reads the current value at
    /// `(target_rank, target_disp)` into the returned [`PendingFetchResult<T>`],
    /// then applies `op(origin, target_old)` at the target — both atomically
    /// with respect to other RMA operations in the same epoch.
    ///
    /// Wraps `MPI_Fetch_and_op`. Restricted to predefined MPI datatypes only
    /// (no derived or custom datatypes); the `T: MpiDatatype` bound enforces
    /// this at compile time.
    ///
    /// **Important:** `MPI_Fetch_and_op` only *initiates* the operation. The
    /// returned `PendingFetchResult<T>` is populated with the pre-update remote
    /// value **only after the surrounding epoch closes** (via `fence`,
    /// `complete`, or `unlock`). Call [`PendingFetchResult::resolve`] **only
    /// after closing the epoch** — reading it earlier is undefined behaviour per
    /// MPI-3 §11.6.
    ///
    /// # Arguments
    ///
    /// * `origin`      — The value to combine with the remote element.
    /// * `target_rank` — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `op`          — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::Mpi`] with class `MpiErrorClass::Op` — if `op` is not valid
    ///   for the element type (e.g., `BitwiseOr` on `f64`).
    /// * [`Error::Mpi`] — if `MPI_Fetch_and_op` fails for any other reason.
    ///
    /// # Safety Contract
    ///
    /// 1. This call must be made **inside** an active access epoch (fence,
    ///    start, or lock). Calling it outside an epoch is undefined per the
    ///    MPI standard.
    /// 2. [`PendingFetchResult::resolve`] must not be called until the epoch
    ///    closes.
    ///
    /// The borrow checker cannot enforce the epoch-completion constraint — this
    /// contract is a documented invariant on [`PendingFetchResult::resolve`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 1).unwrap();
    ///
    /// // Rank 1 initialises its window slot 0 to 100
    /// if world.rank() == 1 {
    ///     win.local_slice_mut()[0] = 100;
    /// }
    ///
    /// // Open fence epoch on all ranks
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0 atomically adds 1 and fetches the old value (operation initiated)
    /// let pending = if world.rank() == 0 {
    ///     Some(win.fetch_and_op(1, 1, 0, ReduceOp::Sum).unwrap())
    /// } else {
    ///     None
    /// };
    ///
    /// // Close epoch — fetch_and_op completes here
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // SAFETY: epoch is closed; result buffer is now populated.
    /// if let Some(p) = pending {
    ///     let old = unsafe { p.resolve() };
    ///     assert_eq!(old, 100); // pre-update value; window slot is now 101
    /// }
    /// ```
    pub fn fetch_and_op(
        &self,
        origin: T,
        target_rank: i32,
        target_disp: i64,
        op: ReduceOp,
    ) -> Result<PendingFetchResult<T>> {
        use std::mem::MaybeUninit;
        // Box the origin and result so they have heap-stable addresses
        // that survive across the epoch.  MPI_Fetch_and_op only *initiates*
        // the operation; the actual read of origin and write to result may
        // not happen until the closing fence/complete/unlock.  Stack-local
        // storage would be reused before then (function returns before the
        // epoch closes), causing MPI to read/write into freed memory — UB
        // observed as zeros on MPICH 4.2.x and as latent corruption on
        // other implementations.  See PendingFetchResult docs for the
        // lifetime contract.
        let origin_box: Box<T> = Box::new(origin);
        // SAFETY: `Box::into_raw` never returns a null pointer.
        let result_ptr: NonNull<MaybeUninit<T>> =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(MaybeUninit::uninit()))) };
        // SAFETY: `origin_box` points to a heap allocation moved into the returned
        // `PendingFetchResult`; MPI reads through it until the epoch closes.
        // `result_ptr` is a raw pointer that owns its allocation (obtained from
        // `Box::into_raw` above, never a cast from a shared borrow); MPI writes through
        // it until the epoch closes. The caller must keep the returned
        // `PendingFetchResult` alive and call `resolve` only after the epoch closes —
        // `resolve` is `unsafe` for that reason. `result_ptr`'s allocation is freed
        // exactly once by `PendingFetchResult`'s `Drop`, on both the `resolve` and
        // drop-without-resolve paths. `T::TAG` matches T's memory layout per the
        // `MpiDatatype` invariant. `op as i32` is a valid `ReduceOp` discriminant that
        // the shim maps to an `MPI_Op`. `target_rank`, `target_disp`, and `win_handle`
        // are passed through unchecked.
        let ret = unsafe {
            ffi::ferrompi_fetch_and_op(
                (origin_box.as_ref() as *const T).cast::<std::ffi::c_void>(),
                result_ptr.as_ptr().cast::<std::ffi::c_void>(),
                T::TAG as i32,
                target_rank,
                target_disp,
                op as i32,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "fetch_and_op").map_err(|e| {
            // SAFETY: `result_ptr` was obtained from `Box::into_raw` above and has not
            // been freed; reconstructing the `Box` here and dropping it frees the
            // allocation exactly once on this error path.
            unsafe { drop(Box::from_raw(result_ptr.as_ptr())) };
            e
        })?;
        Ok(PendingFetchResult {
            _origin: origin_box,
            _compare: None,
            result: result_ptr,
        })
    }
}

impl<'a, T: crate::AtomicMpiDatatype + MpiDatatype> Win<'a, T> {
    /// Single-element atomic compare-and-swap: conditionally replace a remote
    /// element if it matches an expected value, returning a
    /// [`PendingFetchResult<T>`] that holds the pre-CAS remote value once the
    /// epoch closes.
    ///
    /// Wraps `MPI_Compare_and_swap`. Atomically reads the remote element at
    /// `(target_rank, target_disp)` and compares it to `compare`. If they are
    /// equal, the remote element is replaced with `origin`. The
    /// `PendingFetchResult<T>` is **always** the pre-CAS remote value,
    /// regardless of whether the swap succeeded — the caller determines swap
    /// success by calling [`resolve`](PendingFetchResult::resolve) after the
    /// epoch closes and comparing with `compare`.
    ///
    /// This method is restricted to types that implement [`AtomicMpiDatatype`]:
    /// `i32`, `i64`, `u32`, `u64`, and `u8`. Floating-point types (`f32`,
    /// `f64`) are excluded because the MPI 4.1 standard does not require
    /// implementations to support floating-point CAS (section 12.5.4).
    ///
    /// [`AtomicMpiDatatype`]: crate::AtomicMpiDatatype
    ///
    /// **Important:** `MPI_Compare_and_swap` only *initiates* the operation.
    /// The `PendingFetchResult<T>` is populated with the pre-CAS remote value
    /// **only after the surrounding epoch closes** (via `fence`, `complete`,
    /// or `unlock`). Call [`PendingFetchResult::resolve`] **only after closing
    /// the epoch** — reading it earlier is undefined behaviour per MPI-3 §11.6.
    ///
    /// # Arguments
    ///
    /// * `origin`      — The new value to swap in if `compare` matches the
    ///   remote element.
    /// * `compare`     — The expected current remote value. The swap occurs
    ///   only if `remote_value == compare`.
    /// * `target_rank` — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Returns
    ///
    /// `Ok(PendingFetchResult<T>)` — call [`resolve`](PendingFetchResult::resolve)
    /// after the epoch closes to obtain the pre-CAS value and check whether
    /// the swap succeeded:
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinFenceAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<i32>::allocate(&world, 1).unwrap();
    /// # win.fence(WinFenceAssert::default()).unwrap();
    /// let expected = 100i32;
    /// let new_val  = 200i32;
    /// let pending = win.compare_and_swap(new_val, expected, 1, 0).unwrap();
    /// // close epoch first
    /// # win.fence(WinFenceAssert::default()).unwrap();
    /// // SAFETY: epoch is closed.
    /// let old = unsafe { pending.resolve() };
    /// let swapped = old == expected;
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::Mpi`] — if `MPI_Compare_and_swap` fails (e.g., invalid rank
    ///   or displacement, or the MPI version is < 3).
    ///
    /// # Safety Contract
    ///
    /// 1. This call must be made **inside** an active access epoch (fence,
    ///    start, or lock). Calling it outside an epoch is undefined per the
    ///    MPI standard.
    /// 2. [`PendingFetchResult::resolve`] must not be called until the epoch
    ///    closes.
    ///
    /// The borrow checker cannot enforce the epoch-completion constraint — this
    /// contract is a documented invariant on [`PendingFetchResult::resolve`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 1).unwrap();
    ///
    /// // Rank 1 initialises its window slot 0 to 100
    /// if world.rank() == 1 {
    ///     win.local_slice_mut()[0] = 100;
    /// }
    ///
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // Rank 0: swap in 200 only if the current value is 100 (initiated here)
    /// let pending = if world.rank() == 0 {
    ///     Some(win.compare_and_swap(200, 100, 1, 0).unwrap())
    /// } else {
    ///     None
    /// };
    ///
    /// // Close epoch — compare_and_swap completes here
    /// win.fence(WinFenceAssert::default()).unwrap();
    ///
    /// // SAFETY: epoch is closed; result buffer is now populated.
    /// if let Some(p) = pending {
    ///     let old = unsafe { p.resolve() };
    ///     assert_eq!(old, 100); // pre-CAS; slot 0 is now 200
    /// }
    /// ```
    pub fn compare_and_swap(
        &self,
        origin: T,
        compare: T,
        target_rank: i32,
        target_disp: i64,
    ) -> Result<PendingFetchResult<T>> {
        use std::mem::MaybeUninit;
        // Box origin, compare, and result for heap-stable addresses that
        // survive across the epoch.  MPI_Compare_and_swap is non-blocking
        // at the MPI level — the actual atomic CAS may not happen until
        // the closing fence/complete/unlock, by which time stack-local
        // storage would have been reused.  See PendingFetchResult docs.
        let origin_box: Box<T> = Box::new(origin);
        let compare_box: Box<T> = Box::new(compare);
        // SAFETY: `Box::into_raw` never returns a null pointer.
        let result_ptr: NonNull<MaybeUninit<T>> =
            unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(MaybeUninit::uninit()))) };
        // SAFETY: `origin_box` and `compare_box` point to heap allocations moved into
        // the returned `PendingFetchResult`; MPI reads through them until the epoch
        // closes. `result_ptr` is a raw pointer that owns its allocation (obtained from
        // `Box::into_raw` above, never a cast from a shared borrow); MPI writes through
        // it until the epoch closes. The caller must keep the returned
        // `PendingFetchResult` alive and call `resolve` only after the epoch closes —
        // `resolve` is `unsafe` for that reason. `result_ptr`'s allocation is freed
        // exactly once by `PendingFetchResult`'s `Drop`, on both the `resolve` and
        // drop-without-resolve paths. `T::TAG` matches T's memory layout per the
        // `MpiDatatype` invariant; `AtomicMpiDatatype` restricts T to the integer and
        // byte types MPI supports for compare-and-swap (MPI 4.1 §12.5.4). `target_rank`,
        // `target_disp`, and `win_handle` are passed through unchecked.
        let ret = unsafe {
            ffi::ferrompi_compare_and_swap(
                (origin_box.as_ref() as *const T).cast::<std::ffi::c_void>(),
                (compare_box.as_ref() as *const T).cast::<std::ffi::c_void>(),
                result_ptr.as_ptr().cast::<std::ffi::c_void>(),
                T::TAG as i32,
                target_rank,
                target_disp,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "compare_and_swap").map_err(|e| {
            // SAFETY: `result_ptr` was obtained from `Box::into_raw` above and has not
            // been freed; reconstructing the `Box` here and dropping it frees the
            // allocation exactly once on this error path.
            unsafe { drop(Box::from_raw(result_ptr.as_ptr())) };
            e
        })?;
        Ok(PendingFetchResult {
            _origin: origin_box,
            _compare: Some(compare_box),
            result: result_ptr,
        })
    }
}

impl<T: MpiDatatype> Drop for Win<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // ferrompi_win_create or ferrompi_win_allocate. It has not been freed
        // yet because Drop is only called once. MPI_Win_free leaves the user
        // buffer alone for Win::create (WinKind::Created) and frees the
        // MPI-allocated buffer for Win::allocate (WinKind::Allocated) per the
        // MPI standard — the C layer handles this distinction correctly.
        unsafe { ffi::ferrompi_win_free(self.win_handle) };
    }
}

// Win<'_, T> is intentionally NOT Send or Sync. MPI windows have specific
// thread-safety rules that depend on the MPI thread level, and incorrect
// cross-thread usage can cause data corruption or MPI errors.

/// RAII guard for a single-rank window lock.
///
/// Created by [`SharedWindow::lock()`]. When dropped, the lock on the
/// specified rank is automatically released via `MPI_Win_unlock`.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, SharedWindow, LockType};
/// # let mpi = Mpi::init().unwrap();
/// # let node = mpi.world().split_shared().unwrap();
/// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
/// {
///     let guard = win.lock(LockType::Shared, 0).unwrap();
///     let remote = win.remote_slice(0).unwrap();
///     // Ensure all operations are completed
///     guard.flush().unwrap();
/// } // Lock released here
/// ```
pub struct LockGuard<'a, T: MpiDatatype> {
    window: &'a SharedWindow<T>,
    rank: i32,
}

impl<T: MpiDatatype> LockGuard<'_, T> {
    /// Flush pending RMA operations to the locked rank.
    ///
    /// Ensures that all preceding RMA operations (put, get, accumulate)
    /// issued to the locked rank are completed at the target.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush operation fails.
    pub fn flush(&self) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and the rank was locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush(self.rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for LockGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and the rank was locked in the constructor. This unlock matches
        // the lock call that created this guard.
        unsafe { ffi::ferrompi_win_unlock(self.rank, self.window.win_handle) };
    }
}

/// RAII guard for a lock-all window lock.
///
/// Created by [`SharedWindow::lock_all()`]. When dropped, all locks
/// are automatically released via `MPI_Win_unlock_all`.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, SharedWindow};
/// # let mpi = Mpi::init().unwrap();
/// # let node = mpi.world().split_shared().unwrap();
/// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
/// {
///     let guard = win.lock_all().unwrap();
///     // Access shared memory from any rank
///     guard.flush_all().unwrap();
/// } // All locks released here
/// ```
pub struct LockAllGuard<'a, T: MpiDatatype> {
    window: &'a SharedWindow<T>,
}

impl<T: MpiDatatype> LockAllGuard<'_, T> {
    /// Flush all pending RMA operations to all ranks.
    ///
    /// Ensures that all preceding RMA operations issued to any rank
    /// in the window are completed at their respective targets.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush_all operation fails.
    pub fn flush_all(&self) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush_all(self.window.win_handle) };
        Error::check_with_op(ret, "win_flush_all")
    }

    /// Flush pending RMA operations to a specific rank.
    ///
    /// Ensures that all preceding RMA operations issued to the specified
    /// rank are completed at the target.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank to flush (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush operation fails.
    pub fn flush(&self, rank: i32) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush(rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for LockAllGuard<'_, T> {
    fn drop(&mut self) {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor. This unlock_all
        // matches the lock_all call that created this guard.
        unsafe { ffi::ferrompi_win_unlock_all(self.window.win_handle) };
    }
}

// ============================================================================
// WinLockGuard — RAII single-rank lock guard for Win<'a, T>
// ============================================================================

/// RAII guard for a single-rank passive-target lock on a [`Win`].
///
/// Created by [`Win::lock()`]. When dropped, the lock on the specified rank
/// is automatically released via `MPI_Win_unlock`.
///
/// The `'g` lifetime tracks the borrow of the parent `Win<'a, T>`, and `'a`
/// is the buffer lifetime carried by `Win`. The guard cannot outlive the
/// window it was created from.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, Win, LockType};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let win = Win::<f64>::allocate(&world, 8).unwrap();
///
/// {
///     let guard = win.lock(LockType::Shared, 0).unwrap();
///     guard.flush().unwrap();
/// } // MPI_Win_unlock called here
/// ```
pub struct WinLockGuard<'g, 'a, T: MpiDatatype> {
    window: &'g Win<'a, T>,
    rank: i32,
}

impl<T: MpiDatatype> WinLockGuard<'_, '_, T> {
    /// Flush pending RMA operations to the locked rank.
    ///
    /// Ensures that all preceding RMA operations (put, get, accumulate)
    /// issued to the locked rank are completed at the target before this
    /// call returns.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_flush"), .. }` if the MPI
    /// flush operation fails.
    pub fn flush(&self) -> Result<()> {
        // SAFETY: `window.win_handle` is valid — it is borrowed from a live
        // `Win`. `self.rank` was locked in `Win::lock`, so a flush to that
        // rank is legal within this epoch.
        let ret = unsafe { ffi::ferrompi_win_flush(self.rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for WinLockGuard<'_, '_, T> {
    fn drop(&mut self) {
        // SAFETY: `window.win_handle` is valid — it is borrowed from a live
        // `Win`. The rank was locked in `Win::lock`; this unlock matches that
        // lock call. Drop is only called once.
        unsafe { ffi::ferrompi_win_unlock(self.rank, self.window.win_handle) };
    }
}

// ============================================================================
// WinLockAllGuard — RAII lock-all guard for Win<'a, T>
// ============================================================================

/// RAII guard for a lock-all passive-target epoch on a [`Win`].
///
/// Created by [`Win::lock_all()`]. When dropped, all shared locks are
/// automatically released via `MPI_Win_unlock_all`.
///
/// The `'g` lifetime tracks the borrow of the parent `Win<'a, T>`, and `'a`
/// is the buffer lifetime carried by `Win`. The guard cannot outlive the
/// window it was created from.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, Win};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let win = Win::<f64>::allocate(&world, 8).unwrap();
///
/// {
///     let guard = win.lock_all().unwrap();
///     guard.flush_all().unwrap();
/// } // MPI_Win_unlock_all called here
/// ```
pub struct WinLockAllGuard<'g, 'a, T: MpiDatatype> {
    window: &'g Win<'a, T>,
}

impl<T: MpiDatatype> WinLockAllGuard<'_, '_, T> {
    /// Flush all pending RMA operations to all ranks.
    ///
    /// Ensures that all preceding RMA operations issued to any rank in the
    /// window are completed at their respective targets.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_flush_all"), .. }` if the
    /// MPI flush_all operation fails.
    pub fn flush_all(&self) -> Result<()> {
        // SAFETY: `window.win_handle` is valid — it is borrowed from a live
        // `Win`. All ranks were locked in `Win::lock_all`, so a flush_all is
        // legal within this epoch.
        let ret = unsafe { ffi::ferrompi_win_flush_all(self.window.win_handle) };
        Error::check_with_op(ret, "win_flush_all")
    }

    /// Flush pending RMA operations to a specific rank.
    ///
    /// Ensures that all preceding RMA operations issued to `rank` are
    /// completed at the target.
    ///
    /// # Arguments
    ///
    /// * `rank` - The target rank to flush (0-based).
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_flush"), .. }` if the MPI
    /// flush operation fails.
    pub fn flush(&self, rank: i32) -> Result<()> {
        // SAFETY: `window.win_handle` is valid — it is borrowed from a live
        // `Win`. All ranks were locked in `Win::lock_all`, so flushing any
        // individual rank is legal within this epoch.
        let ret = unsafe { ffi::ferrompi_win_flush(rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for WinLockAllGuard<'_, '_, T> {
    fn drop(&mut self) {
        // SAFETY: `window.win_handle` is valid — it is borrowed from a live
        // `Win`. All ranks were locked in `Win::lock_all`; this unlock_all
        // matches that call. Drop is only called once.
        unsafe { ffi::ferrompi_win_unlock_all(self.window.win_handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::{PendingFetchResult, WinFenceAssert, WinPscwAssert};
    use std::ptr::NonNull;

    // -------------------------------------------------------------------------
    // Win<'a, T> unit tests — exercise the type without an MPI runtime
    // -------------------------------------------------------------------------

    #[test]
    fn pending_fetch_result_resolve_and_drop_without_resolve() {
        use std::mem::MaybeUninit;

        fn make(value: i32) -> PendingFetchResult<i32> {
            // SAFETY: `Box::into_raw` never returns a null pointer.
            let result_ptr =
                unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(MaybeUninit::new(value)))) };
            PendingFetchResult {
                _origin: Box::new(0),
                _compare: None,
                result: result_ptr,
            }
        }

        // Resolve path: reads the value, then Drop frees `result` (and `_origin`) exactly once.
        let resolved = make(42);
        // SAFETY: no MPI epoch is involved; `result` was initialised directly above.
        let value = unsafe { resolved.resolve() };
        assert_eq!(value, 42);

        // Drop-without-resolve path: Drop frees `result` (and `_origin`) exactly once.
        let unresolved = make(7);
        drop(unresolved);

        // Compile-time witness: the manual Send/Sync impls restore the auto-trait
        // behavior PendingFetchResult<T> had when `result` was a `Box<MaybeUninit<T>>`.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<PendingFetchResult<i32>>();
        assert_sync::<PendingFetchResult<i32>>();
    }

    // -------------------------------------------------------------------------
    // WinFenceAssert unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn win_fence_assert_or_combines_bits() {
        // Use placeholder constants for the test; the real values come from MPI.
        let a = WinFenceAssert::from_bits_for_test(1);
        let b = WinFenceAssert::from_bits_for_test(2);
        assert_eq!((a | b).bits(), 3);
    }

    #[test]
    fn win_fence_assert_or_assign_combines_bits() {
        let mut a = WinFenceAssert::from_bits_for_test(1);
        a |= WinFenceAssert::from_bits_for_test(4);
        assert_eq!(a.bits(), 5);
    }

    // -------------------------------------------------------------------------
    // WinPscwAssert unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn win_pscw_assert_or_combines_bits() {
        // Use placeholder constants for the test; the real values come from MPI.
        let a = WinPscwAssert::from_bits_for_test(1);
        let b = WinPscwAssert::from_bits_for_test(2);
        assert_eq!((a | b).bits(), 3);
    }

    #[test]
    fn win_pscw_assert_or_assign_combines_bits() {
        let mut a = WinPscwAssert::from_bits_for_test(1);
        a |= WinPscwAssert::from_bits_for_test(4);
        assert_eq!(a.bits(), 5);
    }
}
