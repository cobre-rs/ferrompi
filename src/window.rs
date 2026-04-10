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

use std::ptr::NonNull;

use crate::error::{Error, Result};
use crate::ffi;
use crate::Communicator;
use crate::MpiDatatype;

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
        Error::check(ret)?;

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
        Error::check(ret)?;

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
        Error::check(ret)
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)
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
        Error::check(ret)
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
        Error::check(ret)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_type_equality() {
        assert_eq!(LockType::Exclusive, LockType::Exclusive);
        assert_eq!(LockType::Shared, LockType::Shared);
        assert_ne!(LockType::Exclusive, LockType::Shared);
    }

    #[test]
    fn lock_type_debug() {
        assert_eq!(format!("{:?}", LockType::Exclusive), "Exclusive");
        assert_eq!(format!("{:?}", LockType::Shared), "Shared");
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn lock_type_clone_copy() {
        let original = LockType::Exclusive;
        let copied = original; // Copy
        let cloned = original.clone(); // Clone
        assert_eq!(original, copied);
        assert_eq!(original, cloned);

        let original = LockType::Shared;
        let copied = original; // Copy
        let cloned = original.clone(); // Clone
        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }
}
