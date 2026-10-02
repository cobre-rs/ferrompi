//! The `PersistentRequest` handle returned by the persistent `*_init` constructors.

use crate::error::{Error, Result, FERROMPI_ERR_THREAD_LEVEL};
use crate::ffi;
use crate::rt;

/// A persistent MPI request handle.
///
/// This type represents a persistent collective operation that has been
/// initialized but not yet started. Unlike regular nonblocking operations,
/// persistent operations can be started multiple times.
///
/// # Lifecycle
///
/// 1. Create with `comm.bcast_init()` or similar
/// 2. Start with `start()` or `start_all()`
/// 3. Wait for completion with `wait()`
/// 4. Repeat steps 2-3 as needed
/// 5. Free on drop
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, ReduceOp};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Buffer that will be used for all broadcasts
/// let mut data = vec![0.0f64; 1000];
///
/// // Initialize persistent broadcast (MPI 4.0, or Open MPI 5)
/// let mut persistent = world.bcast_init(&mut data, 0).unwrap();
///
/// // Run many iterations
/// for iter in 0..1000 {
///     // Update data on root
///     if world.rank() == 0 {
///         for (i, x) in data.iter_mut().enumerate() {
///             *x = (iter * 1000 + i) as f64;
///         }
///     }
///
///     // Start the broadcast
///     persistent.start().unwrap();
///
///     // Optionally do other work here...
///
///     // Wait for completion
///     persistent.wait().unwrap();
///
///     // data now contains broadcast result on all ranks
/// }
///
/// // Cleanup happens automatically on drop
/// ```
pub struct PersistentRequest {
    handle: i64,
    active: bool, // True if started but not yet waited
}

impl PersistentRequest {
    /// Create a new persistent request from a raw handle.
    pub(crate) fn new(handle: i64) -> Self {
        PersistentRequest {
            handle,
            active: false,
        }
    }

    /// Get the raw request handle (for advanced use).
    pub fn raw_handle(&self) -> i64 {
        self.handle
    }

    /// Check if this request is currently active (started but not waited).
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Start the persistent operation.
    ///
    /// This initiates the communication. You must call `wait()` before starting
    /// again or accessing the buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if the operation is already active, or
    /// an error if the start fails.
    #[inline]
    pub fn start(&mut self) -> Result<()> {
        if self.active {
            return Err(Error::InvalidState {
                reason: "request is already active",
            });
        }
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table by the *_init constructor
        // that produced this PersistentRequest; self.active is false, so
        // MPI_Start is not being called on an already-active request.
        let ret = unsafe { ffi::ferrompi_start(self.handle) };
        Error::check_with_op(ret, "start")?;
        self.active = true;
        Ok(())
    }

    /// Wait for the operation to complete.
    ///
    /// This blocks until the communication started by `start()` is finished.
    /// After this returns, the buffers can be safely accessed and the operation
    /// can be started again.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails. A wait that MPI ran leaves the
    /// request inactive either way: MPI completed it with that error. A call
    /// rejected before reaching MPI (wrong thread, after finalize, or, in a
    /// debug build at `Serialized`, overlapping another thread's call)
    /// leaves it active.
    #[inline]
    pub fn wait(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        Error::check_with_op(rt::enter(), "wait")?;
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table; self.active was true on
        // entry (checked above), so start() was called and MPI holds an
        // in-flight operation on this handle for ferrompi_wait to complete.
        let ret = unsafe { ffi::ferrompi_wait(self.handle) };
        // A debug build's Serialized overlap check rejects the call before MPI
        // sees it; the request is then still active. Otherwise MPI completed it
        // whatever it returned: it is inactive now, or freed if the library
        // frees failed persistent requests.
        if ret != FERROMPI_ERR_THREAD_LEVEL {
            self.active = false;
        }
        Error::check_with_op(ret, "wait")
    }

    /// Test if the operation has completed without blocking.
    ///
    /// Returns `true` if complete, `false` if still in progress. A failed
    /// `test` that MPI completed also leaves the request inactive.
    #[inline]
    pub fn test(&mut self) -> Result<bool> {
        if !self.active {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table; self.active was true on
        // entry (checked above). flag is a local out-parameter written by
        // ferrompi_test before this function reads it below.
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag) };
        // flag is set when MPI completed the request, even with an error.
        if flag != 0 {
            self.active = false;
        }
        Error::check_with_op(ret, "test")?;
        Ok(flag != 0)
    }

    /// Start multiple persistent operations.
    ///
    /// This is more efficient than starting each operation individually.
    ///
    /// # Errors
    ///
    /// Returns `Err(`[`Error::InvalidState`]`)` without calling MPI if any
    /// request is already active. If MPI reports an error, it may have started
    /// some of the requests; every request is then marked active, so `wait`,
    /// `wait_all` or `Drop` completes whichever did start.
    pub fn start_all(requests: &mut [PersistentRequest]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        if requests.iter().any(|req| req.active) {
            return Err(Error::InvalidState {
                reason: "a request is already active",
            });
        }

        // SAFETY: with_handles provides a valid, contiguous [i64] of the
        // persistent-request handles and a same-length [u8] started buffer,
        // both sized to the count we pass.
        let ret = crate::request::with_handles(
            requests,
            |r| r.handle,
            |handles, started| unsafe {
                ffi::ferrompi_startall(handles.len() as i64, handles.as_ptr(), started.as_mut_ptr())
            },
            |r| r.active = true,
        );
        Error::check_with_op(ret, "startall")
    }

    /// Wait for all persistent operations to complete.
    ///
    /// Whatever the result, every request MPI completed is marked inactive
    /// in place; the others stay active. This is the same policy
    /// [`Request::wait_all`](crate::Request::wait_all) applies.
    /// Inactive requests are skipped, so a request whose earlier `wait` or
    /// `test` failed can stay in the slice.
    ///
    /// On a failed request, the returned error carries that request's own
    /// class and code, and its message ends with `(request N)`, `N` being
    /// its index in `requests`.
    pub fn wait_all(requests: &mut [PersistentRequest]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let mut failed: i64 = -1;
        // SAFETY: with_handles provides a valid, contiguous [i64] of the
        // persistent-request handles and a same-length [u8] done buffer, both
        // sized to the count we pass; failed is a valid stack-allocated i64
        // output parameter.
        let ret = crate::request::with_handles(
            requests,
            |r| if r.active { r.handle } else { -1 },
            |handles, done| unsafe {
                ffi::ferrompi_waitall(
                    handles.len() as i64,
                    handles.as_ptr(),
                    done.as_mut_ptr(),
                    &mut failed,
                )
            },
            |r| r.active = false,
        );
        crate::request::check_batch(ret, "waitall", failed)
    }
}

impl Drop for PersistentRequest {
    /// Complete any in-flight operation, then free the persistent request handle.
    ///
    /// When `self.active` is `true` (i.e., `start()` was called but `wait()` has
    /// not yet returned), this calls `MPI_Wait` before freeing the handle.
    /// **`MPI_Wait` blocks** until the peer operation completes; if the peer is
    /// unreachable, this deadlocks. This two-step sequence upholds the MPI
    /// standard requirement that `MPI_Request_free` must not be called on an
    /// active request.
    ///
    /// See ADR-0004 §"Drop behavior: wait before free" for the full rationale.
    fn drop(&mut self) {
        if !rt::drop_guard("PersistentRequest") {
            return;
        }
        if self.active {
            // SAFETY: self.handle is a valid MPI request handle registered in the
            // C-side request table by the *_init constructor. self.active is true,
            // so start() was called and MPI holds an in-flight operation on this
            // handle. ferrompi_wait calls MPI_Wait which completes the operation
            // and releases the handle's active state before request_free below.
            // Calls the unguarded raw wrapper (not the lifecycle-guarded one):
            // rt::drop_guard above already handles the FFI lifecycle check, so
            // this call must still attempt the wait once reached.
            unsafe { ffi::raw::ferrompi_wait(self.handle) };
        }
        // SAFETY: self.handle is a valid persistent MPI request handle. If it was
        // active, ferrompi_wait above has already completed the operation, so
        // MPI_Request_free is safe to call. If it was inactive, no operation is
        // in flight and MPI_Request_free is unconditionally safe on the handle.
        unsafe { ffi::ferrompi_request_free(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::PersistentRequest;
    use crate::error::Error;
    use std::mem::forget;

    #[test]
    fn start_when_already_active_returns_error() {
        let mut req = PersistentRequest {
            handle: 0,
            active: true,
        };
        let result = req.start();
        assert!(
            matches!(
                result,
                Err(Error::InvalidState {
                    reason: "request is already active"
                })
            ),
            "expected Err(InvalidState), got: {result:?}"
        );
        forget(req);
    }

    #[test]
    fn test_when_inactive_returns_true() {
        let mut req = PersistentRequest::new(0);
        let result = req.test();
        assert!(
            matches!(result, Ok(true)),
            "expected Ok(true), got: {result:?}"
        );
        forget(req);
    }

    #[test]
    fn start_all_empty_slice_returns_ok() {
        let result = PersistentRequest::start_all(&mut []);
        assert!(result.is_ok());
    }

    #[test]
    fn start_all_with_active_request_returns_error() {
        let mut req = PersistentRequest {
            handle: 0,
            active: true,
        };
        let result = PersistentRequest::start_all(std::slice::from_mut(&mut req));
        assert!(
            matches!(
                result,
                Err(Error::InvalidState {
                    reason: "a request is already active"
                })
            ),
            "expected Err(InvalidState), got: {result:?}"
        );
        forget(req);
    }

    #[test]
    fn wait_all_empty_slice_returns_ok() {
        let result = PersistentRequest::wait_all(&mut []);
        assert!(result.is_ok());
    }
}
