//! Request handles for nonblocking MPI operations.

use crate::error::{Error, Result};
use crate::ffi;

/// Element count at or below which request-handle scratch buffers live on the
/// stack. Draining a handful-to-few-dozen in-flight requests on the completion
/// path (halo exchange, ping-pong, progress loops) then incurs no allocator
/// traffic; larger batches fall back to the heap. Mirrors `FERROMPI_REQ_STACK`
/// in `csrc/ferrompi.c`.
const HANDLE_STACK_CAP: usize = 64;

/// Run `f` with the request handles copied into a stack buffer when the batch is
/// small, falling back to a heap `Vec` only for large batches. Removes the
/// per-call `Vec<i64>` allocation on the completion path (PERF-03).
#[inline]
fn with_handles<R>(requests: &[Request], f: impl FnOnce(&mut [i64]) -> R) -> R {
    let len = requests.len();
    if len <= HANDLE_STACK_CAP {
        let mut buf = [0i64; HANDLE_STACK_CAP];
        for (slot, req) in buf[..len].iter_mut().zip(requests) {
            *slot = req.handle;
        }
        f(&mut buf[..len])
    } else {
        let mut buf: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        f(&mut buf)
    }
}

/// Run `f` with a zeroed `i32` index scratch buffer of length `len`,
/// stack-allocated when small (PERF-03). Used for the `*some` output indices.
#[inline]
fn with_index_buf<R>(len: usize, f: impl FnOnce(&mut [i32]) -> R) -> R {
    if len <= HANDLE_STACK_CAP {
        let mut buf = [0i32; HANDLE_STACK_CAP];
        f(&mut buf[..len])
    } else {
        let mut buf = vec![0i32; len];
        f(&mut buf)
    }
}

/// A handle to a nonblocking MPI operation.
///
/// This type represents an in-flight MPI operation. You must call `wait()` or
/// `test()` to complete the operation before the associated buffers can be
/// safely accessed.
///
/// # Safety — Buffer Lifetime
///
/// **The caller must ensure that all buffers passed to the nonblocking operation
/// (e.g., `isend`, `irecv`, `iallreduce`) remain valid and are not moved,
/// reallocated, or dropped until the `Request` is completed (via `wait()` or
/// `test()` returning `true`) or dropped.** MPI holds raw pointers to these
/// buffers; violating this invariant is undefined behavior.
///
/// This cannot currently be enforced by the Rust type system because `Request`
/// does not carry a lifetime parameter tying it to the buffers.
///
/// # Drop Behavior
///
/// **Dropping a `Request` before calling `wait()` will call `MPI_Wait` inside
/// `Drop`, which blocks until the peer operation completes.** If the peer never
/// posts a matching send or receive, the drop call deadlocks permanently.
///
/// This is intentional: blocking in `Drop` is preferred over leaking the MPI
/// request handle or silently cancelling the operation (see
/// [`doc::adr_0004_persistent_collective_approach`](crate::doc::adr_0004_persistent_collective_approach)
/// for the rationale).
///
/// **On any code path that may bypass `wait()` — including early returns via `?`,
/// `break`, or a panic unwind — prefer calling `wait()` or `test()` explicitly
/// so that failure modes remain observable.** See also the migration guide note
/// in [`doc::migrating_from_rsmpi`](crate::doc::migrating_from_rsmpi).
///
/// A `MPI_Cancel`-then-`MPI_Wait`-with-timeout approach to make drop non-blocking
/// is under consideration and planned for v0.5.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, ReduceOp};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// let send = vec![world.rank() as f64; 10];
/// let mut recv = vec![0.0; 10];
///
/// // Start nonblocking all-reduce
/// let request = world.iallreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
///
/// // Do other work while communication proceeds...
///
/// // Wait for completion
/// request.wait().unwrap();
///
/// // Now recv contains the result
/// println!("Sum: {:?}", recv);
/// ```
pub struct Request {
    handle: i64,
    completed: bool,
}

impl Request {
    /// Create a new request from a raw handle.
    pub(crate) fn new(handle: i64) -> Self {
        Request {
            handle,
            completed: false,
        }
    }

    /// Get the raw request handle (for advanced use).
    pub fn raw_handle(&self) -> i64 {
        self.handle
    }

    /// Check if this request has been completed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// Wait for this operation to complete.
    ///
    /// Blocks until the operation is finished. After this returns successfully,
    /// the associated buffers can be safely accessed.
    #[inline]
    pub fn wait(mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        // Mark completed BEFORE the FFI call so that Drop does not attempt a
        // second MPI_Wait on error.  A request handed to MPI_Wait is consumed
        // by MPI regardless of whether MPI reports an error; re-waiting on it
        // would be a use-after-free of the request handle.
        self.completed = true;
        let ret = unsafe { ffi::ferrompi_wait(self.handle) };
        Error::check_with_op(ret, "wait")
    }

    /// Test if this operation has completed without blocking.
    ///
    /// Returns `true` if the operation is complete, `false` otherwise.
    ///
    /// # Note
    ///
    /// If this returns `true`, the request is consumed and you should not call
    /// `wait()` or `test()` again.
    #[inline]
    pub fn test(&mut self) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag) };
        Error::check_with_op(ret, "test")?;
        if flag != 0 {
            self.completed = true;
        }
        Ok(flag != 0)
    }

    /// Wait for any one request in a collection to complete.
    ///
    /// Blocks until at least one request completes and returns its index. Returns
    /// `Ok(None)` when all requests were already `MPI_REQUEST_NULL` on entry.
    ///
    /// The completed `Request` is marked `completed = true` in place. Removing it
    /// from the vector is the caller's responsibility.
    pub fn wait_any(requests: &mut [Request]) -> Result<Option<usize>> {
        if requests.is_empty() {
            return Ok(None);
        }
        let mut index: i32 = 0;
        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles whose length we pass as count; index is a valid stack-allocated
        // i32 output parameter.
        let ret = with_handles(requests, |handles| unsafe {
            ffi::ferrompi_waitany(handles.len() as i64, handles.as_mut_ptr(), &mut index)
        });
        Error::check_with_op(ret, "waitany")?;
        if index < 0 {
            return Ok(None);
        }
        let idx = index as usize;
        requests[idx].completed = true;
        Ok(Some(idx))
    }

    /// Wait until at least one request in a collection completes.
    ///
    /// Returns the indices of all requests that completed in this call.
    /// Returns `Ok(vec![])` when no requests were active (all null or all already done).
    ///
    /// The completed `Request`s are marked `completed = true` in place. Removing
    /// them from the vector is the caller's responsibility.
    pub fn wait_some(requests: &mut [Request]) -> Result<Vec<usize>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let len = requests.len();
        let mut outcount: i64 = 0;
        // SAFETY: with_handles / with_index_buf supply valid, appropriately-sized
        // [i64] handle and [i32] index buffers whose lengths match `count`;
        // outcount is a valid stack-allocated output parameter.
        let (ret, completed) = with_handles(requests, |handles| {
            with_index_buf(len, |indices| {
                let ret = unsafe {
                    ffi::ferrompi_waitsome(
                        handles.len() as i64,
                        handles.as_mut_ptr(),
                        &mut outcount,
                        indices.as_mut_ptr(),
                    )
                };
                // outcount == -1 means all null; 0 means none completed (should
                // not happen for waitsome, but guard defensively). Only collect
                // the completed indices while the index buffer is in scope.
                let completed: Vec<usize> = if ret == 0 && outcount > 0 {
                    indices[..outcount as usize]
                        .iter()
                        .map(|&i| i as usize)
                        .collect()
                } else {
                    Vec::new()
                };
                (ret, completed)
            })
        });
        Error::check_with_op(ret, "waitsome")?;
        for &idx in &completed {
            requests[idx].completed = true;
        }
        Ok(completed)
    }

    /// Test whether any one request in a collection has completed (non-blocking).
    ///
    /// Returns `Ok(Some(idx))` if a request completed, `Ok(None)` if no request
    /// has completed yet or all requests were already null.
    ///
    /// The completed `Request` is marked `completed = true` in place. Removing it
    /// from the vector is the caller's responsibility.
    pub fn test_any(requests: &mut [Request]) -> Result<Option<usize>> {
        if requests.is_empty() {
            return Ok(None);
        }
        let mut index: i32 = 0;
        let mut flag: i32 = 0;
        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles whose length we pass as count; index and flag are valid
        // stack-allocated i32 output parameters.
        let ret = with_handles(requests, |handles| unsafe {
            ffi::ferrompi_testany(
                handles.len() as i64,
                handles.as_mut_ptr(),
                &mut index,
                &mut flag,
            )
        });
        Error::check_with_op(ret, "testany")?;
        if flag == 0 {
            return Ok(None);
        }
        if index < 0 {
            // All requests were null — nothing to mark.
            return Ok(None);
        }
        let idx = index as usize;
        requests[idx].completed = true;
        Ok(Some(idx))
    }

    /// Test how many requests in a collection have completed (non-blocking).
    ///
    /// Returns the indices of all requests that have completed at the moment of
    /// the call. Returns `Ok(vec![])` when none have completed or all were null.
    ///
    /// The completed `Request`s are marked `completed = true` in place. Removing
    /// them from the vector is the caller's responsibility.
    pub fn test_some(requests: &mut [Request]) -> Result<Vec<usize>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let len = requests.len();
        let mut outcount: i64 = 0;
        // SAFETY: with_handles / with_index_buf supply valid, appropriately-sized
        // [i64] handle and [i32] index buffers whose lengths match `count`;
        // outcount is a valid stack-allocated output parameter.
        let (ret, completed) = with_handles(requests, |handles| {
            with_index_buf(len, |indices| {
                let ret = unsafe {
                    ffi::ferrompi_testsome(
                        handles.len() as i64,
                        handles.as_mut_ptr(),
                        &mut outcount,
                        indices.as_mut_ptr(),
                    )
                };
                // outcount == -1 means all null; 0 means none completed yet.
                // Collect completed indices only while the index buffer is alive.
                let completed: Vec<usize> = if ret == 0 && outcount > 0 {
                    indices[..outcount as usize]
                        .iter()
                        .map(|&i| i as usize)
                        .collect()
                } else {
                    Vec::new()
                };
                (ret, completed)
            })
        });
        Error::check_with_op(ret, "testsome")?;
        for &idx in &completed {
            requests[idx].completed = true;
        }
        Ok(completed)
    }

    /// Non-destructive query: check whether this request has completed
    /// without consuming it. Unlike [`test`](Request::test), this does NOT
    /// free the request on completion; it only probes.
    ///
    /// Returns `Ok(true)` if the MPI runtime reports the request is complete,
    /// `Ok(false)` otherwise. Does NOT mutate `completed` — this is a probe,
    /// not a commit.
    pub fn get_status(&self) -> Result<bool> {
        if self.completed {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        // SAFETY: self.handle is a valid request handle issued by the C shim.
        // flag is a valid stack-allocated i32 output parameter.
        let ret = unsafe { ffi::ferrompi_request_get_status(self.handle, &mut flag) };
        Error::check_with_op(ret, "request_get_status")?;
        Ok(flag != 0)
    }

    /// Request cancellation of a pending nonblocking operation.
    ///
    /// # Portability
    ///
    /// Per the MPI 4.0 standard, `MPI_Cancel` is effectively deprecated for
    /// send requests. Open MPI refuses to cancel sends; MPICH may report
    /// success but not actually cancel the send. Cancellation reliably works
    /// only for receives.
    ///
    /// # Usage
    ///
    /// `cancel` does NOT complete the request. The caller must follow up with
    /// [`wait`](Request::wait) to reclaim the handle:
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Result};
    /// # fn main() -> Result<()> {
    /// # let mpi = Mpi::init()?;
    /// # let world = mpi.world();
    /// # let mut buf = vec![0u8; 10];
    /// # let mut req = world.irecv(&mut buf, 0, 0)?;
    /// req.cancel()?;
    /// req.wait()?;
    /// # Ok(()) }
    /// ```
    pub fn cancel(&mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        // SAFETY: self.handle is a valid request handle issued by the C shim.
        let ret = unsafe { ffi::ferrompi_cancel(self.handle) };
        Error::check_with_op(ret, "cancel")
    }

    /// Wait for all requests in a slice to complete.
    ///
    /// This is more efficient than waiting for each request individually.
    ///
    /// Takes the requests by `&mut [Request]` (rather than consuming a
    /// `Vec<Request>`) so a caller can reuse one backing buffer across a drain
    /// loop, mirroring
    /// [`PersistentRequest::wait_all`](crate::PersistentRequest::wait_all). On
    /// success every request is marked completed in place, so the caller's later
    /// `Drop` of each is a no-op; on error the requests are left active so their
    /// `Drop` re-waits each one (preserving the prior cleanup semantics).
    pub fn wait_all(requests: &mut [Request]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles whose length we pass as count.
        let ret = with_handles(requests, |handles| unsafe {
            ffi::ferrompi_waitall(handles.len() as i64, handles.as_mut_ptr())
        });

        if ret == 0 {
            // Success: MPI consumed and freed every handle. Mark each completed
            // so the caller's eventual Drop does not re-wait it (which would be a
            // use-after-free of an already-freed request handle).
            for req in requests.iter_mut() {
                req.completed = true;
            }
            Ok(())
        } else {
            // Error: leave requests active so each one's Drop re-waits it,
            // matching the prior by-value behavior's cleanup path.
            Err(Error::from_code_with_op(ret, "waitall"))
        }
    }
}

impl Drop for Request {
    /// Block until the in-flight operation completes, then release the handle.
    ///
    /// Calls `MPI_Wait` on the underlying request handle when `self.completed`
    /// is `false`. **This call blocks** until the peer posts the matching
    /// operation; if the peer never does, this deadlocks.
    ///
    /// Maintainers: the `self.completed = true` assignment in `Request::wait`
    /// is the only guard that prevents a double-wait here. Any refactoring of
    /// `wait()` must preserve that assignment, or this `Drop` impl becomes
    /// unsound (double-freeing the request handle).
    ///
    /// The `MPI_Cancel`-then-`MPI_Wait`-with-timeout alternative is deferred
    /// to v0.5 (see ADR-0004 §"Drop behavior for nonblocking Request").
    fn drop(&mut self) {
        if !self.completed {
            // SAFETY: self.handle is a valid MPI request handle registered in the
            // C-side request table by the nonblocking constructor (e.g., iallreduce).
            // The handle has not been freed because self.completed is false, meaning
            // wait() was never called. ferrompi_wait calls MPI_Wait which frees the
            // handle on success; the completed flag guards against a double-free.
            unsafe { ffi::ferrompi_wait(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::forget;

    #[test]
    fn new_request_is_not_completed() {
        let req = Request::new(0);
        assert!(!req.is_completed());
        assert_eq!(req.raw_handle(), 0);
        forget(req);
    }

    #[test]
    fn raw_handle_returns_constructor_value() {
        let req = Request::new(99);
        assert_eq!(req.raw_handle(), 99);
        forget(req);
    }

    #[test]
    fn test_when_already_completed_returns_true() {
        let mut req = Request {
            handle: 0,
            completed: true,
        };
        let result = req.test();
        assert!(matches!(result, Ok(true)));
        forget(req);
    }

    #[test]
    fn wait_when_already_completed_returns_ok() {
        // wait() takes self by value (consuming).
        // With completed: true, it returns Ok(()) on line 63 before any FFI.
        // Drop then runs, but !self.completed is false, so Drop is a no-op.
        let req = Request {
            handle: 0,
            completed: true,
        };
        let result = req.wait();
        assert!(result.is_ok());
        // No forget() needed — wait() consumed the value, and Drop was a no-op
    }

    #[test]
    fn wait_all_empty_slice_returns_ok() {
        let result = Request::wait_all(&mut []);
        assert!(result.is_ok());
    }

    #[test]
    fn wait_any_empty_vec_returns_none() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::wait_any(&mut v).unwrap(), None);
    }

    #[test]
    fn wait_some_empty_vec_returns_empty() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::wait_some(&mut v).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn test_any_empty_vec_returns_none() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::test_any(&mut v).unwrap(), None);
    }

    #[test]
    fn test_some_empty_vec_returns_empty() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::test_some(&mut v).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn get_status_on_completed_request_returns_true_without_ffi() {
        let req = Request {
            handle: 0,
            completed: true,
        };
        let result = req.get_status();
        assert!(matches!(result, Ok(true)));
        forget(req);
    }

    #[test]
    fn cancel_on_completed_request_returns_ok_without_ffi() {
        let mut req = Request {
            handle: 0,
            completed: true,
        };
        let result = req.cancel();
        assert!(matches!(result, Ok(())));
        forget(req);
    }
}
