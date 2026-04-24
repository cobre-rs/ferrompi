//! Request handles for nonblocking MPI operations.

use crate::error::{Error, Result};
use crate::ffi;

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
    pub fn wait(mut self) -> Result<()> {
        if self.completed {
            return Ok(());
        }
        let ret = unsafe { ffi::ferrompi_wait(self.handle) };
        Error::check_with_op(ret, "wait")?;
        self.completed = true;
        Ok(())
    }

    /// Test if this operation has completed without blocking.
    ///
    /// Returns `true` if the operation is complete, `false` otherwise.
    ///
    /// # Note
    ///
    /// If this returns `true`, the request is consumed and you should not call
    /// `wait()` or `test()` again.
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
        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let mut index: i32 = 0;
        // SAFETY: handles is a valid, mutable Vec<i64> whose length we pass as count.
        // index is a valid stack-allocated i32 output parameter.
        let ret = unsafe {
            ffi::ferrompi_waitany(handles.len() as i64, handles.as_mut_ptr(), &mut index)
        };
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
        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let mut outcount: i64 = 0;
        let mut indices: Vec<i32> = vec![0; requests.len()];
        // SAFETY: handles, outcount, and indices are valid, appropriately-sized
        // output buffers. handles.len() is passed as count.
        let ret = unsafe {
            ffi::ferrompi_waitsome(
                handles.len() as i64,
                handles.as_mut_ptr(),
                &mut outcount,
                indices.as_mut_ptr(),
            )
        };
        Error::check_with_op(ret, "waitsome")?;
        if outcount <= 0 {
            // outcount == -1 means all null; 0 means none completed (shouldn't
            // happen for waitsome, but guard defensively).
            return Ok(vec![]);
        }
        let completed: Vec<usize> = indices[..outcount as usize]
            .iter()
            .map(|&i| i as usize)
            .collect();
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
        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let mut index: i32 = 0;
        let mut flag: i32 = 0;
        // SAFETY: handles is a valid, mutable Vec<i64> whose length we pass as count.
        // index and flag are valid stack-allocated i32 output parameters.
        let ret = unsafe {
            ffi::ferrompi_testany(
                handles.len() as i64,
                handles.as_mut_ptr(),
                &mut index,
                &mut flag,
            )
        };
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
        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let mut outcount: i64 = 0;
        let mut indices: Vec<i32> = vec![0; requests.len()];
        // SAFETY: handles, outcount, and indices are valid, appropriately-sized
        // output buffers. handles.len() is passed as count.
        let ret = unsafe {
            ffi::ferrompi_testsome(
                handles.len() as i64,
                handles.as_mut_ptr(),
                &mut outcount,
                indices.as_mut_ptr(),
            )
        };
        Error::check_with_op(ret, "testsome")?;
        if outcount <= 0 {
            // outcount == -1 means all null; 0 means none completed yet.
            return Ok(vec![]);
        }
        let completed: Vec<usize> = indices[..outcount as usize]
            .iter()
            .map(|&i| i as usize)
            .collect();
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

    /// Wait for all requests in a collection to complete.
    ///
    /// This is more efficient than waiting for each request individually.
    pub fn wait_all(requests: Vec<Request>) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let ret = unsafe { ffi::ferrompi_waitall(handles.len() as i64, handles.as_mut_ptr()) };

        if ret == 0 {
            // Success: all requests completed, skip Drop (handles already freed by MPI)
            for mut req in requests {
                req.completed = true;
                std::mem::forget(req);
            }
            Ok(())
        } else {
            // Error: let Drop handle cleanup (will re-wait each active request)
            Err(Error::from_code_with_op(ret, "waitall"))
        }
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if !self.completed {
            // If the request wasn't waited on, we need to wait now to avoid
            // leaving the operation in an undefined state
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
    fn wait_all_empty_vec_returns_ok() {
        let result = Request::wait_all(vec![]);
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
