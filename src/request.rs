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
        Error::check(ret)?;
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
        Error::check(ret)?;
        if flag != 0 {
            self.completed = true;
        }
        Ok(flag != 0)
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
            Err(Error::from_code(ret))
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
}
