//! Persistent request handles for MPI 4.0+ persistent collectives.
//!
//! Persistent collectives allow you to initialize a collective operation once
//! and then start it multiple times. This amortizes the setup cost across many
//! iterations, which is particularly beneficial for iterative algorithms like SDDP.
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::{Mpi, ReduceOp};
//!
//! let mpi = Mpi::init().unwrap();
//! let world = mpi.world();
//!
//! // Buffer that will be used for all broadcasts
//! let mut data = vec![0.0f64; 1000];
//!
//! // Initialize persistent broadcast (MPI 4.0+)
//! let mut persistent = world.bcast_init(&mut data, 0).unwrap();
//!
//! // Run many iterations
//! for iter in 0..1000 {
//!     // Update data on root
//!     if world.rank() == 0 {
//!         for (i, x) in data.iter_mut().enumerate() {
//!             *x = (iter * 1000 + i) as f64;
//!         }
//!     }
//!
//!     // Start the broadcast
//!     persistent.start().unwrap();
//!
//!     // Optionally do other work here...
//!
//!     // Wait for completion
//!     persistent.wait().unwrap();
//!
//!     // data now contains broadcast result on all ranks
//! }
//!
//! // Cleanup happens automatically on drop
//! ```

use crate::error::{Error, Result};
use crate::ffi;

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
    /// Returns an error if the operation is already active or if the start fails.
    pub fn start(&mut self) -> Result<()> {
        if self.active {
            return Err(Error::Internal("Request is already active".into()));
        }
        let ret = unsafe { ffi::ferrompi_start(self.handle) };
        Error::check(ret)?;
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
    /// Returns an error if the operation is not active or if the wait fails.
    pub fn wait(&mut self) -> Result<()> {
        if !self.active {
            // Not started, nothing to wait for
            return Ok(());
        }
        let ret = unsafe { ffi::ferrompi_wait(self.handle) };
        Error::check(ret)?;
        self.active = false;
        Ok(())
    }

    /// Test if the operation has completed without blocking.
    ///
    /// Returns `true` if complete, `false` if still in progress.
    pub fn test(&mut self) -> Result<bool> {
        if !self.active {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag) };
        Error::check(ret)?;
        if flag != 0 {
            self.active = false;
        }
        Ok(flag != 0)
    }

    /// Start multiple persistent operations.
    ///
    /// This is more efficient than starting each operation individually.
    pub fn start_all(requests: &mut [PersistentRequest]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        // Check none are already active
        for req in requests.iter() {
            if req.active {
                return Err(Error::Internal(
                    "One or more requests already active".into(),
                ));
            }
        }

        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let ret = unsafe { ffi::ferrompi_startall(handles.len() as i64, handles.as_mut_ptr()) };
        Error::check(ret)?;

        // Mark all as active
        for req in requests.iter_mut() {
            req.active = true;
        }

        Ok(())
    }

    /// Wait for all persistent operations to complete.
    pub fn wait_all(requests: &mut [PersistentRequest]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let mut handles: Vec<i64> = requests.iter().map(|r| r.handle).collect();
        let ret = unsafe { ffi::ferrompi_waitall(handles.len() as i64, handles.as_mut_ptr()) };
        Error::check(ret)?;

        // Only mark as inactive after successful wait
        for req in requests.iter_mut() {
            req.active = false;
        }

        Ok(())
    }
}

impl Drop for PersistentRequest {
    fn drop(&mut self) {
        // If active, wait for completion first
        if self.active {
            unsafe { ffi::ferrompi_wait(self.handle) };
        }
        // Free the persistent request
        unsafe { ffi::ferrompi_request_free(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::forget;

    #[test]
    fn new_request_is_inactive() {
        let req = PersistentRequest::new(0);
        assert!(!req.is_active());
        assert_eq!(req.raw_handle(), 0);
        forget(req);
    }

    #[test]
    fn raw_handle_returns_constructor_value() {
        let req = PersistentRequest::new(42);
        assert_eq!(req.raw_handle(), 42);
        forget(req);
    }

    #[test]
    fn start_when_already_active_returns_error() {
        let mut req = PersistentRequest {
            handle: 0,
            active: true,
        };
        let result = req.start();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, Error::Internal(msg) if msg.contains("already active")),
            "expected Error::Internal containing 'already active', got: {err}"
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
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, Error::Internal(msg) if msg.contains("already active")),
            "expected Error::Internal containing 'already active', got: {err}"
        );
        forget(req);
    }

    #[test]
    fn wait_all_empty_slice_returns_ok() {
        let result = PersistentRequest::wait_all(&mut []);
        assert!(result.is_ok());
    }
}
