//! The `PersistentRequest` handle returned by the persistent `*_init` constructors.

use crate::datatype::MpiDatatype;
use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ffi;
use crate::op::OpRegistration;
use crate::rt;
use std::ffi::c_void;
use std::sync::Arc;

/// Describes the heap allocation of `v` for MPI FFI as `(pointer, count, tag)`,
/// like `datatype::buf` but with the pointer taken from the `Vec` itself
/// (`Vec::as_ptr`) instead of from a slice reborrow.
///
/// A pointer that MPI keeps across calls, as a persistent request does, must
/// be derived this way: no slice reference then sits between it and the
/// allocation, so the slices the request's accessors create later are its
/// children. A pointer taken through a slice reference is invalidated by those
/// accessors under the Stacked and Tree Borrows models.
#[allow(
    clippy::ptr_arg,
    reason = "the pointer must come from the Vec, not from a slice reborrow"
)]
pub(crate) fn vec_buf<T: MpiDatatype>(v: &Vec<T>) -> (*const c_void, i64, i32) {
    (v.as_ptr().cast(), v.len() as i64, T::TAG as i32)
}

/// See [`vec_buf`]; the pointer comes from `Vec::as_mut_ptr`.
pub(crate) fn vec_buf_mut<T: MpiDatatype>(v: &mut Vec<T>) -> (*mut c_void, i64, i32) {
    (v.as_mut_ptr().cast(), v.len() as i64, T::TAG as i32)
}

/// A persistent MPI request that owns its buffers.
///
/// A persistent request is initialized once and then started and completed
/// many times, so one setup serves a whole loop of iterations. The `*_init`
/// constructors take the buffers by value and the request keeps them until it
/// is freed: MPI holds a pointer into each allocation, so no safe call can
/// resize or free one behind MPI's back. Reach the buffer with
/// [`buffer`](Self::buffer) and [`buffer_mut`](Self::buffer_mut) while the
/// request is inactive, or take it back with
/// [`into_buffer`](Self::into_buffer).
///
/// A collective with separate send and receive buffers owns both. `buffer` and
/// `buffer_mut` reach the receive buffer, [`send_buffer_mut`](Self::send_buffer_mut)
/// reaches the send buffer, and [`into_buffers`](Self::into_buffers) returns
/// both. A request that owns one buffer (a point-to-point request, a
/// broadcast or an in-place collective) has an empty send buffer.
///
/// # Lifecycle
///
/// 1. Create with `comm.send_init(vec, ..)`, `comm.recv_init(vec, ..)` or
///    another `*_init`, which takes the buffers by value.
/// 2. Update the buffer through `buffer_mut()` (or `send_buffer_mut()`). The
///    accessors work only while the request is inactive and return
///    [`Error::InvalidState`] otherwise.
/// 3. Start with `start()` or `start_all()`.
/// 4. Wait for completion with `wait()`, then read the result through
///    `buffer()`.
/// 5. Repeat steps 2-4 as needed.
/// 6. Free on drop, or take the buffer back with `into_buffer()` or
///    `into_buffers()`.
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// # fn main() -> ferrompi::Result<()> {
/// let mpi = Mpi::init()?;
/// let world = mpi.world();
/// let rank = world.rank();
///
/// // Initialize a self-send pair once; each request owns its buffer.
/// let mut send = world.send_init(vec![0.0f64; 1000], rank, 0)?;
/// let mut recv = world.recv_init(vec![0.0f64; 1000], rank, 0)?;
///
/// for iter in 0..1000 {
///     for (i, x) in send.buffer_mut()?.iter_mut().enumerate() {
///         *x = (iter * 1000 + i) as f64;
///     }
///
///     recv.start()?;
///     send.start()?;
///     send.wait()?;
///     recv.wait()?;
///
///     assert_eq!(recv.buffer()?[0], (iter * 1000) as f64);
/// }
///
/// // Cleanup happens automatically on drop.
/// # Ok(())
/// # }
/// ```
///
/// # Dropping
///
/// Dropping an active request completes it (`MPI_Wait`, which blocks until the
/// peer operation completes), then frees it (`MPI_Request_free`), and only
/// then frees the buffers.
///
/// - At `Single`/`Funneled`, drop it on the thread that initialized MPI;
///   another thread aborts the process.
/// - At `Serialized` and `Multiple`, the drop is an MPI call, which the
///   program serializes like any other.
/// - Once [`Mpi`](crate::Mpi) is being dropped, or after it was dropped, the
///   drop makes no MPI call. It leaks the request handle, and also the buffers
///   when the request is active, because MPI may still write into them. The
///   buffers of an inactive request are freed normally.
/// - A drop on another thread while `Mpi` is being dropped either completes
///   before `MPI_Finalize`, because it holds a per-call count the finalizer
///   waits for, or makes no MPI call and leaks as above. A drop that is blocked
///   in `MPI_Wait` for a peer that never arrives blocks `Mpi`'s drop the same
///   way a blocked `wait()` does.
/// - Dropping `Mpi` while the request is active skips `MPI_Finalize`, whatever
///   thread holds the request (see [`Mpi`](crate::Mpi)). The request can still
///   be completed with `wait` or `test` afterwards; dropping it then makes no
///   MPI call, as above.
/// - The drop does not assume that a failed wait completed the request
///   (MPI-4.1 §9.4). If completing an active request fails, it makes no
///   further MPI call and leaks the request handle, the buffers and the user
///   op, which the request keeps alive (see [`UserOp`](crate::UserOp)). The
///   request also keeps counting as active, so a later [`Mpi`](crate::Mpi)
///   drop skips `MPI_Finalize`.
pub struct PersistentRequest<T> {
    handle: i64,
    token: Option<rt::HandleToken>,
    send: Vec<T>,
    buffer: Vec<T>,
    counts: Vec<i32>,
    pub(crate) op: Option<Arc<OpRegistration>>,
}

impl<T> PersistentRequest<T> {
    /// Create a new persistent request from a raw handle and the one buffer it
    /// owns.
    pub(crate) fn new(handle: i64, buffer: Vec<T>) -> Self {
        Self::new_pair(handle, Vec::new(), buffer, Vec::new())
    }

    /// Create a persistent request that owns a send buffer, a receive buffer
    /// and the copy of the count and displacement arrays a v-collective was
    /// given (empty for the other collectives).
    pub(crate) fn new_pair(handle: i64, send: Vec<T>, buffer: Vec<T>, counts: Vec<i32>) -> Self {
        PersistentRequest {
            handle,
            token: None,
            send,
            buffer,
            counts,
            op: None,
        }
    }

    /// Leaks what MPI may still use: the buffers, the count and displacement
    /// copy, the user op the request keeps alive, and the token that makes
    /// `Mpi::drop` skip `MPI_Finalize` while the request may be active.
    fn leak(&mut self) {
        std::mem::forget(self.token.take());
        std::mem::forget(std::mem::take(&mut self.send));
        std::mem::forget(std::mem::take(&mut self.buffer));
        std::mem::forget(std::mem::take(&mut self.counts));
        std::mem::forget(self.op.take());
    }

    /// Get the raw request handle (for advanced use).
    pub fn raw_handle(&self) -> i64 {
        self.handle
    }

    /// Check if this request is currently active (started but not waited).
    pub fn is_active(&self) -> bool {
        self.token.is_some()
    }

    /// Start the persistent operation.
    ///
    /// This initiates the communication. You must call `wait()` before starting
    /// again or reaching the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] if the operation is already active,
    /// [`Error::Finalized`] once [`Mpi`](crate::Mpi) is being dropped,
    /// [`Error::ThreadLevelViolation`] on a thread other than the one that
    /// initialized MPI at `Single`/`Funneled`, or the error of a failed start.
    #[inline]
    pub fn start(&mut self) -> Result<()> {
        if self.is_active() {
            return Err(Error::InvalidState {
                reason: "request is already active",
            });
        }
        let tokens =
            rt::open_handle_tokens(1).map_err(|code| Error::from_code_with_op(code, "start"))?;
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table by the *_init constructor
        // that produced this PersistentRequest; the request is inactive (no
        // token), so MPI_Start is not being called on an already-active
        // request.
        let ret = unsafe { ffi::ferrompi_start(self.handle) };
        Error::check_with_op(ret, "start")?;
        self.token = tokens.take();
        Ok(())
    }

    /// Wait for the operation to complete.
    ///
    /// This blocks until the communication started by `start()` is finished.
    /// After this returns, the buffer can be reached with
    /// [`buffer`](Self::buffer) and [`buffer_mut`](Self::buffer_mut), and the
    /// operation can be started again.
    ///
    /// # Errors
    ///
    /// Returns an error if the wait fails. A wait that MPI ran leaves the
    /// request inactive either way: MPI completed it with that error. This
    /// assumes that a request MPI reported an error for is complete, as MPICH
    /// and Open MPI leave it; MPI-4.1 §9.4 (p.460) says an operation that
    /// raised an error may not have satisfied its specification, so the
    /// standard does not promise it. The same assumption holds for
    /// [`test`](Self::test) and [`wait_all`](Self::wait_all); only `Drop`
    /// does not make it. A call rejected before reaching MPI (wrong thread,
    /// after finalize, or, in a debug build at `Serialized`, overlapping
    /// another thread's call) leaves the request active.
    #[inline]
    pub fn wait(&mut self) -> Result<()> {
        if !self.is_active() {
            return Ok(());
        }
        Error::check_with_op(rt::check_completion(), "wait")?;
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table; the request was active on
        // entry (checked above), so start() was called and MPI holds an
        // in-flight operation on this handle for ferrompi_wait to complete.
        let ret = unsafe { ffi::ferrompi_wait(self.handle, std::ptr::null_mut()) };
        // A call the guard rejects before MPI sees it leaves the request
        // active, token included: a debug build's Serialized overlap check, or
        // a finalize that stored `Finalized` after the pre-check above.
        // Otherwise MPI completed it whatever it returned: it is inactive now,
        // or freed if the library frees failed persistent requests.
        if ret != FERROMPI_ERR_THREAD_LEVEL && ret != FERROMPI_ERR_FINALIZED {
            self.token = None;
        }
        Error::check_with_op(ret, "wait")
    }

    /// Test if the operation has completed without blocking.
    ///
    /// Returns `true` if complete, `false` if still in progress. A failed
    /// `test` that MPI completed also leaves the request inactive, on the
    /// assumption described under [`wait`](Self::wait).
    #[inline]
    pub fn test(&mut self) -> Result<bool> {
        if !self.is_active() {
            return Ok(true);
        }
        let mut flag: i32 = 0;
        // SAFETY: self.handle is a valid persistent MPI request handle
        // registered in the C-side request table; the request was active on
        // entry (checked above). flag is a local out-parameter written by
        // ferrompi_test before this function reads it below.
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag, std::ptr::null_mut()) };
        // flag is set when MPI completed the request, even with an error.
        if flag != 0 {
            self.token = None;
        }
        Error::check_with_op(ret, "test")?;
        Ok(flag != 0)
    }

    /// The request's buffer, while the request is inactive.
    ///
    /// This is the send data of a send request, and the received data of a
    /// receive request after its `wait()`. For a collective with separate send
    /// and receive buffers it is the receive buffer; reach the send buffer with
    /// [`send_buffer_mut`](Self::send_buffer_mut). The slice cannot be resized,
    /// because MPI holds a pointer into the allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] while the request is active: MPI owns
    /// the buffer from `start()` until `wait()`, `wait_all()`, or a `test()`
    /// that reports completion, returns.
    #[inline]
    pub fn buffer(&self) -> Result<&[T]> {
        if self.is_active() {
            return Err(Error::InvalidState {
                reason: "request is active",
            });
        }
        Ok(&self.buffer)
    }

    /// The request's buffer for writing, while the request is inactive.
    ///
    /// Write the next iteration's input here between `wait()` and the next
    /// `start()`. For a collective with separate send and receive buffers it is
    /// the receive buffer; reach the send buffer with
    /// [`send_buffer_mut`](Self::send_buffer_mut). The slice cannot be resized,
    /// because MPI holds a pointer into the allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] while the request is active.
    #[inline]
    pub fn buffer_mut(&mut self) -> Result<&mut [T]> {
        if self.is_active() {
            return Err(Error::InvalidState {
                reason: "request is active",
            });
        }
        Ok(&mut self.buffer)
    }

    /// The send buffer of a collective with separate send and receive buffers,
    /// for writing, while the request is inactive.
    ///
    /// Write the next iteration's input here between `wait()` and the next
    /// `start()`. The slice is empty for a request that owns one buffer: a
    /// point-to-point request, a broadcast or an in-place collective. The slice
    /// cannot be resized, because MPI holds a pointer into the allocation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidState`] while the request is active.
    #[inline]
    pub fn send_buffer_mut(&mut self) -> Result<&mut [T]> {
        if self.is_active() {
            return Err(Error::InvalidState {
                reason: "request is active",
            });
        }
        Ok(&mut self.send)
    }

    /// Complete the request if it is active, free it, and return its buffer.
    ///
    /// The request is freed before this returns. Freeing a request is an MPI
    /// call, so `Drop`'s thread rule applies: at `Single`/`Funneled`, on a
    /// thread other than the one that initialized MPI, the process aborts.
    ///
    /// For a collective with separate send and receive buffers this returns the
    /// receive buffer and drops the send buffer after the free; use
    /// [`into_buffers`](Self::into_buffers) to keep both.
    ///
    /// # Errors
    ///
    /// Returns the error of [`wait`](Self::wait) if completing an active
    /// request fails. The request is then dropped as `Drop` does, and the
    /// buffer is not returned.
    pub fn into_buffer(mut self) -> Result<Vec<T>> {
        self.wait()?;
        Ok(std::mem::take(&mut self.buffer))
    }

    /// Complete the request if it is active, free it, and return its send and
    /// receive buffers as `(send, buffer)`.
    ///
    /// The send buffer is empty for a request that owns one buffer. Otherwise
    /// this behaves as [`into_buffer`](Self::into_buffer).
    ///
    /// # Errors
    ///
    /// Returns the error of [`wait`](Self::wait) if completing an active
    /// request fails. The request is then dropped as `Drop` does, and the
    /// buffers are not returned.
    pub fn into_buffers(mut self) -> Result<(Vec<T>, Vec<T>)> {
        self.wait()?;
        Ok((
            std::mem::take(&mut self.send),
            std::mem::take(&mut self.buffer),
        ))
    }

    /// Start multiple persistent operations.
    ///
    /// This is more efficient than starting each operation individually.
    ///
    /// # Errors
    ///
    /// Returns `Err(`[`Error::InvalidState`]`)` without calling MPI if any
    /// request is already active, and [`Error::Finalized`] or
    /// [`Error::ThreadLevelViolation`] as [`start`](Self::start) does. If MPI
    /// reports an error, it may have started some of the requests; every
    /// request is then marked active, so `wait`, `wait_all` or `Drop`
    /// completes whichever did start.
    pub fn start_all(requests: &mut [PersistentRequest<T>]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        if requests.iter().any(PersistentRequest::is_active) {
            return Err(Error::InvalidState {
                reason: "a request is already active",
            });
        }

        let tokens = rt::open_handle_tokens(requests.len())
            .map_err(|code| Error::from_code_with_op(code, "startall"))?;

        // SAFETY: with_handles provides a valid, contiguous [i64] of the
        // persistent-request handles and a same-length [u8] started buffer,
        // both sized to the count we pass.
        let ret = crate::request::with_handles(
            requests,
            |r| r.handle,
            |handles, started| unsafe {
                ffi::ferrompi_startall(handles.len() as i64, handles.as_ptr(), started.as_mut_ptr())
            },
            |r| r.token = tokens.take(),
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
    pub fn wait_all(requests: &mut [PersistentRequest<T>]) -> Result<()> {
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
            |r| if r.is_active() { r.handle } else { -1 },
            |handles, done| unsafe {
                ffi::ferrompi_waitall(
                    handles.len() as i64,
                    handles.as_ptr(),
                    done.as_mut_ptr(),
                    &mut failed,
                )
            },
            |r| r.token = None,
        );
        crate::request::check_batch(ret, "waitall", failed)
    }
}

impl<T> Drop for PersistentRequest<T> {
    /// Complete any in-flight operation, free the persistent request handle,
    /// then free the buffers and release the user op.
    ///
    /// When the request is active (i.e., `start()` was called but `wait()` has
    /// not yet returned), this calls `MPI_Wait` before freeing the handle.
    /// **`MPI_Wait` blocks** until the peer operation completes; if the peer is
    /// unreachable, this deadlocks. This two-step sequence upholds the MPI
    /// standard requirement that `MPI_Request_free` must not be called on an
    /// active request. The buffers are freed last, once MPI is done with them,
    /// and the user op after them: MPI may invoke it at every start.
    ///
    /// A drop that began before [`Mpi`](crate::Mpi) started to drop is an MPI
    /// call in flight at `Serialized` and `Multiple`. MPI requires it to
    /// complete before `MPI_Finalize` (MPI-4.1 §11.2.2 and §11.6.2 p.516), so
    /// `Mpi`'s drop waits for it, and a blocked `MPI_Wait` here blocks that
    /// drop, as a blocked `wait()` does.
    ///
    /// Once `Mpi` is being dropped, or after it was dropped, the drop makes no
    /// MPI call and leaks the request handle. An active request's buffers are
    /// leaked too, send and receive buffers and the copy of a v-collective's
    /// count and displacement arrays, and so are its user op and the token that
    /// made `Mpi`'s drop skip `MPI_Finalize`, because MPI may still use them; an
    /// inactive request's are freed, because MPI uses them only after a new
    /// start, and no start can be issued any more.
    ///
    /// The drop does not assume that a failed wait completed the request
    /// (MPI-4.1 §9.4), unlike `wait`, `test` and `wait_all`, which trust the
    /// completion MPI reports. It then leaks the same things, the token
    /// included so that `Mpi`'s drop still skips `MPI_Finalize`, and skips
    /// `MPI_Request_free`: freeing an active collective request is erroneous,
    /// and a point-to-point request's buffer must outlive MPI's use of it.
    fn drop(&mut self) {
        let Some(_call) = rt::drop_guard("PersistentRequest") else {
            if self.is_active() {
                self.leak();
            }
            return;
        };
        if self.is_active() {
            // SAFETY: self.handle is a valid MPI request handle registered in the
            // C-side request table by the *_init constructor. The request is
            // active, so start() was called and MPI holds an in-flight operation
            // on this handle. ferrompi_wait calls MPI_Wait which completes the
            // operation and releases the handle's active state before
            // request_free below.
            // Calls the unguarded raw wrapper (not the lifecycle-guarded one):
            // rt::drop_guard above already handles the FFI lifecycle check, so
            // this call must still attempt the wait once reached.
            let ret = unsafe { ffi::raw::ferrompi_wait(self.handle, std::ptr::null_mut()) };
            if ret != 0 {
                self.leak();
                return;
            }
        }
        // SAFETY: self.handle is a valid persistent MPI request handle. If it was
        // active, ferrompi_wait above has already completed the operation, so
        // MPI_Request_free is safe to call. If it was inactive, no operation is
        // in flight and MPI_Request_free is unconditionally safe on the handle.
        unsafe { ffi::ferrompi_request_free(self.handle) };
        // Released before `_call`, so a finalizer that waited for `_call` also
        // sees the request gone.
        self.token = None;
    }
}

#[cfg(test)]
mod tests {
    use super::{vec_buf, vec_buf_mut, PersistentRequest};
    use crate::datatype::MpiDatatype;
    use crate::error::Error;
    use crate::op::OpRegistration;
    use crate::rt;
    use std::mem::forget;
    use std::sync::Arc;

    /// What a started request holds. Counted before init, on the SeqCst path;
    /// the test must release it, or drop the request that holds it, so the
    /// count does not outlive it.
    fn active_token() -> rt::HandleToken {
        rt::open_handle_tokens(1)
            .expect("an uninitialized process opens a token")
            .take()
            .expect("one token was opened")
    }

    #[test]
    fn start_when_already_active_returns_error() {
        let mut req = PersistentRequest::<u8> {
            handle: 0,
            token: Some(active_token()),
            send: Vec::new(),
            buffer: Vec::new(),
            counts: Vec::new(),
            op: None,
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
        drop(req.token.take());
        forget(req);
    }

    #[test]
    fn test_when_inactive_returns_true() {
        let mut req = PersistentRequest::<u8>::new(0, Vec::new());
        let result = req.test();
        assert!(
            matches!(result, Ok(true)),
            "expected Ok(true), got: {result:?}"
        );
        forget(req);
    }

    #[test]
    fn start_all_empty_slice_returns_ok() {
        let result = PersistentRequest::<u8>::start_all(&mut []);
        assert!(result.is_ok());
    }

    #[test]
    fn start_all_with_active_request_returns_error() {
        let mut req = PersistentRequest::<u8> {
            handle: 0,
            token: Some(active_token()),
            send: Vec::new(),
            buffer: Vec::new(),
            counts: Vec::new(),
            op: None,
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
        drop(req.token.take());
        forget(req);
    }

    #[test]
    fn wait_all_empty_slice_returns_ok() {
        let result = PersistentRequest::<u8>::wait_all(&mut []);
        assert!(result.is_ok());
    }

    #[test]
    fn buffer_when_active_returns_invalid_state() {
        let mut req = PersistentRequest {
            handle: 0,
            token: Some(active_token()),
            send: Vec::new(),
            buffer: vec![0u8],
            counts: Vec::new(),
            op: None,
        };
        let shared = req.buffer();
        assert!(
            matches!(
                shared,
                Err(Error::InvalidState {
                    reason: "request is active"
                })
            ),
            "expected Err(InvalidState), got: {shared:?}"
        );
        let exclusive = req.buffer_mut();
        assert!(
            matches!(
                exclusive,
                Err(Error::InvalidState {
                    reason: "request is active"
                })
            ),
            "expected Err(InvalidState), got: {exclusive:?}"
        );
        drop(req.token.take());
        drop(std::mem::take(&mut req.buffer));
        forget(req);
    }

    #[test]
    fn send_buffer_when_active_returns_invalid_state() {
        let mut req = PersistentRequest {
            handle: 0,
            token: Some(active_token()),
            send: vec![0u8],
            buffer: Vec::new(),
            counts: Vec::new(),
            op: None,
        };
        let result = req.send_buffer_mut();
        assert!(
            matches!(
                result,
                Err(Error::InvalidState {
                    reason: "request is active"
                })
            ),
            "expected Err(InvalidState), got: {result:?}"
        );
        drop(req.token.take());
        drop(std::mem::take(&mut req.send));
        forget(req);
    }

    #[test]
    fn send_buffer_of_one_buffer_request_is_empty() {
        let mut req = PersistentRequest::new(0, vec![0u8; 3]);
        assert!(req.send_buffer_mut().expect("inactive").is_empty());
        assert_eq!(req.buffer().expect("inactive").len(), 3);
        drop(std::mem::take(&mut req.buffer));
        forget(req);
    }

    /// The request is `Send` for every element type and `Sync` for the
    /// concrete ones, so a field added later cannot silently remove either.
    #[test]
    fn persistent_request_is_send_and_sync() {
        fn is_send<X: Send>() {}
        fn is_send_for_every_element_type<T: MpiDatatype>() {
            is_send::<PersistentRequest<T>>();
        }
        fn is_sync<X: Sync>() {}
        is_send_for_every_element_type::<f64>();
        is_sync::<PersistentRequest<f64>>();
        is_sync::<PersistentRequest<i32>>();
        is_sync::<PersistentRequest<u8>>();
    }

    /// Stands in for MPI writing a receive between `start` and completion:
    /// the pointer `vec_buf_mut` returned must stay usable across the
    /// accessors. Run under Miri, with and without `-Zmiri-tree-borrows`, to
    /// check its provenance.
    #[test]
    fn recorded_receive_pointer_survives_accessors() {
        let mut req = PersistentRequest::new(0, vec![0u64; 4]);
        let (p, n, _) = vec_buf_mut(&mut req.buffer);
        let p = p.cast::<u64>();
        assert_eq!(n, 4);
        for round in 1..=3u64 {
            req.buffer_mut().expect("inactive").fill(round);
            req.token = Some(active_token());
            for i in 0..4 {
                // SAFETY: `p` addresses the live allocation of four `u64`s that
                // `vec_buf_mut` derived it from, and no accessor slice is live
                // while the flag is set.
                unsafe { p.add(i).write(round * 10 + i as u64) };
            }
            req.token = None;
            let expected = [round * 10, round * 10 + 1, round * 10 + 2, round * 10 + 3];
            assert_eq!(req.buffer().expect("inactive"), expected);
        }
        drop(std::mem::take(&mut req.buffer));
        forget(req);
    }

    /// The send counterpart: MPI reads through the pointer `vec_buf`
    /// returned after the accessors rewrote the buffer.
    #[test]
    fn recorded_send_pointer_survives_accessors() {
        let mut req = PersistentRequest::new(0, vec![0u64; 4]);
        let (p, _, _) = vec_buf(&req.buffer);
        let p = p.cast::<u64>();
        for round in 1..=3u64 {
            req.buffer_mut().expect("inactive").fill(round);
            req.token = Some(active_token());
            for i in 0..4 {
                // SAFETY: `p` addresses the live allocation of four `u64`s that
                // `vec_buf` derived it from, and no accessor slice is live while
                // the flag is set.
                assert_eq!(unsafe { p.add(i).read() }, round);
            }
            req.token = None;
        }
        drop(std::mem::take(&mut req.buffer));
        forget(req);
    }

    /// The two-buffer counterpart: MPI reads the send buffer and writes the
    /// receive buffer through the pointers `vec_buf` and `vec_buf_mut`
    /// returned, while the accessors rewrite the send buffer and read the
    /// receive buffer.
    #[test]
    fn recorded_send_and_receive_pointers_survive_accessors() {
        let send = vec![0u64; 4];
        let mut recv = vec![0u64; 4];
        let (sp, _, _) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        let (sp, rp) = (sp.cast::<u64>(), rp.cast::<u64>());
        let mut req = PersistentRequest::new_pair(0, send, recv, Vec::new());
        for round in 1..=3u64 {
            req.send_buffer_mut().expect("inactive").fill(round);
            req.token = Some(active_token());
            for i in 0..4 {
                // SAFETY: `sp` and `rp` address the live allocations of four
                // `u64`s that `vec_buf` and `vec_buf_mut` derived them from, and
                // no accessor slice is live while the flag is set.
                unsafe { rp.add(i).write(sp.add(i).read() * 10 + i as u64) };
            }
            req.token = None;
            let expected = [round * 10, round * 10 + 1, round * 10 + 2, round * 10 + 3];
            assert_eq!(req.buffer().expect("inactive"), expected);
            assert_eq!(req.send_buffer_mut().expect("inactive"), [round; 4]);
        }
        drop(std::mem::take(&mut req.send));
        drop(std::mem::take(&mut req.buffer));
        forget(req);
    }

    /// MPI keeps the pointers into a v-collective's count and displacement
    /// copy until the request is freed. The pointer is taken from the `Vec`
    /// before it moves into the request, as the constructors do, and must stay
    /// readable across the move and the accessors.
    #[test]
    fn recorded_counts_pointer_survives_accessors() {
        let counts = vec![3i32, 1, 4, 1];
        let cp = counts.as_ptr();
        let mut req = PersistentRequest::new_pair(0, vec![0u8; 2], vec![0u8; 2], counts);
        for round in 1..=3u8 {
            req.send_buffer_mut().expect("inactive").fill(round);
            req.buffer_mut().expect("inactive").fill(round);
            req.token = Some(active_token());
            for (i, want) in [3, 1, 4, 1].into_iter().enumerate() {
                // SAFETY: `cp` addresses the live allocation of four `i32`s that
                // `Vec::as_ptr` derived it from, and the request only ever moves
                // that `Vec`.
                assert_eq!(unsafe { cp.add(i).read() }, want);
            }
            req.token = None;
        }
        drop(std::mem::take(&mut req.send));
        drop(std::mem::take(&mut req.buffer));
        drop(std::mem::take(&mut req.counts));
        forget(req);
    }

    /// A drop does not assume that a failed raw wait completed the request, so
    /// it must neither free the request nor release what MPI may still use,
    /// the token that keeps `Mpi::drop` from finalizing included.
    /// Handle -1 names no request: the C layer refuses the wait without
    /// calling MPI, which stands in for an `MPI_Wait` that returned an error.
    #[test]
    #[cfg_attr(miri, ignore = "calls the C layer")]
    fn drop_after_a_failed_wait_leaks_the_buffers_and_the_op() {
        let registration = Arc::new(OpRegistration { slot: 9 });
        let mut send = vec![0u64; 4];
        let mut recv = vec![0u64; 4];
        let mut counts = vec![0i32; 4];
        let (sp, rp, cp) = (send.as_mut_ptr(), recv.as_mut_ptr(), counts.as_mut_ptr());
        let mut req = PersistentRequest::new_pair(-1, send, recv, counts);
        req.op = Some(Arc::clone(&registration));
        req.token = Some(active_token());
        drop(req);
        assert_eq!(Arc::strong_count(&registration), 2);
        // The token stays counted, so `Mpi::drop` still skips `MPI_Finalize`.
        // Other tests hold tokens too, so the sum is only bounded below.
        assert!(rt::live_handle_tokens() >= 1);
        // SAFETY: the failed wait made the drop forget the three `Vec`s (each
        // of length and capacity four, as built above) and the op clone, so
        // each allocation is live and has no other owner, and the leaked strong
        // count is the one released here.
        unsafe {
            drop(Vec::from_raw_parts(sp, 4, 4));
            drop(Vec::from_raw_parts(rp, 4, 4));
            drop(Vec::from_raw_parts(cp, 4, 4));
            Arc::decrement_strong_count(Arc::as_ptr(&registration));
        }
        assert_eq!(Arc::strong_count(&registration), 1);
        rt::release_forgotten_handle_token();
    }

    /// An inactive request releases its op clone with the request, after the
    /// free. Handle -1 makes the free a no-op in the C layer.
    #[test]
    #[cfg_attr(miri, ignore = "calls the C layer")]
    fn drop_of_an_inactive_request_releases_the_op() {
        let registration = Arc::new(OpRegistration { slot: 9 });
        let mut req = PersistentRequest::new(-1, vec![0u8; 4]);
        req.op = Some(Arc::clone(&registration));
        assert_eq!(Arc::strong_count(&registration), 2);
        drop(req);
        assert_eq!(Arc::strong_count(&registration), 1);
    }
}
