//! Request handles for nonblocking MPI operations.

use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ffi;
use crate::rt;
use crate::scope::Registry;
use crate::status::Status;
#[cfg(debug_assertions)]
use std::io::Write;
use std::mem::MaybeUninit;

/// Element count at or below which request-handle scratch buffers live on the
/// stack. Draining a handful-to-few-dozen in-flight requests on the completion
/// path (halo exchange, ping-pong, progress loops) then incurs no allocator
/// traffic; larger batches fall back to the heap. Used by both `Request` and
/// `PersistentRequest`. Mirrors `FERROMPI_REQ_STACK` in `csrc/ferrompi.c`.
const HANDLE_STACK_CAP: usize = 64;

/// Run `f` with the handles of `items` copied into a stack buffer when the
/// batch is small, falling back to a heap `Vec` only for large batches, and
/// a same-length zeroed `done` buffer for `f` to report completions into.
/// After `f` returns, calls `on_done` on every item whose `done` byte is
/// non-zero, whatever `f`'s result. Removes the per-call `Vec<i64>`
/// allocation on the completion path. Shared by `Request` and
/// `PersistentRequest`.
#[inline]
pub(crate) fn with_handles<E, R>(
    items: &mut [E],
    handle: impl Fn(&E) -> i64,
    f: impl FnOnce(&[i64], &mut [u8]) -> R,
    on_done: impl Fn(&mut E),
) -> R {
    let len = items.len();
    if len <= HANDLE_STACK_CAP {
        let mut hbuf = [0i64; HANDLE_STACK_CAP];
        let mut dbuf = [0u8; HANDLE_STACK_CAP];
        for (slot, item) in hbuf[..len].iter_mut().zip(items.iter()) {
            *slot = handle(item);
        }
        let result = f(&hbuf[..len], &mut dbuf[..len]);
        for (item, &done) in items.iter_mut().zip(&dbuf[..len]) {
            if done != 0 {
                on_done(item);
            }
        }
        result
    } else {
        let hvec: Vec<i64> = items.iter().map(&handle).collect();
        let mut dvec = vec![0u8; len];
        let result = f(&hvec, &mut dvec);
        for (item, &done) in items.iter_mut().zip(&dvec) {
            if done != 0 {
                on_done(item);
            }
        }
        result
    }
}

/// The handle a batch call passes for `r`: `-1` (the C layer's null
/// handle) once `r` has completed.
fn handle_or_skip(r: &Request<'_>) -> i64 {
    if r.completed {
        -1
    } else {
        r.handle
    }
}

/// Records that MPI completed `r`.
fn mark_completed(r: &mut Request<'_>) {
    r.completed = true;
    r.release();
}

/// Run `f` with a zeroed `i32` index scratch buffer and an uninitialized
/// status scratch buffer, both of length `len` and stack-allocated when small.
/// Used for the `*some` outputs: the shim writes `statuses[k]` for each
/// reported index, and nothing else.
#[inline]
fn with_index_buf<R>(
    len: usize,
    f: impl FnOnce(&mut [i32], &mut [MaybeUninit<ffi::FerrompiStatus>]) -> R,
) -> R {
    if len <= HANDLE_STACK_CAP {
        let mut buf = [0i32; HANDLE_STACK_CAP];
        let mut statuses = [const { MaybeUninit::uninit() }; HANDLE_STACK_CAP];
        f(&mut buf[..len], &mut statuses[..len])
    } else {
        let mut buf = vec![0i32; len];
        let mut statuses = Vec::with_capacity(len);
        f(&mut buf, &mut statuses.spare_capacity_mut()[..len])
    }
}

/// The `(index, Status)` pairs a `*some` shim reported. Empty unless `ret` is
/// success and `outcount` is positive.
fn some_results(
    ret: i32,
    outcount: i64,
    indices: &[i32],
    statuses: &[MaybeUninit<ffi::FerrompiStatus>],
) -> Vec<(usize, Status)> {
    if ret != 0 || outcount <= 0 {
        return Vec::new();
    }
    let n = outcount as usize;
    indices[..n]
        .iter()
        .zip(&statuses[..n])
        .map(|(&index, status)| {
            // SAFETY: the shim returned success with `outcount` positive, and then
            // writes every field of statuses[k] for each k < outcount (it
            // fills a status for each reported index on success and on
            // MPI_ERR_IN_STATUS, which it turns into success only when a
            // reported request failed). `n` is that outcount, and slicing to
            // it bounds-checks it against the buffer.
            let raw = unsafe { status.assume_init_read() };
            (index as usize, Status::from_ffi(raw))
        })
        .collect()
}

/// Check a batch wait/test return code, appending the failing request's
/// slice index to an `Error::Mpi` message. `failed` is the caller-slice
/// index of the failing request as the C shim wrote it (`-1` when it
/// found none). Shared by `Request::wait_all`, `Request::wait_any`,
/// `Request::test_any` and `PersistentRequest::wait_all`.
pub(crate) fn check_batch(ret: i32, operation: &'static str, failed: i64) -> Result<()> {
    match Error::check_with_op(ret, operation) {
        Err(Error::Mpi {
            class,
            code,
            message,
            operation,
        }) if failed >= 0 => Err(Error::Mpi {
            class,
            code,
            message: format!("{message} (request {failed})"),
            operation,
        }),
        other => other,
    }
}

/// The operation family a [`Request`] came from; only point-to-point
/// requests can be cancelled.
pub(crate) enum RequestKind {
    PointToPoint,
    Collective,
    #[cfg(feature = "rma")]
    Rma,
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
/// `test()` returning `Some`) or dropped.** MPI holds raw pointers to these
/// buffers; violating this invariant is undefined behavior.
///
/// For a request that is not created through a [`scope`](crate::scope) this
/// cannot currently be enforced by the Rust type system, because such a request
/// does not borrow its buffers.
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
/// A request created through a [`scope`](crate::scope) is completed by the
/// scope, which waits for it before returning or unwinding, so dropping it
/// early is harmless; it belongs to the thread that created it. The other
/// constructors still return requests that are not tied to a scope, and those
/// keep the behavior described above.
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
/// ferrompi::scope(|s| {
///     // Start nonblocking all-reduce
///     let request = world.iallreduce(s, &send, &mut recv, ReduceOp::Sum)?;
///
///     // Do other work while communication proceeds...
///
///     // Wait for completion
///     request.wait()?;
///     Ok(())
/// })
/// .unwrap();
///
/// // Now recv contains the result
/// println!("Sum: {:?}", recv);
/// ```
pub struct Request<'s> {
    handle: i64,
    completed: bool,
    kind: RequestKind,
    owner: Owner<'s>,
}

/// Who completes a request that was not waited for: the scope whose registry
/// holds its slot, or the request itself when it is dropped.
#[derive(Clone, Copy)]
enum Owner<'s> {
    Scoped(&'s Registry, u32),
    Unscoped,
}

impl Request<'static> {
    /// Create a request that no scope owns, from a raw handle.
    pub(crate) fn new(handle: i64, kind: RequestKind) -> Self {
        Request {
            handle,
            completed: false,
            kind,
            owner: Owner::Unscoped,
        }
    }
}

impl<'s> Request<'s> {
    /// Create a request owned by the scope whose registry holds `slot`.
    pub(crate) fn scoped(
        handle: i64,
        kind: RequestKind,
        registry: &'s Registry,
        slot: u32,
    ) -> Self {
        Request {
            handle,
            completed: false,
            kind,
            owner: Owner::Scoped(registry, slot),
        }
    }
}

impl Request<'_> {
    /// Frees this request's registry slot, once MPI completed the request.
    fn release(&self) {
        if let Owner::Scoped(registry, slot) = self.owner {
            registry.release(slot);
        }
    }

    /// Get the raw request handle (for advanced use).
    ///
    /// The value is an opaque table handle carrying a generation counter in
    /// its high bits; it names no request once this `Request` completes,
    /// because completion frees the underlying table slot for reuse.
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
    ///
    /// For a receive the returned [`Status`] holds the matched source, tag and
    /// element count. For a send, collective or RMA request the `Status` is the
    /// empty status, and so is a cancelled receive or a request that was
    /// already completed (a [`test`](Request::test) that failed after MPI
    /// completed it, say); only `error` has meaning, and this single-request
    /// call leaves it `None`.
    ///
    /// On a thread the active thread level does not allow, the wait is
    /// rejected and this call drops the still-in-flight `self` before
    /// returning, which aborts the process for a request that no scope owns (see
    /// the `Drop` impl below), except while `Mpi` is dropping or after it
    /// skipped `MPI_Finalize`: the wait then returns
    /// `Err(`[`Error::ThreadLevelViolation`]`)` and the request is leaked.
    ///
    /// In a debug build at `Serialized`, a wait that overlaps another
    /// thread's MPI call can neither run nor hand the request back, so it
    /// prints a message and aborts the process.
    #[inline]
    pub fn wait(self) -> Result<Status> {
        if self.completed {
            return Ok(Status::EMPTY);
        }
        let mut status = ffi::FerrompiStatus::default();
        self.wait_raw(&mut status)?;
        Ok(Status::from_ffi(status))
    }

    /// The body of [`wait`](Request::wait) for a request that is not yet
    /// completed, out of line so that the inline wrapper decodes `status` only
    /// when its caller uses the result.
    #[inline(never)]
    fn wait_raw(mut self, status: &mut ffi::FerrompiStatus) -> Result<()> {
        Error::check_with_op(rt::check_completion(), "wait")?;
        // SAFETY: self.handle is a valid MPI request handle registered in the
        // C-side request table by the nonblocking constructor that produced
        // this Request; self.completed was false on entry (checked by wait),
        // so MPI_Wait has not already consumed this handle. status is a valid
        // out-parameter.
        let ret = unsafe { ffi::ferrompi_wait(self.handle, status) };
        #[cfg(debug_assertions)]
        if ret == FERROMPI_ERR_THREAD_LEVEL {
            // rt::check_completion() above already returns this same sentinel
            // for a call from a non-init thread below Serialized, and
            // propagates it via `?` before reaching here; so this can only be
            // the Serialized overlap check rejecting the call before MPI saw
            // it. `self` cannot be handed back, and letting Drop wait would
            // overlap the other thread's MPI call.
            let _ = std::io::stderr().write_all(
                b"ferrompi: Request::wait overlapped another thread's MPI call at ThreadLevel::Serialized\n",
            );
            std::process::abort();
        }
        // MPI consumed the request whatever it returned; Drop must not wait on it again.
        self.completed = true;
        // A refused call leaves the request pending, so its slot stays live
        // for the scope end.
        if ret != FERROMPI_ERR_FINALIZED && ret != FERROMPI_ERR_THREAD_LEVEL {
            self.release();
        }
        Error::check_with_op(ret, "wait")
    }

    /// Test if this operation has completed without blocking.
    ///
    /// Returns `Some(`[`Status`]`)` if the operation is complete, `None`
    /// otherwise. For a receive the `Status` holds the matched source, tag and
    /// element count. For a send, collective or RMA request the `Status` is the
    /// empty status, and so is a cancelled receive or a request that was
    /// already completed; only `error` has meaning, and these single-request
    /// calls leave it `None`.
    ///
    /// # Note
    ///
    /// If this returns `Some`, the request is consumed and you should not call
    /// `wait()` or `test()` again. A test that fails with an error still marks
    /// the request completed when MPI completed it (e.g. a truncated
    /// receive), so a later `Drop` does not attempt a second `MPI_Wait` on the
    /// same slot.
    #[inline]
    pub fn test(&mut self) -> Result<Option<Status>> {
        if self.completed {
            return Ok(Some(Status::EMPTY));
        }
        let mut status = ffi::FerrompiStatus::default();
        Ok(self
            .test_raw(&mut status)?
            .then(|| Status::from_ffi(status)))
    }

    /// The body of [`test`](Request::test) for a request that is not yet
    /// completed, out of line like [`wait_raw`](Request::wait_raw). Returns
    /// whether MPI completed it.
    #[inline(never)]
    fn test_raw(&mut self, status: &mut ffi::FerrompiStatus) -> Result<bool> {
        let mut flag: i32 = 0;
        // SAFETY: self.handle is a valid MPI request handle registered in the
        // C-side request table; self.completed was false on entry (checked by
        // test), so MPI_Test has not already consumed this handle. flag is a
        // local out-parameter written before this function reads it below;
        // status is a valid out-parameter.
        let ret = unsafe { ffi::ferrompi_test(self.handle, &mut flag, status) };
        // Set completed from flag BEFORE the `?` below: ferrompi_test frees
        // the slot and reports flag=1 whenever MPI completed the request,
        // even when it returns an error (e.g. MPI_ERR_TRUNCATE), so Drop must
        // not re-wait on that now-freed handle.
        if flag != 0 {
            self.completed = true;
            self.release();
        }
        Error::check_with_op(ret, "test")?;
        Ok(flag != 0)
    }

    /// Wait for any one request in a collection to complete.
    ///
    /// Blocks until at least one not-yet-completed request completes and
    /// returns its index and [`Status`]. Completed entries are skipped;
    /// `wait_any` returns `Ok(None)` once every entry in the slice is completed
    /// (or the slice was empty), which is what lets the standard MPI Waitany
    /// loop idiom — calling `wait_any` repeatedly on the same slice without
    /// removing completed entries — terminate.
    ///
    /// For a receive the `Status` holds the matched source, tag and element
    /// count. For a send, collective or RMA request the `Status` is the empty
    /// status, and so is a cancelled receive; only `error` has meaning, and
    /// this call leaves it `None`.
    ///
    /// The completed `Request` is marked completed in place. Removing it
    /// from the vector is optional, not required for correctness.
    ///
    /// On a failed request, the returned error carries that request's own
    /// class and code. When the MPI library reports which request failed
    /// (MPICH and Open MPI do), the message also ends with `(request N)`,
    /// `N` being its index in `requests`.
    pub fn wait_any(requests: &mut [Request<'_>]) -> Result<Option<(usize, Status)>> {
        if requests.is_empty() {
            return Ok(None);
        }
        let mut index: i32 = 0;
        let mut status = ffi::FerrompiStatus::default();
        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles and a same-length [u8] done buffer, both sized to the count we
        // pass; index and status are valid stack-allocated output parameters.
        let ret = with_handles(
            requests,
            handle_or_skip,
            |handles, done| unsafe {
                ffi::ferrompi_waitany(
                    handles.len() as i64,
                    handles.as_ptr(),
                    &mut index,
                    done.as_mut_ptr(),
                    &mut status,
                )
            },
            mark_completed,
        );
        check_batch(ret, "waitany", index as i64)?;
        if index < 0 {
            return Ok(None);
        }
        Ok(Some((index as usize, Status::from_ffi(status))))
    }

    /// Wait until at least one request in a collection completes.
    ///
    /// Returns each completed request's index in `requests` and [`Status`], in
    /// completion order. A request that failed has `error: Some(class)`, and
    /// the others in the same call still complete and are reported. `Err` is
    /// returned only when the call itself fails. Returns `Ok(vec![])` when no
    /// requests were active (all null, all already completed, or a mix of the
    /// two).
    ///
    /// For a receive the `Status` holds the matched source, tag and element
    /// count. For an entry with `error: Some(_)` they are as MPI reported them,
    /// defined when the receive matched (a `Truncate`, say). For a send,
    /// collective or RMA request it is the empty status, and so is a cancelled
    /// receive; only `error` has meaning.
    ///
    /// Completed entries are skipped. The completed `Request`s are marked
    /// completed in place. Removing them from the vector is optional, not
    /// required for correctness.
    pub fn wait_some(requests: &mut [Request<'_>]) -> Result<Vec<(usize, Status)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let len = requests.len();
        let mut outcount: i64 = 0;
        let (ret, completed) = with_handles(
            requests,
            handle_or_skip,
            |handles, done| {
                with_index_buf(len, |indices, statuses| {
                    // SAFETY: with_handles / with_index_buf supply valid,
                    // appropriately-sized [i64] handle, [u8] done, [i32] index
                    // and status buffers whose lengths match `count`; outcount
                    // is a valid stack-allocated output parameter. MaybeUninit
                    // has the layout of FerrompiStatus.
                    let ret = unsafe {
                        ffi::ferrompi_waitsome(
                            handles.len() as i64,
                            handles.as_ptr(),
                            &mut outcount,
                            indices.as_mut_ptr(),
                            done.as_mut_ptr(),
                            statuses.as_mut_ptr().cast(),
                        )
                    };
                    // outcount == -1 means all null or a rejected result.
                    (ret, some_results(ret, outcount, indices, statuses))
                })
            },
            mark_completed,
        );
        Error::check_with_op(ret, "waitsome")?;
        Ok(completed)
    }

    /// Test whether any one request in a collection has completed (non-blocking).
    ///
    /// Returns `Ok(Some((idx, status)))` if a request completed, `Ok(None)` if
    /// no request has completed yet or all requests were already null.
    ///
    /// For a receive the [`Status`] holds the matched source, tag and element
    /// count. For a send, collective or RMA request the `Status` is the empty
    /// status, and so is a cancelled receive; only `error` has meaning, and
    /// this call leaves it `None`.
    ///
    /// Completed entries are skipped, so calling `test_any` again after every
    /// entry has completed keeps returning `Ok(None)` rather than erroring.
    /// The completed `Request` is marked completed in place. Removing it
    /// from the vector is optional, not required for correctness.
    ///
    /// On a failed request, the returned error carries that request's own
    /// class and code. When the MPI library reports which request failed
    /// (MPICH and Open MPI do), the message also ends with `(request N)`,
    /// `N` being its index in `requests`.
    pub fn test_any(requests: &mut [Request<'_>]) -> Result<Option<(usize, Status)>> {
        if requests.is_empty() {
            return Ok(None);
        }
        let mut index: i32 = 0;
        let mut flag: i32 = 0;
        let mut status = ffi::FerrompiStatus::default();
        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles and a same-length [u8] done buffer, both sized to the count we
        // pass; index, flag and status are valid stack-allocated output
        // parameters.
        let ret = with_handles(
            requests,
            handle_or_skip,
            |handles, done| unsafe {
                ffi::ferrompi_testany(
                    handles.len() as i64,
                    handles.as_ptr(),
                    &mut index,
                    &mut flag,
                    done.as_mut_ptr(),
                    &mut status,
                )
            },
            mark_completed,
        );
        check_batch(ret, "testany", index as i64)?;
        if flag == 0 {
            return Ok(None);
        }
        if index < 0 {
            // All requests were null — nothing to mark.
            return Ok(None);
        }
        Ok(Some((index as usize, Status::from_ffi(status))))
    }

    /// Test how many requests in a collection have completed (non-blocking).
    ///
    /// Returns each request that has completed at the moment of the call, as
    /// its index in `requests` and [`Status`], in completion order. A request
    /// that failed has `error: Some(class)`, and the others in the same call
    /// still complete and are reported. `Err` is returned only when the call
    /// itself fails. Returns `Ok(vec![])` when none have completed or all were
    /// null.
    ///
    /// For a receive the `Status` holds the matched source, tag and element
    /// count. For an entry with `error: Some(_)` they are as MPI reported them,
    /// defined when the receive matched (a `Truncate`, say). For a send,
    /// collective or RMA request it is the empty status, and so is a cancelled
    /// receive; only `error` has meaning.
    ///
    /// Completed entries are skipped. The completed `Request`s are marked
    /// completed in place. Removing them from the vector is optional, not
    /// required for correctness.
    pub fn test_some(requests: &mut [Request<'_>]) -> Result<Vec<(usize, Status)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        let len = requests.len();
        let mut outcount: i64 = 0;
        let (ret, completed) = with_handles(
            requests,
            handle_or_skip,
            |handles, done| {
                with_index_buf(len, |indices, statuses| {
                    // SAFETY: with_handles / with_index_buf supply valid,
                    // appropriately-sized [i64] handle, [u8] done, [i32] index
                    // and status buffers whose lengths match `count`; outcount
                    // is a valid stack-allocated output parameter. MaybeUninit
                    // has the layout of FerrompiStatus.
                    let ret = unsafe {
                        ffi::ferrompi_testsome(
                            handles.len() as i64,
                            handles.as_ptr(),
                            &mut outcount,
                            indices.as_mut_ptr(),
                            done.as_mut_ptr(),
                            statuses.as_mut_ptr().cast(),
                        )
                    };
                    // outcount == -1 means all null; 0 means none completed yet.
                    (ret, some_results(ret, outcount, indices, statuses))
                })
            },
            mark_completed,
        );
        Error::check_with_op(ret, "testsome")?;
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

    /// Request cancellation of a pending point-to-point operation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotSupported`] without calling into MPI when `self`
    /// is a nonblocking-collective or RMA request: the MPI standard defines
    /// `MPI_Cancel` only for point-to-point requests, and both MPICH and
    /// Open MPI reject it for other request kinds. The request is left
    /// pending; follow up with [`wait`](Request::wait) as usual.
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
    /// ferrompi::scope(|s| {
    ///     let mut req = world.irecv(s, &mut buf, 0, 0)?;
    ///     req.cancel()?;
    ///     req.wait()?;
    ///     Ok(())
    /// })?;
    /// # Ok(()) }
    /// ```
    pub fn cancel(&mut self) -> Result<()> {
        match self.kind {
            RequestKind::PointToPoint => {}
            RequestKind::Collective => {
                return Err(Error::NotSupported(
                    "cancel of a nonblocking collective request".into(),
                ));
            }
            #[cfg(feature = "rma")]
            RequestKind::Rma => {
                return Err(Error::NotSupported("cancel of an RMA request".into()));
            }
        }
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
    /// [`PersistentRequest::wait_all`](crate::PersistentRequest::wait_all).
    /// Whatever the result, every request MPI completed — one that completed
    /// with an error included — is marked completed in place; the others
    /// stay pending. The same policy applies to `PersistentRequest::wait_all`.
    ///
    /// On a failed request, the returned error carries that request's own
    /// class and code, and its message ends with `(request N)`, `N` being
    /// its index in `requests`.
    pub fn wait_all(requests: &mut [Request<'_>]) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let mut failed: i64 = -1;
        // SAFETY: with_handles provides a valid, contiguous [i64] of the request
        // handles and a same-length [u8] done buffer, both sized to the count we
        // pass; failed is a valid stack-allocated i64 output parameter.
        let ret = with_handles(
            requests,
            handle_or_skip,
            |handles, done| unsafe {
                ffi::ferrompi_waitall(
                    handles.len() as i64,
                    handles.as_ptr(),
                    done.as_mut_ptr(),
                    &mut failed,
                )
            },
            mark_completed,
        );

        check_batch(ret, "waitall", failed)
    }
}

impl Drop for Request<'_> {
    /// Block until the in-flight operation completes, then release the handle.
    ///
    /// Calls `MPI_Wait` on the underlying request handle when `self.completed`
    /// is `false` and no scope owns the request. **This call blocks** until the
    /// peer posts the matching operation; if the peer never does, this
    /// deadlocks. A scoped request does nothing here: its scope completes it.
    ///
    /// Maintainers: the `self.completed = true` assignment in
    /// `Request::wait_raw` is the only guard that prevents a double-wait here.
    /// Any refactoring of `wait()` must preserve that assignment, or this
    /// `Drop` impl becomes unsound (double-freeing the request handle).
    ///
    /// After `Mpi` is dropped this does nothing; below `Serialized` on a
    /// non-init thread it aborts the process.
    fn drop(&mut self) {
        if !self.completed && matches!(self.owner, Owner::Unscoped) {
            let Some(_call) = rt::drop_guard("Request") else {
                return;
            };
            // SAFETY: self.handle is a valid MPI request handle registered in the
            // C-side request table by the nonblocking constructor (e.g., iallreduce).
            // The handle has not been freed because self.completed is false, meaning
            // wait() was never called. ferrompi_wait calls MPI_Wait which frees the
            // handle on success; the completed flag guards against a double-free.
            // Calls the unguarded raw wrapper (not the lifecycle-guarded one):
            // rt::drop_guard above already handles the FFI lifecycle check, so
            // this call must still attempt the wait once reached.
            unsafe { ffi::raw::ferrompi_wait(self.handle, std::ptr::null_mut()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        some_results, with_handles, with_index_buf, Owner, Request, RequestKind, HANDLE_STACK_CAP,
    };
    use crate::error::Error;
    use crate::ffi::FerrompiStatus;
    use crate::status::{Source, Status, Tag};
    use std::mem::{forget, MaybeUninit};

    fn test_request(completed: bool, kind: RequestKind) -> Request<'static> {
        Request {
            handle: 0,
            completed,
            kind,
            owner: Owner::Unscoped,
        }
    }

    #[test]
    fn test_when_already_completed_returns_empty_status() {
        let mut req = test_request(true, RequestKind::PointToPoint);
        let result = req.test();
        assert!(matches!(result, Ok(Some(Status::EMPTY))));
        forget(req);
    }

    #[test]
    fn wait_when_already_completed_returns_ok() {
        // wait() takes self by value (consuming).
        // With completed: true, it returns Ok(Status::EMPTY) before any FFI call.
        // Drop then runs, but !self.completed is false, so Drop is a no-op.
        let req = test_request(true, RequestKind::PointToPoint);
        let result = req.wait();
        assert!(matches!(result, Ok(Status::EMPTY)));
        // No forget() needed — wait() consumed the value, and Drop was a no-op
    }

    #[test]
    fn wait_all_empty_slice_returns_ok() {
        let result = Request::wait_all(&mut []);
        assert!(result.is_ok());
    }

    #[test]
    fn with_handles_passes_every_handle_in_order() {
        let mut stack: Vec<i64> = (0..3).collect();
        let expected = stack.clone();
        let seen = with_handles(
            &mut stack,
            |h| *h,
            |handles, _done| handles.to_vec(),
            |_| {},
        );
        assert_eq!(seen, expected);

        let mut heap: Vec<i64> = (0..(HANDLE_STACK_CAP as i64 + 1)).collect();
        let expected = heap.clone();
        let seen = with_handles(&mut heap, |h| *h, |handles, _done| handles.to_vec(), |_| {});
        assert_eq!(seen, expected);
    }

    #[test]
    fn with_handles_calls_on_done_for_done_items() {
        // Stack path (len=3): the C side reports index 1 done.
        let mut marks = vec![0u8; 3];
        with_handles(
            &mut marks,
            |_| 0i64,
            |_handles, done| done[1] = 1,
            |item| *item = 1,
        );
        assert_eq!(marks, vec![0, 1, 0]);

        // Heap path (len=HANDLE_STACK_CAP+1=65): the C side reports index 64 done.
        let len = HANDLE_STACK_CAP + 1;
        let mut marks = vec![0u8; len];
        with_handles(
            &mut marks,
            |_| 0i64,
            |_handles, done| done[len - 1] = 1,
            |item| *item = 1,
        );
        let mut expected = vec![0u8; len];
        expected[len - 1] = 1;
        assert_eq!(marks, expected);
    }

    #[test]
    fn with_index_buf_hands_out_len_entries_on_both_paths() {
        for len in [3, HANDLE_STACK_CAP, HANDLE_STACK_CAP + 1] {
            let lens = with_index_buf(len, |indices, statuses| (indices.len(), statuses.len()));
            assert_eq!(lens, (len, len));
        }
    }

    #[test]
    fn some_results_reports_only_the_filled_entries_on_success() {
        let raw = |source, tag, error| {
            MaybeUninit::new(FerrompiStatus {
                source,
                tag,
                count: 2,
                error,
            })
        };
        let statuses = [raw(1, 21, 0), raw(4, 9, 0), MaybeUninit::uninit()];
        let indices = [2, 0, 0];
        let got = some_results(0, 2, &indices, &statuses);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, 2);
        assert_eq!(got[0].1.source, Source::Rank(1));
        assert_eq!(got[0].1.tag, Tag::Value(21));
        assert_eq!(got[1].0, 0);
        assert_eq!(got[1].1.source, Source::Rank(4));
        assert!(some_results(0, 0, &indices, &statuses).is_empty());
        assert!(some_results(0, -1, &indices, &statuses).is_empty());
        assert!(some_results(1, 2, &indices, &statuses).is_empty());
    }

    #[test]
    fn wait_any_empty_vec_returns_none() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::wait_any(&mut v).unwrap(), None);
    }

    #[test]
    fn wait_some_empty_vec_returns_empty() {
        let mut v: Vec<Request> = vec![];
        assert!(Request::wait_some(&mut v).unwrap().is_empty());
    }

    #[test]
    fn test_any_empty_vec_returns_none() {
        let mut v: Vec<Request> = vec![];
        assert_eq!(Request::test_any(&mut v).unwrap(), None);
    }

    #[test]
    fn test_some_empty_vec_returns_empty() {
        let mut v: Vec<Request> = vec![];
        assert!(Request::test_some(&mut v).unwrap().is_empty());
    }

    #[test]
    fn get_status_on_completed_request_returns_true_without_ffi() {
        let req = test_request(true, RequestKind::PointToPoint);
        let result = req.get_status();
        assert!(matches!(result, Ok(true)));
        forget(req);
    }

    #[test]
    fn cancel_on_completed_request_returns_ok_without_ffi() {
        let mut req = test_request(true, RequestKind::PointToPoint);
        let result = req.cancel();
        assert!(matches!(result, Ok(())));
        forget(req);
    }

    fn assert_cancel_not_supported(kind: RequestKind) {
        let mut req = test_request(false, kind);
        let result = req.cancel();
        assert!(matches!(result, Err(Error::NotSupported(_))));
        forget(req);
    }

    #[test]
    fn cancel_on_collective_request_returns_not_supported_without_ffi() {
        assert_cancel_not_supported(RequestKind::Collective);
    }

    #[cfg(feature = "rma")]
    #[test]
    fn cancel_on_rma_request_returns_not_supported_without_ffi() {
        assert_cancel_not_supported(RequestKind::Rma);
    }
}
