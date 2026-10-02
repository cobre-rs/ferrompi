//! Point-to-point communication: send, recv, isend, irecv, the persistent send/recv
//! constructors, sendrecv, probe, iprobe.

use crate::comm::Communicator;
use crate::datatype::{buf, buf_mut, MpiDatatype, PlainData};
use crate::datatype_builder::CustomDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::{Request, RequestKind};
use crate::status::{Source, Status, Tag};

impl Communicator {
    /// Send a slice of values to another process.
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag` - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = vec![1.0f64, 2.0, 3.0];
    /// world.send(&data, 1, 0).unwrap();
    /// ```
    pub fn send<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<()> {
        let dest = dest.into().dest_code()?;
        let (p, n, dt) = buf(data);
        // SAFETY: this blocking call returns only after MPI is done with the buffer.
        let ret = unsafe { ffi::ferrompi_send(p, n, dt, dest, tag, self.handle) };
        Error::check_with_op(ret, "send")
    }

    /// Receive a slice of values from another process.
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// Returns `(source, tag, count)`; `count` is `-1` when the message is not
    /// a whole number of `T`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut buf = vec![0.0f64; 10];
    /// let (source, tag, count) = world.recv(&mut buf, 0, 0).unwrap();
    /// ```
    pub fn recv<T: MpiDatatype>(
        &self,
        data: &mut [T],
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<(i32, i32, i64)> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        let mut status = ffi::FerrompiStatus::default();

        let (p, n, dt) = buf_mut(data);
        // SAFETY: this blocking call returns only after MPI is done with the buffer.
        let ret = unsafe { ffi::ferrompi_recv(p, n, dt, source, tag, self.handle, &mut status) };
        Error::check_with_op(ret, "recv")?;
        Ok((status.source, status.tag, status.count))
    }

    /// Nonblocking send.
    ///
    /// Initiates a send operation and returns immediately with a [`Request`]
    /// handle. The send buffer **must not be modified** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send (must remain valid until the request completes)
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag` - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = vec![1.0f64, 2.0, 3.0];
    /// let req = world.isend(&data, 1, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn isend<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<Request> {
        let dest = dest.into().dest_code()?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf(data);
        // SAFETY: the returned Request does not borrow data; keeping it alive and unmodified
        // until completion is the caller's obligation, which this signature does not enforce.
        let ret =
            unsafe { ffi::ferrompi_isend(p, n, dt, dest, tag, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "isend")?;
        Ok(Request::new(request_handle, RequestKind::PointToPoint))
    }

    /// Nonblocking receive.
    ///
    /// Initiates a receive operation and returns immediately with a [`Request`]
    /// handle. The receive buffer **must not be read** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// # Arguments
    ///
    /// * `data` - Receive buffer (must remain valid until the request completes)
    /// * `source` - Source rank, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag` - Message tag or [`Tag::Any`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut buf = vec![0.0f64; 10];
    /// let req = world.irecv(&mut buf, 0, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn irecv<T: MpiDatatype>(
        &self,
        data: &mut [T],
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<Request> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: the returned Request does not borrow data; keeping it alive and unread
        // until completion is the caller's obligation, which this signature does not enforce.
        let ret =
            unsafe { ffi::ferrompi_irecv(p, n, dt, source, tag, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "irecv")?;
        Ok(Request::new(request_handle, RequestKind::PointToPoint))
    }

    /// Blocking send-receive.
    ///
    /// Sends data to one process and receives from another (or the same) in a
    /// single operation. This is useful for avoiding deadlocks in ring-style
    /// communication patterns where each process both sends and receives.
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// Returns `(source, tag, count)`; `count` is `-1` when the message is not
    /// a whole number of `T`.
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `send` - Buffer to send
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `sendtag` - Send message tag
    /// * `recv` - Receive buffer
    /// * `source` - Source rank, [`Source::Any`] or [`Source::ProcNull`]
    /// * `recvtag` - Receive message tag or [`Tag::Any`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 5];
    /// let mut recv = vec![0.0f64; 5];
    /// let next = (world.rank() + 1) % world.size();
    /// let prev = (world.rank() - 1 + world.size()) % world.size();
    /// let (src, tag, count) = world.sendrecv(&send, next, 0, &mut recv, prev, 0).unwrap();
    /// ```
    pub fn sendrecv<T: MpiDatatype>(
        &self,
        send: &[T],
        dest: impl Into<Source>,
        sendtag: i32,
        recv: &mut [T],
        source: impl Into<Source>,
        recvtag: impl Into<Tag>,
    ) -> Result<(i32, i32, i64)> {
        let dest = dest.into().dest_code()?;
        let source = source.into().source_code("source")?;
        let recvtag = recvtag.into().tag_code("recvtag")?;
        let mut status = ffi::FerrompiStatus::default();

        let (sp, sn, sdt) = buf(send);
        let (rp, rn, rdt) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]); this blocking call returns
        // only after MPI is done with both buffers.
        let ret = unsafe {
            ffi::ferrompi_sendrecv(
                sp,
                sn,
                sdt,
                dest,
                sendtag,
                rp,
                rn,
                rdt,
                source,
                recvtag,
                self.handle,
                &mut status,
            )
        };
        Error::check_with_op(ret, "sendrecv")?;
        Ok((status.source, status.tag, status.count))
    }

    /// Blocking probe for an incoming message.
    ///
    /// Waits until a matching message is available and returns status
    /// information (source rank, tag, element count) without actually
    /// receiving the message. This is useful for determining the size of an
    /// incoming message before allocating a receive buffer.
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// The type parameter `T` determines the MPI datatype used by
    /// `MPI_Get_count` to compute the element count in the returned
    /// [`Status`].
    ///
    /// # Arguments
    ///
    /// * `source` - Source rank to match, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag` - Message tag to match or [`Tag::Any`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Source, Tag};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Probe for any incoming f64 message
    /// let status = world.probe::<f64>(Source::Any, Tag::Any).unwrap();
    /// // Allocate a buffer of exactly the right size
    /// assert!(status.count >= 0, "message is not a whole number of f64");
    /// let mut buf = vec![0.0f64; status.count as usize];
    /// world.recv(&mut buf, status.source, status.tag).unwrap();
    /// ```
    pub fn probe<T: MpiDatatype>(
        &self,
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<Status> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        let mut status = ffi::FerrompiStatus::default();

        // SAFETY: all arguments are scalar integers or exclusive output pointers; self.handle is owned.
        let ret =
            unsafe { ffi::ferrompi_probe(source, tag, self.handle, &mut status, T::TAG as i32) };
        Error::check_with_op(ret, "probe")?;
        Ok(Status {
            source: status.source,
            tag: status.tag,
            count: status.count,
        })
    }

    /// Nonblocking probe for an incoming message.
    ///
    /// Checks whether a matching message is available without blocking.
    /// Returns `Some(Status)` if a message is available, `None` otherwise.
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// The type parameter `T` determines the MPI datatype used by
    /// `MPI_Get_count` to compute the element count in the returned
    /// [`Status`].
    ///
    /// # Arguments
    ///
    /// * `source` - Source rank to match, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag` - Message tag to match or [`Tag::Any`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Source, Tag};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Poll for an incoming f64 message without blocking
    /// if let Some(status) = world.iprobe::<f64>(Source::Any, Tag::Any).unwrap() {
    ///     assert!(status.count >= 0, "message is not a whole number of f64");
    ///     let mut buf = vec![0.0f64; status.count as usize];
    ///     world.recv(&mut buf, status.source, status.tag).unwrap();
    /// }
    /// ```
    pub fn iprobe<T: MpiDatatype>(
        &self,
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<Option<Status>> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        let mut flag: i32 = 0;
        let mut status = ffi::FerrompiStatus::default();

        // SAFETY: all arguments are scalar integers or exclusive output pointers; self.handle is owned.
        let ret = unsafe {
            ffi::ferrompi_iprobe(
                source,
                tag,
                self.handle,
                &mut flag,
                &mut status,
                T::TAG as i32,
            )
        };
        Error::check_with_op(ret, "iprobe")?;
        if flag != 0 {
            Ok(Some(Status {
                source: status.source,
                tag: status.tag,
                count: status.count,
            }))
        } else {
            Ok(None)
        }
    }

    // ========================================================================
    // Persistent Point-to-Point (MPI 1.1+)
    // ========================================================================

    /// Initialize a persistent send operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// The caller must not modify `data` while the request is active
    /// (between `start()` and `wait()`).
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag`  - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 100];
    /// let mut req = world.send_init(&send, 1, 7).unwrap();
    /// for _ in 0..10 {
    ///     req.start().unwrap();
    ///     req.wait().unwrap();
    /// }
    /// ```
    pub fn send_init<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let dest = dest.into().dest_code()?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf(data);
        // SAFETY: the returned PersistentRequest records `data`'s pointer until the request
        // is freed and does not borrow it; keeping `data` alive and untouched between
        // start() and completion is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_send_init(p, n, dt, dest, tag, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "send_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent buffered-mode send operation.
    ///
    /// Buffered sends copy the outgoing message into a user-attached buffer
    /// and complete immediately at the local side, regardless of whether the
    /// destination has posted a matching receive. The returned handle can be
    /// started multiple times with `start()`.
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// # Buffer Requirement
    ///
    /// A buffer must be attached via `Mpi::buffer_attach` **before** `start()` is
    /// called on this request. If no buffer is attached when `start()` fires,
    /// MPI will return an error.
    ///
    /// The recommended buffer size is `MPI_BSEND_OVERHEAD + sum(send sizes)`.
    /// `MPI_BSEND_OVERHEAD` is implementation-specific (typically a few hundred
    /// bytes); use a generous margin in practice.
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag`  - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Attach a 64 KiB buffer before creating buffered send requests.
    /// mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice()).unwrap();
    ///
    /// let send = vec![1.0f64; 100];
    /// let mut req = world.bsend_init(&send, 1, 7).unwrap();
    /// for _ in 0..10 {
    ///     req.start().unwrap();
    ///     req.wait().unwrap();
    /// }
    ///
    /// let _ = mpi.buffer_detach().unwrap();
    /// ```
    pub fn bsend_init<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let dest = dest.into().dest_code()?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf(data);
        // SAFETY: the returned PersistentRequest records `data`'s pointer until the request
        // is freed and does not borrow it; keeping `data` alive and untouched between
        // start() and completion is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_bsend_init(p, n, dt, dest, tag, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "bsend_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent ready-mode send operation.
    ///
    /// Ready-mode sends skip the MPI protocol negotiation step and are a
    /// performance optimization. The returned handle can be started multiple
    /// times with `start()`.
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// # Safety Contract
    ///
    /// The matching receive **must** be posted on the destination rank before
    /// `start()` is called on this request. This means the destination must
    /// have already called `recv_init` + `start()`, `irecv`, or `recv` before
    /// the sender calls `start()` here.
    ///
    /// Failure to ensure this is **undefined behavior in MPI**: it typically
    /// results in a hang, but it may also cause a crash or silent data
    /// corruption depending on the MPI implementation.
    ///
    /// The Rust borrow checker cannot enforce this ordering — it is a runtime
    /// contract between communicating processes. In tests, use an explicit
    /// `barrier()` after the receiver posts its receive and before the sender
    /// calls `start()` to ensure the ordering is respected.
    ///
    /// The caller must not modify `data` while the request is active
    /// (between `start()` and `wait()`).
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag`  - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Safety contract: receiver must post recv before we call start().
    /// // Use a barrier to guarantee the ordering.
    /// let send = vec![42.0f64; 10];
    /// let mut req = world.rsend_init(&send, 1, 7).unwrap();
    /// world.barrier().unwrap(); // recv on rank 1 is posted by now
    /// req.start().unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn rsend_init<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let dest = dest.into().dest_code()?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf(data);
        // SAFETY: the returned PersistentRequest records `data`'s pointer until the request
        // is freed and does not borrow it; keeping `data` alive and untouched between
        // start() and completion is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_rsend_init(p, n, dt, dest, tag, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "rsend_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent synchronous-mode send operation.
    ///
    /// Synchronous-mode sends complete only after the matching receive has
    /// begun on the destination rank. Unlike standard sends, the MPI
    /// implementation cannot buffer the message internally: `wait()` on this
    /// request blocks until the receiver has started its matching receive.
    ///
    /// This eliminates the possibility of silent buffering, making it useful
    /// for debugging deadlocks and for algorithms that require a strict
    /// sender/receiver handshake. The trade-off is reduced throughput compared
    /// to standard or buffered sends.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// The caller must not modify `data` while the request is active
    /// (between `start()` and `wait()`).
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank or [`Source::ProcNull`]
    /// * `tag`  - Message tag
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1i32; 5];
    /// let mut req = world.ssend_init(&send, 1, 3).unwrap();
    /// for _ in 0..5 {
    ///     req.start().unwrap();
    ///     req.wait().unwrap(); // returns only after receiver has started
    /// }
    /// ```
    pub fn ssend_init<T: MpiDatatype>(
        &self,
        data: &[T],
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let dest = dest.into().dest_code()?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf(data);
        // SAFETY: the returned PersistentRequest records `data`'s pointer until the request
        // is freed and does not borrow it; keeping `data` alive and untouched between
        // start() and completion is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_ssend_init(p, n, dt, dest, tag, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "ssend_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent receive operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// # Arguments
    ///
    /// * `data`   - Receive buffer (must remain valid for lifetime of handle)
    /// * `source` - Source rank, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag`    - Message tag or [`Tag::Any`]
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut recv = vec![0.0f64; 100];
    /// let mut req = world.recv_init(&mut recv, 0, 7).unwrap();
    /// for _ in 0..10 {
    ///     req.start().unwrap();
    ///     req.wait().unwrap();
    /// }
    /// ```
    pub fn recv_init<T: MpiDatatype>(
        &self,
        data: &mut [T],
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<PersistentRequest> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: the returned PersistentRequest records `data`'s pointer until the request
        // is freed and does not borrow it; keeping `data` alive and untouched between
        // start() and completion is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_recv_init(p, n, dt, source, tag, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "recv_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Send a slice of values to another process using a committed custom datatype.
    ///
    /// This is the custom-datatype counterpart of [`send`](Self::send). The element
    /// type `T` must satisfy the [`PlainData`](crate::PlainData) bound. `datatype`'s
    /// extent must equal `size_of::<T>()` and its data must lie within one `T`,
    /// otherwise the call returns [`Error::InvalidArgument`] without calling MPI.
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `buf`      - Buffer to send; MPI count is `buf.len()`
    /// * `datatype` - Committed custom datatype describing each element
    /// * `dest`     - Destination rank or [`Source::ProcNull`]
    /// * `tag`      - Message tag
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `datatype`'s extent does not equal
    ///   `size_of::<T>()`, or its data does not lie within one `T` — checked
    ///   locally before any MPI call. If a peer already posted the matching
    ///   receive, that peer operation is not cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    /// # let _mpi = Mpi::init().unwrap();
    /// # let world = _mpi.world();
    /// #[repr(C)]
    /// #[derive(Clone, Copy)]
    /// struct Pair { v: f64, i: i32 }
    /// // SAFETY: Pair is #[repr(C)] of an f64 and an i32, so any bit pattern is valid.
    /// unsafe impl ferrompi::PlainData for Pair {}
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// let buf = [Pair { v: 1.23456789, i: 42 }];
    /// world.send_custom(&buf, &dt, 1, 0).unwrap();
    /// ```
    pub fn send_custom<T: PlainData>(
        &self,
        buf: &[T],
        datatype: &CustomDatatype,
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<()> {
        let dest = dest.into().dest_code()?;
        datatype.check_layout::<T>()?;
        // SAFETY: buf.as_ptr() is valid for buf.len() elements; datatype.handle is an owned,
        // committed CustomDatatype; the buffer outlives this blocking call; check_layout above
        // guarantees each element's data lies within its T, so MPI touches only buf.
        let ret = unsafe {
            ffi::ferrompi_send_custom(
                buf.as_ptr().cast::<std::ffi::c_void>(),
                buf.len() as i64,
                datatype.handle,
                dest,
                tag,
                self.handle,
            )
        };
        Error::check_with_op(ret, "send_custom")
    }

    /// Receive a slice of values from another process using a committed custom datatype.
    ///
    /// This is the custom-datatype counterpart of [`recv`](Self::recv). The element
    /// type `T` must satisfy the [`PlainData`](crate::PlainData) bound. `datatype`'s
    /// extent must equal `size_of::<T>()` and its data must lie within one `T`,
    /// otherwise the call returns [`Error::InvalidArgument`] without calling MPI.
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// Returns a [`Status`] whose `count` is the number of `T` elements
    /// received; `count` is `-1` when the message is not a whole number of `T`.
    ///
    /// # Arguments
    ///
    /// * `buf`      - Receive buffer; MPI count is `buf.len()`
    /// * `datatype` - Committed custom datatype describing each element
    /// * `source`   - Source rank, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag`      - Message tag or [`Tag::Any`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `datatype`'s extent does not equal
    ///   `size_of::<T>()`, or its data does not lie within one `T` — checked
    ///   locally before any MPI call. If a peer already posted the matching
    ///   send, that peer operation is not cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    /// # let _mpi = Mpi::init().unwrap();
    /// # let world = _mpi.world();
    /// #[repr(C)]
    /// #[derive(Clone, Copy)]
    /// struct Pair { v: f64, i: i32 }
    /// // SAFETY: Pair is #[repr(C)] of an f64 and an i32, so any bit pattern is valid.
    /// unsafe impl ferrompi::PlainData for Pair {}
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// let mut buf = [Pair { v: 0.0, i: 0 }];
    /// let status = world.recv_custom(&mut buf, &dt, 0, 0).unwrap();
    /// assert_eq!(status.count, 1);
    /// ```
    pub fn recv_custom<T: PlainData>(
        &self,
        buf: &mut [T],
        datatype: &CustomDatatype,
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<Status> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        datatype.check_layout::<T>()?;
        let mut status = ffi::FerrompiStatus::default();

        // SAFETY: buf.as_mut_ptr() is exclusively writable for buf.len() elements; datatype.handle
        // is an owned, committed CustomDatatype; the buffer outlives this blocking call; the
        // PlainData bound on T makes any bytes MPI writes a valid T, and check_layout above
        // guarantees each element's data lies within its T, so MPI touches only buf.
        let ret = unsafe {
            ffi::ferrompi_recv_custom(
                buf.as_mut_ptr().cast::<std::ffi::c_void>(),
                buf.len() as i64,
                datatype.handle,
                source,
                tag,
                self.handle,
                &mut status,
            )
        };
        Error::check_with_op(ret, "recv_custom")?;
        Ok(Status {
            source: status.source,
            tag: status.tag,
            count: status.count,
        })
    }

    /// Nonblocking send using a committed custom datatype.
    ///
    /// This is the custom-datatype counterpart of [`isend`](Self::isend). The
    /// send buffer **must not be modified** until the request is completed via
    /// [`Request::wait()`] or [`Request::test()`].
    ///
    /// The element type `T` must satisfy the [`PlainData`](crate::PlainData) bound.
    /// `datatype`'s extent must equal `size_of::<T>()` and its data must lie
    /// within one `T`, otherwise the call returns [`Error::InvalidArgument`]
    /// without calling MPI.
    ///
    /// A `dest` of `Source::ProcNull` sends nothing and completes at once;
    /// `Source::Any` or a negative rank returns [`Error::InvalidArgument`] before
    /// any MPI call.
    ///
    /// # Arguments
    ///
    /// * `buf`      - Buffer to send (must remain valid until the request completes)
    /// * `datatype` - Committed custom datatype describing each element
    /// * `dest`     - Destination rank or [`Source::ProcNull`]
    /// * `tag`      - Message tag
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `datatype`'s extent does not equal
    ///   `size_of::<T>()`, or its data does not lie within one `T` — checked
    ///   locally before any MPI call. If a peer already posted the matching
    ///   receive, that peer operation is not cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    /// # let _mpi = Mpi::init().unwrap();
    /// # let world = _mpi.world();
    /// #[repr(C)]
    /// #[derive(Clone, Copy)]
    /// struct Pair { v: f64, i: i32 }
    /// // SAFETY: Pair is #[repr(C)] of an f64 and an i32, so any bit pattern is valid.
    /// unsafe impl ferrompi::PlainData for Pair {}
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// let buf = [Pair { v: 1.23456789, i: 42 }];
    /// let req = world.isend_custom(&buf, &dt, 1, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn isend_custom<T: PlainData>(
        &self,
        buf: &[T],
        datatype: &CustomDatatype,
        dest: impl Into<Source>,
        tag: i32,
    ) -> Result<Request> {
        let dest = dest.into().dest_code()?;
        datatype.check_layout::<T>()?;
        let mut request_handle: i64 = 0;
        // SAFETY: buf.as_ptr() is valid for buf.len() elements; datatype.handle is an owned,
        // committed CustomDatatype; the caller must keep the buffer alive until Request completion;
        // check_layout above guarantees each element's data lies within its T, so MPI touches only buf.
        let ret = unsafe {
            ffi::ferrompi_isend_custom(
                buf.as_ptr().cast::<std::ffi::c_void>(),
                buf.len() as i64,
                datatype.handle,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "isend_custom")?;
        Ok(Request::new(request_handle, RequestKind::PointToPoint))
    }

    /// Nonblocking receive using a committed custom datatype.
    ///
    /// This is the custom-datatype counterpart of [`irecv`](Self::irecv). The
    /// receive buffer **must not be read** until the request is completed via
    /// [`Request::wait()`] or [`Request::test()`].
    ///
    /// `Source::Any`/`Tag::Any` match any source/tag; `Source::ProcNull` completes
    /// at once with no data.
    ///
    /// The element type `T` must satisfy the [`PlainData`](crate::PlainData) bound.
    /// `datatype`'s extent must equal `size_of::<T>()` and its data must lie
    /// within one `T`, otherwise the call returns [`Error::InvalidArgument`]
    /// without calling MPI.
    ///
    /// # Arguments
    ///
    /// * `buf`      - Receive buffer (must remain valid until the request completes)
    /// * `datatype` - Committed custom datatype describing each element
    /// * `source`   - Source rank, [`Source::Any`] or [`Source::ProcNull`]
    /// * `tag`      - Message tag or [`Tag::Any`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] if `datatype`'s extent does not equal
    ///   `size_of::<T>()`, or its data does not lie within one `T` — checked
    ///   locally before any MPI call. If a peer already posted the matching
    ///   send, that peer operation is not cancelled.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    /// # let _mpi = Mpi::init().unwrap();
    /// # let world = _mpi.world();
    /// #[repr(C)]
    /// #[derive(Clone, Copy)]
    /// struct Pair { v: f64, i: i32 }
    /// // SAFETY: Pair is #[repr(C)] of an f64 and an i32, so any bit pattern is valid.
    /// unsafe impl ferrompi::PlainData for Pair {}
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// let mut buf = [Pair { v: 0.0, i: 0 }];
    /// let req = world.irecv_custom(&mut buf, &dt, 0, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn irecv_custom<T: PlainData>(
        &self,
        buf: &mut [T],
        datatype: &CustomDatatype,
        source: impl Into<Source>,
        tag: impl Into<Tag>,
    ) -> Result<Request> {
        let source = source.into().source_code("source")?;
        let tag = tag.into().tag_code("tag")?;
        datatype.check_layout::<T>()?;
        let mut request_handle: i64 = 0;
        // SAFETY: buf.as_mut_ptr() is exclusively writable for buf.len() elements; datatype.handle
        // is an owned, committed CustomDatatype; the caller must not read the buffer until
        // Request completion; the PlainData bound on T makes any bytes MPI writes a valid T, and
        // check_layout above guarantees each element's data lies within its T, so MPI touches only buf.
        let ret = unsafe {
            ffi::ferrompi_irecv_custom(
                buf.as_mut_ptr().cast::<std::ffi::c_void>(),
                buf.len() as i64,
                datatype.handle,
                source,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "irecv_custom")?;
        Ok(Request::new(request_handle, RequestKind::PointToPoint))
    }
}
