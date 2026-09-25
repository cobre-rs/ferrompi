//! Persistent point-to-point (MPI 1.1+) and persistent collective operations (MPI 4.0+).

use crate::comm::Communicator;
use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::ReduceOp;

impl Communicator {
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
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank
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
        dest: i32,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid slice of T; cast to *const c_void is the standard
            // pattern for MPI send buffers. The caller is responsible for not modifying
            // data while the request is active (between start and wait). The slice must
            // remain valid for the entire lifetime of the returned PersistentRequest.
            ffi::ferrompi_send_init(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
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
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank
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
        dest: i32,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid slice of T; cast to *const c_void is the standard
            // pattern for MPI send buffers. The caller is responsible for not modifying
            // data while the request is active (between start and wait). The slice must
            // remain valid for the entire lifetime of the returned PersistentRequest.
            // A buffer must be attached via Mpi::buffer_attach before start() is called.
            ffi::ferrompi_bsend_init(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
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
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank
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
        dest: i32,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid slice of T; cast to *const c_void is the standard
            // pattern for MPI send buffers. The caller is responsible for not modifying
            // data while the request is active (between start and wait). The slice must
            // remain valid for the entire lifetime of the returned PersistentRequest.
            // Additionally, per the MPI ready-mode safety contract, the caller must
            // ensure the matching receive is already posted before calling start().
            ffi::ferrompi_rsend_init(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
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
    /// # Arguments
    ///
    /// * `data` - Send buffer (must remain valid for lifetime of handle)
    /// * `dest` - Destination rank
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
        dest: i32,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid slice of T; cast to *const c_void is the standard
            // pattern for MPI send buffers. The caller is responsible for not modifying
            // data while the request is active (between start and wait). The slice must
            // remain valid for the entire lifetime of the returned PersistentRequest.
            ffi::ferrompi_ssend_init(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "ssend_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent receive operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// Available in all MPI versions (MPI 1.1+).
    ///
    /// # Arguments
    ///
    /// * `data`   - Receive buffer (must remain valid for lifetime of handle)
    /// * `source` - Source rank, or `-1` for `MPI_ANY_SOURCE`
    /// * `tag`    - Message tag, or `-1` for `MPI_ANY_TAG`
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
        source: i32,
        tag: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice of T; cast to
            // *mut c_void is the standard pattern for MPI receive buffers. The buffer
            // must remain valid for the entire lifetime of the returned PersistentRequest;
            // the caller must call wait() after each start() before accessing the data.
            ffi::ferrompi_recv_init(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                source,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "recv_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    // ========================================================================
    // Generic Persistent Collectives (MPI 4.0+)
    // ========================================================================

    /// Initialize a persistent broadcast operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to use for broadcasts (must remain valid for lifetime of handle)
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// let mut persistent = world.bcast_init(&mut data, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn bcast_init<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice of T (Rust borrow rules
            // prevent aliasing). T::TAG matches T's MPI datatype per ADR-0003. The caller must
            // keep `data` alive for the entire lifetime of the returned PersistentRequest
            // (including all start/wait cycles) per ADR-0004 §"Buffer-lifetime invariant".
            ffi::ferrompi_bcast_init(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "bcast_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-reduce operation.
    ///
    /// Requires MPI 4.0+.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.allreduce_init(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allreduce_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() verified above.
            // T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive for
            // the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant".
            ffi::ferrompi_allreduce_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allreduce_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place all-reduce operation.
    ///
    /// Requires MPI 4.0+.
    pub fn allreduce_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice of T. NULL as sendbuf is
            // the shim's in-place marker, which ferrompi_allreduce_init maps to MPI_IN_PLACE, so
            // data serves as both send and receive buffer. T::TAG matches T's MPI datatype per
            // ADR-0003. `data` must remain alive for the entire lifetime of the returned
            // PersistentRequest per ADR-0004 §"Buffer-lifetime invariant".
            ffi::ferrompi_allreduce_init(
                std::ptr::null(),
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allreduce_init_inplace")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent reduce operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer (significant only at root)
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.reduce_init(&send, &mut recv, ReduceOp::Sum, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn reduce_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() verified above.
            // T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive for
            // the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant". recv is ignored by MPI at non-root.
            ffi::ferrompi_reduce_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "reduce_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent gather operation.
    ///
    /// Requires MPI 4.0+.
    pub fn gather_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). recv is ignored by MPI at non-root.
            // T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive for
            // the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant".
            ffi::ferrompi_gather_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "gather_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent scatter operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (significant only at root)
    /// * `recv` - Receive buffer
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![0.0f64; 40]; // 4 ranks × 10 elements
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.scatter_init(&send, &mut recv, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn scatter_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice (ignored by MPI at non-root) and recv is a
            // valid exclusive slice of T; they cannot alias (Rust borrow rules). T::TAG matches
            // T's MPI datatype per ADR-0003. Both slices must remain alive for the entire
            // lifetime of the returned PersistentRequest per ADR-0004 §"Buffer-lifetime invariant".
            ffi::ferrompi_scatter_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scatter_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-gather operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (each rank sends `send.len()` elements)
    /// * `recv` - Receive buffer (must hold `send.len() * size` elements)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 40]; // 4 ranks × 10
    /// let mut persistent = world.allgather_init(&send, &mut recv).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allgather_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
    ) -> Result<PersistentRequest> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). T::TAG matches T's MPI datatype per ADR-0003.
            // Both slices must remain alive for the entire lifetime of the returned
            // PersistentRequest per ADR-0004 §"Buffer-lifetime invariant".
            ffi::ferrompi_allgather_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allgather_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent scan (inclusive prefix reduction) operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer
    /// * `op` - Reduction operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.scan_init(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn scan_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() verified above.
            // T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive for
            // the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant".
            ffi::ferrompi_scan_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scan_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent exclusive scan (exclusive prefix reduction) operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer (undefined on rank 0 after operation)
    /// * `op` - Reduction operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.exscan_init(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn exscan_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() verified above.
            // T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive for
            // the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant". Note: MPI leaves recv undefined on rank 0.
            ffi::ferrompi_exscan_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "exscan_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-to-all operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (must contain `sendcount * size` elements)
    /// * `recv` - Receive buffer (must contain `recvcount * size` elements)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![1.0f64; 10 * size];
    /// let mut recv = vec![0.0f64; 10 * size];
    /// let mut persistent = world.alltoall_init(&send, &mut recv).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn alltoall_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
    ) -> Result<PersistentRequest> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let size = self.size() as usize;
        if size == 0 || send.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let count_per_rank = send.len() / size;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() and divisibility
            // by size are both verified above. T::TAG matches T's MPI datatype per ADR-0003.
            // Both slices must remain alive for the entire lifetime of the returned
            // PersistentRequest per ADR-0004 §"Buffer-lifetime invariant".
            ffi::ferrompi_alltoall_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                count_per_rank as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                count_per_rank as i64,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoall_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent reduce-scatter-block operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (must contain `recvcount * size` elements)
    /// * `recv` - Receive buffer
    /// * `op` - Reduction operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![1.0f64; 10 * size];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.reduce_scatter_block_init(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn reduce_scatter_block_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest> {
        let size = self.size() as usize;
        if size == 0 || send.len() != recv.len() * size {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T;
            // they cannot alias (Rust borrow rules). send.len() == recv.len() * size is verified
            // above. T::TAG matches T's MPI datatype per ADR-0003. Both slices must remain alive
            // for the entire lifetime of the returned PersistentRequest per ADR-0004
            // §"Buffer-lifetime invariant".
            ffi::ferrompi_reduce_scatter_block_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "reduce_scatter_block_init")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place gather operation at root. Non-root ranks
    /// must use `gather_init` — this method returns `Error::InvalidOp` on non-root.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `recvcount * size()` where `recvcount` is the
    /// per-rank count. Rank `r`'s contribution lives at offset `r * recvcount`.
    /// Root's own contribution must be pre-written before each `start()` call.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidOp` if this rank is not `root`.
    /// - `Error::InvalidBuffer` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// if world.rank() == 0 {
    ///     let mut data = vec![0i32; 4 * world.size() as usize];
    ///     let mut persistent = world.gather_init_inplace(&mut data, 0).unwrap();
    ///     for _ in 0..3 {
    ///         persistent.start().unwrap();
    ///         persistent.wait().unwrap();
    ///     }
    /// }
    /// ```
    pub fn gather_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        if self.rank() != root {
            return Err(Error::InvalidOp);
        }
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice. NULL as sendbuf is the
            // shim's in-place marker, which ferrompi_gather_init maps to MPI_IN_PLACE; data
            // serves as both root's send contribution and the receive buffer. The guard above
            // guarantees self.rank() == root, which is the only rank MPI_IN_PLACE is valid for
            // in MPI_Gather_init. The buffer must remain valid for the lifetime of the returned
            // PersistentRequest; the caller must call wait() after each start().
            ffi::ferrompi_gather_init(
                std::ptr::null(),
                0,
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "gather_init_inplace")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place all-gather operation. Every rank's `data`
    /// is both send contribution and receive buffer.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `recvcount * size()`. Rank `r`'s contribution
    /// lives at offset `r * recvcount` and must be pre-written before each
    /// `start()` call.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidBuffer` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank() as usize;
    /// let size = world.size() as usize;
    /// let mut data = vec![0i32; size];
    /// data[rank] = rank as i32 * 10;
    /// let mut persistent = world.allgather_init_inplace(&mut data).unwrap();
    /// for _ in 0..3 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allgather_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
    ) -> Result<PersistentRequest> {
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice. NULL as sendbuf is the
            // shim's in-place marker, which ferrompi_allgather_init maps to MPI_IN_PLACE. Each
            // rank's contribution (at offset rank*recvcount) must be pre-written by the caller
            // before each start(). The buffer must remain valid for the lifetime of the
            // returned PersistentRequest.
            ffi::ferrompi_allgather_init(
                std::ptr::null(),
                0,
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allgather_init_inplace")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place scatter operation. At root, `data` is the
    /// `sendcount * size()` send buffer; root's own slot is retained in place. At
    /// non-root, `data` is the `recvcount`-element receive buffer.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `sendcount * size()`. Rank `r`'s slot is
    /// `data[r*sendcount .. (r+1)*sendcount]`. After each wait, only root's own
    /// slot is guaranteed to remain intact; other slots are unspecified.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidBuffer` at root if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// if world.rank() == 0 {
    ///     let mut data = vec![0i32, 10, 20, 30];
    ///     let mut persistent = world.scatter_init_inplace(&mut data, 0).unwrap();
    ///     for _ in 0..3 {
    ///         persistent.start().unwrap();
    ///         persistent.wait().unwrap();
    ///     }
    /// } else {
    ///     let mut data = vec![0i32; 1];
    ///     let mut persistent = world.scatter_init_inplace(&mut data, 0).unwrap();
    ///     for _ in 0..3 {
    ///         persistent.start().unwrap();
    ///         persistent.wait().unwrap();
    ///     }
    /// }
    /// ```
    pub fn scatter_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        let is_root = self.rank() == root;
        let size = self.size() as usize;
        let (sendbuf, sendcount, recvbuf, recvcount) = if is_root {
            if size == 0 || data.len() % size != 0 {
                return Err(Error::InvalidBuffer);
            }
            let per = (data.len() / size) as i64;
            (
                data.as_ptr().cast::<std::ffi::c_void>(),
                per,
                std::ptr::null_mut::<std::ffi::c_void>(),
                0i64,
            )
        } else {
            (
                std::ptr::null::<std::ffi::c_void>(),
                0i64,
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
            )
        };
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: At root, sendbuf points to valid data of length sendcount*size elements
            // (guaranteed by the divisibility check above); recvbuf is NULL, the shim's in-place
            // marker, which ferrompi_scatter_init maps to MPI_IN_PLACE so root's own slot is
            // retained. At non-root, recvbuf points to a valid mutable slice of length recvcount
            // elements; sendbuf is null (MPI standard ignores sendbuf on non-root scatter). Both
            // pointers are cast to *const/*mut c_void as required by the C FFI. The buffer must
            // remain valid for the lifetime of the returned PersistentRequest.
            ffi::ferrompi_scatter_init(
                sendbuf,
                sendcount,
                recvbuf,
                recvcount,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scatter_init_inplace")?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent in-place all-to-all personalized communication.
    /// `data` is both send and receive buffer on every rank.
    ///
    /// Before each `start()` call, rank `r` must pre-write into slot `s` (at
    /// offset `s * count`) the payload it wishes to send to rank `s`. After each
    /// `wait()`, slot `s` contains the data received FROM rank `s`.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `count * size()`. Slot `s` at
    /// `data[s*count..(s+1)*count]` holds data sent to (and later received from)
    /// rank `s`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidBuffer` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let r = world.rank() as i32;
    /// let size = world.size() as usize;
    /// let mut data: Vec<i32> = (0..size as i32).map(|s| r * 10 + s).collect();
    /// let mut persistent = world.alltoall_init_inplace(&mut data).unwrap();
    /// for _ in 0..3 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn alltoall_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
    ) -> Result<PersistentRequest> {
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice of length recvcount*size
            // elements (guaranteed by the divisibility check above). NULL as sendbuf is the
            // shim's in-place marker, which ferrompi_alltoall_init maps to MPI_IN_PLACE; the
            // caller must pre-write each slot before each start() call. The buffer must remain
            // valid for the lifetime of the returned PersistentRequest.
            ffi::ferrompi_alltoall_init(
                std::ptr::null(),
                0,
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoall_init_inplace")?;
        Ok(PersistentRequest::new(request_handle))
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::Communicator;
    use crate::error::{Error, Result};
    use crate::persistent::PersistentRequest;
    use crate::ReduceOp;

    fn dummy_comm() -> Communicator {
        Communicator {
            handle: 0,
            rank: 0,
            size: 1,
        }
    }

    #[test]
    fn send_init_signature_compiles() {
        fn _check(c: &Communicator, buf: &[i32]) -> Result<PersistentRequest> {
            c.send_init(buf, 0, 0)
        }
    }

    #[test]
    fn bsend_init_signature_compiles() {
        fn _check(c: &Communicator, buf: &[i32]) -> Result<PersistentRequest> {
            c.bsend_init(buf, 0, 0)
        }
    }

    #[test]
    fn rsend_init_signature_compiles() {
        fn _check(c: &Communicator, buf: &[i32]) -> Result<PersistentRequest> {
            c.rsend_init(buf, 0, 0)
        }
    }

    #[test]
    fn ssend_init_signature_compiles() {
        fn _check(c: &Communicator, buf: &[i32]) -> Result<PersistentRequest> {
            c.ssend_init(buf, 0, 0)
        }
    }

    #[test]
    fn recv_init_signature_compiles() {
        fn _check(c: &Communicator, buf: &mut [i32]) -> Result<PersistentRequest> {
            c.recv_init(buf, 0, 0)
        }
    }

    #[test]
    fn allreduce_init_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.allreduce_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn reduce_init_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.reduce_init(&send, &mut recv, ReduceOp::Sum, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scan_init_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.scan_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn exscan_init_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.exscan_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoall_init_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5]; // different length → fires before self.size()
        let result = comm.alltoall_init(&send, &mut recv);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn gather_init_inplace_nonroot_returns_invalid_op() {
        let comm = Communicator {
            handle: 0,
            rank: 1,
            size: 4,
        };
        let mut data = vec![0u32; 4];
        let result = comm.gather_init_inplace(&mut data, 0);
        assert!(matches!(result, Err(Error::InvalidOp)));
    }

    #[test]
    fn allgather_init_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7];
        let result = comm.allgather_init_inplace(&mut data);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoall_init_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7];
        let result = comm.alltoall_init_inplace(&mut data);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }
}
