//! Persistent collective operations (MPI 4.0, or Open MPI 5).

use crate::comm::{
    check_rank_slots, check_same_len, rank_block, scatter_inplace_args, Communicator,
};
use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::{vec_buf, vec_buf_mut, PersistentRequest};
use crate::ReduceOp;

impl Communicator {
    /// Initialize a persistent broadcast operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer, owned by the request; the root writes the value to broadcast
    ///   through `buffer_mut`, and every rank reads the result through `buffer`, while the
    ///   request is inactive
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut persistent = world.broadcast_init(vec![0.0f64; 100], 0).unwrap();
    /// for iter in 0..100 {
    ///     if world.rank() == 0 {
    ///         persistent.buffer_mut().unwrap().fill(iter as f64);
    ///     }
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn broadcast_init<T: MpiDatatype>(
        &self,
        mut data: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        let mut request_handle: i64 = 0;
        let (p, n, dt) = vec_buf_mut(&mut data);
        // SAFETY: `p` and `n` describe `data`'s heap allocation, and `p` comes from
        // `Vec::as_mut_ptr`, so no slice reference sits between it and the allocation and
        // the accessors' slices are its children. The returned request owns `data` until
        // it is freed, and moving a `Vec` leaves its allocation in place. The allocation
        // is never resized, because the accessors hand out slices only. From start() to
        // completion MPI is its only user, because the accessors refuse while the request
        // is active, and Drop leaks it when MPI may still use it. If the call fails, MPI
        // recorded nothing (or the shim freed the inactive request), and `data` is dropped
        // here.
        let ret =
            unsafe { ffi::ferrompi_bcast_init(p, n, dt, root, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "broadcast_init")?;
        Ok(PersistentRequest::new(request_handle, data))
    }

    /// Initialize a persistent all-reduce operation.
    ///
    /// Requires MPI 4.0, or Open MPI 5.
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
    ) -> Result<PersistentRequest<T>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned PersistentRequest records both pointers until the
        // request is freed and does not borrow either slice; keeping both alive and
        // untouched between start() and completion is the caller's documented obligation,
        // which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_allreduce_init(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "allreduce_init")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent in-place all-reduce operation.
    ///
    /// Requires MPI 4.0, or Open MPI 5.
    pub fn allreduce_init_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        op: ReduceOp,
    ) -> Result<PersistentRequest<T>> {
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_allreduce_init maps it to MPI_IN_PLACE, so data
        // serves as both send and receive buffer. The returned PersistentRequest records
        // `data`'s pointer until the request is freed and does not borrow it; keeping it
        // alive and untouched between start() and completion is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_allreduce_init(
                std::ptr::null(),
                p,
                n,
                dt,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allreduce_init_inplace")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent reduce operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer; must have `send.len()` elements on every
    ///   rank (its contents matter only at the root)
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] if `recv.len() != send.len()`, on
    /// any rank.
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
    ) -> Result<PersistentRequest<T>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]); recv is ignored by MPI at non-root. The returned
        // PersistentRequest records both pointers until the request is freed and does not
        // borrow either slice; keeping both alive and untouched between start() and
        // completion is the caller's documented obligation, which this signature does not
        // enforce.
        let ret = unsafe {
            ffi::ferrompi_reduce_init(
                sp,
                rp,
                n,
                dt,
                op as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "reduce_init")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent gather operation.
    ///
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer, owned by the request; write it through `send_buffer_mut`
    ///   while the request is inactive
    /// * `recv` - Receive buffer, owned by the request; read it through `buffer` while the
    ///   request is inactive. It is significant only at the root and may be empty
    ///   elsewhere
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if this rank is `root` and `recv.len() < send.len() *
    /// size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let recv = if world.rank() == 0 { vec![0.0f64; 10 * size] } else { Vec::new() };
    /// let mut persistent = world.gather_init(vec![0.0f64; 10], recv, 0).unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     if world.rank() == 0 {
    ///         assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    ///     }
    /// }
    /// ```
    pub fn gather_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        if self.rank == root {
            check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        // SAFETY: `sp`, `n` and `rp` describe the heap allocations of `send` and `recv`,
        // and each pointer comes from `Vec::as_ptr`/`as_mut_ptr`, so no slice reference
        // sits between it and its allocation and the accessors' slices are its children.
        // The two Vecs are distinct allocations, and `recv` is ignored by MPI at non-root.
        // The root-side receive-length relation (recv.len() >= send.len() * size) is
        // checked above. The returned request owns both Vecs until it is freed, and moving
        // a `Vec` leaves its allocation in place. The allocations are never resized,
        // because the accessors hand out slices only. From start() to completion MPI is
        // their only user, because the accessors refuse while the request is active, and
        // Drop leaks them when MPI may still use them. If the call fails, MPI recorded
        // nothing (or the shim freed the inactive request), and both Vecs are dropped
        // here.
        let ret = unsafe {
            ffi::ferrompi_gather_init(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "gather_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            Vec::new(),
        ))
    }

    /// Initialize a persistent scatter operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer, owned by the request; write it through `send_buffer_mut`
    ///   while the request is inactive. It is significant only at the root and may be empty
    ///   elsewhere
    /// * `recv` - Receive buffer, owned by the request; read it through `buffer` while the
    ///   request is inactive
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if this rank is `root` and `send.len() < recv.len() *
    /// size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = if world.rank() == 0 { vec![0.0f64; 10 * size] } else { Vec::new() };
    /// let mut persistent = world.scatter_init(send, vec![0.0f64; 10], 0).unwrap();
    /// for iter in 0..100 {
    ///     if world.rank() == 0 {
    ///         persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     }
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn scatter_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        if self.rank == root {
            check_rank_slots("send", send.len(), recv.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = vec_buf(&send);
        let (rp, n, dt) = vec_buf_mut(&mut recv);
        // SAFETY: `sp`, `rp` and `n` describe the heap allocations of `send` and `recv`,
        // and each pointer comes from `Vec::as_ptr`/`as_mut_ptr`, so no slice reference
        // sits between it and its allocation and the accessors' slices are its children.
        // The two Vecs are distinct allocations, and `send` is ignored by MPI at non-root.
        // The root-side send-length relation (send.len() >= recv.len() * size) is checked
        // above. The returned request owns both Vecs until it is freed, and moving a `Vec`
        // leaves its allocation in place. The allocations are never resized, because the
        // accessors hand out slices only. From start() to completion MPI is their only
        // user, because the accessors refuse while the request is active, and Drop leaks
        // them when MPI may still use them. If the call fails, MPI recorded nothing (or
        // the shim freed the inactive request), and both Vecs are dropped here.
        let ret = unsafe {
            ffi::ferrompi_scatter_init(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "scatter_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            Vec::new(),
        ))
    }

    /// Initialize a persistent all-gather operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (each rank sends `send.len()` elements), owned by the
    ///   request; write it through `send_buffer_mut` while the request is inactive
    /// * `recv` - Receive buffer (must hold at least `send.len() * size` elements), owned
    ///   by the request; read it through `buffer` while the request is inactive
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if `recv.len() < send.len() * size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let mut persistent = world
    ///     .allgather_init(vec![1.0f64; 10], vec![0.0f64; 10 * size])
    ///     .unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn allgather_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
    ) -> Result<PersistentRequest<T>> {
        check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        // SAFETY: `sp`, `n` and `rp` describe the heap allocations of `send` and `recv`,
        // and each pointer comes from `Vec::as_ptr`/`as_mut_ptr`, so no slice reference
        // sits between it and its allocation and the accessors' slices are its children.
        // The two Vecs are distinct allocations. The every-rank receive-length relation
        // (recv.len() >= send.len() * size) is checked above. The returned request owns
        // both Vecs until it is freed, and moving a `Vec` leaves its allocation in place.
        // The allocations are never resized, because the accessors hand out slices only.
        // From start() to completion MPI is their only user, because the accessors refuse
        // while the request is active, and Drop leaks them when MPI may still use them. If
        // the call fails, MPI recorded nothing (or the shim freed the inactive request),
        // and both Vecs are dropped here.
        let ret = unsafe {
            ffi::ferrompi_allgather_init(sp, n, rp, n, dt, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "allgather_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            Vec::new(),
        ))
    }

    /// Initialize a persistent scan (inclusive prefix reduction) operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
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
    ) -> Result<PersistentRequest<T>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned PersistentRequest records both pointers until the
        // request is freed and does not borrow either slice; keeping both alive and
        // untouched between start() and completion is the caller's documented obligation,
        // which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_scan_init(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "scan_init")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent exclusive scan (exclusive prefix reduction) operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
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
    ) -> Result<PersistentRequest<T>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). MPI leaves recv undefined on rank 0, documented above. The
        // returned PersistentRequest records both pointers until the request is freed and
        // does not borrow either slice; keeping both alive and untouched between start() and
        // completion is the caller's documented obligation, which this signature does not
        // enforce.
        let ret = unsafe {
            ffi::ferrompi_exscan_init(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "exscan_init")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent all-to-all operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (must contain `sendcount * size` elements), owned by the
    ///   request; write it through `send_buffer_mut` while the request is inactive
    /// * `recv` - Receive buffer (must contain `recvcount * size` elements), owned by the
    ///   request; read it through `buffer` while the request is inactive
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let mut persistent = world
    ///     .alltoall_init(vec![1.0f64; 10 * size], vec![0.0f64; 10 * size])
    ///     .unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn alltoall_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
    ) -> Result<PersistentRequest<T>> {
        check_same_len("recv", send.len(), recv.len())?;
        let count_per_rank = rank_block("send", send.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        // SAFETY: `sp` and `rp` describe the heap allocations of `send` and `recv`, and
        // each pointer comes from `Vec::as_ptr`/`as_mut_ptr`, so no slice reference sits
        // between it and its allocation and the accessors' slices are its children. The
        // two Vecs are distinct allocations. send.len() == recv.len() and divisibility by
        // size are both verified above. The returned request owns both Vecs until it is
        // freed, and moving a `Vec` leaves its allocation in place. The allocations are
        // never resized, because the accessors hand out slices only. From start() to
        // completion MPI is their only user, because the accessors refuse while the
        // request is active, and Drop leaks them when MPI may still use them. If the call
        // fails, MPI recorded nothing (or the shim freed the inactive request), and both
        // Vecs are dropped here.
        let ret = unsafe {
            ffi::ferrompi_alltoall_init(
                sp,
                count_per_rank,
                rp,
                count_per_rank,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoall_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            Vec::new(),
        ))
    }

    /// Initialize a persistent reduce-scatter-block operation.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
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
    ) -> Result<PersistentRequest<T>> {
        check_same_len(
            "recv",
            rank_block("send", send.len(), self.size)?,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len() *
        // size is verified above. The returned PersistentRequest records both pointers until
        // the request is freed and does not borrow either slice; keeping both alive and
        // untouched between start() and completion is the caller's documented obligation,
        // which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_reduce_scatter_block_init(
                sp,
                rp,
                n,
                dt,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "reduce_scatter_block_init")?;
        Ok(PersistentRequest::new(request_handle, Vec::new()))
    }

    /// Initialize a persistent in-place gather operation. At root, `data` is both
    /// the send contribution and the receive buffer. At non-root, `data` is that
    /// rank's send block.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer, owned by the request; reach it with `buffer` and `buffer_mut`
    ///   while the request is inactive, or take it back with `into_buffer`
    /// * `root` - Rank of the root process
    ///
    /// # Buffer Layout
    ///
    /// At root, `data` must have length `recvcount * size()` where `recvcount` is the
    /// per-rank count. Rank `r`'s contribution lives at offset `r * recvcount`.
    /// Root's own contribution must be pre-written through `buffer_mut()` before each
    /// `start()` call.
    ///
    /// At non-root, `data` is the rank's block of `recvcount` elements. Only root
    /// knows `recvcount`, so a block of another length is not checked locally; MPI
    /// reports the mismatch.
    ///
    /// # Errors
    ///
    /// `Error::InvalidArgument` at root if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = if world.rank() == 0 {
    ///     vec![0i32; 4 * world.size() as usize]
    /// } else {
    ///     vec![world.rank(); 4]
    /// };
    /// let mut persistent = world.gather_init_inplace(data, 0).unwrap();
    /// for _ in 0..3 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// let _data = persistent.into_buffer().unwrap();
    /// ```
    pub fn gather_init_inplace<T: MpiDatatype>(
        &self,
        mut data: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        let mut request_handle: i64 = 0;
        let ret = if self.rank() == root {
            let recvcount = rank_block("data", data.len(), self.size)? as i64;
            let (p, _, dt) = vec_buf_mut(&mut data);
            // SAFETY: `p` describes `data`'s heap allocation and comes from
            // `Vec::as_mut_ptr`, so no slice reference sits between it and the allocation
            // and the accessors' slices are its children. NULL sendbuf is the in-place
            // marker (a Vec's pointer is never null, so this NULL is unambiguous);
            // ferrompi_gather_init maps it to MPI_IN_PLACE, so data serves as both root's
            // send contribution and the receive buffer. recvcount is checked to evenly
            // divide data.len() above, and this branch runs only at root, the only rank
            // MPI_IN_PLACE is valid for in MPI_Gather_init. The returned request owns
            // `data` until it is freed, and moving a `Vec` leaves its allocation in place.
            // The allocation is never resized, because the accessors hand out slices only.
            // From start() to completion MPI is its only user, because the accessors
            // refuse while the request is active, and Drop leaks it when MPI may still use
            // it. If the call fails, MPI recorded nothing (or the shim freed the inactive
            // request), and `data` is dropped.
            unsafe {
                ffi::ferrompi_gather_init(
                    std::ptr::null(),
                    0,
                    p,
                    recvcount,
                    dt,
                    root,
                    self.handle,
                    &mut request_handle,
                )
            }
        } else {
            let (p, n, dt) = vec_buf_mut(&mut data);
            // SAFETY: `p` and `n` describe `data`'s heap allocation, and `p` comes from
            // `Vec::as_mut_ptr`, so no slice reference sits between it and the allocation
            // and the accessors' slices are its children. p is non-null, so
            // ferrompi_gather_init does not map it to MPI_IN_PLACE; data is this rank's
            // send block. It is also passed as recvbuf because strict MPI builds reject a
            // NULL recvbuf at non-root; MPI ignores recvbuf and recvcount there. The
            // returned request owns `data` until it is freed, and moving a `Vec` leaves its
            // allocation in place. The allocation is never resized, because the accessors
            // hand out slices only. From start() to completion MPI is its only user,
            // because the accessors refuse while the request is active, and Drop leaks it
            // when MPI may still use it. If the call fails, MPI recorded nothing (or the
            // shim freed the inactive request), and `data` is dropped.
            unsafe {
                ffi::ferrompi_gather_init(p, n, p, n, dt, root, self.handle, &mut request_handle)
            }
        };
        Error::check_with_op(ret, "gather_init_inplace")?;
        Ok(PersistentRequest::new(request_handle, data))
    }

    /// Initialize a persistent in-place all-gather operation. Every rank's `data`
    /// is both send contribution and receive buffer.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer, owned by the request; reach it with `buffer` and `buffer_mut`
    ///   while the request is inactive, or take it back with `into_buffer`
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `recvcount * size()`. Rank `r`'s contribution
    /// lives at offset `r * recvcount` and must be pre-written through `buffer_mut()`
    /// before each `start()` call.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank() as usize;
    /// let size = world.size() as usize;
    /// let mut persistent = world.allgather_init_inplace(vec![0i32; size]).unwrap();
    /// for _ in 0..3 {
    ///     persistent.buffer_mut().unwrap()[rank] = rank as i32 * 10;
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allgather_init_inplace<T: MpiDatatype>(
        &self,
        mut data: Vec<T>,
    ) -> Result<PersistentRequest<T>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = vec_buf_mut(&mut data);
        // SAFETY: `p` describes `data`'s heap allocation and comes from `Vec::as_mut_ptr`,
        // so no slice reference sits between it and the allocation and the accessors'
        // slices are its children. NULL sendbuf is the in-place marker (a Vec's pointer is
        // never null, so this NULL is unambiguous); ferrompi_allgather_init maps it to
        // MPI_IN_PLACE. recvcount is checked to evenly divide data.len() above; each rank's
        // slot (at offset rank*recvcount) must be pre-written through buffer_mut() before
        // each start(). The returned request owns `data` until it is freed, and moving a
        // `Vec` leaves its allocation in place. The allocation is never resized, because
        // the accessors hand out slices only. From start() to completion MPI is its only
        // user, because the accessors refuse while the request is active, and Drop leaks it
        // when MPI may still use it. If the call fails, MPI recorded nothing (or the shim
        // freed the inactive request), and `data` is dropped here.
        let ret = unsafe {
            ffi::ferrompi_allgather_init(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allgather_init_inplace")?;
        Ok(PersistentRequest::new(request_handle, data))
    }

    /// Initialize a persistent in-place scatter operation. At root, `data` is the
    /// `sendcount * size()` send buffer; root's own slot is retained in place. At
    /// non-root, `data` is the `recvcount`-element receive buffer.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer, owned by the request; reach it with `buffer` and `buffer_mut`
    ///   while the request is inactive, or take it back with `into_buffer`
    /// * `root` - Rank of the root process
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `sendcount * size()`. Rank `r`'s slot is
    /// `data[r*sendcount .. (r+1)*sendcount]`. After each wait, only root's own
    /// slot is guaranteed to remain intact; other slots are unspecified.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` at root if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let data = if world.rank() == 0 {
    ///     vec![0i32, 10, 20, 30]
    /// } else {
    ///     vec![0i32; 1]
    /// };
    /// let mut persistent = world.scatter_init_inplace(data, 0).unwrap();
    /// for _ in 0..3 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// let _data = persistent.into_buffer().unwrap();
    /// ```
    pub fn scatter_init_inplace<T: MpiDatatype>(
        &self,
        mut data: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        let is_root = self.rank() == root;
        let (_, sendcount, _, recvcount, dt) = scatter_inplace_args(&mut data, is_root, self.size)?;
        let mut request_handle: i64 = 0;
        let (p, _, _) = vec_buf_mut(&mut data);
        let (sendbuf, recvbuf) = if is_root {
            (p.cast_const(), std::ptr::null_mut())
        } else {
            (std::ptr::null(), p)
        };
        // SAFETY: `p` describes `data`'s heap allocation and comes from `Vec::as_mut_ptr`,
        // so no slice reference sits between it and the allocation and the accessors'
        // slices are its children; the pointers scatter_inplace_args derived from a slice
        // reborrow are discarded, and only its counts and tag are used. At root, recvbuf is
        // NULL, the in-place marker (a Vec's pointer is never null, so this NULL is
        // unambiguous); ferrompi_scatter_init maps it to MPI_IN_PLACE so root's own slot is
        // retained, and sendbuf is `data`. The one `vec_buf_mut` pointer serves both
        // branches: it is recvbuf at non-root and, cast to const, sendbuf at root.
        // scatter_inplace_args checks that the block size evenly divides data.len(). At
        // non-root, sendbuf is null, which the MPI standard ignores on non-root scatter. The returned request owns `data` until it is freed, and moving
        // a `Vec` leaves its allocation in place. The allocation is never resized, because
        // the accessors hand out slices only. From start() to completion MPI is its only
        // user, because the accessors refuse while the request is active, and Drop leaks it
        // when MPI may still use it. If the call fails, MPI recorded nothing (or the shim
        // freed the inactive request), and `data` is dropped.
        let ret = unsafe {
            ffi::ferrompi_scatter_init(
                sendbuf,
                sendcount,
                recvbuf,
                recvcount,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scatter_init_inplace")?;
        Ok(PersistentRequest::new(request_handle, data))
    }

    /// Initialize a persistent in-place all-to-all personalized communication.
    /// `data` is both send and receive buffer on every rank.
    ///
    /// Before each `start()` call, rank `r` must pre-write into slot `s` (at
    /// offset `s * count`) the payload it wishes to send to rank `s`, through
    /// `buffer_mut()`. After each `wait()`, slot `s` contains the data received FROM
    /// rank `s`, which `buffer()` reads.
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer, owned by the request; reach it with `buffer` and `buffer_mut`
    ///   while the request is inactive, or take it back with `into_buffer`
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `count * size()`. Slot `s` at
    /// `data[s*count..(s+1)*count]` holds data sent to (and later received from)
    /// rank `s`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let r = world.rank() as i32;
    /// let size = world.size() as usize;
    /// let data: Vec<i32> = (0..size as i32).map(|s| r * 10 + s).collect();
    /// let mut persistent = world.alltoall_init_inplace(data).unwrap();
    /// for _ in 0..3 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     for (s, x) in persistent.buffer_mut().unwrap().iter_mut().enumerate() {
    ///         *x = r * 10 + s as i32;
    ///     }
    /// }
    /// ```
    pub fn alltoall_init_inplace<T: MpiDatatype>(
        &self,
        mut data: Vec<T>,
    ) -> Result<PersistentRequest<T>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = vec_buf_mut(&mut data);
        // SAFETY: `p` describes `data`'s heap allocation and comes from `Vec::as_mut_ptr`,
        // so no slice reference sits between it and the allocation and the accessors'
        // slices are its children. NULL sendbuf is the in-place marker (a Vec's pointer is
        // never null, so this NULL is unambiguous); ferrompi_alltoall_init maps it to
        // MPI_IN_PLACE. recvcount is checked to evenly divide data.len() above; the caller
        // must pre-write each slot through buffer_mut() before each start() call. The
        // returned request owns `data` until it is freed, and moving a `Vec` leaves its
        // allocation in place. The allocation is never resized, because the accessors hand
        // out slices only. From start() to completion MPI is its only user, because the
        // accessors refuse while the request is active, and Drop leaks it when MPI may
        // still use it. If the call fails, MPI recorded nothing (or the shim freed the
        // inactive request), and `data` is dropped here.
        let ret = unsafe {
            ffi::ferrompi_alltoall_init(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoall_init_inplace")?;
        Ok(PersistentRequest::new(request_handle, data))
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::test_comm;
    use crate::error::Error;
    use crate::ReduceOp;

    #[test]
    fn allreduce_init_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.allreduce_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn reduce_init_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.reduce_init(&send, &mut recv, ReduceOp::Sum, 0);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn scan_init_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.scan_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn exscan_init_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.exscan_init(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn alltoall_init_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let recv = vec![0.0f64; 5]; // different length → fires before self.size()
        let result = comm.alltoall_init(send, recv);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn allgather_init_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let data = vec![0u32; 7];
        let result = comm.allgather_init_inplace(data);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }

    #[test]
    fn alltoall_init_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let data = vec![0u32; 7];
        let result = comm.alltoall_init_inplace(data);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }
}
