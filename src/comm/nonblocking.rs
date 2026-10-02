//! Nonblocking collective operations: ibroadcast, iallreduce, ireduce, igather, etc.

use crate::comm::{
    check_rank_slots, check_same_len, rank_block, scatter_inplace_args, Communicator,
};
use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::request::{Request, RequestKind};
use crate::scope::Scope;
use crate::ReduceOp;

impl Communicator {
    // ========================================================================
    // Generic Nonblocking Collectives
    // ========================================================================

    /// Nonblocking broadcast.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    ///
    /// # Safety Note
    ///
    /// The buffer must remain valid until the request is completed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// let req = world.ibroadcast(&mut data, 0).unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn ibroadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<Request<'_>> {
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: the returned Request does not borrow `data`; keeping it alive and
        // untouched until the request completes is the caller's documented obligation,
        // which this signature does not enforce.
        let ret = unsafe { ffi::ferrompi_ibcast(p, n, dt, root, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibcast")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-reduce.
    ///
    /// Returns a request handle that must be waited on before accessing the buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let req = world.iallreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<Request<'_>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned Request does not borrow either slice; keeping
        // both alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iallreduce(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iallreduce")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking reduce to root.
    ///
    /// Initiates a reduction operation and returns immediately with a [`Request`]
    /// handle. The buffers must remain valid until the request is completed.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for the result; must have `send.len()` elements on
    ///   every rank (its contents matter only at the root)
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
    /// let req = world.ireduce(&send, &mut recv, ReduceOp::Sum, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ireduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<Request<'_>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned Request does not borrow either slice; keeping
        // both alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_ireduce(
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
        Error::check_with_op(ret, "ireduce")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking gather to root.
    ///
    /// Initiates a gather operation and returns immediately with a [`Request`]
    /// handle. Each process sends `send.len()` elements. Root receives
    /// `send.len() * size` elements total.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root)
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
    /// let send = vec![world.rank() as f64; 5];
    /// let mut recv = vec![0.0f64; 5 * world.size() as usize];
    /// let req = world.igather(&send, &mut recv, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn igather<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        root: i32,
    ) -> Result<Request<'_>> {
        if self.rank == root {
            check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]); at non-root, recv is
        // ignored by MPI. The root-side receive-length relation (recv.len() >= send.len()
        // * size) is checked above. The returned Request does not borrow either slice;
        // keeping both alive and untouched until the request completes is the caller's
        // documented obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_igather(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "igather")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-gather.
    ///
    /// Initiates an all-gather operation and returns immediately with a
    /// [`Request`] handle. Each process sends `send.len()` elements and
    /// receives from all.
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
    /// let send = vec![world.rank() as i32; 3];
    /// let mut recv = vec![0i32; 3 * world.size() as usize];
    /// let req = world.iallgather(&send, &mut recv).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallgather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<Request<'_>> {
        check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). The every-rank
        // receive-length relation (recv.len() >= send.len() * size) is checked above.
        // The returned Request does not borrow either slice; keeping both alive and
        // untouched until the request completes is the caller's documented obligation,
        // which this signature does not enforce.
        let ret =
            unsafe { ffi::ferrompi_iallgather(sp, n, rp, n, dt, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "iallgather")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking scatter from root.
    ///
    /// Initiates a scatter operation and returns immediately with a [`Request`]
    /// handle. Root sends `recv.len() * size` elements total, each process
    /// receives `recv.len()` elements.
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
    /// let send = vec![0.0f64; 5 * world.size() as usize];
    /// let mut recv = vec![0.0f64; 5];
    /// let req = world.iscatter(&send, &mut recv, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iscatter<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        root: i32,
    ) -> Result<Request<'_>> {
        if self.rank == root {
            check_rank_slots("send", send.len(), recv.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). The root-side send-length relation (send.len() >= recv.len() * size)
        // is checked above. The returned Request does not borrow either slice; keeping
        // both alive and untouched until the request completes is the caller's
        // documented obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iscatter(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iscatter")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking barrier.
    ///
    /// Initiates a barrier synchronization and returns immediately with a
    /// [`Request`] handle that belongs to the scope `s`. The barrier is complete
    /// when the request is waited on, or when the scope ends.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// ferrompi::scope(|s| {
    ///     let req = world.ibarrier(s)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub fn ibarrier<'s>(&self, s: &'s Scope<'s, '_>) -> Result<Request<'s>> {
        let mut request_handle: i64 = 0;
        // SAFETY: this call takes only the communicator handle and a request out-pointer;
        // no buffer is involved, so the request has no buffer-lifetime obligation, and
        // the scope completes it.
        let ret = unsafe { ffi::ferrompi_ibarrier(self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibarrier")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking inclusive prefix reduction (scan).
    ///
    /// Initiates an inclusive scan and returns immediately with a [`Request`]
    /// handle. On rank `i`, `recv` will contain the reduction of `send` values
    /// from ranks `0..=i` once the request completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] if `send.len() != recv.len()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let req = world.iscan(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iscan<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<Request<'_>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned Request does not borrow either slice; keeping
        // both alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iscan(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iscan")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking exclusive prefix reduction (exscan).
    ///
    /// Initiates an exclusive scan and returns immediately with a [`Request`]
    /// handle. On rank `i`, `recv` will contain the reduction of `send` values
    /// from ranks `0..i` once the request completes.
    ///
    /// # Rank 0 Behavior
    ///
    /// **Per the MPI standard, the contents of `recv` on rank 0 are undefined.**
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] if `send.len() != recv.len()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// let req = world.iexscan(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iexscan<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<Request<'_>> {
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). The returned Request does not borrow either slice; keeping
        // both alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iexscan(sp, rp, n, dt, op as i32, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iexscan")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-to-all personalized communication.
    ///
    /// Initiates an all-to-all operation and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] if `send.len() != recv.len()`, or
    /// [`Error::InvalidArgument`] if `send.len()` is not evenly divisible by the
    /// communicator size.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![world.rank() as f64; size * 3];
    /// let mut recv = vec![0.0f64; size * 3];
    /// let req = world.ialltoall(&send, &mut recv).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ialltoall<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<Request<'_>> {
        check_same_len("recv", send.len(), recv.len())?;
        let count = rank_block("send", send.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len()
        // and divisibility by size are both verified above. The returned Request does not
        // borrow either slice; keeping both alive and untouched until the request
        // completes is the caller's documented obligation, which this signature does not
        // enforce.
        let ret = unsafe {
            ffi::ferrompi_ialltoall(sp, count, rp, count, dt, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "ialltoall")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking reduce-scatter with uniform block size.
    ///
    /// Initiates a reduce-scatter operation and returns immediately with a
    /// [`Request`] handle. Performs an element-wise reduction across all
    /// processes, then scatters the result so that each process receives
    /// `recv.len()` elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if `send.len()` is not evenly divisible
    /// by the communicator size, or [`Error::BufferSize`] if
    /// `send.len() != recv.len() * size`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![1.0f64; size * 5];
    /// let mut recv = vec![0.0f64; 5];
    /// let req = world.ireduce_scatter_block(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ireduce_scatter_block<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<Request<'_>> {
        check_same_len(
            "recv",
            rank_block("send", send.len(), self.size)?,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len()
        // * size is verified above. The returned Request does not borrow either slice;
        // keeping both alive and untouched until the request completes is the caller's
        // documented obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_ireduce_scatter_block(
                sp,
                rp,
                n,
                dt,
                op as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "ireduce_scatter_block")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place gather. At root, `data` is both the send contribution
    /// and the receive buffer. At non-root, `data` is that rank's own block (here its
    /// send block), as for [`iscatter_inplace`](Self::iscatter_inplace).
    ///
    /// # Buffer Layout
    ///
    /// At root, `data` must have length `recvcount * size()` where `recvcount` is
    /// the per-rank count. Rank `r`'s contribution lives at offset
    /// `r * recvcount`. Root's own contribution must be pre-written into
    /// `data[rank() * recvcount .. (rank()+1) * recvcount]` before the call.
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
    /// let mut data = if world.rank() == 0 {
    ///     vec![0i32; 4 * world.size() as usize]
    /// } else {
    ///     vec![world.rank(); 4]
    /// };
    /// let req = world.igather_inplace(&mut data, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn igather_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<Request<'_>> {
        let mut request_handle: i64 = 0;
        let ret = if self.rank() == root {
            let recvcount = rank_block("data", data.len(), self.size)? as i64;
            let (p, _, dt) = buf_mut(data);
            // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
            // this NULL is unambiguous); ferrompi_igather maps it to MPI_IN_PLACE, so data
            // serves as both root's send contribution and the receive buffer. recvcount is
            // checked to evenly divide data.len() above, and this branch runs only at root,
            // the only rank MPI_IN_PLACE is valid for in MPI_Igather. The returned Request does
            // not borrow data; keeping it alive and untouched until the request completes is
            // the caller's documented obligation, which this signature does not enforce.
            unsafe {
                ffi::ferrompi_igather(
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
            let (p, n, dt) = buf_mut(data);
            // SAFETY: p is non-null, so ferrompi_igather does not map it to MPI_IN_PLACE; data is
            // this rank's send block. It is also passed as recvbuf because strict MPI builds
            // reject a NULL recvbuf at non-root; MPI ignores recvbuf and recvcount there. The
            // returned Request does not borrow data; keeping it alive and untouched until the
            // request completes is the caller's documented obligation, which this signature
            // does not enforce.
            unsafe { ffi::ferrompi_igather(p, n, p, n, dt, root, self.handle, &mut request_handle) }
        };
        Error::check_with_op(ret, "igather_inplace")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place all-gather. Every rank's `data` is both send
    /// contribution and receive buffer.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `recvcount * size()`. Rank `r`'s contribution
    /// lives at offset `r * recvcount` and must be pre-written before the call.
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
    /// let mut data = vec![0i32; size];
    /// data[rank] = rank as i32 * 10;
    /// let req = world.iallgather_inplace(&mut data).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallgather_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<Request<'_>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_iallgather maps it to MPI_IN_PLACE. recvcount
        // is checked to evenly divide data.len() above; each rank's slot must be pre-written
        // by the caller before this call. The returned Request does not borrow data; keeping
        // it alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iallgather(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iallgather_inplace")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place scatter. At root, `data` is the `sendcount * size()`
    /// send buffer; root's own slot is retained in place. At non-root, `data` is
    /// the `recvcount`-element receive buffer.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `sendcount * size()`. Rank `r`'s slot is
    /// `data[r*sendcount .. (r+1)*sendcount]`. After the wait, only root's own
    /// slot is guaranteed to remain intact.
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
    /// if world.rank() == 0 {
    ///     let mut data = vec![0i32, 10, 20, 30];
    ///     let req = world.iscatter_inplace(&mut data, 0).unwrap();
    ///     req.wait().unwrap();
    /// } else {
    ///     let mut data = vec![0i32; 1];
    ///     let req = world.iscatter_inplace(&mut data, 0).unwrap();
    ///     req.wait().unwrap();
    /// }
    /// ```
    pub fn iscatter_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        root: i32,
    ) -> Result<Request<'_>> {
        let (sendbuf, sendcount, recvbuf, recvcount, dt) =
            scatter_inplace_args(data, self.rank() == root, self.size)?;
        let mut request_handle: i64 = 0;
        // SAFETY: at root, recvbuf is NULL, the in-place marker (buf's pointer is never
        // null, so this NULL is unambiguous); ferrompi_iscatter maps it to MPI_IN_PLACE so
        // root's own slot is retained. scatter_inplace_args checks that the block size
        // evenly divides data.len(). At non-root, sendbuf is null, which the MPI standard
        // ignores on non-root scatter. The returned Request does not borrow data; keeping
        // it alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iscatter(
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
        Error::check_with_op(ret, "iscatter_inplace")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place all-to-all personalized communication. `data` is
    /// both send and receive buffer on every rank.
    ///
    /// Before the call, rank `r` must pre-write into slot `s` (at offset
    /// `s * count`) the payload it wishes to send to rank `s`. After the wait,
    /// the same slot contains the data received FROM rank `s`.
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
    /// let mut data: Vec<i32> = (0..size as i32).map(|s| r * 10 + s).collect();
    /// let req = world.ialltoall_inplace(&mut data).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ialltoall_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<Request<'_>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_ialltoall maps it to MPI_IN_PLACE. recvcount
        // is checked to evenly divide data.len() above; the caller must pre-write each slot
        // before calling this method. The returned Request does not borrow data; keeping it
        // alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_ialltoall(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "ialltoall_inplace")?;
        Ok(Request::new(request_handle, RequestKind::Collective))
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::test_comm;
    use crate::error::Error;
    use crate::ReduceOp;

    #[test]
    fn iallreduce_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iallreduce(&send, &mut recv, ReduceOp::Sum);
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
    fn ireduce_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.ireduce(&send, &mut recv, ReduceOp::Sum, 0);
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
    fn iscan_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iscan(&send, &mut recv, ReduceOp::Sum);
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
    fn iexscan_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iexscan(&send, &mut recv, ReduceOp::Sum);
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
    fn iallgather_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let mut data = vec![0u32; 7];
        let result = comm.iallgather_inplace(&mut data);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }

    #[test]
    fn ialltoall_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let mut data = vec![0u32; 7];
        let result = comm.ialltoall_inplace(&mut data);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }
}
