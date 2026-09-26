//! Nonblocking collective operations: ibroadcast, iallreduce, ireduce, igather, etc.

use crate::comm::{check_rank_slots, check_same_len, rank_block, Communicator};
use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::request::Request;
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
    pub fn ibroadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: the returned Request does not borrow `data`; keeping it alive and
        // untouched until the request completes is the caller's documented obligation,
        // which this signature does not enforce.
        let ret = unsafe { ffi::ferrompi_ibcast(p, n, dt, root, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibcast")?;
        Ok(Request::new(request_handle))
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
    ) -> Result<Request> {
        check_same_len(send.len(), recv.len())?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking reduce to root.
    ///
    /// Initiates a reduction operation and returns immediately with a [`Request`]
    /// handle. The buffers must remain valid until the request is completed.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result (only significant at root)
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
    /// let req = world.ireduce(&send, &mut recv, ReduceOp::Sum, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ireduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<Request> {
        check_same_len(send.len(), recv.len())?;
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
        Ok(Request::new(request_handle))
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
    /// [`Error::InvalidBuffer`] if this rank is `root` and `recv.len() < send.len() *
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
    ) -> Result<Request> {
        if self.rank == root {
            check_rank_slots(recv.len(), send.len(), self.size)?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-gather.
    ///
    /// Initiates an all-gather operation and returns immediately with a
    /// [`Request`] handle. Each process sends `send.len()` elements and
    /// receives from all.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBuffer`] if `recv.len() < send.len() * size()`.
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
    pub fn iallgather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<Request> {
        check_rank_slots(recv.len(), send.len(), self.size)?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking scatter from root.
    ///
    /// Initiates a scatter operation and returns immediately with a [`Request`]
    /// handle. Root sends `recv.len() * size` elements total, each process
    /// receives `recv.len()` elements.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBuffer`] if this rank is `root` and `send.len() < recv.len() *
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
    ) -> Result<Request> {
        if self.rank == root {
            check_rank_slots(send.len(), recv.len(), self.size)?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking barrier.
    ///
    /// Initiates a barrier synchronization and returns immediately with a
    /// [`Request`] handle. The barrier is complete when the request is waited on.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let req = world.ibarrier().unwrap();
    /// // ... do other work ...
    /// req.wait().unwrap();
    /// ```
    pub fn ibarrier(&self) -> Result<Request> {
        let mut request_handle: i64 = 0;
        // SAFETY: this call takes only the communicator handle and a request out-pointer;
        // there is no data buffer, so the returned Request has no buffer-lifetime
        // obligation.
        let ret = unsafe { ffi::ferrompi_ibarrier(self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibarrier")?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking inclusive prefix reduction (scan).
    ///
    /// Initiates an inclusive scan and returns immediately with a [`Request`]
    /// handle. On rank `i`, `recv` will contain the reduction of `send` values
    /// from ranks `0..=i` once the request completes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBuffer`] if `send.len() != recv.len()`.
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
    ) -> Result<Request> {
        check_same_len(send.len(), recv.len())?;
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
        Ok(Request::new(request_handle))
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
    /// Returns [`Error::InvalidBuffer`] if `send.len() != recv.len()`.
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
    ) -> Result<Request> {
        check_same_len(send.len(), recv.len())?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-to-all personalized communication.
    ///
    /// Initiates an all-to-all operation and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidBuffer`] if `send.len() != recv.len()` or
    /// `send.len()` is not evenly divisible by the communicator size.
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
    pub fn ialltoall<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<Request> {
        check_same_len(send.len(), recv.len())?;
        let count = rank_block(send.len(), self.size)? as i64;
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
        Ok(Request::new(request_handle))
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
    /// Returns [`Error::InvalidBuffer`] if `send.len() != recv.len() * size`.
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
    ) -> Result<Request> {
        check_same_len(rank_block(send.len(), self.size)?, recv.len())?;
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
        Ok(Request::new(request_handle))
    }

    /// Nonblocking in-place gather at root. Non-root ranks must use
    /// `igather` — this method returns `Error::InvalidOp` on non-root.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `recvcount * size()` where `recvcount` is
    /// the per-rank count. Rank `r`'s contribution lives at offset
    /// `r * recvcount`. Root's own contribution must be pre-written into
    /// `data[rank() * recvcount .. (rank()+1) * recvcount]` before the call.
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
    ///     let req = world.igather_inplace(&mut data, 0).unwrap();
    ///     req.wait().unwrap();
    /// }
    /// ```
    pub fn igather_inplace<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<Request> {
        if self.rank() != root {
            return Err(Error::InvalidOp);
        }
        let recvcount = rank_block(data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_igather maps it to MPI_IN_PLACE, so data
        // serves as both root's send contribution and the receive buffer. recvcount is
        // checked to evenly divide data.len() above, and the guard above guarantees
        // self.rank() == root, the only rank MPI_IN_PLACE is valid for in MPI_Igather. The
        // returned Request does not borrow data; keeping it alive and untouched until the
        // request completes is the caller's documented obligation, which this signature does
        // not enforce.
        let ret = unsafe {
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
        };
        Error::check_with_op(ret, "igather_inplace")?;
        Ok(Request::new(request_handle))
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
    /// let req = world.iallgather_inplace(&mut data).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallgather_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<Request> {
        let recvcount = rank_block(data.len(), self.size)? as i64;
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
        Ok(Request::new(request_handle))
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
    ///     let req = world.iscatter_inplace(&mut data, 0).unwrap();
    ///     req.wait().unwrap();
    /// } else {
    ///     let mut data = vec![0i32; 1];
    ///     let req = world.iscatter_inplace(&mut data, 0).unwrap();
    ///     req.wait().unwrap();
    /// }
    /// ```
    pub fn iscatter_inplace<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<Request> {
        let is_root = self.rank() == root;
        let (sendbuf, sendcount, recvbuf, recvcount, dt) = if is_root {
            let per = rank_block(data.len(), self.size)? as i64;
            let (sp, _, dt) = buf(data);
            (sp, per, std::ptr::null_mut::<std::ffi::c_void>(), 0i64, dt)
        } else {
            let (rp, rn, dt) = buf_mut(data);
            (std::ptr::null::<std::ffi::c_void>(), 0i64, rp, rn, dt)
        };
        let mut request_handle: i64 = 0;
        // SAFETY: at root, recvbuf is NULL, the in-place marker (buf's pointer is never
        // null, so this NULL is unambiguous); ferrompi_iscatter maps it to MPI_IN_PLACE so
        // root's own slot is retained. per is checked to evenly divide data.len() above. At
        // non-root, sendbuf is null, which the MPI standard ignores on non-root scatter.
        // The returned Request does not borrow data; keeping it alive and untouched until
        // the request completes is the caller's documented obligation, which this
        // signature does not enforce.
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
        Ok(Request::new(request_handle))
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
    /// let req = world.ialltoall_inplace(&mut data).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ialltoall_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<Request> {
        let recvcount = rank_block(data.len(), self.size)? as i64;
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
        Ok(Request::new(request_handle))
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::Communicator;
    use crate::error::Error;
    use crate::ReduceOp;

    fn dummy_comm() -> Communicator {
        Communicator {
            handle: 0,
            rank: 0,
            size: 1,
        }
    }

    #[test]
    fn iallreduce_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iallreduce(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn ireduce_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.ireduce(&send, &mut recv, ReduceOp::Sum, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn iscan_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iscan(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn iexscan_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.iexscan(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn igather_inplace_nonroot_returns_invalid_op() {
        let comm = Communicator {
            handle: 0,
            rank: 1,
            size: 4,
        };
        let mut data = vec![0u32; 4];
        let result = comm.igather_inplace(&mut data, 0);
        assert!(matches!(result, Err(Error::InvalidOp)));
    }

    #[test]
    fn iallgather_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7];
        let result = comm.iallgather_inplace(&mut data);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn ialltoall_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7];
        let result = comm.ialltoall_inplace(&mut data);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }
}
