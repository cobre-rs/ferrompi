//! Nonblocking collective operations: ibroadcast, iallreduce, ireduce, igather, etc.

use crate::comm::Communicator;
use crate::datatype::MpiDatatype;
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
        let ret = unsafe {
            ffi::ferrompi_ibcast(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
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
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iallreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
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
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ireduce(
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
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_igather(
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
        Error::check_with_op(ret, "igather")?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-gather.
    ///
    /// Initiates an all-gather operation and returns immediately with a
    /// [`Request`] handle. Each process sends `send.len()` elements and
    /// receives from all.
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
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iallgather(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iallgather")?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking scatter from root.
    ///
    /// Initiates a scatter operation and returns immediately with a [`Request`]
    /// handle. Root sends `recv.len() * size` elements total, each process
    /// receives `recv.len()` elements.
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
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iscatter(
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
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iscan(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
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
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iexscan(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
                &mut request_handle,
            )
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
        let size = self.size() as usize;
        if send.len() != recv.len() || send.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let count = (send.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ialltoall(
                send.as_ptr().cast::<std::ffi::c_void>(),
                count,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                count,
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
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
        let size = self.size() as usize;
        if send.len() != recv.len() * size {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ireduce_scatter_block(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
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
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice. We cast to *mut c_void
            // as required by the C FFI. The guard above guarantees self.rank() == root, so
            // is_root is hardcoded to 1. The buffer outlives the returned Request handle;
            // the caller must call wait() before accessing or dropping the buffer.
            ffi::ferrompi_igather_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
                root,
                1, // is_root = true by the rank guard above
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
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice. We cast to *mut c_void
            // as required by the C FFI. Each rank's contribution (at offset rank*recvcount)
            // must be pre-written by the caller. The buffer outlives the returned Request;
            // the caller must call wait() before accessing or dropping the buffer.
            ffi::ferrompi_iallgather_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
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
        let size = self.size() as usize;
        let (sendbuf, sendcount, recvbuf, recvcount, is_root_flag) = if is_root {
            if size == 0 || data.len() % size != 0 {
                return Err(Error::InvalidBuffer);
            }
            let per = (data.len() / size) as i64;
            (
                data.as_ptr().cast::<std::ffi::c_void>(),
                per,
                std::ptr::null_mut::<std::ffi::c_void>(),
                0i64,
                1i32,
            )
        } else {
            (
                std::ptr::null::<std::ffi::c_void>(),
                0i64,
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                0i32,
            )
        };
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: At root, sendbuf points to valid data of length sendcount*size elements
            // (guaranteed by the divisibility check above); recvbuf is null (MPI_IN_PLACE path).
            // At non-root, recvbuf points to a valid mutable slice of length recvcount elements;
            // sendbuf is null (MPI standard ignores sendbuf on non-root scatter). Both pointers
            // are cast to *const/*mut c_void as required by the C FFI. The buffer outlives the
            // returned Request; the caller must call wait() before accessing or dropping it.
            ffi::ferrompi_iscatter_inplace(
                sendbuf,
                sendcount,
                recvbuf,
                recvcount,
                T::TAG as i32,
                root,
                is_root_flag,
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
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            // SAFETY: data is a valid, exclusively-owned mutable slice of length recvcount*size
            // elements (guaranteed by the divisibility check above). We cast to *mut c_void as
            // required by the C FFI. MPI_IN_PLACE is passed as sendbuf in the C wrapper; the
            // caller must pre-write each slot before calling this method. The buffer outlives
            // the returned Request; the caller must call wait() before accessing or dropping it.
            ffi::ferrompi_ialltoall_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
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
