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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
}
