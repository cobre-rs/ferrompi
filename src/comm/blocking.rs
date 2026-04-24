//! Blocking collective operations: barrier, broadcast, reduce, allreduce, scan, gather, scatter, alltoall.

use crate::comm::Communicator;
use crate::datatype::{MpiDatatype, MpiIndexedDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::ReduceOp;

impl Communicator {
    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Barrier synchronization.
    ///
    /// All processes in the communicator must call this function. No process
    /// will return until all processes have entered the barrier.
    pub fn barrier(&self) -> Result<()> {
        let ret = unsafe { ffi::ferrompi_barrier(self.handle) };
        Error::check(ret)
    }

    // ========================================================================
    // Generic Blocking Collectives
    // ========================================================================

    /// Broadcast a slice from root to all processes.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to broadcast (input at root, output at others)
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// if world.rank() == 0 {
    ///     data.fill(42.0);
    /// }
    /// world.broadcast(&mut data, 0).unwrap();
    /// ```
    pub fn broadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_bcast(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Reduce values to the root process.
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
    /// world.reduce(&send, &mut recv, ReduceOp::Sum, 0).unwrap();
    /// ```
    pub fn reduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_reduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Reduce a single scalar value to the root process.
    ///
    /// Convenience method that wraps [`reduce`](Self::reduce) for a single element.
    /// The result is only meaningful at the root process.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to contribute from this process
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let sum = world.reduce_scalar(world.rank() as f64, ReduceOp::Sum, 0).unwrap();
    /// if world.rank() == 0 {
    ///     println!("Sum of all ranks: {sum}");
    /// }
    /// ```
    pub fn reduce_scalar<T: MpiDatatype>(&self, value: T, op: ReduceOp, root: i32) -> Result<T> {
        let send = [value];
        let mut recv = [value]; // placeholder, will be overwritten at root
        self.reduce(&send, &mut recv, op, root)?;
        Ok(recv[0])
    }

    /// In-place reduce to the root process.
    ///
    /// At root: `data` is both input and output (the reduction result overwrites
    /// the input).
    /// At non-root: `data` is the send buffer only.
    ///
    /// This avoids allocating a separate receive buffer at the root, which is
    /// useful for large reductions where memory is a concern.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to reduce (input on all ranks, output only at root)
    /// * `op` - Reduction operation
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![world.rank() as f64; 10];
    /// world.reduce_inplace(&mut data, ReduceOp::Sum, 0).unwrap();
    /// if world.rank() == 0 {
    ///     println!("Reduced result: {:?}", &data[..3]);
    /// }
    /// ```
    pub fn reduce_inplace<T: MpiDatatype>(
        &self,
        data: &mut [T],
        op: ReduceOp,
        root: i32,
    ) -> Result<()> {
        let is_root = if self.rank() == root { 1i32 } else { 0i32 };
        let ret = unsafe {
            ffi::ferrompi_reduce_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                op as i32,
                root,
                is_root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce values (reduce and broadcast result to all).
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for result
    /// * `op` - Reduction operation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// world.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// ```
    pub fn allreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_allreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce values in place.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![world.rank() as f64; 10];
    /// world.allreduce_inplace(&mut data, ReduceOp::Sum).unwrap();
    /// ```
    pub fn allreduce_inplace<T: MpiDatatype>(&self, data: &mut [T], op: ReduceOp) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allreduce_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-reduce a single scalar value.
    ///
    /// Convenience method for reducing a single element.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let sum = world.allreduce_scalar(world.rank() as f64, ReduceOp::Sum).unwrap();
    /// println!("Sum of all ranks: {sum}");
    /// ```
    pub fn allreduce_scalar<T: MpiDatatype>(&self, value: T, op: ReduceOp) -> Result<T> {
        let send = [value];
        // SAFETY: T is Copy, so zero-init is safe for numeric types
        let mut recv = [value]; // placeholder, will be overwritten
        self.allreduce(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// All-reduce paired value+index types using `MPI_MAXLOC` or `MPI_MINLOC`.
    ///
    /// This method finds the global maximum (or minimum) value across all ranks
    /// together with the rank index where it occurred. Only [`ReduceOp::MaxLoc`]
    /// and [`ReduceOp::MinLoc`] are accepted; passing any other op returns
    /// [`Error::InvalidOp`].
    ///
    /// The type parameter `T` must implement [`MpiIndexedDatatype`], which is
    /// only satisfied by the six MPI predefined paired types: [`FloatInt`],
    /// [`DoubleInt`], [`LongInt`], [`Int2`], [`ShortInt`], [`LongDoubleInt`].
    /// These types are **not** interchangeable with the primitive types used by
    /// `allreduce` — they are distinct at the type-system level.
    ///
    /// # Arguments
    ///
    /// * `send` - Slice of paired values contributed by this process
    /// * `recv` - Output buffer; must be the same length as `send`
    /// * `op` - Must be [`ReduceOp::MaxLoc`] or [`ReduceOp::MinLoc`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if `send.len() != recv.len()`
    /// - [`Error::InvalidOp`] if `op` is not `MaxLoc` or `MinLoc`
    /// - An MPI error if the library rejects the combination
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp, DoubleInt};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let rank = world.rank();
    ///
    /// // Each rank contributes its rank as value and index.
    /// let send = [DoubleInt { value: rank as f64, index: rank }];
    /// let mut recv = [DoubleInt { value: 0.0, index: 0 }];
    /// world.allreduce_indexed(&send, &mut recv, ReduceOp::MaxLoc).unwrap();
    /// // Every rank now holds { value: (size-1) as f64, index: size-1 }
    /// ```
    ///
    /// [`FloatInt`]: crate::FloatInt
    /// [`DoubleInt`]: crate::DoubleInt
    /// [`LongInt`]: crate::LongInt
    /// [`Int2`]: crate::Int2
    /// [`ShortInt`]: crate::ShortInt
    /// [`LongDoubleInt`]: crate::LongDoubleInt
    pub fn allreduce_indexed<T: MpiIndexedDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        if !matches!(op, ReduceOp::MaxLoc | ReduceOp::MinLoc) {
            return Err(Error::InvalidOp);
        }
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_allreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Inclusive prefix reduction (scan).
    ///
    /// On rank `i`, `recv` contains the reduction of `send` values from ranks
    /// `0..=i`. This is the inclusive variant: every rank's own contribution is
    /// included in its result.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to contribute from this process
    /// * `recv` - Buffer for the prefix-reduced result (must be same length as `send`)
    /// * `op` - Reduction operation
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
    /// world.scan(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// // On rank i, recv[j] == (i + 1) * send[j]
    /// ```
    pub fn scan<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], op: ReduceOp) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_scan(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Exclusive prefix reduction (exscan).
    ///
    /// On rank `i`, `recv` contains the reduction of `send` values from ranks
    /// `0..i` (i.e., excluding rank `i`'s own contribution).
    ///
    /// # Rank 0 Behavior
    ///
    /// **Per the MPI standard, the contents of `recv` on rank 0 are undefined.**
    /// Callers must not rely on the receive buffer contents on rank 0.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to contribute from this process
    /// * `recv` - Buffer for the prefix-reduced result (must be same length as `send`;
    ///   **undefined on rank 0**)
    /// * `op` - Reduction operation
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
    /// world.exscan(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// // On rank i > 0, recv[j] == i * send[j]
    /// // On rank 0, recv is undefined per the MPI standard.
    /// ```
    pub fn exscan<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], op: ReduceOp) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_exscan(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Inclusive scan of a single scalar value.
    ///
    /// Convenience method for scanning a single element. On rank `i`, returns
    /// the reduction of the input values from ranks `0..=i`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let prefix_sum = world.scan_scalar(1.0f64, ReduceOp::Sum).unwrap();
    /// // On rank i, prefix_sum == (i + 1) as f64
    /// ```
    pub fn scan_scalar<T: MpiDatatype>(&self, value: T, op: ReduceOp) -> Result<T> {
        let send = [value];
        let mut recv = [value]; // placeholder, will be overwritten
        self.scan(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// Exclusive scan of a single scalar value.
    ///
    /// Convenience method for exclusive-scanning a single element. On rank `i`,
    /// returns the reduction of input values from ranks `0..i`.
    ///
    /// # Rank 0 Behavior
    ///
    /// **Per the MPI standard, the return value on rank 0 is undefined.**
    /// Callers must not rely on the result on rank 0.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let prefix_sum = world.exscan_scalar(1.0f64, ReduceOp::Sum).unwrap();
    /// // On rank i > 0, prefix_sum == i as f64
    /// // On rank 0, the result is undefined per the MPI standard.
    /// ```
    pub fn exscan_scalar<T: MpiDatatype>(&self, value: T, op: ReduceOp) -> Result<T> {
        let send = [value];
        let mut recv = [value]; // placeholder, will be overwritten by MPI (except rank 0)
        self.exscan(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// Gather values to the root process.
    ///
    /// Each process sends `send.len()` elements. Root receives
    /// `send.len() * size` elements total.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root, must be
    ///   `send.len() * size` elements)
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
    /// world.gather(&send, &mut recv, 0).unwrap();
    /// ```
    pub fn gather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_gather(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-gather values (gather and broadcast to all).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as i32; 3];
    /// let mut recv = vec![0i32; 3 * world.size() as usize];
    /// world.allgather(&send, &mut recv).unwrap();
    /// ```
    pub fn allgather<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allgather(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Scatter values from root to all processes.
    ///
    /// Root sends `recv.len() * size` elements total, each process receives
    /// `recv.len()` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![0.0f64; 5 * world.size() as usize];
    /// let mut recv = vec![0.0f64; 5];
    /// world.scatter(&send, &mut recv, 0).unwrap();
    /// ```
    pub fn scatter<T: MpiDatatype>(&self, send: &[T], recv: &mut [T], root: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_scatter(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-to-all personalized communication.
    ///
    /// Each process sends `send.len() / size` elements to every other process
    /// and receives the same amount from each.
    ///
    /// `send` must have exactly `count * size` elements, where `count`
    /// is the number of elements sent to each process.
    /// `recv` must have the same length as `send`.
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
    /// world.alltoall(&send, &mut recv).unwrap();
    /// ```
    pub fn alltoall<T: MpiDatatype>(&self, send: &[T], recv: &mut [T]) -> Result<()> {
        let size = self.size() as usize;
        if send.len() != recv.len() || send.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let count = (send.len() / size) as i64;
        let ret = unsafe {
            ffi::ferrompi_alltoall(
                send.as_ptr().cast::<std::ffi::c_void>(),
                count,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                count,
                T::TAG as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Reduce-scatter with uniform block size.
    ///
    /// Performs an element-wise reduction across all processes, then scatters
    /// the result so that each process receives `recv.len()` elements.
    /// `send` must have exactly `recv.len() * size` elements.
    ///
    /// This is equivalent to [`allreduce`](Self::allreduce) followed by each
    /// process keeping only its portion, but is more efficient because the MPI
    /// implementation can fuse the two operations.
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
    /// world.reduce_scatter_block(&send, &mut recv, ReduceOp::Sum).unwrap();
    /// ```
    pub fn reduce_scatter_block<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        let size = self.size() as usize;
        if send.len() != recv.len() * size {
            return Err(Error::InvalidBuffer);
        }
        let ret = unsafe {
            ffi::ferrompi_reduce_scatter_block(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::Communicator;
    use crate::datatype::DoubleInt;
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
    fn reduce_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5]; // different length
        let result = comm.reduce(&send, &mut recv, ReduceOp::Sum, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allreduce_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.allreduce(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scan_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.scan(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn exscan_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = comm.exscan(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allreduce_indexed_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![
            DoubleInt {
                value: 1.0,
                index: 0,
            };
            10
        ];
        let mut recv = vec![
            DoubleInt {
                value: 0.0,
                index: 0,
            };
            5
        ];
        let result = comm.allreduce_indexed(&send, &mut recv, ReduceOp::MaxLoc);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allreduce_indexed_invalid_op_returns_invalid_op() {
        let comm = dummy_comm();
        let send = vec![
            DoubleInt {
                value: 1.0,
                index: 0,
            };
            4
        ];
        let mut recv = vec![
            DoubleInt {
                value: 0.0,
                index: 0,
            };
            4
        ];
        for op in [
            ReduceOp::Sum,
            ReduceOp::Max,
            ReduceOp::Min,
            ReduceOp::Prod,
            ReduceOp::BitwiseOr,
            ReduceOp::BitwiseAnd,
            ReduceOp::BitwiseXor,
            ReduceOp::LogicalOr,
            ReduceOp::LogicalAnd,
            ReduceOp::LogicalXor,
        ] {
            let result = comm.allreduce_indexed(&send, &mut recv, op);
            assert!(
                matches!(result, Err(Error::InvalidOp)),
                "Expected InvalidOp for op {op:?} on indexed type"
            );
        }
    }
}
