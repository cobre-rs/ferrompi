//! Blocking collective operations: barrier, broadcast, reduce, allreduce, scan, gather, scatter, alltoall.

use crate::comm::Communicator;
use crate::datatype::{buf, buf_mut, BytePermutable, DatatypeTag, MpiDatatype, MpiIndexedDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::op::UserOp;
use crate::ReduceOp;

impl Communicator {
    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Barrier synchronization.
    ///
    /// All processes in the communicator must call this function. No process
    /// will return until all processes have entered the barrier.
    #[inline]
    pub fn barrier(&self) -> Result<()> {
        // SAFETY: self.handle is a valid communicator handle registered in the C-side
        // communicator table. ferrompi_barrier delegates to MPI_Barrier which requires no
        // buffers; only a valid communicator handle is needed.
        let ret = unsafe { ffi::ferrompi_barrier(self.handle) };
        Error::check_with_op(ret, "barrier")
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
    #[inline]
    pub fn broadcast<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<()> {
        let (p, n, dt) = buf_mut(data);
        // SAFETY: this blocking call returns only after MPI is done with the buffer.
        let ret = unsafe { ffi::ferrompi_bcast(p, n, dt, root, self.handle) };
        Error::check_with_op(ret, "bcast")
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
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]).
        let ret = unsafe { ffi::ferrompi_reduce(sp, rp, n, dt, op as i32, root, self.handle) };
        Error::check_with_op(ret, "reduce")
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
        let (p, n, dt) = buf_mut(data);
        let ret = if self.rank() == root {
            // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
            // this NULL is unambiguous); ferrompi_reduce maps it to MPI_IN_PLACE, so data is
            // both input and output at root.
            unsafe {
                ffi::ferrompi_reduce(std::ptr::null(), p, n, dt, op as i32, root, self.handle)
            }
        } else {
            // SAFETY: data is passed as both sendbuf and recvbuf because strict MPI builds
            // reject a NULL recvbuf at non-root; MPI ignores recvbuf there, so the computation
            // is unaffected.
            unsafe { ffi::ferrompi_reduce(p, p, n, dt, op as i32, root, self.handle) }
        };
        Error::check_with_op(ret, "reduce_inplace")
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
    #[inline]
    pub fn allreduce<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]).
        let ret = unsafe { ffi::ferrompi_allreduce(sp, rp, n, dt, op as i32, self.handle) };
        Error::check_with_op(ret, "allreduce")
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
        let (p, n, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so this
        // NULL is unambiguous); ferrompi_allreduce maps it to MPI_IN_PLACE, so data serves as
        // both send and receive buffer.
        let ret =
            unsafe { ffi::ferrompi_allreduce(std::ptr::null(), p, n, dt, op as i32, self.handle) };
        Error::check_with_op(ret, "allreduce_inplace")
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
        // T is Copy, so zero-init is safe for numeric types.
        let mut recv = [value]; // placeholder, will be overwritten
        self.allreduce(&send, &mut recv, op)?;
        Ok(recv[0])
    }

    /// All-reduce values using a user-defined reduction operation.
    ///
    /// Invokes `MPI_Allreduce` with the `MPI_Op` registered inside `op`.
    /// Every rank must call this with the same `op`, the same count, and the
    /// same datatype `T`.
    ///
    /// # Arguments
    ///
    /// * `send` - Data contributed by this process
    /// * `recv` - Output buffer; must be the same length as `send`
    /// * `op`   - A user-defined reduction op created with [`UserOp::new`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if `send.len() != recv.len()`
    /// - An MPI error if the library rejects the call
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, UserOp};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    ///
    /// let op: UserOp<f64> = UserOp::new(|invec: &[f64], inoutvec: &mut [f64]| {
    ///     for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
    ///         *y = x.max(*y);
    ///     }
    /// }).unwrap();
    ///
    /// let send = vec![world.rank() as f64 + 1.5_f64];
    /// let mut recv = vec![0.0_f64];
    /// world.allreduce_with_op(&send, &mut recv, &op).unwrap();
    /// // recv[0] == (world.size() - 1) as f64 + 1.5
    /// ```
    pub fn allreduce_with_op<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: &UserOp<T>,
    ) -> Result<()> {
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). op.raw_handle() is a valid MPI_Op registered by UserOp::new and
        // kept alive for the lifetime of `op`.
        let ret =
            unsafe { ffi::ferrompi_allreduce_user_op(sp, rp, n, dt, op.raw_handle(), self.handle) };
        Error::check_with_op(ret, "allreduce_user_op")
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
        // SAFETY: send is a valid shared slice and recv is a valid exclusive slice of T
        // (T: MpiIndexedDatatype — one of the six predefined MPI paired types). They cannot
        // alias (Rust borrow rules). send.len() == recv.len() verified above. op has been
        // validated to be MaxLoc or MinLoc, which are the only valid ops for indexed types.
        // T::TAG matches T's MPI paired datatype. Slices outlive this call.
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
        Error::check_with_op(ret, "allreduce_indexed")
    }

    /// All-reduce arbitrary `Copy` types using `MPI_BYTE`-typed bitwise reductions.
    ///
    /// Performs a bitwise reduction across all ranks in the communicator. Each
    /// element of `recv` receives the result of applying `op` element-wise across
    /// the corresponding elements of each rank's `send` buffer.
    ///
    /// The buffer is transmitted as a flat array of bytes via `MPI_BYTE`, so the
    /// count passed to MPI is `send.len() * size_of::<T>()`.
    ///
    /// Only `BitwiseOr`, `BitwiseAnd`, `BitwiseXor` are accepted. For
    /// floating-point or indexed reductions, use [`allreduce`] or
    /// [`allreduce_indexed`].
    ///
    /// # Arguments
    ///
    /// * `send` - Data contributed by this process
    /// * `recv` - Output buffer; must be the same length as `send`
    /// * `op` - Must be [`ReduceOp::BitwiseOr`], [`ReduceOp::BitwiseAnd`], or
    ///   [`ReduceOp::BitwiseXor`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOp`] if `op` is not one of the three bitwise ops
    /// - [`Error::InvalidBuffer`] if `send.len() != recv.len()` or the total
    ///   byte count overflows `i64::MAX`
    /// - [`Error::Mpi`] if the MPI layer rejects the call
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let rank = world.rank() as u64;
    ///
    /// // Each rank contributes a different bit; OR across all ranks gives 0b1111
    /// let data: [u64; 4] = [1u64 << rank; 4];
    /// let mut recv = [0u64; 4];
    /// world.allreduce_bytes(&data, &mut recv, ReduceOp::BitwiseOr).unwrap();
    /// assert_eq!(recv, [0b1111u64; 4]);
    /// ```
    ///
    /// [`allreduce`]: Communicator::allreduce
    /// [`allreduce_indexed`]: Communicator::allreduce_indexed
    pub fn allreduce_bytes<T: BytePermutable>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<()> {
        if !matches!(
            op,
            ReduceOp::BitwiseOr | ReduceOp::BitwiseAnd | ReduceOp::BitwiseXor
        ) {
            return Err(Error::InvalidOp);
        }
        if send.len() != recv.len() {
            return Err(Error::InvalidBuffer);
        }
        let byte_count = send
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::InvalidBuffer)?;
        if byte_count > i64::MAX as usize {
            return Err(Error::InvalidBuffer);
        }
        // SAFETY:
        // - send and recv are valid slices of T where T: BytePermutable (Copy + Send + 'static).
        // - byte_count = send.len() * size_of::<T>() bytes, which is the exact memory
        //   footprint of each slice. The cast to *const c_void / *mut c_void is safe
        //   because we pass the byte count to MPI (MPI_BYTE datatype), so MPI treats
        //   the buffer as raw bytes matching exactly the memory of the slices.
        // - DatatypeTag::Byte maps to MPI_BYTE in the C layer (case FERROMPI_BYTE).
        // - send and recv do not alias (send is &[T], recv is &mut [T]).
        let ret = unsafe {
            ffi::ferrompi_allreduce(
                send.as_ptr().cast::<std::ffi::c_void>(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                byte_count as i64,
                DatatypeTag::Byte as i32,
                op as i32,
                self.handle,
            )
        };
        Error::check_with_op(ret, "allreduce_bytes")
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
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]).
        let ret = unsafe { ffi::ferrompi_scan(sp, rp, n, dt, op as i32, self.handle) };
        Error::check_with_op(ret, "scan")
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
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). MPI leaves recv undefined on rank 0, documented above.
        let ret = unsafe { ffi::ferrompi_exscan(sp, rp, n, dt, op as i32, self.handle) };
        Error::check_with_op(ret, "exscan")
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
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]); at non-root, recv is ignored by
        // MPI. The root-side receive-length relation (recv.len() >= send.len() * size) is not
        // checked by this function.
        let ret = unsafe { ffi::ferrompi_gather(sp, n, rp, n, dt, root, self.handle) };
        Error::check_with_op(ret, "gather")
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
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). The every-rank receive-length
        // relation (recv.len() >= send.len() * size) is not checked by this function.
        let ret = unsafe { ffi::ferrompi_allgather(sp, n, rp, n, dt, self.handle) };
        Error::check_with_op(ret, "allgather")
    }

    /// Gather values in place. At root, `data` is both the send contribution and the
    /// receive buffer; non-root ranks must call `gather` (not `gather_inplace`) — this
    /// method returns `Error::InvalidOp` on non-root.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `recvcount * size()` where `recvcount` is the per-rank
    /// count. Rank `r`'s contribution lives at offset `r * recvcount`. Root's own
    /// contribution must be pre-written into `data[rank() * recvcount .. (rank()+1) *
    /// recvcount]`.
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
    /// let rank = world.rank() as usize;
    /// let size = world.size() as usize;
    /// // Root allocates the full buffer; each rank's slot is at offset rank * recvcount.
    /// // recvcount = 1 in this example.
    /// if world.rank() == 0 {
    ///     let mut data = vec![0i32; size]; // slot 0..size
    ///     data[rank] = rank as i32 * 10;   // root pre-writes its own slot
    ///     world.gather_inplace(&mut data, 0).unwrap();
    ///     // data[r] == r * 10 for all r
    /// }
    /// ```
    pub fn gather_inplace<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<()> {
        if self.rank() != root {
            return Err(Error::InvalidOp);
        }
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so this
        // NULL is unambiguous); ferrompi_gather maps it to MPI_IN_PLACE, so data serves as both
        // root's send contribution (at offset rank*recvcount) and the receive buffer.
        // data.len() % size == 0 is checked above, and the guard above guarantees
        // self.rank() == root, the only rank MPI_IN_PLACE is valid for in MPI_Gather.
        let ret = unsafe {
            ffi::ferrompi_gather(std::ptr::null(), 0, p, recvcount, dt, root, self.handle)
        };
        Error::check_with_op(ret, "gather_inplace")
    }

    /// All-gather values in place. Every rank's `data` is both send contribution and
    /// receive buffer.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `recvcount * size()`. Rank `r`'s contribution lives at
    /// offset `r * recvcount` and must be pre-written before the call.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidBuffer` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank() as usize;
    /// let size = world.size() as usize;
    /// // Each rank allocates the full buffer and pre-writes its own slot at offset rank.
    /// let mut data = vec![0i32; size];
    /// data[rank] = rank as i32 * 10;
    /// world.allgather_inplace(&mut data).unwrap();
    /// // data[r] == r * 10 for all r, on every rank
    /// ```
    pub fn allgather_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<()> {
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so this
        // NULL is unambiguous); ferrompi_allgather maps it to MPI_IN_PLACE. data.len() % size
        // == 0 is checked above; each rank's slot must be pre-written by the caller before this
        // call.
        let ret =
            unsafe { ffi::ferrompi_allgather(std::ptr::null(), 0, p, recvcount, dt, self.handle) };
        Error::check_with_op(ret, "allgather_inplace")
    }

    /// Scatter values in place. At root, `data` is the `sendcount * size()` send buffer;
    /// root's own slot is retained in place. At non-root, `data` is the
    /// `recvcount`-element receive buffer.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `sendcount * size()`. Rank `r`'s slot is
    /// `data[r*sendcount .. (r+1)*sendcount]`. After the call, only root's own slot is
    /// guaranteed to remain intact; other slots are unspecified.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidBuffer` at root if `data.len()` is not divisible by
    /// `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // With 4 ranks: root pre-populates [0, 10, 20, 30], each non-root has a 1-element buf.
    /// // After the call: rank 0 retains data[0]==0, rank 1 gets [10], rank 2 [20], rank 3 [30].
    /// if world.rank() == 0 {
    ///     let mut data = vec![0i32, 10, 20, 30];
    ///     world.scatter_inplace(&mut data, 0).unwrap();
    ///     assert_eq!(data[0], 0); // root retains its own slot
    /// } else {
    ///     let mut data = vec![0i32; 1];
    ///     world.scatter_inplace(&mut data, 0).unwrap();
    /// }
    /// ```
    pub fn scatter_inplace<T: MpiDatatype>(&self, data: &mut [T], root: i32) -> Result<()> {
        let is_root = self.rank() == root;
        let size = self.size() as usize;
        let (sendbuf, sendcount, recvbuf, recvcount, dt) = if is_root {
            if size == 0 || data.len() % size != 0 {
                return Err(Error::InvalidBuffer);
            }
            let per = (data.len() / size) as i64;
            let (sp, _, dt) = buf(data);
            (sp, per, std::ptr::null_mut::<std::ffi::c_void>(), 0i64, dt)
        } else {
            let (rp, rn, dt) = buf_mut(data);
            (std::ptr::null::<std::ffi::c_void>(), 0i64, rp, rn, dt)
        };
        // SAFETY: at root, recvbuf is NULL, the in-place marker (buf's pointer is never null,
        // so this NULL is unambiguous); ferrompi_scatter maps it to MPI_IN_PLACE so root's own
        // slot is retained. data.len() % size == 0 is checked above. At non-root, sendbuf is
        // null, which the MPI standard ignores on non-root scatter.
        let ret = unsafe {
            ffi::ferrompi_scatter(
                sendbuf,
                sendcount,
                recvbuf,
                recvcount,
                dt,
                root,
                self.handle,
            )
        };
        Error::check_with_op(ret, "scatter_inplace")
    }

    /// All-to-all personalized communication in place. `data` is both send and receive
    /// buffer on every rank. Before the call, rank `r` must pre-write into
    /// `data[s*count..(s+1)*count]` the payload it wishes to send to rank `s` (for `s`
    /// in `0..size()`). After the call, the same slot contains the data received FROM
    /// rank `s`.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `count * size()` where `count` is the per-rank element
    /// count. Slot `s` at `data[s*count..(s+1)*count]` holds data sent to (and later
    /// received from) rank `s`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidBuffer` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // With 4 ranks: rank r pre-writes data[s] = r*10 + s (payload destined for rank s).
    /// // After the call: data[s] == s*10 + r (data received FROM rank s).
    /// let r = world.rank() as i32;
    /// let size = world.size() as usize;
    /// let mut data: Vec<i32> = (0..size as i32).map(|s| r * 10 + s).collect();
    /// world.alltoall_inplace(&mut data).unwrap();
    /// for s in 0..size as i32 {
    ///     assert_eq!(data[s as usize], s * 10 + r);
    /// }
    /// ```
    pub fn alltoall_inplace<T: MpiDatatype>(&self, data: &mut [T]) -> Result<()> {
        let size = self.size() as usize;
        if size == 0 || data.len() % size != 0 {
            return Err(Error::InvalidBuffer);
        }
        let recvcount = (data.len() / size) as i64;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so this
        // NULL is unambiguous); ferrompi_alltoall maps it to MPI_IN_PLACE. data.len() % size ==
        // 0 is checked above; the caller must pre-write each slot before calling this method.
        let ret =
            unsafe { ffi::ferrompi_alltoall(std::ptr::null(), 0, p, recvcount, dt, self.handle) };
        Error::check_with_op(ret, "alltoall_inplace")
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
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). The root-side send-length relation (send.len() >= recv.len() * size) is
        // not checked by this function.
        let ret = unsafe { ffi::ferrompi_scatter(sp, n, rp, n, dt, root, self.handle) };
        Error::check_with_op(ret, "scatter")
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
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len() and
        // divisibility by size are both verified above.
        let ret = unsafe { ffi::ferrompi_alltoall(sp, count, rp, count, dt, self.handle) };
        Error::check_with_op(ret, "alltoall")
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
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len() * size
        // is verified above.
        let ret =
            unsafe { ffi::ferrompi_reduce_scatter_block(sp, rp, n, dt, op as i32, self.handle) };
        Error::check_with_op(ret, "reduce_scatter_block")
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
    fn gather_inplace_nonroot_returns_invalid_op() {
        let comm = Communicator {
            handle: 0,
            rank: 1,
            size: 4,
        };
        let mut data = vec![0u32; 4];
        let result = comm.gather_inplace(&mut data, 0);
        assert!(matches!(result, Err(Error::InvalidOp)));
    }

    #[test]
    fn gather_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 5]; // 5 is not divisible by 4
        let result = comm.gather_inplace(&mut data, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allgather_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7]; // 7 is not divisible by 4
        let result = comm.allgather_inplace(&mut data);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scatter_inplace_root_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 5]; // 5 is not divisible by 4
        let result = comm.scatter_inplace(&mut data, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoall_inplace_mismatched_len_returns_invalid_buffer() {
        let comm = Communicator {
            handle: 0,
            rank: 0,
            size: 4,
        };
        let mut data = vec![0u32; 7]; // 7 is not divisible by 4
        let result = comm.alltoall_inplace(&mut data);
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

    #[test]
    fn allreduce_bytes_invalid_op_returns_invalid_op() {
        let comm = dummy_comm();
        let send = [1u32; 4];
        let mut recv = [0u32; 4];
        let result = comm.allreduce_bytes(&send, &mut recv, ReduceOp::Sum);
        assert!(matches!(result, Err(Error::InvalidOp)));
    }

    #[test]
    fn allreduce_bytes_mismatched_buffers_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = [1u32; 4];
        let mut recv = [0u32; 3];
        let result = comm.allreduce_bytes(&send, &mut recv, ReduceOp::BitwiseOr);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }
}
