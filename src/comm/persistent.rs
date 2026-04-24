//! Persistent collective operations (MPI 4.0+): bcast_init, allreduce_init, etc.

use crate::comm::Communicator;
use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::ReduceOp;

impl Communicator {
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
            ffi::ferrompi_allreduce_init_inplace(
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
            // SAFETY: data is a valid, exclusively-owned mutable slice. We cast to *mut c_void
            // as required by the C FFI. The guard above guarantees self.rank() == root, so
            // is_root is hardcoded to 1. The buffer must remain valid for the lifetime of the
            // returned PersistentRequest; the caller must call wait() after each start().
            ffi::ferrompi_gather_init_inplace(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcount,
                T::TAG as i32,
                root,
                1, // is_root = true by the rank guard above
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
            // SAFETY: data is a valid, exclusively-owned mutable slice. We cast to *mut c_void
            // as required by the C FFI. Each rank's contribution (at offset rank*recvcount)
            // must be pre-written by the caller before each start(). The buffer must remain
            // valid for the lifetime of the returned PersistentRequest.
            ffi::ferrompi_allgather_init_inplace(
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
            // are cast to *const/*mut c_void as required by the C FFI. The buffer must remain
            // valid for the lifetime of the returned PersistentRequest.
            ffi::ferrompi_scatter_init_inplace(
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
            // elements (guaranteed by the divisibility check above). We cast to *mut c_void as
            // required by the C FFI. MPI_IN_PLACE is passed as sendbuf in the C wrapper; the
            // caller must pre-write each slot before each start() call. The buffer must remain
            // valid for the lifetime of the returned PersistentRequest.
            ffi::ferrompi_alltoall_init_inplace(
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
