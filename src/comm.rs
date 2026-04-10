//! Safe wrappers for MPI communicator operations.
//!
//! All communication methods are generic over [`MpiDatatype`], supporting
//! `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.

use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::Request;
use crate::status::Status;
use crate::ReduceOp;

/// Split types for [`Communicator::split_type`].
///
/// These constants map to MPI communicator split type values. Currently only
/// shared-memory splits are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SplitType {
    /// Split by shared memory domain (same physical node).
    /// Maps to `MPI_COMM_TYPE_SHARED`.
    Shared = 0,
}

/// An MPI communicator.
///
/// This type wraps an MPI communicator handle and provides safe methods for
/// collective and point-to-point communication operations.
///
/// # Thread Safety
///
/// `Communicator` is `Send + Sync`, matching the thread-safety model of
/// C/Fortran MPI. The actual thread-safety guarantees depend on the
/// thread level provided by [`Mpi::init_thread()`](crate::Mpi::init_thread):
///
/// - [`ThreadLevel::Single`](crate::ThreadLevel::Single) /
///   [`Funneled`](crate::ThreadLevel::Funneled): MPI calls only from main thread
/// - [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized): MPI calls
///   from any thread, but serialized by the user (e.g., via a `Mutex`)
/// - [`ThreadLevel::Multiple`](crate::ThreadLevel::Multiple): MPI calls from
///   any thread concurrently without external synchronization
///
/// For hybrid MPI + threads programs, request at least
/// [`ThreadLevel::Funneled`](crate::ThreadLevel::Funneled) (master thread
/// makes MPI calls) or [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized)
/// (any thread, but only one at a time).
///
/// # Example
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// println!("I am rank {} of {}", world.rank(), world.size());
/// ```
pub struct Communicator {
    handle: i32,
}

// SAFETY: Communicator handles are integer indices into a C-side table.
// The C MPI library manages its own thread safety based on the thread level
// requested via MPI_Init_thread. Sending a Communicator to another thread is
// safe because MPI_Comm operations are defined to be callable from any thread
// when the appropriate thread level (Serialized or Multiple) was requested.
// Users must ensure they requested sufficient thread support and serialize
// access themselves when using ThreadLevel::Serialized.
//
// NOTE: The C-layer handle tables (comm_table, request_table, etc.) are not
// internally synchronized. At ThreadLevel::Single and Funneled, only the main
// thread calls MPI so no data race occurs. At Serialized, the user is required
// to serialize all MPI calls. At Multiple, concurrent MPI calls are safe per
// the MPI standard, but our handle table allocation/deallocation is not
// protected by mutexes. This is acceptable because MPI_THREAD_MULTIPLE only
// guarantees that MPI calls are thread-safe — the handle tables are modified
// only inside MPI entry/exit points which the user serializes or the MPI
// implementation serializes internally.
unsafe impl Send for Communicator {}
unsafe impl Sync for Communicator {}

impl Communicator {
    /// Constant for opting out of a communicator split.
    ///
    /// Processes passing this as the `color` to [`split()`](Self::split) will not be
    /// included in any resulting communicator.
    pub const UNDEFINED: i32 = -1;

    /// Get a handle to `MPI_COMM_WORLD`.
    pub(crate) fn world() -> Self {
        Communicator {
            handle: unsafe { ffi::ferrompi_comm_world() },
        }
    }

    /// Get the raw communicator handle (for advanced use).
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }

    /// Get the rank of the calling process in this communicator.
    pub fn rank(&self) -> i32 {
        let mut rank: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_rank(self.handle, &mut rank) };
        debug_assert_eq!(ret, 0, "MPI_Comm_rank failed with code {ret}");
        rank
    }

    /// Get the number of processes in this communicator.
    pub fn size(&self) -> i32 {
        let mut size: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_size(self.handle, &mut size) };
        debug_assert_eq!(ret, 0, "MPI_Comm_size failed with code {ret}");
        size
    }

    /// Get the processor name for this process.
    pub fn processor_name(&self) -> Result<String> {
        let mut buf = [0u8; 256];
        let mut len: i32 = 0;
        let ret = unsafe {
            ffi::ferrompi_get_processor_name(buf.as_mut_ptr().cast::<std::ffi::c_char>(), &mut len)
        };
        Error::check(ret)?;
        let len = (len.max(0) as usize).min(buf.len());
        let s = std::str::from_utf8(&buf[..len])
            .map_err(|_| Error::Internal("Invalid UTF-8 in processor name".into()))?;
        Ok(s.to_string())
    }

    /// Gather topology information from all ranks in this communicator.
    ///
    /// This is a **collective operation** — all ranks must call it.
    /// Every rank receives the complete [`TopologyInfo`], which includes the
    /// rank-to-host mapping, MPI version metadata, and (with the `numa`
    /// feature) SLURM job information.
    ///
    /// [`TopologyInfo`]: crate::TopologyInfo
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let topo = world.topology(&mpi).unwrap();
    /// if world.rank() == 0 {
    ///     println!("{topo}");
    /// }
    /// ```
    pub fn topology(&self, mpi: &crate::Mpi) -> Result<crate::TopologyInfo> {
        crate::topology::gather_topology(self, mpi)
    }

    /// Duplicate this communicator.
    pub fn duplicate(&self) -> Result<Self> {
        let mut new_handle: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_dup(self.handle, &mut new_handle) };
        Error::check(ret)?;
        Ok(Communicator { handle: new_handle })
    }

    /// Split this communicator into sub-communicators based on color and key.
    ///
    /// Processes with the same `color` are placed in the same new communicator.
    /// The `key` controls the rank ordering within the new communicator.
    ///
    /// Returns `None` if this process used [`Communicator::UNDEFINED`] as color.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let color = world.rank() % 2; // Even/odd split
    /// if let Some(sub) = world.split(color, world.rank()).unwrap() {
    ///     println!("Rank {} in sub-communicator of size {}", sub.rank(), sub.size());
    /// }
    /// ```
    pub fn split(&self, color: i32, key: i32) -> Result<Option<Communicator>> {
        let mut new_handle: i32 = 0;
        let ret = unsafe { ffi::ferrompi_comm_split(self.handle, color, key, &mut new_handle) };
        Error::check(ret)?;
        if new_handle < 0 {
            Ok(None)
        } else {
            Ok(Some(Communicator { handle: new_handle }))
        }
    }

    /// Split this communicator by type.
    ///
    /// Processes that share the same resource (determined by `split_type`) are
    /// placed in the same new communicator. The `key` controls the rank ordering
    /// within the new communicator.
    ///
    /// Returns `None` if MPI returns `MPI_COMM_NULL` for this process.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, SplitType};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// if let Some(node) = world.split_type(SplitType::Shared, world.rank()).unwrap() {
    ///     println!("Node has {} processes", node.size());
    /// }
    /// ```
    pub fn split_type(&self, split_type: SplitType, key: i32) -> Result<Option<Communicator>> {
        let mut new_handle: i32 = 0;
        let ret = unsafe {
            ffi::ferrompi_comm_split_type(self.handle, split_type as i32, key, &mut new_handle)
        };
        Error::check(ret)?;
        if new_handle < 0 {
            Ok(None)
        } else {
            Ok(Some(Communicator { handle: new_handle }))
        }
    }

    /// Create a communicator containing only processes that share memory.
    ///
    /// This is equivalent to `split_type(SplitType::Shared, self.rank())`.
    /// All processes on the same physical node will be in the same communicator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`] if MPI unexpectedly returns a null communicator,
    /// which should not happen for `MPI_COMM_TYPE_SHARED` under normal conditions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let node = world.split_shared().unwrap();
    /// println!("Node has {} processes, I am local rank {}", node.size(), node.rank());
    /// ```
    pub fn split_shared(&self) -> Result<Communicator> {
        self.split_type(SplitType::Shared, self.rank())?
            .ok_or_else(|| Error::Internal("split_shared returned null communicator".into()))
    }

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
    // Generic Point-to-Point Communication
    // ========================================================================

    /// Send a slice of values to another process.
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send
    /// * `dest` - Destination rank
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
    pub fn send<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_send(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Receive a slice of values from another process.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// Returns `(actual_source, actual_tag, actual_count)`.
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
        source: i32,
        tag: i32,
    ) -> Result<(i32, i32, i64)> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut actual_count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_recv(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                source,
                tag,
                self.handle,
                &mut actual_source,
                &mut actual_tag,
                &mut actual_count,
            )
        };
        Error::check(ret)?;
        Ok((actual_source, actual_tag, actual_count))
    }

    /// Nonblocking send.
    ///
    /// Initiates a send operation and returns immediately with a [`Request`]
    /// handle. The send buffer **must not be modified** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// # Arguments
    ///
    /// * `data` - Buffer to send (must remain valid until the request completes)
    /// * `dest` - Destination rank
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
    pub fn isend<T: MpiDatatype>(&self, data: &[T], dest: i32, tag: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_isend(
                data.as_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                dest,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking receive.
    ///
    /// Initiates a receive operation and returns immediately with a [`Request`]
    /// handle. The receive buffer **must not be read** until the request is
    /// completed via [`Request::wait()`] or [`Request::test()`].
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// # Arguments
    ///
    /// * `data` - Receive buffer (must remain valid until the request completes)
    /// * `source` - Source rank (or -1 for any source)
    /// * `tag` - Message tag (or -1 for any tag)
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
    pub fn irecv<T: MpiDatatype>(&self, data: &mut [T], source: i32, tag: i32) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_irecv(
                data.as_mut_ptr().cast::<std::ffi::c_void>(),
                data.len() as i64,
                T::TAG as i32,
                source,
                tag,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Blocking send-receive.
    ///
    /// Sends data to one process and receives from another (or the same) in a
    /// single operation. This is useful for avoiding deadlocks in ring-style
    /// communication patterns where each process both sends and receives.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `recvtag = -1` for `MPI_ANY_TAG`.
    ///
    /// Returns `(actual_source, actual_tag, actual_count)`.
    ///
    /// # Arguments
    ///
    /// * `send` - Buffer to send
    /// * `dest` - Destination rank
    /// * `sendtag` - Send message tag
    /// * `recv` - Receive buffer
    /// * `source` - Source rank (or -1 for any source)
    /// * `recvtag` - Receive message tag (or -1 for any tag)
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
        dest: i32,
        sendtag: i32,
        recv: &mut [T],
        source: i32,
        recvtag: i32,
    ) -> Result<(i32, i32, i64)> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut actual_count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_sendrecv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                T::TAG as i32,
                dest,
                sendtag,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                source,
                recvtag,
                self.handle,
                &mut actual_source,
                &mut actual_tag,
                &mut actual_count,
            )
        };
        Error::check(ret)?;
        Ok((actual_source, actual_tag, actual_count))
    }

    // ========================================================================
    // Message Probing
    // ========================================================================

    /// Blocking probe for an incoming message.
    ///
    /// Waits until a matching message is available and returns status
    /// information (source rank, tag, element count) without actually
    /// receiving the message. This is useful for determining the size of an
    /// incoming message before allocating a receive buffer.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// The type parameter `T` determines the MPI datatype used by
    /// `MPI_Get_count` to compute the element count in the returned
    /// [`Status`].
    ///
    /// # Arguments
    ///
    /// * `source` - Source rank to match (or -1 for any source)
    /// * `tag` - Message tag to match (or -1 for any tag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Probe for any incoming f64 message
    /// let status = world.probe::<f64>(-1, -1).unwrap();
    /// // Allocate a buffer of exactly the right size (count may be negative on error)
    /// assert!(status.count >= 0, "MPI_Get_count returned MPI_UNDEFINED");
    /// let mut buf = vec![0.0f64; status.count as usize];
    /// world.recv(&mut buf, status.source, status.tag).unwrap();
    /// ```
    pub fn probe<T: MpiDatatype>(&self, source: i32, tag: i32) -> Result<Status> {
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_probe(
                source,
                tag,
                self.handle,
                &mut actual_source,
                &mut actual_tag,
                &mut count,
                T::TAG as i32,
            )
        };
        Error::check(ret)?;
        Ok(Status {
            source: actual_source,
            tag: actual_tag,
            count,
        })
    }

    /// Nonblocking probe for an incoming message.
    ///
    /// Checks whether a matching message is available without blocking.
    /// Returns `Some(Status)` if a message is available, `None` otherwise.
    ///
    /// Use `source = -1` for `MPI_ANY_SOURCE` and `tag = -1` for `MPI_ANY_TAG`.
    ///
    /// The type parameter `T` determines the MPI datatype used by
    /// `MPI_Get_count` to compute the element count in the returned
    /// [`Status`].
    ///
    /// # Arguments
    ///
    /// * `source` - Source rank to match (or -1 for any source)
    /// * `tag` - Message tag to match (or -1 for any tag)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// // Poll for an incoming f64 message without blocking
    /// if let Some(status) = world.iprobe::<f64>(-1, -1).unwrap() {
    ///     assert!(status.count >= 0, "MPI_Get_count returned MPI_UNDEFINED");
    ///     let mut buf = vec![0.0f64; status.count as usize];
    ///     world.recv(&mut buf, status.source, status.tag).unwrap();
    /// }
    /// ```
    pub fn iprobe<T: MpiDatatype>(&self, source: i32, tag: i32) -> Result<Option<Status>> {
        let mut flag: i32 = 0;
        let mut actual_source: i32 = 0;
        let mut actual_tag: i32 = 0;
        let mut count: i64 = 0;

        let ret = unsafe {
            ffi::ferrompi_iprobe(
                source,
                tag,
                self.handle,
                &mut flag,
                &mut actual_source,
                &mut actual_tag,
                &mut count,
                T::TAG as i32,
            )
        };
        Error::check(ret)?;
        if flag != 0 {
            Ok(Some(Status {
                source: actual_source,
                tag: actual_tag,
                count,
            }))
        } else {
            Ok(None)
        }
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

    // ========================================================================
    // Generic V-Collectives (variable-count)
    // ========================================================================

    /// Gather variable amounts of data to the root process.
    ///
    /// Each process sends `send.len()` elements. At the root, `recvcounts[i]`
    /// elements are placed at offset `displs[i]` in `recv` from rank `i`.
    /// Both `recvcounts` and `displs` must have length equal to the
    /// communicator size and are only significant at root.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root)
    /// * `recvcounts` - Number of elements received from each rank
    /// * `displs` - Displacement in `recv` for data from each rank
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// // Each rank sends (rank+1) elements
    /// let send = vec![rank as f64; (rank + 1) as usize];
    /// let size = world.size();
    /// let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = recvcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = recvcounts.iter().sum();
    /// let mut recv = vec![0.0f64; total as usize];
    /// world.gatherv(&send, &mut recv, &recvcounts, &displs, 0).unwrap();
    /// ```
    pub fn gatherv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
        root: i32,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_gatherv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// Scatter variable amounts of data from the root process.
    ///
    /// At the root, `sendcounts[i]` elements starting at offset `displs[i]`
    /// in `send` are sent to rank `i`. Each process receives `recv.len()`
    /// elements. Both `sendcounts` and `displs` must have length equal to
    /// the communicator size and are only significant at root.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to scatter (only significant at root)
    /// * `sendcounts` - Number of elements sent to each rank
    /// * `displs` - Displacement in `send` for data to each rank
    /// * `recv` - Buffer for received data
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// let size = world.size();
    /// let sendcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = sendcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = sendcounts.iter().sum();
    /// let send = vec![0.0f64; total as usize];
    /// let mut recv = vec![0.0f64; (rank + 1) as usize];
    /// world.scatterv(&send, &sendcounts, &displs, &mut recv, 0).unwrap();
    /// ```
    pub fn scatterv<T: MpiDatatype>(
        &self,
        send: &[T],
        sendcounts: &[i32],
        displs: &[i32],
        recv: &mut [T],
        root: i32,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_scatterv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                displs.as_ptr(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-gather variable amounts of data (gather and broadcast to all).
    ///
    /// Each process sends `send.len()` elements. In `recv`, `recvcounts[i]`
    /// elements from rank `i` are placed at offset `displs[i]`. Both
    /// `recvcounts` and `displs` must have length equal to the communicator
    /// size.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data
    /// * `recvcounts` - Number of elements received from each rank
    /// * `displs` - Displacement in `recv` for data from each rank
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// let size = world.size();
    /// let send = vec![rank as f64; (rank + 1) as usize];
    /// let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = recvcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = recvcounts.iter().sum();
    /// let mut recv = vec![0.0f64; total as usize];
    /// world.allgatherv(&send, &mut recv, &recvcounts, &displs).unwrap();
    /// ```
    pub fn allgatherv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
    ) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_allgatherv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

    /// All-to-all with variable counts.
    ///
    /// Each process sends `sendcounts[i]` elements starting at offset
    /// `sdispls[i]` in `send` to rank `i`, and receives `recvcounts[i]`
    /// elements from rank `i` at offset `rdispls[i]` in `recv`. All four
    /// arrays must have length equal to the communicator size.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `sdispls` - Send displacement for each rank
    /// * `recv` - Receive buffer
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `rdispls` - Receive displacement for each rank
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let sendcounts = vec![1i32; size];
    /// let sdispls: Vec<i32> = (0..size as i32).collect();
    /// let recvcounts = vec![1i32; size];
    /// let rdispls: Vec<i32> = (0..size as i32).collect();
    /// let send = vec![world.rank() as f64; size];
    /// let mut recv = vec![0.0f64; size];
    /// world.alltoallv(&send, &sendcounts, &sdispls, &mut recv, &recvcounts, &rdispls).unwrap();
    /// ```
    pub fn alltoallv<T: MpiDatatype>(
        &self,
        send: &[T],
        sendcounts: &[i32],
        sdispls: &[i32],
        recv: &mut [T],
        recvcounts: &[i32],
        rdispls: &[i32],
    ) -> Result<()> {
        let ret = unsafe {
            ffi::ferrompi_alltoallv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
                T::TAG as i32,
                self.handle,
            )
        };
        Error::check(ret)
    }

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

    /// Nonblocking gather variable amounts of data to root.
    ///
    /// Initiates a variable-count gather and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data (only significant at root)
    /// * `recvcounts` - Number of elements received from each rank
    /// * `displs` - Displacement in `recv` for data from each rank
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// let send = vec![rank as f64; (rank + 1) as usize];
    /// let size = world.size();
    /// let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = recvcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = recvcounts.iter().sum();
    /// let mut recv = vec![0.0f64; total as usize];
    /// let req = world.igatherv(&send, &mut recv, &recvcounts, &displs, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn igatherv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
        root: i32,
    ) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_igatherv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking scatter variable amounts of data from root.
    ///
    /// Initiates a variable-count scatter and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to scatter (only significant at root)
    /// * `recv` - Buffer for received data
    /// * `sendcounts` - Number of elements sent to each rank
    /// * `displs` - Displacement in `send` for data to each rank
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// let size = world.size();
    /// let sendcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = sendcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = sendcounts.iter().sum();
    /// let send = vec![0.0f64; total as usize];
    /// let mut recv = vec![0.0f64; (rank + 1) as usize];
    /// let req = world.iscatterv(&send, &mut recv, &sendcounts, &displs, 0).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iscatterv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        sendcounts: &[i32],
        displs: &[i32],
        root: i32,
    ) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iscatterv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                displs.as_ptr(),
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

    /// Nonblocking all-gather variable amounts of data.
    ///
    /// Initiates a variable-count all-gather and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Arguments
    ///
    /// * `send` - Data to send from this process
    /// * `recv` - Buffer for received data
    /// * `recvcounts` - Number of elements received from each rank
    /// * `displs` - Displacement in `recv` for data from each rank
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank();
    /// let size = world.size();
    /// let send = vec![rank as f64; (rank + 1) as usize];
    /// let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
    /// let displs: Vec<i32> = recvcounts.iter()
    ///     .scan(0, |acc, &c| { let d = *acc; *acc += c; Some(d) })
    ///     .collect();
    /// let total: i32 = recvcounts.iter().sum();
    /// let mut recv = vec![0.0f64; total as usize];
    /// let req = world.iallgatherv(&send, &mut recv, &recvcounts, &displs).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn iallgatherv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
    ) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_iallgatherv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(Request::new(request_handle))
    }

    /// Nonblocking all-to-all with variable counts.
    ///
    /// Initiates a variable-count all-to-all and returns immediately with a
    /// [`Request`] handle.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `sdispls` - Send displacement for each rank
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `rdispls` - Receive displacement for each rank
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let sendcounts = vec![1i32; size];
    /// let sdispls: Vec<i32> = (0..size as i32).collect();
    /// let recvcounts = vec![1i32; size];
    /// let rdispls: Vec<i32> = (0..size as i32).collect();
    /// let send = vec![world.rank() as f64; size];
    /// let mut recv = vec![0.0f64; size];
    /// let req = world.ialltoallv(&send, &mut recv, &sendcounts, &sdispls, &recvcounts, &rdispls).unwrap();
    /// req.wait().unwrap();
    /// ```
    pub fn ialltoallv<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        sendcounts: &[i32],
        sdispls: &[i32],
        recvcounts: &[i32],
        rdispls: &[i32],
    ) -> Result<Request> {
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_ialltoallv(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
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
        Error::check(ret)?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent gatherv operation (variable-count gather).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer (significant only at root)
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `displs` - Displacement for each rank in the receive buffer
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 40];
    /// let recvcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut persistent = world.gatherv_init(&send, &mut recv, &recvcounts, &displs, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn gatherv_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
        root: i32,
    ) -> Result<PersistentRequest> {
        if recvcounts.len() != displs.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_gatherv_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent scatterv operation (variable-count scatter).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (significant only at root)
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `displs` - Displacement for each rank in the send buffer
    /// * `recv` - Receive buffer
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 40];
    /// let sendcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut recv = vec![0.0f64; 10];
    /// let mut persistent = world.scatterv_init(&send, &sendcounts, &displs, &mut recv, 0).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn scatterv_init<T: MpiDatatype>(
        &self,
        send: &[T],
        sendcounts: &[i32],
        displs: &[i32],
        recv: &mut [T],
        root: i32,
    ) -> Result<PersistentRequest> {
        if sendcounts.len() != displs.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_scatterv_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                displs.as_ptr(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recv.len() as i64,
                T::TAG as i32,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-gatherv operation (variable-count all-gather).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `recv` - Receive buffer
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `displs` - Displacement for each rank in the receive buffer
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 40];
    /// let recvcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut persistent = world.allgatherv_init(&send, &mut recv, &recvcounts, &displs).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn allgatherv_init<T: MpiDatatype>(
        &self,
        send: &[T],
        recv: &mut [T],
        recvcounts: &[i32],
        displs: &[i32],
    ) -> Result<PersistentRequest> {
        if recvcounts.len() != displs.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_allgatherv_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                send.len() as i64,
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                displs.as_ptr(),
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
        Ok(PersistentRequest::new(request_handle))
    }

    /// Initialize a persistent all-to-allv operation (variable-count all-to-all).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0+.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `sdispls` - Send displacement for each rank
    /// * `recv` - Receive buffer
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `rdispls` - Receive displacement for each rank
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 40];
    /// let sendcounts = vec![10i32; 4];
    /// let sdispls = vec![0i32, 10, 20, 30];
    /// let mut recv = vec![0.0f64; 40];
    /// let recvcounts = vec![10i32; 4];
    /// let rdispls = vec![0i32, 10, 20, 30];
    /// let mut persistent = world.alltoallv_init(
    ///     &send, &sendcounts, &sdispls,
    ///     &mut recv, &recvcounts, &rdispls,
    /// ).unwrap();
    /// for _ in 0..100 {
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    /// }
    /// ```
    pub fn alltoallv_init<T: MpiDatatype>(
        &self,
        send: &[T],
        sendcounts: &[i32],
        sdispls: &[i32],
        recv: &mut [T],
        recvcounts: &[i32],
        rdispls: &[i32],
    ) -> Result<PersistentRequest> {
        if sendcounts.len() != sdispls.len() || recvcounts.len() != rdispls.len() {
            return Err(Error::InvalidBuffer);
        }
        let mut request_handle: i64 = 0;
        let ret = unsafe {
            ffi::ferrompi_alltoallv_init(
                send.as_ptr().cast::<std::ffi::c_void>(),
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                recv.as_mut_ptr().cast::<std::ffi::c_void>(),
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
                T::TAG as i32,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check(ret)?;
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
        Error::check(ret)?;
        Ok(PersistentRequest::new(request_handle))
    }

    // ========================================================================
    // Process Control
    // ========================================================================

    /// Abort MPI execution across all processes in this communicator.
    ///
    /// This function terminates all processes associated with the communicator.
    /// It calls `MPI_Abort` and then exits the process. This function never
    /// returns.
    ///
    /// # Arguments
    ///
    /// * `errorcode` - Error code to return to the invoking environment
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// if world.rank() == 0 {
    ///     // Fatal error detected, abort all processes
    ///     world.abort(1);
    /// }
    /// ```
    pub fn abort(&self, errorcode: i32) -> ! {
        unsafe { ffi::ferrompi_abort(self.handle, errorcode) };
        std::process::exit(errorcode)
    }
}

impl Drop for Communicator {
    fn drop(&mut self) {
        // Don't free COMM_WORLD (handle 0)
        if self.handle != 0 {
            unsafe { ffi::ferrompi_comm_free(self.handle) };
        }
    }
}

// Compile-time assertions: Communicator must be Send + Sync
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<Communicator>();
    }
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a Communicator with handle 0 (COMM_WORLD).
    /// Drop for handle 0 is a no-op, so this is safe without MPI.
    fn dummy_comm() -> Communicator {
        Communicator { handle: 0 }
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
    fn split_type_shared_repr_value_and_traits() {
        // SplitType::Shared has repr value 0
        assert_eq!(SplitType::Shared as i32, 0);

        // Clone works (Copy implies Clone)
        let st = SplitType::Shared;
        let cloned = st;
        assert_eq!(cloned, SplitType::Shared);

        // Debug works
        assert_eq!(format!("{:?}", st), "Shared");
    }

    #[test]
    fn communicator_undefined_is_negative_one() {
        assert_eq!(Communicator::UNDEFINED, -1);
    }

    #[test]
    fn communicator_raw_handle_returns_correct_value() {
        let comm = dummy_comm();
        assert_eq!(comm.raw_handle(), 0);
    }

    // Nonblocking collective buffer validation tests (ticket-005)

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

    // Persistent collective buffer validation tests (ticket-006)

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

    // V-collective init array length validation tests (ticket-006)

    #[test]
    fn gatherv_init_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.gatherv_init(&send, &mut recv, &recvcounts, &displs, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scatterv_init_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 10];
        let result = comm.scatterv_init(&send, &sendcounts, &displs, &mut recv, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allgatherv_init_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.allgatherv_init(&send, &mut recv, &recvcounts, &displs);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoallv_init_mismatched_send_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20, 30];
        let result = comm.alltoallv_init(
            &send,
            &sendcounts,
            &sdispls,
            &mut recv,
            &recvcounts,
            &rdispls,
        );
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoallv_init_mismatched_recv_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20, 30];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.alltoallv_init(
            &send,
            &sendcounts,
            &sdispls,
            &mut recv,
            &recvcounts,
            &rdispls,
        );
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    // ========================================================================
    // InvalidBuffer validation coverage audit
    // ========================================================================
    //
    // This documents every `return Err(Error::InvalidBuffer)` site in comm.rs
    // and whether it is covered by unit tests (this module) or requires MPI
    // integration tests (examples/test_collectives.rs).
    //
    // COVERED BY UNIT TESTS (18 paths — no MPI runtime needed):
    //
    //   Line  Function             Condition                              Test
    //   ----  -------------------  -------------------------------------  --------------------------------
    //   632   reduce               send.len() != recv.len()               reduce_mismatched_buffers_*
    //   750   allreduce            send.len() != recv.len()               allreduce_mismatched_buffers_*
    //   839   scan                 send.len() != recv.len()               scan_mismatched_buffers_*
    //   889   exscan               send.len() != recv.len()               exscan_mismatched_buffers_*
    //   1420  iallreduce           send.len() != recv.len()               iallreduce_mismatched_buffers_*
    //   1469  ireduce              send.len() != recv.len()               ireduce_mismatched_buffers_*
    //   1658  iscan                send.len() != recv.len()               iscan_mismatched_buffers_*
    //   1708  iexscan              send.len() != recv.len()               iexscan_mismatched_buffers_*
    //   2120  allreduce_init       send.len() != recv.len()               allreduce_init_mismatched_*
    //   2233  reduce_init          send.len() != recv.len()               reduce_init_mismatched_*
    //   2389  scan_init            send.len() != recv.len()               scan_init_mismatched_*
    //   2443  exscan_init          send.len() != recv.len()               exscan_init_mismatched_*
    //   2496  alltoall_init        send.len() != recv.len() (1st check)   alltoall_init_mismatched_*
    //   2561  gatherv_init         recvcounts.len() != displs.len()       gatherv_init_mismatched_*
    //   2623  scatterv_init        sendcounts.len() != displs.len()       scatterv_init_mismatched_*
    //   2683  allgatherv_init      recvcounts.len() != displs.len()       allgatherv_init_mismatched_*
    //   2751  alltoallv_init       sendcounts != sdispls (send side)      alltoallv_init_mismatched_send_*
    //   2751  alltoallv_init       recvcounts != rdispls (recv side)      alltoallv_init_mismatched_recv_*
    //
    // REQUIRE MPI RUNTIME — covered by integration tests (5 paths):
    //
    // These paths call `self.size()` (FFI into MPI_Comm_size) BEFORE the
    // validation check, so they cannot be tested without an initialized MPI
    // runtime. They would segfault with the dummy_comm(handle=0) helper.
    //
    //   Line  Function                    Condition
    //   ----  --------------------------  -------------------------------------------
    //   1073  alltoall                    send.len() != recv.len() || len % size != 0
    //   1122  reduce_scatter_block        send.len() != recv.len() * size
    //   1751  ialltoall                   send.len() != recv.len() || len % size != 0
    //   2026  ireduce_scatter_block       send.len() != recv.len() * size
    //   2500  alltoall_init (2nd check)   size == 0 || send.len() % size != 0
    //   2809  reduce_scatter_block_init   size == 0 || send.len() != recv.len() * size
    //
    // The MPI-dependent paths are exercised by:
    //   - examples/test_collectives.rs (alltoall, reduce_scatter_block)
    //   - The successful-path integration tests validate that correct buffers
    //     pass the checks; incorrect buffers for these functions must be
    //     tested under `mpirun` where self.size() returns a valid value.
    //
    // Total: 23 InvalidBuffer return sites, 18 covered by unit tests,
    //        5 require MPI integration tests (6 logical paths across
    //        5 return statements, since alltoallv_init line 2751 has
    //        two conditions tested separately).
    // ========================================================================
}
