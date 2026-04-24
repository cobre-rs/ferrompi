//! Point-to-point communication: send, recv, isend, irecv, sendrecv, probe, iprobe.

use crate::comm::Communicator;
use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::request::Request;
use crate::status::Status;

impl Communicator {
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
        Error::check_with_op(ret, "send")
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
        Error::check_with_op(ret, "recv")?;
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
        Error::check_with_op(ret, "isend")?;
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
        Error::check_with_op(ret, "irecv")?;
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
        Error::check_with_op(ret, "sendrecv")?;
        Ok((actual_source, actual_tag, actual_count))
    }

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
        Error::check_with_op(ret, "probe")?;
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
        Error::check_with_op(ret, "iprobe")?;
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
}
