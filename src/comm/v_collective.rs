//! Variable-count collective operations: gatherv, scatterv, allgatherv, alltoallv
//! and their nonblocking (i*) and persistent (*_init) variants.

use crate::comm::Communicator;
use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::Request;

/// Checks one counts/displacements pair that MPI reads on this rank: each array
/// holds exactly `size` entries, every count is non-negative, and every block
/// with a positive count lies inside a `buf_len`-element buffer (computed in i64).
fn check_v_args(counts: &[i32], displs: &[i32], size: i32, buf_len: usize) -> Result<()> {
    if counts.len() != size as usize || displs.len() != size as usize {
        return Err(Error::InvalidBuffer);
    }
    let buf_len = buf_len as i64;
    for (&count, &displ) in counts.iter().zip(displs) {
        if count < 0 {
            return Err(Error::InvalidBuffer);
        }
        if count > 0 && (displ < 0 || i64::from(displ) + i64::from(count) > buf_len) {
            return Err(Error::InvalidBuffer);
        }
    }
    Ok(())
}

impl Communicator {
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(recvcounts, displs, self.size, recv.len())?;
        }
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). At root, recvcounts and
        // displs are checked above to have size() entries, non-negative counts, and each
        // positive-count block inside recv; at non-root MPI does not read recvcounts or
        // displs, so they are unchecked here.
        let ret = unsafe {
            ffi::ferrompi_gatherv(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                root,
                self.handle,
            )
        };
        Error::check_with_op(ret, "gatherv")
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `send`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(sendcounts, displs, self.size, send.len())?;
        }
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). At root, sendcounts and displs are checked above to have size()
        // entries, non-negative counts, and each positive-count block inside send; at
        // non-root MPI does not read sendcounts or displs, so they are unchecked here.
        let ret = unsafe {
            ffi::ferrompi_scatterv(
                sp,
                sendcounts.as_ptr(),
                displs.as_ptr(),
                rp,
                n,
                dt,
                root,
                self.handle,
            )
        };
        Error::check_with_op(ret, "scatterv")
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(recvcounts, displs, self.size, recv.len())?;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). recvcounts and displs are
        // checked above, on every rank, to have size() entries, non-negative counts, and
        // each positive-count block inside recv.
        let ret = unsafe {
            ffi::ferrompi_allgatherv(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                self.handle,
            )
        };
        Error::check_with_op(ret, "allgatherv")
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts`/`sdispls`
    ///   or `recvcounts`/`rdispls` does not have `size()` entries, a count is negative, or a
    ///   block with a positive count starts at a negative displacement or ends past the end
    ///   of `send`/`recv` respectively.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(sendcounts, sdispls, self.size, send.len())?;
        check_v_args(recvcounts, rdispls, self.size, recv.len())?;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). sendcounts/sdispls and
        // recvcounts/rdispls are each checked above, on every rank, to have size() entries,
        // non-negative counts, and each positive-count block inside send/recv respectively.
        let ret = unsafe {
            ffi::ferrompi_alltoallv(
                sp,
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                rp,
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
                dt,
                self.handle,
            )
        };
        Error::check_with_op(ret, "alltoallv")
    }

    // ========================================================================
    // Nonblocking V-Collectives
    // ========================================================================

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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(recvcounts, displs, self.size, recv.len())?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). At root, recvcounts and
        // displs are checked above to have size() entries, non-negative counts, and each
        // positive-count block inside recv; at non-root MPI does not read recvcounts or
        // displs, so they are unchecked here. The returned Request does not borrow send,
        // recv, recvcounts or displs; keeping all four alive and untouched until the
        // request completes is the caller's documented obligation, which this signature
        // does not enforce.
        let ret = unsafe {
            ffi::ferrompi_igatherv(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "igatherv")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `send`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(sendcounts, displs, self.size, send.len())?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). At root, sendcounts and displs are checked above to have size()
        // entries, non-negative counts, and each positive-count block inside send; at
        // non-root MPI does not read sendcounts or displs, so they are unchecked here. The
        // returned Request does not borrow send, recv, sendcounts or displs; keeping all
        // four alive and untouched until the request completes is the caller's documented
        // obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iscatterv(
                sp,
                sendcounts.as_ptr(),
                displs.as_ptr(),
                rp,
                n,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iscatterv")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(recvcounts, displs, self.size, recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). recvcounts and displs are
        // checked above, on every rank, to have size() entries, non-negative counts, and
        // each positive-count block inside recv. The returned Request does not borrow
        // send, recv, recvcounts or displs; keeping all four alive and untouched until the
        // request completes is the caller's documented obligation, which this signature
        // does not enforce.
        let ret = unsafe {
            ffi::ferrompi_iallgatherv(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iallgatherv")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts`/`sdispls`
    ///   or `recvcounts`/`rdispls` does not have `size()` entries, a count is negative, or a
    ///   block with a positive count starts at a negative displacement or ends past the end
    ///   of `send`/`recv` respectively.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(sendcounts, sdispls, self.size, send.len())?;
        check_v_args(recvcounts, rdispls, self.size, recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). sendcounts/sdispls and
        // recvcounts/rdispls are each checked above, on every rank, to have size() entries,
        // non-negative counts, and each positive-count block inside send/recv respectively.
        // The returned Request does not borrow send, recv, or any of the four
        // count/displacement arrays; keeping all of them alive and untouched until the
        // request completes is the caller's documented obligation, which this signature
        // does not enforce.
        let ret = unsafe {
            ffi::ferrompi_ialltoallv(
                sp,
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                rp,
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "ialltoallv")?;
        Ok(Request::new(request_handle))
    }

    // ========================================================================
    // Persistent V-Collectives (MPI 4.0+)
    // ========================================================================

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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(recvcounts, displs, self.size, recv.len())?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). At root, recvcounts and
        // displs are checked above to have size() entries, non-negative counts, and each
        // positive-count block inside recv; at non-root MPI does not read recvcounts or
        // displs, so they are unchecked here. The returned PersistentRequest does not
        // borrow send, recv, recvcounts or displs; keeping all four alive and untouched
        // until the request is freed is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_gatherv_init(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "gatherv_init")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `send`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        if self.rank == root {
            check_v_args(sendcounts, displs, self.size, send.len())?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). At root, sendcounts and displs are checked above to have size()
        // entries, non-negative counts, and each positive-count block inside send; at
        // non-root MPI does not read sendcounts or displs, so they are unchecked here. The
        // returned PersistentRequest does not borrow send, recv, sendcounts or displs;
        // keeping all four alive and untouched until the request is freed is the caller's
        // documented obligation, which this signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_scatterv_init(
                sp,
                sendcounts.as_ptr(),
                displs.as_ptr(),
                rp,
                n,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scatterv_init")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `recvcounts` or
    ///   `displs` does not have `size()` entries, a count is negative, or a block with a
    ///   positive count starts at a negative displacement or ends past the end of `recv`.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(recvcounts, displs, self.size, recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). recvcounts and displs are
        // checked above, on every rank, to have size() entries, non-negative counts, and
        // each positive-count block inside recv. The returned PersistentRequest does not
        // borrow send, recv, recvcounts or displs; keeping all four alive and untouched
        // until the request is freed is the caller's documented obligation, which this
        // signature does not enforce.
        let ret = unsafe {
            ffi::ferrompi_allgatherv_init(
                sp,
                n,
                rp,
                recvcounts.as_ptr(),
                displs.as_ptr(),
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "allgatherv_init")?;
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
    /// # Errors
    ///
    /// - [`Error::InvalidBuffer`] if, on a rank where MPI reads them, `sendcounts`/`sdispls`
    ///   or `recvcounts`/`rdispls` does not have `size()` entries, a count is negative, or a
    ///   block with a positive count starts at a negative displacement or ends past the end
    ///   of `send`/`recv` respectively.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
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
        check_v_args(sendcounts, sdispls, self.size, send.len())?;
        check_v_args(recvcounts, rdispls, self.size, recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). sendcounts/sdispls and
        // recvcounts/rdispls are each checked above, on every rank, to have size() entries,
        // non-negative counts, and each positive-count block inside send/recv respectively.
        // The returned PersistentRequest does not borrow send, recv, or any of the four
        // count/displacement arrays; keeping all of them alive and untouched until the
        // request is freed is the caller's documented obligation, which this signature
        // does not enforce.
        let ret = unsafe {
            ffi::ferrompi_alltoallv_init(
                sp,
                sendcounts.as_ptr(),
                sdispls.as_ptr(),
                rp,
                recvcounts.as_ptr(),
                rdispls.as_ptr(),
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoallv_init")?;
        Ok(PersistentRequest::new(request_handle))
    }
}

#[cfg(test)]
mod tests {
    use super::check_v_args;
    use crate::comm::Communicator;
    use crate::error::Error;

    fn dummy_comm() -> Communicator {
        Communicator {
            handle: 0,
            rank: 0,
            size: 1,
        }
    }

    #[test]
    fn check_v_args_boundaries() {
        // exact fit
        assert!(check_v_args(&[2, 2], &[0, 2], 2, 4).is_ok());
        // one element past the end
        assert!(matches!(
            check_v_args(&[2, 2], &[0, 2], 2, 3),
            Err(Error::InvalidBuffer)
        ));
        // counts.len() != size
        assert!(matches!(
            check_v_args(&[2, 2, 2], &[0, 2, 4], 2, 6),
            Err(Error::InvalidBuffer)
        ));
        // displs.len() != size
        assert!(matches!(
            check_v_args(&[2, 2], &[0, 2, 4], 2, 6),
            Err(Error::InvalidBuffer)
        ));
        // negative count
        assert!(matches!(
            check_v_args(&[-1, 2], &[0, 2], 2, 4),
            Err(Error::InvalidBuffer)
        ));
        // negative displacement with count > 0
        assert!(matches!(
            check_v_args(&[2, 2], &[-1, 2], 2, 4),
            Err(Error::InvalidBuffer)
        ));
        // negative displacement with count 0
        assert!(check_v_args(&[0, 2], &[-1, 0], 2, 2).is_ok());
        // i32 arithmetic would wrap: i32::MAX + 1 overflows i32 but not i64
        assert!(matches!(
            check_v_args(&[1], &[i32::MAX], 1, i32::MAX as usize),
            Err(Error::InvalidBuffer)
        ));
    }

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

    // ── blocking / nonblocking v-collective length-mismatch guards ────────

    #[test]
    fn gatherv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.gatherv(&send, &mut recv, &recvcounts, &displs, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn scatterv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 10];
        let result = comm.scatterv(&send, &sendcounts, &displs, &mut recv, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn allgatherv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.allgatherv(&send, &mut recv, &recvcounts, &displs);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn alltoallv_mismatched_send_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20, 30];
        let result = comm.alltoallv(
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
    fn alltoallv_mismatched_recv_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20, 30];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.alltoallv(
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
    fn igatherv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.igatherv(&send, &mut recv, &recvcounts, &displs, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn iscatterv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 10];
        let result = comm.iscatterv(&send, &mut recv, &sendcounts, &displs, 0);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn iallgatherv_mismatched_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.iallgatherv(&send, &mut recv, &recvcounts, &displs);
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn ialltoallv_mismatched_send_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20, 30];
        let result = comm.ialltoallv(
            &send,
            &mut recv,
            &sendcounts,
            &sdispls,
            &recvcounts,
            &rdispls,
        );
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }

    #[test]
    fn ialltoallv_mismatched_recv_counts_displs_returns_invalid_buffer() {
        let comm = dummy_comm();
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20, 30];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.ialltoallv(
            &send,
            &mut recv,
            &sendcounts,
            &sdispls,
            &recvcounts,
            &rdispls,
        );
        assert!(matches!(result, Err(Error::InvalidBuffer)));
    }
}
