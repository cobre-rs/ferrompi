//! Variable-count collective operations: gatherv, scatterv, allgatherv, alltoallv
//! and their nonblocking (i*) and persistent (*_init) variants.

use crate::comm::Communicator;
use crate::datatype::{buf, buf_mut, recv_ptr_of_empty_pair, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::{vec_buf, vec_buf_mut, PersistentRequest};
use crate::request::{Request, RequestKind};
use crate::scope::Scope;

#[derive(Clone, Copy)]
enum VFault {
    CountsLen,
    DisplsLen,
    NegativeCount,
    NegativeDispl,
    PastEnd(i64),
}

#[inline]
fn v_fault(counts: &[i32], displs: &[i32], size: usize, buf_len: usize) -> Option<VFault> {
    if counts.len() != size {
        return Some(VFault::CountsLen);
    }
    if displs.len() != size {
        return Some(VFault::DisplsLen);
    }
    for (&count, &displ) in counts.iter().zip(displs) {
        if count < 0 {
            return Some(VFault::NegativeCount);
        }
        if count > 0 {
            if displ < 0 {
                return Some(VFault::NegativeDispl);
            }
            let end = i64::from(displ) + i64::from(count);
            if end > buf_len as i64 {
                return Some(VFault::PastEnd(end));
            }
        }
    }
    None
}

#[cold]
#[inline(never)]
fn v_error(
    (counts_arg, displs_arg, buf_arg): (&'static str, &'static str, &'static str),
    fault: VFault,
    size: usize,
    counts_len: usize,
    displs_len: usize,
    buf_len: usize,
) -> Error {
    match fault {
        VFault::CountsLen => Error::BufferSize {
            arg: counts_arg,
            required: size,
            actual: counts_len,
        },
        VFault::DisplsLen => Error::BufferSize {
            arg: displs_arg,
            required: size,
            actual: displs_len,
        },
        VFault::NegativeCount => Error::InvalidArgument {
            arg: counts_arg,
            reason: "negative count",
        },
        VFault::NegativeDispl => Error::InvalidArgument {
            arg: displs_arg,
            reason: "negative displacement",
        },
        VFault::PastEnd(end) => Error::BufferSize {
            arg: buf_arg,
            required: end as usize,
            actual: buf_len,
        },
    }
}

/// Checks one counts/displacements pair that MPI reads on this rank, naming the
/// offending parameter in the error: each array holds exactly `size` entries
/// (`BufferSize`), every count is non-negative and every positive-count block
/// starts at a non-negative displacement (`InvalidArgument`), and every
/// positive-count block lies inside a `buf_len`-element buffer (`BufferSize`,
/// computed in i64). The first fault in that order is reported.
fn check_v_args(
    counts_arg: &'static str,
    displs_arg: &'static str,
    buf_arg: &'static str,
    counts: &[i32],
    displs: &[i32],
    size: i32,
    buf_len: usize,
) -> Result<()> {
    let size = size as usize;
    match v_fault(counts, displs, size, buf_len) {
        None => Ok(()),
        Some(fault) => Err(v_error(
            (counts_arg, displs_arg, buf_arg),
            fault,
            size,
            counts.len(),
            displs.len(),
            buf_len,
        )),
    }
}

/// Copies `arrays` end to end into one `Vec<i32>` and returns it with a pointer
/// to the start of each array inside it. A persistent request keeps these
/// pointers until it is freed (MPI-4.1 §6.13, p.274: the arrays must stay
/// unmodified until then), so the pointers come from the `Vec` itself and the
/// request owns it.
fn copy_arrays<const N: usize>(arrays: [&[i32]; N]) -> (Vec<i32>, [*const i32; N]) {
    let copy = arrays.concat();
    let base = copy.as_ptr();
    let mut offset = 0;
    let starts = arrays.map(|a| {
        let start = base.wrapping_add(offset);
        offset += a.len();
        start
    });
    (copy, starts)
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
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
            check_v_args(
                "recvcounts",
                "displs",
                "recv",
                recvcounts,
                displs,
                self.size,
                recv.len(),
            )?;
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
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
            check_v_args(
                "sendcounts",
                "displs",
                "send",
                sendcounts,
                displs,
                self.size,
                send.len(),
            )?;
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
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
        check_v_args(
            "recvcounts",
            "displs",
            "recv",
            recvcounts,
            displs,
            self.size,
            recv.len(),
        )?;
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
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts`/`sdispls` or `recvcounts`/`rdispls`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`/`recv` respectively.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
        check_v_args(
            "sendcounts",
            "sdispls",
            "send",
            sendcounts,
            sdispls,
            self.size,
            send.len(),
        )?;
        check_v_args(
            "recvcounts",
            "rdispls",
            "recv",
            recvcounts,
            rdispls,
            self.size,
            recv.len(),
        )?;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        let rp = recv_ptr_of_empty_pair(send.len(), recv.len(), rp);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). sendcounts/sdispls and
        // recvcounts/rdispls are each checked above, on every rank, to have size() entries,
        // non-negative counts, and each positive-count block inside send/recv respectively.
        // When both buffers are empty every count is zero, and `rp` is then a fixed address
        // of its own, through which MPI writes nothing.
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
    /// [`Request`] that belongs to the scope `s`. `send`, `recvcounts` and
    /// `displs` are borrowed for the scope, and `recv` is borrowed mutably for
    /// it: `recv` cannot be read until [`scope`](crate::scope) returns, even
    /// after the request was waited, and the scope completes the gather at the
    /// latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for received data (only significant at root), borrowed for
    ///   the scope on every rank
    /// * `recvcounts` - Number of elements received from each rank, borrowed for the
    ///   scope on every rank
    /// * `displs` - Displacement in `recv` for data from each rank, borrowed for the
    ///   scope on every rank
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
    /// ferrompi::scope(|s| {
    ///     let req = world.igatherv(s, &send, &mut recv, &recvcounts, &displs, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn igatherv<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        recvcounts: &'s [i32],
        displs: &'s [i32],
        root: i32,
    ) -> Result<Request<'s>> {
        if self.rank == root {
            check_v_args(
                "recvcounts",
                "displs",
                "recv",
                recvcounts,
                displs,
                self.size,
                recv.len(),
            )?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). At root, recvcounts and
        // displs are checked above to have size() entries, non-negative counts, and each
        // positive-count block inside recv; at non-root MPI does not read recvcounts or
        // displs, so they are unchecked here. send, recv, recvcounts and displs are all
        // borrowed for 's (recv mutably), and MPI may read the count and displacement arrays
        // until the request completes. The scope completes every request it holds before 's
        // ends, so the buffers and the arrays outlive the span in which MPI may use them, and
        // nothing touches them meanwhile.
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
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking scatter variable amounts of data from root.
    ///
    /// Initiates a variable-count scatter and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. `send`, `sendcounts` and
    /// `displs` are borrowed for the scope, and `recv` is borrowed mutably for
    /// it: `recv` cannot be read until [`scope`](crate::scope) returns, even
    /// after the request was waited, and the scope completes the scatter at the
    /// latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to scatter (only significant at root), borrowed for the scope
    ///   on every rank
    /// * `sendcounts` - Number of elements sent to each rank, borrowed for the scope
    ///   on every rank
    /// * `displs` - Displacement in `send` for data to each rank, borrowed for the
    ///   scope on every rank
    /// * `recv` - Buffer for received data, borrowed for the scope
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
    /// ferrompi::scope(|s| {
    ///     let req = world.iscatterv(s, &send, &sendcounts, &displs, &mut recv, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn iscatterv<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        sendcounts: &'s [i32],
        displs: &'s [i32],
        recv: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        if self.rank == root {
            check_v_args(
                "sendcounts",
                "displs",
                "send",
                sendcounts,
                displs,
                self.size,
                send.len(),
            )?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). At root, sendcounts and displs are checked above to have size()
        // entries, non-negative counts, and each positive-count block inside send; at
        // non-root MPI does not read sendcounts or displs, so they are unchecked here. send,
        // recv, sendcounts and displs are all borrowed for 's (recv mutably), and MPI may read
        // the count and displacement arrays until the request completes. The scope completes
        // every request it holds before 's ends, so the buffers and the arrays outlive the
        // span in which MPI may use them, and nothing touches them meanwhile.
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
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-gather variable amounts of data.
    ///
    /// Initiates a variable-count all-gather and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. `send`, `recvcounts` and
    /// `displs` are borrowed for the scope, and `recv` is borrowed mutably for
    /// it: `recv` cannot be read until [`scope`](crate::scope) returns, even
    /// after the request was waited, and the scope completes the all-gather at
    /// the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for received data, borrowed for the scope
    /// * `recvcounts` - Number of elements received from each rank, borrowed for the
    ///   scope
    /// * `displs` - Displacement in `recv` for data from each rank, borrowed for the
    ///   scope
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
    /// ferrompi::scope(|s| {
    ///     let req = world.iallgatherv(s, &send, &mut recv, &recvcounts, &displs)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn iallgatherv<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        recvcounts: &'s [i32],
        displs: &'s [i32],
    ) -> Result<Request<'s>> {
        check_v_args(
            "recvcounts",
            "displs",
            "recv",
            recvcounts,
            displs,
            self.size,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). recvcounts and displs are
        // checked above, on every rank, to have size() entries, non-negative counts, and
        // each positive-count block inside recv. send, recv, recvcounts and displs are all
        // borrowed for 's (recv mutably), and MPI may read the count and displacement arrays
        // until the request completes. The scope completes every request it holds before 's
        // ends, so the buffers and the arrays outlive the span in which MPI may use them, and
        // nothing touches them meanwhile.
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
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-to-all with variable counts.
    ///
    /// Initiates a variable-count all-to-all and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. `send` and the four count and
    /// displacement arrays are borrowed for the scope, and `recv` is borrowed
    /// mutably for it: `recv` cannot be read until [`scope`](crate::scope)
    /// returns, even after the request was waited, and the scope completes the
    /// all-to-all at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Send buffer, borrowed for the scope
    /// * `sendcounts` - Number of elements to send to each rank, borrowed for the scope
    /// * `sdispls` - Send displacement for each rank, borrowed for the scope
    /// * `recv` - Receive buffer, borrowed for the scope
    /// * `recvcounts` - Number of elements to receive from each rank, borrowed for the
    ///   scope
    /// * `rdispls` - Receive displacement for each rank, borrowed for the scope
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts`/`sdispls` or `recvcounts`/`rdispls`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`/`recv` respectively.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
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
    /// ferrompi::scope(|s| {
    ///     let req = world.ialltoallv(s, &send, &sendcounts, &sdispls, &mut recv, &recvcounts, &rdispls)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    #[allow(clippy::too_many_arguments)] // the scope plus MPI_Ialltoallv's two buffers and four arrays
    pub fn ialltoallv<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        sendcounts: &'s [i32],
        sdispls: &'s [i32],
        recv: &'s mut [T],
        recvcounts: &'s [i32],
        rdispls: &'s [i32],
    ) -> Result<Request<'s>> {
        check_v_args(
            "sendcounts",
            "sdispls",
            "send",
            sendcounts,
            sdispls,
            self.size,
            send.len(),
        )?;
        check_v_args(
            "recvcounts",
            "rdispls",
            "recv",
            recvcounts,
            rdispls,
            self.size,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        let rp = recv_ptr_of_empty_pair(send.len(), recv.len(), rp);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). sendcounts/sdispls and
        // recvcounts/rdispls are each checked above, on every rank, to have size() entries,
        // non-negative counts, and each positive-count block inside send/recv respectively.
        // When both buffers are empty every count is zero, and `rp` is then a fixed address
        // of its own, through which MPI writes nothing.
        // send, recv and all four count/displacement arrays are borrowed for 's (recv
        // mutably), and MPI may read the arrays until the request completes. The scope
        // completes every request it holds before 's ends, so the buffers and the arrays
        // outlive the span in which MPI may use them, and nothing touches them meanwhile.
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
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    // ========================================================================
    // Persistent V-Collectives (MPI 4.0, or Open MPI 5)
    // ========================================================================

    /// Initialize a persistent gatherv operation (variable-count gather).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer, owned by the request; write it through `send_buffer_mut`
    ///   while the request is inactive
    /// * `recv` - Receive buffer (significant only at root), owned by the request; read it
    ///   through `buffer` while the request is inactive
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `displs` - Displacement for each rank in the receive buffer
    /// * `root` - Rank of the root process
    ///
    /// The request copies `recvcounts` and `displs`, which MPI reads until the request is
    /// freed; the caller may reuse its arrays as soon as the call returns.
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let recvcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut persistent = world
    ///     .gatherv_init(vec![1.0f64; 10], vec![0.0f64; 40], &recvcounts, &displs, 0)
    ///     .unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     if world.rank() == 0 {
    ///         assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    ///     }
    /// }
    /// ```
    pub fn gatherv_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
        recvcounts: &[i32],
        displs: &[i32],
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        if self.rank == root {
            check_v_args(
                "recvcounts",
                "displs",
                "recv",
                recvcounts,
                displs,
                self.size,
                recv.len(),
            )?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        let (counts, [cp, dp]) = copy_arrays([recvcounts, displs]);
        // SAFETY: `sp`, `n` and `rp` describe the heap allocations of `send` and `recv`,
        // taken from the Vecs by `vec_buf`/`vec_buf_mut`. The two Vecs are distinct
        // allocations. At root, recvcounts and displs are checked above to have size()
        // entries, non-negative counts, and each positive-count block inside recv; at
        // non-root MPI does not read recvcounts or displs, so they are unchecked here. `cp`
        // and `dp` point into `counts`, the request's own copy of both arrays, taken after
        // the check and from the Vec itself. The request built below owns `send`, `recv`
        // and `counts` under the ownership invariant documented on
        // `PersistentRequest::new_pair`.
        let ret = unsafe {
            ffi::ferrompi_gatherv_init(
                sp,
                n,
                rp,
                cp,
                dp,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "gatherv_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            counts,
        ))
    }

    /// Initialize a persistent scatterv operation (variable-count scatter).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer (significant only at root), owned by the request; write it
    ///   through `send_buffer_mut` while the request is inactive
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `displs` - Displacement for each rank in the send buffer
    /// * `recv` - Receive buffer, owned by the request; read it through `buffer` while the
    ///   request is inactive
    /// * `root` - Rank of the root process
    ///
    /// The request copies `sendcounts` and `displs`, which MPI reads until the request is
    /// freed; the caller may reuse its arrays as soon as the call returns.
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let sendcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut persistent = world
    ///     .scatterv_init(vec![1.0f64; 40], &sendcounts, &displs, vec![0.0f64; 10], 0)
    ///     .unwrap();
    /// for iter in 0..100 {
    ///     if world.rank() == 0 {
    ///         persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     }
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn scatterv_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        sendcounts: &[i32],
        displs: &[i32],
        mut recv: Vec<T>,
        root: i32,
    ) -> Result<PersistentRequest<T>> {
        if self.rank == root {
            check_v_args(
                "sendcounts",
                "displs",
                "send",
                sendcounts,
                displs,
                self.size,
                send.len(),
            )?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = vec_buf(&send);
        let (rp, n, dt) = vec_buf_mut(&mut recv);
        let (counts, [cp, dp]) = copy_arrays([sendcounts, displs]);
        // SAFETY: `sp`, `rp` and `n` describe the heap allocations of `send` and `recv`,
        // taken from the Vecs by `vec_buf`/`vec_buf_mut`. The two Vecs are distinct
        // allocations, and `send` is ignored by MPI at non-root. At root, sendcounts and
        // displs are checked above to have size() entries, non-negative counts, and each
        // positive-count block inside send; at non-root MPI does not read sendcounts or
        // displs, so they are unchecked here. `cp` and `dp` point into `counts`, the
        // request's own copy of both arrays, taken after the check and from the Vec itself.
        // The request built below owns `send`, `recv` and `counts` under the ownership
        // invariant documented on `PersistentRequest::new_pair`.
        let ret = unsafe {
            ffi::ferrompi_scatterv_init(
                sp,
                cp,
                dp,
                rp,
                n,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "scatterv_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            counts,
        ))
    }

    /// Initialize a persistent all-gatherv operation (variable-count all-gather).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer, owned by the request; write it through `send_buffer_mut`
    ///   while the request is inactive
    /// * `recv` - Receive buffer, owned by the request; read it through `buffer` while the
    ///   request is inactive
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `displs` - Displacement for each rank in the receive buffer
    ///
    /// The request copies `recvcounts` and `displs`, which MPI reads until the request is
    /// freed; the caller may reuse its arrays as soon as the call returns.
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `recvcounts` or `displs`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `recv`.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let recvcounts = vec![10i32; 4];
    /// let displs = vec![0i32, 10, 20, 30];
    /// let mut persistent = world
    ///     .allgatherv_init(vec![1.0f64; 10], vec![0.0f64; 40], &recvcounts, &displs)
    ///     .unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn allgatherv_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        mut recv: Vec<T>,
        recvcounts: &[i32],
        displs: &[i32],
    ) -> Result<PersistentRequest<T>> {
        check_v_args(
            "recvcounts",
            "displs",
            "recv",
            recvcounts,
            displs,
            self.size,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        let (counts, [cp, dp]) = copy_arrays([recvcounts, displs]);
        // SAFETY: `sp`, `n` and `rp` describe the heap allocations of `send` and `recv`,
        // taken from the Vecs by `vec_buf`/`vec_buf_mut`. The two Vecs are distinct
        // allocations. recvcounts and displs are checked above, on every rank, to have
        // size() entries, non-negative counts, and each positive-count block inside recv.
        // `cp` and `dp` point into `counts`, the request's own copy of both arrays, taken
        // after the check and from the Vec itself. The request built below owns `send`,
        // `recv` and `counts` under the ownership invariant documented on
        // `PersistentRequest::new_pair`.
        let ret = unsafe {
            ffi::ferrompi_allgatherv_init(sp, n, rp, cp, dp, dt, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "allgatherv_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            counts,
        ))
    }

    /// Initialize a persistent all-to-allv operation (variable-count all-to-all).
    ///
    /// The returned handle can be started multiple times with `start()`.
    /// Requires MPI 4.0, or Open MPI 5.
    ///
    /// # Arguments
    ///
    /// * `send` - Send buffer, owned by the request; write it through `send_buffer_mut`
    ///   while the request is inactive
    /// * `sendcounts` - Number of elements to send to each rank
    /// * `sdispls` - Send displacement for each rank
    /// * `recv` - Receive buffer, owned by the request; read it through `buffer` while the
    ///   request is inactive
    /// * `recvcounts` - Number of elements to receive from each rank
    /// * `rdispls` - Receive displacement for each rank
    ///
    /// The request copies `sendcounts`, `sdispls`, `recvcounts` and `rdispls`, which MPI
    /// reads until the request is freed; the caller may reuse its arrays as soon as the
    /// call returns.
    ///
    /// # Errors
    ///
    /// - [`Error::BufferSize`] if, on a rank where MPI reads them, `sendcounts`/`sdispls` or `recvcounts`/`rdispls`
    ///   does not have `size()` entries, or a block with a positive count ends past
    ///   the end of `send`/`recv` respectively.
    /// - [`Error::InvalidArgument`] if, on a rank where MPI reads them, a count is
    ///   negative or a block with a positive count starts at a negative displacement.
    /// - [`Error::Mpi`] if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let sendcounts = vec![10i32; 4];
    /// let sdispls = vec![0i32, 10, 20, 30];
    /// let recvcounts = vec![10i32; 4];
    /// let rdispls = vec![0i32, 10, 20, 30];
    /// let mut persistent = world.alltoallv_init(
    ///     vec![1.0f64; 40], &sendcounts, &sdispls,
    ///     vec![0.0f64; 40], &recvcounts, &rdispls,
    /// ).unwrap();
    /// for iter in 0..100 {
    ///     persistent.send_buffer_mut().unwrap().fill(iter as f64);
    ///     persistent.start().unwrap();
    ///     persistent.wait().unwrap();
    ///     assert_eq!(persistent.buffer().unwrap()[0], iter as f64);
    /// }
    /// ```
    pub fn alltoallv_init<T: MpiDatatype>(
        &self,
        send: Vec<T>,
        sendcounts: &[i32],
        sdispls: &[i32],
        mut recv: Vec<T>,
        recvcounts: &[i32],
        rdispls: &[i32],
    ) -> Result<PersistentRequest<T>> {
        check_v_args(
            "sendcounts",
            "sdispls",
            "send",
            sendcounts,
            sdispls,
            self.size,
            send.len(),
        )?;
        check_v_args(
            "recvcounts",
            "rdispls",
            "recv",
            recvcounts,
            rdispls,
            self.size,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = vec_buf(&send);
        let (rp, _, _) = vec_buf_mut(&mut recv);
        let (counts, [scp, sdp, rcp, rdp]) =
            copy_arrays([sendcounts, sdispls, recvcounts, rdispls]);
        // SAFETY: `sp` and `rp` describe the heap allocations of `send` and `recv`, taken
        // from the Vecs by `vec_buf`/`vec_buf_mut`. Two non-empty Vecs are distinct
        // allocations, and MPI reads and writes nothing through an empty one.
        // sendcounts/sdispls and recvcounts/rdispls are each checked above, on every rank,
        // to have size() entries, non-negative counts, and each positive-count block inside
        // send/recv respectively. `scp`, `sdp`, `rcp` and `rdp` point into `counts`, the
        // request's own copy of the four arrays, taken after the checks and from the Vec
        // itself. The request built below owns `send`, `recv` and `counts` under the
        // ownership invariant documented on `PersistentRequest::new_pair`.
        let ret = unsafe {
            ffi::ferrompi_alltoallv_init(
                sp,
                scp,
                sdp,
                rp,
                rcp,
                rdp,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "alltoallv_init")?;
        Ok(PersistentRequest::new_pair(
            request_handle,
            send,
            recv,
            counts,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{check_v_args, copy_arrays};
    use crate::comm::test_comm;
    use crate::error::{Error, Result};

    fn check(counts: &[i32], displs: &[i32], size: i32, buf_len: usize) -> Result<()> {
        check_v_args(
            "recvcounts",
            "displs",
            "recv",
            counts,
            displs,
            size,
            buf_len,
        )
    }

    #[test]
    fn check_v_args_boundaries() {
        // exact fit
        assert!(check(&[2, 2], &[0, 2], 2, 4).is_ok());
        // one element past the end
        assert!(matches!(
            check(&[2, 2], &[0, 2], 2, 3),
            Err(Error::BufferSize {
                arg: "recv",
                required: 4,
                actual: 3
            })
        ));
        // counts.len() != size
        assert!(matches!(
            check(&[2, 2, 2], &[0, 2, 4], 2, 6),
            Err(Error::BufferSize {
                arg: "recvcounts",
                required: 2,
                actual: 3
            })
        ));
        // displs.len() != size
        assert!(matches!(
            check(&[2, 2], &[0, 2, 4], 2, 6),
            Err(Error::BufferSize {
                arg: "displs",
                required: 2,
                actual: 3
            })
        ));
        // negative count
        assert!(matches!(
            check(&[-1, 2], &[0, 2], 2, 4),
            Err(Error::InvalidArgument {
                arg: "recvcounts",
                reason: "negative count"
            })
        ));
        // negative displacement with count > 0
        assert!(matches!(
            check(&[2, 2], &[-1, 2], 2, 4),
            Err(Error::InvalidArgument {
                arg: "displs",
                reason: "negative displacement"
            })
        ));
        // negative displacement with count 0
        assert!(check(&[0, 2], &[-1, 0], 2, 2).is_ok());
        // i32 arithmetic would wrap: i32::MAX + 1 overflows i32 but not i64
        assert!(matches!(
            check(&[1], &[i32::MAX], 1, i32::MAX as usize),
            Err(Error::BufferSize {
                arg: "recv",
                required: 0x8000_0000,
                actual: 0x7fff_ffff
            })
        ));
    }

    #[test]
    fn check_v_args_reports_the_first_fault() {
        // both arrays too long: the counts length is reported
        assert!(matches!(
            check(&[2, 2, 2], &[0, 2, 4, 6], 2, 6),
            Err(Error::BufferSize {
                arg: "recvcounts",
                ..
            })
        ));
        // negative displacement and a block past the buffer: the displacement is reported
        assert!(matches!(
            check(&[1], &[-1], 1, 0),
            Err(Error::InvalidArgument { arg: "displs", .. })
        ));
    }

    /// Each pointer addresses the start of its array in the copy and stays
    /// readable after the `Vec` moves, as a request's owner moves it while MPI
    /// holds the pointers. Run under Miri, with and without
    /// `-Zmiri-tree-borrows`, to check the pointers' provenance.
    #[test]
    fn copy_arrays_pointers_address_each_array_after_a_move() {
        let (counts, [a, b, c, d]) = copy_arrays([&[1, 2, 3], &[], &[4], &[5, 6]]);
        let moved = counts;
        assert_eq!(moved, [1, 2, 3, 4, 5, 6]);
        assert_eq!(b, c, "an empty array starts where the next one does");
        // SAFETY: each pointer addresses its array inside the copy's live
        // allocation of six `i32`s, which `moved` owns and does not resize.
        unsafe {
            assert_eq!([a.read(), a.add(2).read()], [1, 3]);
            assert_eq!(c.read(), 4);
            assert_eq!([d.read(), d.add(1).read()], [5, 6]);
        }
    }

    #[test]
    fn gatherv_init_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.gatherv_init(send, recv, &recvcounts, &displs, 0);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn scatterv_init_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let recv = vec![0.0f64; 10];
        let result = comm.scatterv_init(send, &sendcounts, &displs, recv, 0);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn allgatherv_init_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.allgatherv_init(send, recv, &recvcounts, &displs);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn alltoallv_init_mismatched_send_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20, 30];
        let result = comm.alltoallv_init(send, &sendcounts, &sdispls, recv, &recvcounts, &rdispls);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "sdispls",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn alltoallv_init_mismatched_recv_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20, 30];
        let recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.alltoallv_init(send, &sendcounts, &sdispls, recv, &recvcounts, &rdispls);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "rdispls",
                required: 4,
                actual: 3
            })
        ));
    }

    // ── blocking / nonblocking v-collective length-mismatch guards ────────

    #[test]
    fn gatherv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.gatherv(&send, &mut recv, &recvcounts, &displs, 0);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn scatterv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 10];
        let result = comm.scatterv(&send, &sendcounts, &displs, &mut recv, 0);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn allgatherv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = comm.allgatherv(&send, &mut recv, &recvcounts, &displs);
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn alltoallv_mismatched_send_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
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
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "sdispls",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn alltoallv_mismatched_recv_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
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
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "rdispls",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn igatherv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = crate::scope(|s| {
            comm.igatherv(s, &send, &mut recv, &recvcounts, &displs, 0)
                .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn iscatterv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 10];
        let result = crate::scope(|s| {
            comm.iscatterv(s, &send, &sendcounts, &displs, &mut recv, 0)
                .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn iallgatherv_mismatched_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let displs = vec![0i32, 10, 20]; // 3 elements != 4
        let result = crate::scope(|s| {
            comm.iallgatherv(s, &send, &mut recv, &recvcounts, &displs)
                .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "displs",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn ialltoallv_mismatched_send_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20, 30];
        let result = crate::scope(|s| {
            comm.ialltoallv(
                s,
                &send,
                &sendcounts,
                &sdispls,
                &mut recv,
                &recvcounts,
                &rdispls,
            )
            .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "sdispls",
                required: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn ialltoallv_mismatched_recv_counts_displs_returns_buffer_size() {
        let comm = test_comm(0, 4);
        let send = vec![1.0f64; 40];
        let sendcounts = vec![10i32; 4];
        let sdispls = vec![0i32, 10, 20, 30];
        let mut recv = vec![0.0f64; 40];
        let recvcounts = vec![10i32; 4];
        let rdispls = vec![0i32, 10, 20]; // 3 elements != 4
        let result = crate::scope(|s| {
            comm.ialltoallv(
                s,
                &send,
                &sendcounts,
                &sdispls,
                &mut recv,
                &recvcounts,
                &rdispls,
            )
            .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "rdispls",
                required: 4,
                actual: 3
            })
        ));
    }
}
