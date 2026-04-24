//! Variable-count collective operations: gatherv, scatterv, allgatherv, alltoallv
//! and their nonblocking (i*) and persistent (*_init) variants.

use crate::comm::Communicator;
use crate::datatype::MpiDatatype;
use crate::error::{Error, Result};
use crate::ffi;
use crate::persistent::PersistentRequest;
use crate::request::Request;

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
}

#[cfg(test)]
mod tests {
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
}
