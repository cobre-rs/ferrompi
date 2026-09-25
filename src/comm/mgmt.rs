//! Communicator management operations: duplicate, split, abort, topology.

use crate::comm::{Communicator, SplitType};
use crate::error::{Error, Result};
use crate::ffi;
use crate::group::Group;

impl Communicator {
    /// Get the processor name for this process.
    pub fn processor_name(&self) -> Result<String> {
        let mut buf = [0u8; 256];
        let mut len: i32 = 0;
        // SAFETY: buf is a 256-byte stack array, exclusively writable; the C shim writes at most
        // MPI_MAX_PROCESSOR_NAME bytes and sets len to the actual length.
        let ret = unsafe {
            ffi::ferrompi_get_processor_name(buf.as_mut_ptr().cast::<std::ffi::c_char>(), &mut len)
        };
        Error::check_with_op(ret, "get_processor_name")?;
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

    /// Get the group associated with this communicator.
    ///
    /// Returns a [`Group`] containing all processes in this communicator.
    /// Use [`Group::include`] or [`Group::exclude`] on the returned group
    /// to derive sub-groups, which can then be used with
    /// [`Communicator::create_from_group`](Self::create_from_group).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `MPI_Comm_group` call fails or if
    /// the C-side group table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let g = world.group().unwrap();
    /// assert_eq!(g.size().unwrap(), world.size());
    /// ```
    pub fn group(&self) -> Result<Group> {
        let mut group_handle: i32 = 0;
        // SAFETY: self.handle is owned by this Communicator; group_handle is an exclusive output.
        let ret = unsafe { ffi::ferrompi_comm_group(self.handle, &mut group_handle) };
        Error::check_with_op(ret, "comm_group")?;
        Ok(Group {
            handle: group_handle,
        })
    }

    /// Duplicate this communicator.
    pub fn duplicate(&self) -> Result<Self> {
        let mut new_handle: i32 = 0;
        // SAFETY: self.handle is owned by this Communicator; new_handle is an exclusive output.
        let ret = unsafe { ffi::ferrompi_comm_dup(self.handle, &mut new_handle) };
        Error::check_with_op(ret, "comm_dup")?;
        Self::from_handle(new_handle)
    }

    /// Convert a communicator-handle result into `Option<Self>`, treating a
    /// negative handle as `MPI_COMM_NULL` (used by split, split_type, and
    /// create_from_group).
    fn from_optional_handle(new_handle: i32) -> Result<Option<Self>> {
        if new_handle < 0 {
            Ok(None)
        } else {
            Self::from_handle(new_handle).map(Some)
        }
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
        // SAFETY: self.handle is owned by this Communicator; remaining arguments are scalars.
        let ret = unsafe { ffi::ferrompi_comm_split(self.handle, color, key, &mut new_handle) };
        Error::check_with_op(ret, "comm_split")?;
        Self::from_optional_handle(new_handle)
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
        // SAFETY: self.handle is owned by this Communicator; remaining arguments are scalars.
        let ret = unsafe {
            ffi::ferrompi_comm_split_type(self.handle, split_type as i32, key, &mut new_handle)
        };
        Error::check_with_op(ret, "comm_split_type")?;
        Self::from_optional_handle(new_handle)
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

    /// Create a sub-communicator from a group.
    ///
    /// `group` must be a subset of `self`'s group. This call is collective
    /// over `self` — every rank in the parent communicator must invoke it,
    /// even ranks not in `group`.
    ///
    /// Returns `Ok(None)` for ranks that are not members of `group`
    /// (MPI returns `MPI_COMM_NULL` for those ranks). Returns
    /// `Ok(Some(comm))` for ranks that are members.
    ///
    /// # Errors
    ///
    /// Returns an error if `group` contains a rank not present in the parent
    /// communicator (`MPI_ERR_GROUP`), or if the C-side communicator table is
    /// full (`MPI_ERR_OTHER`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let g = world.group().unwrap().include(&[0, 2]).unwrap();
    /// // Collective: every rank calls create_from_group
    /// if let Some(sub) = world.create_from_group(&g).unwrap() {
    ///     assert_eq!(sub.size(), 2);
    /// }
    /// ```
    pub fn create_from_group(&self, group: &Group) -> Result<Option<Communicator>> {
        let mut new_handle: i32 = -1;
        // SAFETY: self.handle is owned by this Communicator; group.handle is owned by the Group
        // argument and remains valid for the duration of this call.
        let ret = unsafe {
            ffi::ferrompi_comm_create_from_group_parent(self.handle, group.handle, &mut new_handle)
        };
        Error::check_with_op(ret, "comm_create_from_group_parent")?;
        Self::from_optional_handle(new_handle)
    }

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
    ///
    /// # Fallback Behavior
    ///
    /// In the standard case `MPI_Abort` terminates the entire process
    /// group and this function never returns. If `MPI_Abort` itself
    /// returns (non-standard but observed in some implementations during
    /// internal failures), this function falls back to
    /// [`std::process::abort`], which raises `SIGABRT` and terminates
    /// the current process immediately. The original `errorcode` is lost
    /// in that fallback path because `process::abort` does not accept an
    /// exit code; the signal code (usually 134 = 128 + 6) is the best
    /// evidence the caller has that the fallback triggered.
    pub fn abort(&self, errorcode: i32) -> ! {
        // SAFETY: ferrompi_abort delegates to MPI_Abort, which is
        // defined to terminate all processes in the communicator and
        // never return. This call is safe to make with any valid
        // communicator handle; self.handle is always valid because
        // Communicator is only constructed through safe methods that
        // guarantee handle validity.
        unsafe { ffi::ferrompi_abort(self.handle, errorcode) };

        // Defense in depth: the MPI standard says MPI_Abort "should"
        // terminate all processes but does not strictly guarantee the
        // calling process aborts before return. If MPI_Abort ever
        // returns (non-standard implementation behavior), we must still
        // honor the `-> !` contract. std::process::abort() raises
        // SIGABRT and is guaranteed to diverge without running
        // destructors — which is the right outcome, because any
        // destructor that touches MPI state after a failed MPI_Abort
        // has undefined behavior.
        std::process::abort()
    }
}
