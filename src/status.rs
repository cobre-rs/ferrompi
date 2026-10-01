//! MPI message status information.
//!
//! This module provides the [`Status`] struct returned by probe operations,
//! containing metadata about a message without actually receiving it.

/// Information about a probed or received MPI message.
///
/// Returned by [`Communicator::probe`](crate::Communicator::probe) and
/// [`Communicator::iprobe`](crate::Communicator::iprobe) to describe an
/// incoming message without consuming it.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::Mpi;
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Blocking probe for any f64 message
/// let status = world.probe::<f64>(-1, -1).unwrap();
/// println!("Message from rank {} with tag {}, {} elements",
///          status.source, status.tag, status.count);
/// ```
#[derive(Debug, Clone)]
pub struct Status {
    /// Source rank of the message.
    pub source: i32,
    /// Tag of the message.
    pub tag: i32,
    /// Number of elements of the probed type in the message, from `MPI_Get_count`;
    /// `-1` when the message is not a whole number of elements (MPI reports
    /// `MPI_UNDEFINED`).
    pub count: i64,
}
