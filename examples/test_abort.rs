//! Integration test for `Communicator::abort`.
//!
//! Rank 0 prints the abort marker to stderr and calls `abort`, which never
//! returns. The other ranks block in `barrier` and are terminated when the
//! abort tears down the process group.
//!
//! Rank 0's marker print is followed by a barrier and a short sleep before
//! `abort`. Without them, the launcher can tear the job down before
//! forwarding rank 0's already-written stderr: reproduced at a 6% rate
//! (6/100 runs, entirely empty combined output) on MPICH 4.2.1 + Hydra with
//! `MPI_Abort` called immediately after the marker write; a bare barrier cut
//! that to 1/200 runs, and barrier+sleep(100ms) held at 0/500 runs.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_abort
// mpi-test: np=2 expect=abort
// mpi-test-stderr: test_abort: calling abort

use ferrompi::{Mpi, Result};
use std::time::Duration;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    if world.rank() == 0 {
        eprintln!("test_abort: calling abort");
    }

    world.barrier()?;
    // Covers the launcher's stderr-forwarding race with MPI_Abort (Hydra can tear the job down before forwarding rank 0's already-written marker).
    std::thread::sleep(Duration::from_millis(100));

    if world.rank() == 0 {
        world.abort(3);
    }

    world.barrier()?;

    Ok(())
}
