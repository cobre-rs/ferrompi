//! Integration test for `Communicator::abort`.
//!
//! Rank 0 prints the abort marker to stderr and calls `abort`, which never
//! returns. The other ranks block in `barrier` and are terminated when the
//! abort tears down the process group.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_abort
// mpi-test: np=2 expect=abort
// mpi-test-stderr: test_abort: calling abort

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    if world.rank() == 0 {
        eprintln!("test_abort: calling abort");
        world.abort(3);
    }

    world.barrier()?;

    Ok(())
}
