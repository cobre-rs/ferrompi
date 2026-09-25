//! Hello World example - basic MPI initialization and communication.
//!
//! Run with: mpiexec -n 4 cargo run --example hello_world
// mpi-test: np=2..

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    // Initialize MPI
    let mpi = Mpi::init()?;

    // Get the world communicator
    let world = mpi.world();

    // Get our rank and the total number of processes
    let rank = world.rank();
    let size = world.size();

    // Get processor name
    let processor_name = world.processor_name()?;

    // Get MPI version
    let version = Mpi::version()?;

    println!(
        "Hello from rank {} of {} on {} ({})",
        rank, size, processor_name, version
    );

    // Synchronize before exiting
    world.barrier()?;

    if rank == 0 {
        println!("\nAll processes reported in. Test passed!");
    }

    // MPI is finalized when `mpi` is dropped
    Ok(())
}
