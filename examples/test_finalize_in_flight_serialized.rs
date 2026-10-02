//! Dropping `Mpi` at `ThreadLevel::Serialized` while one thread calls MPI
//! waits for the call in progress, and a call that starts after the drop
//! began returns `Error::Finalized` without reaching MPI. One worker keeps
//! the program legal at `Serialized`: the init thread's `MPI_Finalize` is
//! the only other MPI call, and it now follows the worker's last call.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_finalize_in_flight_serialized
// mpi-test: np=1

use ferrompi::{Mpi, ThreadLevel};

mod common;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");

    if mpi.thread_level() < ThreadLevel::Serialized {
        common::skip(
            &mpi.world(),
            &format!(
                "MPI provided {:?}, MPI_THREAD_SERIALIZED required; skipping test",
                mpi.thread_level()
            ),
        );
        return;
    }

    common::finalize_under_load(mpi, 1, "test_finalize_in_flight_serialized");
}
