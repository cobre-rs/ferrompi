//! Dropping `Mpi` at `ThreadLevel::Multiple` while four threads call MPI
//! waits for the calls in progress, and a call that starts after the drop
//! began returns `Error::Finalized` without reaching MPI.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_finalize_in_flight
// mpi-test: np=1

use ferrompi::{Mpi, ThreadLevel};

mod common;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");

    // Some builds (e.g. certain Cray MPT configurations) deliberately refuse it.
    if mpi.thread_level() < ThreadLevel::Multiple {
        common::skip(
            &mpi.world(),
            &format!(
                "MPI provided {:?}, MPI_THREAD_MULTIPLE required; skipping test",
                mpi.thread_level()
            ),
        );
        return;
    }

    common::finalize_under_load(mpi, 4, "test_finalize_in_flight");
}
