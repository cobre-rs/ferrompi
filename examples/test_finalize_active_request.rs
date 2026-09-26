//! Regression for `MPI_Finalize`'s request sweep freeing an active request.
//!
//! Each rank creates an inactive persistent request (`send_init`, never
//! started) and an active nonblocking request (`iallreduce`, never waited),
//! then drops `Mpi`. The finalize sweep must free the inactive persistent
//! request without counting it, and leave the active nonblocking request
//! alone, counting it. The runner always builds the dev profile, so the
//! count is reported on stderr whenever it is greater than zero.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_finalize_active_request
// mpi-test: np=2
// mpi-test-stderr: ferrompi: MPI_Finalize leaves 1 active request(s) unfreed

use ferrompi::{Mpi, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    let persistent = world
        .send_init(&[7i32], 1 - rank, 1)
        .expect("send_init failed");
    let mut recv = [0.0f64; 1];
    let active = world
        .iallreduce(&[1.0f64], &mut recv, ReduceOp::Sum)
        .expect("iallreduce failed");

    drop(mpi);

    drop(persistent);
    drop(active);

    println!("PASS: test_finalize_active_request");
}
