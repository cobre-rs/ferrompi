//! Regression for a `Drop` that would call MPI from a non-init thread.
//!
//! Converts a repro where a `Communicator` duplicate moved to a worker
//! thread and dropped there called `MPI_Comm_free` from that thread. Below
//! `Serialized`, a `Drop` that would call MPI on a non-init thread must
//! abort the process instead, with a diagnostic naming the type and thread.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_drop_wrong_thread
// mpi-test: np=1 expect=abort
// mpi-test-stderr: dropped on thread

use ferrompi::{Mpi, Result, ThreadLevel};

fn main() -> Result<()> {
    let mpi = Mpi::init_thread(ThreadLevel::Funneled)?;
    assert_eq!(mpi.thread_level(), ThreadLevel::Funneled);
    let world = mpi.world();

    let dup = world.duplicate()?;
    std::thread::Builder::new()
        .name("worker".into())
        .spawn(move || drop(dup))
        .expect("spawn failed")
        .join()
        .expect("worker thread panicked");

    // Reached only if the wrong-thread drop failed to abort the process.
    println!("test_drop_wrong_thread: drop returned");
    Ok(())
}
