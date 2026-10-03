//! Regression for the debug-only Serialized overlap check on the scope-end
//! wait: the final `MPI_Waitall` of a nonblocking scope that overlaps another
//! thread's in-flight MPI call cannot run, so the scope prints a message that
//! names the overlap and aborts the process. Needs a debug build: release
//! builds do not check overlaps.
//!
//! A worker blocks in a receive nothing will match, once the scope's own
//! receive is posted: a worker inside MPI earlier would make that post fail.
//! The scope polls a barrier until the overlap check rejects it, which shows
//! the worker is inside MPI, and then ends with its receive still pending.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_scope_wait_overlap
// mpi-test: np=1 expect=abort
// mpi-test-stderr: ferrompi: a nonblocking scope's final wait overlapped another thread's MPI call

use std::sync::atomic::{AtomicBool, Ordering};

use ferrompi::{Error, Mpi, ThreadLevel};

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");
    let world = mpi.world();
    assert_eq!(mpi.thread_level(), ThreadLevel::Serialized);

    let mut buf = [0i32; 1];
    let posted = AtomicBool::new(false);

    std::thread::scope(|t| {
        t.spawn(|| {
            while !posted.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            while matches!(
                world.recv(&mut [0i32; 1], 0, 99),
                Err(Error::ThreadLevelViolation)
            ) {}
        });

        let r = ferrompi::scope(|s| {
            world.irecv(s, &mut buf, 0, 98).expect("irecv");
            posted.store(true, Ordering::Release);
            while !matches!(world.barrier(), Err(Error::ThreadLevelViolation)) {
                std::thread::yield_now();
            }
            Ok(())
        });
        println!("FAIL: the scope ended under an overlap: {r:?}");
        std::process::exit(1);
    });
}
