//! Dropping `Mpi` at `ThreadLevel::Multiple` while a worker thread is inside a
//! nonblocking scope waits for that scope to end before it finalizes. The
//! worker posts an `ibarrier` and leaves it to the scope end; the init thread
//! drops `Mpi` right after the worker signals. The drop must return only after
//! the worker's scope returned `Ok`, and MPI must really finalize, which the
//! runner checks by the absence of the `MPI_Finalize skipped` warning.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_scope_finalize
// mpi-test: np=1

use std::sync::mpsc;
use std::time::{Duration, Instant};

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

    let world = mpi.world();
    let (posted, posted_rx) = mpsc::channel();

    let (scope_result, drop_end) = std::thread::scope(|t| {
        let worker = t.spawn(|| {
            ferrompi::scope(|s| {
                world.ibarrier(s)?;
                posted.send(()).expect("init thread hung up");
                std::thread::sleep(Duration::from_millis(200));
                Ok(Instant::now())
            })
        });

        posted_rx.recv().expect("worker hung up before posting");
        drop(mpi);
        let drop_end = Instant::now();

        (worker.join().expect("worker panicked"), drop_end)
    });

    let scope_end = scope_result.expect("the worker's scope must return Ok");
    assert!(
        drop_end >= scope_end,
        "Mpi::drop returned before the worker's scope ended"
    );
    assert!(
        Mpi::is_finalized(),
        "Mpi::is_finalized() must be true after the drop"
    );

    println!("test_scope_finalize: PASS");
}
