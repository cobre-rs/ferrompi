//! A window that is refused a free, or dropped after a skipped finalize, makes
//! no MPI call and stays counted.
//!
//! Part 1: rank 0's worker thread blocks in a receive that rank 1 matches only
//! after 2 s, and meanwhile the main thread calls `Win::into_buffer` on a
//! window of a self-communicator. In a debug build at `Serialized` the call
//! overlaps the worker's, so it must return `ThreadLevelViolation` without
//! freeing, and leak the window counted. The mutant is `into_buffer` without
//! the overlap check: the free runs and returns `Ok`, which prints `FAIL:`.
//!
//! Part 2: both ranks create a window on `world`, then drop `Mpi`, which skips
//! `MPI_Finalize`: rank 0 has two live windows, the leaked one and this one,
//! and rank 1 has one. Rank 1 forgets its window, so that it frees nothing even
//! under the mutant, and sleeps 3 s. Rank 0 drops its window and prints the
//! literal below. The mutant is a `Drop` that calls `MPI_Win_free` after the
//! skip even though `rt::drop_guard` refused: the free is collective, so rank 0
//! blocks in it until the launcher or the runner's timeout kills it, and the
//! literal never prints.
//!
//! At np 2, once one rank exits the launcher may kill the other, so only the
//! literal and `FAIL` lines are evidence. Every check therefore runs before
//! `drop(mpi)` through `common::check`, or right after it through a local check
//! that exits 2, because `common::check`'s allreduce is refused after the drop.
//!
//! Needs a debug build: release builds do not check overlaps.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_window_leak_paths
// mpi-test: np=2 expect=unfinalized
// mpi-test-stderr: test_window_leak_paths: rank 0 dropped its window after the skipped finalize without an MPI call

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrompi::{Error, Mpi, ThreadLevel, Win};

mod common;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    assert_eq!(
        world.size(),
        2,
        "test_window_leak_paths requires exactly 2 processes"
    );

    let me = world
        .split(Some(rank), 0)
        .expect("split failed")
        .expect("split gave this rank no communicator");

    let refused = if rank == 0 {
        let w1 = Win::<u64>::allocate(&me, 4).expect("Win::allocate failed");
        let started = AtomicBool::new(false);
        let freed = std::thread::scope(|s| {
            let worker = s.spawn(|| {
                started.store(true, Ordering::SeqCst);
                world.recv(&mut [0i32; 1], 1, 99)
            });
            while !started.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(200));
            let freed = w1.into_buffer();
            worker
                .join()
                .expect("worker thread panicked")
                .expect("recv failed");
            freed
        });
        matches!(freed, Err(Error::ThreadLevelViolation))
    } else {
        std::thread::sleep(Duration::from_secs(2));
        world.send(&[1i32], 0, 99).expect("send failed");
        true
    };
    common::check(
        &world,
        refused,
        "an overlapping into_buffer returned ThreadLevelViolation",
    );

    let w2 = Win::<u64>::allocate(&world, 4).expect("Win::allocate failed");
    world.barrier().expect("barrier failed");

    drop(mpi);
    if rank == 1 {
        std::mem::forget(w2);
        std::thread::sleep(Duration::from_secs(3));
        return;
    }
    if !Mpi::is_finalized() {
        eprintln!("FAIL: the skipped finalize reports finalized");
        std::process::exit(2);
    }
    drop(w2);
    eprintln!(
        "test_window_leak_paths: rank 0 dropped its window after the skipped finalize without an MPI call"
    );
}
