//! Regression for zeroed window memory.
//!
//! `Win::allocate` and `SharedWindow::allocate` must never hand a caller a
//! fresh window's memory before it has been zeroed: reading it is an
//! uninitialised-memory read (undefined behaviour). The "fresh" checks
//! below are the valgrind oracle for this — run under
//! `valgrind --track-origins=yes` at np=1, any uninitialised byte in a
//! freshly allocated segment fails the run even where the value itself
//! happens to read zero by luck. The "reused" checks are a value oracle
//! only: they prove a window's previous contents (`u64::MAX`) never leak
//! into a new window allocated after the old one was dropped, which
//! valgrind alone cannot see once the memory has been written at least
//! once.
//!
//! Also checks that a zero-count allocation on both window kinds returns
//! `Ok` with an empty `local_slice()`, rather than leaking the registered
//! window on every rank.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_zeroed
// mpi-test: np=1.. valgrind

use ferrompi::{Mpi, SharedWindow, Win};

mod common;

const N: usize = 4096;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let node = world.split_shared().expect("split_shared failed");

    // Fresh Win::allocate memory reads zero (valgrind oracle).
    {
        let win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let ok = win.local_slice().iter().all(|&v| v == 0);
        common::check(&world, ok, "fresh Win::allocate memory reads zero");
    }

    // Reused Win::allocate memory reads zero (value oracle).
    {
        {
            let mut win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
            win.local_slice_mut().fill(u64::MAX);
        }
        let win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let ok = win.local_slice().iter().all(|&v| v == 0);
        common::check(&world, ok, "reused Win::allocate memory reads zero");
    }

    // Fresh SharedWindow::allocate memory reads zero (valgrind oracle).
    {
        let win = SharedWindow::<u64>::allocate(&node, N).expect("SharedWindow::allocate failed");
        let ok = win.local_slice().iter().all(|&v| v == 0);
        common::check(&world, ok, "fresh SharedWindow::allocate memory reads zero");
    }

    // Reused SharedWindow::allocate memory reads zero (value oracle).
    {
        {
            let mut win =
                SharedWindow::<u64>::allocate(&node, N).expect("SharedWindow::allocate failed");
            win.local_slice_mut().fill(u64::MAX);
        }
        let win = SharedWindow::<u64>::allocate(&node, N).expect("SharedWindow::allocate failed");
        let ok = win.local_slice().iter().all(|&v| v == 0);
        common::check(
            &world,
            ok,
            "reused SharedWindow::allocate memory reads zero",
        );
    }

    // A zero-count allocation returns Ok with an empty local_slice() on
    // both window kinds, rather than leaking the registered window.
    {
        let result = Win::<u64>::allocate(&world, 0);
        let ok = match &result {
            Ok(win) => win.local_slice().is_empty(),
            Err(e) => {
                eprintln!("FAIL: Win::allocate(0) returned Err: {e}");
                false
            }
        };
        common::check(
            &world,
            ok,
            "Win::allocate(0) returns Ok with empty local_slice",
        );
    }
    {
        let result = SharedWindow::<u64>::allocate(&node, 0);
        let ok = match &result {
            Ok(win) => win.local_slice().is_empty(),
            Err(e) => {
                eprintln!("FAIL: SharedWindow::allocate(0) returned Err: {e}");
                false
            }
        };
        common::check(
            &world,
            ok,
            "SharedWindow::allocate(0) returns Ok with empty local_slice",
        );
    }

    if world.rank() == 0 {
        println!("PASS: test_rma_zeroed");
    }
}
