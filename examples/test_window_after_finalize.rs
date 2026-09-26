//! Regression for window memory outliving `Mpi`.
//!
//! `Mpi::drop` skips `MPI_Finalize` while any window is still alive — a
//! `Win::allocate` window and a `SharedWindow` here — because some MPI
//! implementations free MPI-allocated window memory inside `MPI_Finalize`
//! itself, and some abort while tearing down internal state that still
//! tracks a live window's buffer. Either way, the memory stays valid, but
//! `MPI_Finalize` is never called for this process.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_after_finalize
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: MPI_Finalize skipped

use ferrompi::{Mpi, SharedWindow, Win};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let node = world.split_shared().expect("split_shared failed");

    let mut win = Win::<f64>::allocate(&world, 1024).expect("Win::allocate failed");
    win.local_slice_mut().fill(1.5);

    let mut shared =
        SharedWindow::<f64>::allocate(&node, 1 << 16).expect("SharedWindow::allocate failed");
    shared.local_slice_mut()[0] = 42.0;

    drop(node);
    drop(mpi);

    assert!(
        Mpi::is_finalized(),
        "Mpi::is_finalized() must be true after a skipped MPI_Finalize"
    );

    let sum: f64 = win.local_slice().iter().sum();
    assert_eq!(
        sum, 1536.0,
        "win.local_slice() must still be readable after the skipped MPI_Finalize"
    );

    assert_eq!(
        shared.local_slice()[0],
        42.0,
        "shared.local_slice() must still be readable after the skipped MPI_Finalize"
    );

    drop(shared);
    drop(win);

    println!("test_window_after_finalize: PASS");
}
