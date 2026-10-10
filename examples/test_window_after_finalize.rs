//! Regression for window memory outliving `Mpi`.
//!
//! `Mpi::drop` skips `MPI_Finalize` while any window is still alive — a
//! `Win::allocate` window and a `Win::allocate_shared` window here — because some MPI
//! implementations free MPI-allocated window memory inside `MPI_Finalize`
//! itself, and some abort while tearing down internal state that still
//! tracks a live window's buffer. Either way, the memory stays valid, but
//! `MPI_Finalize` is never called for this process.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_after_finalize
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: MPI_Finalize skipped

use ferrompi::{Mpi, Win};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let node = world.split_shared().expect("split_shared failed");

    let win = Win::<f64>::allocate(&world, 1024).expect("Win::allocate failed");
    win.local().copy_from_slice(&[1.5; 1024]);

    let shared = Win::<f64>::allocate_shared(&node, 1 << 16).expect("Win::allocate_shared failed");
    shared.local().write(0, 42.0);

    drop(node);
    drop(mpi);

    assert!(
        Mpi::is_finalized(),
        "Mpi::is_finalized() must be true after a skipped MPI_Finalize"
    );

    let mut seen = [0.0f64; 1024];
    win.local().copy_to_slice(&mut seen);
    let sum: f64 = seen.iter().sum();
    assert_eq!(
        sum, 1536.0,
        "win.local() must still be readable after the skipped MPI_Finalize"
    );

    assert_eq!(
        shared.local().read(0),
        42.0,
        "shared.local() must still be readable after the skipped MPI_Finalize"
    );

    drop(shared);
    drop(win);

    println!("test_window_after_finalize: PASS");
}
