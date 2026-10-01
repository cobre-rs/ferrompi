//! Regression for a window the table cannot hold when it is full.
//!
//! Fills the 256-slot window table with `Win::allocate` calls, then checks
//! that `SharedWindow::allocate` sees the same table-full error. Each failed
//! call leaves one leaked window counted as alive, so dropping every
//! returned window still leaves two and `Mpi` skips `MPI_Finalize`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_table_full
// mpi-test: np=1 expect=unfinalized
// mpi-test-stderr: ferrompi: MPI_Finalize skipped: 2 window(s) still alive

use ferrompi::{Error, Mpi, ResourceKind, SharedWindow, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    let mut windows = Vec::new();
    loop {
        match Win::<f64>::allocate(&world, 1) {
            Ok(win) => windows.push(win),
            Err(err) => {
                common::check(
                    &world,
                    matches!(
                        err,
                        Error::ResourceExhausted {
                            resource: ResourceKind::Window,
                            ..
                        }
                    ),
                    "Win::allocate table-full error",
                );
                break;
            }
        }
    }
    common::check(&world, !windows.is_empty(), "at least one window created");

    let node = world.split_shared().expect("split_shared failed");
    let shared_err = SharedWindow::<f64>::allocate(&node, 1);
    common::check(
        &world,
        matches!(
            shared_err,
            Err(Error::ResourceExhausted {
                resource: ResourceKind::Window,
                ..
            })
        ),
        "SharedWindow::allocate table-full error",
    );

    drop(windows);
    drop(node);
    drop(mpi);

    println!("test_window_table_full: PASS");
}
