//! Regression for a live `Win::create` window at finalize.
//!
//! `MPI_Win_free` is collective, so the finalize sweep must not call it on a
//! window ranks may have leaked inconsistently; it skips the free instead,
//! leaking the window's MPI-side state while leaving the caller's buffer
//! untouched. `Mpi::drop` still runs `MPI_Finalize` for a `Win::create`
//! window (caller-owned memory; unlike `Win::allocate`/`SharedWindow`, Open
//! MPI does not free it inside `MPI_Finalize` itself).
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_create_at_finalize
// mpi-test: np=1 skip-ok=openmpi-4
// mpi-test-stderr: MPI_Win_free skipped for 1 window

use ferrompi::{Error, Mpi, MpiErrorClass, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let mut buf = vec![1.5f64; 64];

    let win = match Win::create(&world, &mut buf) {
        Ok(win) => win,
        Err(Error::Mpi {
            class: MpiErrorClass::Win,
            ..
        }) => {
            common::skip(
                &world,
                "Win::create returned MPI_ERR_WIN — likely OpenMPI 4.x \
                 with a BTL that does not support one-sided over caller-owned \
                 memory (e.g., --btl=self,tcp in CI).",
            );
            return;
        }
        Err(e) => panic!("Win::create failed: {e}"),
    };

    drop(mpi);
    drop(win);

    assert!(
        buf.iter().all(|&x| x == 1.5),
        "buf must be untouched: finalize must skip MPI_Win_free for the live window"
    );

    println!("test_window_create_at_finalize: PASS");
}
