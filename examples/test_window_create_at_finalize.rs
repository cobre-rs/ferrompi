//! Regression for a live `Win::create` window at finalize.
//!
//! `Mpi::drop` skips `MPI_Finalize` while the `Win::create` window below is
//! still alive, so the caller-supplied buffer is left untouched: the window
//! is never freed and `MPI_Finalize` is never called for this process.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_create_at_finalize
// mpi-test: np=1 expect=unfinalized skip-ok=openmpi-4
// mpi-test-stderr: MPI_Finalize skipped

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
