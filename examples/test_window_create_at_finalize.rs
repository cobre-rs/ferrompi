//! Regression for a live `Win::create` window at finalize.
//!
//! `Mpi::drop` skips `MPI_Finalize` while the `Win::create` window below is
//! still alive, so the window's buffer is left untouched: the window is
//! never freed and `MPI_Finalize` is never called for this process. Calling
//! `into_buffer` after the drop is refused with `Error::Finalized` and leaks
//! the buffer.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_window_create_at_finalize
// mpi-test: np=1 expect=unfinalized skip-ok=openmpi-4
// mpi-test-stderr: MPI_Finalize skipped

use ferrompi::{Error, Mpi, MpiErrorClass, Win};

mod common;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    let win = match Win::create(&world, vec![1.5f64; 64]) {
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

    check(
        win.local_slice().iter().all(|&x| x == 1.5),
        "buffer must be untouched: finalize must skip MPI_Win_free for the live window",
    );
    check(
        matches!(win.into_buffer(), Err(Error::Finalized)),
        "into_buffer must be refused with Error::Finalized once Mpi has been dropped",
    );

    println!("test_window_create_at_finalize: PASS");
}
