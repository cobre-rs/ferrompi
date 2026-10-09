//! Dropping `Mpi` inside a passive-target epoch, at `ThreadLevel::Funneled`,
//! skips `MPI_Finalize`, and the epoch still completes and closes.
//!
//! The closure of `lock_all` puts into the window and drops `Mpi`. The drop
//! does not call `MPI_Finalize` because the window is alive: an open epoch
//! holds no token of its own, and the live-window count alone selects the skip
//! (MPI-4.1 §11.2.2). MPI stays initialized, so the closing `MPI_Win_unlock_all`
//! is a valid call that completes the put (MPI-4.1 §12.5.3), while every call
//! that is not a completion returns `Error::Finalized`, including the
//! `flush_all` right after the drop. Freeing the window is refused the same
//! way, and the buffer it owns is leaked with no MPI call.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_rma_epoch_drop_mpi
// mpi-test: np=1 expect=unfinalized valgrind skip-ok=openmpi-4
// mpi-test-stderr: MPI_Finalize skipped: 1 window(s) still alive

use ferrompi::{Error, Mpi, MpiErrorClass, ThreadLevel, Win};

mod common;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Funneled).expect("MPI init failed");
    check(
        mpi.thread_level() == ThreadLevel::Funneled,
        "MPI provides the level asked for",
    );
    let world = mpi.world();

    // Open MPI 4 refuses `Win::create` over caller memory.
    let mut win = match Win::create(&world, vec![0u64; 4]) {
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

    let data = [7u64; 4];
    let r = win.lock_all(|e| {
        e.put(&data, 0, 0)?;
        drop(mpi);
        Ok(e.flush_all())
    });

    check(
        matches!(r, Ok(Err(Error::Finalized))),
        "lock_all returns Ok, and a call that is not a completion is refused inside it",
    );
    check(
        win.local_slice() == [7u64; 4],
        "the closing unlock completed the put after the drop",
    );
    check(
        matches!(win.into_buffer(), Err(Error::Finalized)),
        "into_buffer is refused with Finalized, leaking the buffer",
    );
    println!("test_rma_epoch_drop_mpi: PASS");
}
