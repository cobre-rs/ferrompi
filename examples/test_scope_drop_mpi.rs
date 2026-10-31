//! Dropping `Mpi` inside a nonblocking scope, at `ThreadLevel::Funneled`,
//! skips `MPI_Finalize`, and the scope still completes its requests.
//!
//! The scope holds a self `irecv`/`isend` pair and the closure drops `Mpi`. The
//! drop does not call `MPI_Finalize` because a scope is open on the thread that
//! drops (there is no window in this process, so the scope is the only reason),
//! and MPI stays initialized and not finalized (MPI-4.1 §11.2.2). The scope-end
//! `MPI_Waitall` that follows is therefore a valid call: it completes both
//! requests, the buffer holds the message after the scope, and every call that
//! is not a completion returns `Error::Finalized`.
//!
//! The scope also holds a receive that nothing will match. After the drop the
//! closure cancels it: a cancel must be admitted while finalizing, or the
//! scope-end wait on that receive would never return. A refused cancel fails
//! the example at once, through the local check.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_scope_drop_mpi
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: MPI_Finalize skipped: a nonblocking scope is still open

use ferrompi::{Error, Mpi, Source, Tag, ThreadLevel};

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

    let data = [6u8; 64];
    let mut buf = vec![0u8; 64];
    let mut unmatched = [9u8; 8];
    let result = ferrompi::scope(|s| {
        world.irecv(s, &mut buf, 0, 6)?;
        world.isend(s, &data, 0, 6)?;
        let mut never = world.irecv(s, &mut unmatched, Source::Any, Tag::Value(99))?;
        drop(mpi);
        check(never.cancel().is_ok(), "cancel is admitted after the drop");
        Ok(())
    });

    check(result.is_ok(), "the scope returns Ok after the dropped Mpi");
    check(
        buf == data,
        "scope end completes the receive after the drop",
    );
    check(unmatched == [9u8; 8], "the cancelled receive wrote nothing");
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    check(
        matches!(world.barrier(), Err(Error::Finalized)),
        "a call that is not a completion returns Finalized",
    );
    println!("test_scope_drop_mpi: PASS");
}
