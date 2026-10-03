//! Dropping `Mpi` at `ThreadLevel::Multiple` while persistent requests are
//! active on other threads skips `MPI_Finalize` and does not wait for them.
//!
//! A request is movable, so whichever thread holds it, an active request must
//! make the drop skip, and the drop must not wait for it: the thread that holds
//! the request may itself be waiting for the drop.
//!
//! - **Worker-started, init-dropped.** A worker takes a self pair by move, starts
//!   both requests and hands them back to the init thread still active. That the
//!   pair crosses threads at all is the compile-time proof that
//!   `PersistentRequest<i32>` is `Send`.
//! - **Worker-held.** A second worker starts a receive nothing will match,
//!   signals, and keeps it until the init thread announces that the drop
//!   returned.
//!
//! The init thread then drops `Mpi`. The drop returns without waiting for either
//! worker's request, which a local check on the elapsed time pins (under 5 s). It
//! prints the skip line, which the runner requires, and MPI stays initialized and
//! not finalized. Afterwards:
//! - the init thread drops the handed-back pair, which leaks it without an MPI
//!   call, although another thread started it;
//! - the second worker tests its receive, which is a completion call and is
//!   still admitted (`Ok(None)`: nothing matched), and then drops it, which
//!   leaks it without an MPI call.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused by
//! then. A failed check prints `FAIL:` and exits with status 2, which the runner
//! reports as a failure; status 1 would be accepted under `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_persistent_finalize_multiple
// mpi-test: np=1 expect=unfinalized
// mpi-test-stderr: persistent request(s) still active

use std::sync::mpsc;
use std::time::{Duration, Instant};

use ferrompi::{Mpi, PersistentRequest, Source, Tag, ThreadLevel};

mod common;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");

    // Some builds (e.g. certain Cray MPT configurations) deliberately refuse it.
    if mpi.thread_level() < ThreadLevel::Multiple {
        common::skip(
            &mpi.world(),
            &format!(
                "MPI provided {:?}, MPI_THREAD_MULTIPLE required; skipping test",
                mpi.thread_level()
            ),
        );
        return;
    }

    let world = mpi.world();
    let pair = [
        world
            .recv_init(vec![0i32; 4], 0, 10)
            .expect("recv_init failed"),
        world
            .send_init(vec![1i32, 2, 3, 4], 0, 10)
            .expect("send_init failed"),
    ];
    let unmatched = world
        .recv_init(vec![0u8; 8], Source::Any, Tag::Value(99))
        .expect("recv_init (unmatched) failed");
    let (started, started_rx) = mpsc::channel();
    let (dropped, dropped_rx) = mpsc::channel::<()>();

    let (pair, elapsed) = std::thread::scope(|t| {
        let starter = t.spawn(move || {
            let mut pair = pair;
            for request in &mut pair {
                request.start().expect("start (worker) failed");
            }
            pair
        });
        t.spawn(move || {
            let mut never = unmatched;
            never.start().expect("start (unmatched) failed");
            started.send(()).expect("init thread hung up");
            dropped_rx
                .recv()
                .expect("init thread hung up before the drop returned");
            check(never.is_active(), "the held receive is still active");
            check(
                matches!(never.test(), Ok(None)),
                "test is admitted after the drop and finds nothing",
            );
            drop(never);
        });

        let pair: [PersistentRequest<i32>; 2] = starter.join().expect("starter panicked");
        started_rx.recv().expect("worker hung up before starting");
        check(
            pair.iter().all(PersistentRequest::is_active),
            "the handed-back pair is active",
        );

        let begin = Instant::now();
        drop(mpi);
        let elapsed = begin.elapsed();
        dropped.send(()).expect("worker hung up");
        (pair, elapsed)
    });

    check(
        elapsed < Duration::from_secs(5),
        "Mpi::drop returned without waiting for a worker's request",
    );
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    check(
        pair.iter().all(PersistentRequest::is_active),
        "the skipped finalize leaves the handed-back pair active",
    );
    drop(pair);

    println!("test_persistent_finalize_multiple: PASS");
}
