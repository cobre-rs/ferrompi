//! A persistent request dropped while active, whose completion fails, keeps
//! counting as active, so `Mpi`'s drop skips `MPI_Finalize`.
//!
//! The example starts a 2-element persistent self send, then a 1-element
//! persistent receive that matches it, so the receive completes with
//! `MPI_ERR_TRUNCATE` (errors are returned, not fatal). The send starts first
//! because Open MPI (4.1.6 and 5.0.7 observed) reports no truncation when a
//! self send meets a receive that is already posted, and reports it when the
//! receive is started over a message that is waiting; a persistent send, unlike
//! a blocking one, does not wait for MPICH to match it. The sent message is 8 bytes: a truncation test
//! must keep the sent size at or below 4096 bytes, because Open MPI 5.0.7
//! corrupts the heap when a truncated message exceeds the shared-memory eager
//! limit and CMA single-copy is on.
//!
//! 1. A control pair, completed with `wait`, shows that this library reports
//!    the truncation and leaves the receive inactive.
//! 2. The receive under test is dropped while active. `Drop` does not assume
//!    that a failed wait completed the request: its raw `MPI_Wait` fails, so
//!    it leaks the handle and the buffer and keeps the token. Its send is
//!    completed first, so that the receive is the only request left counting.
//!    `Mpi`'s drop then skips `MPI_Finalize` and prints `persistent request(s)
//!    still active`, which the runner requires; a receive that released its
//!    token would let the finalize run.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_persistent_failed_wait
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: persistent request(s) still active

use ferrompi::{Communicator, Mpi, MpiErrorClass, PersistentRequest, ThreadLevel};

mod common;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

/// A started 1-element receive and the started 2-element send it truncates.
fn truncating_pair(
    world: &Communicator,
    tag: i32,
) -> (PersistentRequest<i32>, PersistentRequest<i32>) {
    let mut recv = world
        .recv_init(vec![0i32; 1], 0, tag)
        .expect("recv_init failed");
    let mut send = world
        .send_init(vec![1i32, 2], 0, tag)
        .expect("send_init failed");
    send.start().expect("start (send) failed");
    recv.start().expect("start (recv) failed");
    (recv, send)
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Funneled).expect("MPI init failed");
    check(
        mpi.thread_level() == ThreadLevel::Funneled,
        "MPI provides the level asked for",
    );
    let world = mpi.world();

    let (mut control, mut control_send) = truncating_pair(&world, 1);
    let result = control.wait();
    common::check(
        &world,
        common::class_of(&result) == Some(MpiErrorClass::Truncate),
        "a 2-element message into a 1-element persistent receive fails with Truncate",
    );
    common::check(
        &world,
        !control.is_active(),
        "the failed wait leaves the control receive inactive",
    );
    control_send.wait().expect("wait (control send) failed");

    let (request, mut send) = truncating_pair(&world, 2);
    common::check(&world, request.is_active(), "the receive is active");
    drop(request);
    send.wait().expect("wait (send) failed");
    common::check(
        &world,
        !send.is_active(),
        "the send completed, so only the receive is left counting",
    );

    drop(mpi);
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    println!("test_persistent_failed_wait: PASS");
}
