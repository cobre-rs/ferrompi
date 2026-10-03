//! Dropping `Mpi` while persistent requests are active, at `ThreadLevel::Funneled`,
//! skips `MPI_Finalize`, and the requests can still be completed afterwards.
//!
//! The example creates:
//! - an idle `send_init`, never started. A `start` or `start_all` on it from
//!   another thread is refused with `Error::ThreadLevelViolation`;
//! - a self pair started with `start` and never completed;
//! - a second self pair started with `start_all` and never completed;
//! - when the library has persistent collectives, an `allreduce_init` with a
//!   user op whose `UserOp` handle is dropped right after the init, started and
//!   never completed. Only the request keeps the op alive, and the op's closure
//!   reports when it is dropped.
//!
//! The drop does not call `MPI_Finalize`, because requests are active (there is
//! no window or scope in this process, so they are the only reason), and MPI
//! stays initialized and not finalized (MPI-4.1 §11.2.2). A `wait` or `test` that
//! follows is therefore a valid call:
//! - the first pair completes, the receive holds the sent data, and its
//!   `into_buffer` returns it;
//! - the allreduce completes, and its op's closure was not dropped, so the
//!   finalize sweep that frees user ops never ran;
//! - the idle request cannot be started, because starting is not a completion
//!   and returns `Error::Finalized`;
//! - the second pair is dropped while still active, which leaks it without an
//!   MPI call.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused by
//! then. A failed check prints `FAIL:` and exits with status 2, which the runner
//! reports as a failure; status 1 would be accepted under `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_persistent_finalize
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: persistent request(s) still active

use std::sync::atomic::{AtomicUsize, Ordering};

use ferrompi::{Error, Mpi, PersistentRequest, ThreadLevel, UserOp};

mod common;

static CLOSURES_DROPPED: AtomicUsize = AtomicUsize::new(0);

struct CountsDrop;

impl Drop for CountsDrop {
    fn drop(&mut self) {
        CLOSURES_DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

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

    let mut idle = world
        .send_init(vec![7i32], 0, 1)
        .expect("send_init (idle) failed");

    // A start from another thread is refused below `Serialized`, and counts
    // nothing.
    let refused_off_thread = std::thread::scope(|t| {
        t.spawn(|| {
            matches!(idle.start(), Err(Error::ThreadLevelViolation))
                && matches!(
                    PersistentRequest::start_all(std::slice::from_mut(&mut idle)),
                    Err(Error::ThreadLevelViolation)
                )
                && !idle.is_active()
        })
        .join()
        .expect("worker thread panicked")
    });
    check(
        refused_off_thread,
        "start and start_all are refused off the init thread",
    );

    let mut recv_a = world
        .recv_init(vec![0i32; 4], 0, 10)
        .expect("recv_init a failed");
    let mut send_a = world
        .send_init(vec![1i32, 2, 3, 4], 0, 10)
        .expect("send_init a failed");
    recv_a.start().expect("start a (recv) failed");
    send_a.start().expect("start a (send) failed");

    let recv_b = world
        .recv_init(vec![0i32; 4], 0, 11)
        .expect("recv_init b failed");
    let send_b = world
        .send_init(vec![5i32, 6, 7, 8], 0, 11)
        .expect("send_init b failed");
    let mut pair_b = [recv_b, send_b];
    PersistentRequest::start_all(&mut pair_b).expect("start_all b failed");

    let mut reduction = None;
    if common::has_mpi4_collectives() {
        let guard = CountsDrop;
        let op: UserOp<i32> = UserOp::new(move |invec: &[i32], inoutvec: &mut [i32]| {
            let _keep = &guard;
            for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
                *y += *x;
            }
        })
        .expect("UserOp::new failed");
        let mut request = world
            .allreduce_init(vec![3i32; 2], vec![0i32; 2], &op)
            .expect("allreduce_init failed");
        drop(op);
        request.start().expect("start (allreduce) failed");
        reduction = Some(request);
    } else {
        println!("test_persistent_finalize: no persistent collectives, skipping the allreduce");
    }
    check(
        CLOSURES_DROPPED.load(Ordering::SeqCst) == 0,
        "the request keeps the user op's closure alive",
    );

    drop(mpi);

    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    check(
        matches!(idle.start(), Err(Error::Finalized)),
        "start is refused after the drop",
    );
    check(
        recv_a.is_active() && send_a.is_active(),
        "the skipped finalize leaves the first pair active",
    );
    check(
        send_a.wait().is_ok() && recv_a.wait().is_ok(),
        "wait completes the first pair after the drop",
    );
    check(
        !recv_a.is_active() && recv_a.buffer().is_ok_and(|b| b == [1, 2, 3, 4]),
        "the receive holds the sent data",
    );
    check(
        recv_a.into_buffer().is_ok_and(|b| b == [1, 2, 3, 4]),
        "into_buffer returns the received data with no MPI call",
    );
    if let Some(mut request) = reduction {
        check(
            request.wait().is_ok(),
            "wait completes the allreduce after the drop",
        );
        check(
            request.buffer().is_ok_and(|b| b == [3, 3]),
            "the allreduce delivered its result",
        );
        check(
            CLOSURES_DROPPED.load(Ordering::SeqCst) == 0,
            "the finalize sweep never ran: the user op's closure is still registered",
        );
    }
    check(
        pair_b.iter().all(PersistentRequest::is_active),
        "the second pair is still active",
    );
    drop(pair_b);

    println!("test_persistent_finalize: PASS");
}
