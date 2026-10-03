//! Dropping `Mpi` at `ThreadLevel::Multiple` while a worker thread is inside a
//! nonblocking scope waits for that scope to end before it finalizes. The
//! worker posts an `ibarrier` and a self `irecv`/`isend` pair and leaves all
//! three to the scope end; the init thread drops `Mpi` right after the worker
//! signals. The drop must return only after the worker's scope returned `Ok`
//! with the message delivered, and MPI must really finalize, which the runner
//! checks by the absence of the `MPI_Finalize skipped` warning. The worker's
//! scope-end `MPI_Waitall` therefore runs while `Mpi::drop` is finalizing, with
//! four live slots.
//!
//! The worker also posts a receive that nothing will match, waits until
//! `Mpi::is_finalized()` reports the drop, and cancels it. The cancel must be
//! admitted while finalizing: refused, the scope could never end, and the drop
//! would wait for it forever.
//!
//! A second worker polls `test()` on pending self `irecv`s until the init thread
//! announces the drop. It then posts the matching `isend`s, and the drop begins
//! only after the last one is posted, because a request cannot be posted once
//! the drop stored `Finalizing`. The worker keeps testing the receives, and the
//! drop starts after a random delay of up to 100 us, so it lands inside one of
//! those calls on some runs. A completion call that read the active state just
//! before the store must still be admitted: no completion call on either worker
//! may return `Err(Finalized)`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_scope_finalize
// mpi-test: np=1

use std::hash::{BuildHasher, RandomState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ferrompi::{Error, Mpi, Source, Tag, ThreadLevel};

mod common;

const POLLED: usize = 1024;

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
    let (posted, posted_rx) = mpsc::channel();
    let (polling, polling_rx) = mpsc::channel();
    let dropping = AtomicBool::new(false);
    let sent = AtomicBool::new(false);
    let message = [7.0f64; 4];
    let polled_message = [8.0f64; 4];

    let (scope_result, delivered, admitted, drop_start, drop_end) = std::thread::scope(|t| {
        let worker = t.spawn(|| {
            let mut delivered = [0.0f64; 4];
            let mut unmatched = [0u8; 1];
            let result = ferrompi::scope(|s| {
                world.ibarrier(s)?;
                world.irecv(s, &mut delivered, 0, 7)?;
                world.isend(s, &message, 0, 7)?;
                let mut never = world.irecv(s, &mut unmatched, Source::Any, Tag::Value(99))?;
                posted.send(()).expect("init thread hung up");
                while !Mpi::is_finalized() {
                    std::thread::yield_now();
                }
                if never.cancel().is_err() {
                    // The scope would wait for the receive forever, so no
                    // return path can report the failure.
                    eprintln!("FAIL: cancel was refused while Mpi began to drop");
                    std::process::exit(2);
                }
                Ok(Instant::now())
            });
            (result, delivered)
        });

        let poller = t.spawn(|| {
            let mut received = vec![[0.0f64; 4]; POLLED];
            ferrompi::scope(|s| {
                let mut requests = received
                    .iter_mut()
                    .map(|buf| world.irecv(s, buf, 0, 8))
                    .collect::<ferrompi::Result<Vec<_>>>()?;
                polling.send(()).expect("init thread hung up");
                let mut admitted = true;
                while !dropping.load(Ordering::Acquire) {
                    for request in &mut requests {
                        admitted &= matches!(request.test(), Ok(None));
                    }
                }
                for _ in 0..POLLED {
                    world.isend(s, &polled_message, 0, 8)?;
                }
                sent.store(true, Ordering::Release);
                for request in &mut requests {
                    loop {
                        match request.test() {
                            Ok(None) => {}
                            Ok(Some(_)) => break,
                            Err(Error::Finalized) => {
                                admitted = false;
                                break;
                            }
                            Err(e) => panic!("test failed: {e}"),
                        }
                    }
                }
                Ok(admitted)
            })
        });

        posted_rx.recv().expect("worker hung up before posting");
        polling_rx.recv().expect("poller hung up before posting");
        dropping.store(true, Ordering::Release);
        while !sent.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        let delay = Duration::from_nanos(RandomState::new().hash_one(()) % 100_000);
        let start = Instant::now();
        while start.elapsed() < delay {
            std::hint::spin_loop();
        }
        let drop_start = Instant::now();
        drop(mpi);
        let drop_end = Instant::now();

        let (scope_result, delivered) = worker.join().expect("worker panicked");
        let admitted = poller.join().expect("poller panicked");
        (scope_result, delivered, admitted, drop_start, drop_end)
    });

    let scope_end = scope_result.expect("the worker's scope must return Ok");
    assert!(
        drop_start < scope_end,
        "the worker's scope ended before Mpi::drop started"
    );
    assert!(
        drop_end >= scope_end,
        "Mpi::drop returned before the worker's scope ended"
    );
    assert_eq!(
        delivered, message,
        "the worker's scope end must deliver the message"
    );
    assert!(
        admitted.expect("the poller's scope must return Ok"),
        "a completion call was refused while Mpi began to drop"
    );
    assert!(
        Mpi::is_finalized(),
        "Mpi::is_finalized() must be true after the drop"
    );

    println!("test_scope_finalize: PASS");
}
