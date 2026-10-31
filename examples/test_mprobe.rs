//! Matched probes and matched receives: `mprobe`, `improbe`, `Message::mrecv`
//! and the discard of a dropped `Message`, at `ThreadLevel::Multiple`.
//!
//! Rank 1 sends and rank 0 receives; every other rank only joins the checks.
//!
//! 1. `improbe` with nothing queued returns `None`.
//! 2. `mprobe(Source::Any, Tag::Any)` reports the source, the tag and the
//!    element count, and `mrecv` receives the data with the same status.
//! 3. `improbe` returns a queued message, and `mrecv` receives it.
//! 4. A probe of `Source::ProcNull` returns at once. Its message receives
//!    nothing and leaves the buffer alone, and one that is dropped is discarded
//!    like any other.
//! 5. Dropping a `Message` receives it: the synchronous send it matched
//!    completes, and the next message with that tag is the one sent after it.
//! 6. An `mrecv` into a buffer that is too small returns class `Truncate`.
//! 7. Four threads match and receive 64 messages of eight lengths. Each thread
//!    receives the message it matched: the count and every element agree with
//!    what its probe reported.
//!
//! The process then finalizes MPI. A `Message` that kept its count would make
//! the drop of `Mpi` skip `MPI_Finalize`, which the runner reports as a
//! failure.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_mprobe
// mpi-test: np=2.. valgrind

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ferrompi::{Mpi, MpiErrorClass, Source, Status, Tag, ThreadLevel};

mod common;

const PATIENCE: Duration = Duration::from_secs(20);
const MESSAGES: usize = 64;

fn is(status: &Status, source: Source, tag: Tag, count: usize) -> bool {
    status.source == source && status.tag == tag && status.count == Some(count)
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    assert!(
        world.size() >= 2,
        "test_mprobe requires at least 2 processes, got {}",
        world.size()
    );
    common::check(
        &world,
        mpi.thread_level() == ThreadLevel::Multiple,
        "MPI provides ThreadLevel::Multiple",
    );

    // 1. Nothing is queued yet: rank 1 sends only after this check.
    let nothing = rank != 0 || matches!(world.improbe::<f64>(1, 7), Ok(None));
    common::check(&world, nothing, "improbe with nothing queued returns None");

    // 2. mprobe with wildcards, then mrecv.
    let payload = [1.5f64, 2.5, 3.5, 4.5];
    let mut ok = true;
    if rank == 1 {
        world.send(&payload, 0, 7).expect("send failed");
    } else if rank == 0 {
        let (message, probed) = world
            .mprobe::<f64>(Source::Any, Tag::Any)
            .expect("mprobe failed");
        let mut buf = [0.0f64; 4];
        let received = message.mrecv(&mut buf);
        ok = is(&probed, Source::Rank(1), Tag::Value(7), 4)
            && matches!(&received, Ok(st) if is(st, Source::Rank(1), Tag::Value(7), 4))
            && buf == payload;
    }
    common::check(&world, ok, "mprobe reports the message that mrecv receives");

    // 3. improbe until the message is there, then mrecv.
    let mut ok = true;
    if rank == 1 {
        world.send(&[1i32, 2, 3], 0, 8).expect("send failed");
    } else if rank == 0 {
        let deadline = Instant::now() + PATIENCE;
        let mut matched = None;
        while matched.is_none() && Instant::now() < deadline {
            matched = world.improbe::<i32>(1, 8).expect("improbe failed");
        }
        ok = match matched {
            Some((message, probed)) => {
                let mut buf = [0i32; 3];
                is(&probed, Source::Rank(1), Tag::Value(8), 3)
                    && message.mrecv(&mut buf).is_ok()
                    && buf == [1, 2, 3]
            }
            None => false,
        };
    }
    common::check(&world, ok, "improbe returns a queued message");

    // 4. Source::ProcNull: a message with no data, on every rank.
    let (message, probed) = world
        .mprobe::<f64>(Source::ProcNull, Tag::Any)
        .expect("mprobe of ProcNull failed");
    let mut untouched = [7.0f64; 2];
    let received = message.mrecv(&mut untouched);
    let dropped = world.improbe::<f64>(Source::ProcNull, 3);
    let ok = is(&probed, Source::ProcNull, Tag::Any, 0)
        && matches!(&received, Ok(st) if is(st, Source::ProcNull, Tag::Any, 0))
        && untouched == [7.0, 7.0]
        && matches!(&dropped, Ok(Some((_, st))) if is(st, Source::ProcNull, Tag::Any, 0));
    drop(dropped);
    common::check(
        &world,
        ok,
        "a probe of ProcNull matches a message of no data",
    );

    // 5. A dropped message is received: the synchronous send completes. The
    // send request lives until after the check: dropped while still active, it
    // would wait for a receive that a failed discard never makes.
    let mut ok = true;
    let mut synchronous = None;
    if rank == 1 {
        let request = synchronous.insert(
            world
                .ssend_init(vec![9i32; 4], 0, 9)
                .expect("ssend_init failed"),
        );
        request.start().expect("start failed");
        world.send(&[5i32], 0, 9).expect("send failed");
        let deadline = Instant::now() + PATIENCE;
        let mut completed = false;
        while !completed && Instant::now() < deadline {
            completed = request.test().expect("test failed").is_some();
        }
        ok = completed;
    } else if rank == 0 {
        let (message, probed) = world.mprobe::<i32>(1, 9).expect("mprobe failed");
        drop(message);
        let mut next = [0i32; 1];
        let received = world.recv(&mut next, 1, 9);
        ok = is(&probed, Source::Rank(1), Tag::Value(9), 4) && received.is_ok() && next == [5];
    }
    common::check(&world, ok, "dropping a Message receives it");
    drop(synchronous);

    // 6. A buffer that is too small.
    let mut ok = true;
    if rank == 1 {
        world.send(&common::PAYLOAD, 0, 10).expect("send failed");
    } else if rank == 0 {
        let (message, _) = world.mprobe::<i32>(1, 10).expect("mprobe failed");
        let mut small = [0i32; 1];
        ok = common::class_of(&message.mrecv(&mut small)) == Some(MpiErrorClass::Truncate);
    }
    common::check(&world, ok, "mrecv into a short buffer returns Truncate");

    // 7. Four threads, each receiving exactly the message it matched.
    let mut ok = true;
    if rank == 1 {
        for i in 0..MESSAGES {
            let len = 1 + i % 8;
            world
                .send(&vec![len as i32; len], 0, 20)
                .expect("send failed");
        }
    } else if rank == 0 {
        let received = AtomicUsize::new(0);
        let wrong = AtomicUsize::new(0);
        let deadline = Instant::now() + PATIENCE;
        std::thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    while received.load(Ordering::SeqCst) < MESSAGES && Instant::now() < deadline {
                        let Ok(matched) = world.improbe::<i32>(1, 20) else {
                            wrong.fetch_add(1, Ordering::SeqCst);
                            return;
                        };
                        let Some((message, probed)) = matched else {
                            std::thread::yield_now();
                            continue;
                        };
                        let len = probed.count.unwrap_or(0);
                        let mut buf = vec![0i32; len];
                        let good = matches!(message.mrecv(&mut buf), Ok(st) if st.count == Some(len))
                            && len > 0
                            && buf.iter().all(|&v| v == len as i32);
                        if !good {
                            wrong.fetch_add(1, Ordering::SeqCst);
                        }
                        received.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        });
        ok = received.load(Ordering::SeqCst) == MESSAGES && wrong.load(Ordering::SeqCst) == 0;
    }
    common::check(&world, ok, "each thread receives the message it matched");

    if rank == 0 {
        println!("PASS: test_mprobe");
    }
}
