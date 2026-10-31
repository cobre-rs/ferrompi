//! A matched message that is not received keeps `Mpi` from finalizing, and it
//! can still be received after `Mpi` was dropped, at `ThreadLevel::Funneled`.
//!
//! The process sends itself two messages, with buffered sends, and matches both.
//! - The first `Message` moves to another thread, which may not call MPI below
//!   `Serialized`. Its `mrecv` there is refused with
//!   `Error::ThreadLevelViolation`, and the drop that follows makes no MPI call
//!   and does not abort: the message stays matched, unreceived and counted.
//! - The second `Message` stays on the main thread across the drop of `Mpi`.
//!
//! The drop of `Mpi` skips `MPI_Finalize` and counts both messages. MPI stays
//! initialized, so the second message is then received with its data: a
//! matched receive completes something that is already matched. A new probe
//! is refused with `Error::Finalized`.
//!
//! No check after the drop can use `common::check`, whose allreduce is refused by
//! then. A failed check prints `FAIL:` and exits with status 2, which the runner
//! reports as a failure; status 1 would be accepted under `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_mprobe_finalize
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: 2 persistent request(s) still active or matched message(s) not yet received

use ferrompi::{Error, Mpi, ThreadLevel};

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
    let rank = world.rank();

    // A buffered send completes without a receive, so the process can send to
    // itself and then leave the matched messages unreceived.
    mpi.buffer_attach(vec![0u8; 4096].into_boxed_slice())
        .expect("buffer_attach failed");
    let second = [4i32, 5];
    for (data, tag) in [(vec![1i32, 2, 3], 1), (second.to_vec(), 2)] {
        let mut send = world
            .bsend_init(data, rank, tag)
            .expect("bsend_init failed");
        send.start().expect("start failed");
        send.wait().expect("wait failed");
    }
    let leaked = world.mprobe::<i32>(rank, 1);
    let kept = world.mprobe::<i32>(rank, 2);
    check(
        leaked.is_ok() && kept.is_ok(),
        "two self-sent messages are matched",
    );
    let (Ok((leaked, _)), Ok((kept, _))) = (leaked, kept) else {
        return;
    };

    let refused = std::thread::scope(|threads| {
        threads
            .spawn(move || {
                let mut buf = [0i32; 3];
                matches!(leaked.mrecv(&mut buf), Err(Error::ThreadLevelViolation))
            })
            .join()
            .expect("the receiving thread panicked")
    });
    check(
        refused,
        "mrecv on another thread is refused below Serialized",
    );
    check(
        world.barrier().is_ok(),
        "the main thread still calls MPI after the refused receive",
    );

    drop(mpi);

    let mut buf = [0i32; 2];
    let received = kept.mrecv(&mut buf);
    check(
        matches!(&received, Ok(st) if st.count == Some(2)) && buf == second,
        "a matched message is received after Mpi was dropped",
    );
    check(
        matches!(world.improbe::<i32>(rank, 3), Err(Error::Finalized)),
        "a new matched probe is refused after Mpi was dropped",
    );

    println!("PASS: test_mprobe_finalize");
}
