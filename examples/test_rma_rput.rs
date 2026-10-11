//! Integration test for `PassiveEpoch::rput` — request-returning one-sided write.
//!
//! Verifies that:
//!
//! 1. Rank 0 locks rank 1 exclusively, calls `PassiveEpoch::rput` inside the
//!    closure to post a write of `[10, 20, 30, 40]` into rank 1's window, waits
//!    on the returned `Request` (local completion), then the closure ends and
//!    the unlock gives remote completion. After a barrier, rank 1 locks its
//!    own window, reads it and asserts the correct values.
//! 2. A request the closure never waits is completed by the epoch before it
//!    unlocks, whether the closure returns `Ok`, returns an error or panics.
//!    Rank 0 runs many `lock` epochs per exit, each posting one `rput` it
//!    never waits. That the epoch completed the request is checked by the
//!    `checks_that_need_mpi` unit test in `src/lib.rs`, which can read the
//!    request's table slot.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_rput
// mpi-test: np=2

use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use ferrompi::{LockType, Mpi, Win};

mod common;

/// Epochs per exit in test 2.
const ROUNDS: usize = 300;

#[derive(Clone, Copy)]
enum Exit {
    Ok,
    Err,
    Panic,
}

/// Runs `ROUNDS` `lock` epochs on rank 1, each posting one `rput` that it never
/// waits and ending as `exit` says, and returns whether every epoch ended that
/// way.
fn pending_rput_epochs(win: &mut Win<i32>, buf: &[i32; 4], exit: Exit) -> bool {
    let mut ended_as_asked = true;
    for _ in 0..ROUNDS {
        let ended = catch_unwind(AssertUnwindSafe(|| {
            win.lock(LockType::Exclusive, 1, |e| {
                let _req = e.rput(buf, 1, 0)?;
                match exit {
                    Exit::Ok => Ok(()),
                    // Rank 99 is outside the window's communicator.
                    Exit::Err => e.rput(buf, 99, 0).map(|_| ()),
                    Exit::Panic => panic!("closure panics with a request pending"),
                }
            })
        }));
        ended_as_asked &= matches!(
            (exit, ended),
            (Exit::Ok, Ok(Ok(()))) | (Exit::Err, Ok(Err(_))) | (Exit::Panic, Err(_))
        );
    }
    ended_as_asked
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_rput requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: rank 0 rput [10, 20, 30, 40] into rank 1's window.
    //
    // Protocol:
    //   Rank 0: lock(Exclusive, rank=1, |e| rput → req.wait()) → unlock
    //   Rank 1: barrier → lock(Shared, 1) → read local window → assert [10, 20, 30, 40]
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");

        if rank == 0 {
            let buf = [10i32, 20, 30, 40];

            // Exclusive passive-target lock on rank 1's window; the unlock when
            // the closure ends triggers remote completion — rank 1 will see the
            // write after this point.
            let locked = win.lock(LockType::Exclusive, 1, |e| {
                // Post the rput — request completes when local buffer is safe to reuse.
                let req = match e.rput(&buf, 1, 0) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("rank 0: FAIL: PassiveEpoch::rput returned error: {err}");
                        world.abort(1);
                    }
                };

                if let Err(err) = req.wait() {
                    eprintln!("rank 0: FAIL: Request::wait after rput failed: {err}");
                    local_ok = false;
                }
                Ok(())
            });
            if let Err(e) = locked {
                eprintln!("rank 0: FAIL: Win::lock(Exclusive, 1) failed: {e}");
                world.abort(1);
            }
        }

        // All ranks synchronize here, so rank 1 reads after rank 0's unlock
        // has completed.
        world.barrier().expect("barrier after rput failed");

        if rank == 1 {
            let expected = [10i32, 20, 30, 40];
            let mut local = [0i32; N];
            // Locking its own window makes rank 0's update visible to this
            // rank's loads, and nothing writes the window after the unlock.
            win.lock(LockType::Shared, 1, |e| {
                e.local().copy_to_slice(&mut local);
                Ok(())
            })
            .expect("lock(Shared, 1) failed");
            if local != expected {
                eprintln!(
                    "FAIL: rank 1 local window after rput: expected {expected:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::rput with local completion");
    }

    // ========================================================================
    // Test 2: a request left pending is completed by its epoch, whatever the
    // closure does. Rank 1 waits in the barrier below, which makes the MPI
    // progress rank 0's unlocks may need.
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 4).expect("Win::allocate failed");
        if rank == 0 {
            let buf = [10i32, 20, 30, 40];
            set_hook(Box::new(|_| {}));
            for (exit, name) in [
                (Exit::Ok, "returns Ok"),
                (Exit::Err, "returns an error"),
                (Exit::Panic, "panics"),
            ] {
                if !pending_rput_epochs(&mut win, &buf, exit) {
                    eprintln!("FAIL: a closure that {name} did not end its epoch that way");
                    local_ok = false;
                }
            }
            drop(take_hook());
        }
        world.barrier().expect("barrier after test 2 failed");
    }

    common::check(&world, local_ok, "test_rma_rput");
}
