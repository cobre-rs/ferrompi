//! Integration test for `PassiveEpoch::raccumulate` — request-returning
//! one-sided accumulate with local-completion semantics.
//!
//! Verifies that:
//!
//! 1. Rank 1 initializes its local window to `[1, 2, 3, 4]`; a fence with
//!    `no_succeed` and a barrier make the write visible before rank 0 locks.
//! 2. Rank 0 locks rank 1 exclusively and calls `PassiveEpoch::raccumulate`
//!    with `ReduceOp::Sum` inside the closure, waits on the returned `Request`
//!    (local completion — MPI no longer reads the origin buffer), then the
//!    closure ends and the unlock gives remote completion — accumulation
//!    visible at rank 1.
//! 3. After a barrier, rank 1 locks its own window, and once the lock is
//!    released it reads the window and asserts it equals `[11, 22, 33, 44]`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_raccumulate
// mpi-test: np=2

use ferrompi::{LockType, Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_raccumulate requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test: rank 1 initialises its window to [1, 2, 3, 4]; rank 0 locks rank 1
    // (Exclusive), raccumulates [10, 20, 30, 40] with ReduceOp::Sum, waits on
    // the Request (local completion), then unlocks (remote completion).
    // After a barrier, rank 1 locks its own window (an empty epoch) and, once
    // it is unlocked, asserts the window equals [11, 22, 33, 44].
    //
    // Protocol:
    //   Rank 1: initialize window to [1, 2, 3, 4]
    //   All:    fence(no_succeed) + barrier (rank 1's init visible before the lock)
    //   Rank 0: lock(Exclusive, rank=1, |e|
    //             raccumulate(&[10, 20, 30, 40], 1, 0, ReduceOp::Sum)
    //             → req.wait()  (local completion — MPI no longer reads the origin))
    //           → unlock  (remote completion — visible at rank 1)
    //   All:    barrier
    //   Rank 1: lock(Shared, 1, |_| Ok(())) → assert local window == [11, 22, 33, 44]
    //   All:    `common::check` verdict
    // ========================================================================
    const N: usize = 4;
    let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");

    // Rank 1 initializes its local window slice.
    if rank == 1 {
        win.local().copy_from_slice(&[1i32, 2, 3, 4]);
    }

    // Make rank 1's stores visible in its window; `no_succeed` says no fence
    // epoch follows. The barrier orders the stores before rank 0's lock.
    win.fence(WinFenceAssert::none(), WinFenceAssert::no_succeed(), |_| {
        Ok(())
    })
    .expect("fence failed");
    world.barrier().expect("barrier before lock failed");

    if rank == 0 {
        let buf = [10i32, 20, 30, 40];

        // Exclusive passive-target lock on rank 1's window; the unlock when the
        // closure ends completes the epoch and guarantees the accumulation is
        // visible at rank 1.
        let locked = win.lock(LockType::Exclusive, 1, |e| {
            let req = match e.raccumulate(&buf, 1, 0, ReduceOp::Sum) {
                Ok(r) => r,
                Err(err) => {
                    eprintln!("rank 0: FAIL: PassiveEpoch::raccumulate returned error: {err}");
                    world.abort(1);
                }
            };

            // Wait for local completion — MPI no longer reads the origin buffer
            // (`buf`) after this point. Remote visibility requires the unlock.
            if let Err(err) = req.wait() {
                eprintln!("rank 0: FAIL: Request::wait after raccumulate failed: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank 0: FAIL: Win::lock(Exclusive, 1) failed: {e}");
            world.abort(1);
        }
    }

    // Barrier: ensure rank 0 has unlocked before rank 1 reads its window.
    world.barrier().expect("barrier failed");

    // Rank 1 verifies its local window contains the accumulated result.
    if rank == 1 {
        let expected = [11i32, 22, 33, 44];
        // Locking its own window makes rank 0's update visible to this rank's
        // loads, and nothing writes the window after the unlock.
        win.lock(LockType::Shared, 1, |_| Ok(()))
            .expect("lock(Shared, 1) failed");
        let mut local = [0i32; N];
        win.local().copy_to_slice(&mut local);
        if local != expected {
            eprintln!(
                "FAIL: rank 1 window after raccumulate Sum: expected {expected:?}, got {local:?}"
            );
            local_ok = false;
        }
    }

    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::raccumulate Sum with local completion");
    }

    common::check(&world, local_ok, "test_rma_raccumulate");
}
