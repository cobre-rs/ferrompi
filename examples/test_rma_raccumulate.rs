//! Integration test for `Win::raccumulate` — request-returning one-sided
//! accumulate with local-completion semantics.
//!
//! Verifies that:
//!
//! 1. Rank 1 initializes its local window to `[1, 2, 3, 4]` and participates
//!    in a fence to make the write visible.
//! 2. Rank 0 acquires an exclusive passive-target lock on rank 1, calls
//!    `Win::raccumulate` with `ReduceOp::Sum`, waits on the returned `Request`
//!    (local completion — origin buffer safe to reuse), then drops the lock
//!    guard (remote completion — accumulation visible at rank 1).
//! 3. After a barrier, rank 1 reads its local window and asserts it equals
//!    `[11, 22, 33, 44]`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_raccumulate

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
    // After a barrier, rank 1 asserts its window equals [11, 22, 33, 44].
    //
    // Protocol:
    //   Rank 1: initialize window to [1, 2, 3, 4]
    //   All:    fence (write phase — makes rank 1's init visible)
    //   Rank 0: lock(Exclusive, rank=1)
    //           → raccumulate(&[10, 20, 30, 40], 1, 0, 4, ReduceOp::Sum)
    //           → req.wait()  (local completion — origin buffer safe to reuse)
    //           → drop guard  (remote completion — visible at rank 1)
    //   All:    barrier
    //   Rank 1: assert local window == [11, 22, 33, 44]
    //   All:    sentinel allreduce before exit
    // ========================================================================
    const N: usize = 4;
    let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");

    // Rank 1 initializes its local window slice.
    if rank == 1 {
        win.local_slice_mut().copy_from_slice(&[1i32, 2, 3, 4]);
    }

    // Active-target fence: make rank 1's initialization visible to all.
    win.fence(WinFenceAssert::default())
        .expect("fence (write phase) failed");

    if rank == 0 {
        let buf = [10i32, 20, 30, 40];

        // Acquire exclusive passive-target lock on rank 1's window.
        let guard = match win.lock(LockType::Exclusive, 1) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank 0: FAIL: Win::lock(Exclusive, 1) failed: {e}");
                local_ok = false;
                let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                return;
            }
        };

        // Post raccumulate — returns a Request signaling local completion.
        let count = buf.len() as i64;
        let req = match win.raccumulate(&buf, 1, 0, count, ReduceOp::Sum) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rank 0: FAIL: Win::raccumulate returned error: {e}");
                local_ok = false;
                drop(guard);
                let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                return;
            }
        };

        // Wait for local completion — origin buffer (`buf`) is safe to
        // reuse after this point. Remote visibility requires unlock below.
        if let Err(e) = req.wait() {
            eprintln!("rank 0: FAIL: Request::wait after raccumulate failed: {e}");
            local_ok = false;
        }

        // Drop guard issues MPI_Win_unlock, completing the epoch and
        // guaranteeing the accumulation is visible at rank 1.
        drop(guard);
    }

    // Barrier: ensure rank 0 has unlocked before rank 1 reads its window.
    world.barrier().expect("barrier failed");

    // Rank 1 verifies its local window contains the accumulated result.
    if rank == 1 {
        let expected = [11i32, 22, 33, 44];
        let local = win.local_slice();
        if local != expected {
            eprintln!(
                "FAIL: rank 1 window after raccumulate Sum: expected {expected:?}, got {local:?}"
            );
            local_ok = false;
        }
    }

    if rank == 0 && local_ok {
        println!("PASS: Win::raccumulate Sum with local completion");
    }

    common::check(&world, local_ok, "test_rma_raccumulate");
}
