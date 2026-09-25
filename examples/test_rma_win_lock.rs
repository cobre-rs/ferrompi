//! Integration test for Win passive-target lock/unlock epoch helpers.
//!
//! Verifies that `Win::lock`, `Win::lock_all`, `WinLockGuard::flush`, and
//! `WinLockAllGuard::flush` / `flush_all` work correctly without issuing any
//! RMA data operations. Data-movement tests are deferred to ticket-034 / 057–058.
//!
//! Test matrix (all on a 2-rank world):
//!
//! 1. **Shared lock on rank 0** — every rank acquires `LockType::Shared` on
//!    rank 0, calls `guard.flush()`, then drops the guard (→ unlock).
//! 2. **Exclusive self-lock** — rank 0 acquires `LockType::Exclusive` on
//!    itself, then drops the guard.
//! 3. **lock_all** — all ranks call `lock_all()`, invoke `guard.flush(0)`
//!    and `guard.flush_all()`, then drop the guard (→ unlock_all).
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_lock

use ferrompi::{LockType, Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_win_lock requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Allocate an 8-element f64 window on every rank.
    // ========================================================================
    let win = match Win::<f64>::allocate(&world, 8) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: Win::allocate failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            return;
        }
    };

    // Opening fence to satisfy MPI epoch rules before passive-target use.
    if let Err(e) = win.fence(WinFenceAssert::none()) {
        eprintln!("rank {rank}: FAIL: initial fence failed: {e}");
        local_ok = false;
    }

    // ========================================================================
    // Test 1: Shared lock on rank 0, flush, then unlock (via Drop).
    //
    // Every rank acquires a shared lock on rank 0's window. Because shared
    // locks allow multiple concurrent holders, this is valid even when all
    // ranks execute the block simultaneously.
    // ========================================================================
    {
        let guard = match win.lock(LockType::Shared, 0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: Win::lock(Shared, 0) failed: {e}");
                local_ok = false;
                // Still need to participate in test-2 and test-3 barriers, so
                // we synthesise a dummy guard path by jumping ahead.
                // Use a sentinel allreduce and return.
                let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                return;
            }
        };

        if let Err(e) = guard.flush() {
            eprintln!("rank {rank}: FAIL: WinLockGuard::flush failed: {e}");
            local_ok = false;
        }
        // guard drops here → MPI_Win_unlock(0, win)
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock with LockGuard");
    }

    // ========================================================================
    // Test 2: Exclusive self-lock (rank 0 locks its own window).
    //
    // Only rank 0 performs the lock. The other ranks skip but wait at the
    // barrier so the test progresses collectively.
    // ========================================================================
    if rank == 0 {
        let guard = match win.lock(LockType::Exclusive, 0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: Win::lock(Exclusive, 0) failed: {e}");
                local_ok = false;
                let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                return;
            }
        };

        if let Err(e) = guard.flush() {
            eprintln!("rank {rank}: FAIL: WinLockGuard::flush (exclusive) failed: {e}");
            local_ok = false;
        }
        // guard drops here → MPI_Win_unlock(0, win)
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock Exclusive self-lock");
    }

    // ========================================================================
    // Test 3: lock_all — shared locks on all ranks simultaneously.
    //
    // All ranks call lock_all(), issue flush(0) and flush_all(), then drop
    // the guard (→ unlock_all).
    // ========================================================================
    {
        let guard = match win.lock_all() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: Win::lock_all failed: {e}");
                local_ok = false;
                let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                return;
            }
        };

        if let Err(e) = guard.flush(0) {
            eprintln!("rank {rank}: FAIL: WinLockAllGuard::flush(0) failed: {e}");
            local_ok = false;
        }

        if let Err(e) = guard.flush_all() {
            eprintln!("rank {rank}: FAIL: WinLockAllGuard::flush_all failed: {e}");
            local_ok = false;
        }
        // guard drops here → MPI_Win_unlock_all(win)
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock_all with WinLockAllGuard");
    }

    common::check(&world, local_ok, "test_rma_win_lock");
}
