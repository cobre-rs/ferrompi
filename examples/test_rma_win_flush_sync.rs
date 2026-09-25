//! Integration test for Win::flush_local / flush_local_all / sync helpers.
//!
//! Verifies that the three local-completion helpers introduced in ticket-056
//! compile and run correctly without issuing any RMA data operations.
//! Data-movement tests are deferred to tickets 034 / 057–062.
//!
//! Test matrix (all on a 2-rank world):
//!
//! 1. **flush_local inside a shared lock epoch** — every rank acquires
//!    `LockType::Shared` on rank 0, calls `win.flush_local(0)`, then drops
//!    the guard (→ unlock).
//! 2. **flush_local_all and sync inside a lock_all epoch** — all ranks call
//!    `lock_all()`, invoke `win.flush_local_all()` and `win.sync()`, then drop
//!    the guard (→ unlock_all).
//!    Note: `MPI_Win_sync` is defined by the MPI standard as a local operation
//!    valid at any point, but MPICH 4.2.x enforces it only within a
//!    passive-target epoch. The test therefore calls `sync` inside `lock_all`
//!    to be portable across all conformant implementations.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_flush_sync
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
        "test_rma_win_flush_sync requires at least 2 processes, got {size}"
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
    // Test 1: flush_local inside a shared lock epoch on rank 0.
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
                world.abort(1);
            }
        };

        if let Err(e) = win.flush_local(0) {
            eprintln!("rank {rank}: FAIL: Win::flush_local(0) failed: {e}");
            local_ok = false;
        }
        // guard drops here → MPI_Win_unlock(0, win)
        drop(guard);
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::flush_local inside shared lock epoch");
    }

    // ========================================================================
    // Test 2: flush_local_all and sync inside a lock_all epoch.
    //
    // All ranks call lock_all(), invoke flush_local_all() and sync(), then
    // drop the guard (→ unlock_all).
    //
    // Note: MPI_Win_sync is a local memory barrier that the MPI standard
    // defines as valid outside any epoch, but MPICH 4.2.x rejects it unless
    // a passive-target epoch is active. The test therefore calls sync() inside
    // the lock_all epoch to be portable across all conformant implementations.
    // ========================================================================
    {
        let guard = match win.lock_all() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: Win::lock_all failed: {e}");
                world.abort(1);
            }
        };

        if let Err(e) = win.flush_local_all() {
            eprintln!("rank {rank}: FAIL: Win::flush_local_all failed: {e}");
            local_ok = false;
        }

        if let Err(e) = win.sync() {
            eprintln!("rank {rank}: FAIL: Win::sync failed: {e}");
            local_ok = false;
        }
        // guard drops here → MPI_Win_unlock_all(win)
        drop(guard);
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::flush_local_all and Win::sync inside lock_all epoch");
    }

    common::check(&world, local_ok, "test_rma_win_flush_sync");
}
