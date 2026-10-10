//! Integration test for PassiveEpoch::flush_local / flush_local_all / sync.
//!
//! Verifies that the three local-completion helpers compile and run
//! correctly without issuing any RMA data operations, and that `flush` and
//! `flush_local` reject a rank outside the window's communicator before any
//! MPI call.
//!
//! Test matrix (all on a 2-rank world):
//!
//! 1. **flush_local inside a shared lock epoch** — every rank locks rank 0 with
//!    `LockType::Shared`, calls `e.flush_local(0)`, then the closure ends
//!    (→ unlock).
//! 2. **flush_local_all and sync inside a lock_all epoch** — all ranks call
//!    `lock_all`, invoke `e.flush_local_all()` and `e.sync()`, then the closure
//!    ends (→ unlock_all).
//!    Note: the MPI standard allows `MPI_Win_sync` only within a passive-target
//!    epoch, so the test calls `sync` inside `lock_all`.
//! 3. **A rank outside the communicator** — inside `lock_all`, `e.flush` and
//!    `e.flush_local` return `Error::InvalidArgument { arg: "rank" }` for the
//!    communicator size and for -1, and the epoch still works afterwards.
//!    Open MPI accepts such a rank, and with the pt2pt component reads past its
//!    peer table; MPICH returns `MPI_ERR_RANK`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_flush_sync
// mpi-test: np=2

use ferrompi::{Error, LockType, Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

fn is_rank_rejected(result: &ferrompi::Result<()>) -> bool {
    matches!(result, Err(Error::InvalidArgument { arg: "rank", .. }))
}

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
    let mut win = match Win::<f64>::allocate(&world, 8) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: Win::allocate failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            return;
        }
    };

    // `no_succeed` says no fence epoch follows; the barrier puts every rank's
    // fence before any rank's lock.
    if let Err(e) = win.fence(WinFenceAssert::none(), WinFenceAssert::no_succeed(), |_| {
        Ok(())
    }) {
        eprintln!("rank {rank}: FAIL: initial fence failed: {e}");
        local_ok = false;
    }
    if let Err(e) = world.barrier() {
        eprintln!("rank {rank}: FAIL: barrier before lock failed: {e}");
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
        let locked = win.lock(LockType::Shared, 0, |e| {
            if let Err(err) = e.flush_local(0) {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush_local(0) failed: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank {rank}: FAIL: Win::lock(Shared, 0) failed: {e}");
            world.abort(1);
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::flush_local inside shared lock epoch");
    }

    // ========================================================================
    // Test 2: flush_local_all and sync inside a lock_all epoch.
    //
    // All ranks call lock_all, invoke flush_local_all() and sync(), then the
    // closure ends (→ unlock_all).
    //
    // Note: the MPI standard allows MPI_Win_sync only within a passive-target
    // epoch, so the test calls sync() inside the lock_all epoch.
    // ========================================================================
    {
        let locked = win.lock_all(|e| {
            if let Err(err) = e.flush_local_all() {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush_local_all failed: {err}");
                local_ok = false;
            }

            if let Err(err) = e.sync() {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::sync failed: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank {rank}: FAIL: Win::lock_all failed: {e}");
            world.abort(1);
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::flush_local_all and sync inside lock_all epoch");
    }

    // ========================================================================
    // Test 3: a rank outside the window's communicator is rejected before any
    // MPI call, so the epoch is unchanged and still flushes.
    // ========================================================================
    {
        let locked = win.lock_all(|e| {
            for bad in [size, -1] {
                let flushed = e.flush(bad);
                if !is_rank_rejected(&flushed) {
                    eprintln!("rank {rank}: FAIL: PassiveEpoch::flush({bad}) returned {flushed:?}");
                    local_ok = false;
                }
                let flushed = e.flush_local(bad);
                if !is_rank_rejected(&flushed) {
                    eprintln!(
                        "rank {rank}: FAIL: PassiveEpoch::flush_local({bad}) returned {flushed:?}"
                    );
                    local_ok = false;
                }
            }
            if let Err(err) = e.flush(0) {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush(0) after a rejected rank: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank {rank}: FAIL: Win::lock_all [test 3] failed: {e}");
            world.abort(1);
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::flush and flush_local reject a rank outside the group");
    }

    common::check(&world, local_ok, "test_rma_win_flush_sync");
}
