//! Integration test for Win passive-target lock/unlock epoch closures.
//!
//! Verifies that `Win::lock` and `Win::lock_all` run their closure in an
//! epoch, that `PassiveEpoch::flush` / `flush_all` work inside it without
//! issuing any RMA data operations, and that the epoch is unlocked whether the
//! closure returns, returns an error or panics.
//!
//! Test matrix (all on a 2-rank world):
//!
//! 1. **Shared lock on rank 0** — every rank locks rank 0 with
//!    `LockType::Shared`, calls `e.flush(0)`, then the closure ends
//!    (→ unlock).
//! 2. **Exclusive self-lock** — rank 0 locks itself with `LockType::Exclusive`.
//! 3. **lock_all** — all ranks call `lock_all`, invoke `e.flush(0)` and
//!    `e.flush_all()`, then the closure ends (→ unlock_all).
//! 4. **Unlock after an error** — rank 0's closure returns an error; the lock
//!    is released, so locking rank 0 exclusively again succeeds.
//! 5. **Unlock on unwind** — rank 0's closure panics; the panic reaches the
//!    caller and the lock is released, so locking rank 0 exclusively again
//!    succeeds. A second exclusive lock on a window still locked would fail or
//!    hang.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_lock
// mpi-test: np=2

use std::panic::{catch_unwind, AssertUnwindSafe};

use ferrompi::{Error, LockType, Mpi, ReduceOp, Win, WinFenceAssert};

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
    // Test 1: Shared lock on rank 0, flush, then unlock when the closure ends.
    //
    // Every rank acquires a shared lock on rank 0's window. Because shared
    // locks allow multiple concurrent holders, this is valid even when all
    // ranks execute the block simultaneously.
    // ========================================================================
    {
        let locked = win.lock(LockType::Shared, 0, |e| {
            if let Err(err) = e.flush(0) {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush failed: {err}");
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
        println!("PASS: Win::lock with PassiveEpoch");
    }

    // ========================================================================
    // Test 2: Exclusive self-lock (rank 0 locks its own window).
    //
    // Only rank 0 performs the lock. The other ranks skip but wait at the
    // barrier so the test progresses collectively.
    // ========================================================================
    if rank == 0 {
        let locked = win.lock(LockType::Exclusive, 0, |e| {
            if let Err(err) = e.flush(0) {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush (exclusive) failed: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank {rank}: FAIL: Win::lock(Exclusive, 0) failed: {e}");
            world.abort(1);
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock Exclusive self-lock");
    }

    // ========================================================================
    // Test 3: lock_all — shared locks on all ranks simultaneously.
    //
    // All ranks call lock_all, issue flush(0) and flush_all(), then the
    // closure ends (→ unlock_all).
    // ========================================================================
    {
        let locked = win.lock_all(|e| {
            if let Err(err) = e.flush(0) {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush(0) failed: {err}");
                local_ok = false;
            }

            if let Err(err) = e.flush_all() {
                eprintln!("rank {rank}: FAIL: PassiveEpoch::flush_all failed: {err}");
                local_ok = false;
            }
            Ok(())
        });
        if let Err(e) = locked {
            eprintln!("rank {rank}: FAIL: Win::lock_all failed: {e}");
            world.abort(1);
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock_all with PassiveEpoch");
    }

    // ========================================================================
    // Test 4: the lock is released when the closure returns an error.
    //
    // The put names a rank outside the window's communicator, so it is rejected
    // before any MPI call and the closure returns that error. Locking rank 0
    // exclusively again succeeds only if the first lock was released.
    // ========================================================================
    if rank == 0 {
        let buf = [1.0f64];
        let epoch = win.lock(LockType::Exclusive, 0, |e| e.put(&buf, size, 0));
        if !matches!(
            epoch,
            Err(Error::InvalidArgument {
                arg: "target_rank",
                ..
            })
        ) {
            eprintln!("rank {rank}: FAIL: Win::lock [test 4] returned {epoch:?}");
            local_ok = false;
        }
        if let Err(e) = win.lock(LockType::Exclusive, 0, |e| e.flush(0)) {
            eprintln!("rank {rank}: FAIL: the lock was not released after an error: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 4 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock unlocks after the closure returns an error");
    }

    // ========================================================================
    // Test 5: the lock is released when the closure panics.
    //
    // The panic is caught around the `lock` call, so it must reach that far.
    // Locking rank 0 exclusively again succeeds only if the unwind released
    // the first lock.
    // ========================================================================
    if rank == 0 {
        // The panic is the test's input; keep its message out of the output.
        std::panic::set_hook(Box::new(|_| {}));
        let unwound = catch_unwind(AssertUnwindSafe(|| {
            win.lock(LockType::Exclusive, 0, |_| -> ferrompi::Result<()> {
                panic!("unwind test")
            })
        }));
        drop(std::panic::take_hook());
        if unwound.is_ok() {
            eprintln!("rank {rank}: FAIL: the panic did not reach the caller of Win::lock");
            local_ok = false;
        }
        if let Err(e) = win.lock(LockType::Exclusive, 0, |e| e.flush(0)) {
            eprintln!("rank {rank}: FAIL: the lock was not released on unwind: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 5 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::lock unlocks and keeps unwinding when the closure panics");
    }

    common::check(&world, local_ok, "test_rma_win_lock");
}
