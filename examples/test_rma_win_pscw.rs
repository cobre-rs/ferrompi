//! Integration test for Win PSCW (post/start/complete/wait) active-target epoch helpers.
//!
//! Verifies that the four PSCW epoch methods — `Win::post`, `Win::start`,
//! `Win::complete`, and `Win::wait_exposure` — correctly open and close epochs
//! without issuing any RMA data operations. Data-movement tests are deferred to
//! the RMA data-op tickets (ticket-034 / ticket-057 / ticket-058).
//!
//! Rank 0 acts as the *target* (exposure side): calls `post` then `wait_exposure`.
//! Rank 1 acts as the *origin* (access side): calls `start` then `complete`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_pscw
// mpi-test: np=2

use ferrompi::{Mpi, ReduceOp, Win, WinPscwAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_rma_win_pscw requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Allocate an 8-element f64 window on both ranks.
    // ========================================================================
    let win = match Win::<f64>::allocate(&world, 8) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: Win::allocate failed: {e}");
            // Bail out rather than hang — reduce will propagate the failure.
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            return;
        }
    };

    // ========================================================================
    // Test 1: Basic PSCW epoch open/close (WinPscwAssert::default())
    //
    // Rank 0 (target): post → wait_exposure
    // Rank 1 (origin): start → complete
    //
    // No data movement — we are only exercising epoch helpers.
    // ========================================================================
    if rank == 0 {
        // Build the access group: ranks that will call Win::start against us.
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
                world.abort(1);
            }
        };
        let access_group = match world_group.include(&[1]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([1]) failed: {e}");
                world.abort(1);
            }
        };

        // Expose our window to rank 1.
        if let Err(e) = win.post(&access_group, WinPscwAssert::default()) {
            eprintln!("rank {rank}: FAIL: Win::post failed: {e}");
            local_ok = false;
        }

        // Wait for rank 1 to complete its access epoch.
        if let Err(e) = win.wait_exposure() {
            eprintln!("rank {rank}: FAIL: Win::wait_exposure failed: {e}");
            local_ok = false;
        }
    } else {
        // rank == 1: access side
        // Build the exposure group: ranks that will call Win::post.
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
                world.abort(1);
            }
        };
        let exposure_group = match world_group.include(&[0]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([0]) failed: {e}");
                world.abort(1);
            }
        };

        // Open access epoch to rank 0's window.
        if let Err(e) = win.start(&exposure_group, WinPscwAssert::default()) {
            eprintln!("rank {rank}: FAIL: Win::start failed: {e}");
            local_ok = false;
        }

        // Close the access epoch (no actual RMA data operations in this test).
        if let Err(e) = win.complete() {
            eprintln!("rank {rank}: FAIL: Win::complete failed: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: PSCW epoch (post/start/complete/wait)");
    }

    // ========================================================================
    // Test 2: Win::test_exposure — poll until exposure epoch closes.
    //
    // Rank 0 (target): post → spin on test_exposure until true
    // Rank 1 (origin): start → complete
    // ========================================================================
    if rank == 0 {
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() [test 2] failed: {e}");
                world.abort(1);
            }
        };
        let access_group = match world_group.include(&[1]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([1]) [test 2] failed: {e}");
                world.abort(1);
            }
        };

        if let Err(e) = win.post(&access_group, WinPscwAssert::default()) {
            eprintln!("rank {rank}: FAIL: Win::post [test 2] failed: {e}");
            local_ok = false;
        }

        // Poll until rank 1 has completed its access epoch.
        let mut done = false;
        while !done {
            match win.test_exposure() {
                Ok(flag) => done = flag,
                Err(e) => {
                    eprintln!("rank {rank}: FAIL: Win::test_exposure failed: {e}");
                    local_ok = false;
                    break;
                }
            }
        }
    } else {
        // rank == 1
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() [test 2] failed: {e}");
                world.abort(1);
            }
        };
        let exposure_group = match world_group.include(&[0]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([0]) [test 2] failed: {e}");
                world.abort(1);
            }
        };

        if let Err(e) = win.start(&exposure_group, WinPscwAssert::default()) {
            eprintln!("rank {rank}: FAIL: Win::start [test 2] failed: {e}");
            local_ok = false;
        }

        if let Err(e) = win.complete() {
            eprintln!("rank {rank}: FAIL: Win::complete [test 2] failed: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::test_exposure (nonblocking poll)");
    }

    // ========================================================================
    // Test 3: WinPscwAssert::no_check() with an actual put.
    //
    // MPI_MODE_NOCHECK requires the matching post to have already completed
    // before the paired start, hence the barrier between post and start.
    //
    // Rank 0 (target): post(no_check) -> barrier -> wait_exposure -> check data
    // Rank 1 (origin):  barrier -> start(no_check) -> put -> complete
    // ========================================================================
    if rank == 0 {
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() [test 3] failed: {e}");
                world.abort(1);
            }
        };
        let access_group = match world_group.include(&[1]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([1]) [test 3] failed: {e}");
                world.abort(1);
            }
        };

        if let Err(e) = win.post(&access_group, WinPscwAssert::no_check()) {
            eprintln!("rank {rank}: FAIL: Win::post(no_check) failed: {e}");
            local_ok = false;
        }

        world
            .barrier()
            .expect("barrier between post and start [test 3] failed");

        if let Err(e) = win.wait_exposure() {
            eprintln!("rank {rank}: FAIL: Win::wait_exposure [test 3] failed: {e}");
            local_ok = false;
        }

        let expected = [1.0f64, 2.0, 3.0, 4.0];
        if win.local_slice()[0..4] != expected {
            eprintln!(
                "rank {rank}: FAIL: expected {expected:?}, got {:?}",
                &win.local_slice()[0..4]
            );
            local_ok = false;
        }
    } else {
        // rank == 1
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: world.group() [test 3] failed: {e}");
                world.abort(1);
            }
        };
        let exposure_group = match world_group.include(&[0]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL: group.include([0]) [test 3] failed: {e}");
                world.abort(1);
            }
        };

        world
            .barrier()
            .expect("barrier between post and start [test 3] failed");

        if let Err(e) = win.start(&exposure_group, WinPscwAssert::no_check()) {
            eprintln!("rank {rank}: FAIL: Win::start(no_check) failed: {e}");
            local_ok = false;
        }

        let buf = [1.0f64, 2.0, 3.0, 4.0];
        if let Err(e) = win.put(&buf, 0, 0, buf.len() as i64) {
            eprintln!("rank {rank}: FAIL: Win::put [test 3] failed: {e}");
            local_ok = false;
        }

        if let Err(e) = win.complete() {
            eprintln!("rank {rank}: FAIL: Win::complete [test 3] failed: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: PSCW epoch with WinPscwAssert::no_check() and put");
    }

    common::check(&world, local_ok, "test_rma_win_pscw");
}
