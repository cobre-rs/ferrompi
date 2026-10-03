//! Integration test for Win PSCW (post/start/wait) active-target epoch helpers.
//!
//! Verifies that the PSCW epoch methods — `Win::post`, `Win::start` (whose
//! closure ends with `MPI_Win_complete`), and `Win::wait_exposure` — correctly
//! open and close epochs. Tests 1 and 2 exercise the epoch helpers alone (no
//! RMA data operations); test 3 issues a real `Win::put` inside a
//! `WinPscwAssert::no_check()` epoch on its own window and checks the
//! transferred data; test 4 returns an error from the access closure after a
//! put and checks that the epoch is still closed and the put delivered.
//!
//! Rank 0 acts as the *target* (exposure side): calls `post` then `wait_exposure`.
//! Rank 1 acts as the *origin* (access side): calls `start` with a closure.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_pscw
// mpi-test: np=2

use ferrompi::{Communicator, Error, Group, Mpi, ReduceOp, Win, WinPscwAssert};

mod common;

fn peer_group(world: &Communicator, peer: i32) -> Group {
    world
        .group()
        .and_then(|g| g.include(&[peer]))
        .expect("peer group failed")
}

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
    let mut win = match Win::<f64>::allocate(&world, 8) {
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

        // Open an access epoch to rank 0's window and close it again (no
        // actual RMA data operations in this test).
        if let Err(e) = win.start(&exposure_group, WinPscwAssert::default(), |_| Ok(())) {
            eprintln!("rank {rank}: FAIL: Win::start failed: {e}");
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

        if let Err(e) = win.start(&exposure_group, WinPscwAssert::default(), |_| Ok(())) {
            eprintln!("rank {rank}: FAIL: Win::start [test 2] failed: {e}");
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
    // Uses its own freshly allocated window rather than `win`: Open MPI 4.1's
    // osc/sm mis-completes a NOCHECK epoch on a window reused across PSCW
    // epochs.
    //
    // MPI_MODE_NOCHECK requires the matching post to have already completed
    // before the paired start, hence the barrier between post and start.
    //
    // Rank 0 (target): post(no_check) -> barrier -> wait_exposure -> check data
    // Rank 1 (origin):  barrier -> start(no_check) -> put -> complete
    // ========================================================================
    {
        let mut win3 = Win::<f64>::allocate(&world, 4).expect("Win::allocate [test 3] failed");

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

            if let Err(e) = win3.post(&access_group, WinPscwAssert::no_check()) {
                eprintln!("rank {rank}: FAIL: Win::post(no_check) failed: {e}");
                local_ok = false;
            }

            world
                .barrier()
                .expect("barrier between post and start [test 3] failed");

            if let Err(e) = win3.wait_exposure() {
                eprintln!("rank {rank}: FAIL: Win::wait_exposure [test 3] failed: {e}");
                local_ok = false;
            }

            let expected = [1.0f64, 2.0, 3.0, 4.0];
            if win3.local_slice()[0..4] != expected {
                eprintln!(
                    "rank {rank}: FAIL: expected {expected:?}, got {:?}",
                    &win3.local_slice()[0..4]
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

            let buf = [1.0f64, 2.0, 3.0, 4.0];
            let epoch = win3.start(&exposure_group, WinPscwAssert::no_check(), |e| {
                e.put(&buf, 0, 0, buf.len() as i64)
            });
            if let Err(e) = epoch {
                eprintln!("rank {rank}: FAIL: Win::start(no_check) with put failed: {e}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: PSCW epoch with WinPscwAssert::no_check() and put");
    }

    // ========================================================================
    // Test 4: the access closure returns an Err after its put.
    //
    // `start` returns the closure's error, and the closing MPI_Win_complete
    // still runs: the target's wait_exposure returns (it would block forever
    // on an epoch left open) and the put before the error is delivered.
    // Uses its own freshly allocated window, as test 3 does.
    //
    // Rank 0 (target): post -> wait_exposure -> check data
    // Rank 1 (origin): start(put, then put to a rank that does not exist)
    // ========================================================================
    {
        let mut win4 = Win::<f64>::allocate(&world, 4).expect("Win::allocate [test 4] failed");

        if rank == 0 {
            if let Err(e) = win4.post(&peer_group(&world, 1), WinPscwAssert::default()) {
                eprintln!("rank {rank}: FAIL: Win::post [test 4] failed: {e}");
                local_ok = false;
            }

            if let Err(e) = win4.wait_exposure() {
                eprintln!("rank {rank}: FAIL: Win::wait_exposure [test 4] failed: {e}");
                local_ok = false;
            }

            let expected = [5.0f64, 6.0, 7.0, 8.0];
            if win4.local_slice() != expected {
                eprintln!(
                    "rank {rank}: FAIL: [test 4] expected {expected:?}, got {:?}",
                    win4.local_slice()
                );
                local_ok = false;
            }
        } else {
            let buf = [5.0f64, 6.0, 7.0, 8.0];
            let epoch = win4.start(&peer_group(&world, 0), WinPscwAssert::default(), |e| {
                e.put(&buf, 0, 0, buf.len() as i64)?;
                e.put(&buf, 2, 0, buf.len() as i64)
            });
            if !matches!(
                epoch,
                Err(Error::InvalidArgument {
                    arg: "target_rank",
                    ..
                })
            ) {
                eprintln!("rank {rank}: FAIL: Win::start [test 4] returned {epoch:?}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 4 failed");
    if rank == 0 && local_ok {
        println!("PASS: PSCW epoch closes after the access closure returns an error");
    }

    common::check(&world, local_ok, "test_rma_win_pscw");
}
