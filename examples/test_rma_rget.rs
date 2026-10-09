//! Integration test for `PassiveEpoch::rget` — request-returning one-sided read.
//!
//! Verifies that:
//!
//! 1. Rank 1 initializes its local window to `[100, 200, 300, 400]`; a
//!    fence with `no_succeed` and a barrier make the write visible before
//!    rank 0 locks.
//! 2. Rank 0 locks rank 1 with a shared passive-target lock, calls
//!    `PassiveEpoch::rget` inside the closure to post a read, waits on the
//!    returned `Request` (local completion), then the closure ends and the
//!    window is unlocked. The local buffer stays borrowed until then, so rank 0
//!    reads the fetched data after the closure returns.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_rget
// mpi-test: np=2

use ferrompi::{LockType, Mpi, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_rget requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: rank 0 rget [100, 200, 300, 400] from rank 1's window.
    //
    // Protocol:
    //   Rank 1: initialize window to [100, 200, 300, 400]
    //   All:    fence(no_succeed) + barrier (rank 1's init visible before the lock)
    //   Rank 0: lock(Shared, rank=1, |e| rget(&mut buf, 1, 0, 4) → req.wait())
    //           → unlock → assert buf == [100, 200, 300, 400]
    //   All:    `common::check` verdict
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");

        // Rank 1 initializes its local window slice.
        if rank == 1 {
            let local = win.local_slice_mut();
            local[0] = 100;
            local[1] = 200;
            local[2] = 300;
            local[3] = 400;
        }

        // Make rank 1's stores visible in its window; `no_succeed` says no fence
        // epoch follows. The barrier orders the stores before rank 0's lock.
        win.fence(WinFenceAssert::none(), WinFenceAssert::no_succeed(), |_| {
            Ok(())
        })
        .expect("fence failed");
        world.barrier().expect("barrier before lock failed");

        if rank == 0 {
            let mut local_buf = [0i32; N];

            // Shared passive-target lock on rank 1's window.
            let locked = win.lock(LockType::Shared, 1, |e| {
                // Post the rget — request completes when local buffer contains data.
                let req = match e.rget(&mut local_buf, 1, 0) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("rank 0: FAIL: PassiveEpoch::rget returned error: {err}");
                        world.abort(1);
                    }
                };

                // Wait for local completion — the data is in local_buf after
                // this, which stays borrowed until the unlock.
                if let Err(err) = req.wait() {
                    eprintln!("rank 0: FAIL: Request::wait after rget failed: {err}");
                    local_ok = false;
                }
                Ok(())
            });
            if let Err(e) = locked {
                eprintln!("rank 0: FAIL: Win::lock(Shared, 1) failed: {e}");
                world.abort(1);
            }

            // Verify the fetched data.
            if local_ok {
                let expected = [100i32, 200, 300, 400];
                if local_buf != expected {
                    eprintln!(
                        "FAIL: rank 0 local_buf after rget: expected {expected:?}, got {local_buf:?}"
                    );
                    local_ok = false;
                }
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: PassiveEpoch::rget with local completion");
    }

    common::check(&world, local_ok, "test_rma_rget");
}
