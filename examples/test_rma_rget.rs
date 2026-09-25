//! Integration test for `Win::rget` — request-returning one-sided read.
//!
//! Verifies that:
//!
//! 1. Rank 1 initializes its local window to `[100, 200, 300, 400]` and
//!    participates in a fence to make the write visible.
//! 2. Rank 0 acquires a shared passive-target lock on rank 1, calls
//!    `Win::rget` to post a read, waits on the returned `Request` (local
//!    completion), then drops the lock guard. After `req.wait()` returns,
//!    the local buffer already contains the fetched data.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_rget

use ferrompi::{LockType, Mpi, ReduceOp, Win};

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
    //   All:    fence (write phase — makes rank 1's init visible)
    //   Rank 0: lock(Shared, rank=1) → rget(&mut buf, 1, 0, 4)
    //           → req.wait() → assert buf == [100, 200, 300, 400]
    //           → drop guard (unlock)
    //   All:    sentinel allreduce before exit
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

        // Active-target fence: make rank 1's initialization visible to all.
        win.fence(ferrompi::WinFenceAssert::default())
            .expect("fence (write phase) failed");

        if rank == 0 {
            let mut local_buf = [0i32; N];

            // Acquire shared passive-target lock on rank 1's window.
            let guard = match win.lock(LockType::Shared, 1) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("rank 0: FAIL: Win::lock(Shared, 1) failed: {e}");
                    local_ok = false;
                    let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                    return;
                }
            };

            // Post the rget — request completes when local buffer contains data.
            let count = local_buf.len() as i64;
            let req = match win.rget(&mut local_buf, 1, 0, count) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("rank 0: FAIL: Win::rget returned error: {e}");
                    local_ok = false;
                    drop(guard);
                    let _ = world.allreduce_scalar(local_ok as i32, ReduceOp::Min);
                    return;
                }
            };

            // Wait for local completion — data is in local_buf after this.
            if let Err(e) = req.wait() {
                eprintln!("rank 0: FAIL: Request::wait after rget failed: {e}");
                local_ok = false;
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

            // Drop guard issues MPI_Win_unlock.
            drop(guard);
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::rget with local completion");
    }

    // ========================================================================
    // Sentinel allreduce(Min) — confirms no rank diverged silently.
    // ========================================================================
    let global_ok = world
        .allreduce_scalar(local_ok as i32, ReduceOp::Min)
        .expect("sentinel allreduce failed");

    assert!(
        global_ok != 0,
        "test_rma_rget: one or more ranks reported failure"
    );

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win::rget tests passed! (1 test)");
        println!("========================================");
    }
}
