//! Integration test for `ActiveEpoch::put` — one-sided write with a fence epoch.
//!
//! Verifies that rank 0 can write `[10.0, 20.0, 30.0, 40.0]` into rank 1's
//! window at displacement 0, and that rank 1 observes the correct data after
//! the closing fence.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_put
// mpi-test: np=2

use ferrompi::{Mpi, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_put requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: rank 0 puts [10.0, 20.0, 30.0, 40.0] into rank 1's window at
    // displacement 0. Verified by rank 1 reading its local window memory after
    // the closing fence (through Win::local).
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<f64>::allocate(&world, N).expect("Win::allocate failed");
        let buf = [10.0f64, 20.0, 30.0, 40.0];

        // Open the fence epoch on all ranks; the put completes when it closes
        let epoch = win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
            if rank == 0 {
                e.put(&buf, 1, 0)
            } else {
                Ok(())
            }
        });
        if let Err(e) = epoch {
            eprintln!("FAIL: rank {rank} fence epoch with put returned error: {e}");
            local_ok = false;
        }

        // Rank 1 inspects its local window memory through Win::local.
        // The closing fence guarantees the put from rank 0 is visible.
        if rank == 1 {
            let expected = [10.0f64, 20.0, 30.0, 40.0];
            let mut local = [0.0f64; N];
            win.local().copy_to_slice(&mut local);
            if local != expected {
                eprintln!(
                    "FAIL: rank 1 local window after put: expected {expected:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: ActiveEpoch::put with fence epoch");
    }

    // ========================================================================
    // Test 2: invalid target rank returns an MPI error.
    //
    // Rank 0 opens a fence epoch and calls put with target_rank = 5 (which
    // does not exist in a 2-process job). MPI must return an error. We close
    // the epoch on the other rank before checking the result.
    // ========================================================================
    {
        let mut win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 2) failed");
        let buf = [1.0f64, 2.0, 3.0, 4.0];

        win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
            if rank == 0 && e.put(&buf, 5, 0).is_ok() {
                eprintln!("FAIL: expected Err for invalid rank 5, got Ok");
                local_ok = false;
            }
            // The put to a non-existent rank failed immediately; we did not
            // actually post an operation, so closing the fence is safe.
            Ok(())
        })
        .expect("test 2 fence epoch failed");
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: ActiveEpoch::put invalid rank returns error");
    }

    common::check(&world, local_ok, "test_rma_put");
}
