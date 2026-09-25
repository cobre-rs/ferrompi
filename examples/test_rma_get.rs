//! Integration test for `Win::get` — one-sided read with a fence epoch.
//!
//! Verifies that rank 0 can read `[100.0, 200.0, 300.0, 400.0]` from rank 1's
//! window at displacement 0, and that rank 0 observes the correct data after
//! the closing fence.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_get
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
        "test_rma_get requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: rank 1 writes [100.0, 200.0, 300.0, 400.0] into its own window;
    // rank 0 reads that data via Win::get. Verified by rank 0 after the
    // closing fence.
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<f64>::allocate(&world, N).expect("Win::allocate failed");

        // Rank 1 initialises its local window before the epoch opens.
        if rank == 1 {
            let local = win.local_slice_mut();
            local.copy_from_slice(&[100.0f64, 200.0, 300.0, 400.0]);
        }

        // Open the fence epoch on all ranks
        win.fence(WinFenceAssert::default())
            .expect("opening fence failed");

        let mut local_buf = [0.0f64; N];
        if rank == 0 {
            if let Err(e) = win.get(&mut local_buf, 1, 0, N as i64) {
                eprintln!("FAIL: rank 0 Win::get returned error: {e}");
                local_ok = false;
            }
        }

        // Close the epoch — get completes here, local_buf is now valid
        win.fence(WinFenceAssert::default())
            .expect("closing fence failed");

        if rank == 0 {
            let expected = [100.0f64, 200.0, 300.0, 400.0];
            if local_buf != expected {
                eprintln!(
                    "FAIL: rank 0 local_buf after get: expected {expected:?}, got {local_buf:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::get with fence epoch");
    }

    // ========================================================================
    // Test 2: invalid target rank returns an MPI error.
    //
    // Rank 0 opens a fence epoch and calls get with target_rank = 5 (which
    // does not exist in a 2-process job). MPI must return an error. We close
    // the epoch on the other rank before checking the result.
    // ========================================================================
    {
        let win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 2) failed");

        win.fence(WinFenceAssert::default())
            .expect("test 2 opening fence failed");

        if rank == 0 {
            let mut buf = [0.0f64; 4];
            let result = win.get(&mut buf, 5, 0, 4);
            if result.is_ok() {
                eprintln!("FAIL: expected Err for invalid rank 5, got Ok");
                local_ok = false;
            }
            // The get to a non-existent rank failed immediately; we did not
            // actually post an operation, so closing the fence is safe.
        }

        win.fence(WinFenceAssert::default())
            .expect("test 2 closing fence failed");
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::get invalid rank returns error");
    }

    common::check(&world, local_ok, "test_rma_get");
}
