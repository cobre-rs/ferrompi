//! Integration test for `Win::fetch_and_op` — single-element atomic fetch-and-update
//! over a fence epoch.
//!
//! Verifies two test cases:
//!   1. Sum: rank 1's window slot 0 starts at `100i32`; rank 0 calls
//!      `fetch_and_op(1, 1, 0, Sum)`; after the closing fence, rank 0's
//!      returned value equals `100` (pre-update) and rank 1's window slot 0
//!      equals `101` (post-update).
//!   2. Replace: rank 1's window slot 0 starts at `42i32`; rank 0 calls
//!      `fetch_and_op(999, 1, 0, Replace)`; after the closing fence, rank 0's
//!      returned value equals `42` (pre-update) and rank 1's window slot 0
//!      equals `999` (post-update).
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_fetch_and_op

use ferrompi::{Mpi, PendingFetchResult, ReduceOp, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_fetch_and_op requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1 (Sum + fetch): rank 1 initialises its window slot 0 to 100i32.
    // Rank 0 calls fetch_and_op(1, 1, 0, Sum). After the closing fence:
    //   - rank 0's returned value must equal 100 (pre-update)
    //   - rank 1's window slot 0 must equal 101 (post-update)
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 1).expect("Win::allocate failed");

        if rank == 1 {
            win.local_slice_mut()[0] = 100;
        }

        win.fence(WinFenceAssert::default())
            .expect("test 1 opening fence failed");

        // fetch_and_op returns a PendingFetchResult — result is not yet
        // populated; must call .resolve() only after the epoch closes.
        let mut pending: Option<PendingFetchResult<i32>> = None;
        if rank == 0 {
            match win.fetch_and_op(1i32, 1, 0, ReduceOp::Sum) {
                Ok(p) => pending = Some(p),
                Err(e) => {
                    eprintln!("FAIL: rank 0 Win::fetch_and_op Sum returned error: {e}");
                    local_ok = false;
                }
            }
        }

        // Close the epoch — MPI_Fetch_and_op completes here.
        win.fence(WinFenceAssert::default())
            .expect("test 1 closing fence failed");

        // SAFETY: epoch is closed; the result buffer is now populated.
        let mut old = 0i32;
        if let Some(p) = pending {
            old = unsafe { p.resolve() };
        }

        if rank == 0 && old != 100 {
            eprintln!("FAIL: rank 0 old value after Sum: expected 100, got {old}");
            local_ok = false;
        }

        if rank == 1 {
            let got = win.local_slice()[0];
            if got != 101 {
                eprintln!("FAIL: rank 1 window slot 0 after Sum: expected 101, got {got}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fetch_and_op Sum");
    }

    // ========================================================================
    // Test 2 (Replace + fetch): rank 1 initialises its window slot 0 to 42i32.
    // Rank 0 calls fetch_and_op(999, 1, 0, Replace). After the closing fence:
    //   - rank 0's returned value must equal 42 (pre-update)
    //   - rank 1's window slot 0 must equal 999 (post-update)
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 1).expect("Win::allocate (test 2) failed");

        if rank == 1 {
            win.local_slice_mut()[0] = 42;
        }

        win.fence(WinFenceAssert::default())
            .expect("test 2 opening fence failed");

        // fetch_and_op returns a PendingFetchResult — result is not yet
        // populated; must call .resolve() only after the epoch closes.
        let mut pending: Option<PendingFetchResult<i32>> = None;
        if rank == 0 {
            match win.fetch_and_op(999i32, 1, 0, ReduceOp::Replace) {
                Ok(p) => pending = Some(p),
                Err(e) => {
                    eprintln!("FAIL: rank 0 Win::fetch_and_op Replace returned error: {e}");
                    local_ok = false;
                }
            }
        }

        // Close the epoch — MPI_Fetch_and_op completes here.
        win.fence(WinFenceAssert::default())
            .expect("test 2 closing fence failed");

        // SAFETY: epoch is closed; the result buffer is now populated.
        let mut old = 0i32;
        if let Some(p) = pending {
            old = unsafe { p.resolve() };
        }

        if rank == 0 && old != 42 {
            eprintln!("FAIL: rank 0 old value after Replace: expected 42, got {old}");
            local_ok = false;
        }

        if rank == 1 {
            let got = win.local_slice()[0];
            if got != 999 {
                eprintln!("FAIL: rank 1 window slot 0 after Replace: expected 999, got {got}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fetch_and_op Replace");
    }

    common::check(&world, local_ok, "test_rma_fetch_and_op");
}
