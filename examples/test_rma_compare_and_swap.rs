//! Integration test for `Win::compare_and_swap` — atomic compare-and-swap
//! over a fence epoch.
//!
//! Verifies two test cases:
//!   1. Match: rank 1's window slot 0 starts at `100i32`; rank 0 calls
//!      `compare_and_swap(200, 100, 1, 0)`; after the closing fence:
//!      rank 0's returned value equals `100` (pre-CAS, swap succeeded)
//!      and rank 1's window slot 0 equals `200` (updated).
//!   2. No match: rank 1's window slot 0 starts at `100i32`; rank 0 calls
//!      `compare_and_swap(200, 99, 1, 0)`; after the closing fence:
//!      rank 0's returned value equals `100` (pre-CAS, swap did not occur)
//!      and rank 1's window slot 0 equals `100` (unchanged).
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_compare_and_swap
// mpi-test: np=2

use ferrompi::{Mpi, PendingFetchResult, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_compare_and_swap requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1 (match): rank 1 initialises its window slot 0 to 100i32.
    // Rank 0 calls compare_and_swap(200, 100, 1, 0). After the closing fence:
    //   - rank 0's returned value must equal 100 (pre-CAS value)
    //   - rank 1's window slot 0 must equal 200 (swap succeeded)
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 1).expect("Win::allocate failed (test 1)");

        if rank == 1 {
            win.local_slice_mut()[0] = 100;
        }

        win.fence(WinFenceAssert::default())
            .expect("test 1 opening fence failed");

        // compare_and_swap returns a PendingFetchResult — result is not yet
        // populated; must call .resolve() only after the epoch closes.
        let mut pending: Option<PendingFetchResult<i32>> = None;
        if rank == 0 {
            match win.compare_and_swap(200, 100, 1, 0) {
                Ok(p) => pending = Some(p),
                Err(e) => {
                    eprintln!("FAIL: rank 0 Win::compare_and_swap (match) returned error: {e}");
                    local_ok = false;
                }
            }
        }

        // Close the epoch — MPI_Compare_and_swap completes here.
        win.fence(WinFenceAssert::default())
            .expect("test 1 closing fence failed");

        let mut old = 0i32;
        if let Some(p) = pending {
            // SAFETY: the preceding fence closed the epoch, so MPI has
            // populated the result buffer per PendingFetchResult::resolve's
            // safety contract.
            old = unsafe { p.resolve() };
        }

        if rank == 0 && old != 100 {
            eprintln!("FAIL: rank 0 returned value (match): expected 100, got {old}");
            local_ok = false;
        }

        if rank == 1 {
            let got = win.local_slice()[0];
            if got != 200 {
                eprintln!("FAIL: rank 1 window slot 0 after match CAS: expected 200, got {got}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::compare_and_swap (match)");
    }

    // ========================================================================
    // Test 2 (no match): rank 1 initialises its window slot 0 to 100i32.
    // Rank 0 calls compare_and_swap(200, 99, 1, 0). After the closing fence:
    //   - rank 0's returned value must equal 100 (pre-CAS value)
    //   - rank 1's window slot 0 must equal 100 (unchanged, swap did not occur)
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 1).expect("Win::allocate failed (test 2)");

        if rank == 1 {
            win.local_slice_mut()[0] = 100;
        }

        win.fence(WinFenceAssert::default())
            .expect("test 2 opening fence failed");

        // compare_and_swap returns a PendingFetchResult — result is not yet
        // populated; must call .resolve() only after the epoch closes.
        let mut pending: Option<PendingFetchResult<i32>> = None;
        if rank == 0 {
            match win.compare_and_swap(200, 99, 1, 0) {
                Ok(p) => pending = Some(p),
                Err(e) => {
                    eprintln!("FAIL: rank 0 Win::compare_and_swap (no match) returned error: {e}");
                    local_ok = false;
                }
            }
        }

        // Close the epoch — MPI_Compare_and_swap completes here.
        win.fence(WinFenceAssert::default())
            .expect("test 2 closing fence failed");

        let mut old = 0i32;
        if let Some(p) = pending {
            // SAFETY: the preceding fence closed the epoch, so MPI has
            // populated the result buffer per PendingFetchResult::resolve's
            // safety contract.
            old = unsafe { p.resolve() };
        }

        if rank == 0 && old != 100 {
            eprintln!("FAIL: rank 0 returned value (no match): expected 100, got {old}");
            local_ok = false;
        }

        if rank == 1 {
            let got = win.local_slice()[0];
            if got != 100 {
                eprintln!("FAIL: rank 1 window slot 0 after no-match CAS: expected 100, got {got}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::compare_and_swap (no match)");
    }

    common::check(&world, local_ok, "test_rma_compare_and_swap");
}
