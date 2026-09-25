//! Integration test for `Win::get_accumulate` — atomic read-modify-write over a
//! fence epoch.
//!
//! Verifies two test cases:
//!   1. Sum: rank 1's window starts at `[10, 20, 30, 40]`; rank 0 calls
//!      `get_accumulate([1, 2, 3, 4], &mut result, 1, 0, 4, Sum)`; after the
//!      closing fence, `result == [10, 20, 30, 40]` (pre-update) and rank 1's
//!      window equals `[11, 22, 33, 44]` (post-update).
//!   2. NoOp: rank 1's window starts at `[99]`; rank 0 calls
//!      `get_accumulate([0], &mut result, 1, 0, 1, NoOp)`; after the closing
//!      fence, `result == [99]` and rank 1's window is unchanged at `[99]`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_get_accumulate

use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_get_accumulate requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1 (Sum + fetch): rank 1 initialises its window to [10, 20, 30, 40];
    // rank 0 calls get_accumulate with origin=[1, 2, 3, 4] and ReduceOp::Sum.
    // After the closing fence:
    //   - rank 0's result must equal the pre-update value [10, 20, 30, 40]
    //   - rank 1's window must equal the post-update value [11, 22, 33, 44]
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");
        let mut result = [0i32; N];

        if rank == 1 {
            win.local_slice_mut().copy_from_slice(&[10i32, 20, 30, 40]);
        }

        win.fence(WinFenceAssert::default())
            .expect("test 1 opening fence failed");

        if rank == 0 {
            let origin = [1i32, 2, 3, 4];
            if let Err(e) = win.get_accumulate(&origin, &mut result, 1, 0, N as i64, ReduceOp::Sum)
            {
                eprintln!("FAIL: rank 0 Win::get_accumulate Sum returned error: {e}");
                local_ok = false;
            }
        }

        win.fence(WinFenceAssert::default())
            .expect("test 1 closing fence failed");

        if rank == 0 {
            let expected_result = [10i32, 20, 30, 40];
            if result != expected_result {
                eprintln!(
                    "FAIL: rank 0 result after Sum: expected {expected_result:?}, got {result:?}"
                );
                local_ok = false;
            }
        }

        if rank == 1 {
            let expected_win = [11i32, 22, 33, 44];
            let local = win.local_slice();
            if local != expected_win {
                eprintln!(
                    "FAIL: rank 1 window after Sum: expected {expected_win:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::get_accumulate Sum (fetch + accumulate)");
    }

    // ========================================================================
    // Test 2 (NoOp = atomic read): rank 1 initialises its window to [99];
    // rank 0 calls get_accumulate with origin=[0] and ReduceOp::NoOp.
    // After the closing fence:
    //   - rank 0's result must equal [99] (the pre-update value)
    //   - rank 1's window must remain [99] (unchanged)
    // ========================================================================
    {
        const N: usize = 1;
        let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate (test 2) failed");
        let mut result = [0i32; N];

        if rank == 1 {
            win.local_slice_mut().copy_from_slice(&[99i32]);
        }

        win.fence(WinFenceAssert::default())
            .expect("test 2 opening fence failed");

        if rank == 0 {
            let origin = [0i32];
            if let Err(e) = win.get_accumulate(&origin, &mut result, 1, 0, N as i64, ReduceOp::NoOp)
            {
                eprintln!("FAIL: rank 0 Win::get_accumulate NoOp returned error: {e}");
                local_ok = false;
            }
        }

        win.fence(WinFenceAssert::default())
            .expect("test 2 closing fence failed");

        if rank == 0 {
            let expected_result = [99i32];
            if result != expected_result {
                eprintln!(
                    "FAIL: rank 0 result after NoOp: expected {expected_result:?}, got {result:?}"
                );
                local_ok = false;
            }
        }

        if rank == 1 {
            let expected_win = [99i32];
            let local = win.local_slice();
            if local != expected_win {
                eprintln!(
                    "FAIL: rank 1 window after NoOp: expected {expected_win:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::get_accumulate NoOp (atomic read)");
    }

    common::check(&world, local_ok, "test_rma_get_accumulate");
}
