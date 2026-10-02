//! Integration test for `Win::accumulate` — one-sided reduction with a fence
//! epoch.
//!
//! Verifies three test cases:
//!   1. Sum: rank 0 accumulates `[10, 20, 30, 40]` onto rank 1's window which
//!      starts at `[1, 2, 3, 4]`; rank 1 must observe `[11, 22, 33, 44]`.
//!   2. Replace: rank 0 accumulates `[100, 200, 300, 400]` with
//!      `AccumulateOp::REPLACE` onto rank 1's window; rank 1 must observe
//!      `[100, 200, 300, 400]`.
//!   3. Rejected ops: a bitwise or logical op on an `f64` window returns
//!      `Error::InvalidArgument` from `accumulate` and `get_accumulate`
//!      before any MPI call, and the window is unchanged.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_accumulate
// mpi-test: np=2

use ferrompi::{AccumulateOp, Error, Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_accumulate requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1 (Sum): rank 1 initialises its window to [1, 2, 3, 4]; rank 0
    // accumulates [10, 20, 30, 40] with ReduceOp::Sum; rank 1 asserts
    // [11, 22, 33, 44] after the closing fence.
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<f64>::allocate(&world, N).expect("Win::allocate failed");

        // Rank 1 initialises its local window before the epoch opens.
        if rank == 1 {
            win.local_slice_mut()
                .copy_from_slice(&[1.0f64, 2.0, 3.0, 4.0]);
        }

        // Open the fence epoch on all ranks
        win.fence(WinFenceAssert::default())
            .expect("opening fence failed");

        if rank == 0 {
            let buf = [10.0f64, 20.0, 30.0, 40.0];
            if let Err(e) = win.accumulate(&buf, 1, 0, buf.len() as i64, ReduceOp::Sum) {
                eprintln!("FAIL: rank 0 Win::accumulate Sum returned error: {e}");
                local_ok = false;
            }
        }

        // Close the epoch — accumulate completes here
        win.fence(WinFenceAssert::default())
            .expect("closing fence failed");

        if rank == 1 {
            let expected = [11.0f64, 22.0, 33.0, 44.0];
            let local = win.local_slice();
            if local != expected {
                eprintln!("FAIL: rank 1 window after Sum: expected {expected:?}, got {local:?}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::accumulate Sum");
    }

    // ========================================================================
    // Test 2 (Replace): rank 1 initialises its window to [1, 2, 3, 4]; rank 0
    // accumulates [100, 200, 300, 400] with AccumulateOp::REPLACE; rank 1 asserts
    // [100, 200, 300, 400] after the closing fence.
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<f64>::allocate(&world, N).expect("Win::allocate (test 2) failed");

        // Rank 1 initialises its local window before the epoch opens.
        if rank == 1 {
            win.local_slice_mut()
                .copy_from_slice(&[1.0f64, 2.0, 3.0, 4.0]);
        }

        // Open the fence epoch on all ranks
        win.fence(WinFenceAssert::default())
            .expect("test 2 opening fence failed");

        if rank == 0 {
            let buf = [100.0f64, 200.0, 300.0, 400.0];
            if let Err(e) = win.accumulate(&buf, 1, 0, buf.len() as i64, AccumulateOp::REPLACE) {
                eprintln!("FAIL: rank 0 Win::accumulate Replace returned error: {e}");
                local_ok = false;
            }
        }

        // Close the epoch — accumulate completes here
        win.fence(WinFenceAssert::default())
            .expect("test 2 closing fence failed");

        if rank == 1 {
            let expected = [100.0f64, 200.0, 300.0, 400.0];
            let local = win.local_slice();
            if local != expected {
                eprintln!(
                    "FAIL: rank 1 window after Replace: expected {expected:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::accumulate Replace");
    }

    // ========================================================================
    // Test 3 (rejected ops): on an f64 window every rank calls accumulate with
    // BitwiseOr and get_accumulate with LogicalAnd; both must return
    // InvalidArgument { arg: "op" } before reaching MPI, so the window is
    // unchanged after the closing fence.
    // ========================================================================
    {
        const N: usize = 4;
        let initial = [1.0f64, 2.0, 3.0, 4.0];
        let mut win = Win::<f64>::allocate(&world, N).expect("Win::allocate (test 3) failed");
        win.local_slice_mut().copy_from_slice(&initial);

        win.fence(WinFenceAssert::default())
            .expect("test 3 opening fence failed");

        let buf = [10.0f64, 20.0, 30.0, 40.0];
        let mut out = [0.0f64; N];
        let accumulate = win.accumulate(&buf, 1, 0, N as i64, ReduceOp::BitwiseOr);
        if !matches!(accumulate, Err(Error::InvalidArgument { arg: "op", .. })) {
            eprintln!("FAIL: rank {rank} accumulate BitwiseOr on f64: expected InvalidArgument for op, got {accumulate:?}");
            local_ok = false;
        }
        let get_accumulate =
            win.get_accumulate(&buf, &mut out, 1, 0, N as i64, ReduceOp::LogicalAnd);
        if !matches!(
            get_accumulate,
            Err(Error::InvalidArgument { arg: "op", .. })
        ) {
            eprintln!("FAIL: rank {rank} get_accumulate LogicalAnd on f64: expected InvalidArgument for op, got {get_accumulate:?}");
            local_ok = false;
        }

        win.fence(WinFenceAssert::default())
            .expect("test 3 closing fence failed");

        let local = win.local_slice();
        if local != initial {
            eprintln!(
                "FAIL: rank {rank} window after rejected ops: expected {initial:?}, got {local:?}"
            );
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::accumulate and get_accumulate reject a bitwise op on f64");
    }

    common::check(&world, local_ok, "test_rma_accumulate");
}
