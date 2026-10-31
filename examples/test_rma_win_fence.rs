//! Integration test for Win::fence active-target epoch synchronization.
//!
//! Verifies that `Win::fence` opens and closes an epoch on all ranks, with and
//! without assertions, and that the two assertions that would leave the epoch's
//! operations uncompleted are rejected before any MPI call.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_fence
// mpi-test: np=2

use ferrompi::{Error, Mpi, Win, WinFenceAssert};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_win_fence requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: Win::fence with WinFenceAssert::default() (no assertion)
    //
    // Allocate a window and run an empty closure in a fence epoch (opening
    // and closing fence). This is the minimal valid epoch around RMA ops.
    // ========================================================================
    {
        let mut win = Win::<f64>::allocate(&world, 16).expect("Win::allocate failed");

        win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |_| {
            Ok(())
        })
        .expect("Win::fence(default) failed");
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 {
        println!("PASS: Win::fence with no assertion");
    }

    // ========================================================================
    // Test 2: Win::fence with WinFenceAssert::none() — explicit zero
    // ========================================================================
    {
        let mut win = Win::<i32>::allocate(&world, 8).expect("Win::allocate (i32) failed");

        win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |_| Ok(()))
            .expect("Win::fence(none) failed");
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 {
        println!("PASS: Win::fence with explicit none()");
    }

    // ========================================================================
    // Test 3: Win::fence with no_precede()/no_succeed() and an actual put.
    //
    // Uses a fresh window — MPI_MODE_NOPRECEDE is valid only when no RMA
    // epoch was previously open on the window.
    // ========================================================================
    {
        let no_precede = WinFenceAssert::no_precede();
        let no_succeed = WinFenceAssert::no_succeed();

        let mut win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 3) failed");
        let buf = [1.0f64, 2.0, 3.0, 4.0];

        let epoch = win.fence(no_precede, no_succeed, |e| {
            if rank == 0 {
                e.put(&buf, 1, 0)?;
            }
            Ok(())
        });
        if let Err(e) = epoch {
            eprintln!("rank {rank}: FAIL: Win::fence(no_precede, no_succeed) with put failed: {e}");
            local_ok = false;
        }

        if rank == 1 {
            let expected = [1.0f64, 2.0, 3.0, 4.0];
            let mut local = [0.0f64; 4];
            win.local().copy_to_slice(&mut local);
            if local != expected {
                eprintln!("rank {rank}: FAIL: expected {expected:?}, got {local:?}");
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence with no_precede/no_succeed and put");
    }

    // ========================================================================
    // Test 4: fence rejects no_succeed() on the opening fence and no_precede()
    // on the closing one with InvalidArgument, on every rank, without running
    // the closure and before any MPI call: no rank enters a collective, so the
    // window still takes a normal epoch afterwards.
    // ========================================================================
    {
        let mut win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 4) failed");
        let mut ran = false;

        let open = win.fence(WinFenceAssert::no_succeed(), WinFenceAssert::none(), |_| {
            ran = true;
            Ok(())
        });
        if !matches!(open, Err(Error::InvalidArgument { arg: "open", .. })) {
            eprintln!("rank {rank}: FAIL: fence(no_succeed, none) returned {open:?}");
            local_ok = false;
        }

        let close = win.fence(WinFenceAssert::none(), WinFenceAssert::no_precede(), |_| {
            ran = true;
            Ok(())
        });
        if !matches!(close, Err(Error::InvalidArgument { arg: "close", .. })) {
            eprintln!("rank {rank}: FAIL: fence(none, no_precede) returned {close:?}");
            local_ok = false;
        }

        if ran {
            eprintln!("rank {rank}: FAIL: a rejected fence ran its closure");
            local_ok = false;
        }

        if let Err(e) = win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |_| Ok(())) {
            eprintln!("rank {rank}: FAIL: fence after the rejections failed: {e}");
            local_ok = false;
        }
    }

    world.barrier().expect("barrier after test 4 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence rejects no_succeed open and no_precede close");
    }

    common::check(&world, local_ok, "test_rma_win_fence");

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win::fence tests passed! (4 tests)");
        println!("========================================");
    }
}
