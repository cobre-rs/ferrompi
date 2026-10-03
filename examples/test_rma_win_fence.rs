//! Integration test for Win::fence active-target epoch synchronization.
//!
//! Verifies that `Win::fence` opens and closes an epoch correctly on all
//! ranks, that `WinFenceAssert` composition works as expected, and that the two
//! assertions that would leave the epoch's operations uncompleted are rejected
//! before any MPI call.
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
    // Test 3: WinFenceAssert bitflag composition (non-MPI, compile+logic only)
    //
    // Verify that the OR of two non-zero flags has at least two bits set.
    // We use no_store() | no_put() as an example. On real MPI installations
    // these constants are non-zero and distinct, so their OR has >=2 bits set.
    // If the MPI < 3 fallback is active, both equal 0 and the OR is also 0
    // (which is the safe "no assertion" value).
    // ========================================================================
    {
        let combined = WinFenceAssert::no_store() | WinFenceAssert::no_put();
        let no_store_bits = WinFenceAssert::no_store().bits();
        let no_put_bits = WinFenceAssert::no_put().bits();

        // If both are non-zero they must be distinct (MPI guarantees that) and
        // their combination must differ from each individually.
        if no_store_bits != 0 && no_put_bits != 0 {
            assert!(
                combined.bits() != no_store_bits && combined.bits() != no_put_bits,
                "WinFenceAssert OR result ({}) is not a proper superset of \
                 no_store ({}) | no_put ({})",
                combined.bits(),
                no_store_bits,
                no_put_bits
            );
        }
    }

    world.barrier().expect("barrier after test 3 failed");
    if rank == 0 {
        println!("PASS: WinFenceAssert bitflag composition");
    }

    // ========================================================================
    // Test 4: Win::fence with no_precede()/no_succeed() and an actual put.
    //
    // Uses a fresh window — MPI_MODE_NOPRECEDE is valid only when no RMA
    // epoch was previously open on the window.
    // ========================================================================
    {
        let no_precede = WinFenceAssert::no_precede();
        let no_succeed = WinFenceAssert::no_succeed();
        if no_precede.bits() == 0
            || no_succeed.bits() == 0
            || no_precede.bits() == no_succeed.bits()
        {
            eprintln!(
                "rank {rank}: FAIL: no_precede/no_succeed bits invalid \
                 (no_precede={}, no_succeed={})",
                no_precede.bits(),
                no_succeed.bits()
            );
            local_ok = false;
        }

        let mut win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 4) failed");
        let buf = [1.0f64, 2.0, 3.0, 4.0];

        let epoch = win.fence(no_precede, no_succeed, |e| {
            if rank == 0 {
                e.put(&buf, 1, 0, buf.len() as i64)?;
            }
            Ok(())
        });
        if let Err(e) = epoch {
            eprintln!("rank {rank}: FAIL: Win::fence(no_precede, no_succeed) with put failed: {e}");
            local_ok = false;
        }

        if rank == 1 {
            let expected = [1.0f64, 2.0, 3.0, 4.0];
            if win.local_slice() != expected {
                eprintln!(
                    "rank {rank}: FAIL: expected {expected:?}, got {:?}",
                    win.local_slice()
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 4 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence with no_precede/no_succeed and put");
    }

    // ========================================================================
    // Test 5: fence rejects no_succeed() on the opening fence and no_precede()
    // on the closing one with InvalidArgument, on every rank, without running
    // the closure and before any MPI call: no rank enters a collective, so the
    // window still takes a normal epoch afterwards.
    // ========================================================================
    {
        let mut win = Win::<f64>::allocate(&world, 4).expect("Win::allocate (test 5) failed");
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

    world.barrier().expect("barrier after test 5 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::fence rejects no_succeed open and no_precede close");
    }

    common::check(&world, local_ok, "test_rma_win_fence");

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win::fence tests passed! (5 tests)");
        println!("========================================");
    }
}
