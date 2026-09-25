//! Integration test for `Group::compare` and `GroupComparison`.
//!
//! Exercises all four acceptance criteria from ticket-048:
//!
//! 1. `include(&[1, 2])` vs `include(&[2, 1])` — same members, different
//!    ordering → `Similar`.  Using reversed ranks forces MPI_SIMILAR even on
//!    MPICH, which canonicalises equal-order groups to MPI_IDENT.
//! 2. `include(&[0, 1, 2])` vs `include(&[2, 1, 0])` — same member set,
//!    reversed order → `Similar`.
//! 3. `include(&[0, 1])` vs `include(&[2, 3])` — disjoint rank sets →
//!    `Unequal`.
//! 4. `gw.compare(&gw)` — same group object → `Identical`.
//!
//! All assertions are guarded by a sentinel `allreduce(Min)` before any
//! `process::exit` so that no rank exits while others are still inside MPI
//! collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_compare
// mpi-test: np=4

use ferrompi::{GroupComparison, Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_group_compare requires exactly 4 processes, got {size}"
    );

    let mut local_ok = true;

    // Obtain the world group once; used as the source for all sub-groups.
    let gw = match world.group() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 1: same member set, different ordering → Similar
    //
    //   g1 = include([1, 2])  — ordering: rank 1 first, rank 2 second
    //   g2 = include([2, 1])  — ordering: rank 2 first, rank 1 second
    //
    //   Both groups contain the same two processes {1, 2} but in a different
    //   order.  MPI guarantees MPI_SIMILAR for this case on all conforming
    //   implementations.  (Using same-order groups with MPICH can yield
    //   MPI_IDENT instead of MPI_SIMILAR because MPICH may canonicalise
    //   identical-order groups to the same internal object.)
    // ========================================================================
    let g1 = match gw.include(&[1, 2]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[1,2]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    let g2 = match gw.include(&[2, 1]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[2,1]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    match g1.compare(&g2) {
        Ok(GroupComparison::Similar) => {
            if rank == 0 {
                println!("PASS: Test 1 — same members, different order → Similar");
            }
        }
        Ok(other) => {
            eprintln!("rank {rank}: FAIL Test 1: expected Similar, got {other:?}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 1: compare() error: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 2: same member set, reversed 3-element ordering → Similar
    //
    //   g_fwd = include([0, 1, 2])
    //   g_rev = include([2, 1, 0])
    //
    //   Same set {0, 1, 2}, different rank order → MPI_SIMILAR.
    // ========================================================================
    let g_fwd = match gw.include(&[0, 1, 2]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[0,1,2]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    let g_rev = match gw.include(&[2, 1, 0]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[2,1,0]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    match g_fwd.compare(&g_rev) {
        Ok(GroupComparison::Similar) => {
            if rank == 0 {
                println!("PASS: Test 2 — same members, reversed 3-element order → Similar");
            }
        }
        Ok(other) => {
            eprintln!("rank {rank}: FAIL Test 2: expected Similar, got {other:?}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 2: compare() error: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 3: different rank sets → Unequal
    //
    //   g_lo = include([0, 1])
    //   g_hi = include([2, 3])
    //
    //   Completely disjoint sets → MPI_UNEQUAL.
    // ========================================================================
    let g_lo = match gw.include(&[0, 1]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[0,1]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    let g_hi = match gw.include(&[2, 3]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[2,3]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    match g_lo.compare(&g_hi) {
        Ok(GroupComparison::Unequal) => {
            if rank == 0 {
                println!("PASS: Test 3 — disjoint rank sets → Unequal");
            }
        }
        Ok(other) => {
            eprintln!("rank {rank}: FAIL Test 3: expected Unequal, got {other:?}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 3: compare() error: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 4: same group object → Identical
    //
    //   gw.compare(&gw) — both sides are the same MPI group object.
    //   MPI guarantees MPI_IDENT for this case.
    // ========================================================================
    match gw.compare(&gw) {
        Ok(GroupComparison::Identical) => {
            if rank == 0 {
                println!("PASS: Test 4 — same group object → Identical");
            }
        }
        Ok(other) => {
            eprintln!("rank {rank}: FAIL Test 4: expected Identical, got {other:?}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 4: compare() error: {e}");
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_group_compare");
}
