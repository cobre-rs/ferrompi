//! Integration test for MPI Group set-theoretic operations.
//!
//! Exercises `Group::union`, `Group::intersection`, and `Group::difference`
//! using a 4-rank world. All assertions are guarded by a sentinel
//! allreduce(Min) before any `process::exit` so that no rank exits while
//! others are still inside MPI collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_set_ops

use ferrompi::{Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_group_set_ops requires exactly 4 processes, got {size}"
    );

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // Build the base groups shared across all tests.
    // gw = world group; g1 = {0, 1}; g2 = {1, 2}
    let gw = match world.group() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    let g1 = match gw.include(&[0, 1]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[0,1]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    let g2 = match gw.include(&[1, 2]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[1,2]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 1: g1.union(&g2).size() == 3
    //   g1 = {0, 1}, g2 = {1, 2} → union = {0, 1, 2}
    // ========================================================================
    {
        match g1.union(&g2) {
            Ok(u) => match u.size() {
                Ok(s) if s == 3 => {
                    if rank == 0 {
                        println!("PASS: Test 1 — g1.union(&g2).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 1: union size = {s}, expected 3");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 1: union.size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1: g1.union(&g2) failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: g1.intersection(&g2).size() == 1
    //   g1 = {0, 1}, g2 = {1, 2} → intersection = {1}
    // ========================================================================
    {
        match g1.intersection(&g2) {
            Ok(i) => match i.size() {
                Ok(s) if s == 1 => {
                    if rank == 0 {
                        println!("PASS: Test 2 — g1.intersection(&g2).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 2: intersection size = {s}, expected 1");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 2: intersection.size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2: g1.intersection(&g2) failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 3: g1.difference(&g2).size() == 1
    //   g1 = {0, 1}, g2 = {1, 2} → difference = {0}
    // ========================================================================
    {
        match g1.difference(&g2) {
            Ok(d) => match d.size() {
                Ok(s) if s == 1 => {
                    if rank == 0 {
                        println!("PASS: Test 3 — g1.difference(&g2).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 3: difference size = {s}, expected 1");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 3: difference.size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: g1.difference(&g2) failed: {e}");
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_group_set_ops");
}
