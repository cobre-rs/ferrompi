//! Integration test for MPI Group range-based constructors.
//!
//! Exercises `Group::range_include` and `Group::range_exclude` using a
//! 4-rank `MPI_COMM_WORLD`. All assertions are guarded by a sentinel
//! `allreduce(Min)` before any `process::exit` so that no rank exits while
//! others are still inside MPI collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_ranges
// mpi-test: np=4

use ferrompi::{Mpi, RankRange, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_group_ranges requires exactly 4 processes, got {size}"
    );

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    let parent = match world.group() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 1: range_include with stride 1 → ranks {0, 1, 2} → size 3
    // ========================================================================
    {
        let ranges = [RankRange {
            first: 0,
            last: 2,
            stride: 1,
        }];
        match parent.range_include(&ranges) {
            Ok(sub) => match sub.size() {
                Ok(s) if s == 3 => {
                    if rank == 0 {
                        println!("PASS: Test 1 — range_include([0..=2 step 1]).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 1: expected size 3, got {s}");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 1: size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1: range_include failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: range_include with stride 2 → ranks {0, 2} → size 2
    // ========================================================================
    {
        let ranges = [RankRange {
            first: 0,
            last: 3,
            stride: 2,
        }];
        match parent.range_include(&ranges) {
            Ok(sub) => match sub.size() {
                Ok(s) if s == 2 => {
                    if rank == 0 {
                        println!("PASS: Test 2 — range_include([0..=3 step 2]).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 2: expected size 2, got {s}");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 2: size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2: range_include failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 3: range_exclude ranks {1, 2} → surviving ranks {0, 3} → size 2
    // ========================================================================
    {
        let ranges = [RankRange {
            first: 1,
            last: 2,
            stride: 1,
        }];
        match parent.range_exclude(&ranges) {
            Ok(sub) => match sub.size() {
                Ok(s) if s == 2 => {
                    if rank == 0 {
                        println!("PASS: Test 3 — range_exclude([1..=2 step 1]).size() = {s}");
                    }
                }
                Ok(s) => {
                    eprintln!("rank {rank}: FAIL Test 3: expected size 2, got {s}");
                    local_ok = false;
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 3: size() error: {e}");
                    local_ok = false;
                }
            },
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: range_exclude failed: {e}");
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_group_ranges");
}
