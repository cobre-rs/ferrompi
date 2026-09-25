//! Integration test for `Group::translate_ranks`.
//!
//! Exercises the acceptance criteria from ticket-049:
//!
//! 1. `gw.translate_ranks(&[0, 1, 2, 3], &gsub)` where `gsub = gw.include(&[1, 3])`
//!    → `[None, Some(0), None, Some(1)]`.
//! 2. `gsub.translate_ranks(&[0, 1], &gw)` → `[Some(1), Some(3)]`.
//! 3. `gw.translate_ranks(&[], &gsub)` → `[]` (empty input).
//!
//! All assertions are guarded by a sentinel `allreduce(Min)` before any
//! `process::exit` so that no rank exits while others are still inside MPI
//! collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_translate
// mpi-test: np=4

use ferrompi::{Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_group_translate requires exactly 4 processes, got {size}"
    );

    let mut local_ok = true;

    // Obtain the world group once.
    let gw = match world.group() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: world.group() failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // Build gsub = {rank 1, rank 3} (in that order within gsub).
    let gsub = match gw.include(&[1, 3]) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: gw.include(&[1, 3]) failed: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 1: translate world ranks [0,1,2,3] into gsub's rank space.
    //
    //   gsub contains world-rank 1 (→ gsub-rank 0) and world-rank 3 (→ gsub-rank 1).
    //   World-ranks 0 and 2 are not in gsub → None.
    //   Expected: [None, Some(0), None, Some(1)]
    // ========================================================================
    match gw.translate_ranks(&[0, 1, 2, 3], &gsub) {
        Ok(result) => {
            let expected = vec![None, Some(0), None, Some(1)];
            if result == expected {
                if rank == 0 {
                    println!("PASS: Test 1 — gw→gsub translate [0,1,2,3] = {result:?}");
                }
            } else {
                eprintln!("rank {rank}: FAIL Test 1: expected {expected:?}, got {result:?}");
                local_ok = false;
            }
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 1: translate_ranks error: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 2: translate gsub ranks [0,1] back into world's rank space.
    //
    //   gsub-rank 0 → world-rank 1; gsub-rank 1 → world-rank 3.
    //   Expected: [Some(1), Some(3)]
    // ========================================================================
    match gsub.translate_ranks(&[0, 1], &gw) {
        Ok(result) => {
            let expected = vec![Some(1), Some(3)];
            if result == expected {
                if rank == 0 {
                    println!("PASS: Test 2 — gsub→gw translate [0,1] = {result:?}");
                }
            } else {
                eprintln!("rank {rank}: FAIL Test 2: expected {expected:?}, got {result:?}");
                local_ok = false;
            }
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 2: translate_ranks error: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 3: empty input slice → empty output.
    //
    //   translate_ranks(&[], &gsub) must return Ok(vec![]) regardless of
    //   the group handles involved.
    // ========================================================================
    match gw.translate_ranks(&[], &gsub) {
        Ok(result) => {
            if result.is_empty() {
                if rank == 0 {
                    println!("PASS: Test 3 — empty input → empty output");
                }
            } else {
                eprintln!("rank {rank}: FAIL Test 3: expected empty vec, got {result:?}");
                local_ok = false;
            }
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 3: translate_ranks error: {e}");
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_group_translate");
}
