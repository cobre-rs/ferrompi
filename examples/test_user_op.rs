//! Integration test for `UserOp<T>` and `Communicator::allreduce_with_op`.
//!
//! Tests:
//!   1. Element-wise max of f64 via a user-defined commutative op.
//!   2. Bitwise-OR reimplementation on i32 (each rank contributes 1 << rank).
//!   3. Drop-after-use: verifies no MPI corruption after the UserOp is dropped.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_user_op
// mpi-test: np=4 valgrind

use ferrompi::{Mpi, ReduceOp, UserOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_user_op requires at least 2 MPI processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: element-wise max of f64
    //   Each rank contributes [rank + 1.5].
    //   Expected result: [(size - 1) as f64 + 1.5] on every rank.
    // ========================================================================
    {
        let op: UserOp<f64> = match UserOp::new(|invec, inoutvec| {
            for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
                if *x > *y {
                    *y = *x;
                }
            }
        }) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1: UserOp::new failed: {e}");
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        let send = vec![rank as f64 + 1.5_f64];
        let mut recv = vec![0.0_f64];

        if let Err(e) = world.allreduce_with_op(&send, &mut recv, &op) {
            eprintln!("rank {rank}: FAIL Test 1: allreduce_with_op failed: {e}");
            local_ok = false;
        }

        let expected = (size - 1) as f64 + 1.5_f64;
        if (recv[0] - expected).abs() > 1e-12 {
            eprintln!(
                "rank {rank}: FAIL Test 1: expected {expected}, got {}",
                recv[0]
            );
            local_ok = false;
        } else if rank == 0 {
            println!("PASS: UserOp max-of-floats (result = {})", recv[0]);
        }
    }

    // ========================================================================
    // Test 2: bitwise-OR on i32
    //   Each rank contributes 1 << rank.
    //   Expected result: (1 << size) - 1  (all bits 0..size-1 set).
    // ========================================================================
    {
        let op: UserOp<i32> = match UserOp::new(|invec, inoutvec| {
            for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
                *y |= *x;
            }
        }) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2: UserOp::new failed: {e}");
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        // Guard against overflow: if size > 30 the shift wraps on i32.
        let contrib = if rank < 30 { 1i32 << rank } else { 0i32 };
        let send = vec![contrib];
        let mut recv = vec![0i32];

        if let Err(e) = world.allreduce_with_op(&send, &mut recv, &op) {
            eprintln!("rank {rank}: FAIL Test 2: allreduce_with_op failed: {e}");
            local_ok = false;
        }

        let effective_size = size.min(30);
        let expected = (1i32 << effective_size) - 1;
        if recv[0] != expected {
            eprintln!(
                "rank {rank}: FAIL Test 2: expected {expected:#010x}, got {:#010x}",
                recv[0]
            );
            local_ok = false;
        } else if rank == 0 {
            println!(
                "PASS: UserOp custom combine i32 bitwise-OR (result = {:#010x})",
                recv[0]
            );
        }
    }

    // ========================================================================
    // Test 3: Drop-after-use
    //   Create a UserOp, use it once, drop it explicitly, then verify that
    //   ordinary MPI operations still work correctly (no MPI state corruption).
    // ========================================================================
    {
        let op: UserOp<i32> = match UserOp::new(|invec, inoutvec| {
            for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
                *y += *x;
            }
        }) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: UserOp::new failed: {e}");
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        let send = vec![1i32];
        let mut recv = vec![0i32];
        if let Err(e) = world.allreduce_with_op(&send, &mut recv, &op) {
            eprintln!("rank {rank}: FAIL Test 3: allreduce_with_op failed: {e}");
            local_ok = false;
        }

        // Explicit drop — exercises Drop for UserOp.
        drop(op);

        // Verify subsequent MPI operations still work.
        match world.allreduce_scalar(rank, ReduceOp::Sum) {
            Ok(total) if total == (size * (size - 1) / 2) => {
                if rank == 0 {
                    println!("PASS: UserOp Drop after use (MPI still healthy, sum = {total})");
                }
            }
            Ok(total) => {
                eprintln!(
                    "rank {rank}: FAIL Test 3: post-drop allreduce_scalar sum = {total}, \
                     expected {}",
                    size * (size - 1) / 2
                );
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: post-drop allreduce_scalar failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 4: non-commutative reduction — min operation
    //   Each rank contributes [rank as i32 + 1] (values: 1, 2, ..., size).
    //   With commute=0, the min reduction is deterministic regardless of MPI's
    //   allreduce tree algorithm: min(1..=size) == 1 on every rank.
    //   Expected result: 1 on every rank.
    // ========================================================================
    {
        let op: UserOp<i32> = match UserOp::new_noncommutative(|invec, inoutvec| {
            for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
                if *x < *y {
                    *y = *x;
                }
            }
        }) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 4: UserOp::new_noncommutative failed: {e}");
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        let send = vec![rank + 1];
        let mut recv = vec![0i32];

        if let Err(e) = world.allreduce_with_op(&send, &mut recv, &op) {
            eprintln!("rank {rank}: FAIL Test 4: allreduce_with_op failed: {e}");
            local_ok = false;
        }

        if recv[0] != 1 {
            eprintln!("rank {rank}: FAIL Test 4: expected 1, got {}", recv[0]);
            local_ok = false;
        } else if rank == 0 {
            println!("PASS: UserOp non-commutative min (result = {})", recv[0]);
        }
    }

    common::check(&world, local_ok, "test_user_op");
}
