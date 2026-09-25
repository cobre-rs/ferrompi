//! Integration test for `Communicator::create_from_group`.
//!
//! Exercises `MPI_Comm_create` via the `ferrompi_comm_create_from_group_parent`
//! shim. Verifies that:
//!
//! - Ranks 0 and 2 receive `Ok(Some(comm))` with `comm.size() == 2`.
//! - Ranks 1 and 3 receive `Ok(None)` (they are not in the sub-group).
//! - The sub-communicator is functional: `allreduce_scalar(1.0, Sum)` returns
//!   `2.0` on ranks that are members.
//!
//! All assertions are guarded by a sentinel `allreduce_scalar(Min)` before any
//! `process::exit` so that no rank exits while others are still inside MPI
//! collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_comm_from_group

use ferrompi::{Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_comm_from_group requires exactly 4 processes, got {size}"
    );

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Build the sub-group {0, 2} from the world group.
    // ========================================================================
    let world_group = world.group().expect("world.group() failed");
    let sub_group = world_group
        .include(&[0, 2])
        .expect("group.include(&[0, 2]) failed");

    // ========================================================================
    // Test 1: create_from_group — collective over all 4 ranks.
    //
    // Ranks 0 and 2 must receive Some(comm); ranks 1 and 3 must receive None.
    // ========================================================================
    let sub_comm = match world.create_from_group(&sub_group) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: create_from_group returned error: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    if rank == 0 || rank == 2 {
        // ====================================================================
        // Test 1a: member ranks must receive Some(comm).
        // ====================================================================
        let comm = match &sub_comm {
            Some(c) => c,
            None => {
                eprintln!("rank {rank}: FAIL Test 1a: expected Some(comm), got None");
                local_ok = false;
                // Participate in subsequent collective calls with fallback.
                // We cannot join sub-comm collectives without a comm, so we
                // skip them and let the sentinel allreduce catch the failure.
                let ok = local_ok as i32;
                let _ = world.allreduce_scalar(ok, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        // ====================================================================
        // Test 1b: sub-communicator must have size 2.
        // ====================================================================
        let sub_size = comm.size();
        if sub_size == 2 {
            if rank == 0 {
                println!("PASS: Test 1b — sub-comm size = {sub_size}");
            }
        } else {
            eprintln!("rank {rank}: FAIL Test 1b: sub-comm size = {sub_size}, expected 2");
            local_ok = false;
        }

        // ====================================================================
        // Test 2: allreduce on the sub-communicator.
        //
        // Each of the 2 member ranks contributes 1.0; the sum must be 2.0.
        // ====================================================================
        match comm.allreduce_scalar(1.0f64, ReduceOp::Sum) {
            Ok(result) if (result - 2.0f64).abs() < f64::EPSILON => {
                if rank == 0 {
                    println!("PASS: Test 2 — sub-comm allreduce(1.0, Sum) = {result}");
                }
            }
            Ok(result) => {
                eprintln!("rank {rank}: FAIL Test 2: allreduce result = {result}, expected 2.0");
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2: allreduce failed: {e}");
                local_ok = false;
            }
        }
    } else {
        // ====================================================================
        // Test 1a (non-member): ranks 1 and 3 must receive None.
        // ====================================================================
        if sub_comm.is_none() {
            if rank == 1 {
                println!("PASS: Test 1a — non-member rank {rank} received None");
            }
        } else {
            eprintln!("rank {rank}: FAIL Test 1a: expected None for non-member rank, got Some");
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_comm_from_group");
}
