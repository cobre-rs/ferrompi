//! Integration test for `Mpi::create_from_group` (MPI 4.0+).
//!
//! Exercises `MPI_Comm_create_from_group` via the
//! `ferrompi_comm_create_from_group` shim.  Verifies that:
//!
//! - On MPI 4.0+: every rank that calls `mpi.create_from_group(&g, tag)`
//!   with the same world group and the same tag receives an `Ok(comm)` with
//!   `comm.size() == world.size()`.
//! - On MPI < 4.0: the example prints `SKIP` and exits 0 before calling
//!   `create_from_group`, so an `Err` from that call on MPI >= 4 fails.
//!
//! All assertions are guarded by a sentinel `allreduce_scalar(Min)` before
//! any `process::exit` so that no rank exits while others are still inside
//! MPI collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_mpi_from_group
// mpi-test: np=4 skip-ok=openmpi

use ferrompi::{Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_mpi_from_group requires exactly 4 processes, got {size}"
    );

    // ========================================================================
    // Version check — skip on MPI < 4.0.
    // ========================================================================
    if common::mpi_major() < 4 {
        let version = Mpi::version().expect("Mpi::version() failed");
        common::skip(
            &world,
            &format!("create_from_group needs MPI 4 ({version})"),
        );
        return;
    }

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Build the world group — all ranks participate with the same tag.
    // ========================================================================
    let world_group = world.group().expect("world.group() failed");

    // ========================================================================
    // Test 1: create_from_group returns Ok(comm).
    // ========================================================================
    let comm = match mpi.create_from_group(&world_group, "ferrompi-test") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rank {rank}: FAIL: create_from_group returned error: {e}");
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 2: the new communicator has the same size as COMM_WORLD.
    // ========================================================================
    let comm_size = comm.size();
    if comm_size == size {
        if rank == 0 {
            println!("PASS: Test 2 — comm.size() == world.size() == {comm_size}");
        }
    } else {
        eprintln!("rank {rank}: FAIL Test 2: comm.size() = {comm_size}, expected {size}");
        local_ok = false;
    }

    // ========================================================================
    // Test 3: allreduce on the new communicator is functional.
    // ========================================================================
    match comm.allreduce_scalar(1i32, ReduceOp::Sum) {
        Ok(result) if result == size => {
            if rank == 0 {
                println!("PASS: Test 3 — allreduce(1, Sum) == {result} (== world.size())");
            }
        }
        Ok(result) => {
            eprintln!("rank {rank}: FAIL Test 3: allreduce result = {result}, expected {size}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 3: allreduce failed: {e}");
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_mpi_from_group");
}
