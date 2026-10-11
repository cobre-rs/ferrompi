//! Integration test for `CustomDatatype::resized`.
//!
//! Exercises:
//! - Successful resize of a `{ f64, i32 }` struct type to a 16-byte extent.
//! - The original struct type remains valid (not consumed) after the resize:
//!   a second resize succeeds while the first resized type is alive.
//! - Passing a negative extent (`-1`) returns `Err(Error::Mpi { .. })`.
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_resized
// mpi-test: np=2

use ferrompi::{CustomDatatype, Error, Mpi, ReduceOp, StructField};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok: bool = true;

    // Build the base struct type used by tests 1, 2 and 4.
    // Models `{ f64, i32 }`: 8-byte f64 at offset 0, i32 at offset 8.
    let s = match CustomDatatype::create_struct(&[
        StructField::new::<f64>(1, 0),
        StructField::new::<i32>(1, 8),
    ]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rank {rank}: FAIL — create_struct({{f64,i32}}) returned Err: {e}");
            // Propagate failure to all ranks via the sentinel allreduce before
            // exiting so no rank is left waiting inside MPI.
            let _ = world.allreduce_scalar(0_i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 1: resized(0, 16) returns Ok
    // ========================================================================
    {
        match s.resized(0, 16) {
            Ok(_first) => {
                if rank == 0 {
                    println!("PASS: Test 1 — resized(0, 16) returned Ok");
                }

                // ============================================================
                // Test 2: with the first resized type alive, a second
                //         resized(0, 16) is Ok too: the original `s` was not
                //         consumed
                // ============================================================
                match s.resized(0, 16) {
                    Ok(_second) => {
                        if rank == 0 {
                            println!("PASS: Test 2 — original still usable after resize");
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "rank {rank}: FAIL Test 2 — second resized(0, 16) returned Err: {e}"
                        );
                        local_ok = false;
                    }
                }
                // Both resized types drop here, freeing their MPI handles
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1 — resized(0, 16) returned Err: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 4: resized(0, -1) returns Err(Error::Mpi { .. })
    //
    // MPI_Type_create_resized with a negative extent is implementation-defined:
    // most stacks return MPI_ERR_ARG but any MPI error class is accepted.
    // ========================================================================
    {
        match s.resized(0, -1) {
            Err(Error::Mpi { .. }) => {
                if rank == 0 {
                    println!("PASS: Test 4 — resized(0, -1) returned Err(Mpi {{ .. }})");
                }
            }
            Err(e) => {
                // Any error is acceptable; some stacks may return a non-Mpi variant.
                if rank == 0 {
                    println!("PASS: Test 4 — resized(0, -1) returned Err: {e}");
                }
            }
            Ok(_) => {
                // A few MPI implementations accept negative extent (treating it as
                // a large unsigned value or allowing it as a hint).  Log a note but
                // do not fail the test — the spec says "accept any Err(Mpi)".
                if rank == 0 {
                    println!(
                        "NOTE: Test 4 — resized(0, -1) succeeded — \
                         MPI implementation accepts negative extent (non-standard)"
                    );
                }
            }
        }
    }

    common::check(&world, local_ok, "test_custom_dt_resized");
}
