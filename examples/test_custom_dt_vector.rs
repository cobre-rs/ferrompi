//! Integration test for `CustomDatatype::vector`.
//!
//! Exercises:
//! - Successful construction
//! - Count=0 returns an MPI error or a valid zero-size type
//!   (implementation-defined; both outcomes are accepted)
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_vector
// mpi-test: np=2

use ferrompi::{CustomDatatype, Error, Mpi, MpiErrorClass};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: vector(3, 2, 5, F64) returns Ok
    // ========================================================================
    {
        match CustomDatatype::vector::<f64>(3, 2, 5) {
            Ok(_) => {
                if rank == 0 {
                    println!("PASS: Test 1 — vector(3, 2, 5, F64) returned Ok");
                }
                // The datatype drops at the end of the match, freeing the MPI handle
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1 — vector(3, 2, 5, F64) returned Err: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: vector(0, 2, 5, F64) returns an MPI error or a valid type.
    //
    // MPI_Type_vector with count=0 returns MPI_ERR_COUNT on most
    // implementations, but some (e.g., Open MPI >= 5.0) accept it and produce
    // a valid zero-size datatype. Both outcomes are accepted.
    // ========================================================================
    {
        match CustomDatatype::vector::<f64>(0, 2, 5) {
            Err(Error::Mpi { class, .. }) => {
                if matches!(class, MpiErrorClass::Count | MpiErrorClass::Arg) {
                    if rank == 0 {
                        println!("PASS: Test 2 — vector(0, 2, 5, F64) returned Err({class:?}) (implementation-defined Count or Arg)");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 2 — vector(0, 2, 5, F64) returned unexpected class {class:?}");
                    local_ok = false;
                }
            }
            Err(e) => {
                eprintln!(
                    "rank {rank}: FAIL Test 2 — vector(0, 2, 5, F64) returned unexpected error: {e}"
                );
                local_ok = false;
            }
            Ok(_) => {
                // Some MPI implementations accept count=0 and produce a valid
                // zero-size datatype. This is technically conforming; accept it
                // but emit a note.
                if rank == 0 {
                    println!(
                        "NOTE: Test 2 — vector(0, 2, 5, F64) succeeded — \
                         MPI implementation accepts count=0 (conforming)"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Test 4: vector with blocklength=0 is accepted (produces an empty-block type)
    // ========================================================================
    {
        match CustomDatatype::vector::<f64>(3, 0, 5) {
            Ok(_) => {
                if rank == 0 {
                    println!("PASS: Test 4 — vector(3, 0, 5, F64) accepted");
                }
            }
            Err(e) => {
                eprintln!(
                    "rank {rank}: FAIL Test 4 — vector(3, 0, 5, F64) returned unexpected error: {e}"
                );
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_custom_dt_vector");
}
