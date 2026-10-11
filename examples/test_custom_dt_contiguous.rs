//! Integration test for `CustomDatatype::contiguous`.
//!
//! Exercises:
//! - Successful construction
//! - Count=0 returns an MPI error (`Count` or `Arg` class, implementation-defined)
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_contiguous
// mpi-test: np=2

use ferrompi::{CustomDatatype, Error, Mpi, MpiDatatype, MpiErrorClass};

mod common;

/// Builds `contiguous::<T>(4)` and drops it; reports a failure on this rank.
fn builds<T: MpiDatatype>(rank: i32, name: &str) -> bool {
    match CustomDatatype::contiguous::<T>(4) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 4 — contiguous::<{name}>(4) failed: {e}");
            false
        }
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: contiguous(10, F64) returns Ok
    // ========================================================================
    {
        match CustomDatatype::contiguous::<f64>(10) {
            Ok(_) => {
                if rank == 0 {
                    println!("PASS: Test 1 — contiguous(10, F64) returned Ok");
                }
                // The datatype drops at the end of the match, freeing the MPI handle
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1 — contiguous(10, F64) returned Err: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: contiguous(0, F64) returns an MPI error.
    //
    // MPI_Type_contiguous with count=0 returns MPI_ERR_COUNT on most
    // implementations, but MPI_ERR_ARG on some (e.g., MPICH 4.x). Both are
    // acceptable per the MPI standard (the standard says count must be >= 0,
    // but error reporting is implementation-defined).
    // ========================================================================
    {
        match CustomDatatype::contiguous::<f64>(0) {
            Err(Error::Mpi { class, .. }) => {
                let accepted = matches!(class, MpiErrorClass::Count | MpiErrorClass::Arg);
                if accepted {
                    if rank == 0 {
                        println!("PASS: Test 2 — contiguous(0, F64) returned Err({class:?}) (implementation-defined Count or Arg)");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 2 — contiguous(0, F64) returned unexpected class {class:?}");
                    local_ok = false;
                }
            }
            Err(e) => {
                eprintln!(
                    "rank {rank}: FAIL Test 2 — contiguous(0, F64) returned unexpected error: {e}"
                );
                local_ok = false;
            }
            Ok(_) => {
                // Some MPI implementations (notably Open MPI >= 5.0) accept
                // count=0 and produce a valid zero-size datatype. This is
                // technically conforming; accept it but emit a note.
                if rank == 0 {
                    println!(
                        "NOTE: Test 2 — contiguous(0, F64) succeeded — \
                         MPI implementation accepts count=0 (conforming)"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Test 4: Multiple CustomDatatype instances all drop cleanly
    // ========================================================================
    {
        let success = [
            builds::<f32>(rank, "f32"),
            builds::<f64>(rank, "f64"),
            builds::<i32>(rank, "i32"),
            builds::<i64>(rank, "i64"),
            builds::<u8>(rank, "u8"),
            builds::<u32>(rank, "u32"),
            builds::<u64>(rank, "u64"),
        ]
        .iter()
        .all(|&built| built);
        if success {
            if rank == 0 {
                println!("PASS: Test 4 — all primitive base types construct and drop cleanly");
            }
        } else {
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_custom_dt_contiguous");
}
