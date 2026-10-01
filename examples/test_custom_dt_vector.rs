//! Integration test for `CustomDatatype::vector`.
//!
//! Exercises:
//! - Successful construction and `raw_handle() >= 0`
//! - Count=0 returns an MPI error or a valid zero-size type
//!   (implementation-defined; both outcomes are accepted)
//! - Indexed-basetype rejection returns `Error::InvalidOp` before any FFI call
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_vector
// mpi-test: np=2

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, MpiErrorClass};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: vector(3, 2, 5, F64) returns Ok with raw_handle() >= 0
    // ========================================================================
    {
        match CustomDatatype::vector(3, 2, 5, DatatypeTag::F64) {
            Ok(v) => {
                let h = v.raw_handle();
                if h >= 0 {
                    if rank == 0 {
                        println!("PASS: Test 1 — vector(3, 2, 5, F64) raw_handle = {h}");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 1 — raw_handle = {h}, expected >= 0");
                    local_ok = false;
                }
                // v drops here, freeing the MPI handle
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
        match CustomDatatype::vector(0, 2, 5, DatatypeTag::F64) {
            Err(Error::Mpi { class, .. }) => {
                let accepted = matches!(class, MpiErrorClass::Count | MpiErrorClass::Arg);
                if accepted {
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
            Ok(v) => {
                // Some MPI implementations accept count=0 and produce a valid
                // zero-size datatype. This is technically conforming; accept it
                // but emit a note.
                if rank == 0 {
                    println!(
                        "NOTE: Test 2 — vector(0, 2, 5, F64) succeeded (raw_handle={}) — \
                         MPI implementation accepts count=0 (conforming)",
                        v.raw_handle()
                    );
                }
            }
        }
    }

    // ========================================================================
    // Test 3: vector(3, 2, 5, FloatInt) returns Err(Error::InvalidOp) without
    //         calling into MPI (pre-FFI validation in the Rust wrapper)
    // ========================================================================
    {
        match CustomDatatype::vector(3, 2, 5, DatatypeTag::FloatInt) {
            Err(Error::InvalidOp) => {
                if rank == 0 {
                    println!("PASS: Test 3 — vector(3, 2, 5, FloatInt) returned Err(InvalidOp)");
                }
            }
            other => {
                eprintln!(
                    "rank {rank}: FAIL Test 3 — expected Err(InvalidOp), got: {:?}",
                    other.err()
                );
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 4: vector with blocklength=0 is accepted (produces an empty-block type)
    // ========================================================================
    {
        match CustomDatatype::vector(3, 0, 5, DatatypeTag::F64) {
            Ok(v) => {
                if v.raw_handle() >= 0 {
                    if rank == 0 {
                        println!(
                            "PASS: Test 4 — vector(3, 0, 5, F64) accepted (raw_handle={})",
                            v.raw_handle()
                        );
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 4 — vector(3, 0, 5, F64) raw_handle < 0");
                    local_ok = false;
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
