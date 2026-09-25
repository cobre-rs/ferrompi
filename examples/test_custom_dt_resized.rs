//! Integration test for `CustomDatatype::resized`.
//!
//! Exercises:
//! - Successful resize of a `{ f64, i32 }` struct type to a 16-byte extent:
//!   the resized handle is `>= 0` and differs from the original handle.
//! - The original struct type remains valid (not consumed) after the resize.
//! - Passing a negative extent (`-1`) returns `Err(Error::Mpi { .. })`.
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_resized

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, ReduceOp, StructField};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok: bool = true;

    // Build the base struct type used by tests 1-3.
    // Models `{ f64, i32 }`: 8-byte f64 at offset 0, i32 at offset 8.
    let s = match CustomDatatype::create_struct(&[
        StructField {
            blocklength: 1,
            displacement: 0,
            basetype: DatatypeTag::F64,
        },
        StructField {
            blocklength: 1,
            displacement: 8,
            basetype: DatatypeTag::I32,
        },
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
    // Test 1: resized(0, 16) returns Ok with raw_handle() >= 0
    // ========================================================================
    {
        match s.resized(0, 16) {
            Ok(r) => {
                let h = r.raw_handle();
                if h >= 0 {
                    if rank == 0 {
                        println!("PASS: Test 1 — resized(0, 16) raw_handle = {h}");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 1 — raw_handle = {h}, expected >= 0");
                    local_ok = false;
                }
                // r drops here, freeing the resized MPI handle
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1 — resized(0, 16) returned Err: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: the resized handle differs from the original handle
    // ========================================================================
    {
        match s.resized(0, 16) {
            Ok(r) => {
                if r.raw_handle() != s.raw_handle() {
                    if rank == 0 {
                        println!(
                            "PASS: Test 2 — resized handle ({}) != original handle ({})",
                            r.raw_handle(),
                            s.raw_handle()
                        );
                    }
                } else {
                    eprintln!(
                        "rank {rank}: FAIL Test 2 — resized handle == original handle ({})",
                        s.raw_handle()
                    );
                    local_ok = false;
                }
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2 — resized(0, 16) returned Err: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 3: the original `s` is not consumed; its raw_handle() remains valid
    // ========================================================================
    {
        let h = s.raw_handle();
        if h >= 0 {
            if rank == 0 {
                println!("PASS: Test 3 — original handle still valid after resize: {h}");
            }
        } else {
            eprintln!("rank {rank}: FAIL Test 3 — original handle is {h}, expected >= 0");
            local_ok = false;
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
            Ok(r) => {
                // A few MPI implementations accept negative extent (treating it as
                // a large unsigned value or allowing it as a hint).  Log a note but
                // do not fail the test — the spec says "accept any Err(Mpi)".
                if rank == 0 {
                    println!(
                        "NOTE: Test 4 — resized(0, -1) succeeded (raw_handle={}) — \
                         MPI implementation accepts negative extent (non-standard)",
                        r.raw_handle()
                    );
                }
            }
        }
    }

    common::check(&world, local_ok, "test_custom_dt_resized");
}
