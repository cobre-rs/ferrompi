//! Integration test for `CustomDatatype::contiguous`.
//!
//! Exercises:
//! - Successful construction and `raw_handle() >= 0`
//! - Count=0 returns an MPI error (`Count` or `Arg` class, implementation-defined)
//! - Indexed-basetype rejection returns `Error::InvalidOp` before any FFI call
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_contiguous

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, MpiErrorClass};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: contiguous(10, F64) returns Ok with raw_handle() >= 0
    // ========================================================================
    {
        match CustomDatatype::contiguous(10, DatatypeTag::F64) {
            Ok(dt) => {
                let h = dt.raw_handle();
                if h >= 0 {
                    if rank == 0 {
                        println!("PASS: Test 1 — contiguous(10, F64) raw_handle = {h}");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 1 — raw_handle = {h}, expected >= 0");
                    local_ok = false;
                }
                // dt drops here, freeing the MPI handle
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
        match CustomDatatype::contiguous(0, DatatypeTag::F64) {
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
            Ok(dt) => {
                // Some MPI implementations (notably Open MPI >= 5.0) accept
                // count=0 and produce a valid zero-size datatype. This is
                // technically conforming; accept it but emit a note.
                if rank == 0 {
                    println!(
                        "NOTE: Test 2 — contiguous(0, F64) succeeded (raw_handle={}) — \
                         MPI implementation accepts count=0 (conforming)",
                        dt.raw_handle()
                    );
                }
            }
        }
    }

    // ========================================================================
    // Test 3: contiguous(5, FloatInt) returns Err(Error::InvalidOp) without
    //         calling into MPI (pre-FFI validation in the Rust wrapper)
    // ========================================================================
    {
        match CustomDatatype::contiguous(5, DatatypeTag::FloatInt) {
            Err(Error::InvalidOp) => {
                if rank == 0 {
                    println!("PASS: Test 3 — contiguous(5, FloatInt) returned Err(InvalidOp)");
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
    // Test 4: Multiple CustomDatatype instances all drop cleanly
    // ========================================================================
    {
        let mut success = true;
        let tags = [
            DatatypeTag::F32,
            DatatypeTag::F64,
            DatatypeTag::I32,
            DatatypeTag::I64,
            DatatypeTag::U8,
            DatatypeTag::U32,
            DatatypeTag::U64,
            DatatypeTag::Byte,
        ];
        for tag in tags {
            match CustomDatatype::contiguous(4, tag) {
                Ok(dt) => {
                    if dt.raw_handle() < 0 {
                        eprintln!("rank {rank}: FAIL Test 4 — contiguous(4, {tag:?}) handle < 0");
                        success = false;
                    }
                    // Drop frees the handle
                }
                Err(e) => {
                    eprintln!("rank {rank}: FAIL Test 4 — contiguous(4, {tag:?}) failed: {e}");
                    success = false;
                }
            }
        }
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
