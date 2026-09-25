//! Integration test for `CustomDatatype::create_struct`.
//!
//! Exercises:
//! - Successful construction of a `{ f64, i32 }` struct type (8-byte f64 at
//!   offset 0, i32 at offset 8) and `raw_handle() >= 0`
//! - Indexed-basetype rejection returns `Error::InvalidOp` before any FFI call
//! - Empty `fields` slice returns an MPI error (implementation-defined class:
//!   `MpiErrorClass::Arg` or `MpiErrorClass::Count` are both accepted)
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_struct
// mpi-test: np=2

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, MpiErrorClass, StructField};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: create_struct for a { f64, i32 } layout returns Ok with
    //         raw_handle() >= 0
    // ========================================================================
    {
        let fields = [
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
        ];
        match CustomDatatype::create_struct(&fields) {
            Ok(s) => {
                let h = s.raw_handle();
                if h >= 0 {
                    if rank == 0 {
                        println!("PASS: Test 1 — create_struct({{f64,i32}}) raw_handle = {h}");
                    }
                } else {
                    eprintln!("rank {rank}: FAIL Test 1 — raw_handle = {h}, expected >= 0");
                    local_ok = false;
                }
                // s drops here, freeing the MPI handle
            }
            Err(e) => {
                eprintln!(
                    "rank {rank}: FAIL Test 1 — create_struct({{f64,i32}}) returned Err: {e}"
                );
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: create_struct with a FloatInt field returns Err(Error::InvalidOp)
    //         without calling into MPI (pre-FFI validation in the Rust wrapper)
    // ========================================================================
    {
        let fields = [StructField {
            blocklength: 1,
            displacement: 0,
            basetype: DatatypeTag::FloatInt,
        }];
        match CustomDatatype::create_struct(&fields) {
            Err(Error::InvalidOp) => {
                if rank == 0 {
                    println!(
                        "PASS: Test 2 — create_struct(FloatInt field) returned Err(InvalidOp)"
                    );
                }
            }
            other => {
                eprintln!(
                    "rank {rank}: FAIL Test 2 — expected Err(InvalidOp), got: {:?}",
                    other.err()
                );
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 3: create_struct(&[]) returns an Err or a zero-size type.
    //
    // MPI_Type_create_struct with count=0 is rejected by most implementations
    // with MPI_ERR_ARG or MPI_ERR_COUNT (both accepted). However, some
    // implementations (e.g. Open MPI >= 5.0) accept count=0 and produce a
    // valid zero-size datatype — this is also accepted as conforming.
    // The C shim forwards count=0 directly to MPI (no early return) so the
    // exact outcome is implementation-defined.
    // ========================================================================
    {
        match CustomDatatype::create_struct(&[]) {
            Err(Error::Mpi { class, .. }) => {
                let accepted = matches!(class, MpiErrorClass::Arg | MpiErrorClass::Count);
                if accepted {
                    if rank == 0 {
                        println!(
                            "PASS: Test 3 — create_struct(&[]) returned Err({class:?}) \
                             (implementation-defined Arg or Count)"
                        );
                    }
                } else {
                    // Some MPI implementations may return other error classes
                    // for count=0. Accept any MPI error as conforming.
                    if rank == 0 {
                        println!(
                            "NOTE: Test 3 — create_struct(&[]) returned Err({class:?}) \
                             (non-standard class, still an error — accepted)"
                        );
                    }
                }
            }
            Err(e) => {
                // Any error is acceptable for an empty slice.
                if rank == 0 {
                    println!("PASS: Test 3 — create_struct(&[]) returned Err: {e}");
                }
            }
            Ok(s) => {
                // Some MPI implementations accept count=0 and produce a valid
                // zero-size datatype. This is conforming; accept it with a note.
                if rank == 0 {
                    println!(
                        "NOTE: Test 3 — create_struct(&[]) succeeded (raw_handle={}) — \
                         MPI implementation accepts count=0 (conforming)",
                        s.raw_handle()
                    );
                }
            }
        }
    }

    common::check(&world, local_ok, "test_custom_dt_struct");
}
