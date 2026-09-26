//! Integration test for `CustomDatatype::create_struct`.
//!
//! Exercises:
//! - Successful construction of a `{ f64, i32 }` struct type (8-byte f64 at
//!   offset 0, i32 at offset 8) and `raw_handle() >= 0`
//! - Indexed-basetype rejection returns `Error::InvalidOp` before any FFI call
//! - Empty `fields` slice returns `Err` with class `Arg` without calling into
//!   MPI
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
                if h < 0 {
                    eprintln!("rank {rank}: FAIL Test 1 — raw_handle = {h}, expected >= 0");
                    local_ok = false;
                } else if rank == 0 {
                    println!("PASS: Test 1 — create_struct({{f64,i32}}) raw_handle = {h}");
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
    // Test 3: create_struct(&[]) returns Err with class Arg.
    //
    // The C shim rejects an empty field list before the stack arrays are
    // filled, returning MPI_ERR_ARG without calling into MPI, on every MPI
    // implementation.
    // ========================================================================
    {
        match CustomDatatype::create_struct(&[]) {
            Err(Error::Mpi {
                class: MpiErrorClass::Arg,
                ..
            }) => {
                if rank == 0 {
                    println!("PASS: Test 3 — create_struct(&[]) returned Err(class: Arg)");
                }
            }
            other => {
                eprintln!(
                    "rank {rank}: FAIL Test 3 — expected Err(class: Arg), got: {:?}",
                    other
                );
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_custom_dt_struct");
}
