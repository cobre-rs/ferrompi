//! Integration test for `CustomDatatype::create_struct`.
//!
//! Exercises:
//! - Successful construction of a `{ f64, i32 }` struct type (8-byte f64 at
//!   offset 0, i32 at offset 8)
//! - Empty `fields` slice returns `Err` with class `Arg` without calling into
//!   MPI
//! - Drop frees the underlying MPI handle (no double-free on exit)
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_struct
// mpi-test: np=2

use ferrompi::{CustomDatatype, Error, Mpi, MpiErrorClass, StructField};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    let mut local_ok = true;

    // ========================================================================
    // Test 1: create_struct for a { f64, i32 } layout returns Ok
    // ========================================================================
    {
        let fields = [StructField::new::<f64>(1, 0), StructField::new::<i32>(1, 8)];
        match CustomDatatype::create_struct(&fields) {
            Ok(_) => {
                if rank == 0 {
                    println!("PASS: Test 1 — create_struct({{f64,i32}}) returned Ok");
                }
                // The datatype drops at the end of the match, freeing the MPI handle
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
