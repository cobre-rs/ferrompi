//! Integration test for custom-datatype point-to-point communication.
//!
//! Exercises:
//! - Blocking `send_custom` / `recv_custom` round-trip with a `{ f64, i32 }`
//!   struct datatype built via `CustomDatatype::create_struct`.
//! - Nonblocking `isend_custom` / `irecv_custom` round-trip with the same
//!   struct datatype, completed via `Request::wait`.
//! - Mismatched-buffer-length receive returns `Err(Error::Mpi { .. })`.
//!
//! All assertions are protected by a sentinel allreduce(Min) before any
//! `process::exit` call so that no rank exits while others are still inside MPI.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_custom_dt_p2p
// mpi-test: np=2

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, StructField};

mod common;

/// A simple `#[repr(C)]` struct whose layout we model with `create_struct`.
///
/// Layout: f64 at offset 0 (8 bytes), i32 at offset 8 (4 bytes).
/// Total natural size = 12 bytes; MPI will see the same offsets.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
struct Pair {
    v: f64,
    i: i32,
}

/// Build the custom datatype for `Pair`.
fn make_pair_datatype() -> CustomDatatype {
    CustomDatatype::create_struct(&[
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
    ])
    .expect("create_struct for Pair failed")
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    assert!(
        world.size() >= 2,
        "test_custom_dt_p2p requires at least 2 processes, got {}",
        world.size()
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: Blocking send_custom / recv_custom round-trip
    // ========================================================================
    {
        let dt = make_pair_datatype();

        if rank == 0 {
            let buf = [Pair {
                v: 1.23456789,
                i: 42,
            }];
            if let Err(e) = world.send_custom(&buf, &dt, 1, 0) {
                eprintln!("rank 0: FAIL Test 1 — send_custom returned Err: {e}");
                local_ok = false;
            }
        } else if rank == 1 {
            let mut buf = [Pair { v: 0.0, i: 0 }];
            match world.recv_custom(&mut buf, &dt, 0, 0) {
                Ok(status) => {
                    let expected = Pair {
                        v: 1.23456789,
                        i: 42,
                    };
                    if buf[0] != expected {
                        eprintln!(
                            "rank 1: FAIL Test 1 — recv_custom payload mismatch: got {:?}, expected {:?}",
                            buf[0], expected
                        );
                        local_ok = false;
                    } else if status.count != 1 {
                        eprintln!(
                            "rank 1: FAIL Test 1 — recv_custom status.count = {}, expected 1",
                            status.count
                        );
                        local_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("rank 1: FAIL Test 1 — recv_custom returned Err: {e}");
                    local_ok = false;
                }
            }
        }

        world.barrier().expect("barrier after Test 1 failed");
        if rank == 0 {
            println!("PASS: Test 1 — blocking send_custom / recv_custom");
        }
    }

    // ========================================================================
    // Test 2: Nonblocking isend_custom / irecv_custom round-trip
    // ========================================================================
    {
        let dt = make_pair_datatype();

        if rank == 0 {
            let buf = [Pair { v: 2.71, i: 99 }];
            match world.isend_custom(&buf, &dt, 1, 1) {
                Ok(req) => {
                    if let Err(e) = req.wait() {
                        eprintln!("rank 0: FAIL Test 2 — isend_custom req.wait() Err: {e}");
                        local_ok = false;
                    }
                }
                Err(e) => {
                    eprintln!("rank 0: FAIL Test 2 — isend_custom returned Err: {e}");
                    local_ok = false;
                }
            }
        } else if rank == 1 {
            let mut buf = [Pair { v: 0.0, i: 0 }];
            match world.irecv_custom(&mut buf, &dt, 0, 1) {
                Ok(req) => {
                    if let Err(e) = req.wait() {
                        eprintln!("rank 1: FAIL Test 2 — irecv_custom req.wait() Err: {e}");
                        local_ok = false;
                    } else {
                        let expected = Pair { v: 2.71, i: 99 };
                        if buf[0] != expected {
                            eprintln!(
                                "rank 1: FAIL Test 2 — irecv_custom payload mismatch: got {:?}, expected {:?}",
                                buf[0], expected
                            );
                            local_ok = false;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("rank 1: FAIL Test 2 — irecv_custom returned Err: {e}");
                    local_ok = false;
                }
            }
        }

        world.barrier().expect("barrier after Test 2 failed");
        if rank == 0 {
            println!("PASS: Test 2 — nonblocking isend_custom / irecv_custom");
        }
    }

    // ========================================================================
    // Test 3: Mismatched-buffer-length recv returns Err(Mpi)
    //
    // Rank 0 sends 1 Pair element; rank 1 posts a recv with a 0-element buffer.
    // MPI must return an error (MPI_ERR_TRUNCATE or another class — both are
    // accepted).
    // ========================================================================
    {
        let dt = make_pair_datatype();

        if rank == 0 {
            let buf = [Pair { v: 1.0, i: 1 }];
            // Send proceeds normally; the error manifests on the receive side.
            if let Err(e) = world.send_custom(&buf, &dt, 1, 2) {
                eprintln!("rank 0: FAIL Test 3 — send_custom (truncation test) Err: {e}");
                local_ok = false;
            }
        } else if rank == 1 {
            // Receive into a 0-element buffer — MPI must report an error.
            let mut buf: [Pair; 0] = [];
            match world.recv_custom(&mut buf, &dt, 0, 2) {
                Err(Error::Mpi { .. }) => {
                    // Any MPI error class is acceptable (truncate, count, etc.)
                }
                Err(e) => {
                    eprintln!("rank 1: FAIL Test 3 — expected Err(Mpi), got different error: {e}");
                    local_ok = false;
                }
                Ok(_) => {
                    eprintln!(
                        "rank 1: FAIL Test 3 — recv_custom with 0-element buffer returned Ok, \
                         expected an MPI error"
                    );
                    local_ok = false;
                }
            }
        }

        world.barrier().expect("barrier after Test 3 failed");
        if rank == 0 {
            println!("PASS: Test 3 — mismatched-buffer recv returns Err(Mpi)");
        }
    }

    common::check(&world, local_ok, "test_custom_dt_p2p");
}
