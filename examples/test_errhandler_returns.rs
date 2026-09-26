//! Integration test for MPI_ERRORS_RETURN error handler installation.
//!
//! Verifies that MPI errors on COMM_WORLD, derived communicators, and
//! MPI_COMM_SELF (the handler MPI uses for errors not associated with any
//! communicator, such as group construction) return as `Err(Error::Mpi {
//! .. })` rather than aborting the process via the default
//! `MPI_ERRORS_ARE_FATAL` handler.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_errhandler_returns
// mpi-test: np=2..

use ferrompi::{Error, Mpi, MpiErrorClass};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_errhandler_returns requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Test 1: broadcast with invalid root returns Err(Error::Mpi { class: Root })
    //
    // Root 999 is out of range for any realistic communicator size. With the
    // default MPI_ERRORS_ARE_FATAL handler this would abort the process; with
    // MPI_ERRORS_RETURN installed it must return an error code.
    // ========================================================================
    {
        let mut data = vec![0.0f64; 10];
        let result = world.broadcast(&mut data, 999);

        match result {
            Err(Error::Mpi {
                class: MpiErrorClass::Root,
                ..
            }) => {
                // Expected: invalid root error propagated as Err
                if rank == 0 {
                    println!("PASS: broadcast with invalid root returns Err(MpiErrorClass::Root)");
                }
            }
            Err(Error::Mpi { class, code, .. }) => {
                // Some MPI implementations may return a different class for an
                // out-of-range root (e.g. MpiErrorClass::Arg on certain builds).
                // Accept any Err(Mpi) variant as proof the handler is installed.
                if rank == 0 {
                    println!(
                        "PASS: broadcast with invalid root returns Err(Mpi) \
                         (class={class:?}, code={code}); note: class is \
                         implementation-defined for invalid-root errors"
                    );
                }
            }
            Ok(()) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast with invalid root returned Ok — \
                     MPI_ERRORS_RETURN may not be installed"
                );
                std::process::exit(1);
            }
            Err(other) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast with invalid root returned \
                     unexpected error variant: {other:?}"
                );
                std::process::exit(1);
            }
        }
    }

    // ========================================================================
    // Test 2: send with dest = size() (out-of-range rank) returns
    //         Err(Error::Mpi { class: Rank, .. })
    //
    // Only performed on rank 0 to avoid requiring a paired recv. We use
    // MPI_ANY_SOURCE-style: rank 0 attempts to send to a non-existent rank.
    // This is a local check (the error is detectable before any network I/O).
    // ========================================================================
    {
        if rank == 0 {
            let data = vec![1.0f64; 4];
            let invalid_dest = size; // Out-of-range rank
            let result = world.send(&data, invalid_dest, 0);

            match result {
                Err(Error::Mpi {
                    class: MpiErrorClass::Rank,
                    ..
                }) => {
                    println!(
                        "PASS: send to invalid dest={invalid_dest} returns \
                         Err(MpiErrorClass::Rank)"
                    );
                }
                Err(Error::Mpi { class, code, .. }) => {
                    // Accept any Mpi error: some implementations may return
                    // a different class (e.g. Arg) for out-of-range destination.
                    println!(
                        "PASS: send to invalid dest={invalid_dest} returns Err(Mpi) \
                         (class={class:?}, code={code}); note: class is \
                         implementation-defined for out-of-range rank"
                    );
                }
                Ok(()) => {
                    eprintln!(
                        "FAIL: send to invalid dest={invalid_dest} returned Ok — \
                         MPI_ERRORS_RETURN may not be installed on COMM_WORLD"
                    );
                    std::process::exit(1);
                }
                Err(other) => {
                    eprintln!(
                        "FAIL: send to invalid dest={invalid_dest} returned \
                         unexpected error variant: {other:?}"
                    );
                    std::process::exit(1);
                }
            }
        }
        // Barrier so non-root ranks wait for rank 0's test to complete before
        // the communicator is used further.
        world.barrier().expect("barrier failed");
    }

    // ========================================================================
    // Test 3: errors on a dup'd communicator also return (not abort)
    //
    // dup inherits the errhandler; this test verifies the belt-and-braces
    // install in ferrompi_comm_dup is in effect.
    // ========================================================================
    {
        let dup = world.duplicate().expect("comm_dup failed");
        let mut data = vec![0.0f64; 10];
        let result = dup.broadcast(&mut data, 999);

        match result {
            Err(Error::Mpi { .. }) => {
                if rank == 0 {
                    println!("PASS: broadcast on dup'd comm with invalid root returns Err(Mpi)");
                }
            }
            Ok(()) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast on dup'd comm with invalid root \
                     returned Ok"
                );
                std::process::exit(1);
            }
            Err(other) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast on dup'd comm with invalid root \
                     returned unexpected error variant: {other:?}"
                );
                std::process::exit(1);
            }
        }
    }

    // ========================================================================
    // Test 4: errors on a split communicator also return (not abort)
    //
    // Split all ranks into a single group (color=0). The new comm covers all
    // ranks, so an invalid root still triggers MPI_ERR_ROOT.
    // ========================================================================
    {
        let split_comm = world
            .split(0, rank)
            .expect("comm_split failed")
            .expect("expected Some communicator, got None");

        let mut data = vec![0.0f64; 10];
        let result = split_comm.broadcast(&mut data, 999);

        match result {
            Err(Error::Mpi { .. }) => {
                if rank == 0 {
                    println!("PASS: broadcast on split comm with invalid root returns Err(Mpi)");
                }
            }
            Ok(()) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast on split comm with invalid root \
                     returned Ok"
                );
                std::process::exit(1);
            }
            Err(other) => {
                eprintln!(
                    "rank {rank}: FAIL: broadcast on split comm with invalid root \
                     returned unexpected error variant: {other:?}"
                );
                std::process::exit(1);
            }
        }
    }

    // ========================================================================
    // Test 5: group construction is not associated with any communicator
    //
    // MPI_Group_incl takes no communicator argument. Since MPI 4.0, an error
    // raised by such a call is delivered through MPI_COMM_SELF's error
    // handler, whose default is MPI_ERRORS_ARE_FATAL. Rank 999 is out of
    // range for every group size used in this test suite.
    // ========================================================================
    {
        let group = world.group().expect("group failed");
        let result = group.include(&[999]);

        let ok = matches!(
            result,
            Err(Error::Mpi {
                class: MpiErrorClass::Rank,
                ..
            })
        );
        common::check(&world, ok, "group include with rank 999 returns Err(Rank)");
        if rank == 0 {
            println!("PASS: group include with rank 999 returns Err(MpiErrorClass::Rank)");
        }
    }

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All errhandler tests passed! (5 tests)");
        println!("========================================");
    }
}
