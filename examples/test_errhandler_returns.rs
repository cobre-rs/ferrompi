//! Integration test for MPI_ERRORS_RETURN error handler installation.
//!
//! Verifies that MPI errors on COMM_WORLD, derived communicators, and
//! MPI_COMM_SELF (the handler MPI uses for errors not associated with any
//! communicator, such as group construction) return as `Err` carrying the
//! exact error class of the MPI library the test ran on, rather than
//! aborting the process via the default `MPI_ERRORS_ARE_FATAL` handler.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_errhandler_returns
// mpi-test: np=2..

use ferrompi::{Error, Mpi, MpiErrorClass, ReduceOp};

mod common;

/// Returns `true` only if `result` is `Err(Error::Mpi { class, .. })` with
/// `class == expected`; otherwise prints the expected and actual outcome.
fn expect_class<T>(result: Result<T, Error>, expected: MpiErrorClass) -> bool {
    match result {
        Err(Error::Mpi { class, .. }) if class == expected => true,
        Err(e) => {
            println!("expected class {expected:?}, got error: {e:?}");
            false
        }
        Ok(_) => {
            println!("expected class {expected:?}, got Ok");
            false
        }
    }
}

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
    // Test 1: broadcast with invalid root returns Err(class: Root)
    // ========================================================================
    {
        let mut data = vec![0.0f64; 10];
        let ok = expect_class(world.broadcast(&mut data, 999), MpiErrorClass::Root);
        common::check(&world, ok, "broadcast root=999 -> Root");
        if rank == 0 {
            println!("PASS: broadcast root=999 -> Root");
        }
    }

    // ========================================================================
    // Test 2: send with dest = size() (out-of-range rank) returns
    //         Err(class: Rank)
    //
    // A local check (detectable before any network I/O), so only rank 0
    // issues the send; every rank still verifies through common::check.
    // ========================================================================
    {
        let ok = if rank == 0 {
            let data = vec![1.0f64; 4];
            expect_class(world.send(&data, size, 0), MpiErrorClass::Rank)
        } else {
            true
        };
        common::check(&world, ok, "send dest=size -> Rank");
        if rank == 0 {
            println!("PASS: send dest=size -> Rank");
        }
    }

    // ========================================================================
    // Test 3: errors on a dup'd communicator also return with class Root
    //
    // dup inherits the errhandler; this verifies the belt-and-braces install
    // in ferrompi_comm_dup is in effect.
    // ========================================================================
    {
        let dup = world.duplicate().expect("comm_dup failed");
        let mut data = vec![0.0f64; 10];
        let ok = expect_class(dup.broadcast(&mut data, 999), MpiErrorClass::Root);
        common::check(&world, ok, "dup broadcast root=999 -> Root");
        if rank == 0 {
            println!("PASS: dup broadcast root=999 -> Root");
        }
    }

    // ========================================================================
    // Test 4: errors on a split communicator also return with class Root
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
        let ok = expect_class(split_comm.broadcast(&mut data, 999), MpiErrorClass::Root);
        common::check(&world, ok, "split broadcast root=999 -> Root");
        if rank == 0 {
            println!("PASS: split broadcast root=999 -> Root");
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
        let ok = expect_class(group.include(&[999]), MpiErrorClass::Rank);
        common::check(&world, ok, "group include rank=999 -> Rank");
        if rank == 0 {
            println!("PASS: group include rank=999 -> Rank");
        }
    }

    // ========================================================================
    // Test 6: a bitwise reduction on f64 is not defined for floating-point
    //         types and returns Err(class: Op)
    // ========================================================================
    {
        let send = [1.0f64];
        let mut recv = [0.0f64];
        let ok = expect_class(
            world.allreduce(&send, &mut recv, ReduceOp::BitwiseOr),
            MpiErrorClass::Op,
        );
        common::check(&world, ok, "allreduce f64 BitwiseOr -> Op");
        if rank == 0 {
            println!("PASS: allreduce f64 BitwiseOr -> Op");
        }
    }

    // ========================================================================
    // Test 7: a truncated blocking receive returns Err(class: Truncate)
    //
    // A truncated nonblocking receive of a self-message returns success on
    // Open MPI 4.1.6 and 5.0.7, while the blocking form reports
    // MPI_ERR_TRUNCATE everywhere; this posts a nonblocking self-send and
    // completes it with an undersized blocking recv.
    // ========================================================================
    {
        let outgoing = [1i32; 4];
        let req = world.isend(&outgoing, rank, 7).expect("isend failed");
        let mut incoming = [0i32; 1];
        let ok = expect_class(world.recv(&mut incoming, rank, 7), MpiErrorClass::Truncate);
        req.wait().expect("wait failed");
        common::check(&world, ok, "recv truncate -> Truncate");
        if rank == 0 {
            println!("PASS: recv truncate -> Truncate");
        }
    }

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All errhandler tests passed! (7 tests)");
        println!("========================================");
    }
}
