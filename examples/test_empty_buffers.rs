//! Regression test for zero-length `u8` buffers in collectives.
//!
//! An empty `Vec<u8>` points at address 1, which some MPI libraries read as
//! `MPI_IN_PLACE`. Every call below passes an empty byte buffer where a
//! process has nothing to send or receive, and must return `Ok`:
//!
//! 1. `gather`, `gather_inplace`, `allgather`, `allreduce`, `allreduce_bytes`,
//!    `reduce`, `scatter` and `alltoallv` with every buffer empty. `alltoallv`
//!    gets one slice as its send and its receive counts, which is when MPICH
//!    refuses a send and a receive buffer at the same address.
//! 2. `gatherv` and `allgatherv` where rank 1 contributes nothing and every
//!    other rank contributes four bytes; the gathered bytes are checked.
//! 3. A persistent `allreduce_init` and a persistent `reduce_init` over empty
//!    buffers, each started and waited once. Open MPI 5 refuses a
//!    `reduce_init` whose two buffers share an address at the root. A library
//!    without persistent collectives must refuse both with
//!    `Error::NotSupported`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_empty_buffers
// mpi-test: np=2..

use ferrompi::{Error, Mpi, PersistentRequest, ReduceOp, Result};

mod common;

/// True iff `request` was built, started and waited, or, on a library without
/// persistent collectives, refused with `Error::NotSupported`.
fn persistent_ok(request: Result<PersistentRequest<u8>>) -> bool {
    if common::has_mpi4_collectives() {
        match request {
            Ok(mut request) => request.start().is_ok() && request.wait().is_ok(),
            Err(_) => false,
        }
    } else {
        matches!(request, Err(Error::NotSupported(_)))
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size() as usize;
    assert!(
        size >= 2,
        "test_empty_buffers requires at least 2 processes, got {size}"
    );

    let empty: Vec<u8> = Vec::new();
    let mut out: Vec<u8> = Vec::new();

    let gather = world.gather(&empty, &mut out, 0);
    common::check(&world, gather.is_ok(), "gather of empty buffers");

    let gather_inplace = world.gather_inplace(&mut out, 0);
    common::check(
        &world,
        gather_inplace.is_ok(),
        "gather_inplace of an empty buffer",
    );

    let allgather = world.allgather(&empty, &mut out);
    common::check(&world, allgather.is_ok(), "allgather of empty buffers");

    let allreduce = world.allreduce(&empty, &mut out, ReduceOp::Max);
    common::check(&world, allreduce.is_ok(), "allreduce of empty buffers");

    let allreduce_bytes = world.allreduce_bytes(&empty, &mut out, ReduceOp::BitwiseOr);
    common::check(
        &world,
        allreduce_bytes.is_ok(),
        "allreduce_bytes of empty buffers",
    );

    let reduce = world.reduce(&empty, &mut out, ReduceOp::Max, 0);
    common::check(&world, reduce.is_ok(), "reduce of empty buffers");

    let scatter = world.scatter(&empty, &mut out, 0);
    common::check(&world, scatter.is_ok(), "scatter of empty buffers");

    let zeros = vec![0i32; size];
    let alltoallv = world.alltoallv(&empty, &zeros, &zeros, &mut out, &zeros, &zeros);
    common::check(&world, alltoallv.is_ok(), "alltoallv of empty buffers");

    // Rank 1 contributes nothing; every other rank contributes four bytes
    // holding its rank.
    let counts: Vec<i32> = (0..size).map(|r| if r == 1 { 0 } else { 4 }).collect();
    let displs = common::displs_from_counts(&counts);
    let mine = vec![rank as u8; counts[rank as usize] as usize];
    let expected: Vec<u8> = (0..size)
        .flat_map(|r| vec![r as u8; counts[r] as usize])
        .collect();

    let mut gathered = vec![0u8; if rank == 0 { expected.len() } else { 0 }];
    let gatherv = world.gatherv(&mine, &mut gathered, &counts, &displs, 0);
    common::check(
        &world,
        gatherv.is_ok() && (rank != 0 || gathered == expected),
        "gatherv with one empty contribution",
    );

    let mut all = vec![0u8; expected.len()];
    let allgatherv = world.allgatherv(&mine, &mut all, &counts, &displs);
    common::check(
        &world,
        allgatherv.is_ok() && all == expected,
        "allgatherv with one empty contribution",
    );

    let allreduce_init = world.allreduce_init(Vec::<u8>::new(), Vec::<u8>::new(), ReduceOp::Max);
    common::check(
        &world,
        persistent_ok(allreduce_init),
        "persistent allreduce of empty buffers",
    );

    let reduce_init = world.reduce_init(Vec::<u8>::new(), Vec::<u8>::new(), ReduceOp::Max, 0);
    common::check(
        &world,
        persistent_ok(reduce_init),
        "persistent reduce of empty buffers",
    );

    if rank == 0 {
        println!("PASS: test_empty_buffers");
    }
}
