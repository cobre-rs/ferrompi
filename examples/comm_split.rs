//! Communicator split example - split world into even/odd rank sub-communicators.
//!
//! Demonstrates `MPI_Comm_split` by partitioning `MPI_COMM_WORLD` into two
//! groups based on rank parity: even ranks (color 0) and odd ranks (color 1).
//! Each sub-communicator then performs an independent allreduce to verify
//! the split is correct.
//!
//! Run with: mpiexec -n 4 cargo run --example comm_split
// mpi-test: np=2..

use ferrompi::{Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This example requires at least 2 processes");
        }
        return Ok(());
    }

    // Split world into even (color=0) and odd (color=1) sub-communicators.
    // The key parameter controls rank ordering within the new communicator;
    // using the world rank preserves the original relative ordering.
    let color = rank % 2;
    let sub = world
        .split(color, rank)?
        .expect("split with valid color should return a communicator");

    let sub_rank = sub.rank();
    let sub_size = sub.size();
    let group_name = if color == 0 { "even" } else { "odd" };

    println!(
        "World rank {}/{}: {} group — sub-communicator rank {}/{}",
        rank, size, group_name, sub_rank, sub_size,
    );

    // Verify the sub-communicator works by performing an allreduce within it.
    // Each process contributes its world rank; the sum should equal the sum of
    // all world ranks that share the same parity.
    let local_sum = world.allreduce_scalar(rank as f64, ReduceOp::Sum)?;
    let sub_sum = sub.allreduce_scalar(rank as f64, ReduceOp::Sum)?;

    // Compute expected sub-communicator sum based on parity
    let expected_sub_sum: f64 = (0..size).filter(|r| r % 2 == color).map(|r| r as f64).sum();

    assert!(
        (sub_sum - expected_sub_sum).abs() < f64::EPSILON,
        "Rank {}: sub-communicator allreduce mismatch: got {}, expected {}",
        rank,
        sub_sum,
        expected_sub_sum,
    );

    world.barrier()?;

    if rank == 0 {
        println!(
            "\nComm split test passed! World sum={}, even sum={}, odd sum={}",
            local_sum,
            expected_sub_sum,
            (0..size)
                .filter(|r| r % 2 == 1)
                .map(|r| r as f64)
                .sum::<f64>(),
        );
    }

    // sub-communicator is freed on drop; MPI is finalized when `mpi` is dropped
    Ok(())
}
