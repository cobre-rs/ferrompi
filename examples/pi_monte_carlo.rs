//! Monte Carlo Pi Estimation - A classic parallel computing example.
//!
//! This example estimates the value of π by randomly sampling points in a
//! unit square and counting how many fall inside a quarter circle.
//!
//! Run with: mpiexec -n 4 cargo run --release --example pi_monte_carlo
//!
//! The more processes and samples, the more accurate the estimate.

use ferrompi::{Mpi, ReduceOp, Result};
// rand 0.10 moved `random()` from `Rng` to the `RngExt` extension trait.
use rand::RngExt;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    // Total number of samples (distributed across all processes)
    let total_samples: u64 = 100_000_000;
    let samples_per_process = total_samples / size as u64;

    if rank == 0 {
        println!("╔════════════════════════════════════════════════════╗");
        println!("║       Monte Carlo Pi Estimation with FerroMPI      ║");
        println!("╠════════════════════════════════════════════════════╣");
        println!("║ Processes: {:>8}                                ║", size);
        println!(
            "║ Total samples: {:>12}                        ║",
            total_samples
        );
        println!(
            "║ Samples/process: {:>10}                        ║",
            samples_per_process
        );
        println!("╚════════════════════════════════════════════════════╝");
        println!();
    }

    world.barrier()?;

    // Start timing
    let start_time = Mpi::wtime();

    // Each process gets a different random seed based on rank
    let mut rng = rand::rng();

    // Count points inside the quarter circle
    let mut local_inside: u64 = 0;

    for _ in 0..samples_per_process {
        let x: f64 = rng.random();
        let y: f64 = rng.random();

        // Check if point is inside quarter circle (x² + y² ≤ 1)
        if x * x + y * y <= 1.0 {
            local_inside += 1;
        }
    }

    // Convert to f64 for reduction (MPI doesn't have u64 reduction)
    let local_inside_f64 = local_inside as f64;

    // Reduce all counts to rank 0
    let send = [local_inside_f64];
    let mut recv = [0.0];
    world.reduce(&send, &mut recv, ReduceOp::Sum, 0)?;
    let global_inside_f64 = recv[0];

    let elapsed = Mpi::wtime() - start_time;

    // Also get global timing statistics
    world.reduce(&[elapsed], [0.0].as_mut_slice(), ReduceOp::Max, 0)?;
    let max_time = recv[0];
    world.reduce(&[elapsed], &mut recv, ReduceOp::Min, 0)?;
    let min_time = recv[0];

    if rank == 0 {
        let global_inside = global_inside_f64 as u64;
        let actual_samples = samples_per_process * size as u64;

        // π/4 = (points inside circle) / (total points)
        // π = 4 * (points inside circle) / (total points)
        let pi_estimate = 4.0 * global_inside as f64 / actual_samples as f64;
        let error = (pi_estimate - std::f64::consts::PI).abs();
        let relative_error = error / std::f64::consts::PI * 100.0;

        println!("Results:");
        println!("--------");
        println!("  Points inside quarter circle: {}", global_inside);
        println!("  Total points sampled: {}", actual_samples);
        println!();
        println!("  Estimated π: {:.10}", pi_estimate);
        println!("  Actual π:    {:.10}", std::f64::consts::PI);
        println!("  Error:       {:.10} ({:.6}%)", error, relative_error);
        println!();
        println!("Performance:");
        println!("-----------");
        println!("  Total time: {:.4}s", elapsed);
        println!("  Min time: {:.4}s", min_time);
        println!("  Max time: {:.4}s", max_time);
        println!("  Samples/second: {:.2e}", actual_samples as f64 / elapsed);
        println!(
            "  Samples/second/process: {:.2e}",
            samples_per_process as f64 / elapsed
        );
    }

    // Broadcast the result to all processes for verification
    let mut pi_estimate = if rank == 0 {
        4.0 * global_inside_f64 / (samples_per_process * size as u64) as f64
    } else {
        0.0
    };

    let mut pi_buf = [pi_estimate];
    world.broadcast(&mut pi_buf, 0)?;
    pi_estimate = pi_buf[0];

    // Each process verifies the result is reasonable
    assert!(
        (pi_estimate - std::f64::consts::PI).abs() < 0.01,
        "Pi estimate {} is too far from actual value",
        pi_estimate
    );

    world.barrier()?;

    if rank == 0 {
        println!("\n✓ All processes verified the result!");
    }

    Ok(())
}
