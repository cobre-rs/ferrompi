//! Persistent collectives example (MPI 4.0+).
//!
//! Demonstrates the use of persistent collectives for iterative algorithms
//! where the same communication pattern is repeated many times.
//!
//! Run with: mpiexec -n 4 cargo run --example persistent_bcast
//!
//! Note: This example requires MPICH 4.0+ or OpenMPI 5.0+ with MPI 4.0 support.
// mpi-test: np=2.. skip-ok=openmpi

use ferrompi::{Mpi, ReduceOp, Result};

mod common;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    if rank == 0 {
        println!("Testing persistent collectives (MPI 4.0+)");
        println!("MPI Version: {}", Mpi::version()?);
        println!("Running on {} processes\n", size);
    }

    world.barrier()?;

    if common::mpi_major() >= 4 {
        // ============================================================
        // Test 1: Persistent Broadcast
        // ============================================================
        if rank == 0 {
            println!("Test 1: Persistent Broadcast");
            println!("----------------------------");
        }

        // Buffer that will be reused for all broadcasts
        let mut bcast_buffer = vec![0.0f64; 1000];

        let mut persistent_bcast = world.bcast_init(&mut bcast_buffer, 0)?;
        let num_iterations = 100;
        let start_time = Mpi::wtime();

        for iter in 0..num_iterations {
            // Root updates the buffer
            if rank == 0 {
                for (i, x) in bcast_buffer.iter_mut().enumerate() {
                    *x = (iter * 1000 + i) as f64;
                }
            }

            // Start the persistent operation
            persistent_bcast.start()?;

            // Optionally do other work here while communication proceeds...

            // Wait for completion
            persistent_bcast.wait()?;

            // Verify on non-root processes
            if rank != 0 {
                for (i, &x) in bcast_buffer.iter().enumerate() {
                    let expected = (iter * 1000 + i) as f64;
                    assert!(
                        (x - expected).abs() < 1e-10,
                        "Mismatch at iter {}, index {}: expected {}, got {}",
                        iter,
                        i,
                        expected,
                        x
                    );
                }
            }
        }

        let elapsed = Mpi::wtime() - start_time;
        let throughput = num_iterations as f64 / elapsed;

        world.barrier()?;

        if rank == 0 {
            println!(
                "  ✓ {} iterations completed in {:.4}s",
                num_iterations, elapsed
            );
            println!("  ✓ Throughput: {:.1} broadcasts/second", throughput);
        }

        world.barrier()?;

        // ============================================================
        // Test 2: Persistent All-Reduce
        // ============================================================
        if rank == 0 {
            println!("\nTest 2: Persistent All-Reduce");
            println!("-----------------------------");
        }

        let mut allreduce_send = vec![0.0f64; 500];
        let mut allreduce_recv = vec![0.0f64; 500];

        let mut persistent_allreduce =
            world.allreduce_init(&allreduce_send, &mut allreduce_recv, ReduceOp::Sum)?;
        let num_iterations = 100;
        let start_time = Mpi::wtime();

        for iter in 0..num_iterations {
            // Each rank contributes its rank value
            for x in allreduce_send.iter_mut() {
                *x = rank as f64 + iter as f64;
            }

            persistent_allreduce.start()?;
            persistent_allreduce.wait()?;

            // Verify: sum should be (0 + 1 + ... + (size-1)) + iter*size
            let expected: f64 = (0..size).map(|r| r as f64 + iter as f64).sum();
            for &x in &allreduce_recv {
                assert!(
                    (x - expected).abs() < 1e-10,
                    "Allreduce mismatch at iter {}",
                    iter
                );
            }
        }

        let elapsed = Mpi::wtime() - start_time;
        let throughput = num_iterations as f64 / elapsed;

        world.barrier()?;

        if rank == 0 {
            println!(
                "  ✓ {} iterations completed in {:.4}s",
                num_iterations, elapsed
            );
            println!("  ✓ Throughput: {:.1} all-reduces/second", throughput);
        }
    } else {
        common::skip(&world, "persistent collectives need MPI 4");
    }

    world.barrier()?;

    // ============================================================
    // Test 3: Comparison with Non-Persistent
    // ============================================================
    if rank == 0 {
        println!("\nTest 3: Performance Comparison");
        println!("------------------------------");
    }

    let compare_iterations = 1000;
    let buffer_size = 100;

    // Non-persistent broadcast timing
    {
        let mut buffer = vec![0.0f64; buffer_size];
        world.barrier()?;

        let start = Mpi::wtime();
        for iter in 0..compare_iterations {
            if rank == 0 {
                for (i, x) in buffer.iter_mut().enumerate() {
                    *x = (iter + i) as f64;
                }
            }
            world.broadcast(&mut buffer, 0)?;
        }
        let non_persistent_time = Mpi::wtime() - start;

        if rank == 0 {
            println!(
                "  Non-persistent: {} broadcasts in {:.4}s ({:.1}/s)",
                compare_iterations,
                non_persistent_time,
                compare_iterations as f64 / non_persistent_time
            );
        }
    }

    // Persistent broadcast timing
    {
        let mut buffer = vec![0.0f64; buffer_size];

        if let Ok(mut persistent) = world.bcast_init(&mut buffer, 0) {
            world.barrier()?;

            let start = Mpi::wtime();
            for iter in 0..compare_iterations {
                if rank == 0 {
                    for (i, x) in buffer.iter_mut().enumerate() {
                        *x = (iter + i) as f64;
                    }
                }
                persistent.start()?;
                persistent.wait()?;
            }
            let persistent_time = Mpi::wtime() - start;

            if rank == 0 {
                println!(
                    "  Persistent:     {} broadcasts in {:.4}s ({:.1}/s)",
                    compare_iterations,
                    persistent_time,
                    compare_iterations as f64 / persistent_time
                );
            }
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("Persistent collectives tests complete!");
        println!("========================================");
    }

    Ok(())
}
