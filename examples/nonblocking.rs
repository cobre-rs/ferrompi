//! Nonblocking operations example.
//!
//! Demonstrates overlap of communication and computation using
//! nonblocking collectives.
//!
//! Run with: mpiexec -n 4 cargo run --example nonblocking

use ferrompi::{Mpi, ReduceOp, Request, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {}: Starting nonblocking tests", rank);

    // ============================================================
    // Test 1: Nonblocking broadcast
    // ============================================================
    {
        let mut data: Vec<f64> = if rank == 0 {
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        } else {
            vec![0.0; 5]
        };

        // Start nonblocking broadcast
        let start_time = Mpi::wtime();
        let request = world.ibroadcast(&mut data, 0)?;

        // Simulate some computation while communication proceeds
        let mut compute_result = 0.0;
        for i in 0..1000 {
            compute_result += (i as f64).sin();
        }

        // Wait for broadcast to complete
        request.wait()?;
        let elapsed = Mpi::wtime() - start_time;

        // Verify the result
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            data, expected,
            "Nonblocking broadcast failed on rank {}",
            rank
        );

        if rank == 0 {
            println!(
                "✓ Nonblocking broadcast test passed (elapsed: {:.6}s)",
                elapsed
            );
            println!("  (computed {} during communication)", compute_result);
        }
    }

    world.barrier()?;

    // ============================================================
    // Test 2: Nonblocking all-reduce
    // ============================================================
    {
        let send = vec![rank as f64 + 1.0; 100];
        let mut recv = vec![0.0; 100];

        // Start nonblocking all-reduce
        let start_time = Mpi::wtime();
        let request = world.iallreduce(&send, &mut recv, ReduceOp::Sum)?;

        // Do some work
        let mut work_done = 0;
        for i in 0..10000 {
            work_done += i;
        }

        // Wait for completion
        request.wait()?;
        let elapsed = Mpi::wtime() - start_time;

        // Verify: sum should be 1 + 2 + ... + size
        let expected_sum: f64 = (1..=size).map(|x| x as f64).sum();
        for &val in &recv {
            assert!(
                (val - expected_sum).abs() < 1e-10,
                "Nonblocking allreduce failed on rank {}",
                rank
            );
        }

        if rank == 0 {
            println!(
                "✓ Nonblocking all-reduce test passed (elapsed: {:.6}s)",
                elapsed
            );
            println!("  (did {} units of work during communication)", work_done);
        }
    }

    world.barrier()?;

    // ============================================================
    // Test 3: Multiple nonblocking operations with wait_all
    // ============================================================
    {
        // Start multiple broadcasts
        let mut data1: Vec<f64> = if rank == 0 {
            vec![1.0; 10]
        } else {
            vec![0.0; 10]
        };
        let mut data2: Vec<f64> = if rank == 0 {
            vec![2.0; 10]
        } else {
            vec![0.0; 10]
        };
        let mut data3: Vec<f64> = if rank == 0 {
            vec![3.0; 10]
        } else {
            vec![0.0; 10]
        };

        let req1 = world.ibroadcast(&mut data1, 0)?;
        let req2 = world.ibroadcast(&mut data2, 0)?;
        let req3 = world.ibroadcast(&mut data3, 0)?;

        // Wait for all at once
        Request::wait_all(&mut [req1, req2, req3])?;

        // Verify
        assert!(data1.iter().all(|&x| (x - 1.0).abs() < 1e-10));
        assert!(data2.iter().all(|&x| (x - 2.0).abs() < 1e-10));
        assert!(data3.iter().all(|&x| (x - 3.0).abs() < 1e-10));

        if rank == 0 {
            println!("✓ Multiple nonblocking operations with wait_all passed");
        }
    }

    world.barrier()?;

    // ============================================================
    // Test 4: Using test() to poll for completion
    // ============================================================
    {
        let mut data: Vec<f64> = if rank == 0 {
            vec![42.0; 100]
        } else {
            vec![0.0; 100]
        };

        let mut request = world.ibroadcast(&mut data, 0)?;

        // Poll until complete
        let mut polls = 0;
        while !request.test()? {
            polls += 1;
            // Do a tiny bit of work between polls
            std::hint::spin_loop();
        }

        assert!(data.iter().all(|&x| (x - 42.0).abs() < 1e-10));

        if rank == 0 {
            println!("✓ Test/poll completion test passed ({} polls)", polls);
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("All nonblocking tests passed!");
        println!("========================================");
    }

    Ok(())
}
