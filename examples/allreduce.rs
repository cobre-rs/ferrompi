//! All-reduce example - collective reduction operations.
//!
//! Tests various collective operations including broadcast, reduce, and all-reduce.
//!
//! Run with: mpiexec -n 4 cargo run --example allreduce
// mpi-test: np=2..

use ferrompi::{Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {}/{}: Starting collective tests", rank, size);

    // ============================================================
    // Test 1: Broadcast
    // ============================================================
    {
        let mut data: Vec<f64> = if rank == 0 {
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        } else {
            vec![0.0; 5]
        };

        world.broadcast(&mut data, 0)?;

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(data, expected, "Broadcast failed on rank {}", rank);

        if rank == 0 {
            println!("✓ Broadcast test passed");
        }
    }

    // ============================================================
    // Test 2: Reduce (sum)
    // ============================================================
    {
        let send = vec![rank as f64 + 1.0; 3]; // Each rank sends [rank+1, rank+1, rank+1]
        let mut recv = vec![0.0; 3];

        world.reduce(&send, &mut recv, ReduceOp::Sum, 0)?;

        if rank == 0 {
            // Sum of 1 + 2 + ... + size
            let expected_sum: f64 = (1..=size).map(|x| x as f64).sum();
            let expected = vec![expected_sum; 3];
            assert_eq!(recv, expected, "Reduce Sum failed");
            println!("✓ Reduce Sum test passed (sum = {})", expected_sum);
        }
    }

    // ============================================================
    // Test 3: Reduce (max)
    // ============================================================
    {
        let send = vec![rank as f64 * 10.0];
        let mut recv = vec![0.0];

        world.reduce(&send, &mut recv, ReduceOp::Max, 0)?;

        if rank == 0 {
            let expected = (size - 1) as f64 * 10.0;
            assert_eq!(recv[0], expected, "Reduce Max failed");
            println!("✓ Reduce Max test passed (max = {})", expected);
        }
    }

    // ============================================================
    // Test 4: All-reduce (sum)
    // ============================================================
    {
        let send = vec![1.0; 4];
        let mut recv = vec![0.0; 4];

        world.allreduce(&send, &mut recv, ReduceOp::Sum)?;

        let expected = vec![size as f64; 4];
        assert_eq!(recv, expected, "Allreduce Sum failed on rank {}", rank);

        if rank == 0 {
            println!("✓ Allreduce Sum test passed");
        }
    }

    // ============================================================
    // Test 5: All-reduce scalar convenience method
    // ============================================================
    {
        let my_value = rank as f64 + 1.0;
        let sum = world.allreduce_scalar(my_value, ReduceOp::Sum)?;

        let expected: f64 = (1..=size).map(|x| x as f64).sum();
        assert!((sum - expected).abs() < 1e-10, "Allreduce scalar failed");

        if rank == 0 {
            println!("✓ Allreduce scalar test passed (sum = {})", sum);
        }
    }

    // ============================================================
    // Test 6: All-reduce in-place
    // ============================================================
    {
        let mut data = vec![rank as f64; 3];

        world.allreduce_inplace(&mut data, ReduceOp::Sum)?;

        // Sum of 0 + 1 + ... + (size-1)
        let expected: f64 = (0..size).map(|x| x as f64).sum();
        assert_eq!(
            data,
            vec![expected; 3],
            "Allreduce in-place failed on rank {}",
            rank
        );

        if rank == 0 {
            println!("✓ Allreduce in-place test passed");
        }
    }

    // ============================================================
    // Test 7: Gather
    // ============================================================
    {
        let send = vec![rank as f64 * 10.0, rank as f64 * 10.0 + 1.0];
        let mut recv = if rank == 0 {
            vec![0.0; 2 * size as usize]
        } else {
            vec![] // Not used on non-root
        };

        world.gather(&send, &mut recv, 0)?;

        if rank == 0 {
            // Check the gathered data
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(recv[idx], r as f64 * 10.0, "Gather failed at index {}", idx);
                assert_eq!(
                    recv[idx + 1],
                    r as f64 * 10.0 + 1.0,
                    "Gather failed at index {}",
                    idx + 1
                );
            }
            println!("✓ Gather test passed (received {} elements)", recv.len());
        }
    }

    // ============================================================
    // Test 8: All-gather
    // ============================================================
    {
        let send = vec![rank as f64];
        let mut recv = vec![0.0; size as usize];

        world.allgather(&send, &mut recv)?;

        for r in 0..size {
            assert_eq!(
                recv[r as usize], r as f64,
                "Allgather failed on rank {}",
                rank
            );
        }

        if rank == 0 {
            println!("✓ Allgather test passed");
        }
    }

    // ============================================================
    // Test 9: Scatter
    // ============================================================
    {
        let send = if rank == 0 {
            (0..size * 2).map(|x| x as f64).collect::<Vec<_>>()
        } else {
            vec![]
        };
        let mut recv = vec![0.0; 2];

        world.scatter(&send, &mut recv, 0)?;

        let expected_start = rank * 2;
        assert_eq!(
            recv[0], expected_start as f64,
            "Scatter failed on rank {}",
            rank
        );
        assert_eq!(
            recv[1],
            (expected_start + 1) as f64,
            "Scatter failed on rank {}",
            rank
        );

        if rank == 0 {
            println!("✓ Scatter test passed");
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("All collective tests passed!");
        println!("========================================");
    }

    Ok(())
}
