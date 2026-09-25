//! Variable-count gather (gatherv) example.
//!
//! Demonstrates `MPI_Gatherv` where each process sends a different number of
//! elements to the root.  Rank `i` contributes `(i + 1)` elements, each set
//! to `i as f64`.  The root gathers all elements with correct displacements
//! and verifies the result.
//!
//! Run with: mpiexec -n 4 cargo run --example gatherv
// mpi-test: np=2..

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {}/{}: Starting gatherv example", rank, size);

    // ============================================================
    // Test 1: Variable-count gather to root
    // ============================================================
    //
    // Rank i sends (i+1) elements, all set to (i * 10.0).
    // Root receives them in order with proper displacements.
    {
        let send_count = (rank + 1) as usize;
        let send: Vec<f64> = vec![rank as f64 * 10.0; send_count];

        // Receive counts and displacements (only significant at root, but
        // we compute them on all ranks for simplicity).
        let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
        let displs: Vec<i32> = recvcounts
            .iter()
            .scan(0i32, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();
        let total: usize = recvcounts.iter().sum::<i32>() as usize;

        let mut recv = if rank == 0 {
            vec![0.0f64; total]
        } else {
            vec![] // Not used on non-root
        };

        world.gatherv(&send, &mut recv, &recvcounts, &displs, 0)?;

        if rank == 0 {
            // Verify: for each source rank r, the (r+1) elements starting at
            // displs[r] should all equal (r * 10.0).
            for r in 0..size as usize {
                let start = displs[r] as usize;
                let count = recvcounts[r] as usize;
                let expected_value = r as f64 * 10.0;
                for j in 0..count {
                    assert!(
                        (recv[start + j] - expected_value).abs() < f64::EPSILON,
                        "Gatherv mismatch from rank {} at offset {}: got {}, expected {}",
                        r,
                        start + j,
                        recv[start + j],
                        expected_value,
                    );
                }
            }
            println!(
                "  Gatherv test 1 passed ({} total elements gathered)",
                total
            );
        }
    }

    // ============================================================
    // Test 2: Gatherv with non-uniform payload sizes
    // ============================================================
    //
    // Each rank sends a "staircase" pattern: rank i sends elements
    // [i*100, i*100+1, ..., i*100+(i)], i.e., (i+1) elements.
    {
        let send_count = (rank + 1) as usize;
        let send: Vec<f64> = (0..send_count)
            .map(|j| rank as f64 * 100.0 + j as f64)
            .collect();

        let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
        let displs: Vec<i32> = recvcounts
            .iter()
            .scan(0i32, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();
        let total: usize = recvcounts.iter().sum::<i32>() as usize;

        let mut recv = if rank == 0 {
            vec![-1.0f64; total]
        } else {
            vec![]
        };

        world.gatherv(&send, &mut recv, &recvcounts, &displs, 0)?;

        if rank == 0 {
            for r in 0..size as usize {
                let start = displs[r] as usize;
                let count = recvcounts[r] as usize;
                for j in 0..count {
                    let expected = r as f64 * 100.0 + j as f64;
                    assert!(
                        (recv[start + j] - expected).abs() < f64::EPSILON,
                        "Gatherv staircase mismatch from rank {} at offset {}: got {}, expected {}",
                        r,
                        start + j,
                        recv[start + j],
                        expected,
                    );
                }
            }
            println!(
                "  Gatherv test 2 (staircase) passed ({} total elements)",
                total
            );
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("All gatherv tests passed!");
        println!("========================================");
    }

    Ok(())
}
