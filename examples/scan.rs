//! Prefix sum (scan) example — inclusive and exclusive scan operations.
//!
//! Demonstrates `MPI_Scan` (inclusive) and `MPI_Exscan` (exclusive) by computing
//! prefix sums of rank values across all processes.  Also shows the scalar
//! convenience methods `scan_scalar` and `exscan_scalar`.
//!
//! Run with: mpiexec -n 4 cargo run --example scan
// mpi-test: np=2..

use ferrompi::{Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {}/{}: Starting scan examples", rank, size);

    // ============================================================
    // Test 1: Inclusive scan (vector)
    // ============================================================
    //
    // Each rank contributes a vector of its rank value.
    // After inclusive scan, rank i receives the element-wise sum of
    // contributions from ranks 0..=i.
    {
        let send = vec![rank as f64 + 1.0; 3]; // [rank+1, rank+1, rank+1]
        let mut recv = vec![0.0f64; 3];

        world.scan(&send, &mut recv, ReduceOp::Sum)?;

        // Expected: sum of (r+1) for r in 0..=rank  =  (rank+1)*(rank+2)/2
        let expected = (rank + 1) as f64 * (rank + 2) as f64 / 2.0;
        for (j, &val) in recv.iter().enumerate() {
            assert!(
                (val - expected).abs() < f64::EPSILON,
                "Rank {}: inclusive scan mismatch at index {}: got {}, expected {}",
                rank,
                j,
                val,
                expected,
            );
        }

        if rank == 0 {
            println!("  Inclusive scan (vector) passed");
        }
    }

    // ============================================================
    // Test 2: Inclusive scan (scalar convenience)
    // ============================================================
    {
        let prefix_sum = world.scan_scalar(1.0f64, ReduceOp::Sum)?;

        // Each rank contributes 1.0, so rank i gets (i + 1).0
        let expected = (rank + 1) as f64;
        assert!(
            (prefix_sum - expected).abs() < f64::EPSILON,
            "Rank {}: scan_scalar mismatch: got {}, expected {}",
            rank,
            prefix_sum,
            expected,
        );

        if rank == 0 {
            println!("  Inclusive scan (scalar) passed");
        }
    }

    // ============================================================
    // Test 3: Exclusive scan (vector)
    // ============================================================
    //
    // After exclusive scan, rank i receives the element-wise sum of
    // contributions from ranks 0..i (excluding rank i itself).
    // NOTE: On rank 0, the receive buffer is UNDEFINED per the MPI standard.
    {
        let send = vec![rank as f64 + 1.0; 3];
        let mut recv = vec![0.0f64; 3];

        world.exscan(&send, &mut recv, ReduceOp::Sum)?;

        if rank > 0 {
            // Expected: sum of (r+1) for r in 0..rank  =  rank*(rank+1)/2
            let expected = rank as f64 * (rank + 1) as f64 / 2.0;
            for (j, &val) in recv.iter().enumerate() {
                assert!(
                    (val - expected).abs() < f64::EPSILON,
                    "Rank {}: exclusive scan mismatch at index {}: got {}, expected {}",
                    rank,
                    j,
                    val,
                    expected,
                );
            }
        }
        // rank 0: recv is undefined per MPI standard — intentionally not checked

        if rank == 0 {
            println!("  Exclusive scan (vector) passed");
        }
    }

    // ============================================================
    // Test 4: Exclusive scan (scalar convenience)
    // ============================================================
    {
        let prefix_sum = world.exscan_scalar(1.0f64, ReduceOp::Sum)?;

        if rank > 0 {
            // Each rank contributes 1.0, so rank i gets i.0
            let expected = rank as f64;
            assert!(
                (prefix_sum - expected).abs() < f64::EPSILON,
                "Rank {}: exscan_scalar mismatch: got {}, expected {}",
                rank,
                prefix_sum,
                expected,
            );
        }

        if rank == 0 {
            println!("  Exclusive scan (scalar) passed");
        }
    }

    // ============================================================
    // Test 5: Inclusive scan with Max (running maximum)
    // ============================================================
    //
    // Demonstrates scan with an operation other than Sum.
    {
        // Rank i contributes value (size - rank), so the sequence is
        // [size, size-1, ..., 1].  Running max is always `size`.
        let value = (size - rank) as f64;
        let running_max = world.scan_scalar(value, ReduceOp::Max)?;

        let expected = size as f64; // max is always the first rank's value
        assert!(
            (running_max - expected).abs() < f64::EPSILON,
            "Rank {}: scan Max mismatch: got {}, expected {}",
            rank,
            running_max,
            expected,
        );

        if rank == 0 {
            println!("  Inclusive scan (Max) passed");
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("All scan tests passed!");
        println!("========================================");
    }

    Ok(())
}
