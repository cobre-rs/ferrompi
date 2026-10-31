//! MAXLOC and MINLOC reduction example.
//!
//! Demonstrates `allreduce_indexed` with `CollectiveOp::MAX_LOC` and
//! `CollectiveOp::MIN_LOC` using the `DoubleInt` paired value+index type.
//!
//! Each rank contributes `DoubleInt { value: rank as f64, index: rank }`.
//! After `MAX_LOC`: every rank holds `{ value: (size-1) as f64, index: size-1 }`.
//! After `MIN_LOC`: every rank holds `{ value: 0.0, index: 0 }`.
//!
//! Run with: mpiexec -n 4 cargo run --example reduce_op_maxloc
// mpi-test: np=4

use ferrompi::{CollectiveOp, DoubleInt, Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {rank}/{size}: starting MAXLOC/MINLOC tests");

    // ============================================================
    // Test 1: MAX_LOC — find the maximum value and its origin rank
    // ============================================================
    {
        let send = [DoubleInt {
            value: rank as f64,
            index: rank,
        }];
        let mut recv = [DoubleInt {
            value: 0.0,
            index: 0,
        }];

        world.allreduce_indexed(&send, &mut recv, CollectiveOp::MAX_LOC)?;

        let expected_value = (size - 1) as f64;
        let expected_index = size - 1;
        assert_eq!(
            recv[0].value, expected_value,
            "Rank {rank}: MAX_LOC value mismatch: got {}, expected {expected_value}",
            recv[0].value
        );
        assert_eq!(
            recv[0].index, expected_index,
            "Rank {rank}: MAX_LOC index mismatch: got {}, expected {expected_index}",
            recv[0].index
        );

        if rank == 0 {
            println!(
                "MAX_LOC PASS: value={}, index={}",
                recv[0].value, recv[0].index
            );
        }
    }

    // ============================================================
    // Test 2: MIN_LOC — find the minimum value and its origin rank
    // ============================================================
    {
        let send = [DoubleInt {
            value: rank as f64,
            index: rank,
        }];
        let mut recv = [DoubleInt {
            value: 0.0,
            index: 0,
        }];

        world.allreduce_indexed(&send, &mut recv, CollectiveOp::MIN_LOC)?;

        let expected_value = 0.0_f64;
        let expected_index = 0_i32;
        assert_eq!(
            recv[0].value, expected_value,
            "Rank {rank}: MIN_LOC value mismatch: got {}, expected {expected_value}",
            recv[0].value
        );
        assert_eq!(
            recv[0].index, expected_index,
            "Rank {rank}: MIN_LOC index mismatch: got {}, expected {expected_index}",
            recv[0].index
        );

        if rank == 0 {
            println!(
                "MIN_LOC PASS: value={}, index={}",
                recv[0].value, recv[0].index
            );
        }
    }

    // ============================================================
    // Test 3: MAX_LOC with multiple elements
    // ============================================================
    {
        // Each rank contributes two elements with different values:
        //   element 0: value = rank as f64, index = rank
        //   element 1: value = (size - 1 - rank) as f64, index = rank
        // After MAX_LOC on element 0: value = (size-1), index = size-1
        // After MAX_LOC on element 1: value = (size-1), index = 0
        let send = [
            DoubleInt {
                value: rank as f64,
                index: rank,
            },
            DoubleInt {
                value: (size - 1 - rank) as f64,
                index: rank,
            },
        ];
        let mut recv = [
            DoubleInt {
                value: 0.0,
                index: 0,
            },
            DoubleInt {
                value: 0.0,
                index: 0,
            },
        ];

        world.allreduce_indexed(&send, &mut recv, CollectiveOp::MAX_LOC)?;

        assert_eq!(
            recv[0].value,
            (size - 1) as f64,
            "Rank {rank}: MAX_LOC[0] value mismatch"
        );
        assert_eq!(
            recv[0].index,
            size - 1,
            "Rank {rank}: MAX_LOC[0] index mismatch"
        );
        assert_eq!(
            recv[1].value,
            (size - 1) as f64,
            "Rank {rank}: MAX_LOC[1] value mismatch"
        );
        assert_eq!(recv[1].index, 0, "Rank {rank}: MAX_LOC[1] index mismatch");

        if rank == 0 {
            println!(
                "MAX_LOC (multi-element) PASS: [{}, {}] [{}, {}]",
                recv[0].value, recv[0].index, recv[1].value, recv[1].index
            );
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\n========================================");
        println!("All MAXLOC/MINLOC tests passed!");
        println!("========================================");
    }

    Ok(())
}
