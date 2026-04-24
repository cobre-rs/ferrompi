//! MAXLOC and MINLOC reduction example.
//!
//! Demonstrates `allreduce_indexed` with `ReduceOp::MaxLoc` and
//! `ReduceOp::MinLoc` using the `DoubleInt` paired value+index type.
//!
//! Each rank contributes `DoubleInt { value: rank as f64, index: rank }`.
//! After `MaxLoc`: every rank holds `{ value: (size-1) as f64, index: size-1 }`.
//! After `MinLoc`: every rank holds `{ value: 0.0, index: 0 }`.
//!
//! Run with: mpiexec -n 4 cargo run --example reduce_op_maxloc

use ferrompi::{DoubleInt, Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Rank {rank}/{size}: starting MAXLOC/MINLOC tests");

    // ============================================================
    // Test 1: MaxLoc — find the maximum value and its origin rank
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

        world.allreduce_indexed(&send, &mut recv, ReduceOp::MaxLoc)?;

        let expected_value = (size - 1) as f64;
        let expected_index = size - 1;
        assert_eq!(
            recv[0].value, expected_value,
            "Rank {rank}: MaxLoc value mismatch: got {}, expected {expected_value}",
            recv[0].value
        );
        assert_eq!(
            recv[0].index, expected_index,
            "Rank {rank}: MaxLoc index mismatch: got {}, expected {expected_index}",
            recv[0].index
        );

        if rank == 0 {
            println!(
                "MaxLoc PASS: value={}, index={}",
                recv[0].value, recv[0].index
            );
        }
    }

    // ============================================================
    // Test 2: MinLoc — find the minimum value and its origin rank
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

        world.allreduce_indexed(&send, &mut recv, ReduceOp::MinLoc)?;

        let expected_value = 0.0_f64;
        let expected_index = 0_i32;
        assert_eq!(
            recv[0].value, expected_value,
            "Rank {rank}: MinLoc value mismatch: got {}, expected {expected_value}",
            recv[0].value
        );
        assert_eq!(
            recv[0].index, expected_index,
            "Rank {rank}: MinLoc index mismatch: got {}, expected {expected_index}",
            recv[0].index
        );

        if rank == 0 {
            println!(
                "MinLoc PASS: value={}, index={}",
                recv[0].value, recv[0].index
            );
        }
    }

    // ============================================================
    // Test 3: MaxLoc with multiple elements
    // ============================================================
    {
        // Each rank contributes two elements with different values:
        //   element 0: value = rank as f64, index = rank
        //   element 1: value = (size - 1 - rank) as f64, index = rank
        // After MaxLoc on element 0: value = (size-1), index = size-1
        // After MaxLoc on element 1: value = (size-1), index = 0
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

        world.allreduce_indexed(&send, &mut recv, ReduceOp::MaxLoc)?;

        assert_eq!(
            recv[0].value,
            (size - 1) as f64,
            "Rank {rank}: MaxLoc[0] value mismatch"
        );
        assert_eq!(
            recv[0].index,
            size - 1,
            "Rank {rank}: MaxLoc[0] index mismatch"
        );
        assert_eq!(
            recv[1].value,
            (size - 1) as f64,
            "Rank {rank}: MaxLoc[1] value mismatch"
        );
        assert_eq!(recv[1].index, 0, "Rank {rank}: MaxLoc[1] index mismatch");

        if rank == 0 {
            println!(
                "MaxLoc (multi-element) PASS: [{}, {}] [{}, {}]",
                recv[0].value, recv[0].index, recv[1].value, recv[1].index
            );
        }
    }

    // ============================================================
    // Test 4: Invalid op returns Err — guard is enforced
    // ============================================================
    {
        use ferrompi::Error;
        let send = [DoubleInt {
            value: 1.0,
            index: rank,
        }];
        let mut recv = [DoubleInt {
            value: 0.0,
            index: 0,
        }];
        let result = world.allreduce_indexed(&send, &mut recv, ReduceOp::Sum);
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "Rank {rank}: expected Err(InvalidOp) for Sum op on indexed type"
        );
        if rank == 0 {
            println!("InvalidOp guard for non-loc op PASS");
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
