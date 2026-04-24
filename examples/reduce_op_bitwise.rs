//! Bitwise reduction operations example.
//!
//! Demonstrates `BitwiseOr`, `BitwiseAnd`, and `BitwiseXor` via `allreduce`.
//!
//! With 2 ranks:
//!   - Rank 0 contributes `0b0011` (3u32)
//!   - Rank 1 contributes `0b0101` (5u32)
//!
//! Expected results:
//!   - `BitwiseOr`:  `0b0111` (7u32) — union of set bits
//!   - `BitwiseAnd`: `0b0001` (1u32) — intersection of set bits
//!   - `BitwiseXor`: `0b0110` (6u32) — bits set in exactly one rank
//!
//! Run with: mpiexec -n 2 cargo run --example reduce_op_bitwise

use ferrompi::{Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    // ============================================================
    // Bitwise OR: union of all rank contributions
    // ============================================================
    {
        // Each rank contributes a single bit: rank 0 -> bit 0, rank 1 -> bit 1, ...
        let my_bit = 1u32 << rank;
        let result = world.allreduce_scalar(my_bit, ReduceOp::BitwiseOr)?;
        // All bits from rank 0 through (size-1) must be set.
        let expected: u32 = (0..size).fold(0u32, |acc, r| acc | (1u32 << r));
        assert_eq!(
            result, expected,
            "Rank {rank}: BitwiseOr failed: got {result:#010b}, expected {expected:#010b}"
        );
        if rank == 0 {
            println!("BitwiseOr (single bit per rank): {result:#010b} — PASS");
        }
    }

    // ============================================================
    // Fixed 2-rank scenario: rank 0 = 0b0011, rank 1 = 0b0101
    // Only verified when exactly 2 ranks are present.
    // ============================================================
    if size == 2 {
        let send: u32 = if rank == 0 { 0b0011 } else { 0b0101 };

        // BitwiseOr: 0b0011 | 0b0101 = 0b0111
        {
            let mut recv = [0u32; 1];
            world.allreduce(&[send], &mut recv, ReduceOp::BitwiseOr)?;
            assert_eq!(
                recv[0], 0b0111,
                "Rank {rank}: BitwiseOr failed: got {:#06b}, expected 0b0111",
                recv[0]
            );
            if rank == 0 {
                println!("BitwiseOr  (0b0011 | 0b0101) = {:#06b} — PASS", recv[0]);
            }
        }

        // BitwiseAnd: 0b0011 & 0b0101 = 0b0001
        {
            let mut recv = [0u32; 1];
            world.allreduce(&[send], &mut recv, ReduceOp::BitwiseAnd)?;
            assert_eq!(
                recv[0], 0b0001,
                "Rank {rank}: BitwiseAnd failed: got {:#06b}, expected 0b0001",
                recv[0]
            );
            if rank == 0 {
                println!("BitwiseAnd (0b0011 & 0b0101) = {:#06b} — PASS", recv[0]);
            }
        }

        // BitwiseXor: 0b0011 ^ 0b0101 = 0b0110
        {
            let mut recv = [0u32; 1];
            world.allreduce(&[send], &mut recv, ReduceOp::BitwiseXor)?;
            assert_eq!(
                recv[0], 0b0110,
                "Rank {rank}: BitwiseXor failed: got {:#06b}, expected 0b0110",
                recv[0]
            );
            if rank == 0 {
                println!("BitwiseXor (0b0011 ^ 0b0101) = {:#06b} — PASS", recv[0]);
            }
        }
    }

    world.barrier()?;

    if rank == 0 {
        println!("\nAll bitwise reduction tests passed!");
    }

    Ok(())
}
