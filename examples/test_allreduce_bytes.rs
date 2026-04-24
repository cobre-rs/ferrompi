//! Integration test: `allreduce_bytes` with `MPI_BYTE` and `BitwiseOr`.
//!
//! Each of the 4 ranks contributes a `[u64; 4]` buffer where every element
//! equals `1u64 << rank`. After a `BitwiseOr` all-reduce, every element
//! of the receive buffer must equal `(1<<0) | (1<<1) | (1<<2) | (1<<3) = 0b1111`.
//!
//! Run with: mpiexec -n 4 cargo run --example test_allreduce_bytes

use ferrompi::{Mpi, ReduceOp, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();

    // Each rank contributes a buffer where every element is `1 << rank`.
    let data: [u64; 4] = [1u64 << rank; 4];
    let mut recv = [0u64; 4];

    world.allreduce_bytes(&data, &mut recv, ReduceOp::BitwiseOr)?;

    // After OR-reduction across 4 ranks: bits 0-3 are all set => 0b1111 = 15
    let expected = 0b1111u64;
    for (i, &val) in recv.iter().enumerate() {
        assert_eq!(
            val, expected,
            "Rank {rank}: recv[{i}] = {val:#018b}, expected {expected:#018b}"
        );
    }

    world.barrier()?;

    if rank == 0 {
        println!("allreduce_bytes BitwiseOr [u64; 4]: recv = {recv:?} — PASS");
    }

    Ok(())
}
