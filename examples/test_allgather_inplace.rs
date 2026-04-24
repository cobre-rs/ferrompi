//! Integration test for allgather_inplace.
//!
//! Each rank r allocates the full buffer and pre-writes r*10 into its own slot
//! at offset r. After allgather_inplace, every rank asserts the buffer equals
//! [0, 10, 20, 30].
//!
//! Run with: mpiexec -n 4 cargo run --example test_allgather_inplace

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(size, 4, "This test requires exactly 4 MPI processes");

    // recvcount = 1 element per rank; full buffer length = size.
    // Each rank's slot is at offset rank.
    let mut data = vec![0i32; size as usize];
    data[rank as usize] = rank * 10;

    world.allgather_inplace(&mut data)?;

    let expected = vec![0i32, 10, 20, 30];
    assert_eq!(
        data, expected,
        "allgather_inplace: rank {} expected {:?} but got {:?}",
        rank, expected, data
    );

    if rank == 0 {
        println!("allgather_inplace: {:?}", data);
    }

    world.barrier()?;

    Ok(())
}
