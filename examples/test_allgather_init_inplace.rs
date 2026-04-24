//! Integration test for allgather_init_inplace.
//!
//! Each rank r allocates the full buffer and pre-writes r*10 into its own slot
//! at offset r. After allgather_init_inplace, every rank asserts the buffer equals
//! [0, 10, 20, 30].
//!
//! The persistent request is reused across 3 iterations. On each iteration the
//! buffer is re-seeded and the operation is re-started.
//!
//! Run with: mpiexec -n 4 cargo run --example test_allgather_init_inplace

use ferrompi::{Mpi, Result};

const ITERATIONS: usize = 3;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(size, 4, "This test requires exactly 4 MPI processes");

    // Probe: check if persistent collectives are supported (MPI 4.0+).
    let mut probe_data = vec![0.0f64; 1];
    match world.bcast_init(&mut probe_data, 0) {
        Ok(req) => drop(req),
        Err(_) => {
            if rank == 0 {
                println!("SKIP: Persistent collectives not supported (requires MPI 4.0+)");
            }
            return Ok(());
        }
    }

    // recvcount = 1 element per rank; full buffer length = size.
    // Each rank's slot is at offset rank.
    let mut data = vec![0i32; size as usize];
    data[rank as usize] = rank * 10;

    let mut req = world.allgather_init_inplace(&mut data)?;

    for iter in 0..ITERATIONS {
        // Re-seed own slot before each start.
        data[rank as usize] = rank * 10;

        req.start()?;
        req.wait()?;

        let expected = vec![0i32, 10, 20, 30];
        assert_eq!(
            data, expected,
            "allgather_init_inplace iter {iter}: rank {} expected {:?} but got {:?}",
            rank, expected, data
        );
    }

    if rank == 0 {
        println!(
            "allgather_init_inplace: {:?} (x{ITERATIONS} iterations)",
            data
        );
    }

    world.barrier()?;

    Ok(())
}
