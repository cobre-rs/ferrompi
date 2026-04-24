//! Integration test for gather_init_inplace.
//!
//! Each rank r pre-writes r*10 into its in-place slot at offset r of the shared
//! buffer. Only rank 0 (root) calls gather_init_inplace; non-root ranks call the
//! regular gather_init to contribute their value.
//!
//! The persistent request is reused across 3 iterations. On each iteration the
//! buffer is re-seeded and the operation is re-started.
//!
//! After each wait, rank 0 asserts the buffer equals [0, 10, 20, 30].
//!
//! Run with: mpiexec -n 4 cargo run --example test_gather_init_inplace

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

    if rank == 0 {
        // Root: allocate full buffer; slot 0 is root's own contribution.
        let mut data = vec![0i32; size as usize];
        data[rank as usize] = rank * 10; // slot 0 = 0

        let mut req = world.gather_init_inplace(&mut data, 0)?;

        for iter in 0..ITERATIONS {
            // Re-seed root's own slot before each start.
            data[rank as usize] = rank * 10;

            req.start()?;
            req.wait()?;

            let expected = vec![0i32, 10, 20, 30];
            assert_eq!(
                data, expected,
                "gather_init_inplace iter {iter}: rank 0 expected {:?} but got {:?}",
                expected, data
            );
        }

        println!("gather_init_inplace: {:?} (x{ITERATIONS} iterations)", data);
    } else {
        // Non-root: single-element send buffer; recv is empty.
        let send = vec![rank * 10];
        let mut recv = vec![0i32; 0];

        let mut req = world.gather_init(&send, &mut recv, 0)?;

        for _ in 0..ITERATIONS {
            req.start()?;
            req.wait()?;
        }
    }

    world.barrier()?;

    Ok(())
}
