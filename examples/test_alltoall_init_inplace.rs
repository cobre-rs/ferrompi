//! Integration test for alltoall_init_inplace.
//!
//! Each rank r pre-writes data[s] = r*10 + s into slot s (the payload destined for
//! rank s). After alltoall_init_inplace, every rank asserts data[s] == s*10 + r (the
//! data received FROM rank s, which is rank s's contribution destined for rank r).
//!
//! The persistent request is reused across 3 iterations. The buffer is re-seeded
//! before each start.
//!
//! Run with: mpiexec -n 4 cargo run --example test_alltoall_init_inplace

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

    // Each rank r pre-writes slot s with the value it wants to send to rank s.
    // data[s] = r*10 + s  means "from rank r, to rank s: value r*10+s"
    let mut data: Vec<i32> = (0..size).map(|s| rank * 10 + s).collect();

    let mut req = world.alltoall_init_inplace(&mut data)?;

    for iter in 0..ITERATIONS {
        // Re-seed before each start.
        for s in 0..size {
            data[s as usize] = rank * 10 + s;
        }

        req.start()?;
        req.wait()?;

        // After the call: data[s] should contain what rank s sent to rank r.
        // Rank s sent to rank r: s*10 + r.
        for s in 0..size {
            let expected = s * 10 + rank;
            assert_eq!(
                data[s as usize], expected,
                "alltoall_init_inplace iter {iter}: rank {} slot {} expected {} but got {}",
                rank, s, expected, data[s as usize]
            );
        }
    }

    if rank == 0 {
        println!(
            "alltoall_init_inplace: rank 0 result {:?} (x{ITERATIONS} iterations)",
            data
        );
    }

    world.barrier()?;

    Ok(())
}
