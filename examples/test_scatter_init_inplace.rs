//! Integration test for scatter_init_inplace.
//!
//! Root (rank 0) pre-populates a 4-element buffer [0, 10, 20, 30]. After
//! scatter_init_inplace(root=0), rank 0 retains its own slot (data[0] == 0) and each
//! non-root rank receives a single-element buffer matching its slot value.
//!
//! The persistent request is reused across 3 iterations. Root re-seeds the full
//! buffer before each start; non-root verifies the received value each iteration.
//!
//! Run with: mpiexec -n 4 cargo run --example test_scatter_init_inplace

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

    // MPICH 4.2.x has a deterministic deadlock in MPI_Scatter_init when called
    // with MPI_IN_PLACE on the root rank.  Reproduced on Ubuntu 24.04 (MPICH
    // 4.2.0/4.2.1 hotfix) and Fedora MPICH 4.2.3.  OpenMPI 5.x and MPICH 4.3+
    // are unaffected.  Skip this test on the affected MPICH versions until
    // upstream releases a fix; see docs/adr/0002-handle-tables.md follow-up.
    if let Ok(version) = Mpi::library_version() {
        // Match e.g. "MPICH Version:    4.2.3" — whitespace-tolerant match on
        // any MPICH 4.2.x.  The deadlock is fixed in MPICH 4.3+ and absent on
        // OpenMPI, so this is a narrow workaround, not a permanent skip.
        let mpich_42 = version.contains("MPICH")
            && version
                .lines()
                .any(|line| line.contains("Version:") && line.contains("4.2."));
        if mpich_42 {
            if rank == 0 {
                println!("SKIP: MPICH 4.2.x has a known MPI_Scatter_init+MPI_IN_PLACE deadlock");
            }
            return Ok(());
        }
    }

    if rank == 0 {
        // Root: full send buffer with one slot per rank.
        // Slot r contains the value to scatter to rank r.
        let mut data = vec![0i32, 10, 20, 30];

        let mut req = world.scatter_init_inplace(&mut data, 0)?;

        for iter in 0..ITERATIONS {
            // Re-seed the full buffer before each start.
            data.copy_from_slice(&[0i32, 10, 20, 30]);

            req.start()?;
            req.wait()?;

            // Root's own slot (index 0) must be retained unchanged.
            assert_eq!(
                data[0], 0,
                "scatter_init_inplace iter {iter}: rank 0 expected data[0]==0 but got {}",
                data[0]
            );
        }

        println!(
            "scatter_init_inplace: rank 0 retained slot data[0]={} (x{ITERATIONS} iterations)",
            data[0]
        );
    } else {
        // Non-root: single-element receive buffer; scatter fills it with the
        // value root placed in slot `rank`.
        let mut data = vec![0i32; 1];

        let mut req = world.scatter_init_inplace(&mut data, 0)?;

        for iter in 0..ITERATIONS {
            req.start()?;
            req.wait()?;

            let expected = rank * 10;
            assert_eq!(
                data[0], expected,
                "scatter_init_inplace iter {iter}: rank {} expected data[0]=={} but got {}",
                rank, expected, data[0]
            );
        }

        println!(
            "scatter_init_inplace: rank {} received {} (x{ITERATIONS} iterations)",
            rank, data[0]
        );
    }

    world.barrier()?;

    Ok(())
}
