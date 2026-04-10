//! Topology report example - gather and display MPI rank-to-host mapping.
//!
//! Run with: mpiexec -n 4 cargo run --example topology

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let topo = world.topology(&mpi)?;

    if world.rank() == 0 {
        println!("{topo}");
    }

    // Programmatic access is available on all ranks:
    for entry in topo.hosts() {
        if entry.ranks.contains(&world.rank()) {
            eprintln!(
                "Rank {} is on {} with {} co-located rank(s)",
                world.rank(),
                entry.hostname,
                entry.ranks.len() - 1,
            );
        }
    }

    Ok(())
}
