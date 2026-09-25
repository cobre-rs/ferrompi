//! Topology report example - gather and display MPI rank-to-host mapping.
//!
//! Run with: mpiexec -n 4 cargo run --example topology
// mpi-test: np=2..

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let topo = world.topology(&mpi)?;

    let size = world.size();
    assert_eq!(topo.size(), size, "topo.size() disagrees with world.size()");
    assert!(topo.num_hosts() >= 1, "topology reports zero hosts");

    let mut seen = vec![false; size as usize];
    for entry in topo.hosts() {
        for &r in &entry.ranks {
            assert!(
                !seen[r as usize],
                "rank {r} appears in more than one host entry"
            );
            seen[r as usize] = true;
        }
    }
    assert!(
        seen.iter().all(|&s| s),
        "topo.hosts() does not cover every rank in 0..{size} exactly once"
    );

    assert_eq!(
        topo.standard_version(),
        Mpi::version()?,
        "topo.standard_version() disagrees with Mpi::version()"
    );

    #[cfg(feature = "numa")]
    {
        assert_eq!(
            topo.slurm().is_some(),
            ferrompi::slurm::is_slurm_job(),
            "topo.slurm() presence disagrees with slurm::is_slurm_job()"
        );
        if !ferrompi::slurm::is_slurm_job() {
            assert!(ferrompi::slurm::job_id().is_none());
            assert!(ferrompi::slurm::local_rank().is_none());
            assert!(ferrompi::slurm::node_list().is_none());
        }
    }

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
