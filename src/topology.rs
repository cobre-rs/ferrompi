use std::fmt;

use crate::{Communicator, Error, Mpi, Result, ThreadLevel};

/// MPI topology information gathered across all ranks in a communicator.
///
/// This is produced by a collective operation ([`Communicator::topology`]) and
/// contains the rank-to-host mapping, MPI library metadata, and optional SLURM
/// job information.
///
/// # Display
///
/// The `Display` implementation produces a human-readable topology report:
///
/// ```text
/// ================ MPI Topology ================
/// Library:   Open MPI v4.1.6
/// Standard:  MPI 4.0
/// Threads:   Funneled
/// Processes: 8 across 2 nodes
///
///   compute-01: ranks 0, 1, 2, 3  (4 processes)
///   compute-02: ranks 4, 5, 6, 7  (4 processes)
/// ==============================================
/// ```
pub struct TopologyInfo {
    /// MPI library version (e.g., "Open MPI v4.1.6").
    library_version: String,
    /// MPI standard version (e.g., "MPI 4.0").
    standard_version: String,
    /// Thread support level granted by the MPI runtime.
    thread_level: ThreadLevel,
    /// Total number of processes in the communicator.
    size: i32,
    /// Hosts and their assigned ranks, ordered by first rank on each host.
    hosts: Vec<HostEntry>,
    /// SLURM job metadata, populated when running under SLURM with the `numa` feature.
    #[cfg(feature = "numa")]
    slurm: Option<SlurmInfo>,
}

/// A single host and its assigned MPI ranks.
#[derive(Debug, Clone)]
pub struct HostEntry {
    /// Hostname as reported by `MPI_Get_processor_name`.
    pub hostname: String,
    /// Sorted list of global ranks on this host.
    pub ranks: Vec<i32>,
}

/// SLURM job metadata.
#[cfg(feature = "numa")]
#[derive(Debug, Clone)]
pub struct SlurmInfo {
    /// SLURM job ID.
    pub job_id: String,
    /// Compact node list (e.g., "node[001-004]").
    pub node_list: Option<String>,
    /// CPUs allocated per task.
    pub cpus_per_task: Option<i32>,
}

impl TopologyInfo {
    /// Hosts and their assigned ranks, ordered by first rank on each host.
    pub fn hosts(&self) -> &[HostEntry] {
        &self.hosts
    }

    /// MPI library version string (implementation-specific).
    pub fn library_version(&self) -> &str {
        &self.library_version
    }

    /// MPI standard version string.
    pub fn standard_version(&self) -> &str {
        &self.standard_version
    }

    /// Thread support level granted by the MPI runtime.
    pub fn thread_level(&self) -> ThreadLevel {
        self.thread_level
    }

    /// Total number of processes in the communicator.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Number of distinct hosts.
    pub fn num_hosts(&self) -> usize {
        self.hosts.len()
    }

    /// SLURM job metadata, if running under SLURM with the `numa` feature.
    #[cfg(feature = "numa")]
    pub fn slurm(&self) -> Option<&SlurmInfo> {
        self.slurm.as_ref()
    }
}

/// Maximum hostname length used for the fixed-size allgather buffer.
/// Matches `MPI_MAX_PROCESSOR_NAME` (256 in all major implementations).
const HOSTNAME_BUF_LEN: usize = 256;

/// Gather topology information from all ranks in the communicator.
///
/// This is a **collective operation** — all ranks in the communicator must call
/// it. Every rank receives the complete topology.
pub(crate) fn gather_topology(comm: &Communicator, mpi: &Mpi) -> Result<TopologyInfo> {
    let size = comm.size();
    let rank = comm.rank();

    // Each rank fills a fixed-size hostname buffer.
    let name = comm.processor_name()?;
    let mut local_buf = [0u8; HOSTNAME_BUF_LEN];
    let name_bytes = name.as_bytes();
    let copy_len = name_bytes.len().min(HOSTNAME_BUF_LEN);
    local_buf[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

    // Allgather the hostname buffers.
    let mut all_bufs = vec![0u8; HOSTNAME_BUF_LEN * size as usize];
    comm.allgather(&local_buf, &mut all_bufs)?;

    // Build rank-to-host mapping, preserving insertion order (first rank seen per host).
    let mut host_map: Vec<(String, Vec<i32>)> = Vec::new();
    for r in 0..size {
        let start = r as usize * HOSTNAME_BUF_LEN;
        let end = start + HOSTNAME_BUF_LEN;
        let raw = &all_bufs[start..end];
        // Find the first null byte or take the whole buffer.
        let nul_pos = raw.iter().position(|&b| b == 0).unwrap_or(HOSTNAME_BUF_LEN);
        let hostname = std::str::from_utf8(&raw[..nul_pos])
            .map_err(|_| Error::Internal("Invalid UTF-8 in gathered hostname".into()))?
            .to_string();

        if let Some(entry) = host_map.iter_mut().find(|(h, _)| *h == hostname) {
            entry.1.push(r);
        } else {
            host_map.push((hostname, vec![r]));
        }
    }

    let hosts: Vec<HostEntry> = host_map
        .into_iter()
        .map(|(hostname, ranks)| HostEntry { hostname, ranks })
        .collect();

    // Gather metadata — only rank 0 strictly needs these, but they're cheap
    // and having them on every rank avoids conditional logic for the caller.
    let library_version = if rank == 0 {
        Mpi::library_version()?
    } else {
        String::new()
    };
    let standard_version = if rank == 0 {
        Mpi::version()?
    } else {
        String::new()
    };

    // Broadcast the version strings from rank 0 so all ranks have them.
    // We encode as a fixed-size buffer to keep things simple.
    let library_version = broadcast_string(comm, &library_version, 0)?;
    let standard_version = broadcast_string(comm, &standard_version, 0)?;

    let thread_level = mpi.thread_level();

    #[cfg(feature = "numa")]
    let slurm = if crate::slurm::is_slurm_job() {
        Some(SlurmInfo {
            job_id: crate::slurm::job_id().unwrap_or_default(),
            node_list: crate::slurm::node_list(),
            cpus_per_task: crate::slurm::cpus_per_task(),
        })
    } else {
        None
    };

    Ok(TopologyInfo {
        library_version,
        standard_version,
        thread_level,
        size,
        hosts,
        #[cfg(feature = "numa")]
        slurm,
    })
}

/// Broadcast a string from `root` to all ranks using a fixed-size buffer.
fn broadcast_string(comm: &Communicator, s: &str, root: i32) -> Result<String> {
    // Use a generous buffer — library version strings can be long.
    const BUF_LEN: usize = 512;
    let mut buf = [0u8; BUF_LEN];
    if comm.rank() == root {
        let bytes = s.as_bytes();
        let copy_len = bytes.len().min(BUF_LEN);
        buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
    }
    comm.broadcast(&mut buf, root)?;
    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(BUF_LEN);
    let result = std::str::from_utf8(&buf[..nul_pos])
        .map_err(|_| Error::Internal("Invalid UTF-8 in broadcast string".into()))?;
    Ok(result.to_string())
}

impl fmt::Display for TopologyInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "================ MPI Topology ================")?;
        writeln!(f, "Library:   {}", self.library_version)?;
        writeln!(f, "Standard:  {}", self.standard_version)?;
        writeln!(f, "Threads:   {:?}", self.thread_level)?;
        let node_word = if self.hosts.len() == 1 {
            "node"
        } else {
            "nodes"
        };
        writeln!(
            f,
            "Processes: {} across {} {}",
            self.size,
            self.hosts.len(),
            node_word,
        )?;

        #[cfg(feature = "numa")]
        if let Some(ref slurm) = self.slurm {
            writeln!(f, "SLURM Job: {}", slurm.job_id)?;
            if let Some(ref nl) = slurm.node_list {
                writeln!(f, "Nodes:     {}", nl)?;
            }
            if let Some(cpt) = slurm.cpus_per_task {
                writeln!(f, "CPUs/Task: {}", cpt)?;
            }
        }

        writeln!(f)?;
        for entry in &self.hosts {
            let ranks_str: Vec<String> = entry.ranks.iter().map(|r| r.to_string()).collect();
            let proc_word = if entry.ranks.len() == 1 {
                "process"
            } else {
                "processes"
            };
            writeln!(
                f,
                "  {}: ranks {}  ({} {})",
                entry.hostname,
                ranks_str.join(", "),
                entry.ranks.len(),
                proc_word,
            )?;
        }
        write!(f, "==============================================")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_topology() -> TopologyInfo {
        TopologyInfo {
            library_version: "Open MPI v4.1.6".to_string(),
            standard_version: "MPI 4.0".to_string(),
            thread_level: ThreadLevel::Funneled,
            size: 8,
            hosts: vec![
                HostEntry {
                    hostname: "compute-01".to_string(),
                    ranks: vec![0, 1, 2, 3],
                },
                HostEntry {
                    hostname: "compute-02".to_string(),
                    ranks: vec![4, 5, 6, 7],
                },
            ],
            #[cfg(feature = "numa")]
            slurm: None,
        }
    }

    #[test]
    fn display_contains_library_version() {
        let topo = sample_topology();
        let output = format!("{topo}");
        assert!(output.contains("Open MPI v4.1.6"));
    }

    #[test]
    fn display_contains_standard_version() {
        let topo = sample_topology();
        let output = format!("{topo}");
        assert!(output.contains("MPI 4.0"));
    }

    #[test]
    fn display_contains_thread_level() {
        let topo = sample_topology();
        let output = format!("{topo}");
        assert!(output.contains("Funneled"));
    }

    #[test]
    fn display_contains_process_count() {
        let topo = sample_topology();
        let output = format!("{topo}");
        assert!(output.contains("8 across 2 nodes"));
    }

    #[test]
    fn display_contains_host_entries() {
        let topo = sample_topology();
        let output = format!("{topo}");
        assert!(output.contains("compute-01: ranks 0, 1, 2, 3  (4 processes)"));
        assert!(output.contains("compute-02: ranks 4, 5, 6, 7  (4 processes)"));
    }

    #[test]
    fn display_single_node() {
        let topo = TopologyInfo {
            library_version: "MPICH v4.1".to_string(),
            standard_version: "MPI 4.0".to_string(),
            thread_level: ThreadLevel::Single,
            size: 4,
            hosts: vec![HostEntry {
                hostname: "localhost".to_string(),
                ranks: vec![0, 1, 2, 3],
            }],
            #[cfg(feature = "numa")]
            slurm: None,
        };
        let output = format!("{topo}");
        assert!(output.contains("4 across 1 node"));
        assert!(!output.contains("nodes"));
    }

    #[test]
    fn display_single_process() {
        let topo = TopologyInfo {
            library_version: "MPICH v4.1".to_string(),
            standard_version: "MPI 4.0".to_string(),
            thread_level: ThreadLevel::Single,
            size: 1,
            hosts: vec![HostEntry {
                hostname: "localhost".to_string(),
                ranks: vec![0],
            }],
            #[cfg(feature = "numa")]
            slurm: None,
        };
        let output = format!("{topo}");
        assert!(output.contains("1 process)"));
        assert!(!output.contains("processes"));
    }

    #[test]
    fn accessors_return_expected_values() {
        let topo = sample_topology();
        assert_eq!(topo.library_version(), "Open MPI v4.1.6");
        assert_eq!(topo.standard_version(), "MPI 4.0");
        assert_eq!(topo.thread_level(), ThreadLevel::Funneled);
        assert_eq!(topo.size(), 8);
        assert_eq!(topo.num_hosts(), 2);
        assert_eq!(topo.hosts().len(), 2);
        assert_eq!(topo.hosts()[0].hostname, "compute-01");
        assert_eq!(topo.hosts()[0].ranks, vec![0, 1, 2, 3]);
    }

    #[cfg(feature = "numa")]
    #[test]
    fn display_with_slurm_info() {
        let topo = TopologyInfo {
            library_version: "Open MPI v4.1.6".to_string(),
            standard_version: "MPI 4.0".to_string(),
            thread_level: ThreadLevel::Multiple,
            size: 8,
            hosts: vec![
                HostEntry {
                    hostname: "compute-01".to_string(),
                    ranks: vec![0, 1, 2, 3],
                },
                HostEntry {
                    hostname: "compute-02".to_string(),
                    ranks: vec![4, 5, 6, 7],
                },
            ],
            slurm: Some(SlurmInfo {
                job_id: "123456".to_string(),
                node_list: Some("compute-[01-02]".to_string()),
                cpus_per_task: Some(4),
            }),
        };
        let output = format!("{topo}");
        assert!(output.contains("SLURM Job: 123456"));
        assert!(output.contains("Nodes:     compute-[01-02]"));
        assert!(output.contains("CPUs/Task: 4"));
    }

    #[cfg(feature = "numa")]
    #[test]
    fn slurm_accessor_none_when_absent() {
        let topo = sample_topology();
        assert!(topo.slurm().is_none());
    }
}
