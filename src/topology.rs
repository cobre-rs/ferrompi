use std::collections::HashMap;
use std::fmt;

use crate::{Communicator, Error, Mpi, Result, ThreadLevel};

/// MPI topology information gathered across all ranks in a communicator.
///
/// This is produced by a collective operation ([`Communicator::topology`]).
/// The rank-to-host mapping is gathered across all ranks; the MPI library
/// metadata is each rank's own local value (see
/// [`library_version`](Self::library_version) and
/// [`standard_version`](Self::standard_version)).
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
    ///
    /// The calling process's own value, as returned by
    /// [`Mpi::library_version`]; it is not gathered, so ranks linked against
    /// differently built libraries may report different strings.
    pub fn library_version(&self) -> &str {
        &self.library_version
    }

    /// MPI standard version string.
    ///
    /// The calling process's own value, as returned by [`Mpi::version`]; it
    /// is not gathered, so ranks linked against differently built libraries
    /// may report different strings.
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
/// It equals the size of `Communicator::processor_name`'s buffer, which holds any
/// `MPI_MAX_PROCESSOR_NAME` the C layer accepts at build time.
const HOSTNAME_BUF_LEN: usize = 256;

/// Builds one rank's hostname slot for the topology allgather.
///
/// `None` marks a rank whose local queries failed: the slot's first byte is
/// `0xFF`, a byte valid UTF-8 never contains, so every rank's host-table
/// build rejects it.
fn hostname_slot(name: Option<&str>) -> [u8; HOSTNAME_BUF_LEN] {
    let mut buf = [0u8; HOSTNAME_BUF_LEN];
    match name {
        Some(name) => {
            let name_bytes = name.as_bytes();
            let copy_len = name_bytes.len().min(HOSTNAME_BUF_LEN);
            buf[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        }
        None => buf[0] = 0xFF,
    }
    buf
}

/// Builds the rank-to-host table from the gathered hostname slots.
///
/// Preserves insertion order (first rank seen per host). Keying a HashMap on
/// borrowed hostname slices makes this pass O(size) rather than O(size ×
/// distinct_hosts), and allocates one `String` per distinct host instead of
/// one per rank.
fn hosts_from_slots(all_bufs: &[u8], size: i32) -> Result<Vec<HostEntry>> {
    let mut hosts: Vec<HostEntry> = Vec::new();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for r in 0..size {
        let start = r as usize * HOSTNAME_BUF_LEN;
        let raw = &all_bufs[start..start + HOSTNAME_BUF_LEN];
        // Find the first null byte or take the whole buffer.
        let nul_pos = raw.iter().position(|&b| b == 0).unwrap_or(HOSTNAME_BUF_LEN);
        let hostname = std::str::from_utf8(&raw[..nul_pos]).map_err(|_| {
            Error::Internal(format!(
                "rank {r} could not query its processor name or MPI version"
            ))
        })?;

        if let Some(&i) = index.get(hostname) {
            hosts[i].ranks.push(r);
        } else {
            index.insert(hostname, hosts.len());
            hosts.push(HostEntry {
                hostname: hostname.to_string(),
                ranks: vec![r],
            });
        }
    }
    Ok(hosts)
}

/// Gather topology information from all ranks in the communicator.
///
/// This is a **collective operation** — all ranks in the communicator must call
/// it. Every rank receives the complete topology.
pub(crate) fn gather_topology(comm: &Communicator, mpi: &Mpi) -> Result<TopologyInfo> {
    let size = comm.size();

    // Both version queries are local procedures (MPI-4.1 §9.1.1): each rank reports its own library.
    let local = comm
        .processor_name()
        .and_then(|name| Ok((name, Mpi::library_version()?, Mpi::version()?)));
    let local_buf = hostname_slot(local.as_ref().ok().map(|(name, _, _)| name.as_str()));

    // Allgather the hostname buffers.
    let mut all_bufs = vec![0u8; HOSTNAME_BUF_LEN * size as usize];
    comm.allgather(&local_buf, &mut all_bufs)?;

    // A failing rank reports its own error here, after every rank has
    // already reached the allgather.
    let (_, library_version, standard_version) = local?;
    let hosts = hosts_from_slots(&all_bufs, size)?;

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
    #[cfg(feature = "numa")]
    use super::SlurmInfo;
    use super::{hostname_slot, hosts_from_slots, Error, HostEntry, ThreadLevel, TopologyInfo};

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

    #[test]
    fn slots_group_ranks_by_host() {
        let slots = [
            hostname_slot(Some("node-a")),
            hostname_slot(Some("node-b")),
            hostname_slot(Some("node-a")),
        ]
        .concat();
        let hosts = hosts_from_slots(&slots, 3).unwrap();
        assert_eq!(hosts.len(), 2);
        assert_eq!(hosts[0].hostname, "node-a");
        assert_eq!(hosts[0].ranks, vec![0, 2]);
        assert_eq!(hosts[1].hostname, "node-b");
        assert_eq!(hosts[1].ranks, vec![1]);
    }

    #[test]
    fn a_failed_rank_fails_the_host_table() {
        let slots = [
            hostname_slot(Some("node-a")),
            hostname_slot(None),
            hostname_slot(None),
        ]
        .concat();
        let Err(Error::Internal(m)) = hosts_from_slots(&slots, 3) else {
            panic!("expected Error::Internal");
        };
        assert!(m.contains("rank 1"));
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
