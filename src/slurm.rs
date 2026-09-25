//! SLURM scheduler environment helpers.
//!
//! These functions read SLURM environment variables to determine job topology.
//! They return `None` if the variable is not set (e.g., when not running under SLURM).
//!
//! # Environment Variables
//!
//! | Function | Variable | Description |
//! |----------|----------|-------------|
//! | `job_id()` | `SLURM_JOB_ID` | Unique job identifier |
//! | `local_rank()` | `SLURM_LOCALID` | Task ID relative to this node |
//! | `local_size()` | `SLURM_NTASKS_PER_NODE` | Number of tasks on this node |
//! | `num_nodes()` | `SLURM_NNODES` | Total number of nodes |
//! | `cpus_per_task()` | `SLURM_CPUS_PER_TASK` | CPUs allocated per task |
//! | `node_name()` | `SLURM_NODENAME` | Name of this compute node |
//! | `node_list()` | `SLURM_NODELIST` | Compact list of allocated nodes |

use std::env;

/// Check if running under SLURM job scheduler.
pub fn is_slurm_job() -> bool {
    env::var("SLURM_JOB_ID").is_ok()
}

/// Get the SLURM job ID.
pub fn job_id() -> Option<String> {
    env::var("SLURM_JOB_ID").ok()
}

/// Get the local (intra-node) rank of this process.
pub fn local_rank() -> Option<i32> {
    env::var("SLURM_LOCALID").ok().and_then(|s| s.parse().ok())
}

/// Get the number of tasks per node.
pub fn local_size() -> Option<i32> {
    env::var("SLURM_NTASKS_PER_NODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            // Fallback: parse first entry of SLURM_TASKS_PER_NODE (format: "4(x2)")
            env::var("SLURM_TASKS_PER_NODE")
                .ok()
                .and_then(|s| s.split('(').next().and_then(|n| n.parse().ok()))
        })
}

/// Get the total number of nodes allocated.
pub fn num_nodes() -> Option<i32> {
    env::var("SLURM_NNODES").ok().and_then(|s| s.parse().ok())
}

/// Get the number of CPUs per task.
pub fn cpus_per_task() -> Option<i32> {
    env::var("SLURM_CPUS_PER_TASK")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Get the name of the compute node this process is running on.
pub fn node_name() -> Option<String> {
    env::var("SLURM_NODENAME")
        .or_else(|_| env::var("SLURMD_NODENAME"))
        .ok()
}

/// Get the compact node list string (e.g., "node[001-004]").
pub fn node_list() -> Option<String> {
    env::var("SLURM_NODELIST").ok()
}

#[cfg(test)]
mod tests {
    use super::{
        cpus_per_task, is_slurm_job, job_id, local_rank, local_size, node_list, node_name,
        num_nodes,
    };

    #[test]
    fn not_in_slurm_by_default() {
        // In a test environment, we're not in a SLURM job
        // (unless running on a cluster, but CI won't be)
        if std::env::var("SLURM_JOB_ID").is_err() {
            assert!(!is_slurm_job());
            assert!(job_id().is_none());
            assert!(local_rank().is_none());
        }
    }

    /// Set an environment variable for `slurm_env_var_parsing` below.
    fn set_env(key: &str, value: &str) {
        // SAFETY: slurm_env_var_parsing is the only test in this crate that
        // writes SLURM_* variables; this unit-test binary's only concurrent
        // readers of the environment go through std::env, which serializes
        // with this write, and no unit test calls into C code that reads the
        // environment.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    /// Unset an environment variable for `slurm_env_var_parsing` below.
    fn unset_env(key: &str) {
        // SAFETY: slurm_env_var_parsing is the only test in this crate that
        // writes SLURM_* variables; this unit-test binary's only concurrent
        // readers of the environment go through std::env, which serializes
        // with this write, and no unit test calls into C code that reads the
        // environment.
        unsafe {
            std::env::remove_var(key);
        }
    }

    /// Tests that mutate environment variables are combined into a single test
    /// to avoid data races when tests run in parallel. `env::set_var` and
    /// `env::remove_var` are not thread-safe — multiple tests touching the same
    /// env vars concurrently will produce flaky results.
    #[test]
    fn slurm_env_var_parsing() {
        // --- local_size: parses SLURM_TASKS_PER_NODE "4(x2)" format ---
        set_env("SLURM_TASKS_PER_NODE", "4(x2)");
        unset_env("SLURM_NTASKS_PER_NODE");
        assert_eq!(local_size(), Some(4));

        // --- local_size: SLURM_NTASKS_PER_NODE takes priority ---
        set_env("SLURM_NTASKS_PER_NODE", "8");
        set_env("SLURM_TASKS_PER_NODE", "4(x2)");
        assert_eq!(local_size(), Some(8));

        // --- local_size: returns None when neither var is set ---
        unset_env("SLURM_NTASKS_PER_NODE");
        unset_env("SLURM_TASKS_PER_NODE");
        assert_eq!(local_size(), None);

        // --- is_slurm_job: detects SLURM_JOB_ID ---
        set_env("SLURM_JOB_ID", "12345");
        assert!(is_slurm_job());
        assert_eq!(job_id(), Some("12345".to_string()));
        unset_env("SLURM_JOB_ID");

        // --- num_nodes: parses SLURM_NNODES ---
        set_env("SLURM_NNODES", "16");
        assert_eq!(num_nodes(), Some(16));
        unset_env("SLURM_NNODES");

        // --- cpus_per_task: parses SLURM_CPUS_PER_TASK ---
        set_env("SLURM_CPUS_PER_TASK", "4");
        assert_eq!(cpus_per_task(), Some(4));
        unset_env("SLURM_CPUS_PER_TASK");

        // --- node_name: returns None when neither var is set ---
        unset_env("SLURM_NODENAME");
        unset_env("SLURMD_NODENAME");
        assert_eq!(node_name(), None);

        // --- node_name: SLURM_NODENAME takes priority ---
        set_env("SLURM_NODENAME", "node01");
        assert_eq!(node_name(), Some("node01".to_string()));
        unset_env("SLURM_NODENAME");

        // --- node_name: falls back to SLURMD_NODENAME ---
        set_env("SLURMD_NODENAME", "node02");
        assert_eq!(node_name(), Some("node02".to_string()));
        unset_env("SLURMD_NODENAME");

        // --- node_list: returns None when not set ---
        unset_env("SLURM_NODELIST");
        assert_eq!(node_list(), None);

        // --- node_list: returns value when set ---
        set_env("SLURM_NODELIST", "node[001-004]");
        assert_eq!(node_list(), Some("node[001-004]".to_string()));
        unset_env("SLURM_NODELIST");

        // --- local_rank: returns None for non-numeric value ---
        set_env("SLURM_LOCALID", "not_a_number");
        assert_eq!(local_rank(), None);

        // --- local_rank: parses valid numeric value ---
        set_env("SLURM_LOCALID", "3");
        assert_eq!(local_rank(), Some(3));
        unset_env("SLURM_LOCALID");

        // --- local_size: non-numeric NTASKS_PER_NODE falls through to TASKS_PER_NODE ---
        set_env("SLURM_NTASKS_PER_NODE", "garbage");
        set_env("SLURM_TASKS_PER_NODE", "6(x3)");
        assert_eq!(local_size(), Some(6));
        unset_env("SLURM_NTASKS_PER_NODE");
        unset_env("SLURM_TASKS_PER_NODE");

        // --- num_nodes: returns None for non-numeric value ---
        set_env("SLURM_NNODES", "not_a_number");
        assert_eq!(num_nodes(), None);
        unset_env("SLURM_NNODES");

        // --- cpus_per_task: returns None for non-numeric value ---
        set_env("SLURM_CPUS_PER_TASK", "abc");
        assert_eq!(cpus_per_task(), None);
        unset_env("SLURM_CPUS_PER_TASK");
    }
}
