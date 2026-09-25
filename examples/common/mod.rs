#![allow(dead_code)] // each example binary uses a subset of these helpers

use ferrompi::{Communicator, Mpi, ReduceOp};

/// Aggregates this rank's verdict across `world` via `allreduce(Min)`. If any
/// rank reports failure, rank 0 prints `FAIL: {name}` and every rank exits
/// with status 1.
pub fn check(world: &Communicator, ok: bool, name: &str) {
    let global_ok = world
        .allreduce_scalar(i32::from(ok), ReduceOp::Min)
        .expect("check: allreduce failed");

    if global_ok == 0 {
        if world.rank() == 0 {
            eprintln!("FAIL: {name}");
        }
        std::process::exit(1);
    }
}

/// Prints `SKIP: {reason}` from rank 0.
pub fn skip(world: &Communicator, reason: &str) {
    if world.rank() == 0 {
        println!("SKIP: {reason}");
    }
}

/// Parses the major version number from `Mpi::version()` (format `MPI
/// <major>.<minor>`).
///
/// # Panics
///
/// Panics if the version string cannot be parsed. Never defaults to 0,
/// because a 0 would turn a version-parse failure into a silent SKIP.
pub fn mpi_major() -> u32 {
    let version = Mpi::version().expect("mpi_major: Mpi::version() failed");
    version
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.split('.').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("mpi_major: could not parse version string {version:?}"))
}
