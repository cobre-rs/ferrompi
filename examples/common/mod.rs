#![allow(dead_code)] // each example binary uses a subset of these helpers

use ferrompi::{Communicator, Error, Mpi, MpiErrorClass, ReduceOp, Result};

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

/// True iff `r` is `Err(Error::Mpi { class: MpiErrorClass::Count, .. })`.
pub fn is_count<T>(r: &Result<T>) -> bool {
    matches!(
        r,
        Err(Error::Mpi {
            class: MpiErrorClass::Count,
            ..
        })
    )
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

/// True iff the library provides the MPI 4.0 persistent collectives: it
/// reports MPI 4.0 or later, or it is Open MPI 5 or later, which
/// implements them while reporting MPI 3.1. Mirrors the C shim's
/// compile-time gate.
pub fn has_mpi4_collectives() -> bool {
    if mpi_major() >= 4 {
        return true;
    }
    let library =
        Mpi::library_version().expect("has_mpi4_collectives: Mpi::library_version() failed");
    library
        .strip_prefix("Open MPI v")
        .and_then(|v| v.split('.').next())
        .and_then(|major| major.parse::<u32>().ok())
        .is_some_and(|major| major >= 5)
}
