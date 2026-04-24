//! Shared helpers for ferrompi Criterion benchmarks.
//!
//! Every bench binary declares `mod common;` and calls
//! [`init_mpi_for_bench`] at the top of `main` before constructing any
//! [`criterion::Criterion`] instance.

use ferrompi::{Communicator, Mpi};

/// Initialize MPI for a Criterion benchmark binary.
///
/// Must be called **before** `Criterion::default()` so that:
/// 1. `MPI_Init` runs on all ranks before any MPI collective is attempted.
/// 2. `CRITERION_HOME` is redirected for non-root ranks so that rank 0 is
///    the sole owner of `target/criterion/` and its HTML report.
///
/// Non-root ranks write their ephemeral Criterion output to
/// `/tmp/ferrompi-bench-rank<N>/` instead, which is intentionally excluded
/// from version control.
///
/// # Panics
///
/// Panics with `"MPI_Init failed in bench"` if [`Mpi::init`] returns an
/// error.  Bench code is test-grade; `expect` is acceptable here per the
/// Rust coding standards allowance for test builds.
pub fn init_mpi_for_bench() -> Mpi {
    let mpi = Mpi::init().expect("MPI_Init failed in bench");

    // Rank is now available; redirect Criterion output for non-root ranks so
    // they do not clobber the authoritative `target/criterion/` directory that
    // rank 0 writes.
    let rank = mpi.world().rank();
    if rank != 0 {
        // SAFETY (env mutation): bench binaries are single-threaded at this
        // point — Criterion has not yet spawned its measurement threads, and
        // MPI has just been initialized with ThreadLevel::Single.  Setting an
        // environment variable here races with no other thread.
        unsafe {
            std::env::set_var("CRITERION_HOME", format!("/tmp/ferrompi-bench-rank{rank}"));
        }
    }

    mpi
}

/// Execute a closure only on rank 0 of the given communicator.
///
/// Non-root ranks skip the closure entirely. This is used to gate
/// Criterion benchmark registration so that only rank 0 drives the
/// statistical measurement loop while non-root ranks participate in MPI
/// collectives through the same code path.
#[allow(dead_code)]
pub fn rank_zero_only<F: FnOnce()>(comm: &Communicator, f: F) {
    if comm.rank() == 0 {
        f();
    }
}
