//! Shared helpers for ferrompi Criterion benchmarks.
//!
//! Every bench binary declares `mod common;` and calls
//! [`init_mpi_for_bench`] at the top of `main` before constructing any
//! [`criterion::Criterion`] instance. Rank 0 drives Criterion and sends a
//! [`Command`] with [`lead`] before each measured call; the other ranks run
//! [`follow`] until rank 0 sends [`STOP`].

use ferrompi::{Communicator, Mpi, ReduceOp};

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
        // SAFETY: bench binaries are single-threaded at this
        // point — Criterion has not yet spawned its measurement threads, and
        // MPI has just been initialized with ThreadLevel::Single.  Setting an
        // environment variable here races with no other thread.
        unsafe {
            std::env::set_var("CRITERION_HOME", format!("/tmp/ferrompi-bench-rank{rank}"));
        }
    }

    mpi
}

/// What rank 0 tells the other ranks to run next: an operation code and one
/// argument, both chosen by the bench.
pub type Command = [u64; 2];

/// The command that ends [`follow`]. Bench operation codes start at 1.
pub const STOP: Command = [0, 0];

/// Rank 0: sends `cmd` to every other rank.
///
/// The other ranks contribute zeros to a sum allreduce, so they receive
/// `cmd` unchanged.
pub fn lead(world: &Communicator, cmd: Command) {
    let mut received = [0u64; 2];
    world
        .allreduce(&cmd, &mut received, ReduceOp::Sum)
        .expect("bench command allreduce failed");
}

/// Ranks other than 0: runs `run` on each command rank 0 sends with [`lead`],
/// until it sends [`STOP`].
pub fn follow(world: &Communicator, mut run: impl FnMut(Command)) {
    loop {
        let mut cmd = [0u64; 2];
        world
            .allreduce(&[0u64; 2], &mut cmd, ReduceOp::Sum)
            .expect("bench command allreduce failed");
        if cmd == STOP {
            return;
        }
        run(cmd);
    }
}
