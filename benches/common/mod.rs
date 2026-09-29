//! Shared helpers for ferrompi Criterion benchmarks.
//!
//! The `allreduce_roundtrip` and `persistent_vs_iallreduce` Criterion
//! benches declare `mod common;` and call [`init_mpi_for_bench`] at the top
//! of `main` before constructing any [`criterion::Criterion`] instance.
//! Rank 0 drives Criterion and sends a [`Command`] with [`lead`] before each
//! measured call; the other ranks run [`follow`] until rank 0 sends
//! [`STOP`].

use ferrompi::{Communicator, Mpi, ReduceOp};

/// Initialize MPI for a Criterion benchmark binary.
///
/// Must be called **before** `Criterion::default()` so that `MPI_Init` runs
/// on all ranks before any MPI collective is attempted.
///
/// # Panics
///
/// Panics with `"MPI_Init failed in bench"` if [`Mpi::init`] returns an
/// error.  Bench code is test-grade; `expect` is acceptable here per the
/// Rust coding standards allowance for test builds.
pub fn init_mpi_for_bench() -> Mpi {
    Mpi::init().expect("MPI_Init failed in bench")
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
