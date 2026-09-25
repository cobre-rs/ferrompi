//! Hybrid MPI+OpenMP example
//!
//! Demonstrates the `MPI_THREAD_FUNNELED` pattern where only the main thread
//! makes MPI calls, while worker threads handle local computation.
//!
//! Build: `cargo build --example hybrid_openmp`
//! Run:   `OMP_NUM_THREADS=4 mpiexec -n 2 --bind-to none ./target/debug/examples/hybrid_openmp`
//!
//! For SLURM:
//! ```text
//! #SBATCH --ntasks=2
//! #SBATCH --cpus-per-task=4
//! srun ./target/release/examples/hybrid_openmp
//! ```
// mpi-test: np=2

use ferrompi::{Mpi, ReduceOp, Result, ThreadLevel};

fn main() -> Result<()> {
    // Request funneled threading: only main thread calls MPI
    let mpi = Mpi::init_thread(ThreadLevel::Funneled)?;

    if mpi.thread_level() < ThreadLevel::Funneled {
        eprintln!(
            "Warning: MPI only provided {:?}, expected Funneled",
            mpi.thread_level()
        );
    }

    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    // Get OMP_NUM_THREADS from environment (or default to 1)
    let num_threads: usize = std::env::var("OMP_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    println!(
        "Rank {}/{}: using {} OpenMP threads",
        rank, size, num_threads
    );

    // In a real program, the OpenMP parallel region would be here.
    // Since Rust doesn't have native OpenMP, this shows the pattern:
    //
    // 1. Main thread: MPI communication
    // 2. All threads: parallel computation (via std::thread, rayon, or FFI to OpenMP)
    // 3. Main thread: MPI communication
    //
    // For actual OpenMP, call into C/C++ via FFI:
    //   extern "C" { fn compute_parallel(data: *mut f64, n: usize); }

    // Simulate local computation result
    let local_result = rank as f64 * num_threads as f64;

    // Only main thread calls MPI (funneled pattern)
    let global_sum = world.allreduce_scalar(local_result, ReduceOp::Sum)?;

    println!("Rank {}: global sum = {}", rank, global_sum);

    Ok(())
}
