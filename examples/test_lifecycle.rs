//! Integration test for MPI lifecycle functions.
//!
//! Exercises Mpi::init, version, wtime, is_initialized, is_finalized,
//! thread_level, and world.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_lifecycle
// mpi-test: np=2

use ferrompi::{Mpi, ThreadLevel};

fn main() {
    // Test init_thread (which init() delegates to)
    let mpi = Mpi::init().expect("MPI init failed");

    // Test is_initialized after init
    assert!(
        Mpi::is_initialized(),
        "is_initialized should be true after init"
    );
    println!("PASS: is_initialized");

    // Test is_finalized before drop
    assert!(
        !Mpi::is_finalized(),
        "is_finalized should be false before drop"
    );
    println!("PASS: is_finalized (false before drop)");

    // Test thread_level
    let level = mpi.thread_level();
    assert!(
        level >= ThreadLevel::Single,
        "thread_level should be >= Single"
    );
    println!("PASS: thread_level = {:?}", level);

    // Test version
    let version = Mpi::version().expect("version() failed");
    assert!(!version.is_empty(), "version string should not be empty");
    println!("PASS: version = {}", version);

    // Test wtime
    let t1 = Mpi::wtime();
    assert!(t1 >= 0.0, "wtime should return non-negative value");
    let t2 = Mpi::wtime();
    assert!(t2 >= t1, "wtime should be monotonic");
    println!("PASS: wtime = {}", t1);

    // Test world
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();
    assert!(rank >= 0 && rank < size, "rank should be in [0, size)");
    assert!(size >= 1, "size should be >= 1");
    println!("PASS: world rank={} size={}", rank, size);

    // Test processor_name
    let name = world.processor_name().expect("processor_name failed");
    assert!(!name.is_empty(), "processor_name should not be empty");
    println!("PASS: processor_name = {}", name);

    // Test raw_handle
    let handle = world.raw_handle();
    assert!(handle >= 0, "raw_handle should be non-negative");
    println!("PASS: raw_handle = {}", handle);

    // Mpi drops here, which calls MPI_Finalize
    drop(mpi);

    if rank == 0 {
        println!("\n========================================");
        println!("All lifecycle tests passed!");
        println!("========================================");
    }
}
