//! Shared memory window example — intra-node communication via MPI shared memory.
//!
//! Demonstrates allocating a shared memory window with [`SharedWindow<T>`],
//! writing from each rank's local portion, and reading across ranks using
//! both fence (active target) and lock/unlock (passive target) synchronization.
//!
//! Run with: mpiexec -n 4 cargo run --example shared_memory --features rma
// mpi-test: np=2..

use ferrompi::{LockType, Mpi, Result, SharedWindow};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    // Create a communicator containing only processes that share memory
    // (i.e., processes on the same physical node).
    let node = world.split_shared()?;
    let local_rank = node.rank();
    let local_size = node.size();

    println!(
        "Global rank {}, local rank {} of {} on node",
        world.rank(),
        local_rank,
        local_size
    );

    // Each process allocates 10 f64 elements in shared memory.
    // All processes' segments are contiguous and directly accessible.
    let count_per_rank: usize = 10;
    let mut win = SharedWindow::<f64>::allocate(&node, count_per_rank)?;

    // ====================================================================
    // Phase 1: Write to local portion
    // ====================================================================
    // Each rank fills its local segment with a recognizable pattern:
    //   rank 0: [0, 1, 2, ..., 9]
    //   rank 1: [10, 11, 12, ..., 19]
    //   etc.
    {
        let local = win.local_slice_mut();
        for (i, x) in local.iter_mut().enumerate() {
            *x = (local_rank as usize * count_per_rank + i) as f64;
        }
    }

    // ====================================================================
    // Phase 2: Fence synchronization (active target)
    // ====================================================================
    // A collective fence ensures all writes from all ranks are visible
    // before any rank reads remote memory. This is the simplest
    // synchronization mode — all processes must participate.
    win.fence()?;

    // Now every rank can safely read any other rank's shared memory.
    // Read rank 0's first 3 values as a sanity check.
    let remote = win.remote_slice(0)?;
    println!(
        "Rank {}: rank 0's data[0..3] = {:?}",
        local_rank,
        &remote[..3]
    );

    // ====================================================================
    // Phase 3: Lock/unlock (passive target) synchronization
    // ====================================================================
    // Passive target access allows fine-grained, one-sided reads without
    // a collective barrier. Here we acquire a shared lock on rank 1's
    // window to read its first value. The RAII LockGuard automatically
    // releases the lock when it goes out of scope.
    if local_size > 1 {
        let _guard = win.lock(LockType::Shared, 1)?;
        let remote_1 = win.remote_slice(1)?;
        println!(
            "Rank {}: rank 1's first value = {} (via lock/unlock)",
            local_rank, remote_1[0]
        );
        // Lock is released here when `_guard` is dropped
    }

    // ====================================================================
    // Cleanup
    // ====================================================================
    // Final fence to ensure all accesses are complete before the window
    // is freed on drop.
    win.fence()?;

    println!("Rank {}: shared memory example complete", local_rank);

    // SharedWindow is freed on drop; node communicator is freed on drop;
    // MPI is finalized when `mpi` is dropped.
    Ok(())
}
