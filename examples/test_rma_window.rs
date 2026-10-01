//! Integration test for SharedWindow (RMA) operations.
//!
//! Exercises all `SharedWindow<T>` methods including allocation, local/remote
//! slice access, fence synchronization, lock/unlock (passive target) with
//! both `LockGuard` and `LockAllGuard`, flush operations, and multi-type
//! coverage (f64, i32, u64).
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_rma_window
// mpi-test: np=2..

use ferrompi::{LockType, Mpi, SharedWindow};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_window requires at least 2 processes, got {size}"
    );

    // All SharedWindow operations require a shared-memory communicator.
    let node = world.split_shared().expect("split_shared failed");
    let node_rank = node.rank();
    let node_size = node.size();

    // ========================================================================
    // Test 1: Allocate and basic properties
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            SharedWindow::<f64>::allocate(&node, count).expect("SharedWindow allocate failed");

        // Verify raw_handle returns a non-negative handle
        assert!(
            win.raw_handle() >= 0,
            "raw_handle() should be non-negative, got {}",
            win.raw_handle()
        );

        // Verify comm_size matches the node communicator size
        assert_eq!(
            win.comm_size(),
            node_size,
            "comm_size() = {}, expected node_size = {}",
            win.comm_size(),
            node_size
        );

        // Write via local_slice_mut
        {
            let local = win.local_slice_mut();
            assert_eq!(
                local.len(),
                count,
                "local_slice_mut length = {}, expected {count}",
                local.len()
            );
            for (i, x) in local.iter_mut().enumerate() {
                *x = (node_rank as usize * count + i) as f64;
            }
        }

        // Verify via local_slice (read-only)
        {
            let local = win.local_slice();
            assert_eq!(
                local.len(),
                count,
                "local_slice length = {}, expected {count}",
                local.len()
            );
            for (i, &x) in local.iter().enumerate() {
                let expected = (node_rank as usize * count + i) as f64;
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: local_slice[{i}] = {x}, expected {expected}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: allocate and basic properties (raw_handle, comm_size, local_slice, local_slice_mut)");
        }

        // SharedWindow dropped here — exercises Drop
    }

    world.barrier().expect("barrier after test 1 failed");

    // ========================================================================
    // Test 2: Fence synchronization + remote_slice
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            SharedWindow::<f64>::allocate(&node, count).expect("SharedWindow allocate failed");

        // Each rank writes a recognizable pattern to its local segment
        {
            let local = win.local_slice_mut();
            for (i, x) in local.iter_mut().enumerate() {
                *x = (node_rank as usize * count + i) as f64;
            }
        }

        // Fence: ensure all writes are visible to all ranks
        win.fence().expect("fence (write phase) failed");

        // Read rank 0's remote data
        let remote = win.remote_slice(0).expect("remote_slice(0) failed");
        assert_eq!(
            remote.len(),
            count,
            "remote_slice(0) length = {}, expected {count}",
            remote.len()
        );
        for (i, &x) in remote.iter().enumerate() {
            let expected = i as f64; // rank 0 wrote [0, 1, 2, ..., 9]
            assert!(
                (x - expected).abs() < f64::EPSILON,
                "rank {node_rank}: remote_slice(0)[{i}] = {x}, expected {expected}"
            );
        }

        // If multiple ranks exist, also verify a non-zero rank's data
        if node_size > 1 {
            let remote_1 = win.remote_slice(1).expect("remote_slice(1) failed");
            for (i, &x) in remote_1.iter().enumerate() {
                let expected = (count + i) as f64; // rank 1 wrote [10, 11, ..., 19]
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: remote_slice(1)[{i}] = {x}, expected {expected}"
                );
            }
        }

        // Closing fence
        win.fence().expect("fence (read phase) failed");

        if rank == 0 {
            println!("PASS: fence synchronization + remote_slice");
        }
    }

    world.barrier().expect("barrier after test 2 failed");

    // ========================================================================
    // Test 3: Lock/unlock with LockGuard
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            SharedWindow::<f64>::allocate(&node, count).expect("SharedWindow allocate failed");

        // Write local data
        {
            let local = win.local_slice_mut();
            for (i, x) in local.iter_mut().enumerate() {
                *x = (node_rank as usize * 1000 + i) as f64;
            }
        }

        // Fence to make writes visible
        win.fence().expect("fence (lock test write phase) failed");

        // All ranks acquire a Shared lock on rank 0 and read
        {
            let guard = win
                .lock(LockType::Shared, 0)
                .expect("lock(Shared, 0) failed");
            let remote = win.remote_slice(0).expect("remote_slice(0) in lock failed");
            for (i, &x) in remote.iter().enumerate() {
                let expected = i as f64; // rank 0 wrote [0, 1, ..., 9]
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: locked remote_slice(0)[{i}] = {x}, expected {expected}"
                );
            }
            // Flush to ensure all RMA ops complete at target
            guard.flush().expect("LockGuard::flush() failed");
            // LockGuard dropped here — exercises LockGuard::drop (unlock)
        }

        // Only rank 0 acquires an Exclusive lock on itself (safe — no contention)
        if node_rank == 0 {
            let guard = win
                .lock(LockType::Exclusive, 0)
                .expect("lock(Exclusive, 0) failed");
            let remote = win
                .remote_slice(0)
                .expect("remote_slice(0) in exclusive lock failed");
            assert!(
                (remote[0]).abs() < f64::EPSILON,
                "rank 0: exclusive lock remote[0] = {}, expected 0.0",
                remote[0]
            );
            guard.flush().expect("exclusive LockGuard::flush() failed");
            // LockGuard dropped — exercises exclusive unlock
        }

        // Final fence
        win.fence().expect("fence (lock test end) failed");

        if rank == 0 {
            println!("PASS: lock/unlock with LockGuard (Shared + Exclusive)");
        }
    }

    world.barrier().expect("barrier after test 3 failed");

    // ========================================================================
    // Test 4: lock_all / LockAllGuard
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            SharedWindow::<f64>::allocate(&node, count).expect("SharedWindow allocate failed");

        // Write local data
        {
            let local = win.local_slice_mut();
            for (i, x) in local.iter_mut().enumerate() {
                *x = (node_rank as usize * count + i + 100) as f64;
            }
        }

        // Fence to make writes visible
        win.fence()
            .expect("fence (lock_all test write phase) failed");

        // lock_all — acquires shared locks on all ranks
        {
            let guard = win.lock_all().expect("lock_all() failed");

            // Read rank 0's data via remote_slice
            let remote = win
                .remote_slice(0)
                .expect("remote_slice(0) in lock_all failed");
            for (i, &x) in remote.iter().enumerate() {
                let expected = (i + 100) as f64; // rank 0 wrote [100, 101, ..., 109]
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: lock_all remote_slice(0)[{i}] = {x}, expected {expected}"
                );
            }

            // flush(0) — flush ops to rank 0
            guard.flush(0).expect("LockAllGuard::flush(0) failed");

            // flush_all — flush ops to all ranks
            guard.flush_all().expect("LockAllGuard::flush_all() failed");

            // LockAllGuard dropped here — exercises LockAllGuard::drop (unlock_all)
        }

        // Final fence
        win.fence().expect("fence (lock_all test end) failed");

        if rank == 0 {
            println!("PASS: lock_all / LockAllGuard (flush, flush_all)");
        }
    }

    world.barrier().expect("barrier after test 4 failed");

    // ========================================================================
    // Test 5: Multi-type coverage (i32 and u64)
    // ========================================================================
    {
        // --- SharedWindow<i32> ---
        let count: usize = 8;
        let mut win_i32 =
            SharedWindow::<i32>::allocate(&node, count).expect("SharedWindow<i32> allocate failed");

        {
            let local = win_i32.local_slice_mut();
            for (i, x) in local.iter_mut().enumerate() {
                *x = node_rank * 100 + i as i32;
            }
        }

        // Verify via local_slice
        {
            let local = win_i32.local_slice();
            for (i, &x) in local.iter().enumerate() {
                let expected = node_rank * 100 + i as i32;
                assert_eq!(
                    x, expected,
                    "rank {node_rank}: i32 local_slice[{i}] = {x}, expected {expected}"
                );
            }
        }

        // Fence + remote read
        win_i32.fence().expect("SharedWindow<i32> fence failed");
        let remote_i32 = win_i32
            .remote_slice(0)
            .expect("SharedWindow<i32> remote_slice(0) failed");
        for (i, &x) in remote_i32.iter().enumerate() {
            let expected = i as i32; // rank 0 wrote [0, 1, ..., 7]
            assert_eq!(
                x, expected,
                "rank {node_rank}: i32 remote_slice(0)[{i}] = {x}, expected {expected}"
            );
        }
        win_i32
            .fence()
            .expect("SharedWindow<i32> closing fence failed");

        if rank == 0 {
            println!("PASS: SharedWindow<i32> allocate, write, read");
        }

        // --- SharedWindow<u64> ---
        let mut win_u64 =
            SharedWindow::<u64>::allocate(&node, count).expect("SharedWindow<u64> allocate failed");

        {
            let local = win_u64.local_slice_mut();
            for (i, x) in local.iter_mut().enumerate() {
                *x = node_rank as u64 * 1000 + i as u64;
            }
        }

        // Verify via local_slice
        {
            let local = win_u64.local_slice();
            for (i, &x) in local.iter().enumerate() {
                let expected = node_rank as u64 * 1000 + i as u64;
                assert_eq!(
                    x, expected,
                    "rank {node_rank}: u64 local_slice[{i}] = {x}, expected {expected}"
                );
            }
        }

        // Fence + remote read
        win_u64.fence().expect("SharedWindow<u64> fence failed");
        let remote_u64 = win_u64
            .remote_slice(0)
            .expect("SharedWindow<u64> remote_slice(0) failed");
        for (i, &x) in remote_u64.iter().enumerate() {
            let expected = i as u64; // rank 0 wrote [0, 1, ..., 7]
            assert_eq!(
                x, expected,
                "rank {node_rank}: u64 remote_slice(0)[{i}] = {x}, expected {expected}"
            );
        }
        win_u64
            .fence()
            .expect("SharedWindow<u64> closing fence failed");

        if rank == 0 {
            println!("PASS: SharedWindow<u64> allocate, write, read");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All RMA window tests passed! (5 tests)");
        println!("========================================");
    }
}
