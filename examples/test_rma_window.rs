//! Integration test for shared-memory windows (RMA) from `Win::allocate_shared`.
//!
//! Exercises allocation and properties, the window's `local()` view and the
//! epoch handles' `remote(rank)` views, fence epochs, passive-target `lock` and
//! `lock_all` epochs with the passive handle's `flush`, `flush_all` and `sync`,
//! and multi-type coverage (f64, i32, u64).
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_rma_window
// mpi-test: np=2.. release

use ferrompi::{LockType, Mpi, Win, WinFenceAssert};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_window requires at least 2 processes, got {size}"
    );

    // Win::allocate_shared requires a shared-memory communicator.
    let node = world.split_shared().expect("split_shared failed");
    let node_rank = node.rank();
    let node_size = node.size();
    let none = WinFenceAssert::none();

    // ========================================================================
    // Test 1: Allocate and basic properties
    // ========================================================================
    {
        let count: usize = 10;
        let win = Win::<f64>::allocate_shared(&node, count).expect("Win::allocate_shared failed");

        // Verify comm_size matches the node communicator size
        assert_eq!(
            win.comm_size(),
            node_size,
            "comm_size() = {}, expected node_size = {}",
            win.comm_size(),
            node_size
        );

        // Write through the local view
        {
            let local = win.local();
            assert_eq!(
                local.len(),
                count,
                "local() length = {}, expected {count}",
                local.len()
            );
            for i in 0..count {
                local.write(i, (node_rank as usize * count + i) as f64);
            }
        }

        // Verify through a second view of the same region
        {
            let local = win.local();
            assert_eq!(
                local.len(),
                count,
                "second local() length = {}, expected {count}",
                local.len()
            );
            for i in 0..count {
                let x = local.read(i);
                let expected = (node_rank as usize * count + i) as f64;
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: local().read({i}) = {x}, expected {expected}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: allocate and basic properties (comm_size, local view)");
        }

        // Win dropped here — exercises Drop
    }

    world.barrier().expect("barrier after test 1 failed");

    // ========================================================================
    // Test 2: Fence epochs + remote views
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            Win::<f64>::allocate_shared(&node, count).expect("Win::allocate_shared failed");

        // Each rank writes a recognizable pattern to its own region. The
        // closing fence makes the stores visible.
        win.fence(none, none, |e| {
            for i in 0..count {
                e.local().write(i, (node_rank as usize * count + i) as f64);
            }
            Ok(())
        })
        .expect("fence (write phase) failed");

        // The next fence epoch reads the regions of rank 0 and of rank 1
        let (remote_0, remote_1) = win
            .fence(none, none, |e| {
                let view_0 = e.remote(0)?;
                let mut remote_0 = vec![0.0f64; view_0.len()];
                view_0.copy_to_slice(&mut remote_0);

                // If multiple ranks exist, also read a non-zero rank's data
                let mut remote_1 = vec![0.0f64; count];
                if node_size > 1 {
                    e.remote(1)?.copy_to_slice(&mut remote_1);
                }
                Ok((remote_0, remote_1))
            })
            .expect("fence (read phase) failed");

        assert_eq!(
            remote_0.len(),
            count,
            "remote(0) length = {}, expected {count}",
            remote_0.len()
        );
        for (i, &x) in remote_0.iter().enumerate() {
            let expected = i as f64; // rank 0 wrote [0, 1, 2, ..., 9]
            assert!(
                (x - expected).abs() < f64::EPSILON,
                "rank {node_rank}: remote(0)[{i}] = {x}, expected {expected}"
            );
        }

        if node_size > 1 {
            for (i, &x) in remote_1.iter().enumerate() {
                let expected = (count + i) as f64; // rank 1 wrote [10, 11, ..., 19]
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: remote(1)[{i}] = {x}, expected {expected}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: fence epochs + remote views");
        }
    }

    world.barrier().expect("barrier after test 2 failed");

    // ========================================================================
    // Test 3: Lock/unlock (Shared + Exclusive) after a fence
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            Win::<f64>::allocate_shared(&node, count).expect("Win::allocate_shared failed");

        // Write local data
        for i in 0..count {
            win.local().write(i, (node_rank as usize * 1000 + i) as f64);
        }

        // The closing fence asserts that it opens no epoch, and the barrier
        // makes sure every rank left it before any rank takes a lock.
        win.fence(none, WinFenceAssert::no_succeed(), |_| Ok(()))
            .expect("fence (lock test write phase) failed");
        node.barrier()
            .expect("barrier (lock test write phase) failed");

        // All ranks acquire a Shared lock on rank 0 and read
        let remote = win
            .lock(LockType::Shared, 0, |e| {
                e.sync()?;
                let mut remote = vec![0.0f64; count];
                e.remote(0)?.copy_to_slice(&mut remote);
                // Flush the locked rank: the epoch issues no RMA operation, so this
                // completes nothing
                e.flush(0)?;
                Ok(remote)
            })
            .expect("lock(Shared, 0) failed");
        for (i, &x) in remote.iter().enumerate() {
            let expected = i as f64; // rank 0 wrote [0, 1, ..., 9]
            assert!(
                (x - expected).abs() < f64::EPSILON,
                "rank {node_rank}: locked remote(0)[{i}] = {x}, expected {expected}"
            );
        }

        // Only rank 0 acquires an Exclusive lock on itself; it waits for the
        // Shared locks the other ranks may still hold on rank 0
        if node_rank == 0 {
            let first = win
                .lock(LockType::Exclusive, 0, |e| {
                    let first = e.remote(0)?.read(0);
                    e.flush(0)?;
                    Ok(first)
                })
                .expect("lock(Exclusive, 0) failed");
            assert!(
                first.abs() < f64::EPSILON,
                "rank 0: exclusive lock remote(0)[0] = {first}, expected 0.0"
            );
        }

        if rank == 0 {
            println!("PASS: lock/unlock (Shared + Exclusive) after a no_succeed fence");
        }
    }

    world.barrier().expect("barrier after test 3 failed");

    // ========================================================================
    // Test 4: lock_all with the shared-memory pattern
    // ========================================================================
    {
        let count: usize = 10;
        let mut win =
            Win::<f64>::allocate_shared(&node, count).expect("Win::allocate_shared failed");

        // Write local data
        for i in 0..count {
            win.local()
                .write(i, (node_rank as usize * count + i + 100) as f64);
        }

        // The closing fence asserts that it opens no epoch, and the barrier
        // makes sure every rank left it before any rank takes a lock.
        win.fence(none, WinFenceAssert::no_succeed(), |_| Ok(()))
            .expect("fence (lock_all test write phase) failed");
        node.barrier()
            .expect("barrier (lock_all test write phase) failed");

        // lock_all — shared locks on all ranks. Each rank adds 1000 to slot 0
        // of its own region, then store, sync, barrier, sync, load.
        let regions = win
            .lock_all(|e| {
                let own = e.local();
                own.write(0, own.read(0) + 1000.0);
                e.sync()?;
                node.barrier()?;
                e.sync()?;

                let mut regions = Vec::new();
                for r in 0..node_size {
                    let view = e.remote(r)?;
                    let mut region = vec![0.0f64; view.len()];
                    view.copy_to_slice(&mut region);
                    regions.push(region);
                }

                e.flush(0)?;
                e.flush_all()?;
                Ok(regions)
            })
            .expect("lock_all with the shared-memory pattern failed");

        for (r, region) in regions.iter().enumerate() {
            assert_eq!(
                region.len(),
                count,
                "rank {node_rank}: lock_all remote({r}) length = {}, expected {count}",
                region.len()
            );
            for (i, &x) in region.iter().enumerate() {
                // Rank r wrote [r*10 + 100, ...]; slot 0 was then raised by 1000
                let bump = if i == 0 { 1000.0 } else { 0.0 };
                let expected = (r * count + i + 100) as f64 + bump;
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "rank {node_rank}: lock_all remote({r})[{i}] = {x}, expected {expected}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: lock_all shared-memory pattern (sync, barrier, flush, flush_all)");
        }
    }

    world.barrier().expect("barrier after test 4 failed");

    // ========================================================================
    // Test 5: Multi-type coverage (i32 and u64)
    // ========================================================================
    {
        // --- Win<i32> ---
        let count: usize = 8;
        let mut win_i32 =
            Win::<i32>::allocate_shared(&node, count).expect("Win<i32> allocate_shared failed");

        win_i32
            .fence(none, none, |e| {
                for i in 0..count {
                    e.local().write(i, node_rank * 100 + i as i32);
                }
                Ok(())
            })
            .expect("Win<i32> write fence failed");

        // Verify through the local view
        let mut local_i32 = vec![0i32; count];
        win_i32.local().copy_to_slice(&mut local_i32);
        for (i, &x) in local_i32.iter().enumerate() {
            let expected = node_rank * 100 + i as i32;
            assert_eq!(
                x, expected,
                "rank {node_rank}: i32 local()[{i}] = {x}, expected {expected}"
            );
        }

        // Fence epoch + remote read
        let remote_i32 = win_i32
            .fence(none, none, |e| {
                let mut remote = vec![0i32; count];
                e.remote(0)?.copy_to_slice(&mut remote);
                Ok(remote)
            })
            .expect("Win<i32> read fence failed");
        for (i, &x) in remote_i32.iter().enumerate() {
            let expected = i as i32; // rank 0 wrote [0, 1, ..., 7]
            assert_eq!(
                x, expected,
                "rank {node_rank}: i32 remote(0)[{i}] = {x}, expected {expected}"
            );
        }

        if rank == 0 {
            println!("PASS: Win<i32> allocate_shared, write, read");
        }

        // --- Win<u64> ---
        let mut win_u64 =
            Win::<u64>::allocate_shared(&node, count).expect("Win<u64> allocate_shared failed");

        win_u64
            .fence(none, none, |e| {
                for i in 0..count {
                    e.local().write(i, node_rank as u64 * 1000 + i as u64);
                }
                Ok(())
            })
            .expect("Win<u64> write fence failed");

        // Verify through the local view
        let mut local_u64 = vec![0u64; count];
        win_u64.local().copy_to_slice(&mut local_u64);
        for (i, &x) in local_u64.iter().enumerate() {
            let expected = node_rank as u64 * 1000 + i as u64;
            assert_eq!(
                x, expected,
                "rank {node_rank}: u64 local()[{i}] = {x}, expected {expected}"
            );
        }

        // Fence epoch + remote read
        let remote_u64 = win_u64
            .fence(none, none, |e| {
                let mut remote = vec![0u64; count];
                e.remote(0)?.copy_to_slice(&mut remote);
                Ok(remote)
            })
            .expect("Win<u64> read fence failed");
        for (i, &x) in remote_u64.iter().enumerate() {
            let expected = i as u64; // rank 0 wrote [0, 1, ..., 7]
            assert_eq!(
                x, expected,
                "rank {node_rank}: u64 remote(0)[{i}] = {x}, expected {expected}"
            );
        }

        if rank == 0 {
            println!("PASS: Win<u64> allocate_shared, write, read");
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
