//! Integration test for ALL persistent collective operations (MPI 4.0+).
//!
//! Tests the full PersistentRequest lifecycle: init, start, wait, test,
//! start_all, wait_all, and drop. Gracefully skips if MPI < 4.0.
//!
//! Exercises all 15 persistent collective `_init` methods in `Comm`:
//! bcast_init, allreduce_init, allreduce_init_inplace, gather_init,
//! reduce_init, scatter_init, allgather_init, scan_init, exscan_init,
//! alltoall_init, gatherv_init, scatterv_init, allgatherv_init,
//! alltoallv_init, and reduce_scatter_block_init.
//!
//! Additionally tests PersistentRequest::test() polling, start_all/wait_all,
//! drop without start (inactive path), and drop after start without wait
//! (active path) for full Drop coverage.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_persistent

use ferrompi::{Mpi, PersistentRequest, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_persistent requires at least 2 processes, got {size}"
    );

    if common::mpi_major() < 4 {
        let version = Mpi::version().expect("Mpi::version() failed");
        common::skip(&world, &format!("bcast_init needs MPI 4 ({version})"));
        return;
    }

    let mut test_count = 0u32;

    // ========================================================================
    // Test 1: bcast_init (with start/wait lifecycle + reuse)
    // ========================================================================
    {
        let mut data = vec![0.0f64; 10];
        if rank == 0 {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (i + 1) as f64;
            }
        }
        let mut req = world.bcast_init(&mut data, 0).expect("bcast_init failed");

        // Verify initial state
        assert!(!req.is_active(), "request should be inactive after init");

        // First start/wait cycle
        req.start().expect("bcast_init start failed");
        assert!(req.is_active(), "request should be active after start");
        req.wait().expect("bcast_init wait failed");
        assert!(!req.is_active(), "request should be inactive after wait");

        for (i, &x) in data.iter().enumerate() {
            assert!(
                (x - (i + 1) as f64).abs() < f64::EPSILON,
                "rank {rank}: bcast_init data[{i}] = {x}, expected {}",
                (i + 1) as f64
            );
        }

        // Reuse: second start/wait cycle with new data
        if rank == 0 {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (i + 1) as f64 * 100.0;
            }
        }
        req.start().expect("bcast_init reuse start failed");
        req.wait().expect("bcast_init reuse wait failed");

        for (i, &x) in data.iter().enumerate() {
            assert!(
                (x - (i + 1) as f64 * 100.0).abs() < f64::EPSILON,
                "rank {rank}: bcast_init reuse data[{i}] = {x}, expected {}",
                (i + 1) as f64 * 100.0
            );
        }

        test_count += 1;
        if rank == 0 {
            println!("PASS: bcast_init (with reuse)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 2: allreduce_init (Sum)
    // ========================================================================
    {
        let send = vec![rank as f64; 5];
        let mut recv = vec![0.0f64; 5];
        let mut req = world
            .allreduce_init(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce_init failed");

        req.start().expect("allreduce_init start failed");
        req.wait().expect("allreduce_init wait failed");

        // Sum of ranks: 0 + 1 + ... + (size-1) = size*(size-1)/2
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: allreduce_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: allreduce_init (Sum)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 3: allreduce_init_inplace (Sum)
    // ========================================================================
    {
        let mut data = vec![rank as f64; 5];
        let mut req = world
            .allreduce_init_inplace(&mut data, ReduceOp::Sum)
            .expect("allreduce_init_inplace failed");

        req.start().expect("allreduce_init_inplace start failed");
        req.wait().expect("allreduce_init_inplace wait failed");

        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: allreduce_init_inplace data[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: allreduce_init_inplace (Sum)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 4: gather_init
    // ========================================================================
    {
        let send = vec![rank as f64 * 10.0, rank as f64 * 10.0 + 1.0];
        let mut recv = vec![0.0f64; 2 * size as usize];
        let mut req = world
            .gather_init(&send, &mut recv, 0)
            .expect("gather_init failed");

        req.start().expect("gather_init start failed");
        req.wait().expect("gather_init wait failed");

        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert!(
                    (recv[idx] - r as f64 * 10.0).abs() < f64::EPSILON,
                    "gather_init recv[{idx}] = {}, expected {}",
                    recv[idx],
                    r as f64 * 10.0
                );
                assert!(
                    (recv[idx + 1] - (r as f64 * 10.0 + 1.0)).abs() < f64::EPSILON,
                    "gather_init recv[{}] = {}, expected {}",
                    idx + 1,
                    recv[idx + 1],
                    r as f64 * 10.0 + 1.0
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: gather_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 5: reduce_init (Sum, to root=0)
    // ========================================================================
    {
        let send = vec![(rank + 1) as f64; 4];
        let mut recv = vec![0.0f64; 4];
        let mut req = world
            .reduce_init(&send, &mut recv, ReduceOp::Sum, 0)
            .expect("reduce_init failed");

        req.start().expect("reduce_init start failed");
        req.wait().expect("reduce_init wait failed");

        if rank == 0 {
            // Sum of 1 + 2 + ... + size = size*(size+1)/2
            let expected = (size * (size + 1) / 2) as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "reduce_init recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: reduce_init (Sum)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 6: scatter_init
    // ========================================================================
    {
        let send_data = if rank == 0 {
            (0..size * 3).map(|x| x as f64).collect::<Vec<f64>>()
        } else {
            vec![0.0f64; (size * 3) as usize]
        };
        let mut recv = vec![0.0f64; 3];
        let mut req = world
            .scatter_init(&send_data, &mut recv, 0)
            .expect("scatter_init failed");

        req.start().expect("scatter_init start failed");
        req.wait().expect("scatter_init wait failed");

        let base = rank * 3;
        for (i, &v) in recv.iter().enumerate() {
            let expected = (base + i as i32) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: scatter_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: scatter_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 7: allgather_init
    // ========================================================================
    {
        let send = vec![rank as f64; 2];
        let mut recv = vec![0.0f64; 2 * size as usize];
        let mut req = world
            .allgather_init(&send, &mut recv)
            .expect("allgather_init failed");

        req.start().expect("allgather_init start failed");
        req.wait().expect("allgather_init wait failed");

        for r in 0..size {
            let idx = r as usize * 2;
            assert!(
                (recv[idx] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: allgather_init recv[{idx}] = {}, expected {}",
                recv[idx],
                r as f64
            );
            assert!(
                (recv[idx + 1] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: allgather_init recv[{}] = {}, expected {}",
                idx + 1,
                recv[idx + 1],
                r as f64
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: allgather_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 8: scan_init (inclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        let mut req = world
            .scan_init(&send, &mut recv, ReduceOp::Sum)
            .expect("scan_init failed");

        req.start().expect("scan_init start failed");
        req.wait().expect("scan_init wait failed");

        // On rank i, inclusive scan of 1.0 from all ranks 0..=i => (i+1)
        let expected = (rank + 1) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: scan_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: scan_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 9: exscan_init (exclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        let mut req = world
            .exscan_init(&send, &mut recv, ReduceOp::Sum)
            .expect("exscan_init failed");

        req.start().expect("exscan_init start failed");
        req.wait().expect("exscan_init wait failed");

        // On rank i > 0, exclusive scan of 1.0 from ranks 0..i => i
        // On rank 0, result is undefined per MPI standard — skip assertion
        if rank > 0 {
            let expected = rank as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "rank {rank}: exscan_init recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: exscan_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 10: alltoall_init
    // ========================================================================
    {
        // Each rank sends its rank value to every other rank.
        // send = [rank, rank, ..., rank] (one per destination rank)
        let send = vec![rank as f64; size as usize];
        let mut recv = vec![0.0f64; size as usize];
        let mut req = world
            .alltoall_init(&send, &mut recv)
            .expect("alltoall_init failed");

        req.start().expect("alltoall_init start failed");
        req.wait().expect("alltoall_init wait failed");

        // recv[i] should be i (the value that rank i sent)
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - i as f64).abs() < f64::EPSILON,
                "rank {rank}: alltoall_init recv[{i}] = {v}, expected {i}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: alltoall_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 11: gatherv_init (variable-count gather)
    // ========================================================================
    {
        // Each rank sends (rank + 1) elements, each equal to rank as f64.
        let send_count = (rank + 1) as usize;
        let send = vec![rank as f64; send_count];

        let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
        let displs: Vec<i32> = recvcounts
            .iter()
            .scan(0, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();
        let total: usize = recvcounts.iter().map(|&c| c as usize).sum();

        let mut recv = vec![0.0f64; total];

        let mut req = world
            .gatherv_init(&send, &mut recv, &recvcounts, &displs, 0)
            .expect("gatherv_init failed");

        req.start().expect("gatherv_init start failed");
        req.wait().expect("gatherv_init wait failed");

        if rank == 0 {
            for r in 0..size {
                let offset = displs[r as usize] as usize;
                let count = recvcounts[r as usize] as usize;
                for j in 0..count {
                    assert!(
                        (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                        "gatherv_init: recv[{}] = {}, expected {}",
                        offset + j,
                        recv[offset + j],
                        r as f64
                    );
                }
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: gatherv_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 12: scatterv_init (variable-count scatter)
    // ========================================================================
    {
        let recv_count = (rank + 1) as usize;
        let sendcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
        let displs: Vec<i32> = sendcounts
            .iter()
            .scan(0, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();
        let total: usize = sendcounts.iter().map(|&c| c as usize).sum();

        // Root sends: rank r gets (r+1) elements, each = r * 100.0
        let send = if rank == 0 {
            let mut buf = vec![0.0f64; total];
            for r in 0..size {
                let offset = displs[r as usize] as usize;
                let count = sendcounts[r as usize] as usize;
                for j in 0..count {
                    buf[offset + j] = r as f64 * 100.0;
                }
            }
            buf
        } else {
            vec![0.0f64; total]
        };

        let mut recv = vec![0.0f64; recv_count];
        let mut req = world
            .scatterv_init(&send, &sendcounts, &displs, &mut recv, 0)
            .expect("scatterv_init failed");

        req.start().expect("scatterv_init start failed");
        req.wait().expect("scatterv_init wait failed");

        let expected = rank as f64 * 100.0;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: scatterv_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: scatterv_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 13: allgatherv_init (variable-count allgather)
    // ========================================================================
    {
        let send_count = (rank + 1) as usize;
        let send = vec![rank as f64; send_count];

        let recvcounts: Vec<i32> = (0..size).map(|r| r + 1).collect();
        let displs: Vec<i32> = recvcounts
            .iter()
            .scan(0, |acc, &c| {
                let d = *acc;
                *acc += c;
                Some(d)
            })
            .collect();
        let total: usize = recvcounts.iter().map(|&c| c as usize).sum();
        let mut recv = vec![0.0f64; total];

        let mut req = world
            .allgatherv_init(&send, &mut recv, &recvcounts, &displs)
            .expect("allgatherv_init failed");

        req.start().expect("allgatherv_init start failed");
        req.wait().expect("allgatherv_init wait failed");

        for r in 0..size {
            let offset = displs[r as usize] as usize;
            let count = recvcounts[r as usize] as usize;
            for j in 0..count {
                assert!(
                    (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                    "rank {rank}: allgatherv_init recv[{}] = {}, expected {}",
                    offset + j,
                    recv[offset + j],
                    r as f64
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: allgatherv_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 14: alltoallv_init (variable-count all-to-all)
    // ========================================================================
    {
        // Each rank sends 1 element to every other rank.
        // The element sent from rank r to rank d is (r * 1000 + d) as f64.
        let sz = size as usize;
        let sendcounts = vec![1i32; sz];
        let sdispls: Vec<i32> = (0..sz as i32).collect();
        let recvcounts = vec![1i32; sz];
        let rdispls: Vec<i32> = (0..sz as i32).collect();

        let send: Vec<f64> = (0..size).map(|d| (rank * 1000 + d) as f64).collect();
        let mut recv = vec![0.0f64; sz];

        let mut req = world
            .alltoallv_init(
                &send,
                &sendcounts,
                &sdispls,
                &mut recv,
                &recvcounts,
                &rdispls,
            )
            .expect("alltoallv_init failed");

        req.start().expect("alltoallv_init start failed");
        req.wait().expect("alltoallv_init wait failed");

        // recv[i] should be the value that rank i sent to us = i * 1000 + rank
        for (i, &v) in recv.iter().enumerate() {
            let expected = (i as i32 * 1000 + rank) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: alltoallv_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: alltoallv_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 15: reduce_scatter_block_init
    // ========================================================================
    {
        let block_size = 3usize;
        let sz = size as usize;
        // Each rank contributes [1.0; block_size * size]
        let send = vec![1.0f64; block_size * sz];
        let mut recv = vec![0.0f64; block_size];

        let mut req = world
            .reduce_scatter_block_init(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block_init failed");

        req.start().expect("reduce_scatter_block_init start failed");
        req.wait().expect("reduce_scatter_block_init wait failed");

        // Each element is the sum across all ranks of 1.0 => size
        let expected = size as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: reduce_scatter_block_init recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: reduce_scatter_block_init");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 16: PersistentRequest::test() polling
    // ========================================================================
    {
        let send = vec![rank as f64; 5];
        let mut recv = vec![0.0f64; 5];
        let mut req = world
            .allreduce_init(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce_init for test() polling failed");

        // test() on inactive request should return true immediately
        let completed = req.test().expect("test() on inactive failed");
        assert!(completed, "test() on inactive request should return true");
        assert!(!req.is_active(), "should remain inactive after test()");

        // Start and poll until complete
        req.start().expect("start for test() polling failed");
        assert!(req.is_active(), "should be active after start");

        // Poll until complete (bounded loop to avoid infinite spin)
        let mut poll_count = 0u64;
        loop {
            let done = req.test().expect("test() polling failed");
            poll_count += 1;
            if done {
                break;
            }
            assert!(
                poll_count < 10_000_000,
                "test() polling exceeded 10M iterations"
            );
        }
        assert!(
            !req.is_active(),
            "should be inactive after test() returns true"
        );

        // Verify result
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: test() polling recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: PersistentRequest::test() polling");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 17: start_all + wait_all
    // ========================================================================
    {
        // Create two persistent requests: a bcast and an allreduce
        let mut bcast_data = vec![0.0f64; 5];
        if rank == 0 {
            for (i, x) in bcast_data.iter_mut().enumerate() {
                *x = (i + 1) as f64 * 10.0;
            }
        }
        let allreduce_send = vec![rank as f64; 3];
        let mut allreduce_recv = vec![0.0f64; 3];

        let req_bcast = world
            .bcast_init(&mut bcast_data, 0)
            .expect("bcast_init for start_all failed");
        let req_allreduce = world
            .allreduce_init(&allreduce_send, &mut allreduce_recv, ReduceOp::Sum)
            .expect("allreduce_init for start_all failed");

        // Use start_all and wait_all
        let mut requests = [req_bcast, req_allreduce];
        PersistentRequest::start_all(&mut requests).expect("start_all failed");

        for r in &requests {
            assert!(
                r.is_active(),
                "all requests should be active after start_all"
            );
        }

        PersistentRequest::wait_all(&mut requests).expect("wait_all failed");

        for r in &requests {
            assert!(
                !r.is_active(),
                "all requests should be inactive after wait_all"
            );
        }

        // We moved the requests into the array, so verify via the array.
        // The bcast_data and allreduce_recv buffers should still be valid
        // because the persistent requests hold references to the original buffers.
        // However, since we moved the PersistentRequests into the array,
        // the original bindings (req_bcast, req_allreduce) are consumed.
        // Verification of the bcast result:
        for (i, &x) in bcast_data.iter().enumerate() {
            assert!(
                (x - (i + 1) as f64 * 10.0).abs() < f64::EPSILON,
                "rank {rank}: start_all bcast_data[{i}] = {x}, expected {}",
                (i + 1) as f64 * 10.0
            );
        }

        // Verification of the allreduce result:
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in allreduce_recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: start_all allreduce_recv[{i}] = {v}, expected {expected}"
            );
        }

        test_count += 1;
        if rank == 0 {
            println!("PASS: start_all + wait_all");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 18: drop without start (exercises Drop when active=false)
    //
    // Creates a persistent request, does NOT start it, then drops it.
    // This exercises the `!self.active` path in Drop (just calls request_free).
    // ========================================================================
    {
        let mut data = vec![0.0f64; 5];
        let req = world
            .bcast_init(&mut data, 0)
            .expect("bcast_init for drop-without-start failed");

        assert!(!req.is_active(), "request should be inactive before start");

        // Explicitly drop without starting — exercises Drop's !active path
        drop(req);

        test_count += 1;
        if rank == 0 {
            println!("PASS: drop without start (inactive Drop path)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 19: drop after start but before wait (exercises Drop when active=true)
    //
    // Creates a persistent request, starts it, does NOT wait, then drops it.
    // This exercises the `self.active` path in Drop (calls wait then request_free).
    // The Drop impl will wait for completion, so this won't corrupt MPI state.
    // ========================================================================
    {
        let mut data = vec![0.0f64; 5];
        if rank == 0 {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (i + 1) as f64;
            }
        }
        let mut req = world
            .bcast_init(&mut data, 0)
            .expect("bcast_init for drop-while-active failed");

        req.start().expect("start for drop-while-active failed");
        assert!(req.is_active(), "request should be active after start");

        // Explicitly drop while active — exercises Drop's active path (wait + free)
        drop(req);

        // Barrier to confirm all ranks survived the drop-while-active
        world
            .barrier()
            .expect("barrier after drop-while-active failed");

        test_count += 1;
        if rank == 0 {
            println!("PASS: drop after start without wait (active Drop path)");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All persistent collective tests passed! ({test_count} tests)");
        println!("========================================");
    }
}
