//! Integration test for ALL blocking collective operations.
//!
//! Exercises broadcast, allreduce, gather, scatter, barrier, reduce,
//! allgather, alltoall, scan, exscan, gatherv, scatterv, allgatherv,
//! alltoallv, reduce_scatter_block, reduce_scalar, and reduce_inplace.
//!
//! Each operation is verified with meaningful assertions.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_collectives
// mpi-test: np=2.. valgrind

use ferrompi::{Mpi, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_collectives requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Test 1: broadcast
    // ========================================================================
    {
        let mut data = vec![0.0f64; 10];
        if rank == 0 {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (i + 1) as f64;
            }
        }
        world.broadcast(&mut data, 0).expect("broadcast failed");
        for (i, &x) in data.iter().enumerate() {
            assert!(
                (x - (i + 1) as f64).abs() < f64::EPSILON,
                "rank {rank}: broadcast data[{i}] = {x}, expected {}",
                (i + 1) as f64
            );
        }
        if rank == 0 {
            println!("PASS: broadcast");
        }
    }

    // ========================================================================
    // Test 2: allreduce (Sum)
    // ========================================================================
    {
        let send = vec![rank as f64; 5];
        let mut recv = vec![0.0f64; 5];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce failed");
        // Sum of ranks: 0 + 1 + ... + (size-1) = size*(size-1)/2
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: allreduce recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: allreduce (Sum)");
        }
    }

    // ========================================================================
    // Test 3: gather
    // ========================================================================
    {
        let send = vec![rank as f64 * 10.0, rank as f64 * 10.0 + 1.0];
        let mut recv = if rank == 0 {
            vec![0.0f64; 2 * size as usize]
        } else {
            vec![]
        };
        world.gather(&send, &mut recv, 0).expect("gather failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert!(
                    (recv[idx] - r as f64 * 10.0).abs() < f64::EPSILON,
                    "gather recv[{idx}] = {}, expected {}",
                    recv[idx],
                    r as f64 * 10.0
                );
                assert!(
                    (recv[idx + 1] - (r as f64 * 10.0 + 1.0)).abs() < f64::EPSILON,
                    "gather recv[{}] = {}, expected {}",
                    idx + 1,
                    recv[idx + 1],
                    r as f64 * 10.0 + 1.0
                );
            }
            println!("PASS: gather");
        }
    }

    // ========================================================================
    // Test 4: scatter
    // ========================================================================
    {
        let send_data = if rank == 0 {
            (0..size * 3).map(|x| x as f64).collect::<Vec<f64>>()
        } else {
            vec![]
        };
        let mut recv = vec![0.0f64; 3];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter failed");
        let base = rank * 3;
        for (i, &v) in recv.iter().enumerate() {
            let expected = (base + i as i32) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: scatter recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: scatter");
        }
    }

    // ========================================================================
    // Test 5: barrier
    // ========================================================================
    {
        world.barrier().expect("barrier failed");
        if rank == 0 {
            println!("PASS: barrier");
        }
    }

    // ========================================================================
    // Test 6: reduce (Sum, to root=0)
    // ========================================================================
    {
        let send = vec![(rank + 1) as f64; 4];
        let mut recv = vec![0.0f64; 4];
        world
            .reduce(&send, &mut recv, ReduceOp::Sum, 0)
            .expect("reduce failed");
        if rank == 0 {
            // Sum of 1 + 2 + ... + size = size*(size+1)/2
            let expected = (size * (size + 1) / 2) as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "reduce recv[{i}] = {v}, expected {expected}"
                );
            }
            println!("PASS: reduce (Sum)");
        }
    }

    // ========================================================================
    // Test 7: reduce (Max, to root=0)
    // ========================================================================
    {
        let send = vec![rank as f64];
        let mut recv = vec![0.0f64];
        world
            .reduce(&send, &mut recv, ReduceOp::Max, 0)
            .expect("reduce Max failed");
        if rank == 0 {
            let expected = (size - 1) as f64;
            assert!(
                (recv[0] - expected).abs() < f64::EPSILON,
                "reduce Max: got {}, expected {expected}",
                recv[0]
            );
            println!("PASS: reduce (Max)");
        }
    }

    // ========================================================================
    // Test 8: reduce_scalar (Sum, to root=0)
    // ========================================================================
    {
        let result = world
            .reduce_scalar(rank as f64, ReduceOp::Sum, 0)
            .expect("reduce_scalar failed");
        if rank == 0 {
            let expected = (size * (size - 1) / 2) as f64;
            assert!(
                (result - expected).abs() < 1e-10,
                "reduce_scalar: got {result}, expected {expected}"
            );
            println!("PASS: reduce_scalar");
        }
    }

    // ========================================================================
    // Test 9: reduce_inplace (Sum, to root=0)
    // ========================================================================
    {
        let mut data = vec![(rank + 1) as f64; 3];
        world
            .reduce_inplace(&mut data, ReduceOp::Sum, 0)
            .expect("reduce_inplace failed");
        if rank == 0 {
            let expected = (size * (size + 1) / 2) as f64;
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "reduce_inplace data[{i}] = {v}, expected {expected}"
                );
            }
            println!("PASS: reduce_inplace");
        }
    }

    // ========================================================================
    // Test 10: allgather
    // ========================================================================
    {
        let send = vec![rank as f64; 2];
        let mut recv = vec![0.0f64; 2 * size as usize];
        world.allgather(&send, &mut recv).expect("allgather failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert!(
                (recv[idx] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: allgather recv[{idx}] = {}, expected {}",
                recv[idx],
                r as f64
            );
            assert!(
                (recv[idx + 1] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: allgather recv[{}] = {}, expected {}",
                idx + 1,
                recv[idx + 1],
                r as f64
            );
        }
        if rank == 0 {
            println!("PASS: allgather");
        }
    }

    // ========================================================================
    // Test 11: alltoall
    // ========================================================================
    {
        // Each rank sends its rank value to every other rank.
        // send = [rank, rank, ..., rank] (one per destination rank)
        let send = vec![rank as f64; size as usize];
        let mut recv = vec![0.0f64; size as usize];
        world.alltoall(&send, &mut recv).expect("alltoall failed");
        // recv[i] should be i (the value that rank i sent)
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - i as f64).abs() < f64::EPSILON,
                "rank {rank}: alltoall recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: alltoall");
        }
    }

    // ========================================================================
    // Test 12: scan (inclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        world
            .scan(&send, &mut recv, ReduceOp::Sum)
            .expect("scan failed");
        // On rank i, inclusive scan of 1.0 from all ranks 0..=i => (i+1)
        let expected = (rank + 1) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: scan recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: scan");
        }
    }

    // ========================================================================
    // Test 13: exscan (exclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        world
            .exscan(&send, &mut recv, ReduceOp::Sum)
            .expect("exscan failed");
        // On rank i > 0, exclusive scan of 1.0 from ranks 0..i => i
        // On rank 0, result is undefined per MPI standard — skip assertion
        if rank > 0 {
            let expected = rank as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "rank {rank}: exscan recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        if rank == 0 {
            println!("PASS: exscan");
        }
    }

    // ========================================================================
    // Test 14: gatherv (variable-count gather)
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

        let mut recv = if rank == 0 {
            vec![0.0f64; total]
        } else {
            vec![]
        };

        world
            .gatherv(&send, &mut recv, &recvcounts, &displs, 0)
            .expect("gatherv failed");

        if rank == 0 {
            for r in 0..size {
                let offset = displs[r as usize] as usize;
                let count = recvcounts[r as usize] as usize;
                for j in 0..count {
                    assert!(
                        (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                        "gatherv: recv[{}] = {}, expected {}",
                        offset + j,
                        recv[offset + j],
                        r as f64
                    );
                }
            }
            println!("PASS: gatherv");
        }
    }

    // ========================================================================
    // Test 15: scatterv (variable-count scatter)
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
            vec![]
        };

        let mut recv = vec![0.0f64; recv_count];
        world
            .scatterv(&send, &sendcounts, &displs, &mut recv, 0)
            .expect("scatterv failed");

        let expected = rank as f64 * 100.0;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: scatterv recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: scatterv");
        }
    }

    // ========================================================================
    // Test 16: allgatherv (variable-count allgather)
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

        world
            .allgatherv(&send, &mut recv, &recvcounts, &displs)
            .expect("allgatherv failed");

        for r in 0..size {
            let offset = displs[r as usize] as usize;
            let count = recvcounts[r as usize] as usize;
            for j in 0..count {
                assert!(
                    (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                    "rank {rank}: allgatherv recv[{}] = {}, expected {}",
                    offset + j,
                    recv[offset + j],
                    r as f64
                );
            }
        }
        if rank == 0 {
            println!("PASS: allgatherv");
        }
    }

    // ========================================================================
    // Test 17: alltoallv (variable-count all-to-all)
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

        world
            .alltoallv(
                &send,
                &sendcounts,
                &sdispls,
                &mut recv,
                &recvcounts,
                &rdispls,
            )
            .expect("alltoallv failed");

        // recv[i] should be the value that rank i sent to us = i * 1000 + rank
        for (i, &v) in recv.iter().enumerate() {
            let expected = (i as i32 * 1000 + rank) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: alltoallv recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: alltoallv");
        }
    }

    // ========================================================================
    // Test 18: reduce_scatter_block
    // ========================================================================
    {
        let block_size = 3usize;
        let sz = size as usize;
        // Each rank contributes [1.0; block_size * size]
        let send = vec![1.0f64; block_size * sz];
        let mut recv = vec![0.0f64; block_size];

        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block failed");

        // Each element is the sum across all ranks of 1.0 => size
        let expected = size as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: reduce_scatter_block recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: reduce_scatter_block");
        }
    }

    // ========================================================================
    // Test 19: allreduce_scalar
    // ========================================================================
    {
        let result = world
            .allreduce_scalar(rank as f64, ReduceOp::Sum)
            .expect("allreduce_scalar failed");
        let expected = (size * (size - 1) / 2) as f64;
        assert!(
            (result - expected).abs() < 1e-10,
            "rank {rank}: allreduce_scalar = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: allreduce_scalar");
        }
    }

    // ========================================================================
    // Test 20: allreduce_inplace
    // ========================================================================
    {
        let mut data = vec![rank as f64; 4];
        world
            .allreduce_inplace(&mut data, ReduceOp::Sum)
            .expect("allreduce_inplace failed");
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: allreduce_inplace data[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: allreduce_inplace");
        }
    }

    // ========================================================================
    // Test 21: scan_scalar
    // ========================================================================
    {
        let result = world
            .scan_scalar(1.0f64, ReduceOp::Sum)
            .expect("scan_scalar failed");
        let expected = (rank + 1) as f64;
        assert!(
            (result - expected).abs() < 1e-10,
            "rank {rank}: scan_scalar = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: scan_scalar");
        }
    }

    // ========================================================================
    // Test 22: exscan_scalar
    // ========================================================================
    {
        let result = world
            .exscan_scalar(1.0f64, ReduceOp::Sum)
            .expect("exscan_scalar failed");
        // On rank 0, result is undefined per MPI standard — skip assertion
        if rank > 0 {
            let expected = rank as f64;
            assert!(
                (result - expected).abs() < 1e-10,
                "rank {rank}: exscan_scalar = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: exscan_scalar");
        }
    }

    // ========================================================================
    // Test 23: reduce with non-zero root
    // ========================================================================
    {
        let root = (size - 1).min(1); // Use rank 1 if available, else 0
        let send = vec![rank as f64; 2];
        let mut recv = vec![0.0f64; 2];
        world
            .reduce(&send, &mut recv, ReduceOp::Sum, root)
            .expect("reduce to non-zero root failed");
        if rank == root {
            let expected = (size * (size - 1) / 2) as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "reduce (root={root}) recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        if rank == 0 {
            println!("PASS: reduce (non-zero root)");
        }
    }

    // ========================================================================
    // Test 24: allreduce with different ops (Min, Prod)
    // ========================================================================
    {
        // Min
        let min_result = world
            .allreduce_scalar(rank as f64, ReduceOp::Min)
            .expect("allreduce Min failed");
        assert!(
            min_result.abs() < f64::EPSILON,
            "rank {rank}: allreduce Min = {min_result}, expected 0"
        );

        // Prod: product of (rank+1) for all ranks = size!
        let prod_result = world
            .allreduce_scalar((rank + 1) as f64, ReduceOp::Prod)
            .expect("allreduce Prod failed");
        let expected_prod: f64 = (1..=size).map(|r| r as f64).product();
        assert!(
            (prod_result - expected_prod).abs() < 1e-6,
            "rank {rank}: allreduce Prod = {prod_result}, expected {expected_prod}"
        );

        if rank == 0 {
            println!("PASS: allreduce (Min, Prod)");
        }
    }

    // ========================================================================
    // Test 25: broadcast with i32 type (test generic API)
    // ========================================================================
    {
        let mut data = vec![0i32; 5];
        if rank == 0 {
            data = vec![10, 20, 30, 40, 50];
        }
        world.broadcast(&mut data, 0).expect("broadcast i32 failed");
        assert_eq!(
            data,
            vec![10, 20, 30, 40, 50],
            "rank {rank}: broadcast i32 mismatch"
        );
        if rank == 0 {
            println!("PASS: broadcast (i32 generic)");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All collective tests passed! (25 tests)");
        println!("========================================");
    }
}
