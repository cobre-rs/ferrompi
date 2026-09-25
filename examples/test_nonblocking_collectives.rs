//! Integration test for ALL nonblocking collective operations.
//!
//! Exercises all 15 nonblocking collective methods in `Comm`:
//! ibroadcast, iallreduce, ireduce, igather, iallgather, iscatter,
//! ibarrier, iscan, iexscan, ialltoall, igatherv, iscatterv,
//! iallgatherv, ialltoallv, and ireduce_scatter_block.
//!
//! Each operation returns a `Request` that is `.wait()`ed before
//! verifying results with assertions matching the blocking test
//! math in `test_collectives.rs`.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_nonblocking_collectives
// mpi-test: np=2..

use ferrompi::{Mpi, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_nonblocking_collectives requires at least 2 processes, got {size}"
    );

    let mut test_count = 0u32;

    // ========================================================================
    // Test 1: ibroadcast
    // ========================================================================
    {
        let mut data = vec![0.0f64; 10];
        if rank == 0 {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (i + 1) as f64;
            }
        }
        let req = world.ibroadcast(&mut data, 0).expect("ibroadcast failed");
        req.wait().expect("ibroadcast wait failed");

        for (i, &x) in data.iter().enumerate() {
            assert!(
                (x - (i + 1) as f64).abs() < f64::EPSILON,
                "rank {rank}: ibroadcast data[{i}] = {x}, expected {}",
                (i + 1) as f64
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: ibroadcast");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 2: iallreduce (Sum)
    // ========================================================================
    {
        let send = vec![rank as f64; 5];
        let mut recv = vec![0.0f64; 5];
        let req = world
            .iallreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("iallreduce failed");
        req.wait().expect("iallreduce wait failed");

        // Sum of ranks: 0 + 1 + ... + (size-1) = size*(size-1)/2
        let expected = (size * (size - 1) / 2) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: iallreduce recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iallreduce (Sum)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 3: ireduce (Sum, to root=0)
    // ========================================================================
    {
        let send = vec![(rank + 1) as f64; 4];
        let mut recv = vec![0.0f64; 4];
        let req = world
            .ireduce(&send, &mut recv, ReduceOp::Sum, 0)
            .expect("ireduce failed");
        req.wait().expect("ireduce wait failed");

        if rank == 0 {
            // Sum of 1 + 2 + ... + size = size*(size+1)/2
            let expected = (size * (size + 1) / 2) as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "ireduce recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: ireduce (Sum)");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 4: igather
    // ========================================================================
    {
        let send = vec![rank as f64 * 10.0, rank as f64 * 10.0 + 1.0];
        let mut recv = vec![0.0f64; 2 * size as usize];
        let req = world.igather(&send, &mut recv, 0).expect("igather failed");
        req.wait().expect("igather wait failed");

        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert!(
                    (recv[idx] - r as f64 * 10.0).abs() < f64::EPSILON,
                    "igather recv[{idx}] = {}, expected {}",
                    recv[idx],
                    r as f64 * 10.0
                );
                assert!(
                    (recv[idx + 1] - (r as f64 * 10.0 + 1.0)).abs() < f64::EPSILON,
                    "igather recv[{}] = {}, expected {}",
                    idx + 1,
                    recv[idx + 1],
                    r as f64 * 10.0 + 1.0
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: igather");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 5: iallgather
    // ========================================================================
    {
        let send = vec![rank as f64; 2];
        let mut recv = vec![0.0f64; 2 * size as usize];
        let req = world
            .iallgather(&send, &mut recv)
            .expect("iallgather failed");
        req.wait().expect("iallgather wait failed");

        for r in 0..size {
            let idx = r as usize * 2;
            assert!(
                (recv[idx] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: iallgather recv[{idx}] = {}, expected {}",
                recv[idx],
                r as f64
            );
            assert!(
                (recv[idx + 1] - r as f64).abs() < f64::EPSILON,
                "rank {rank}: iallgather recv[{}] = {}, expected {}",
                idx + 1,
                recv[idx + 1],
                r as f64
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iallgather");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 6: iscatter
    // ========================================================================
    {
        let send_data = if rank == 0 {
            (0..size * 3).map(|x| x as f64).collect::<Vec<f64>>()
        } else {
            vec![0.0f64; (size * 3) as usize]
        };
        let mut recv = vec![0.0f64; 3];
        let req = world
            .iscatter(&send_data, &mut recv, 0)
            .expect("iscatter failed");
        req.wait().expect("iscatter wait failed");

        let base = rank * 3;
        for (i, &v) in recv.iter().enumerate() {
            let expected = (base + i as i32) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: iscatter recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iscatter");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 7: ibarrier
    // ========================================================================
    {
        let req = world.ibarrier().expect("ibarrier failed");
        req.wait().expect("ibarrier wait failed");

        test_count += 1;
        if rank == 0 {
            println!("PASS: ibarrier");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 8: iscan (inclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        let req = world
            .iscan(&send, &mut recv, ReduceOp::Sum)
            .expect("iscan failed");
        req.wait().expect("iscan wait failed");

        // On rank i, inclusive scan of 1.0 from all ranks 0..=i => (i+1)
        let expected = (rank + 1) as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: iscan recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iscan");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 9: iexscan (exclusive prefix sum)
    // ========================================================================
    {
        let send = vec![1.0f64; 3];
        let mut recv = vec![0.0f64; 3];
        let req = world
            .iexscan(&send, &mut recv, ReduceOp::Sum)
            .expect("iexscan failed");
        req.wait().expect("iexscan wait failed");

        // On rank i > 0, exclusive scan of 1.0 from ranks 0..i => i
        // On rank 0, result is undefined per MPI standard — skip assertion
        if rank > 0 {
            let expected = rank as f64;
            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-10,
                    "rank {rank}: iexscan recv[{i}] = {v}, expected {expected}"
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iexscan");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 10: ialltoall
    // ========================================================================
    {
        // Each rank sends its rank value to every other rank.
        // send = [rank, rank, ..., rank] (one per destination rank)
        let send = vec![rank as f64; size as usize];
        let mut recv = vec![0.0f64; size as usize];
        let req = world.ialltoall(&send, &mut recv).expect("ialltoall failed");
        req.wait().expect("ialltoall wait failed");

        // recv[i] should be i (the value that rank i sent)
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - i as f64).abs() < f64::EPSILON,
                "rank {rank}: ialltoall recv[{i}] = {v}, expected {i}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: ialltoall");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 11: igatherv (variable-count gather)
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

        let req = world
            .igatherv(&send, &mut recv, &recvcounts, &displs, 0)
            .expect("igatherv failed");
        req.wait().expect("igatherv wait failed");

        if rank == 0 {
            for r in 0..size {
                let offset = displs[r as usize] as usize;
                let count = recvcounts[r as usize] as usize;
                for j in 0..count {
                    assert!(
                        (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                        "igatherv: recv[{}] = {}, expected {}",
                        offset + j,
                        recv[offset + j],
                        r as f64
                    );
                }
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: igatherv");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 12: iscatterv (variable-count scatter)
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
        let req = world
            .iscatterv(&send, &mut recv, &sendcounts, &displs, 0)
            .expect("iscatterv failed");
        req.wait().expect("iscatterv wait failed");

        let expected = rank as f64 * 100.0;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: iscatterv recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iscatterv");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 13: iallgatherv (variable-count allgather)
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

        let req = world
            .iallgatherv(&send, &mut recv, &recvcounts, &displs)
            .expect("iallgatherv failed");
        req.wait().expect("iallgatherv wait failed");

        for r in 0..size {
            let offset = displs[r as usize] as usize;
            let count = recvcounts[r as usize] as usize;
            for j in 0..count {
                assert!(
                    (recv[offset + j] - r as f64).abs() < f64::EPSILON,
                    "rank {rank}: iallgatherv recv[{}] = {}, expected {}",
                    offset + j,
                    recv[offset + j],
                    r as f64
                );
            }
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: iallgatherv");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 14: ialltoallv (variable-count all-to-all)
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

        let req = world
            .ialltoallv(
                &send,
                &mut recv,
                &sendcounts,
                &sdispls,
                &recvcounts,
                &rdispls,
            )
            .expect("ialltoallv failed");
        req.wait().expect("ialltoallv wait failed");

        // recv[i] should be the value that rank i sent to us = i * 1000 + rank
        for (i, &v) in recv.iter().enumerate() {
            let expected = (i as i32 * 1000 + rank) as f64;
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: ialltoallv recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: ialltoallv");
        }
    }

    world.barrier().expect("barrier failed");

    // ========================================================================
    // Test 15: ireduce_scatter_block
    // ========================================================================
    {
        let block_size = 3usize;
        let sz = size as usize;
        // Each rank contributes [1.0; block_size * size]
        let send = vec![1.0f64; block_size * sz];
        let mut recv = vec![0.0f64; block_size];

        let req = world
            .ireduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("ireduce_scatter_block failed");
        req.wait().expect("ireduce_scatter_block wait failed");

        // Each element is the sum across all ranks of 1.0 => size
        let expected = size as f64;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-10,
                "rank {rank}: ireduce_scatter_block recv[{i}] = {v}, expected {expected}"
            );
        }
        test_count += 1;
        if rank == 0 {
            println!("PASS: ireduce_scatter_block");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All nonblocking collective tests passed! ({test_count} tests)");
        println!("========================================");
    }
}
