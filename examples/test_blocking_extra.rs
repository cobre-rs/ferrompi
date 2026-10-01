//! Multi-type blocking collective tests for coverage.
//!
//! Exercises key blocking collective operations with multiple data types
//! (i32, f32, u8, u32, u64, i64) to cover generic monomorphizations
//! beyond the f64 tests in test_collectives.rs.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_blocking_extra
// mpi-test: np=2.. valgrind

use ferrompi::{Mpi, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_blocking_extra requires at least 2 processes, got {size}"
    );

    let mut test_count = 0u32;

    // ========================================================================
    // i32 tests
    // ========================================================================

    // --- i32: broadcast ---
    // (already tested in test_collectives, but we include it for completeness
    //  of this self-contained multi-type suite)
    {
        let mut data = vec![0i32; 4];
        if rank == 0 {
            data = vec![10, 20, 30, 40];
        }
        world.broadcast(&mut data, 0).expect("broadcast i32 failed");
        assert_eq!(
            data,
            vec![10, 20, 30, 40],
            "rank {rank}: broadcast i32 mismatch"
        );
        if rank == 0 {
            println!("PASS: i32 broadcast");
        }
        test_count += 1;
    }

    // --- i32: allreduce ---
    {
        let send = vec![rank; 3];
        let mut recv = vec![0i32; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce i32 failed");
        let expected = size * (size - 1) / 2;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: allreduce i32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i32 allreduce (Sum)");
        }
        test_count += 1;
    }

    // --- i32: allreduce_scalar ---
    {
        let result = world
            .allreduce_scalar(rank, ReduceOp::Sum)
            .expect("allreduce_scalar i32 failed");
        let expected = size * (size - 1) / 2;
        assert_eq!(
            result, expected,
            "rank {rank}: allreduce_scalar i32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: i32 allreduce_scalar");
        }
        test_count += 1;
    }

    // --- i32: gather ---
    {
        let send = vec![rank * 10, rank * 10 + 1];
        let mut recv = if rank == 0 {
            vec![0i32; 2 * size as usize]
        } else {
            vec![]
        };
        world
            .gather(&send, &mut recv, 0)
            .expect("gather i32 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(
                    recv[idx],
                    r * 10,
                    "gather i32 recv[{idx}] = {}, expected {}",
                    recv[idx],
                    r * 10
                );
                assert_eq!(
                    recv[idx + 1],
                    r * 10 + 1,
                    "gather i32 recv[{}] = {}, expected {}",
                    idx + 1,
                    recv[idx + 1],
                    r * 10 + 1
                );
            }
            println!("PASS: i32 gather");
        }
        test_count += 1;
    }

    // --- i32: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).collect::<Vec<i32>>()
        } else {
            vec![]
        };
        let mut recv = vec![0i32; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter i32 failed");
        let base = rank * 2;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as i32;
            assert_eq!(
                v, expected,
                "rank {rank}: scatter i32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i32 scatter");
        }
        test_count += 1;
    }

    // --- i32: alltoall ---
    {
        let send = vec![rank; size as usize];
        let mut recv = vec![0i32; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall i32 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, i as i32,
                "rank {rank}: alltoall i32 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: i32 alltoall");
        }
        test_count += 1;
    }

    // --- i32: allgather ---
    {
        let send = vec![rank; 2];
        let mut recv = vec![0i32; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather i32 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert_eq!(
                recv[idx], r,
                "rank {rank}: allgather i32 recv[{idx}] mismatch"
            );
            assert_eq!(
                recv[idx + 1],
                r,
                "rank {rank}: allgather i32 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: i32 allgather");
        }
        test_count += 1;
    }

    // --- i32: reduce_scatter_block ---
    {
        let block_size = 2usize;
        let send = vec![1i32; block_size * size as usize];
        let mut recv = vec![0i32; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block i32 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, size,
                "rank {rank}: reduce_scatter_block i32 recv[{i}] = {v}, expected {size}"
            );
        }
        if rank == 0 {
            println!("PASS: i32 reduce_scatter_block");
        }
        test_count += 1;
    }

    // --- i32: scan_scalar ---
    {
        let result = world
            .scan_scalar(1i32, ReduceOp::Sum)
            .expect("scan_scalar i32 failed");
        let expected = rank + 1;
        assert_eq!(
            result, expected,
            "rank {rank}: scan_scalar i32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: i32 scan_scalar");
        }
        test_count += 1;
    }

    // --- i32: reduce_scalar ---
    {
        let result = world
            .reduce_scalar(rank, ReduceOp::Sum, 0)
            .expect("reduce_scalar i32 failed");
        if rank == 0 {
            let expected = size * (size - 1) / 2;
            assert_eq!(
                result, expected,
                "reduce_scalar i32: got {result}, expected {expected}"
            );
            println!("PASS: i32 reduce_scalar");
        }
        test_count += 1;
    }

    // --- i32: exscan_scalar ---
    {
        let result = world
            .exscan_scalar(1i32, ReduceOp::Sum)
            .expect("exscan_scalar i32 failed");
        if rank > 0 {
            let expected = rank;
            assert_eq!(
                result, expected,
                "rank {rank}: exscan_scalar i32 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i32 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // f32 tests
    // ========================================================================

    // --- f32: broadcast ---
    {
        let mut data = vec![0.0f32; 4];
        if rank == 0 {
            data = vec![1.5, 2.5, 3.5, 4.5];
        }
        world.broadcast(&mut data, 0).expect("broadcast f32 failed");
        assert_eq!(
            data,
            vec![1.5f32, 2.5, 3.5, 4.5],
            "rank {rank}: broadcast f32 mismatch"
        );
        if rank == 0 {
            println!("PASS: f32 broadcast");
        }
        test_count += 1;
    }

    // --- f32: allreduce ---
    {
        let send = vec![rank as f32; 3];
        let mut recv = vec![0.0f32; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce f32 failed");
        let expected = (size * (size - 1) / 2) as f32;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "rank {rank}: allreduce f32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: f32 allreduce (Sum)");
        }
        test_count += 1;
    }

    // --- f32: allreduce_scalar ---
    {
        let result = world
            .allreduce_scalar(rank as f32, ReduceOp::Sum)
            .expect("allreduce_scalar f32 failed");
        let expected = (size * (size - 1) / 2) as f32;
        assert!(
            (result - expected).abs() < 1e-5,
            "rank {rank}: allreduce_scalar f32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: f32 allreduce_scalar");
        }
        test_count += 1;
    }

    // --- f32: gather ---
    {
        let send = vec![rank as f32 * 10.0, rank as f32 * 10.0 + 1.0];
        let mut recv = if rank == 0 {
            vec![0.0f32; 2 * size as usize]
        } else {
            vec![]
        };
        world
            .gather(&send, &mut recv, 0)
            .expect("gather f32 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                let expected0 = r as f32 * 10.0;
                let expected1 = r as f32 * 10.0 + 1.0;
                assert!(
                    (recv[idx] - expected0).abs() < f32::EPSILON,
                    "gather f32 recv[{idx}] = {}, expected {expected0}",
                    recv[idx]
                );
                assert!(
                    (recv[idx + 1] - expected1).abs() < f32::EPSILON,
                    "gather f32 recv[{}] = {}, expected {expected1}",
                    idx + 1,
                    recv[idx + 1]
                );
            }
            println!("PASS: f32 gather");
        }
        test_count += 1;
    }

    // --- f32: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).map(|x| x as f32).collect::<Vec<f32>>()
        } else {
            vec![]
        };
        let mut recv = vec![0.0f32; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter f32 failed");
        let base = (rank * 2) as f32;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as f32;
            assert!(
                (v - expected).abs() < f32::EPSILON,
                "rank {rank}: scatter f32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: f32 scatter");
        }
        test_count += 1;
    }

    // --- f32: alltoall ---
    {
        let send = vec![rank as f32; size as usize];
        let mut recv = vec![0.0f32; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall f32 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - i as f32).abs() < f32::EPSILON,
                "rank {rank}: alltoall f32 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: f32 alltoall");
        }
        test_count += 1;
    }

    // --- f32: allgather ---
    {
        let send = vec![rank as f32; 2];
        let mut recv = vec![0.0f32; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather f32 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert!(
                (recv[idx] - r as f32).abs() < f32::EPSILON,
                "rank {rank}: allgather f32 recv[{idx}] mismatch"
            );
            assert!(
                (recv[idx + 1] - r as f32).abs() < f32::EPSILON,
                "rank {rank}: allgather f32 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: f32 allgather");
        }
        test_count += 1;
    }

    // --- f32: reduce_scatter_block ---
    {
        let block_size = 2usize;
        let send = vec![1.0f32; block_size * size as usize];
        let mut recv = vec![0.0f32; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block f32 failed");
        let expected = size as f32;
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "rank {rank}: reduce_scatter_block f32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: f32 reduce_scatter_block");
        }
        test_count += 1;
    }

    // --- f32: scan_scalar ---
    {
        let result = world
            .scan_scalar(1.0f32, ReduceOp::Sum)
            .expect("scan_scalar f32 failed");
        let expected = (rank + 1) as f32;
        assert!(
            (result - expected).abs() < 1e-5,
            "rank {rank}: scan_scalar f32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: f32 scan_scalar");
        }
        test_count += 1;
    }

    // --- f32: reduce_scalar ---
    {
        let result = world
            .reduce_scalar(rank as f32, ReduceOp::Sum, 0)
            .expect("reduce_scalar f32 failed");
        if rank == 0 {
            let expected = (size * (size - 1) / 2) as f32;
            assert!(
                (result - expected).abs() < 1e-5,
                "reduce_scalar f32: got {result}, expected {expected}"
            );
            println!("PASS: f32 reduce_scalar");
        }
        test_count += 1;
    }

    // --- f32: exscan_scalar ---
    {
        let result = world
            .exscan_scalar(1.0f32, ReduceOp::Sum)
            .expect("exscan_scalar f32 failed");
        if rank > 0 {
            let expected = rank as f32;
            assert!(
                (result - expected).abs() < 1e-5,
                "rank {rank}: exscan_scalar f32 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: f32 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // u8 tests (use small values to avoid overflow with Sum)
    // ========================================================================

    // --- u8: broadcast ---
    {
        let mut data = vec![0u8; 4];
        if rank == 0 {
            data = vec![10, 20, 30, 40];
        }
        world.broadcast(&mut data, 0).expect("broadcast u8 failed");
        assert_eq!(
            data,
            vec![10u8, 20, 30, 40],
            "rank {rank}: broadcast u8 mismatch"
        );
        if rank == 0 {
            println!("PASS: u8 broadcast");
        }
        test_count += 1;
    }

    // --- u8: allreduce (Max, to avoid overflow) ---
    {
        let send = vec![rank as u8; 3];
        let mut recv = vec![0u8; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Max)
            .expect("allreduce u8 Max failed");
        let expected = (size - 1) as u8;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: allreduce u8 Max recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u8 allreduce (Max)");
        }
        test_count += 1;
    }

    // --- u8: allreduce_scalar (Min) ---
    {
        let result = world
            .allreduce_scalar(rank as u8, ReduceOp::Min)
            .expect("allreduce_scalar u8 Min failed");
        assert_eq!(
            result, 0u8,
            "rank {rank}: allreduce_scalar u8 Min = {result}, expected 0"
        );
        if rank == 0 {
            println!("PASS: u8 allreduce_scalar (Min)");
        }
        test_count += 1;
    }

    // --- u8: gather ---
    {
        let send = vec![rank as u8, rank as u8 + 100];
        let mut recv = if rank == 0 {
            vec![0u8; 2 * size as usize]
        } else {
            vec![]
        };
        world.gather(&send, &mut recv, 0).expect("gather u8 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(
                    recv[idx], r as u8,
                    "gather u8 recv[{idx}] = {}, expected {}",
                    recv[idx], r
                );
                assert_eq!(
                    recv[idx + 1],
                    r as u8 + 100,
                    "gather u8 recv[{}] = {}, expected {}",
                    idx + 1,
                    recv[idx + 1],
                    r as u8 + 100
                );
            }
            println!("PASS: u8 gather");
        }
        test_count += 1;
    }

    // --- u8: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).map(|x| x as u8).collect::<Vec<u8>>()
        } else {
            vec![]
        };
        let mut recv = vec![0u8; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter u8 failed");
        let base = (rank * 2) as u8;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as u8;
            assert_eq!(
                v, expected,
                "rank {rank}: scatter u8 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u8 scatter");
        }
        test_count += 1;
    }

    // --- u8: alltoall ---
    {
        let send = vec![rank as u8; size as usize];
        let mut recv = vec![0u8; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall u8 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, i as u8,
                "rank {rank}: alltoall u8 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: u8 alltoall");
        }
        test_count += 1;
    }

    // --- u8: allgather ---
    {
        let send = vec![rank as u8; 2];
        let mut recv = vec![0u8; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather u8 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert_eq!(
                recv[idx], r as u8,
                "rank {rank}: allgather u8 recv[{idx}] mismatch"
            );
            assert_eq!(
                recv[idx + 1],
                r as u8,
                "rank {rank}: allgather u8 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: u8 allgather");
        }
        test_count += 1;
    }

    // --- u8: reduce_scatter_block (Max) ---
    {
        let block_size = 2usize;
        // Each rank sends its rank as u8 into each block element
        let send = vec![rank as u8; block_size * size as usize];
        let mut recv = vec![0u8; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Max)
            .expect("reduce_scatter_block u8 failed");
        let expected = (size - 1) as u8;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: reduce_scatter_block u8 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u8 reduce_scatter_block (Max)");
        }
        test_count += 1;
    }

    // --- u8: scan_scalar (Sum with value 1, safe for small process counts) ---
    {
        let result = world
            .scan_scalar(1u8, ReduceOp::Sum)
            .expect("scan_scalar u8 failed");
        let expected = (rank + 1) as u8;
        assert_eq!(
            result, expected,
            "rank {rank}: scan_scalar u8 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: u8 scan_scalar");
        }
        test_count += 1;
    }

    // --- u8: reduce_scalar (Max) ---
    {
        let result = world
            .reduce_scalar(rank as u8, ReduceOp::Max, 0)
            .expect("reduce_scalar u8 failed");
        if rank == 0 {
            let expected = (size - 1) as u8;
            assert_eq!(
                result, expected,
                "reduce_scalar u8 Max: got {result}, expected {expected}"
            );
            println!("PASS: u8 reduce_scalar (Max)");
        }
        test_count += 1;
    }

    // --- u8: exscan_scalar (Sum with value 1) ---
    {
        let result = world
            .exscan_scalar(1u8, ReduceOp::Sum)
            .expect("exscan_scalar u8 failed");
        if rank > 0 {
            let expected = rank as u8;
            assert_eq!(
                result, expected,
                "rank {rank}: exscan_scalar u8 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u8 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // u32 tests
    // ========================================================================

    // --- u32: broadcast ---
    {
        let mut data = vec![0u32; 4];
        if rank == 0 {
            data = vec![100, 200, 300, 400];
        }
        world.broadcast(&mut data, 0).expect("broadcast u32 failed");
        assert_eq!(
            data,
            vec![100u32, 200, 300, 400],
            "rank {rank}: broadcast u32 mismatch"
        );
        if rank == 0 {
            println!("PASS: u32 broadcast");
        }
        test_count += 1;
    }

    // --- u32: allreduce ---
    {
        let send = vec![rank as u32; 3];
        let mut recv = vec![0u32; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce u32 failed");
        let expected = (size * (size - 1) / 2) as u32;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: allreduce u32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u32 allreduce (Sum)");
        }
        test_count += 1;
    }

    // --- u32: allreduce_scalar ---
    {
        let result = world
            .allreduce_scalar(rank as u32, ReduceOp::Sum)
            .expect("allreduce_scalar u32 failed");
        let expected = (size * (size - 1) / 2) as u32;
        assert_eq!(
            result, expected,
            "rank {rank}: allreduce_scalar u32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: u32 allreduce_scalar");
        }
        test_count += 1;
    }

    // --- u32: gather ---
    {
        let send = vec![rank as u32 * 10, rank as u32 * 10 + 1];
        let mut recv = if rank == 0 {
            vec![0u32; 2 * size as usize]
        } else {
            vec![]
        };
        world
            .gather(&send, &mut recv, 0)
            .expect("gather u32 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(recv[idx], r as u32 * 10, "gather u32 recv[{idx}] mismatch");
                assert_eq!(
                    recv[idx + 1],
                    r as u32 * 10 + 1,
                    "gather u32 recv[{}] mismatch",
                    idx + 1
                );
            }
            println!("PASS: u32 gather");
        }
        test_count += 1;
    }

    // --- u32: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).map(|x| x as u32).collect::<Vec<u32>>()
        } else {
            vec![]
        };
        let mut recv = vec![0u32; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter u32 failed");
        let base = (rank * 2) as u32;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as u32;
            assert_eq!(
                v, expected,
                "rank {rank}: scatter u32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u32 scatter");
        }
        test_count += 1;
    }

    // --- u32: alltoall ---
    {
        let send = vec![rank as u32; size as usize];
        let mut recv = vec![0u32; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall u32 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, i as u32,
                "rank {rank}: alltoall u32 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: u32 alltoall");
        }
        test_count += 1;
    }

    // --- u32: allgather ---
    {
        let send = vec![rank as u32; 2];
        let mut recv = vec![0u32; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather u32 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert_eq!(
                recv[idx], r as u32,
                "rank {rank}: allgather u32 recv[{idx}] mismatch"
            );
            assert_eq!(
                recv[idx + 1],
                r as u32,
                "rank {rank}: allgather u32 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: u32 allgather");
        }
        test_count += 1;
    }

    // --- u32: reduce_scatter_block ---
    {
        let block_size = 2usize;
        let send = vec![1u32; block_size * size as usize];
        let mut recv = vec![0u32; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block u32 failed");
        let expected = size as u32;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: reduce_scatter_block u32 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u32 reduce_scatter_block");
        }
        test_count += 1;
    }

    // --- u32: scan_scalar ---
    {
        let result = world
            .scan_scalar(1u32, ReduceOp::Sum)
            .expect("scan_scalar u32 failed");
        let expected = (rank + 1) as u32;
        assert_eq!(
            result, expected,
            "rank {rank}: scan_scalar u32 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: u32 scan_scalar");
        }
        test_count += 1;
    }

    // --- u32: reduce_scalar ---
    {
        let result = world
            .reduce_scalar(rank as u32, ReduceOp::Sum, 0)
            .expect("reduce_scalar u32 failed");
        if rank == 0 {
            let expected = (size * (size - 1) / 2) as u32;
            assert_eq!(
                result, expected,
                "reduce_scalar u32: got {result}, expected {expected}"
            );
            println!("PASS: u32 reduce_scalar");
        }
        test_count += 1;
    }

    // --- u32: exscan_scalar ---
    {
        let result = world
            .exscan_scalar(1u32, ReduceOp::Sum)
            .expect("exscan_scalar u32 failed");
        if rank > 0 {
            let expected = rank as u32;
            assert_eq!(
                result, expected,
                "rank {rank}: exscan_scalar u32 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u32 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // u64 tests
    // ========================================================================

    // --- u64: broadcast ---
    {
        let mut data = vec![0u64; 4];
        if rank == 0 {
            data = vec![1000, 2000, 3000, 4000];
        }
        world.broadcast(&mut data, 0).expect("broadcast u64 failed");
        assert_eq!(
            data,
            vec![1000u64, 2000, 3000, 4000],
            "rank {rank}: broadcast u64 mismatch"
        );
        if rank == 0 {
            println!("PASS: u64 broadcast");
        }
        test_count += 1;
    }

    // --- u64: allreduce ---
    {
        let send = vec![rank as u64; 3];
        let mut recv = vec![0u64; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce u64 failed");
        let expected = (size * (size - 1) / 2) as u64;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: allreduce u64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u64 allreduce (Sum)");
        }
        test_count += 1;
    }

    // --- u64: allreduce_scalar ---
    {
        let result = world
            .allreduce_scalar(rank as u64, ReduceOp::Sum)
            .expect("allreduce_scalar u64 failed");
        let expected = (size * (size - 1) / 2) as u64;
        assert_eq!(
            result, expected,
            "rank {rank}: allreduce_scalar u64 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: u64 allreduce_scalar");
        }
        test_count += 1;
    }

    // --- u64: gather ---
    {
        let send = vec![rank as u64 * 10, rank as u64 * 10 + 1];
        let mut recv = if rank == 0 {
            vec![0u64; 2 * size as usize]
        } else {
            vec![]
        };
        world
            .gather(&send, &mut recv, 0)
            .expect("gather u64 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(recv[idx], r as u64 * 10, "gather u64 recv[{idx}] mismatch");
                assert_eq!(
                    recv[idx + 1],
                    r as u64 * 10 + 1,
                    "gather u64 recv[{}] mismatch",
                    idx + 1
                );
            }
            println!("PASS: u64 gather");
        }
        test_count += 1;
    }

    // --- u64: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).map(|x| x as u64).collect::<Vec<u64>>()
        } else {
            vec![]
        };
        let mut recv = vec![0u64; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter u64 failed");
        let base = (rank * 2) as u64;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as u64;
            assert_eq!(
                v, expected,
                "rank {rank}: scatter u64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u64 scatter");
        }
        test_count += 1;
    }

    // --- u64: alltoall ---
    {
        let send = vec![rank as u64; size as usize];
        let mut recv = vec![0u64; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall u64 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, i as u64,
                "rank {rank}: alltoall u64 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: u64 alltoall");
        }
        test_count += 1;
    }

    // --- u64: allgather ---
    {
        let send = vec![rank as u64; 2];
        let mut recv = vec![0u64; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather u64 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert_eq!(
                recv[idx], r as u64,
                "rank {rank}: allgather u64 recv[{idx}] mismatch"
            );
            assert_eq!(
                recv[idx + 1],
                r as u64,
                "rank {rank}: allgather u64 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: u64 allgather");
        }
        test_count += 1;
    }

    // --- u64: reduce_scatter_block ---
    {
        let block_size = 2usize;
        let send = vec![1u64; block_size * size as usize];
        let mut recv = vec![0u64; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block u64 failed");
        let expected = size as u64;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: reduce_scatter_block u64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u64 reduce_scatter_block");
        }
        test_count += 1;
    }

    // --- u64: scan_scalar ---
    {
        let result = world
            .scan_scalar(1u64, ReduceOp::Sum)
            .expect("scan_scalar u64 failed");
        let expected = (rank + 1) as u64;
        assert_eq!(
            result, expected,
            "rank {rank}: scan_scalar u64 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: u64 scan_scalar");
        }
        test_count += 1;
    }

    // --- u64: reduce_scalar ---
    {
        let result = world
            .reduce_scalar(rank as u64, ReduceOp::Sum, 0)
            .expect("reduce_scalar u64 failed");
        if rank == 0 {
            let expected = (size * (size - 1) / 2) as u64;
            assert_eq!(
                result, expected,
                "reduce_scalar u64: got {result}, expected {expected}"
            );
            println!("PASS: u64 reduce_scalar");
        }
        test_count += 1;
    }

    // --- u64: exscan_scalar ---
    {
        let result = world
            .exscan_scalar(1u64, ReduceOp::Sum)
            .expect("exscan_scalar u64 failed");
        if rank > 0 {
            let expected = rank as u64;
            assert_eq!(
                result, expected,
                "rank {rank}: exscan_scalar u64 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: u64 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // i64 tests
    // ========================================================================

    // --- i64: broadcast ---
    {
        let mut data = vec![0i64; 4];
        if rank == 0 {
            data = vec![-10, 20, -30, 40];
        }
        world.broadcast(&mut data, 0).expect("broadcast i64 failed");
        assert_eq!(
            data,
            vec![-10i64, 20, -30, 40],
            "rank {rank}: broadcast i64 mismatch"
        );
        if rank == 0 {
            println!("PASS: i64 broadcast");
        }
        test_count += 1;
    }

    // --- i64: allreduce ---
    {
        let send = vec![rank as i64; 3];
        let mut recv = vec![0i64; 3];
        world
            .allreduce(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce i64 failed");
        let expected = (size * (size - 1) / 2) as i64;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: allreduce i64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i64 allreduce (Sum)");
        }
        test_count += 1;
    }

    // --- i64: allreduce_scalar ---
    {
        let result = world
            .allreduce_scalar(rank as i64, ReduceOp::Sum)
            .expect("allreduce_scalar i64 failed");
        let expected = (size * (size - 1) / 2) as i64;
        assert_eq!(
            result, expected,
            "rank {rank}: allreduce_scalar i64 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: i64 allreduce_scalar");
        }
        test_count += 1;
    }

    // --- i64: gather ---
    {
        let send = vec![rank as i64 * 10, rank as i64 * 10 + 1];
        let mut recv = if rank == 0 {
            vec![0i64; 2 * size as usize]
        } else {
            vec![]
        };
        world
            .gather(&send, &mut recv, 0)
            .expect("gather i64 failed");
        if rank == 0 {
            for r in 0..size {
                let idx = r as usize * 2;
                assert_eq!(recv[idx], r as i64 * 10, "gather i64 recv[{idx}] mismatch");
                assert_eq!(
                    recv[idx + 1],
                    r as i64 * 10 + 1,
                    "gather i64 recv[{}] mismatch",
                    idx + 1
                );
            }
            println!("PASS: i64 gather");
        }
        test_count += 1;
    }

    // --- i64: scatter ---
    {
        let send_data = if rank == 0 {
            (0..size * 2).map(|x| x as i64).collect::<Vec<i64>>()
        } else {
            vec![]
        };
        let mut recv = vec![0i64; 2];
        world
            .scatter(&send_data, &mut recv, 0)
            .expect("scatter i64 failed");
        let base = (rank * 2) as i64;
        for (i, &v) in recv.iter().enumerate() {
            let expected = base + i as i64;
            assert_eq!(
                v, expected,
                "rank {rank}: scatter i64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i64 scatter");
        }
        test_count += 1;
    }

    // --- i64: alltoall ---
    {
        let send = vec![rank as i64; size as usize];
        let mut recv = vec![0i64; size as usize];
        world
            .alltoall(&send, &mut recv)
            .expect("alltoall i64 failed");
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, i as i64,
                "rank {rank}: alltoall i64 recv[{i}] = {v}, expected {i}"
            );
        }
        if rank == 0 {
            println!("PASS: i64 alltoall");
        }
        test_count += 1;
    }

    // --- i64: allgather ---
    {
        let send = vec![rank as i64; 2];
        let mut recv = vec![0i64; 2 * size as usize];
        world
            .allgather(&send, &mut recv)
            .expect("allgather i64 failed");
        for r in 0..size {
            let idx = r as usize * 2;
            assert_eq!(
                recv[idx], r as i64,
                "rank {rank}: allgather i64 recv[{idx}] mismatch"
            );
            assert_eq!(
                recv[idx + 1],
                r as i64,
                "rank {rank}: allgather i64 recv[{}] mismatch",
                idx + 1
            );
        }
        if rank == 0 {
            println!("PASS: i64 allgather");
        }
        test_count += 1;
    }

    // --- i64: reduce_scatter_block ---
    {
        let block_size = 2usize;
        let send = vec![1i64; block_size * size as usize];
        let mut recv = vec![0i64; block_size];
        world
            .reduce_scatter_block(&send, &mut recv, ReduceOp::Sum)
            .expect("reduce_scatter_block i64 failed");
        let expected = size as i64;
        for (i, &v) in recv.iter().enumerate() {
            assert_eq!(
                v, expected,
                "rank {rank}: reduce_scatter_block i64 recv[{i}] = {v}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i64 reduce_scatter_block");
        }
        test_count += 1;
    }

    // --- i64: scan_scalar ---
    {
        let result = world
            .scan_scalar(1i64, ReduceOp::Sum)
            .expect("scan_scalar i64 failed");
        let expected = (rank + 1) as i64;
        assert_eq!(
            result, expected,
            "rank {rank}: scan_scalar i64 = {result}, expected {expected}"
        );
        if rank == 0 {
            println!("PASS: i64 scan_scalar");
        }
        test_count += 1;
    }

    // --- i64: reduce_scalar ---
    {
        let result = world
            .reduce_scalar(rank as i64, ReduceOp::Sum, 0)
            .expect("reduce_scalar i64 failed");
        if rank == 0 {
            let expected = (size * (size - 1) / 2) as i64;
            assert_eq!(
                result, expected,
                "reduce_scalar i64: got {result}, expected {expected}"
            );
            println!("PASS: i64 reduce_scalar");
        }
        test_count += 1;
    }

    // --- i64: exscan_scalar ---
    {
        let result = world
            .exscan_scalar(1i64, ReduceOp::Sum)
            .expect("exscan_scalar i64 failed");
        if rank > 0 {
            let expected = rank as i64;
            assert_eq!(
                result, expected,
                "rank {rank}: exscan_scalar i64 = {result}, expected {expected}"
            );
        }
        if rank == 0 {
            println!("PASS: i64 exscan_scalar");
        }
        test_count += 1;
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All multi-type blocking collective tests passed! ({test_count} tests)");
        println!("========================================");
    }
}
