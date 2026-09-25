//! Integration test for communicator splitting operations.
//!
//! Exercises Communicator::split() to create sub-communicators by color,
//! verifies rank/size in sub-communicators, and tests collective operations
//! within sub-communicators. Also tests split_type and split_shared.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_comm_split
// mpi-test: np=4

use ferrompi::{Communicator, Mpi, ReduceOp, SplitType};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 4,
        "test_comm_split requires at least 4 processes, got {size}"
    );

    // ========================================================================
    // Test 1: Even/odd split
    // ========================================================================
    {
        let color = rank % 2;
        let sub = world
            .split(color, rank)
            .expect("split failed")
            .expect("split returned None for valid color");

        let sub_rank = sub.rank();
        let sub_size = sub.size();

        // Expected: even ranks form one group, odd ranks form another
        let expected_size = if color == 0 {
            (size + 1) / 2 // number of even ranks
        } else {
            size / 2 // number of odd ranks
        };
        assert_eq!(
            sub_size, expected_size,
            "rank {rank}: even/odd split sub_size = {sub_size}, expected {expected_size}"
        );

        // In the sub-communicator, ranks should be 0..sub_size-1
        assert!(
            sub_rank >= 0 && sub_rank < sub_size,
            "rank {rank}: sub_rank {sub_rank} out of range [0, {sub_size})"
        );

        // Verify: the sub_rank should be rank / 2 (since key=rank preserves ordering)
        let expected_sub_rank = rank / 2;
        assert_eq!(
            sub_rank, expected_sub_rank,
            "rank {rank}: sub_rank = {sub_rank}, expected {expected_sub_rank}"
        );

        if rank == 0 {
            println!("PASS: even/odd split (rank/size verification)");
        }
    }

    world.barrier().expect("barrier 1 failed");

    // ========================================================================
    // Test 2: Allreduce within sub-communicator
    // ========================================================================
    {
        let color = rank % 2;
        let sub = world
            .split(color, rank)
            .expect("split failed")
            .expect("split returned None");

        // Each rank contributes its world rank; sum within sub-communicator
        let sub_sum = sub
            .allreduce_scalar(rank as f64, ReduceOp::Sum)
            .expect("sub allreduce failed");

        // Expected: sum of even ranks or sum of odd ranks
        let expected: f64 = (0..size)
            .filter(|&r| r % 2 == color)
            .map(|r| r as f64)
            .sum();

        assert!(
            (sub_sum - expected).abs() < 1e-10,
            "rank {rank}: sub allreduce sum = {sub_sum}, expected {expected}"
        );

        if rank == 0 {
            println!("PASS: allreduce within sub-communicator");
        }
    }

    world.barrier().expect("barrier 2 failed");

    // ========================================================================
    // Test 3: Three-way split (color = rank % 3)
    // ========================================================================
    {
        let color = rank % 3;
        let sub = world
            .split(color, rank)
            .expect("split mod 3 failed")
            .expect("split returned None");

        let sub_size = sub.size();
        let expected_count = (0..size).filter(|&r| r % 3 == color).count() as i32;
        assert_eq!(
            sub_size, expected_count,
            "rank {rank}: mod-3 split sub_size = {sub_size}, expected {expected_count}"
        );

        // Gather world ranks within sub-communicator to verify membership
        let send = vec![rank as f64];
        let mut recv = vec![0.0f64; sub_size as usize];
        sub.allgather(&send, &mut recv)
            .expect("sub allgather failed");

        // All gathered ranks should have the same color
        for (i, &r) in recv.iter().enumerate() {
            let gathered_rank = r as i32;
            assert_eq!(
                gathered_rank % 3,
                color,
                "rank {rank}: mod-3 split recv[{i}] = {gathered_rank}, wrong color"
            );
        }

        if rank == 0 {
            println!("PASS: three-way split");
        }
    }

    world.barrier().expect("barrier 3 failed");

    // ========================================================================
    // Test 4: Split with UNDEFINED color (opt-out)
    // ========================================================================
    {
        // Rank 0 opts out of the split
        let color = if rank == 0 {
            Communicator::UNDEFINED
        } else {
            0 // all other ranks in the same group
        };
        let result = world
            .split(color, rank)
            .expect("split with UNDEFINED failed");

        if rank == 0 {
            assert!(
                result.is_none(),
                "rank 0: split with UNDEFINED should return None"
            );
            println!("PASS: split with UNDEFINED color");
        } else {
            let sub = result.expect("non-UNDEFINED ranks should get a communicator");
            assert_eq!(
                sub.size(),
                size - 1,
                "rank {rank}: UNDEFINED split sub_size = {}, expected {}",
                sub.size(),
                size - 1
            );
        }
    }

    world.barrier().expect("barrier 4 failed");

    // ========================================================================
    // Test 5: Broadcast within sub-communicator
    // ========================================================================
    {
        let color = rank % 2;
        let sub = world
            .split(color, rank)
            .expect("split failed")
            .expect("split returned None");

        let mut data = vec![0.0f64; 5];
        if sub.rank() == 0 {
            data = vec![color as f64 * 100.0; 5];
        }
        sub.broadcast(&mut data, 0).expect("sub broadcast failed");

        let expected = color as f64 * 100.0;
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - expected).abs() < f64::EPSILON,
                "rank {rank}: sub broadcast data[{i}] = {v}, expected {expected}"
            );
        }

        if rank == 0 {
            println!("PASS: broadcast within sub-communicator");
        }
    }

    world.barrier().expect("barrier 5 failed");

    // ========================================================================
    // Test 6: split_type (Shared memory split)
    // ========================================================================
    {
        let result = world
            .split_type(SplitType::Shared, rank)
            .expect("split_type failed");

        // On a single node, all ranks should be in the same shared-memory comm
        let node = result.expect("split_type Shared should return a communicator");
        let node_rank = node.rank();
        let node_size = node.size();

        assert!(
            node_rank >= 0 && node_rank < node_size,
            "rank {rank}: node_rank {node_rank} out of range [0, {node_size})"
        );
        // node_size should be >= 1 and <= size
        assert!(
            node_size >= 1 && node_size <= size,
            "rank {rank}: node_size {node_size} out of range [1, {size}]"
        );

        if rank == 0 {
            println!("PASS: split_type (Shared), node_size = {node_size}");
        }
    }

    world.barrier().expect("barrier 6 failed");

    // ========================================================================
    // Test 7: split_shared convenience method
    // ========================================================================
    {
        let node = world.split_shared().expect("split_shared failed");
        let node_size = node.size();
        let node_rank = node.rank();

        assert!(
            node_rank >= 0 && node_rank < node_size,
            "rank {rank}: split_shared node_rank out of range"
        );

        // Verify the node communicator works by doing an allreduce
        let sum = node
            .allreduce_scalar(1.0f64, ReduceOp::Sum)
            .expect("node allreduce failed");
        assert!(
            (sum - node_size as f64).abs() < 1e-10,
            "rank {rank}: split_shared allreduce sum = {sum}, expected {node_size}"
        );

        if rank == 0 {
            println!("PASS: split_shared");
        }
    }

    world.barrier().expect("barrier 7 failed");

    // ========================================================================
    // Test 8: duplicate communicator
    // ========================================================================
    {
        let dup = world.duplicate().expect("duplicate failed");
        assert_eq!(
            dup.rank(),
            rank,
            "rank {rank}: duplicated comm rank mismatch"
        );
        assert_eq!(
            dup.size(),
            size,
            "rank {rank}: duplicated comm size mismatch"
        );

        // Verify the duplicate works for communication
        let sum = dup
            .allreduce_scalar(1.0f64, ReduceOp::Sum)
            .expect("dup allreduce failed");
        assert!(
            (sum - size as f64).abs() < 1e-10,
            "rank {rank}: dup allreduce sum = {sum}, expected {size}"
        );

        if rank == 0 {
            println!("PASS: duplicate communicator");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All comm split tests passed! (8 tests)");
        println!("========================================");
    }
}
