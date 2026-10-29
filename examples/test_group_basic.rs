//! Integration test for MPI Group operations.
//!
//! Exercises `Communicator::group`, `Group::size`, `Group::rank`,
//! `Group::include`, and `Group::exclude`, and checks that an empty group
//! takes no slot of the group table. All assertions are guarded by
//! a sentinel allreduce(Min) before any `process::exit` so that no rank
//! exits while others are still inside MPI collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_basic
// mpi-test: np=4

use ferrompi::{Mpi, RankRange, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_group_basic requires exactly 4 processes, got {size}"
    );

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Test 1: world.group().size() == world.size()
    // ========================================================================
    {
        let world_group = match world.group() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1: world.group() failed: {e}");
                // Participate in the sentinel allreduce so other ranks are not
                // left hanging, then exit.
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        };

        match world_group.size() {
            Ok(s) if s == size => {
                if rank == 0 {
                    println!("PASS: Test 1 — world_group.size() = {s}");
                }
            }
            Ok(s) => {
                eprintln!("rank {rank}: FAIL Test 1: world_group.size() = {s}, expected {size}");
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 1: world_group.size() failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 2: world.group().rank() == world.rank()
    // ========================================================================
    {
        let world_group = world.group().expect("world.group() failed in Test 2");
        match world_group.rank() {
            Ok(Some(r)) if r == rank => {
                if rank == 0 {
                    println!(
                        "PASS: Test 2 — world_group.rank() matches world.rank() for all ranks"
                    );
                }
            }
            Ok(r) => {
                eprintln!("rank {rank}: FAIL Test 2: world_group.rank() = {r:?}, expected {rank}");
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 2: world_group.rank() failed: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 3: include(&[0, 2]).size() == 2
    // ========================================================================
    {
        let world_group = world.group().expect("world.group() failed in Test 3");
        let sub = match world_group.include(&[0, 2]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: include(&[0,2]) failed: {e}");
                local_ok = false;
                // Create a dummy group to satisfy code flow — we use the
                // world_group again so Drop is valid.
                world.group().expect("fallback group failed")
            }
        };

        match sub.size() {
            Ok(s) if s == 2 => {
                if rank == 0 {
                    println!("PASS: Test 3 — include(&[0, 2]).size() = {s}");
                }
            }
            Ok(s) => {
                eprintln!("rank {rank}: FAIL Test 3: include size = {s}, expected 2");
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 3: include size error: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 4: exclude(&[1, 3]).size() == 2
    // ========================================================================
    {
        let world_group = world.group().expect("world.group() failed in Test 4");
        let sub = match world_group.exclude(&[1, 3]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 4: exclude(&[1,3]) failed: {e}");
                local_ok = false;
                world.group().expect("fallback group failed")
            }
        };

        match sub.size() {
            Ok(s) if s == 2 => {
                if rank == 0 {
                    println!("PASS: Test 4 — exclude(&[1, 3]).size() = {s}");
                }
            }
            Ok(s) => {
                eprintln!("rank {rank}: FAIL Test 4: exclude size = {s}, expected 2");
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 4: exclude size error: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 5: rank in included sub-group
    //   rank 0 of WORLD -> rank 0 of sub-group {0, 2}
    //   rank 2 of WORLD -> rank 1 of sub-group {0, 2}
    //   ranks 1, 3 of WORLD -> None (not a member)
    // ========================================================================
    {
        let world_group = world.group().expect("world.group() failed in Test 5");
        let sub = match world_group.include(&[0, 2]) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 5: include failed: {e}");
                local_ok = false;
                world.group().expect("fallback group failed")
            }
        };

        let expected_sub_rank = match rank {
            0 => Some(0),
            2 => Some(1),
            _ => None,
        };

        match sub.rank() {
            Ok(r) if r == expected_sub_rank => {
                if rank == 0 {
                    println!("PASS: Test 5 — sub-group rank correct for rank 0 (sub-rank = {r:?})");
                }
                if rank == 2 {
                    println!("PASS: Test 5 — sub-group rank correct for rank 2 (sub-rank = {r:?})");
                }
            }
            Ok(r) => {
                eprintln!(
                    "rank {rank}: FAIL Test 5: sub-group rank = {r:?}, expected {expected_sub_rank:?}"
                );
                local_ok = false;
            }
            Err(e) => {
                eprintln!("rank {rank}: FAIL Test 5: sub-group rank error: {e}");
                local_ok = false;
            }
        }
    }

    // ========================================================================
    // Test 6: an empty group takes no slot of the group table.
    //   The table has 63 slots. 64 empty results of each of the seven calls
    //   that can return one are held at once, and a group with a member can
    //   still be made.
    // ========================================================================
    {
        let world_group = world.group().expect("world.group() failed in Test 6");
        let low = world_group.include(&[0, 1]).expect("include failed");
        let high = world_group.include(&[2, 3]).expect("include failed");
        let nobody = world_group.include(&[]).expect("include of no rank failed");
        let everyone = RankRange {
            first: 0,
            last: 3,
            stride: 1,
        };
        let mut held = Vec::new();
        for _ in 0..64 {
            held.push(world_group.include(&[]));
            held.push(world_group.exclude(&[0, 1, 2, 3]));
            held.push(world_group.difference(&world_group));
            held.push(low.intersection(&high));
            held.push(nobody.union(&nobody));
            held.push(world_group.range_include(&[]));
            held.push(world_group.range_exclude(&[everyone]));
        }
        let all_empty = held
            .iter()
            .all(|group| group.as_ref().is_ok_and(|g| g.size().ok() == Some(0)));
        let still_room = world_group
            .include(&[0])
            .is_ok_and(|g| g.size().ok() == Some(1));
        if all_empty && still_room {
            if rank == 0 {
                println!("PASS: Test 6 — {} empty groups held at once", held.len());
            }
        } else {
            eprintln!(
                "rank {rank}: FAIL Test 6: all empty = {all_empty}, room for one more = {still_room}"
            );
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_group_basic");
}
