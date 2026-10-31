//! Integration test for group calls made by two threads at
//! `ThreadLevel::Multiple`.
//!
//! MPICH 4.2 builds a group's ordered member list on its first use by a set
//! operation, a comparison or a rank translation, and does not lock the group
//! while it does so. The crate serialises those calls. Every round here makes
//! fresh groups, so every round is a first use, and releases its two threads
//! together.
//!
//! 1. Both threads make the same call on groups they share: `difference`,
//!    `intersection`, `union`, `compare` and `translate_ranks` in turn. Each
//!    result is compared with the members it must have.
//! 2. One thread creates a communicator from a fresh group with `create_group`
//!    while the other compares that group with a second one.
//!
//! The ordered list matters from three members on, so the test needs at least
//! 3 processes.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_group_threads
// mpi-test: np=3.. timeout=60

use ferrompi::{Group, GroupComparison, Mpi, ReduceOp, ThreadLevel};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod common;

/// Rounds of step 1; each of its five calls runs in a fifth of them.
const ROUNDS: usize = 2000;
/// Rounds of step 2, one communicator each.
const CREATE_ROUNDS: usize = 100;

/// Spins, giving the processor away now and then so that a machine with fewer
/// cores than threads still makes progress.
fn wait_until(done: impl Fn() -> bool) {
    let mut spins = 0u32;
    while !done() {
        std::hint::spin_loop();
        spins += 1;
        if spins % 256 == 0 {
            std::thread::yield_now();
        }
    }
}

/// Runs `first` and `second` on two threads released together. True if both
/// return true.
fn together(first: impl Fn() -> bool + Sync, second: impl Fn() -> bool + Sync) -> bool {
    let ready = AtomicUsize::new(0);
    let go = AtomicBool::new(false);
    let run = |call: &(dyn Fn() -> bool + Sync)| {
        ready.fetch_add(1, Ordering::Release);
        wait_until(|| go.load(Ordering::Acquire));
        call()
    };
    std::thread::scope(|s| {
        let a = s.spawn(|| run(&first));
        let b = s.spawn(|| run(&second));
        wait_until(|| ready.load(Ordering::Acquire) == 2);
        go.store(true, Ordering::Release);
        let a = a.join().unwrap_or(false);
        let b = b.join().unwrap_or(false);
        a && b
    })
}

/// True if `group` holds exactly the world ranks `want`, in that order.
fn has_members(group: &Group, world_group: &Group, want: &[i32]) -> bool {
    let positions: Vec<i32> = (0..want.len() as i32).collect();
    group.size().ok() == Some(want.len() as i32)
        && group
            .translate_ranks(&positions, world_group)
            .is_ok_and(|ranks| ranks.iter().copied().eq(want.iter().map(|&r| Some(r))))
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");
    let world = mpi.world();
    let size = world.size();
    assert!(
        size >= 3,
        "test_group_threads requires at least 3 processes, got {size}"
    );
    if mpi.thread_level() < ThreadLevel::Multiple {
        common::skip(
            &world,
            &format!(
                "MPI provided {:?}, MPI_THREAD_MULTIPLE required; skipping test",
                mpi.thread_level()
            ),
        );
        return;
    }
    let world_group = world.group().expect("world.group() failed");
    let half = (size + 1) / 2;

    // 1. The same call from both threads, on groups made for the round: `a`
    //    holds every rank in descending order from a start that moves with
    //    the round, `b` half of them in ascending order, and `c` the members
    //    of `a` in another order.
    let mut ok = true;
    for round in 0..ROUNDS {
        let start = round as i32 % size;
        let in_a: Vec<i32> = (0..size)
            .map(|i| (start + size - 1 - i).rem_euclid(size))
            .collect();
        let in_b: Vec<i32> = (0..half).map(|i| (start + i) % size).collect();
        let mut in_c = in_a.clone();
        in_c.rotate_left(1);
        let a = world_group.include(&in_a).expect("include failed");
        let b = world_group.include(&in_b).expect("include failed");
        let c = world_group.include(&in_c).expect("include failed");

        let only_a: Vec<i32> = in_a.iter().copied().filter(|r| !in_b.contains(r)).collect();
        let both: Vec<i32> = in_a.iter().copied().filter(|r| in_b.contains(r)).collect();
        let b_then_a: Vec<i32> = in_b.iter().chain(&only_a).copied().collect();
        let b_in_a: Vec<Option<i32>> = in_b
            .iter()
            .map(|r| in_a.iter().position(|x| x == r).map(|p| p as i32))
            .collect();
        let positions: Vec<i32> = (0..half).collect();

        let call = || match round % 5 {
            0 => a
                .difference(&b)
                .is_ok_and(|d| has_members(&d, &world_group, &only_a)),
            1 => a
                .intersection(&b)
                .is_ok_and(|d| has_members(&d, &world_group, &both)),
            2 => b
                .union(&a)
                .is_ok_and(|d| has_members(&d, &world_group, &b_then_a)),
            3 => a.compare(&c).ok() == Some(GroupComparison::Similar),
            _ => b.translate_ranks(&positions, &a).ok().as_ref() == Some(&b_in_a),
        };
        ok &= together(call, call);
    }
    common::check(&world, ok, "group calls from two threads on shared groups");

    // 2. One thread creates a communicator from a group made for the round
    //    while the other compares that group with a second one. Every rank is
    //    a member, so the creating threads of all ranks meet in the call.
    let mut ok = true;
    for round in 0..CREATE_ROUNDS {
        let start = round as i32 % size;
        let in_g: Vec<i32> = (0..size)
            .map(|i| (start + size - 1 - i).rem_euclid(size))
            .collect();
        let mut in_h = in_g.clone();
        in_h.rotate_left(1);
        let g = world_group.include(&in_g).expect("include failed");
        let h = world_group.include(&in_h).expect("include failed");
        let create = || match world.create_group(&g, round as i32) {
            Ok(Some(sub)) => {
                sub.size() == size && sub.allreduce_scalar(1i32, ReduceOp::Sum).ok() == Some(size)
            }
            _ => false,
        };
        let compare = || g.compare(&h).ok() == Some(GroupComparison::Similar);
        ok &= together(create, compare);
    }
    common::check(
        &world,
        ok,
        "create_group while another thread compares the group",
    );

    if world.rank() == 0 {
        println!("PASS: test_group_threads");
    }
}
