//! Integration test for `Communicator::create_group` (`MPI_Comm_create_group`).
//!
//! The group holds the even ranks of the world communicator. Unlike
//! `Communicator::create`, the call is collective over the group only.
//!
//! 1. A negative tag is refused with class `Tag`: the tag is passed to MPI and
//!    is not a wildcard. Each rank passes the group of itself alone, so no
//!    other rank takes part in the call.
//! 2. Only the members call: each receives `Ok(Some(comm))` whose size is the
//!    number of even ranks, and an allreduce on it counts its members.
//! 3. Every rank calls: a member receives `Some(comm)`; for any other rank the
//!    call is local and returns `Ok(None)`.
//! 4. With the empty group every rank receives `Ok(None)`.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_comm_create_group
// mpi-test: np=2.. timeout=30

use ferrompi::{Mpi, MpiErrorClass, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();
    assert!(
        size >= 2,
        "test_comm_create_group requires at least 2 processes, got {size}"
    );

    let evens: Vec<i32> = (0..size).step_by(2).collect();
    let members = evens.len() as i32;
    let member = rank % 2 == 0;
    let world_group = world.group().expect("world.group() failed");
    let group = world_group.include(&evens).expect("include failed");

    // 1. A tag MPI does not accept. With the group of this rank alone, no
    //    other rank takes part in the call.
    let alone = world_group.include(&[rank]).expect("include failed");
    let ok = common::class_of(&world.create_group(&alone, -5)) == Some(MpiErrorClass::Tag);
    common::check(&world, ok, "create_group refuses a negative tag");

    // 2. Only the members call.
    let mut ok = true;
    if member {
        ok = match world.create_group(&group, 3) {
            Ok(Some(sub)) => {
                sub.size() == members
                    && sub.rank() == rank / 2
                    && sub.allreduce_scalar(1i32, ReduceOp::Sum).ok() == Some(members)
            }
            _ => false,
        };
    }
    common::check(&world, ok, "create_group among the members of the group");

    // 3. Every rank calls: local, and None, for a rank outside the group.
    let created = world.create_group(&group, 4);
    let ok = match &created {
        Ok(Some(sub)) => member && sub.size() == members,
        Ok(None) => !member,
        Err(_) => false,
    };
    drop(created);
    common::check(
        &world,
        ok,
        "create_group returns None to a rank outside the group",
    );

    // 4. The empty group.
    let nobody = world_group.include(&[]).expect("include of no rank failed");
    let ok = matches!(world.create_group(&nobody, 5), Ok(None));
    common::check(&world, ok, "create_group of the empty group returns None");

    if rank == 0 {
        println!("PASS: test_comm_create_group");
    }
}
