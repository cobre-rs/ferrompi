//! Regression example for write-back on every batch completion: a failed
//! `wait_all`/`wait_any`/`wait_some`/`test_any`/`test_some` (and the
//! persistent `wait_all`) must still write MPI's own completion state back
//! into the request table, whatever the return code.
//!
//! Rank 1 sends 4 `i32` into rank 0's 1-element receives, which truncates
//! (`MPI_ERR_TRUNCATE`). Barriers order each send after rank 0 has posted
//! the matching receive. Before the fix, the drop in part 1b hung forever
//! (MPI's already-freed request object was reused by a later receive), so
//! the runner timeout is itself part of the oracle.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_waitall_error
// mpi-test: np=2 valgrind

use ferrompi::{Communicator, Error, Mpi, MpiErrorClass, PersistentRequest, Request};

mod common;

/// A 4-element payload rank 1 sends into a 1-element ("small") receive
/// buffer to force `MPI_ERR_TRUNCATE`, or into a 4-element ("big") buffer,
/// which it fits exactly.
const PAYLOAD: [i32; 4] = [1, 2, 3, 4];

fn class_of<T>(result: &ferrompi::Result<T>) -> Option<MpiErrorClass> {
    match result {
        Err(Error::Mpi { class, .. }) => Some(*class),
        _ => None,
    }
}

fn part1_and_1b(world: &Communicator, rank: i32, open_mpi: bool) {
    let mut small = [0i32; 1];
    let mut big = [0i32; 4];

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut small, 1, 1).expect("part1: irecv small"),
            world.irecv(&mut big, 1, 2).expect("part1: irecv big"),
        ];
        world.barrier().expect("part1: barrier after posting");

        let mut ok = true;
        let result = Request::wait_all(&mut reqs);
        ok &= class_of(&result) == Some(MpiErrorClass::InStatus);
        let completed: Vec<bool> = reqs.iter().map(Request::is_completed).collect();
        ok &= completed == [true, open_mpi];

        ok &= Request::wait_all(&mut reqs).is_ok();
        ok &= reqs.iter().all(Request::is_completed);
        ok &= big == PAYLOAD;
        common::check(
            world,
            ok,
            "part 1: failed wait_all completes the failed request",
        );

        let mut ok1b = true;
        ok1b &= matches!(reqs[0].test(), Ok(true));
        ok1b &= matches!(reqs[1].test(), Ok(true));

        let mut fresh_buf = [0i32; 1];
        let fresh = world
            .irecv(&mut fresh_buf, 1, 3)
            .expect("part1b: irecv fresh");
        world.barrier().expect("part1b: barrier before send");

        drop(reqs);

        ok1b &= fresh.wait().is_ok();
        ok1b &= fresh_buf == [3];
        common::check(
            world,
            ok1b,
            "part 1b: test() and drop on a completed slice are no-ops",
        );
    } else {
        world.barrier().expect("part1: barrier after posting");
        world
            .send(&PAYLOAD, 0, 1)
            .expect("part1: send small (truncates)");
        world.send(&PAYLOAD, 0, 2).expect("part1: send big");
        common::check(
            world,
            true,
            "part 1: failed wait_all completes the failed request",
        );

        world.barrier().expect("part1b: barrier before send");
        world.send(&[3i32], 0, 3).expect("part1b: send fresh");
        common::check(
            world,
            true,
            "part 1b: test() and drop on a completed slice are no-ops",
        );
    }
}

fn part2_wait_any_truncate(world: &Communicator, rank: i32) {
    let mut small = [0i32; 1];
    let mut other = [0i32; 1];

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut small, 1, 11).expect("part2: irecv small"),
            world.irecv(&mut other, 1, 12).expect("part2: irecv other"),
        ];
        world.barrier().expect("part2: barrier after posting");

        let result = Request::wait_any(&mut reqs);
        let mut ok = class_of(&result) == Some(MpiErrorClass::Truncate);
        ok &= reqs[0].is_completed() && !reqs[1].is_completed();

        world.barrier().expect("part2: barrier before other send");
        ok &= Request::wait_all(&mut reqs).is_ok();

        common::check(
            world,
            ok,
            "part 2: wait_any reports the truncating request's own error",
        );
    } else {
        world.barrier().expect("part2: barrier after posting");
        world
            .send(&PAYLOAD, 0, 11)
            .expect("part2: send small (truncates)");

        world.barrier().expect("part2: barrier before other send");
        world.send(&[12i32], 0, 12).expect("part2: send other");

        common::check(
            world,
            true,
            "part 2: wait_any reports the truncating request's own error",
        );
    }
}

fn part3_wait_some_in_status(world: &Communicator, rank: i32) {
    let mut other = [0i32; 1];
    let mut small = [0i32; 1];

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut other, 1, 22).expect("part3: irecv other"),
            world.irecv(&mut small, 1, 21).expect("part3: irecv small"),
        ];
        world.barrier().expect("part3: barrier after posting");

        let result = Request::wait_some(&mut reqs);
        let mut ok = class_of(&result) == Some(MpiErrorClass::InStatus);
        ok &= reqs[1].is_completed() && !reqs[0].is_completed();

        world.barrier().expect("part3: barrier before other send");
        ok &= Request::wait_all(&mut reqs).is_ok();

        common::check(
            world,
            ok,
            "part 3: wait_some reports the truncating request in-status",
        );
    } else {
        world.barrier().expect("part3: barrier after posting");
        world
            .send(&PAYLOAD, 0, 21)
            .expect("part3: send small (truncates)");

        world.barrier().expect("part3: barrier before other send");
        world.send(&[22i32], 0, 22).expect("part3: send other");

        common::check(
            world,
            true,
            "part 3: wait_some reports the truncating request in-status",
        );
    }
}

fn part4_test_any_truncate(world: &Communicator, rank: i32) {
    let mut small = [0i32; 1];
    let mut other = [0i32; 1];

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut small, 1, 31).expect("part4: irecv small"),
            world.irecv(&mut other, 1, 32).expect("part4: irecv other"),
        ];
        world.barrier().expect("part4: barrier after posting");

        let result = loop {
            match Request::test_any(&mut reqs) {
                Ok(None) => continue,
                other => break other,
            }
        };
        let mut ok = class_of(&result) == Some(MpiErrorClass::Truncate);
        ok &= reqs[0].is_completed() && !reqs[1].is_completed();

        world.barrier().expect("part4: barrier before other send");
        ok &= Request::wait_all(&mut reqs).is_ok();

        common::check(
            world,
            ok,
            "part 4: test_any reports the truncating request's own error",
        );
    } else {
        world.barrier().expect("part4: barrier after posting");
        world
            .send(&PAYLOAD, 0, 31)
            .expect("part4: send small (truncates)");

        world.barrier().expect("part4: barrier before other send");
        world.send(&[32i32], 0, 32).expect("part4: send other");

        common::check(
            world,
            true,
            "part 4: test_any reports the truncating request's own error",
        );
    }
}

fn part5_test_some_in_status(world: &Communicator, rank: i32) {
    let mut other = [0i32; 1];
    let mut small = [0i32; 1];

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut other, 1, 42).expect("part5: irecv other"),
            world.irecv(&mut small, 1, 41).expect("part5: irecv small"),
        ];
        world.barrier().expect("part5: barrier after posting");

        let result = loop {
            match Request::test_some(&mut reqs) {
                Ok(v) if v.is_empty() => continue,
                other => break other,
            }
        };
        let mut ok = class_of(&result) == Some(MpiErrorClass::InStatus);
        ok &= reqs[1].is_completed() && !reqs[0].is_completed();

        world.barrier().expect("part5: barrier before other send");
        ok &= Request::wait_all(&mut reqs).is_ok();

        common::check(
            world,
            ok,
            "part 5: test_some reports the truncating request in-status",
        );
    } else {
        world.barrier().expect("part5: barrier after posting");
        world
            .send(&PAYLOAD, 0, 41)
            .expect("part5: send small (truncates)");

        world.barrier().expect("part5: barrier before other send");
        world.send(&[42i32], 0, 42).expect("part5: send other");

        common::check(
            world,
            true,
            "part 5: test_some reports the truncating request in-status",
        );
    }
}

fn part6_persistent_wait_all(world: &Communicator, rank: i32, open_mpi: bool) {
    let mut small = [0i32; 1];
    let mut big = [0i32; 4];

    if rank == 0 {
        let mut reqs = vec![
            world
                .recv_init(&mut small, 1, 51)
                .expect("part6: recv_init small"),
            world
                .recv_init(&mut big, 1, 52)
                .expect("part6: recv_init big"),
        ];
        PersistentRequest::start_all(&mut reqs).expect("part6: start_all");
        world.barrier().expect("part6: barrier after start_all");

        let mut ok = true;
        let result = PersistentRequest::wait_all(&mut reqs);
        ok &= if open_mpi {
            result.is_ok()
        } else {
            class_of(&result) == Some(MpiErrorClass::InStatus)
        };
        let active: Vec<bool> = reqs.iter().map(PersistentRequest::is_active).collect();
        ok &= active == [false, !open_mpi];

        ok &= PersistentRequest::wait_all(&mut reqs).is_ok();
        ok &= reqs.iter().all(|r| !r.is_active());

        ok &= PersistentRequest::start_all(&mut reqs).is_ok();
        world.barrier().expect("part6: barrier before restart send");

        ok &= PersistentRequest::wait_all(&mut reqs).is_ok();
        ok &= small == [9];

        common::check(
            world,
            ok,
            "part 6: persistent wait_all completes the failed request",
        );
    } else {
        world.barrier().expect("part6: barrier after start_all");
        world
            .send(&PAYLOAD, 0, 51)
            .expect("part6: send small (truncates)");
        world.send(&PAYLOAD, 0, 52).expect("part6: send big");

        world.barrier().expect("part6: barrier before restart send");
        world
            .send(&[9i32], 0, 51)
            .expect("part6: send small restart");
        world
            .send(&PAYLOAD, 0, 52)
            .expect("part6: send big restart");

        common::check(
            world,
            true,
            "part 6: persistent wait_all completes the failed request",
        );
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_waitall_error requires exactly 2 processes, got {size}"
    );

    let open_mpi = Mpi::library_version()
        .expect("library_version failed")
        .contains("Open MPI");

    part1_and_1b(&world, rank, open_mpi);
    part2_wait_any_truncate(&world, rank);
    part3_wait_some_in_status(&world, rank);
    part4_test_any_truncate(&world, rank);
    part5_test_some_in_status(&world, rank);
    part6_persistent_wait_all(&world, rank, open_mpi);

    if rank == 0 {
        println!("\n========================================");
        println!("All waitall-error tests passed!");
        println!("========================================");
    }
}
