//! Regression example for persistent-request completion bookkeeping: a
//! failed `wait` or `test` on an active persistent request must leave it
//! inactive, the same way MPI itself completes it (successfully or not),
//! instead of leaving it stuck active until a later call recovers it, or,
//! on a library that frees a failed persistent request, stuck forever.
//!
//! Rank 1 sends a 4-element payload into rank 0's 1-element persistent
//! receive to force `MPI_ERR_TRUNCATE`. Part 1 exercises `wait`; part 2
//! exercises `test`. Barriers order each send after rank 0 has posted the
//! matching receive.
//!
//! Part 3 checks the other half of the same bookkeeping: the finalize sweep
//! must not count a request that failed through `wait`/`test` as still
//! active. Rank 0 keeps its two failed persistent requests alive (never
//! dropped) alongside one plain nonblocking receive it deliberately never
//! completes, then drops `Mpi` while all three are still alive. Only the
//! never-completed plain receive should still be counted; the stderr
//! directive below pins the expected count.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_error
// mpi-test: np=2 valgrind
// mpi-test-stderr: ferrompi: MPI_Finalize leaves 1 active request(s) unfreed

use ferrompi::{Communicator, Error, Mpi, MpiErrorClass};

mod common;

const PAYLOAD: [i32; 4] = [1, 2, 3, 4];

fn class_of<T>(result: &ferrompi::Result<T>) -> Option<MpiErrorClass> {
    match result {
        Err(Error::Mpi { class, .. }) => Some(*class),
        _ => None,
    }
}

fn part1_wait(world: &Communicator, rank: i32, mpich: bool) {
    let mut small = [0i32; 1];

    if rank == 0 {
        let mut req = world
            .recv_init(&mut small, 1, 71)
            .expect("part1: recv_init");
        req.start().expect("part1: start");
        world.barrier().expect("part1: barrier A");
        world.barrier().expect("part1: barrier B");

        let result = req.wait();
        let mut ok = class_of(&result) == Some(MpiErrorClass::Truncate);
        ok &= !req.is_active();

        if mpich {
            ok &= req.start().is_ok();
            world.barrier().expect("part1: barrier C");
            world.barrier().expect("part1: barrier D");
            ok &= req.wait().is_ok();
            ok &= small == [5];
        } else {
            let start_result = req.start();
            ok &= matches!(
                start_result,
                Err(Error::Mpi {
                    class: MpiErrorClass::Request,
                    ..
                })
            );
            world.barrier().expect("part1: barrier C");
            world.barrier().expect("part1: barrier D");
        }

        common::check(
            world,
            ok,
            "part 1: a failed persistent wait leaves the request inactive",
        );
    } else {
        world.barrier().expect("part1: barrier A");
        world
            .send(&PAYLOAD, 0, 71)
            .expect("part1: send (truncates)");
        world.barrier().expect("part1: barrier B");

        world.barrier().expect("part1: barrier C");
        if mpich {
            world.send(&[5i32], 0, 71).expect("part1: send restart");
        }
        world.barrier().expect("part1: barrier D");

        common::check(
            world,
            true,
            "part 1: a failed persistent wait leaves the request inactive",
        );
    }
}

fn part2_test(world: &Communicator, rank: i32) {
    let mut small = [0i32; 1];

    if rank == 0 {
        let mut req = world
            .recv_init(&mut small, 1, 72)
            .expect("part2: recv_init");
        req.start().expect("part2: start");
        world.barrier().expect("part2: barrier A");
        world.barrier().expect("part2: barrier B");

        let mut result = Ok(false);
        for _ in 0..1_000_000 {
            result = req.test();
            match &result {
                Ok(false) => continue,
                _ => break,
            }
        }

        let mut ok = class_of(&result) == Some(MpiErrorClass::Truncate);
        ok &= !req.is_active();
        drop(req);

        common::check(
            world,
            ok,
            "part 2: a failed persistent test leaves the request inactive",
        );
    } else {
        world.barrier().expect("part2: barrier A");
        world
            .send(&PAYLOAD, 0, 72)
            .expect("part2: send (truncates)");
        world.barrier().expect("part2: barrier B");

        common::check(
            world,
            true,
            "part 2: a failed persistent test leaves the request inactive",
        );
    }
}

// Rank 0 keeps two failed persistent requests (one completed through
// `wait`, one through `test`) and one never-completed plain receive alive
// across `drop(mpi)`, then finalizes while all three are still registered.
// Only the never-completed plain receive should be counted active; the
// module-level `mpi-test-stderr` directive pins the expected count to
// catch the finalize sweep miscounting either failed persistent request.
fn part3_finalize_accounting(mpi: Mpi, world: &Communicator, rank: i32) {
    let mut small_a = [0i32; 1];
    let mut small_b = [0i32; 1];
    let mut ctrl = [0i32; 1];

    if rank == 0 {
        let mut req_a = world
            .recv_init(&mut small_a, 1, 73)
            .expect("part3: recv_init a");
        let mut req_b = world
            .recv_init(&mut small_b, 1, 74)
            .expect("part3: recv_init b");
        req_a.start().expect("part3: start a");
        req_b.start().expect("part3: start b");
        world.barrier().expect("part3: barrier A");
        world.barrier().expect("part3: barrier B");

        let result_a = req_a.wait();
        let mut ok = class_of(&result_a) == Some(MpiErrorClass::Truncate);
        ok &= !req_a.is_active();

        let mut result_b = Ok(false);
        for _ in 0..1_000_000 {
            result_b = req_b.test();
            match &result_b {
                Ok(false) => continue,
                _ => break,
            }
        }
        ok &= class_of(&result_b) == Some(MpiErrorClass::Truncate);
        ok &= !req_b.is_active();

        // Never waited: stays active until finalize, giving the sweep
        // exactly one request it must count.
        let ctrl_req = world.irecv(&mut ctrl, 1, 75).expect("part3: irecv ctrl");
        world.barrier().expect("part3: barrier C");

        common::check(
            world,
            ok,
            "part 3: a failed persistent wait or test does not leave the request counted active at finalize",
        );

        // Drop Mpi while req_a, req_b and ctrl_req are still alive: their
        // own Drop impls run afterward, see the finalized lifecycle state,
        // and make no further MPI call.
        drop(mpi);
        drop(req_a);
        drop(req_b);
        drop(ctrl_req);
    } else {
        world.barrier().expect("part3: barrier A");
        world
            .send(&PAYLOAD, 0, 73)
            .expect("part3: send a (truncates)");
        world
            .send(&PAYLOAD, 0, 74)
            .expect("part3: send b (truncates)");
        world.barrier().expect("part3: barrier B");

        world.barrier().expect("part3: barrier C");
        world.send(&[9i32], 0, 75).expect("part3: send ctrl");

        common::check(
            world,
            true,
            "part 3: a failed persistent wait or test does not leave the request counted active at finalize",
        );

        drop(mpi);
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_error requires exactly 2 processes, got {size}"
    );

    let mpich = Mpi::library_version()
        .expect("library_version failed")
        .contains("MPICH");

    part1_wait(&world, rank, mpich);
    part2_test(&world, rank);
    part3_finalize_accounting(mpi, &world, rank);

    if rank == 0 {
        println!("PASS: test_persistent_error");
    }
}
