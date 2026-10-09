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
//! Part 3 checks that a batch wait started over one request left inactive
//! by an earlier failed wait, alongside one still-active request, skips the
//! inactive one and completes the other instead of rejecting the whole
//! batch.
//!
//! Part 4 checks a `start_all` over a persistent receive from self and a
//! one-element persistent buffered send: on MPICH, an unattached buffer
//! makes it fail after starting only the receive; both requests must come
//! back `is_active()` regardless. Other libraries observed so far do not
//! enforce the attached-buffer requirement at this size, so there a small
//! buffer is attached anyway and `start_all` succeeds outright — still
//! exercising the same bookkeeping, just without a failure to over-mark
//! from. Either way, a matching send to self and a `wait_all` complete
//! whichever of the two requests did start, and the receive's data must
//! have arrived.
//!
//! Part 5 checks the other half of the same bookkeeping: a request that failed
//! through `wait`/`test` must release the token every active persistent request
//! holds, and must not be counted as still active at finalize. Rank 0 keeps its
//! two failed persistent requests alive (never dropped) and drops `Mpi` while
//! both are still alive. `MPI_Finalize` must run: the runner fails the example
//! if it prints the skip marker, which it does when a request still holds its
//! token. The sweep then frees both.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_error
// mpi-test: np=2 valgrind

use ferrompi::{Communicator, Mpi, MpiErrorClass, PersistentRequest};

mod common;

use common::{class_of, PAYLOAD};

fn part1_wait(world: &Communicator, rank: i32, mpich: bool) {
    if rank == 0 {
        let mut req = world
            .recv_init(vec![0i32; 1], 1, 71)
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
            ok &= req.buffer().is_ok_and(|b| b == [5]);
        } else {
            ok &= class_of(&req.start()) == Some(MpiErrorClass::Request);
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
    if rank == 0 {
        let mut req = world
            .recv_init(vec![0i32; 1], 1, 72)
            .expect("part2: recv_init");
        req.start().expect("part2: start");
        world.barrier().expect("part2: barrier A");
        world.barrier().expect("part2: barrier B");

        let mut result = Ok(None);
        for _ in 0..1_000_000 {
            result = req.test();
            match &result {
                Ok(None) => continue,
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

// A batch wait over [a request a failed wait already left inactive, a
// still-active request] must skip the inactive one and complete the other,
// instead of rejecting the whole batch because the inactive request's
// handle no longer names a live request on a library that frees a failed
// persistent request.
fn part3_wait_all_skips_inactive(world: &Communicator, rank: i32) {
    if rank == 0 {
        let req_small = world
            .recv_init(vec![0i32; 1], 1, 73)
            .expect("part3: recv_init small");
        let req_full = world
            .recv_init(vec![0i32; 4], 1, 74)
            .expect("part3: recv_init full");
        let mut reqs = [req_small, req_full];
        PersistentRequest::start_all(&mut reqs).expect("part3: start_all");
        world.barrier().expect("part3: barrier A");
        world.barrier().expect("part3: barrier B");

        let result = reqs[0].wait();
        let mut ok = class_of(&result) == Some(MpiErrorClass::Truncate);
        ok &= !reqs[0].is_active();

        world.barrier().expect("part3: barrier C");
        world.barrier().expect("part3: barrier D");

        ok &= PersistentRequest::wait_all(&mut reqs).is_ok();
        ok &= reqs[1].buffer().is_ok_and(|b| b == PAYLOAD);
        ok &= !reqs[1].is_active();

        common::check(
            world,
            ok,
            "part 3: wait_all skips a request a failed wait left inactive",
        );
    } else {
        world.barrier().expect("part3: barrier A");
        world
            .send(&PAYLOAD, 0, 73)
            .expect("part3: send small (truncates)");
        world.barrier().expect("part3: barrier B");

        world.barrier().expect("part3: barrier C");
        world.send(&PAYLOAD, 0, 74).expect("part3: send full");
        world.barrier().expect("part3: barrier D");

        common::check(
            world,
            true,
            "part 3: wait_all skips a request a failed wait left inactive",
        );
    }
}

// A `start_all` that fails partway through must still mark every request
// it was given active, not just the ones MPI finished starting: rank 0
// builds a receive from itself (tag 75) and a one-element buffered send
// (tag 79, never matched). On MPICH, `bsend_init` with no buffer attached
// fails `MPI_Startall` outright once it reaches the send, after the
// receive has already started. Other libraries observed so far do not
// enforce the attached-buffer requirement for a message this small, so
// there a buffer is attached anyway — the standard-conforming way to use a
// buffered send — and `start_all` succeeds outright, which still exercises
// the same activity bookkeeping, just without a failure to over-mark from.
// Either way, a matching send to self and a `wait_all` complete whichever
// request did start, and the buffered send's own error, if the library
// reports one, must not stop the receive from completing.
fn part4_partial_start_all(world: &Communicator, mpi: &Mpi, rank: i32, mpich: bool) {
    if rank == 0 {
        if !mpich {
            mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice())
                .expect("part4: buffer_attach");
        }

        let recv_req = world
            .recv_init(vec![0i32; 1], 0, 75)
            .expect("part4: recv_init from self");
        let bsend_req = world
            .bsend_init(vec![0i32; 1], 0, 79)
            .expect("part4: bsend_init");
        let mut reqs = [recv_req, bsend_req];

        let start_result = PersistentRequest::start_all(&mut reqs);
        let mut ok = if mpich {
            start_result.is_err()
        } else {
            start_result.is_ok()
        };
        ok &= reqs[0].is_active();
        ok &= reqs[1].is_active();

        world
            .send(&[42i32], 0, 75)
            .expect("part4: send to self (matches the recv)");

        let wait_result = PersistentRequest::wait_all(&mut reqs);
        // On MPICH the buffered send never had a buffer to send from, so
        // wait_all reports ERR_BUFFER behind it; with a buffer attached it
        // completes normally.
        ok &= if mpich {
            class_of(&wait_result) == Some(MpiErrorClass::Buffer)
        } else {
            wait_result.is_ok()
        };
        ok &= !reqs[0].is_active();
        ok &= !reqs[1].is_active();
        ok &= reqs[0].buffer().is_ok_and(|b| b == [42]);

        if !mpich {
            ok &= mpi.buffer_detach().is_ok();
        }

        common::check(
            world,
            ok,
            "part 4: start_all marks every request active after a partial failure",
        );
    } else {
        common::check(
            world,
            true,
            "part 4: start_all marks every request active after a partial failure",
        );
    }
}

// Rank 0 keeps two failed persistent requests (one completed through
// `wait`, one through `test`) alive across `drop(mpi)`, then finalizes while
// both are still registered. Neither may hold a token or be counted active:
// `MPI_Finalize` must run, which the runner checks by the absence of the skip
// marker.
fn part5_finalize_accounting(mpi: Mpi, world: &Communicator, rank: i32) {
    if rank == 0 {
        let mut req_a = world
            .recv_init(vec![0i32; 1], 1, 76)
            .expect("part5: recv_init a");
        let mut req_b = world
            .recv_init(vec![0i32; 1], 1, 77)
            .expect("part5: recv_init b");
        req_a.start().expect("part5: start a");
        req_b.start().expect("part5: start b");
        world.barrier().expect("part5: barrier A");
        world.barrier().expect("part5: barrier B");

        let result_a = req_a.wait();
        let mut ok = class_of(&result_a) == Some(MpiErrorClass::Truncate);
        ok &= !req_a.is_active();

        let mut result_b = Ok(None);
        for _ in 0..1_000_000 {
            result_b = req_b.test();
            match &result_b {
                Ok(None) => continue,
                _ => break,
            }
        }
        ok &= class_of(&result_b) == Some(MpiErrorClass::Truncate);
        ok &= !req_b.is_active();

        common::check(
            world,
            ok,
            "part 5: a failed persistent wait or test does not leave the request counted active at finalize",
        );

        // Drop Mpi while req_a and req_b are still alive: their own Drop
        // impls run afterward, see the finalized lifecycle state, and make no
        // further MPI call.
        drop(mpi);
        drop(req_a);
        drop(req_b);
    } else {
        world.barrier().expect("part5: barrier A");
        world
            .send(&PAYLOAD, 0, 76)
            .expect("part5: send a (truncates)");
        world
            .send(&PAYLOAD, 0, 77)
            .expect("part5: send b (truncates)");
        world.barrier().expect("part5: barrier B");

        common::check(
            world,
            true,
            "part 5: a failed persistent wait or test does not leave the request counted active at finalize",
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
    part3_wait_all_skips_inactive(&world, rank);
    part4_partial_start_all(&world, &mpi, rank, mpich);
    part5_finalize_accounting(mpi, &world, rank);

    if rank == 0 {
        println!("PASS: test_persistent_error");
    }
}
