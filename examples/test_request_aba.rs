//! Regression example for the request-handle ABA bug: a completed request's
//! table slot is freed and can be reused by the next nonblocking call, so a
//! stale handle left in a slice or held past completion must not act on
//! whatever request now occupies the reused slot.
//!
//! Four parts, each ending in a `common::check`:
//!
//! 1. The Waitany loop idiom: calling `wait_any` repeatedly on the same
//!    slice without removing completed entries reaches `Ok(None)`, and
//!    `test_any`/`wait_some`/`test_some` agree once every entry is done.
//! 2. `wait_all` on a slice that is already fully completed is a no-op and
//!    does not disturb new requests that reused the freed slots.
//! 3. A completed entry left in a slice does not alias a newer request that
//!    reused its slot when a later batch call runs on the same slice.
//! 4. A `test()` that fails with a truncation error still marks the request
//!    completed, so dropping it does not disturb a newer request that
//!    reused its slot.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_request_aba
// mpi-test: np=2

use ferrompi::{Communicator, Error, Mpi, MpiErrorClass, Request};

mod common;

// Drains `reqs` through the Waitany loop idiom. Returns `Err` if any call
// fails outright (the BASE behavior: the loop idiom cannot even reach
// `Ok(None)`), `Ok(false)` if a call succeeds with an unexpected value, or
// `Ok(true)` once every check passes.
fn part1_check(reqs: &mut [Request]) -> ferrompi::Result<bool> {
    let first = Request::wait_any(reqs)?;
    let second = Request::wait_any(reqs)?;
    if !(first.is_some() && second.is_some() && first != second) {
        return Ok(false);
    }
    if Request::wait_any(reqs)?.is_some() {
        return Ok(false);
    }
    if Request::test_any(reqs)?.is_some() {
        return Ok(false);
    }
    if !Request::wait_some(reqs)?.is_empty() {
        return Ok(false);
    }
    if !Request::test_some(reqs)?.is_empty() {
        return Ok(false);
    }
    Ok(true)
}

fn part1_wait_any_loop(world: &Communicator, rank: i32) {
    let mut buf1 = [0u8; 4];
    let mut buf2 = [0u8; 4];
    let mut ok = true;

    if rank == 0 {
        ok = ferrompi::scope(|s| {
            let mut reqs = vec![
                world.irecv(s, &mut buf1, 1, 1).expect("part1: irecv tag1"),
                world.irecv(s, &mut buf2, 1, 2).expect("part1: irecv tag2"),
            ];
            world.barrier().expect("part1: barrier after posting");

            Ok(part1_check(&mut reqs).unwrap_or(false))
        })
        .expect("part1: scope failed");
        ok &= buf1 == [1u8; 4] && buf2 == [2u8; 4];
    } else {
        world.barrier().expect("part1: barrier after posting");
        world.send(&[1u8; 4], 0, 1).expect("part1: send tag1");
        world.send(&[2u8; 4], 0, 2).expect("part1: send tag2");
    }

    common::check(world, ok, "part 1: wait_any loop reaches Ok(None)");
}

fn part2_wait_all_completed_slice(world: &Communicator, rank: i32) {
    let mut old_a = [0u8; 4];
    let mut old_b = [0u8; 4];
    let mut new_a = [0u8; 4];
    let mut new_b = [0u8; 4];
    let mut ok = true;

    if rank == 0 {
        ok &= ferrompi::scope(|s| {
            let mut old = vec![
                world
                    .irecv(s, &mut old_a, 1, 3)
                    .expect("part2: irecv old_a"),
                world
                    .irecv(s, &mut old_b, 1, 4)
                    .expect("part2: irecv old_b"),
            ];
            world.barrier().expect("part2: barrier after posting old");

            Request::wait_all(&mut old).expect("part2: wait_all(old)");

            let new_recv_a = world
                .irecv(s, &mut new_a, 1, 5)
                .expect("part2: irecv new_a");
            let new_recv_b = world
                .irecv(s, &mut new_b, 1, 6)
                .expect("part2: irecv new_b");
            world.barrier().expect("part2: barrier after posting new");

            let mut scope_ok = Request::wait_all(&mut old).is_ok();
            scope_ok &= new_recv_a.wait().is_ok();
            scope_ok &= new_recv_b.wait().is_ok();
            Ok(scope_ok)
        })
        .expect("part2: scope failed");
        ok &= new_a == [5u8; 4] && new_b == [6u8; 4];
    } else {
        world.barrier().expect("part2: barrier after posting old");
        world.send(&[3u8; 4], 0, 3).expect("part2: send tag3");
        world.send(&[4u8; 4], 0, 4).expect("part2: send tag4");
        world.barrier().expect("part2: barrier after posting new");
        world.send(&[5u8; 4], 0, 5).expect("part2: send tag5");
        world.send(&[6u8; 4], 0, 6).expect("part2: send tag6");
    }

    common::check(
        world,
        ok,
        "part 2: wait_all on a completed slice is a no-op",
    );
}

fn part3_stale_wait_any_entry(world: &Communicator, rank: i32) {
    let mut a_buf = [0u8; 4];
    let mut b_buf = [0u8; 4];
    let mut other_buf = [0u8; 4];
    let mut ok = true;

    if rank == 0 {
        ok &= ferrompi::scope(|s| {
            let mut reqs = vec![
                world.irecv(s, &mut a_buf, 1, 11).expect("part3: irecv a"),
                world.irecv(s, &mut b_buf, 1, 12).expect("part3: irecv b"),
            ];
            world.barrier().expect("part3: barrier after posting a,b");

            let mut scope_ok = true;
            let first = Request::wait_any(&mut reqs)
                .expect("part3: first wait_any")
                .map(|(i, _)| i);
            scope_ok &= first == Some(0);

            let other = world
                .irecv(s, &mut other_buf, 1, 13)
                .expect("part3: irecv other");
            world.barrier().expect("part3: barrier after posting other");

            let second = Request::wait_any(&mut reqs)
                .expect("part3: second wait_any")
                .map(|(i, _)| i);
            scope_ok &= second == Some(1);
            scope_ok &= !other.is_completed();
            scope_ok &= other.wait().is_ok();
            Ok(scope_ok)
        })
        .expect("part3: scope failed");
        ok &= a_buf == [11u8; 4] && b_buf == [12u8; 4] && other_buf == [13u8; 4];
    } else {
        world.barrier().expect("part3: barrier after posting a,b");
        world.send(&[11u8; 4], 0, 11).expect("part3: send tag11");
        world.barrier().expect("part3: barrier after posting other");
        world.send(&[13u8; 4], 0, 13).expect("part3: send tag13");
        world.send(&[12u8; 4], 0, 12).expect("part3: send tag12");
    }

    common::check(
        world,
        ok,
        "part 3: stale wait_any entry does not alias a reused slot",
    );
}

fn part4_failed_test(world: &Communicator, rank: i32) {
    let mut a_buf = [0i32; 1];
    let mut b_buf = [0i32; 1];
    let mut ok = true;

    if rank == 0 {
        ok &= ferrompi::scope(|s| {
            let mut a = world.irecv(s, &mut a_buf, 1, 21).expect("part4: irecv a");
            world.barrier().expect("part4: barrier after posting a");

            let result = loop {
                match a.test() {
                    Ok(None) => continue,
                    other => break other,
                }
            };
            let mut scope_ok = matches!(
                result,
                Err(Error::Mpi {
                    class: MpiErrorClass::Truncate,
                    ..
                })
            );
            scope_ok &= a.is_completed();

            let b = world.irecv(s, &mut b_buf, 1, 22).expect("part4: irecv b");
            world.barrier().expect("part4: barrier after posting b");

            #[expect(
                clippy::drop_non_drop,
                reason = "Request has no Drop impl; the drop marks the early release"
            )]
            drop(a);
            scope_ok &= b.wait().is_ok();
            Ok(scope_ok)
        })
        .expect("part4: scope failed");
        ok &= b_buf == [42];
    } else {
        world.barrier().expect("part4: barrier after posting a");
        world
            .send(&[1i32, 2, 3], 0, 21)
            .expect("part4: send truncating tag21");
        world.barrier().expect("part4: barrier after posting b");
        world.send(&[42i32], 0, 22).expect("part4: send tag22");
    }

    common::check(
        world,
        ok,
        "part 4: failed test() still marks the request completed",
    );
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_request_aba requires exactly 2 processes, got {size}"
    );

    part1_wait_any_loop(&world, rank);
    part2_wait_all_completed_slice(&world, rank);
    part3_stale_wait_any_entry(&world, rank);
    part4_failed_test(&world, rank);

    if rank == 0 {
        println!("\n========================================");
        println!("All request-ABA tests passed!");
        println!("========================================");
    }
}
