//! Regression example for the request-handle ABA bug: a completed request's
//! table slot is freed and can be reused by the next nonblocking call, so a
//! stale handle left in a slice or held past completion must not act on
//! whatever request now occupies the reused slot.
//!
//! Five parts, each ending in a `common::check`:
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
//! 5. A stale raw handle passed directly to the C layer is rejected instead
//!    of acting on whatever now occupies the slot.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_request_aba
// mpi-test: np=2

use ferrompi::{Communicator, Error, Mpi, MpiErrorClass, Request};

mod common;

// Raw FFI declaration for the C-side shim under test (Part 5).
//
// SAFETY invariants for the call below:
//   - `request` is a stale i64 handle: valid table bounds, but freed and
//     possibly reused by the time of this call, so the generation check
//     inside the C-side lookup must reject it without touching whatever
//     request now occupies the slot.
//   - `flag` points to a valid i32 on the stack; it is only meaningful if
//     the call succeeds, which it must not.
#[allow(dead_code)]
extern "C" {
    fn ferrompi_test(request: i64, flag: *mut i32) -> std::ffi::c_int;
}

// Drains `reqs` through the Waitany loop idiom. Returns `Err` if any call
// fails outright (the BASE behavior: the loop idiom cannot even reach
// `Ok(None)`), `Ok(false)` if a call succeeds with an unexpected value, or
// `Ok(true)` once every check passes.
fn part1_check(reqs: &mut [Request], buf1: &[u8; 4], buf2: &[u8; 4]) -> ferrompi::Result<bool> {
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
    Ok(buf1 == &[1u8; 4] && buf2 == &[2u8; 4])
}

fn part1_wait_any_loop(world: &Communicator, rank: i32) {
    let mut buf1 = [0u8; 4];
    let mut buf2 = [0u8; 4];
    let mut ok = true;

    if rank == 0 {
        let mut reqs = vec![
            world.irecv(&mut buf1, 1, 1).expect("part1: irecv tag1"),
            world.irecv(&mut buf2, 1, 2).expect("part1: irecv tag2"),
        ];
        world.barrier().expect("part1: barrier after posting");

        ok = part1_check(&mut reqs, &buf1, &buf2).unwrap_or(false);
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
        let mut old = vec![
            world.irecv(&mut old_a, 1, 3).expect("part2: irecv old_a"),
            world.irecv(&mut old_b, 1, 4).expect("part2: irecv old_b"),
        ];
        world.barrier().expect("part2: barrier after posting old");

        Request::wait_all(&mut old).expect("part2: wait_all(old)");

        let new_recv_a = world.irecv(&mut new_a, 1, 5).expect("part2: irecv new_a");
        let new_recv_b = world.irecv(&mut new_b, 1, 6).expect("part2: irecv new_b");
        world.barrier().expect("part2: barrier after posting new");

        ok &= Request::wait_all(&mut old).is_ok();
        ok &= new_recv_a.wait().is_ok();
        ok &= new_recv_b.wait().is_ok();
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
        let mut reqs = vec![
            world.irecv(&mut a_buf, 1, 11).expect("part3: irecv a"),
            world.irecv(&mut b_buf, 1, 12).expect("part3: irecv b"),
        ];
        world.barrier().expect("part3: barrier after posting a,b");

        let first = Request::wait_any(&mut reqs).expect("part3: first wait_any");
        ok &= first == Some(0);

        let other = world
            .irecv(&mut other_buf, 1, 13)
            .expect("part3: irecv other");
        world.barrier().expect("part3: barrier after posting other");

        let second = Request::wait_any(&mut reqs).expect("part3: second wait_any");
        ok &= second == Some(1);
        ok &= !other.is_completed();
        ok &= other.wait().is_ok();
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
        let mut a = world.irecv(&mut a_buf, 1, 21).expect("part4: irecv a");
        world.barrier().expect("part4: barrier after posting a");

        let result = loop {
            match a.test() {
                Ok(false) => continue,
                other => break other,
            }
        };
        ok &= matches!(
            result,
            Err(Error::Mpi {
                class: MpiErrorClass::Truncate,
                ..
            })
        );
        ok &= a.is_completed();

        let b = world.irecv(&mut b_buf, 1, 22).expect("part4: irecv b");
        world.barrier().expect("part4: barrier after posting b");

        drop(a);
        ok &= b.wait().is_ok();
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

fn part5_stale_raw_handle(world: &Communicator, rank: i32) {
    let mut a_buf = [0u8; 4];
    let mut b_buf = [0u8; 4];
    let mut ok = true;

    if rank == 0 {
        let a = world.irecv(&mut a_buf, 1, 31).expect("part5: irecv a");
        let stale_handle = a.raw_handle();
        world.barrier().expect("part5: barrier after posting a");

        a.wait().expect("part5: wait a");

        let b = world.irecv(&mut b_buf, 1, 32).expect("part5: irecv b");
        common::check(
            world,
            (b.raw_handle() & 0xffff_ffff) == (stale_handle & 0xffff_ffff)
                && b.raw_handle() != stale_handle,
            "part 5 precondition: slot reused with a new generation",
        );
        world.barrier().expect("part5: barrier after posting b");

        let mut flag: i32 = 0;
        let raw_ret = unsafe {
            // SAFETY: see the invariant comment on the extern "C" block above.
            ferrompi_test(stale_handle, std::ptr::addr_of_mut!(flag))
        };
        ok &= raw_ret != 0;
        ok &= matches!(
            Error::from_code(raw_ret),
            Error::Mpi {
                class: MpiErrorClass::Request,
                ..
            }
        );

        ok &= b.wait().is_ok();
        ok &= b_buf == [32u8; 4];
    } else {
        world.barrier().expect("part5: barrier after posting a");
        world.send(&[31u8; 4], 0, 31).expect("part5: send tag31");
        common::check(
            world,
            true,
            "part 5 precondition: slot reused with a new generation",
        );
        world.barrier().expect("part5: barrier after posting b");
        world.send(&[32u8; 4], 0, 32).expect("part5: send tag32");
    }

    common::check(
        world,
        ok,
        "part 5: stale raw handle is rejected, not aliased",
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
    part5_stale_raw_handle(&world, rank);

    if rank == 0 {
        println!("\n========================================");
        println!("All request-ABA tests passed!");
        println!("========================================");
    }
}
