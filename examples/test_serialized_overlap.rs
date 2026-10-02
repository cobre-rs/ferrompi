//! Regression for the debug-only Serialized overlap check: a
//! `PersistentRequest` or `Request` call rejected because it overlaps
//! another thread's in-flight MPI call must leave the request's
//! active/completed state unchanged, not mark it as if MPI had completed
//! it. Needs a debug build: release builds do not check overlaps.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_serialized_overlap
// mpi-test: np=2

use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrompi::{Error, Mpi, PersistentRequest, Request, ThreadLevel};

mod common;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    common::check(
        &world,
        mpi.thread_level() == ThreadLevel::Serialized,
        "init_thread provided Serialized",
    );
    world.barrier().expect("barrier A");

    if rank == 1 {
        for _ in 0..3 {
            world.send(&[1i32], 0, 98).expect("send tag 98");
        }
        std::thread::sleep(Duration::from_secs(2));
        world.send(&[2i32], 0, 99).expect("send tag 99");

        common::check(
            &world,
            true,
            "overlap-rejected calls leave request state unchanged",
        );
        world.barrier().expect("barrier B");
        return;
    }

    let mut buf_a = [0i32; 1];
    let mut buf_b = [0i32; 1];
    let mut buf_c = [0i32; 1];

    let mut a = world.recv_init(&mut buf_a, 1, 98).expect("recv_init a");
    a.start().expect("start a");
    let mut b = world.recv_init(&mut buf_b, 1, 98).expect("recv_init b");
    let flag = AtomicBool::new(false);
    let mut ok = true;

    ferrompi::scope(|ms| {
        let mut reqs = vec![world.irecv(ms, &mut buf_c, 1, 98).expect("irecv")];

        std::thread::scope(|s| {
            let world_ref = &world;
            let flag_ref = &flag;
            let handle = s.spawn(move || {
                flag_ref.store(true, Ordering::SeqCst);
                world_ref.recv(&mut [0i32; 1], 1, 99)
            });

            while !flag.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(200));

            ok &= matches!(a.wait(), Err(Error::ThreadLevelViolation));
            ok &= a.is_active();
            ok &= matches!(a.test(), Err(Error::ThreadLevelViolation));
            ok &= a.is_active();

            ok &= matches!(
                PersistentRequest::wait_all(slice::from_mut(&mut a)),
                Err(Error::ThreadLevelViolation)
            );
            ok &= a.is_active();

            ok &= matches!(b.start(), Err(Error::ThreadLevelViolation));
            ok &= !b.is_active();
            ok &= matches!(
                PersistentRequest::start_all(slice::from_mut(&mut b)),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !b.is_active();

            ok &= matches!(reqs[0].test(), Err(Error::ThreadLevelViolation));
            ok &= !reqs[0].is_completed();
            ok &= matches!(
                Request::wait_all(&mut reqs),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !reqs[0].is_completed();
            ok &= matches!(
                Request::wait_any(&mut reqs),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !reqs[0].is_completed();
            ok &= matches!(
                Request::wait_some(&mut reqs),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !reqs[0].is_completed();
            ok &= matches!(
                Request::test_any(&mut reqs),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !reqs[0].is_completed();
            ok &= matches!(
                Request::test_some(&mut reqs),
                Err(Error::ThreadLevelViolation)
            );
            ok &= !reqs[0].is_completed();

            let recv_result = handle.join().expect("worker thread panicked");
            ok &= recv_result.is_ok();
        });

        ok &= a.wait().is_ok();
        ok &= !a.is_active();
        ok &= Request::wait_all(&mut reqs).is_ok();
        ok &= reqs[0].is_completed();
        Ok(())
    })
    .expect("scope failed");

    b.start().expect("start b (restart)");
    ok &= b.wait().is_ok();

    common::check(
        &world,
        ok,
        "overlap-rejected calls leave request state unchanged",
    );
    println!("test_serialized_overlap: PASS");
    world.barrier().expect("barrier B");
}
