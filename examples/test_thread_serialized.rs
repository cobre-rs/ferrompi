//! Regression for the debug-only Serialized overlap check: correctly
//! serialized MPI calls at `ThreadLevel::Serialized` must never be rejected
//! as a false positive.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_thread_serialized
// mpi-test: np=1

use std::sync::Mutex;

use ferrompi::{Mpi, ThreadLevel};

mod common;

const NUM_THREADS: usize = 4;
const ITERS: usize = 200;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");
    let world = mpi.world();

    common::check(
        &world,
        mpi.thread_level() == ThreadLevel::Serialized,
        "init_thread provided Serialized",
    );

    let call_lock = Mutex::new(());

    let all_ok = std::thread::scope(|s| {
        let world_ref = &world;
        let lock_ref = &call_lock;
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|t| {
                s.spawn(move || {
                    let tag = t as i32;
                    for i in 0..ITERS {
                        let _guard = lock_ref.lock().expect("mutex poisoned");
                        let send = [i as i32; 1];
                        let mut recv = [0i32; 1];
                        let req = world_ref.irecv(&mut recv, 0, tag)?;
                        world_ref.send(&send, 0, tag)?;
                        req.wait()?;
                        if recv[0] != i as i32 {
                            return Ok(false);
                        }
                    }
                    Ok::<bool, ferrompi::Error>(true)
                })
            })
            .collect();

        handles
            .into_iter()
            .all(|h| matches!(h.join().expect("worker thread panicked"), Ok(true)))
    });

    common::check(
        &world,
        all_ok,
        "correctly serialized calls at Serialized never rejected",
    );

    println!("test_thread_serialized: PASS");
}
