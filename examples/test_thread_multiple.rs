//! Control for `test_thread_funneled`: at `ThreadLevel::Multiple`, calls from
//! any thread are legal and must not be rejected by the thread-level guard.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_thread_multiple
// mpi-test: np=1

use ferrompi::{Mpi, ThreadLevel};

mod common;

const NUM_THREADS: usize = 4;
const ITERS: usize = 1000;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");
    let world = mpi.world();

    common::check(
        &world,
        mpi.thread_level() == ThreadLevel::Multiple,
        "init_thread provided Multiple",
    );

    let all_ok = std::thread::scope(|s| {
        let world_ref = &world;
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|t| {
                s.spawn(move || {
                    let tag = t as i32;
                    for i in 0..ITERS {
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

    common::check(&world, all_ok, "concurrent MPI calls at Multiple succeed");

    println!("test_thread_multiple: PASS");
}
