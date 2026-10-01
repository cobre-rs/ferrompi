//! Regression for the debug-only Serialized overlap check on a consuming
//! `Request::wait`: a call it rejects because it overlaps another thread's
//! in-flight MPI call cannot run and cannot hand the request back, so it
//! prints a message and aborts the process. Needs a debug build: release
//! builds do not check overlaps.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_request_wait_overlap
// mpi-test: np=1 expect=abort
// mpi-test-stderr: ferrompi: Request::wait overlapped another thread's MPI call

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ferrompi::{Mpi, ThreadLevel};

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Serialized).expect("MPI init failed");
    let world = mpi.world();
    assert_eq!(mpi.thread_level(), ThreadLevel::Serialized);

    let mut buf = [0i32; 1];
    let req = world.irecv(&mut buf, 0, 98).expect("irecv");

    let flag = AtomicBool::new(false);

    std::thread::scope(|s| {
        let world_ref = &world;
        let flag_ref = &flag;
        s.spawn(move || {
            flag_ref.store(true, Ordering::SeqCst);
            let _ = world_ref.recv(&mut [0i32; 1], 0, 99);
        });

        while !flag.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(200));

        let r = req.wait();
        println!("FAIL: Request::wait returned under an overlap: {r:?}");
        std::process::exit(1);
    });
}
