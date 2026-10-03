//! Dropping an active persistent receive waits for the message before it frees
//! the buffer.
//!
//! Rank 0 starts a persistent receive, tells rank 1 to go, and drops the
//! request while it is active. Rank 1 sleeps 200 ms after the go signal and
//! only then sends the matching message, so the data arrives while `Drop`'s
//! `MPI_Wait` is in progress, not before the drop (MPI-4.1 §3.9). A `Drop`
//! that freed the buffer first would let MPI write into freed memory, which
//! valgrind reports as an invalid write (`tests/run_mpi_tests.sh --valgrind`
//! exits 99). A `Drop` that did not wait at all would return at once, which
//! the elapsed time on rank 0 shows without valgrind.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_drop_order
// mpi-test: np=2 valgrind

use std::time::{Duration, Instant};

use ferrompi::Mpi;

mod common;

const DELAY: Duration = Duration::from_millis(200);

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    let waited = if world.rank() == 0 {
        let mut recv = world
            .recv_init(vec![0u64; 64], 1, 7)
            .expect("recv_init failed");
        recv.start().expect("start failed");
        world.send(&[1u8], 1, 8).expect("go signal failed");
        let started = Instant::now();
        drop(recv);
        started.elapsed() >= DELAY / 2
    } else {
        let mut go = [0u8; 1];
        world.recv(&mut go, 0, 8).expect("go recv failed");
        std::thread::sleep(DELAY);
        world
            .send(&[0x5151_5151_5151_5151u64; 64], 0, 7)
            .expect("send failed");
        true
    };

    common::check(
        &world,
        waited,
        "the drop of an active receive waits for the message",
    );
    if world.rank() == 0 {
        println!("test_persistent_drop_order: PASS");
    }
}
