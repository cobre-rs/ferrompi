//! Release-mode regression: polling a peer's flag in a shared window.
//!
//! Rank 0 of the node stores 1 into a one-element shared window after a delay,
//! while every other rank on the node spins on that element through
//! `WinView::read` and then leaves the loop.
//!
//! A `&[u64]` over memory that another process writes lets the compiler load the
//! element once and spin on that value, so the loop never ends. The view's
//! `Relaxed` atomic load is reloaded on every iteration, so the loop ends once
//! the store is visible. A debug build reloads in both forms, so only an
//! optimised build tells them apart: the directive's `release` flag makes
//! `tests/run_mpi_tests.sh --release` run this example, and a hang is the
//! runner's `FAIL timeout`.
//!
//! The loop body holds no call, not even `std::hint::spin_loop`. A call the
//! compiler cannot see through could also stop it from hoisting the load, and a
//! regression back to a slice would then pass.
//!
//! In the unified memory model, which `remote` requires, a process can poll a
//! location that another process stores to (MPI-4.1 §12.7, rules U2 and U3,
//! p.608-609; Example 12.13, p.617). The final read sees 1 because a store,
//! once visible, stays (rule U3). The writer's `sync` after its store and the
//! reader's `sync` after the poll follow the advice to users after rule U5
//! (p.609).
//!
//! Rank 0 stores before its first MPI call after the barrier. The unlock that
//! closes its epoch can wait for the spinning ranks to make MPI progress
//! (MPI-4.1 §12.7.3, p.615-616; MPICH 4.2 does), and they make none until the
//! store takes them out of the loop. A store after such a call would deadlock,
//! as in Example 12.13 (p.617).
//!
//! Run with: mpiexec -n 2 ./target/release/examples/test_window_flag_poll

use std::time::Duration;

use ferrompi::{Mpi, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    let node = world.split_shared().expect("split_shared failed");
    assert!(
        node.size() >= 2,
        "test_window_flag_poll requires at least 2 processes on a node, got {}",
        node.size()
    );
    let mut win = Win::<u64>::allocate_shared(&node, 1).expect("allocate_shared failed");

    let last = win
        .lock_all(|e| {
            node.barrier()?;
            if node.rank() == 0 {
                std::thread::sleep(Duration::from_millis(300));
                e.local().write(0, 1);
                e.sync()?;
                Ok(e.local().read(0))
            } else {
                let flag = e.remote(0)?;
                while flag.read(0) == 0 {}
                e.sync()?;
                Ok(flag.read(0))
            }
        })
        .expect("lock_all with the flag poll failed");

    common::check(&world, last == 1, "every rank's last read of the flag is 1");
    if rank == 0 {
        println!("PASS: a peer's flag was observed through the window view");
    }
}
// mpi-test: np=2.. timeout=30 release
