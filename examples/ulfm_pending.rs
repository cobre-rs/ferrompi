//! Regression example for a receive left pending by a process failure
//! (`MPI_ERR_PROC_FAILED_PENDING`).
//!
//! Rank 2 calls `std::process::abort()`. Rank 1 posts a wildcard-source
//! receive and runs the requested completion mode on it; that call must
//! abort the process instead of reporting the receive complete while MPI
//! still owns its buffer. Rank 0 stays alive for a second so the only
//! failure rank 1 can observe is rank 2's, then exits without calling
//! `MPI_Finalize`: that call is an ordinary collective, not a ULFM-aware
//! one, so if it starts before rank 1 has detected its own failure (or
//! once rank 1 is already dead) it can block forever waiting for a
//! participant that will never arrive.
//!
//! No `// mpi-test:` directive and no `test_` prefix, so the default runner
//! never picks this up; a dedicated CI step runs it directly. Needs
//! fault-tolerant MPI (Open MPI 5's ULFM mode); run it manually:
//!
//! ```text
//! cargo build --example ulfm_pending
//! mpiexec --with-ft ulfm -n 3 target/debug/examples/ulfm_pending <wait|wait_all|wait_any>
//! ```

use ferrompi::{Mpi, Request};
use std::time::Duration;

fn main() {
    let mode = std::env::args()
        .nth(1)
        .expect("usage: ulfm_pending <wait|wait_all|wait_any>");

    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    assert_eq!(world.size(), 3, "ulfm_pending requires exactly 3 processes");

    world.barrier().expect("barrier failed");

    match world.rank() {
        0 => {
            std::thread::sleep(Duration::from_secs(1));
            std::process::exit(0);
        }
        1 => {
            let mut buf = [0i32; 1];
            let req = world.irecv(&mut buf, -1, 7).expect("irecv failed");

            match mode.as_str() {
                "wait" => {
                    let result = req.wait();
                    println!(
                        "FAIL: {mode} returned while the receive is still pending: {result:?}"
                    );
                    std::process::exit(1);
                }
                "wait_all" => {
                    let mut reqs = vec![req];
                    let result = Request::wait_all(&mut reqs);
                    let completed = reqs[0].is_completed();
                    println!(
                        "FAIL: {mode} returned while the receive is still pending: {result:?}; completed={completed}"
                    );
                    std::mem::forget(reqs);
                    std::process::exit(1);
                }
                "wait_any" => {
                    let mut reqs = vec![req];
                    let result = Request::wait_any(&mut reqs);
                    let completed = reqs[0].is_completed();
                    println!(
                        "FAIL: {mode} returned while the receive is still pending: {result:?}; completed={completed}"
                    );
                    std::mem::forget(reqs);
                    std::process::exit(1);
                }
                other => panic!("unknown mode: {other}"),
            }
        }
        2 => std::process::abort(),
        _ => unreachable!(),
    }
}
