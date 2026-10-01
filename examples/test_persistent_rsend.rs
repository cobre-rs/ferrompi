//! Integration test for persistent ready-mode send (`rsend_init`).
//!
//! Ready-mode sends require the matching receive to be posted on the destination
//! rank **before** `start()` is called. This test enforces that ordering with an
//! explicit barrier: rank 1 posts its `recv_init` and calls `start()`, then all
//! ranks synchronize via `barrier()`, then rank 0 calls `rsend_init` + `start()`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_rsend
// mpi-test: np=2

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_rsend requires exactly 2 processes, got {size}"
    );

    const N: usize = 10;
    const TAG: i32 = 7;

    if rank == 1 {
        // Rank 1: post the persistent receive and start it BEFORE the barrier.
        // This guarantees the receive is active when rank 0's rsend start() fires.
        let mut recv = vec![0.0f64; N];
        let mut recv_req = world
            .recv_init(&mut recv, 0, TAG)
            .expect("recv_init failed on rank 1");
        recv_req.start().expect("recv start failed on rank 1");

        // Barrier: signals rank 0 that the receive is already posted.
        world.barrier().expect("barrier failed on rank 1");

        recv_req.wait().expect("recv wait failed on rank 1");

        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - 42.0f64).abs() < f64::EPSILON,
                "rank 1: recv[{i}] = {v}, expected 42.0"
            );
        }
    } else {
        // Rank 0: wait for the barrier (guarantees rank 1's recv is posted),
        // then initialize and start the ready-mode persistent send.
        world.barrier().expect("barrier failed on rank 0");

        let send = vec![42.0f64; N];
        let mut send_req = world
            .rsend_init(&send, 1, TAG)
            .expect("rsend_init failed on rank 0");
        send_req.start().expect("rsend start failed on rank 0");
        send_req.wait().expect("rsend wait failed on rank 0");
    }

    // Sentinel allreduce to confirm both ranks reached this point cleanly.
    let local: i64 = 1;
    let total = world
        .allreduce_scalar(local, ferrompi::ReduceOp::Sum)
        .expect("sentinel allreduce failed");
    assert_eq!(
        total, 2,
        "sentinel allreduce: expected 2 (both ranks), got {total}"
    );

    if rank == 0 {
        println!("PASS: rsend_init with pre-posted recv");
    }
}
