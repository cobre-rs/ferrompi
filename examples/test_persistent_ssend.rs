//! Integration test for persistent synchronous-mode send (`ssend_init`).
//!
//! Synchronous sends complete only after the matching receive has begun — there
//! is no internal buffering. To avoid a deadlock the receiver must post (and
//! start) its `recv_init` before or concurrently with the sender's `start()`.
//! This test achieves safe ordering by starting the receive first, then
//! starting the send, then waiting on the receive, then waiting on the send.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_ssend
// mpi-test: np=2

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_ssend requires exactly 2 processes, got {size}"
    );

    const N: usize = 5;
    const TAG: i32 = 3;
    const ITERS: usize = 5;

    if rank == 0 {
        let mut send = vec![0i32; N];
        let mut send_req = world
            .ssend_init(&send, 1, TAG)
            .expect("ssend_init failed on rank 0");

        // Synchronize both ranks past the init phase before any start().
        // Required for OpenMPI 4.x compatibility — ssend (synchronous send)
        // waits for matching recv, and without this barrier rank 0 may
        // start before rank 1 has finished recv_init.
        world.barrier().expect("post-init barrier failed on rank 0");

        for iter in 0..ITERS {
            // Fill the send buffer for this iteration.
            for (i, slot) in send.iter_mut().enumerate() {
                *slot = (iter * N + i) as i32;
            }

            // Start the synchronous send — it will complete only after rank 1
            // has started its matching receive.
            send_req.start().expect("ssend start failed on rank 0");

            // Wait for the send to complete (rank 0 unblocks when rank 1
            // calls recv_req.start()).
            send_req.wait().expect("ssend wait failed on rank 0");
        }
    } else {
        // Rank 1
        let mut recv = vec![0i32; N];
        let mut recv_req = world
            .recv_init(&mut recv, 0, TAG)
            .expect("recv_init failed on rank 1");

        // See companion comment on rank 0.
        world.barrier().expect("post-init barrier failed on rank 1");

        for iter in 0..ITERS {
            // Post the receive first so the synchronous send can complete.
            recv_req.start().expect("recv start failed on rank 1");
            recv_req.wait().expect("recv wait failed on rank 1");

            // Verify the payload.
            for (i, &val) in recv.iter().enumerate() {
                let expected = (iter * N + i) as i32;
                assert_eq!(
                    val, expected,
                    "rank 1 iter {iter}: recv[{i}] = {val}, expected {expected}",
                );
            }
        }
    }

    // Sentinel allreduce: confirms both ranks completed all iterations cleanly.
    let local: i64 = 1;
    let total = world
        .allreduce_scalar(local, ferrompi::ReduceOp::Sum)
        .expect("sentinel allreduce failed");
    assert_eq!(
        total, 2,
        "sentinel allreduce: expected 2 (both ranks), got {total}"
    );

    if rank == 0 {
        println!("PASS: ssend_init with {ITERS} iterations");
    }
}
