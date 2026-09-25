//! Integration test for persistent buffered send (`bsend_init`) with buffer
//! attach / detach.
//!
//! Buffered sends copy the outgoing message into a user-attached buffer and
//! complete at the local side immediately, without waiting for the receiver
//! to post a matching receive.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_bsend
// mpi-test: np=2

use ferrompi::{Mpi, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_bsend requires exactly 2 processes, got {size}"
    );

    const N: usize = 100;
    const ITERS: usize = 10;
    const TAG: i32 = 42;

    // Attach a 64 KiB buffer on every rank (MPI_Buffer_attach is per-process;
    // only rank 0 will actually use it for sending, but attaching on all ranks
    // keeps the code symmetric and avoids implementation-specific issues).
    mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice())
        .expect("buffer_attach failed");

    if rank == 0 {
        // Rank 0: initialize a persistent buffered send request, loop 10 times,
        // each time filling the send buffer with a per-iteration value.
        let mut send = vec![0.0f64; N];
        let mut send_req = world
            .bsend_init(&send, 1, TAG)
            .expect("bsend_init failed on rank 0");

        // Synchronize both ranks past the init phase before any start().
        // Required for OpenMPI 4.x compatibility — without this, rank 0's
        // first start() may issue before rank 1 has finished recv_init.
        world.barrier().expect("post-init barrier failed on rank 0");

        for iter in 0..ITERS {
            let value = (iter + 1) as f64;
            send.fill(value);
            send_req.start().expect("bsend start failed");
            send_req.wait().expect("bsend wait failed");
        }
    } else {
        // Rank 1: initialize a persistent receive request, loop 10 times,
        // each time verifying the received values.
        let mut recv = vec![0.0f64; N];
        let mut recv_req = world
            .recv_init(&mut recv, 0, TAG)
            .expect("recv_init failed on rank 1");

        // See companion comment on rank 0.
        world.barrier().expect("post-init barrier failed on rank 1");

        for iter in 0..ITERS {
            let expected = (iter + 1) as f64;
            recv_req.start().expect("recv start failed");
            recv_req.wait().expect("recv wait failed");

            for (i, &v) in recv.iter().enumerate() {
                assert!(
                    (v - expected).abs() < f64::EPSILON,
                    "iter {iter}: recv[{i}] = {v}, expected {expected}"
                );
            }
        }
    }

    // Detach the buffer on every rank (blocks until all buffered sends using
    // it have completed — rank 0's sends are already done at this point).
    let _buf = mpi.buffer_detach().expect("buffer_detach failed");
    // _buf is dropped here; the allocation is freed.

    // Sentinel allreduce to confirm both ranks completed without error.
    let local: i64 = 1;
    let total = world
        .allreduce_scalar(local, ReduceOp::Sum)
        .expect("sentinel allreduce failed");
    assert_eq!(
        total, 2,
        "sentinel allreduce: expected 2 (both ranks), got {total}"
    );

    if rank == 0 {
        println!("PASS: bsend_init with attached buffer");
    }
}
