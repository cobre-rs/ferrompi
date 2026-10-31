//! Integration test for persistent P2P send_init and recv_init (MPI 1.1+).
//!
//! Verifies that `Communicator::send_init` and `Communicator::recv_init` produce
//! correct results over 100 iterations and that `PersistentRequest` can be reused
//! without leaking MPI state. Every `wait` also returns the right `Status`: the
//! receiver's reports the matched source, tag and element count on each of the
//! 100 starts, and the sender's is the empty status.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_p2p
// mpi-test: np=2

use ferrompi::{Mpi, Source, Tag};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_p2p requires exactly 2 processes, got {size}"
    );

    const N: usize = 100;
    const ITERS: usize = 100;
    const TAG: i32 = 7;

    let mut sends_empty = true;
    let mut recvs_matched = true;

    if rank == 0 {
        // Rank 0: initialize a persistent send for a 100-element f64 buffer.
        let mut req = world
            .send_init(vec![0.0f64; N], 1, TAG)
            .expect("send_init failed on rank 0");

        for iter in 0..ITERS {
            // Fill send buffer: send[i] = iter * N + i
            let send = req.buffer_mut().expect("buffer_mut failed on rank 0");
            for (i, x) in send.iter_mut().enumerate() {
                *x = (iter * N + i) as f64;
            }
            req.start().expect("start failed on rank 0");
            let st = req.wait().expect("wait failed on rank 0");
            sends_empty &= common::is_empty_status(&st);
        }

        // Drop the request to exercise PersistentRequest::Drop cleanly.
        drop(req);
    } else {
        // Rank 1: initialize a persistent recv for a 100-element f64 buffer.
        let mut req = world
            .recv_init(vec![0.0f64; N], 0, TAG)
            .expect("recv_init failed on rank 1");

        for iter in 0..ITERS {
            req.start().expect("start failed on rank 1");
            let st = req.wait().expect("wait failed on rank 1");
            recvs_matched &= st.source == Source::Rank(0)
                && st.tag == Tag::Value(TAG)
                && st.count == Some(N)
                && st.error.is_none();

            // Verify: recv[i] must equal iter * N + i
            let recv = req.buffer().expect("buffer failed on rank 1");
            for (i, &v) in recv.iter().enumerate() {
                let expected = (iter * N + i) as f64;
                assert!(
                    (v - expected).abs() < f64::EPSILON,
                    "rank 1: iter={iter} recv[{i}] = {v}, expected {expected}"
                );
            }
        }

        // Drop the request to exercise PersistentRequest::Drop cleanly.
        drop(req);
    }

    common::check(
        &world,
        sends_empty,
        "every persistent send wait returns the empty status",
    );
    common::check(
        &world,
        recvs_matched,
        "every persistent receive wait reports the sender, tag and count",
    );

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
        println!("PASS: send_init/recv_init with {ITERS} iterations");
    }
}
