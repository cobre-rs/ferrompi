//! Integration test for persistent P2P send_init and recv_init (MPI 1.1+).
//!
//! Verifies that `Communicator::send_init` and `Communicator::recv_init` produce
//! correct results over 100 iterations and that `PersistentRequest` can be reused
//! without leaking MPI state.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_p2p
// mpi-test: np=2

use ferrompi::Mpi;

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

    if rank == 0 {
        // Rank 0: initialize a persistent send for a 100-element f64 buffer.
        let mut send = vec![0.0f64; N];
        let mut req = world
            .send_init(&send, 1, TAG)
            .expect("send_init failed on rank 0");

        // Synchronize both ranks past the init phase before any start().
        // Required for OpenMPI 4.x compatibility — without this, rank 0's
        // first start() may issue before rank 1 has finished recv_init,
        // and OpenMPI's progress engine may deadlock under --btl=self,tcp
        // (MPICH 4.x progresses without the barrier).
        world.barrier().expect("post-init barrier failed on rank 0");

        for iter in 0..ITERS {
            // Fill send buffer: send[i] = iter * N + i
            for (i, x) in send.iter_mut().enumerate() {
                *x = (iter * N + i) as f64;
            }
            req.start().expect("start failed on rank 0");
            req.wait().expect("wait failed on rank 0");
        }

        // Drop the request to exercise PersistentRequest::Drop cleanly.
        drop(req);
    } else {
        // Rank 1: initialize a persistent recv for a 100-element f64 buffer.
        let mut recv = vec![0.0f64; N];
        let mut req = world
            .recv_init(&mut recv, 0, TAG)
            .expect("recv_init failed on rank 1");

        // See companion comment on rank 0.
        world.barrier().expect("post-init barrier failed on rank 1");

        for iter in 0..ITERS {
            req.start().expect("start failed on rank 1");
            req.wait().expect("wait failed on rank 1");

            // Verify: recv[i] must equal iter * N + i
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
