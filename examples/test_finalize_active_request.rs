//! Regression for `MPI_Finalize`'s request sweep, and for the token that every
//! completion path releases.
//!
//! Each rank creates:
//! - an idle persistent send (`send_init`, never started): inactive, freed
//!   by the sweep;
//! - a nonblocking collective (`iallreduce`) in a scope: the scope completes
//!   it before `Mpi` is dropped;
//! - three matched `send_init`/`recv_init` pairs with its peer rank, one pair
//!   per completion path, each using distinct tags:
//!   - `start` then `wait` on each side individually;
//!   - `start_all` then `wait_all` on the pair together;
//!   - `start` then a `test` loop on each side until it reports complete.
//!
//! Every pair is pairwise-matched (rank 0 and rank 1 run the same code
//! against each other), so nothing deadlocks, and every request is completed
//! before `Mpi` is dropped.
//!
//! `Mpi` is then dropped with every request handle still alive. Completing a
//! request releases the token it holds, so no persistent request is active,
//! and the drop finalizes MPI: the runner fails the example if it prints the
//! skip marker. The finalize sweep frees every inactive persistent request. A
//! request left active at the drop skips `MPI_Finalize` instead;
//! `test_persistent_finalize` covers that. Dropping the remaining request
//! handles after `Mpi` makes no further MPI call.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_finalize_active_request
// mpi-test: np=2

use ferrompi::{Mpi, PersistentRequest, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let peer = 1 - rank;

    let idle = world
        .send_init(vec![7i32], peer, 1)
        .expect("send_init failed");

    let allreduce_send = [1.0f64];
    let mut allreduce_recv = [0.0f64; 1];
    ferrompi::scope(|s| {
        world.iallreduce(s, &allreduce_send, &mut allreduce_recv, ReduceOp::Sum)?;
        Ok(())
    })
    .expect("iallreduce failed");

    // Path A: start + wait on each side -> completes, both freed.
    let mut send_a = world.send_init(vec![10i32], peer, 10).expect("send_init a");
    let mut recv_a = world.recv_init(vec![0i32], peer, 10).expect("recv_init a");
    send_a.start().expect("start a (send)");
    recv_a.start().expect("start a (recv)");
    send_a.wait().expect("wait a (send)");
    recv_a.wait().expect("wait a (recv)");

    // Path B: start_all + wait_all on the pair -> completes, both freed.
    let send_b = world.send_init(vec![11i32], peer, 11).expect("send_init b");
    let recv_b = world.recv_init(vec![0i32], peer, 11).expect("recv_init b");
    let mut pair_b = [send_b, recv_b];
    PersistentRequest::start_all(&mut pair_b).expect("start_all b");
    PersistentRequest::wait_all(&mut pair_b).expect("wait_all b");

    // Path C: start + test loop on each side until complete -> both freed.
    let mut send_c = world.send_init(vec![12i32], peer, 12).expect("send_init c");
    let mut recv_c = world.recv_init(vec![0i32], peer, 12).expect("recv_init c");
    send_c.start().expect("start c (send)");
    recv_c.start().expect("start c (recv)");
    while send_c.test().expect("test c (send)").is_none() {}
    while recv_c.test().expect("test c (recv)").is_none() {}

    drop(mpi);

    drop(idle);
    drop(send_a);
    drop(recv_a);
    drop(pair_b);
    drop(send_c);
    drop(recv_c);

    println!("PASS: test_finalize_active_request");
}
