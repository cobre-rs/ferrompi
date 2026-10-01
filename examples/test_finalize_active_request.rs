//! Regression for `MPI_Finalize`'s request sweep freeing an active request.
//!
//! Each rank creates:
//! - an idle persistent send (`send_init`, never started): inactive, freed
//!   by the sweep, not counted;
//! - an active nonblocking collective (`iallreduce`, never waited): active,
//!   counted;
//! - five matched `send_init`/`recv_init` pairs with its peer rank, one pair
//!   per completion path:
//!   - `start` then `wait` on each side individually: completes, both
//!     freed, not counted;
//!   - `start_all` then `wait_all` on the pair together: completes, both
//!     freed, not counted;
//!   - `start` then a `test` loop on each side until it reports complete:
//!     completes, both freed, not counted;
//!   - `start`, never completed: both sides stay active, both counted;
//!   - `start_all`, never completed: both sides stay active, both counted.
//!
//! Every pair is pairwise-matched (rank 0 and rank 1 run the same code
//! against each other, using distinct tags per path), so nothing deadlocks
//! and the never-completed pairs really stay pending at `Mpi::drop` without
//! blocking it.
//!
//! `Mpi` is then dropped. The finalize sweep frees every inactive
//! persistent request without counting it, and leaves every active request
//! alone, counting it: the iallreduce plus the two never-completed pairs
//! (1 + 2 + 2 = 5). Dropping the remaining request handles after `Mpi`
//! makes no further MPI call.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_finalize_active_request
// mpi-test: np=2
// mpi-test-stderr: ferrompi: MPI_Finalize leaves 5 active request(s) unfreed

use ferrompi::{Mpi, PersistentRequest, ReduceOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let peer = 1 - rank;

    let idle = world.send_init(&[7i32], peer, 1).expect("send_init failed");

    let mut allreduce_recv = [0.0f64; 1];
    let active = world
        .iallreduce(&[1.0f64], &mut allreduce_recv, ReduceOp::Sum)
        .expect("iallreduce failed");

    // Path A: start + wait on each side -> completes, both freed.
    let send_buf_a = [10i32];
    let mut recv_buf_a = [0i32];
    let mut send_a = world.send_init(&send_buf_a, peer, 10).expect("send_init a");
    let mut recv_a = world
        .recv_init(&mut recv_buf_a, peer, 10)
        .expect("recv_init a");
    send_a.start().expect("start a (send)");
    recv_a.start().expect("start a (recv)");
    send_a.wait().expect("wait a (send)");
    recv_a.wait().expect("wait a (recv)");

    // Path B: start_all + wait_all on the pair -> completes, both freed.
    let send_buf_b = [11i32];
    let mut recv_buf_b = [0i32];
    let send_b = world.send_init(&send_buf_b, peer, 11).expect("send_init b");
    let recv_b = world
        .recv_init(&mut recv_buf_b, peer, 11)
        .expect("recv_init b");
    let mut pair_b = [send_b, recv_b];
    PersistentRequest::start_all(&mut pair_b).expect("start_all b");
    PersistentRequest::wait_all(&mut pair_b).expect("wait_all b");

    // Path C: start + test loop on each side until complete -> both freed.
    let send_buf_c = [12i32];
    let mut recv_buf_c = [0i32];
    let mut send_c = world.send_init(&send_buf_c, peer, 12).expect("send_init c");
    let mut recv_c = world
        .recv_init(&mut recv_buf_c, peer, 12)
        .expect("recv_init c");
    send_c.start().expect("start c (send)");
    recv_c.start().expect("start c (recv)");
    while !send_c.test().expect("test c (send)") {}
    while !recv_c.test().expect("test c (recv)") {}

    // Path D: start, never completed -> both sides stay active, both counted.
    let send_buf_d = [13i32];
    let mut recv_buf_d = [0i32];
    let mut send_d = world.send_init(&send_buf_d, peer, 13).expect("send_init d");
    let mut recv_d = world
        .recv_init(&mut recv_buf_d, peer, 13)
        .expect("recv_init d");
    send_d.start().expect("start d (send)");
    recv_d.start().expect("start d (recv)");

    // Path E: start_all, never completed -> both sides stay active, both counted.
    let send_buf_e = [14i32];
    let mut recv_buf_e = [0i32];
    let send_e = world.send_init(&send_buf_e, peer, 14).expect("send_init e");
    let recv_e = world
        .recv_init(&mut recv_buf_e, peer, 14)
        .expect("recv_init e");
    let mut pair_e = [send_e, recv_e];
    PersistentRequest::start_all(&mut pair_e).expect("start_all e");

    drop(mpi);

    drop(idle);
    drop(active);
    drop(send_a);
    drop(recv_a);
    drop(pair_b);
    drop(send_c);
    drop(recv_c);
    drop(send_d);
    drop(recv_d);
    drop(pair_e);

    println!("PASS: test_finalize_active_request");
}
