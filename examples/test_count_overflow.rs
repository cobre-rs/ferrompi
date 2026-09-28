//! Regression test for the `count > INT_MAX` guard across the 30 scalar-count
//! shims (blocking/nonblocking p2p and collectives, plus the user-op
//! reduction).
//!
//! On MPI < 4, a scalar count above `INT_MAX` must return
//! `Err(Error::Mpi { class: MpiErrorClass::Count, .. })` before any MPI
//! function or buffer access, instead of silently truncating to a 16-byte
//! transfer. On MPI >= 4 the same count takes the `_c` large-count path
//! instead (covered by the MPICH large-count job), so this test SKIPs
//! before allocating the two 4 GiB buffers a full run would otherwise need.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_count_overflow
// mpi-test: np=2.. skip-ok=mpich

use ferrompi::{Error, Mpi, MpiErrorClass, ReduceOp, Result, UserOp};

mod common;

const N: usize = (1 << 32) + 16;

/// True iff `r` is `Err(Error::Mpi { class: MpiErrorClass::Count, .. })`.
fn is_count<T>(r: &Result<T>) -> bool {
    matches!(
        r,
        Err(Error::Mpi {
            class: MpiErrorClass::Count,
            ..
        })
    )
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    if common::mpi_major() >= 4 {
        common::skip(&world, "MPI >= 4 takes the _c large-count path");
        return;
    }

    // Zeroed allocations whose pages the fixed shim never touches: every
    // guard below must fire before any MPI call reads or writes them.
    let send = vec![0u8; N];
    let mut recv = vec![0u8; N];

    let mut send_recv_ok = true;
    if rank == 0 {
        send_recv_ok = is_count(&world.send(&send, 1, 7));
    } else if rank == 1 {
        send_recv_ok = is_count(&world.recv(&mut recv, 0, 7));
    }
    common::check(
        &world,
        send_recv_ok,
        "send/recv of 2^32+16 bytes returns Count",
    );

    let broadcast_ok = is_count(&world.broadcast(&mut recv, 0));
    common::check(
        &world,
        broadcast_ok,
        "broadcast of 2^32+16 bytes returns Count",
    );

    let allreduce_ok = is_count(&world.allreduce(&send, &mut recv, ReduceOp::Max));
    common::check(
        &world,
        allreduce_ok,
        "allreduce of 2^32+16 bytes returns Count",
    );

    let max_op: UserOp<u8> = UserOp::new(|invec: &[u8], inoutvec: &mut [u8]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = (*x).max(*y);
        }
    })
    .expect("UserOp::new failed");
    let allreduce_with_op_ok = is_count(&world.allreduce_with_op(&send, &mut recv, &max_op));
    common::check(
        &world,
        allreduce_with_op_ok,
        "allreduce_with_op of 2^32+16 bytes returns Count",
    );

    // A red run's ferrompi_ibcast truncates the count and actually issues a
    // real (16-byte) MPI_Ibcast, so `Ok` must still be drained via `wait()`
    // before counting the failure, rather than leaking the request.
    let ibcast_result = world.ibroadcast(&mut recv, 0);
    let ibroadcast_ok = if is_count(&ibcast_result) {
        true
    } else {
        if let Ok(req) = ibcast_result {
            let _ = req.wait();
        }
        false
    };
    common::check(
        &world,
        ibroadcast_ok,
        "ibroadcast of 2^32+16 bytes returns Count",
    );

    if rank == 0 {
        println!("PASS: test_count_overflow");
    }
}
