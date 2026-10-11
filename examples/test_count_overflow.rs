//! Regression test for the `count > INT_MAX` guard across the 30 scalar-count
//! shims (blocking/nonblocking p2p and collectives, plus every blocking,
//! nonblocking and persistent reduction with a user op), and the 6 v-collective
//! shims.
//!
//! On MPI < 4, a scalar count above `INT_MAX` must return
//! `Err(Error::Mpi { class: MpiErrorClass::Count, .. })` before any MPI
//! function or buffer access, instead of silently truncating to a 16-byte
//! transfer. On MPI >= 4 the same count takes the `_c` large-count path
//! instead (covered by the MPICH large-count job), so this test SKIPs
//! before allocating the two 4 GiB buffers a full run would otherwise need.
//!
//! V-collectives have no `_c` path on any MPI version, and a user op's classic
//! function takes an `int` length, so both are checked before that skip. A
//! persistent reduction without persistent collectives is refused with
//! `Error::NotSupported` instead.
//!
//! A user op in `reduce_scatter_block` reduces `recv.len() * size` elements, so
//! it is also refused when that span exceeds `INT_MAX`, though `recv.len()`
//! does not.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_count_overflow
// mpi-test: np=2.. skip-ok=mpich

use ferrompi::{Communicator, Error, Mpi, ReduceOp, UserOp};

mod common;

const N: usize = (1 << 32) + 16;

/// V-collectives have no `_c` large-count path: they must reject a scalar
/// send count above `INT_MAX` on every MPI version.
fn v_collectives(world: &Communicator) {
    let size = world.size();
    let send = vec![0u8; N];
    let counts = vec![16i32; size as usize];
    let displs: Vec<i32> = (0..size).map(|r| r * 16).collect();
    let mut recv = vec![0u8; 16 * size as usize];

    let allgatherv_ok = common::is_count(&world.allgatherv(&send, &mut recv, &counts, &displs));
    common::check(
        world,
        allgatherv_ok,
        "allgatherv with a 2^32+16-byte send returns Count",
    );

    let igatherv_ok = ferrompi::scope(|s| {
        let igatherv_result = world.igatherv(s, &send, &mut recv, &counts, &displs, 0);
        Ok(if common::is_count(&igatherv_result) {
            true
        } else {
            if let Ok(req) = igatherv_result {
                let _ = req.wait();
            }
            false
        })
    })
    .expect("scope for igatherv");
    common::check(
        world,
        igatherv_ok,
        "igatherv with a 2^32+16-byte send returns Count",
    );
}

/// True iff `r` is what a persistent reduction with a user op above `INT_MAX`
/// returns: `Count` on a library with persistent collectives, `NotSupported`
/// without them.
fn persistent_refused<T>(r: &ferrompi::Result<T>) -> bool {
    if common::has_mpi4_collectives() {
        common::is_count(r)
    } else {
        matches!(r, Err(Error::NotSupported(_)))
    }
}

/// A user op's classic function takes an `int` length: every blocking,
/// nonblocking and persistent reduction with a user op must reject a count
/// above `INT_MAX`, and `reduce_scatter_block` a reduced span above it.
fn user_op_reduction(world: &Communicator) {
    // Zeroed allocations whose pages the fixed shim never touches: the guard
    // must fire before any MPI call reads or writes them.
    let send = vec![0u8; N];
    let mut recv = vec![0u8; N];

    let max_op: UserOp<u8> = UserOp::new(|invec: &[u8], inoutvec: &mut [u8]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = (*x).max(*y);
        }
    })
    .expect("UserOp::new failed");

    let allreduce_ok = common::is_count(&world.allreduce(&send, &mut recv, &max_op));
    common::check(
        world,
        allreduce_ok,
        "allreduce of 2^32+16 bytes with a user op returns Count",
    );

    let reduce_ok = common::is_count(&world.reduce(&send, &mut recv, &max_op, 0));
    common::check(
        world,
        reduce_ok,
        "reduce of 2^32+16 bytes with a user op returns Count",
    );

    let scan_ok = common::is_count(&world.scan(&send, &mut recv, &max_op));
    common::check(
        world,
        scan_ok,
        "scan of 2^32+16 bytes with a user op returns Count",
    );

    let exscan_ok = common::is_count(&world.exscan(&send, &mut recv, &max_op));
    common::check(
        world,
        exscan_ok,
        "exscan of 2^32+16 bytes with a user op returns Count",
    );

    // Each refusal returns before registering a request, so the scope has
    // nothing to wait for.
    let iallreduce_ok = ferrompi::scope(|s| {
        Ok(common::is_count(
            &world.iallreduce(s, &send, &mut recv, &max_op),
        ))
    })
    .expect("scope failed");
    common::check(
        world,
        iallreduce_ok,
        "iallreduce of 2^32+16 bytes with a user op returns Count",
    );

    let ireduce_ok = ferrompi::scope(|s| {
        Ok(common::is_count(
            &world.ireduce(s, &send, &mut recv, &max_op, 0),
        ))
    })
    .expect("scope failed");
    common::check(
        world,
        ireduce_ok,
        "ireduce of 2^32+16 bytes with a user op returns Count",
    );

    let iscan_ok =
        ferrompi::scope(|s| Ok(common::is_count(&world.iscan(s, &send, &mut recv, &max_op))))
            .expect("scope failed");
    common::check(
        world,
        iscan_ok,
        "iscan of 2^32+16 bytes with a user op returns Count",
    );

    let iexscan_ok = ferrompi::scope(|s| {
        Ok(common::is_count(
            &world.iexscan(s, &send, &mut recv, &max_op),
        ))
    })
    .expect("scope failed");
    common::check(
        world,
        iexscan_ok,
        "iexscan of 2^32+16 bytes with a user op returns Count",
    );

    // One block is within INT_MAX, but the reduction covers one block per
    // rank, which is not.
    let block = i32::MAX as usize / world.size() as usize + 1;
    let span_send = vec![0u8; block * world.size() as usize];
    let mut span_recv = vec![0u8; block];

    let reduce_scatter_block_ok =
        common::is_count(&world.reduce_scatter_block(&span_send, &mut span_recv, &max_op));
    common::check(
        world,
        reduce_scatter_block_ok,
        "reduce_scatter_block with a user op and a span above INT_MAX returns Count",
    );

    let ireduce_scatter_block_ok = ferrompi::scope(|s| {
        Ok(common::is_count(&world.ireduce_scatter_block(
            s,
            &span_send,
            &mut span_recv,
            &max_op,
        )))
    })
    .expect("scope failed");
    common::check(
        world,
        ireduce_scatter_block_ok,
        "ireduce_scatter_block with a user op and a span above INT_MAX returns Count",
    );

    // A persistent reduction owns its buffers, so each call below moves in
    // fresh zeroed ones (a refusal never touches their pages and drops them).
    let reduce_scatter_block_init_ok =
        persistent_refused(&world.reduce_scatter_block_init(span_send, span_recv, &max_op));
    common::check(
        world,
        reduce_scatter_block_init_ok,
        "reduce_scatter_block_init with a user op and a span above INT_MAX returns Count",
    );

    let allreduce_init_ok =
        persistent_refused(&world.allreduce_init(vec![0u8; N], vec![0u8; N], &max_op));
    common::check(
        world,
        allreduce_init_ok,
        "allreduce_init of 2^32+16 bytes with a user op returns Count",
    );

    let reduce_init_ok =
        persistent_refused(&world.reduce_init(vec![0u8; N], vec![0u8; N], &max_op, 0));
    common::check(
        world,
        reduce_init_ok,
        "reduce_init of 2^32+16 bytes with a user op returns Count",
    );

    let scan_init_ok = persistent_refused(&world.scan_init(vec![0u8; N], vec![0u8; N], &max_op));
    common::check(
        world,
        scan_init_ok,
        "scan_init of 2^32+16 bytes with a user op returns Count",
    );

    let exscan_init_ok =
        persistent_refused(&world.exscan_init(vec![0u8; N], vec![0u8; N], &max_op));
    common::check(
        world,
        exscan_init_ok,
        "exscan_init of 2^32+16 bytes with a user op returns Count",
    );
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    v_collectives(&world);
    user_op_reduction(&world);

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
        send_recv_ok = common::is_count(&world.send(&send, 1, 7));
    } else if rank == 1 {
        send_recv_ok = common::is_count(&world.recv(&mut recv, 0, 7));
    }
    common::check(
        &world,
        send_recv_ok,
        "send/recv of 2^32+16 bytes returns Count",
    );

    let broadcast_ok = common::is_count(&world.broadcast(&mut recv, 0));
    common::check(
        &world,
        broadcast_ok,
        "broadcast of 2^32+16 bytes returns Count",
    );

    let allreduce_ok = common::is_count(&world.allreduce(&send, &mut recv, ReduceOp::Max));
    common::check(
        &world,
        allreduce_ok,
        "allreduce of 2^32+16 bytes returns Count",
    );

    // A red run's ferrompi_ibcast truncates the count and actually issues a
    // real (16-byte) MPI_Ibcast, so `Ok` must still be drained via `wait()`
    // before counting the failure, rather than leaking the request.
    let ibroadcast_ok = ferrompi::scope(|s| {
        let ibcast_result = world.ibroadcast(s, &mut recv, 0);
        Ok(if common::is_count(&ibcast_result) {
            true
        } else {
            if let Ok(req) = ibcast_result {
                let _ = req.wait();
            }
            false
        })
    })
    .expect("scope for ibroadcast");
    common::check(
        &world,
        ibroadcast_ok,
        "ibroadcast of 2^32+16 bytes returns Count",
    );

    if rank == 0 {
        println!("PASS: test_count_overflow");
    }
}
