//! Regression test for the `count > INT_MAX` guard across the 30 scalar-count
//! shims (blocking/nonblocking p2p and collectives, plus the user-op
//! reduction), the 6 v-collective shims, and the 7 RMA shims.
//!
//! On MPI < 4, a scalar count above `INT_MAX` must return
//! `Err(Error::Mpi { class: MpiErrorClass::Count, .. })` before any MPI
//! function or buffer access, instead of silently truncating to a 16-byte
//! transfer. On MPI >= 4 the same count takes the `_c` large-count path
//! instead (covered by the MPICH large-count job), so this test SKIPs
//! before allocating the two 4 GiB buffers a full run would otherwise need.
//!
//! V-collectives have no `_c` path on any MPI version, so they are checked
//! before that skip. RMA is exercised through a raw `extern "C"` call to
//! `ferrompi_put`, not the safe `Win::put` API: the window's bounds check
//! rejects a mismatched `target_count` in Rust before the C guard is ever
//! reached, so only a direct C call can observe it — and only on MPI < 4,
//! since on MPI >= 4 the fixed shim would hand the oversized count to
//! `MPI_Put_c` over a four-element window.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_count_overflow
// mpi-test: np=2.. skip-ok=mpich

use ferrompi::{Communicator, Mpi, ReduceOp, UserOp};
#[cfg(feature = "rma")]
use ferrompi::{Error, MpiDatatype, MpiErrorClass, Win, WinFenceAssert};

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

    let igatherv_result = world.igatherv(&send, &mut recv, &counts, &displs, 0);
    let igatherv_ok = if common::is_count(&igatherv_result) {
        true
    } else {
        if let Ok(req) = igatherv_result {
            let _ = req.wait();
        }
        false
    };
    common::check(
        world,
        igatherv_ok,
        "igatherv with a 2^32+16-byte send returns Count",
    );
}

// Raw FFI declaration for the C-side RMA shim under test, in the same shape
// as examples/test_waitall_count_overflow.rs. Declared here (rather than via
// ferrompi::ffi, which is pub(crate)) so the example can call it directly
// without an API extension on the Rust side.
//
// SAFETY invariant for the call in rma_put below: the C guard rejects
// target_count before MPI_Put/MPI_Put_c is called, so `origin` is never
// read by MPI and need only be valid for the duration of the call.
#[cfg(feature = "rma")]
#[allow(dead_code)]
extern "C" {
    fn ferrompi_put(
        origin: *const std::ffi::c_void,
        origin_count: i64,
        origin_dt_tag: i32,
        target_rank: i32,
        target_disp: i64,
        target_count: i64,
        target_dt_tag: i32,
        win_handle: i32,
    ) -> std::ffi::c_int;
}

/// Calls the C `ferrompi_put` shim directly with a `target_count` above
/// `INT_MAX` over a real 4-element window, bypassing `Win::put`'s bounds
/// check so the C-side guard is reachable. Only reachable on MPI < 4: the
/// fixed shim on MPI >= 4 would pass the oversized count to `MPI_Put_c`.
#[cfg(feature = "rma")]
fn rma_put(world: &Communicator) {
    let win = Win::<u64>::allocate(world, 4).expect("Win::allocate failed");
    win.fence(WinFenceAssert::default())
        .expect("opening fence failed");

    let mut rma_put_ok = true;
    if world.rank() == 0 {
        let origin = [0u64; 4];
        let dt_tag = <u64 as MpiDatatype>::TAG as i32;
        let raw_ret = unsafe {
            // SAFETY: see the invariant comment on the extern "C" block above.
            ferrompi_put(
                origin.as_ptr().cast(),
                4,
                dt_tag,
                1,
                0,
                (1i64 << 32) + 4,
                dt_tag,
                win.raw_handle(),
            )
        };
        rma_put_ok = matches!(
            Error::from_code(raw_ret),
            Error::Mpi {
                class: MpiErrorClass::Count,
                ..
            }
        );
    }
    win.fence(WinFenceAssert::default())
        .expect("closing fence failed");
    drop(win);

    common::check(
        world,
        rma_put_ok,
        "put with a 2^32+4 target count returns Count",
    );
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    v_collectives(&world);

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

    let max_op: UserOp<u8> = UserOp::new(|invec: &[u8], inoutvec: &mut [u8]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = (*x).max(*y);
        }
    })
    .expect("UserOp::new failed");
    let allreduce_with_op_ok =
        common::is_count(&world.allreduce_with_op(&send, &mut recv, &max_op));
    common::check(
        &world,
        allreduce_with_op_ok,
        "allreduce_with_op of 2^32+16 bytes returns Count",
    );

    // A red run's ferrompi_ibcast truncates the count and actually issues a
    // real (16-byte) MPI_Ibcast, so `Ok` must still be drained via `wait()`
    // before counting the failure, rather than leaking the request.
    let ibcast_result = world.ibroadcast(&mut recv, 0);
    let ibroadcast_ok = if common::is_count(&ibcast_result) {
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

    #[cfg(feature = "rma")]
    rma_put(&world);

    if rank == 0 {
        println!("PASS: test_count_overflow");
    }
}
