//! FFI overhead benchmark — characterizes the fixed per-call cost of the ferrompi/MPI FFI boundary
//! for five representative operations: `rank_cached`, `size_cached`, `barrier`,
//! `broadcast_1elem`, and `allreduce_1elem`.
//!
//! **MPI synchronization**: `rank_cached` and `size_cached` run only on rank 0 (no MPI calls).
//! For collectives, rank 0 drives Criterion and non-root ranks are kept in lockstep via a
//! sentinel `u64[1]` allreduce. See `benches/README.md` for context and output details.

use criterion::{black_box, Criterion};
use ferrompi::{Communicator, ReduceOp};
use std::time::Duration;

mod common;

/// Sentinel variant codes carried in the control allreduce.
const VARIANT_STOP: u64 = 0;
const VARIANT_BARRIER: u64 = 1;
const VARIANT_BROADCAST: u64 = 2;
const VARIANT_ALLREDUCE: u64 = 3;

// ─── Rank-0 benchmark driver ──────────────────────────────────────────────────

/// Register and drive all five benchmarks on rank 0.
///
/// The two local benches (`rank_cached`, `size_cached`) run without any MPI
/// coordination — they are pure field reads.
///
/// For each collective bench the function issues a sentinel allreduce inside
/// `b.iter` before the actual collective so non-root ranks stay in lockstep.
fn bench_ffi_overhead(c: &mut Criterion, world: &Communicator) {
    let mut group = c.benchmark_group("ffi_overhead");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(3));

    // ── rank_cached: field read, zero FFI ────────────────────────────────────
    // No sentinel needed — this call never enters the MPI library.
    group.bench_function("rank_cached", |b| {
        b.iter(|| black_box(world.rank()));
    });

    // ── size_cached: field read, zero FFI ────────────────────────────────────
    group.bench_function("size_cached", |b| {
        b.iter(|| black_box(world.size()));
    });

    // ── barrier: one FFI round-trip to MPI_Barrier ───────────────────────────
    group.bench_function("barrier", |b| {
        b.iter(|| {
            // Sentinel: tell follower ranks "run one barrier".
            let ctl_send = [VARIANT_BARRIER];
            let mut ctl_recv = [0u64; 1];
            world
                .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
                .unwrap();

            world.barrier().unwrap();
        });
    });

    // ── broadcast_1elem: MPI_Bcast on 1 × f64 ────────────────────────────────
    let mut bcast_buf = [0.0f64; 1];
    group.bench_function("broadcast_1elem", |b| {
        b.iter(|| {
            // Sentinel: tell follower ranks "run one broadcast".
            let ctl_send = [VARIANT_BROADCAST];
            let mut ctl_recv = [0u64; 1];
            world
                .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
                .unwrap();

            world.broadcast(&mut bcast_buf, 0).unwrap();
        });
    });

    // ── allreduce_1elem: MPI_Allreduce on 1 × f64 ────────────────────────────
    let send = [1.0f64; 1];
    let mut recv = [0.0f64; 1];
    group.bench_function("allreduce_1elem", |b| {
        b.iter(|| {
            // Sentinel: tell follower ranks "run one allreduce".
            let ctl_send = [VARIANT_ALLREDUCE];
            let mut ctl_recv = [0u64; 1];
            world
                .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
                .unwrap();

            world.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
        });
    });

    group.finish();
}

// ─── Non-root mirror loop ─────────────────────────────────────────────────────

/// Mirror loop for ranks > 0.
///
/// Loops until rank 0 sends the stop sentinel.  Each iteration:
/// 1. Participates in the sentinel allreduce to learn the next variant.
/// 2. Executes the matching collective so all ranks stay in lockstep with
///    rank 0's Criterion-driven `b.iter` calls.
///
/// `rank_cached` and `size_cached` are local operations on rank 0 and do not
/// appear in this loop — non-root ranks simply wait for the first collective
/// sentinel.
fn run_follower(world: &Communicator) {
    let mut bcast_buf = [0.0f64; 1];
    let send = [1.0f64; 1];
    let mut recv = [0.0f64; 1];

    loop {
        // Mirror the sentinel allreduce.
        let ctl_send = [0u64];
        let mut ctl_recv = [0u64; 1];
        world
            .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
            .unwrap();

        match ctl_recv[0] {
            VARIANT_STOP => break,

            VARIANT_BARRIER => {
                world.barrier().unwrap();
            }

            VARIANT_BROADCAST => {
                world.broadcast(&mut bcast_buf, 0).unwrap();
            }

            VARIANT_ALLREDUCE => {
                world.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
            }

            other => panic!("run_follower: unexpected sentinel value {other}"),
        }
    }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let _mpi = common::init_mpi_for_bench();
    let world = _mpi.world();

    let size = world.size();
    if size < 2 {
        panic!("ffi_overhead bench requires at least 2 MPI ranks; got {size}");
    }

    if world.rank() == 0 {
        // ── Rank 0: sole Criterion driver ─────────────────────────────────────
        let mut c = Criterion::default()
            .configure_from_args()
            .measurement_time(Duration::from_secs(3))
            .sample_size(100);

        bench_ffi_overhead(&mut c, &world);

        c.final_summary();

        // Send the stop sentinel so follower ranks exit their mirror loop.
        let stop_send = [VARIANT_STOP];
        let mut stop_recv = [0u64; 1];
        world
            .allreduce(&stop_send, &mut stop_recv, ReduceOp::Sum)
            .unwrap();
    } else {
        // ── Non-root ranks: mirror loop ───────────────────────────────────────
        run_follower(&world);
    }

    // All ranks meet here before MPI_Finalize.
    world.barrier().unwrap();
    drop(_mpi);
}
