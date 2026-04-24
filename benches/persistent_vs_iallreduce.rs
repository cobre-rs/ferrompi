//! Persistent-collective vs iallreduce comparison benchmark.
//!
//! Measures the wall-time cost of 100 consecutive `allreduce` operations.
//! Buffer: 131 072 `f64` elements (1 MiB). Group name: `iterative_allreduce_1mib_100x`.
//!
//! **MPI synchronization**: Rank 0 drives Criterion. Non-root ranks are kept in lockstep via
//! a sentinel `u64[1]` allreduce that encodes the variant. See `benches/README.md`
//! for context and output details.

use criterion::{black_box, Criterion};
use ferrompi::{Communicator, ReduceOp};
use std::time::Duration;

mod common;

/// Number of f64 elements in the benchmark buffer (1 MiB).
const N: usize = 131_072;

/// Number of allreduce iterations measured per Criterion sample.
const ITERS: usize = 100;

/// Sentinel variant codes carried in the control allreduce.
const VARIANT_STOP: u64 = 0;
const VARIANT_PERSISTENT: u64 = 1;
const VARIANT_IALLREDUCE: u64 = 2;

// ─── Rank-0 benchmark driver ─────────────────────────────────────────────────

/// Register and drive the two benchmarks on rank 0.
///
/// For each Criterion `b.iter` call, this function:
/// 1. Issues a sentinel allreduce that broadcasts the variant code to all other ranks.
/// 2. Runs the actual 100-iteration measurement loop.
///
/// Both steps are inside the same `b.iter` closure so the sentinel overhead is
/// included in the reported wall time.  At 1 MiB per allreduce the sentinel's
/// 8-byte allreduce is < 0.001 % overhead.
fn bench_iterative_allreduce(c: &mut Criterion, world: &Communicator) {
    let send = vec![world.rank() as f64; N];
    let mut recv = vec![0.0f64; N];

    let mut group = c.benchmark_group("iterative_allreduce_1mib_100x");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // ── Persistent benchmark ──────────────────────────────────────────────────
    // The PersistentRequest is created once, outside b.iter, so setup cost is
    // not included in the per-sample measurement — matching the ticket spec.
    let mut persistent = world
        .allreduce_init(&send, &mut recv, ReduceOp::Sum)
        .unwrap();

    group.bench_function("persistent", |b| {
        b.iter(|| {
            // Sentinel: tell follower ranks "run 100 persistent start/wait".
            let ctl_send = [VARIANT_PERSISTENT];
            let mut ctl_recv = [0u64; 1];
            world
                .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
                .unwrap();

            for _ in 0..ITERS {
                persistent.start().unwrap();
                persistent.wait().unwrap();
            }

            black_box(&recv);
        });
    });

    // ── iallreduce benchmark ──────────────────────────────────────────────────
    group.bench_function("iallreduce", |b| {
        b.iter(|| {
            // Sentinel: tell follower ranks "run 100 iallreduce+wait".
            let ctl_send = [VARIANT_IALLREDUCE];
            let mut ctl_recv = [0u64; 1];
            world
                .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
                .unwrap();

            for _ in 0..ITERS {
                let req = world
                    .iallreduce(black_box(&send), black_box(&mut recv), ReduceOp::Sum)
                    .unwrap();
                req.wait().unwrap();
            }
        });
    });

    group.finish();
}

// ─── Non-root mirror loop ─────────────────────────────────────────────────────

/// Mirror loop for ranks > 0.
///
/// Loops until rank 0 sends the stop sentinel.  Each iteration:
/// 1. Participates in the sentinel allreduce to learn the next variant.
/// 2. Executes the matching 100-iteration inner loop so all ranks stay
///    in lockstep with rank 0's Criterion-driven `b.iter` calls.
///
/// The `PersistentRequest` for the follower is created lazily on first use and
/// reused for all subsequent persistent samples, mirroring rank 0's strategy.
fn run_follower(world: &Communicator) {
    let send = vec![world.rank() as f64; N];
    let mut recv = vec![0.0f64; N];

    // Persistent request created lazily when first VARIANT_PERSISTENT arrives.
    let mut persistent_opt: Option<ferrompi::PersistentRequest> = None;

    loop {
        // Mirror the sentinel allreduce.
        let ctl_send = [0u64];
        let mut ctl_recv = [0u64; 1];
        world
            .allreduce(&ctl_send, &mut ctl_recv, ReduceOp::Sum)
            .unwrap();

        match ctl_recv[0] {
            VARIANT_STOP => break,

            VARIANT_PERSISTENT => {
                // Initialise persistent request once; reuse across all
                // subsequent persistent samples.
                let persistent = persistent_opt.get_or_insert_with(|| {
                    world
                        .allreduce_init(&send, &mut recv, ReduceOp::Sum)
                        .unwrap()
                });
                for _ in 0..ITERS {
                    persistent.start().unwrap();
                    persistent.wait().unwrap();
                }
            }

            VARIANT_IALLREDUCE => {
                for _ in 0..ITERS {
                    let req = world.iallreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
                    req.wait().unwrap();
                }
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
        panic!("persistent_vs_iallreduce bench requires at least 2 MPI ranks; got {size}");
    }

    if world.rank() == 0 {
        // ── Rank 0: sole Criterion driver ─────────────────────────────────────
        let mut c = Criterion::default()
            .configure_from_args()
            .measurement_time(Duration::from_secs(10))
            .sample_size(10);

        bench_iterative_allreduce(&mut c, &world);

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
