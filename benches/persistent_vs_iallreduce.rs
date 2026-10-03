//! Persistent-collective vs iallreduce comparison benchmark.
//!
//! Measures the wall-time cost of 100 consecutive `allreduce` operations, swept over
//! message sizes from 8 B to 256 KiB in steps of 8x, then 1 MiB. Group name:
//! `iterative_allreduce_100x`.
//!
//! **MPI synchronization**: Rank 0 drives Criterion. Non-root ranks are kept in lockstep via
//! a sentinel `[u64; 2]` allreduce carrying a command code and argument. See
//! `benches/README.md` for context and output details.

use criterion::{BenchmarkId, Criterion};
use ferrompi::{Communicator, PersistentRequest, ReduceOp};
use std::hint::black_box;
use std::time::Duration;

mod common;

/// Message sizes in `f64` elements: 8 B to 256 KiB in steps of 8x, then 1 MiB.
const SIZES: &[usize] = &[1, 8, 64, 512, 4_096, 32_768, 131_072];

/// Number of allreduce iterations measured per Criterion sample.
const ITERS: usize = 100;

/// Command codes carried in the control allreduce.
const SETUP: u64 = 1;
const PERSISTENT: u64 = 2;
const IALLREDUCE: u64 = 3;

/// Register and drive the two benchmarks on rank 0, one pair per size in [`SIZES`].
///
/// For each size, this function sends a `SETUP` command (so every rank re-initializes
/// a persistent request in the same collective order), then measures `persistent`
/// start+wait and `iallreduce`+wait over [`ITERS`] iterations per Criterion sample.
fn bench_iterative_allreduce(c: &mut Criterion, world: &Communicator) {
    let mut group = c.benchmark_group("iterative_allreduce_100x");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for &n in SIZES {
        let bytes = n * std::mem::size_of::<f64>();

        // SETUP: every rank learns the element count and re-initializes its
        // persistent request in the same order before any bench_function runs.
        common::lead(world, [SETUP, n as u64]);

        let send = vec![world.rank() as f64; n];
        let mut recv = vec![0.0f64; n];
        let mut persistent = world
            .allreduce_init(&send, &mut recv, ReduceOp::Sum)
            .expect("allreduce_init needs persistent collectives");

        group.bench_with_input(BenchmarkId::new("persistent", bytes), &n, |b, _| {
            b.iter(|| {
                common::lead(world, [PERSISTENT, 0]);
                for _ in 0..ITERS {
                    persistent.start().unwrap();
                    persistent.wait().unwrap();
                }
                black_box(&recv);
            });
        });

        // Drop rank 0's request before the next size's SETUP so each rank holds at
        // most one persistent request at a time.
        drop(persistent);

        group.bench_with_input(BenchmarkId::new("iallreduce", bytes), &n, |b, _| {
            b.iter(|| {
                common::lead(world, [IALLREDUCE, 0]);
                for _ in 0..ITERS {
                    ferrompi::scope(|s| {
                        let req = world.iallreduce(
                            s,
                            black_box(&send),
                            black_box(&mut recv),
                            ReduceOp::Sum,
                        )?;
                        req.wait()?;
                        Ok(())
                    })
                    .unwrap();
                }
            });
        });
    }

    group.finish();
}

/// Mirror loop for ranks > 0.
///
/// Loops until rank 0 sends `STOP`. On `SETUP` it frees its current persistent
/// request first, reallocates buffers to the new element count, and re-initializes
/// the request. On `PERSISTENT`/`IALLREDUCE` it runs the matching [`ITERS`]-iteration
/// loop, mirroring rank 0's `b.iter` calls.
fn run_follower(world: &Communicator) {
    let mut send: Vec<f64> = Vec::new();
    let mut recv: Vec<f64> = Vec::new();
    let mut persistent: Option<PersistentRequest> = None;

    common::follow(world, |[op, arg]| match op {
        SETUP => {
            // Free the old request before reallocating: a request still pointing
            // into freed buffers is harmless only while inactive, and it must
            // never be started again after the buffers move.
            drop(persistent.take());
            let n = arg as usize;
            send = vec![world.rank() as f64; n];
            recv = vec![0.0f64; n];
            persistent = Some(
                world
                    .allreduce_init(&send, &mut recv, ReduceOp::Sum)
                    .expect("allreduce_init needs persistent collectives"),
            );
        }

        PERSISTENT => {
            let req = persistent
                .as_mut()
                .expect("run_follower: PERSISTENT without a preceding SETUP");
            for _ in 0..ITERS {
                req.start().unwrap();
                req.wait().unwrap();
            }
        }

        IALLREDUCE => {
            for _ in 0..ITERS {
                ferrompi::scope(|s| {
                    let req = world.iallreduce(s, &send, &mut recv, ReduceOp::Sum)?;
                    req.wait()?;
                    Ok(())
                })
                .unwrap();
            }
        }

        other => panic!("run_follower: unexpected command code {other}"),
    });
}

fn main() {
    let _mpi = common::init_mpi_for_bench();
    let world = _mpi.world();

    let size = world.size();
    if size < 2 {
        panic!("persistent_vs_iallreduce bench requires at least 2 MPI ranks; got {size}");
    }

    if world.rank() == 0 {
        // Rank 0: sole Criterion driver.
        let mut c = Criterion::default().configure_from_args();

        bench_iterative_allreduce(&mut c, &world);

        c.final_summary();

        // Send the stop command so follower ranks exit their mirror loop.
        common::lead(&world, common::STOP);
    } else {
        // Non-root ranks: mirror loop.
        run_follower(&world);
    }

    // All ranks meet here before MPI_Finalize.
    world.barrier().unwrap();
    drop(_mpi);
}
