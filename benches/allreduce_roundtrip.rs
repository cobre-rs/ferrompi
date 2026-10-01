//! Allreduce roundtrip benchmark — measures `world.allreduce(&send, &mut recv, ReduceOp::Sum)`
//! latency and throughput for three `f64` buffer sizes: 2, 131 072, and 2 097 152 elements.
//!
//! **MPI synchronization**: Rank 0 drives Criterion. Non-root ranks are kept in lockstep via
//! a `common::lead` allreduce issued before each data allreduce. See `benches/README.md`
//! for context and output details.

use criterion::{BenchmarkId, Criterion, Throughput};
use ferrompi::{Communicator, ReduceOp};
use std::hint::black_box;
use std::time::Duration;

mod common;

/// Run one f64 allreduce of the size in the command's argument.
const ALLREDUCE: u64 = 1;

/// Sizes in number of f64 elements: 16 B, 1 MiB, 16 MiB.
const SIZES: &[usize] = &[2, 131_072, 2_097_152];

/// Run the Criterion benchmark group on rank 0.
///
/// Inside each `b.iter` closure this function issues a sentinel allreduce
/// followed by the actual f64 data allreduce.  The sentinel tells non-root
/// ranks which size is being measured so they can call the matching allreduce.
fn allreduce_f64(c: &mut Criterion, world: &Communicator) {
    let mut group = c.benchmark_group("allreduce_f64");

    for &n in SIZES {
        let send = vec![world.rank() as f64; n];
        let mut recv = vec![0.0f64; n];
        let bytes = (n * std::mem::size_of::<f64>()) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                common::lead(world, [ALLREDUCE, n as u64]);

                // Data allreduce — this is what we want to measure.
                world
                    .allreduce(black_box(&send), black_box(&mut recv), ReduceOp::Sum)
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Non-root rank mirror loop.
///
/// Loops indefinitely, calling the sentinel allreduce then the matching f64
/// allreduce, until the sentinel signals "stop".  Buffers are allocated once
/// per size encountered; if the same size repeats across iterations of the
/// same benchmark, the same pre-allocated buffer is reused.
fn run_follower(world: &Communicator) {
    // Cache the last-seen size to avoid re-allocating on every iteration.
    let mut cached_n: usize = 0;
    let mut send: Vec<f64> = Vec::new();
    let mut recv: Vec<f64> = Vec::new();

    common::follow(world, |[_, n]| {
        let n = n as usize;

        // Reallocate buffers only when the size changes.
        if n != cached_n {
            cached_n = n;
            send = vec![world.rank() as f64; n];
            recv = vec![0.0f64; n];
        }

        // Mirror the f64 data allreduce that rank 0 issues inside b.iter.
        world.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    });
}

fn main() {
    let _mpi = common::init_mpi_for_bench();
    let world = _mpi.world();

    let size = world.size();
    if size < 2 {
        panic!("allreduce_roundtrip bench requires at least 2 MPI ranks; got {size}");
    }

    if world.rank() == 0 {
        // Rank 0: sole Criterion driver.
        let mut c = Criterion::default()
            .configure_from_args()
            .measurement_time(Duration::from_secs(5))
            .sample_size(20);

        allreduce_f64(&mut c, &world);

        c.final_summary();

        // Send the stop command so non-root ranks exit their mirror loop.
        common::lead(&world, common::STOP);
    } else {
        // Non-root ranks: mirror loop.
        run_follower(&world);
    }

    // Both ranks meet here before MPI_Finalize.
    world.barrier().unwrap();
    drop(_mpi);
}
