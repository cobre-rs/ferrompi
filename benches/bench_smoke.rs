//! Smoke benchmark — verifies that the Criterion harness compiles and runs correctly under `mpiexec`.
//!
//! See `benches/README.md` for invocation and output details.

use criterion::{black_box, Criterion};

mod common;

fn smoke(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| black_box(1u64 + 1)));
}

fn main() {
    let _mpi = common::init_mpi_for_bench();

    let mut c = Criterion::default().configure_from_args();
    smoke(&mut c);
    c.final_summary();

    drop(_mpi);
}
