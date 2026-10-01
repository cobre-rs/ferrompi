use ferrompi::{Mpi, ReduceOp};
use perfprobe::*;
use std::ffi::c_void;
use std::hint::black_box;
extern "C" { fn ferrompi_allreduce(s: *const c_void, r: *mut c_void, n: i64, dt: i32, op: i32, comm: i32) -> i32; }
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let sync = || { world.barrier().unwrap(); };
    for &c in &[1usize, 8, 32, 64, 128, 512] {
        let s = vec![1.0f64; c]; let mut r = vec![0.0f64; c]; let s2 = vec![1.0f64; c]; let mut r2 = vec![0.0f64; c];
        let raw_vs_shim = ab(reps, 20000,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), c as i32, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || unsafe { ferrompi_allreduce(black_box(s2.as_ptr()).cast(), black_box(r2.as_mut_ptr()).cast(), c as i64, 1, 0, 0); }, &sync);
        let shim_vs_rust = ab(reps, 20000,
            || unsafe { ferrompi_allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), c as i64, 1, 0, 0); },
            || { world.allreduce(black_box(&s2[..]), black_box(&mut r2[..]), ReduceOp::Sum).unwrap(); }, &sync);
        if rank == 0 { println!("count {:>4}: raw {:>7.1} shim {:>7.1} (shim-raw {:>+6.1}) | shim {:>7.1} rust {:>7.1} (rust-shim {:>+6.1}) ns",
            c, raw_vs_shim.0, raw_vs_shim.1, raw_vs_shim.1 - raw_vs_shim.0, shim_vs_rust.0, shim_vs_rust.1, shim_vs_rust.1 - shim_vs_rust.0); }
    }
    sync();
}
