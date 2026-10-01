use ferrompi::{Mpi, ReduceOp};
use perfprobe::*;
use std::hint::black_box;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let sync = || { world.barrier().unwrap(); };
    let rep = |name: &str, r: (f64, f64, f64, f64)| if rank == 0 { println!("{:<40} A {:>8.1} | B {:>8.1} | B-A {:>6.1} ns ({:>5.1}%)", name, r.0, r.1, r.1 - r.0, 100.0 * (r.1 - r.0) / r.0); };
    let s = [1.0f64; 64]; let mut r = [0.0f64; 64]; let s2 = [1.0f64; 64]; let mut r2 = [0.0f64; 64];
    for _ in 0..3 {
        let x = ab(reps, 20000,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 64, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || unsafe { MPI_Allreduce(black_box(s2.as_ptr()).cast(), black_box(r2.as_mut_ptr()).cast(), 64, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }, &sync);
        rep("A/A raw allreduce(64) vs raw", x);
        let x = ab(reps, 20000,
            || { world.allreduce(black_box(&s), black_box(&mut r), ReduceOp::Sum).unwrap(); },
            || { world.allreduce(black_box(&s2), black_box(&mut r2), ReduceOp::Sum).unwrap(); }, &sync);
        rep("A/A ferro allreduce(64) vs ferro", x);
        let x = ab(reps, 20000,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 64, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || { world.allreduce(black_box(&s2), black_box(&mut r2), ReduceOp::Sum).unwrap(); }, &sync);
        rep("A/B raw vs ferro allreduce(64)", x);
    }
    sync();
}
