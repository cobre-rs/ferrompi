use ferrompi::{Mpi, Request};
use perfprobe::*;
use std::ffi::c_void;
use std::hint::black_box;

extern "C" {
    fn ferrompi_isend(
        buf: *const c_void,
        count: i64,
        tag_dt: i32,
        dest: i32,
        tag: i32,
        comm: i32,
        req: *mut i64,
    ) -> i32;
    fn ferrompi_irecv(
        buf: *mut c_void,
        count: i64,
        tag_dt: i32,
        src: i32,
        tag: i32,
        comm: i32,
        req: *mut i64,
    ) -> i32;
    fn ferrompi_waitall(count: i64, reqs: *mut i64, done: *mut u8, failed_index: *mut i64) -> i32;
    fn ferrompi_wait(req: i64) -> i32;
}
const F64_TAG: i32 = 1;

fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let n = world.size();
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21);
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let sync = || {
        world.barrier().unwrap();
    };
    let dest = (rank + 1) % n;
    let src = (rank - 1 + n) % n;
    let report = |name: &str, r: (f64, f64, f64, f64)| {
        if rank == 0 {
            println!("{:<40} A med {:>8.1} ns | B med {:>8.1} ns | B-A {:>7.1} ns ({:>5.1}%) per-req {:>5.1} ns | minA {:>8.1} minB {:>8.1}",
                name, r.0, r.1, r.1 - r.0, 100.0 * (r.1 - r.0) / r.0, (r.1 - r.0) / (2 * k) as f64, r.2, r.3);
        }
    };
    if rank == 0 {
        println!("# n={n} reps={reps} iters={iters} k={k} (2k requests per iteration)");
    }

    let sb = vec![1.0f64; k];
    let mut rb = vec![0.0f64; k];
    let mut raw = |sb: &[f64], rb: &mut [f64]| unsafe {
        let mut q = [0i32; 128];
        for i in 0..k {
            MPI_Irecv(
                rb.as_mut_ptr().add(i).cast(),
                1,
                MPI_DOUBLE,
                src,
                i as i32,
                MPI_COMM_WORLD,
                &mut q[i],
            );
        }
        for i in 0..k {
            MPI_Isend(
                sb.as_ptr().add(i).cast(),
                1,
                MPI_DOUBLE,
                dest,
                i as i32,
                MPI_COMM_WORLD,
                &mut q[k + i],
            );
        }
        MPI_Waitall((2 * k) as i32, q.as_mut_ptr(), MPI_STATUS_IGNORE);
    };
    let shim = |sb: &[f64], rb: &mut [f64]| unsafe {
        let mut q = [0i64; 128];
        let mut done = [0u8; 128];
        let mut failed_index: i64 = -1;
        for i in 0..k {
            ferrompi_irecv(
                rb.as_mut_ptr().add(i).cast(),
                1,
                F64_TAG,
                src,
                i as i32,
                0,
                &mut q[i],
            );
        }
        for i in 0..k {
            ferrompi_isend(
                sb.as_ptr().add(i).cast(),
                1,
                F64_TAG,
                dest,
                i as i32,
                0,
                &mut q[k + i],
            );
        }
        ferrompi_waitall(
            (2 * k) as i64,
            q.as_mut_ptr(),
            done.as_mut_ptr(),
            &mut failed_index,
        );
    };
    let mut reqs: Vec<Request> = Vec::with_capacity(2 * k);
    let mut rust = |sb: &[f64], rb: &mut [f64], reqs: &mut Vec<Request>| {
        for (i, x) in rb.chunks_mut(1).enumerate() {
            reqs.push(world.irecv(x, src, i as i32).unwrap());
        }
        for (i, x) in sb.chunks(1).enumerate() {
            reqs.push(world.isend(x, dest, i as i32).unwrap());
        }
        Request::wait_all(reqs).unwrap();
        reqs.clear();
    };
    let (mut rb1, mut rb2) = (rb.clone(), rb.clone());
    let r = ab(
        reps,
        iters,
        || raw(&sb, &mut rb1),
        || shim(&sb, &mut rb2),
        &sync,
    );
    report("A=raw MPI  B=C shim (waitall)", r);
    let r = ab(
        reps,
        iters,
        || shim(&sb, &mut rb1),
        || rust(&sb, &mut rb2, &mut reqs),
        &sync,
    );
    report("A=C shim   B=Rust API (wait_all)", r);
    let r = ab(
        reps,
        iters,
        || raw(&sb, &mut rb1),
        || rust(&sb, &mut rb2, &mut reqs),
        &sync,
    );
    report("A=raw MPI  B=Rust API (wait_all)", r);

    // individual waits
    let raw1 = |sb: &[f64], rb: &mut [f64]| unsafe {
        let mut q = [0i32; 128];
        for i in 0..k {
            MPI_Irecv(
                rb.as_mut_ptr().add(i).cast(),
                1,
                MPI_DOUBLE,
                src,
                i as i32,
                MPI_COMM_WORLD,
                &mut q[i],
            );
        }
        for i in 0..k {
            MPI_Isend(
                sb.as_ptr().add(i).cast(),
                1,
                MPI_DOUBLE,
                dest,
                i as i32,
                MPI_COMM_WORLD,
                &mut q[k + i],
            );
        }
        for i in 0..2 * k {
            MPI_Wait(&mut q[i], MPI_STATUS_IGNORE);
        }
    };
    let shim1 = |sb: &[f64], rb: &mut [f64]| unsafe {
        let mut q = [0i64; 128];
        for i in 0..k {
            ferrompi_irecv(
                rb.as_mut_ptr().add(i).cast(),
                1,
                F64_TAG,
                src,
                i as i32,
                0,
                &mut q[i],
            );
        }
        for i in 0..k {
            ferrompi_isend(
                sb.as_ptr().add(i).cast(),
                1,
                F64_TAG,
                dest,
                i as i32,
                0,
                &mut q[k + i],
            );
        }
        for i in 0..2 * k {
            ferrompi_wait(q[i]);
        }
    };
    let rust1 = |sb: &[f64], rb: &mut [f64], reqs: &mut Vec<Request>| {
        for (i, x) in rb.chunks_mut(1).enumerate() {
            reqs.push(world.irecv(x, src, i as i32).unwrap());
        }
        for (i, x) in sb.chunks(1).enumerate() {
            reqs.push(world.isend(x, dest, i as i32).unwrap());
        }
        for r in reqs.drain(..) {
            r.wait().unwrap();
        }
    };
    let r = ab(
        reps,
        iters,
        || raw1(&sb, &mut rb1),
        || shim1(&sb, &mut rb2),
        &sync,
    );
    report("A=raw MPI  B=C shim (wait each)", r);
    let r = ab(
        reps,
        iters,
        || shim1(&sb, &mut rb1),
        || rust1(&sb, &mut rb2, &mut reqs),
        &sync,
    );
    report("A=C shim   B=Rust API (wait each)", r);
    black_box((&rb1, &rb2));
    sync();
}
