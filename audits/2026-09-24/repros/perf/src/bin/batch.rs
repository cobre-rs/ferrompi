use ferrompi::{Mpi, ReduceOp, Request, PersistentRequest};
use perfprobe::*;
use std::hint::black_box;

fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank(); let n = world.size();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let sync = || { world.barrier().unwrap(); };
    let dest = (rank + 1) % n; let src = (rank - 1 + n) % n;
    let report = |name: &str, r: (f64, f64, f64, f64)| {
        if rank == 0 {
            println!("{:<44} raw med {:>9.1} ns | ferrompi med {:>9.1} ns | delta {:>8.1} ns ({:>5.1}%) | min raw {:>9.1} ferro {:>9.1}",
                name, r.0, r.1, r.1 - r.0, 100.0 * (r.1 - r.0) / r.0, r.2, r.3);
        }
    };
    if rank == 0 { println!("# n={n} reps={reps}"); }

    // A. Post N irecv + N isend, drain with waitany one at a time.
    for &nn in &[16usize, 64] {
        let sb = vec![1.0f64; nn]; let mut rb1 = vec![0.0f64; nn]; let mut rb2 = vec![0.0f64; nn];
        let mut q = vec![0i32; 2 * nn];
        let mut reqs: Vec<Request> = Vec::with_capacity(2 * nn);
        let iters = 40000 / nn;
        let r = ab(reps, iters,
            || unsafe {
                for i in 0..nn { MPI_Irecv(rb1.as_mut_ptr().add(i).cast(), 1, MPI_DOUBLE, src, i as i32, MPI_COMM_WORLD, &mut q[i]); }
                for i in 0..nn { MPI_Isend(sb.as_ptr().add(i).cast(), 1, MPI_DOUBLE, dest, i as i32, MPI_COMM_WORLD, &mut q[nn + i]); }
                let mut idx = 0;
                for _ in 0..2 * nn { MPI_Waitany((2 * nn) as i32, q.as_mut_ptr(), &mut idx, MPI_STATUS_IGNORE); black_box(idx); }
            },
            || {
                for (i, x) in rb2.chunks_mut(1).enumerate() { reqs.push(world.irecv(x, src, i as i32).unwrap()); }
                for (i, x) in sb.chunks(1).enumerate() { reqs.push(world.isend(x, dest, i as i32).unwrap()); }
                while !reqs.is_empty() {
                    let idx = Request::wait_any(&mut reqs).unwrap().unwrap();
                    reqs.swap_remove(idx);
                }
            }, &sync);
        report(&format!("drain {} reqs via waitany (per drain)", 2 * nn), r);
    }
    // B. test_some polling: post N pairs, poll with testsome until all done.
    {
        let nn = 16usize;
        let sb = vec![1.0f64; nn]; let mut rb1 = vec![0.0f64; nn]; let mut rb2 = vec![0.0f64; nn];
        let mut q = vec![0i32; 2 * nn]; let mut idxs = vec![0i32; 2 * nn];
        let mut reqs: Vec<Request> = Vec::with_capacity(2 * nn);
        let r = ab(reps, 2000,
            || unsafe {
                for i in 0..nn { MPI_Irecv(rb1.as_mut_ptr().add(i).cast(), 1, MPI_DOUBLE, src, i as i32, MPI_COMM_WORLD, &mut q[i]); }
                for i in 0..nn { MPI_Isend(sb.as_ptr().add(i).cast(), 1, MPI_DOUBLE, dest, i as i32, MPI_COMM_WORLD, &mut q[nn + i]); }
                let mut done = 0;
                while done < 2 * nn { let mut out = 0; MPI_Testsome((2 * nn) as i32, q.as_mut_ptr(), &mut out, idxs.as_mut_ptr(), MPI_STATUS_IGNORE); if out > 0 { done += out as usize; } }
            },
            || {
                for (i, x) in rb2.chunks_mut(1).enumerate() { reqs.push(world.irecv(x, src, i as i32).unwrap()); }
                for (i, x) in sb.chunks(1).enumerate() { reqs.push(world.isend(x, dest, i as i32).unwrap()); }
                while !reqs.is_empty() {
                    let mut done = Request::test_some(&mut reqs).unwrap();
                    done.sort_unstable_by(|a, b| b.cmp(a));
                    for i in done { reqs.swap_remove(i); }
                }
            }, &sync);
        report("post 32 + poll test_some until drained", r);
    }
    // C. persistent start_all + wait_all of 2 allreduces (1 f64 each)
    {
        let s = [1.0f64; 2]; let mut r1 = [0.0f64; 2]; let mut r2 = [0.0f64; 2];
        let mut q = [0i32; 2];
        unsafe {
            MPI_Allreduce_init(s.as_ptr().cast(), r1.as_mut_ptr().cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &mut q[0]);
            MPI_Allreduce_init(s.as_ptr().add(1).cast(), r1.as_mut_ptr().add(1).cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &mut q[1]);
        }
        let (a, b) = r2.split_at_mut(1);
        let mut p = vec![world.allreduce_init(&s[..1], a, ReduceOp::Sum).unwrap(), world.allreduce_init(&s[1..], b, ReduceOp::Sum).unwrap()];
        let res = ab(reps, 20000,
            || unsafe { MPI_Startall(2, q.as_mut_ptr()); MPI_Waitall(2, q.as_mut_ptr(), MPI_STATUS_IGNORE); },
            || { PersistentRequest::start_all(&mut p).unwrap(); PersistentRequest::wait_all(&mut p).unwrap(); },
            &sync);
        report("persistent start_all+wait_all (2 reqs)", res);
        unsafe { MPI_Request_free(&mut q[0]); MPI_Request_free(&mut q[1]); }
        drop(p);
    }
    // D. repeat allreduce 64 with more reps, buffers swapped between arms to detect alignment/noise
    {
        let s = [1.0f64; 64]; let mut r = [0.0f64; 64];
        let s2 = [1.0f64; 64]; let mut r2 = [0.0f64; 64];
        let res = ab(reps, 20000,
            || unsafe { MPI_Allreduce(black_box(s2.as_ptr()).cast(), black_box(r2.as_mut_ptr()).cast(), 64, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || { world.allreduce(black_box(&s), black_box(&mut r), ReduceOp::Sum).unwrap(); },
            &sync);
        report("allreduce(64 f64) [buffers swapped]", res);
    }
    sync();
}
