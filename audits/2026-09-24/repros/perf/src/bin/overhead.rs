use ferrompi::{Mpi, ReduceOp, Request, Win, SharedWindow};
use perfprobe::*;
use std::ffi::c_void;
use std::hint::black_box;

fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let n = world.size();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let sync = || { world.barrier().unwrap(); };
    let report = |name: &str, r: (f64, f64, f64, f64)| {
        if rank == 0 {
            println!("{:<34} raw med {:>9.1} ns | ferrompi med {:>9.1} ns | delta {:>7.1} ns ({:>5.1}%) | min raw {:>8.1} ferro {:>8.1}",
                name, r.0, r.1, r.1 - r.0, 100.0 * (r.1 - r.0) / r.0, r.2, r.3);
        }
    };
    if rank == 0 { println!("# n={n} reps={reps} iters={iters}"); }

    // 1. allreduce 1 f64
    {
        let s = [1.0f64]; let mut r = [0.0f64];
        let s2 = [1.0f64]; let mut r2 = [0.0f64];
        let res = ab(reps, iters,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || { world.allreduce(black_box(&s2), black_box(&mut r2), ReduceOp::Sum).unwrap(); },
            &sync);
        report("allreduce(1 f64)", res);
    }
    // 2. allreduce_scalar
    {
        let s = [1.0f64]; let mut r = [0.0f64];
        let res = ab(reps, iters,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || { black_box(world.allreduce_scalar(black_box(1.0f64), ReduceOp::Sum).unwrap()); },
            &sync);
        report("allreduce_scalar(f64)", res);
    }
    // 3. allreduce 64 f64
    {
        let s = [1.0f64; 64]; let mut r = [0.0f64; 64];
        let s2 = [1.0f64; 64]; let mut r2 = [0.0f64; 64];
        let res = ab(reps, iters,
            || unsafe { MPI_Allreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 64, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); },
            || { world.allreduce(black_box(&s2), black_box(&mut r2), ReduceOp::Sum).unwrap(); },
            &sync);
        report("allreduce(64 f64)", res);
    }
    // 4. barrier
    {
        let res = ab(reps, iters,
            || unsafe { MPI_Barrier(MPI_COMM_WORLD); },
            || { world.barrier().unwrap(); },
            &sync);
        report("barrier", res);
    }
    // 5. bcast 1 f64
    {
        let mut b = [1.0f64]; let mut b2 = [1.0f64];
        let res = ab(reps, iters,
            || unsafe { MPI_Bcast(black_box(b.as_mut_ptr()).cast(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); },
            || { world.broadcast(black_box(&mut b2), 0).unwrap(); },
            &sync);
        report("bcast(1 f64)", res);
    }
    // 6. iallreduce + wait
    {
        let s = [1.0f64]; let mut r = [0.0f64];
        let s2 = [1.0f64]; let mut r2 = [0.0f64];
        let res = ab(reps, iters,
            || unsafe { let mut q = 0; MPI_Iallreduce(black_box(s.as_ptr()).cast(), black_box(r.as_mut_ptr()).cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &mut q); MPI_Wait(&mut q, MPI_STATUS_IGNORE); },
            || { world.iallreduce(black_box(&s2), black_box(&mut r2), ReduceOp::Sum).unwrap().wait().unwrap(); },
            &sync);
        report("iallreduce+wait(1 f64)", res);
    }
    // 7. persistent allreduce start+wait
    {
        let s = [1.0f64]; let mut r = [0.0f64];
        let s2 = [1.0f64]; let mut r2 = [0.0f64];
        let mut q = 0;
        unsafe { MPI_Allreduce_init(s.as_ptr().cast(), r.as_mut_ptr().cast(), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &mut q); }
        let mut p = world.allreduce_init(&s2, &mut r2, ReduceOp::Sum).unwrap();
        let res = ab(reps, iters,
            || unsafe { MPI_Start(&mut q); MPI_Wait(&mut q, MPI_STATUS_IGNORE); },
            || { p.start().unwrap(); p.wait().unwrap(); },
            &sync);
        report("persistent allreduce start+wait", res);
        unsafe { MPI_Request_free(&mut q); }
        drop(p);
        black_box((&r, &r2));
    }
    // 8. neighbor exchange: 8 isend + 8 irecv + waitall (1 f64 each)
    {
        let dest = (rank + 1) % n; let src = (rank - 1 + n) % n;
        let sb = [1.0f64; 8]; let mut rb = [0.0f64; 8];
        let sb2 = [1.0f64; 8]; let mut rb2 = [0.0f64; 8];
        let mut reqs: Vec<Request> = Vec::with_capacity(16);
        let res = ab(reps, iters / 4,
            || unsafe {
                let mut q = [0i32; 16];
                for i in 0..8 { MPI_Irecv(rb.as_mut_ptr().add(i).cast(), 1, MPI_DOUBLE, src, i as i32, MPI_COMM_WORLD, &mut q[i]); }
                for i in 0..8 { MPI_Isend(sb.as_ptr().add(i).cast(), 1, MPI_DOUBLE, dest, i as i32, MPI_COMM_WORLD, &mut q[8 + i]); }
                MPI_Waitall(16, q.as_mut_ptr(), MPI_STATUS_IGNORE);
            },
            || {
                reqs.clear();
                // split borrows per element to satisfy borrowck
                let (a, _) = rb2.split_at_mut(8);
                for (i, x) in a.chunks_mut(1).enumerate() { reqs.push(world.irecv(x, src, i as i32).unwrap()); }
                for (i, x) in sb2.chunks(1).enumerate() { reqs.push(world.isend(x, dest, i as i32).unwrap()); }
                Request::wait_all(&mut reqs).unwrap();
                reqs.clear();
            },
            &sync);
        report("8x(isend+irecv)+waitall", res);
    }
    // 9. RMA fetch_and_op + flush (passive target, lock_all epoch)
    {
        let target = (rank + 1) % n;
        let mut wraw: i32 = 0; let mut base: *mut c_void = std::ptr::null_mut();
        unsafe { MPI_Win_allocate(8, 8, MPI_INFO_NULL, MPI_COMM_WORLD, (&mut base as *mut *mut c_void).cast(), &mut wraw); MPI_Win_lock_all(0, wraw); }
        let win = Win::<i64>::allocate(&world, 1).unwrap();
        let guard = win.lock_all().unwrap();
        let res = ab(reps, iters,
            || unsafe {
                let o: i64 = 1; let mut r: i64 = 0;
                MPI_Fetch_and_op((&o as *const i64).cast(), (&mut r as *mut i64).cast(), MPI_INT64_T, target, 0, MPI_SUM, wraw);
                MPI_Win_flush(target, wraw);
                black_box(r);
            },
            || {
                let p = win.fetch_and_op(1i64, target, 0, ReduceOp::Sum).unwrap();
                guard.flush(target).unwrap();
                black_box(unsafe { p.resolve() });
            },
            &sync);
        report("fetch_and_op+flush(i64)", res);
        // 10. put 1 element + flush
        let v = [1i64]; let v2 = [1i64];
        let res = ab(reps, iters,
            || unsafe { MPI_Put(v.as_ptr().cast(), 1, MPI_INT64_T, target, 0, 1, MPI_INT64_T, wraw); MPI_Win_flush(target, wraw); },
            || { win.put(&v2, target, 0, 1).unwrap(); guard.flush(target).unwrap(); },
            &sync);
        report("put(1 i64)+flush", res);
        drop(guard);
        drop(win);
        unsafe { MPI_Win_unlock_all(wraw); MPI_Win_free(&mut wraw); }
    }
    // 11. SharedWindow remote_slice (MPI_Win_shared_query per call) vs cached slice
    {
        let node = world.split_shared().unwrap();
        let win = SharedWindow::<f64>::allocate(&node, 16).unwrap();
        let peer = (node.rank() + 1) % node.size();
        let cached = win.remote_slice(peer).unwrap();
        let res = ab(reps, iters * 10,
            || { black_box(black_box(cached)[3]); },
            || { black_box(win.remote_slice(black_box(peer)).unwrap()[3]); },
            &sync);
        report("SharedWindow::remote_slice+read", res);
    }
    sync();
}
