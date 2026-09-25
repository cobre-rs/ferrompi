use ferrompi::{Mpi, Request, ThreadLevel, Communicator};
use perfprobe::*;
use std::sync::{Arc, Barrier};
use std::hint::black_box;

// Each thread: K irecv + K isend to self on its own communicator, then waitall. Repeat ITERS.
fn run_raw(comm: i32, k: usize, iters: usize, bar: &Barrier) -> f64 {
    let sb = vec![1.0f64; k]; let mut rb = vec![0.0f64; k];
    let mut q = vec![0i32; 2 * k];
    bar.wait();
    let t0 = std::time::Instant::now();
    for _ in 0..iters { unsafe {
        for i in 0..k { MPI_Irecv(rb.as_mut_ptr().add(i).cast(), 1, MPI_DOUBLE, 0, i as i32, comm, &mut q[i]); }
        for i in 0..k { MPI_Isend(sb.as_ptr().add(i).cast(), 1, MPI_DOUBLE, 0, i as i32, comm, &mut q[k + i]); }
        MPI_Waitall((2 * k) as i32, q.as_mut_ptr(), MPI_STATUS_IGNORE);
    } }
    let e = t0.elapsed().as_nanos() as f64;
    black_box(&rb);
    e
}
fn run_ferro(comm: &Communicator, k: usize, iters: usize, bar: &Barrier) -> f64 {
    let sb = vec![1.0f64; k]; let mut rb = vec![0.0f64; k];
    let mut reqs: Vec<Request> = Vec::with_capacity(2 * k);
    bar.wait();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for (i, x) in rb.chunks_mut(1).enumerate() { reqs.push(comm.irecv(x, 0, i as i32).unwrap()); }
        for (i, x) in sb.chunks(1).enumerate() { reqs.push(comm.isend(x, 0, i as i32).unwrap()); }
        Request::wait_all(&mut reqs).unwrap();
        reqs.clear();
    }
    let e = t0.elapsed().as_nanos() as f64;
    black_box(&rb);
    e
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).unwrap();
    assert!(matches!(mpi.thread_level(), ThreadLevel::Multiple));
    let world = mpi.world();
    assert_eq!(world.size(), 1, "run with -n 1");
    let k: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20000);
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let threads: Vec<usize> = std::env::var("THREADS").ok().map(|s| s.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1, 2, 4, 8]);
    println!("# k={k} iters={iters} reps={reps} (ns per request = aggregate wall / total requests; lower is better)");
    for &t in &threads {
        // per-thread communicators (created on main thread)
        let fcomms: Vec<Arc<Communicator>> = (0..t).map(|_| Arc::new(world.duplicate().unwrap())).collect();
        let mut rcomms: Vec<i32> = (0..t).map(|_| { let mut c = 0; unsafe { MPI_Comm_dup(MPI_COMM_WORLD, &mut c); } c }).collect();
        let (mut vr, mut vf) = (Vec::new(), Vec::new());
        for rep in 0..reps {
            for pass in 0..2 {
                let do_raw = (rep + pass) % 2 == 0;
                let bar = Arc::new(Barrier::new(t));
                let t0 = std::time::Instant::now();
                std::thread::scope(|s| {
                    for i in 0..t {
                        let bar = bar.clone();
                        let rc = rcomms[i];
                        let fc = fcomms[i].clone();
                        s.spawn(move || if do_raw { run_raw(rc, k, iters, &bar) } else { run_ferro(&fc, k, iters, &bar) });
                    }
                });
                let wall = t0.elapsed().as_nanos() as f64 / (t * iters * 2 * k) as f64;
                if do_raw { vr.push(wall) } else { vf.push(wall) }
            }
        }
        let (mr, mf) = (median(&mut vr), median(&mut vf));
        println!("threads={t:<2} raw {mr:>7.1} ns/req | ferrompi {mf:>7.1} ns/req | delta {:>6.1} ns/req ({:>5.1}%)", mf - mr, 100.0 * (mf - mr) / mr);
        for c in rcomms.iter_mut() { unsafe { MPI_Comm_free(c); } }
        drop(fcomms);
    }
}
