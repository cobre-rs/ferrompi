#![allow(non_snake_case, non_camel_case_types, dead_code)]
use std::ffi::{c_int, c_long, c_void};
pub type MPI_Comm = c_int;
pub type MPI_Datatype = c_int;
pub type MPI_Op = c_int;
pub type MPI_Request = c_int;
pub type MPI_Win = c_int;
pub const MPI_COMM_WORLD: MPI_Comm = 0x44000000;
pub const MPI_DOUBLE: MPI_Datatype = 0x4c00080b;
pub const MPI_INT64_T: MPI_Datatype = 0x4c00083a;
pub const MPI_SUM: MPI_Op = 0x58000003;
pub const MPI_INFO_NULL: c_int = 0x1c000000;
pub const MPI_REQUEST_NULL: MPI_Request = 0x2c000000;
pub const MPI_STATUS_IGNORE: *mut c_void = 1 as *mut c_void;
extern "C" {
    pub fn MPI_Allreduce(s: *const c_void, r: *mut c_void, n: c_int, dt: MPI_Datatype, op: MPI_Op, c: MPI_Comm) -> c_int;
    pub fn MPI_Barrier(c: MPI_Comm) -> c_int;
    pub fn MPI_Bcast(b: *mut c_void, n: c_int, dt: MPI_Datatype, root: c_int, c: MPI_Comm) -> c_int;
    pub fn MPI_Iallreduce(s: *const c_void, r: *mut c_void, n: c_int, dt: MPI_Datatype, op: MPI_Op, c: MPI_Comm, req: *mut MPI_Request) -> c_int;
    pub fn MPI_Allreduce_init(s: *const c_void, r: *mut c_void, n: c_int, dt: MPI_Datatype, op: MPI_Op, c: MPI_Comm, info: c_int, req: *mut MPI_Request) -> c_int;
    pub fn MPI_Wait(req: *mut MPI_Request, st: *mut c_void) -> c_int;
    pub fn MPI_Waitall(n: c_int, req: *mut MPI_Request, st: *mut c_void) -> c_int;
    pub fn MPI_Start(req: *mut MPI_Request) -> c_int;
    pub fn MPI_Request_free(req: *mut MPI_Request) -> c_int;
    pub fn MPI_Isend(b: *const c_void, n: c_int, dt: MPI_Datatype, dest: c_int, tag: c_int, c: MPI_Comm, req: *mut MPI_Request) -> c_int;
    pub fn MPI_Irecv(b: *mut c_void, n: c_int, dt: MPI_Datatype, src: c_int, tag: c_int, c: MPI_Comm, req: *mut MPI_Request) -> c_int;
    pub fn MPI_Win_allocate(size: c_long, du: c_int, info: c_int, c: MPI_Comm, base: *mut c_void, win: *mut MPI_Win) -> c_int;
    pub fn MPI_Win_lock_all(a: c_int, w: MPI_Win) -> c_int;
    pub fn MPI_Win_unlock_all(w: MPI_Win) -> c_int;
    pub fn MPI_Win_flush(r: c_int, w: MPI_Win) -> c_int;
    pub fn MPI_Win_free(w: *mut MPI_Win) -> c_int;
    pub fn MPI_Fetch_and_op(o: *const c_void, r: *mut c_void, dt: MPI_Datatype, tr: c_int, td: c_long, op: MPI_Op, w: MPI_Win) -> c_int;
    pub fn MPI_Put(o: *const c_void, oc: c_int, odt: MPI_Datatype, tr: c_int, td: c_long, tc: c_int, tdt: MPI_Datatype, w: MPI_Win) -> c_int;
    pub fn MPI_Wtime() -> f64;
}

/// Run `f` `iters` times, return ns/op.
#[inline(never)]
pub fn time_loop<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let t0 = std::time::Instant::now();
    for _ in 0..iters { f(); }
    t0.elapsed().as_nanos() as f64 / iters as f64
}

pub fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Interleaved A/B: `reps` rounds, each round times A then B for `iters`.
/// Returns (median_a, median_b, min_a, min_b).
pub fn ab<A: FnMut(), B: FnMut()>(reps: usize, iters: usize, mut a: A, mut b: B, sync: &dyn Fn()) -> (f64, f64, f64, f64) {
    // warmup
    sync(); time_loop(iters / 4 + 1, &mut a); sync(); time_loop(iters / 4 + 1, &mut b);
    let (mut va, mut vb) = (Vec::new(), Vec::new());
    for r in 0..reps {
        if r % 2 == 0 {
            sync(); va.push(time_loop(iters, &mut a));
            sync(); vb.push(time_loop(iters, &mut b));
        } else {
            sync(); vb.push(time_loop(iters, &mut b));
            sync(); va.push(time_loop(iters, &mut a));
        }
    }
    let mina = va.iter().cloned().fold(f64::INFINITY, f64::min);
    let minb = vb.iter().cloned().fold(f64::INFINITY, f64::min);
    (median(&mut va), median(&mut vb), mina, minb)
}
extern "C" {
    pub fn MPI_Comm_dup(c: MPI_Comm, n: *mut MPI_Comm) -> c_int;
    pub fn MPI_Comm_free(c: *mut MPI_Comm) -> c_int;
}
extern "C" {
    pub fn MPI_Waitany(n: c_int, req: *mut MPI_Request, idx: *mut c_int, st: *mut c_void) -> c_int;
    pub fn MPI_Testsome(n: c_int, req: *mut MPI_Request, out: *mut c_int, idx: *mut c_int, st: *mut c_void) -> c_int;
    pub fn MPI_Startall(n: c_int, req: *mut MPI_Request) -> c_int;
}
extern "C" {
    pub fn MPI_Send(b: *const c_void, n: c_int, dt: MPI_Datatype, dest: c_int, tag: c_int, c: MPI_Comm) -> c_int;
    pub fn MPI_Recv(b: *mut c_void, n: c_int, dt: MPI_Datatype, src: c_int, tag: c_int, c: MPI_Comm, st: *mut c_void) -> c_int;
}
