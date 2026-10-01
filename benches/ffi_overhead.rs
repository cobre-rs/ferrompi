//! Interleaved A/B FFI-overhead benchmark: ferrompi vs direct `MPI_*`, at one rank.
//!
//! Each case alternates rounds of a direct `MPI_*` call and the matching ferrompi
//! call, then reports each arm's median ns/call and their delta. The first case,
//! `A/A direct iallreduce+wait`, runs the direct arm on both sides; its delta is
//! the noise floor the other cases' deltas are judged against.
//!
//! `FERROMPI_BENCH_LEVEL` selects the thread level MPI is initialized with:
//! `funneled` (the default) or `multiple`. The binary prints one line and exits 0
//! when the library does not provide the requested level.
//!
//! The direct arm links MPICH's integer handle values and calls only MPI-1/MPI-3
//! symbols, so it links on every MPI implementation. At run time it runs only when
//! `Mpi::library_version()` reports MPICH; on any other library it prints one line
//! and exits 0.
//!
//! Run with `cargo bench --bench ffi_overhead` as a singleton, or build with
//! `cargo bench --bench ffi_overhead --no-run` and launch the binary under
//! `mpiexec -n 1`.

#![allow(non_snake_case)]

use ferrompi::{Mpi, PersistentRequest, ReduceOp, Request, ThreadLevel};
use std::ffi::{c_int, c_void};
use std::hint::black_box;
use std::time::Instant;

/// MPICH's `MPI_Request` is a plain `int`.
type MpiRequest = c_int;

/// MPICH handle values from `mpi.h`.
const MPI_COMM_WORLD: c_int = 0x4400_0000;
const MPI_DOUBLE: c_int = 0x4c00_080b;
const MPI_SUM: c_int = 0x5800_0003;
const MPI_UINT8_T: c_int = 0x4c00_013b;
const MPI_UINT64_T: c_int = 0x4c00_083e;
const MPI_BOR: c_int = 0x5800_0008;

/// Interleaved rounds per case.
const ROUNDS: usize = 21;
/// Messages per batch in the 8x cases.
const BATCH: usize = 8;

extern "C" {
    fn MPI_Isend(
        buf: *const c_void,
        count: c_int,
        datatype: c_int,
        dest: c_int,
        tag: c_int,
        comm: c_int,
        request: *mut MpiRequest,
    ) -> c_int;
    fn MPI_Irecv(
        buf: *mut c_void,
        count: c_int,
        datatype: c_int,
        source: c_int,
        tag: c_int,
        comm: c_int,
        request: *mut MpiRequest,
    ) -> c_int;
    fn MPI_Send_init(
        buf: *const c_void,
        count: c_int,
        datatype: c_int,
        dest: c_int,
        tag: c_int,
        comm: c_int,
        request: *mut MpiRequest,
    ) -> c_int;
    fn MPI_Recv_init(
        buf: *mut c_void,
        count: c_int,
        datatype: c_int,
        source: c_int,
        tag: c_int,
        comm: c_int,
        request: *mut MpiRequest,
    ) -> c_int;
    fn MPI_Iallreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: c_int,
        datatype: c_int,
        op: c_int,
        comm: c_int,
        request: *mut MpiRequest,
    ) -> c_int;
    fn MPI_Allreduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: c_int,
        datatype: c_int,
        op: c_int,
        comm: c_int,
    ) -> c_int;
    fn MPI_Allgatherv(
        sendbuf: *const c_void,
        sendcount: c_int,
        sendtype: c_int,
        recvbuf: *mut c_void,
        recvcounts: *const c_int,
        displs: *const c_int,
        recvtype: c_int,
        comm: c_int,
    ) -> c_int;
    fn MPI_Bcast(
        buffer: *mut c_void,
        count: c_int,
        datatype: c_int,
        root: c_int,
        comm: c_int,
    ) -> c_int;
    fn MPI_Barrier(comm: c_int) -> c_int;
    fn MPI_Start(request: *mut MpiRequest) -> c_int;
    fn MPI_Startall(count: c_int, requests: *mut MpiRequest) -> c_int;
    fn MPI_Wait(request: *mut MpiRequest, status: *mut c_void) -> c_int;
    fn MPI_Waitall(count: c_int, requests: *mut MpiRequest, statuses: *mut c_void) -> c_int;
    fn MPI_Request_free(request: *mut MpiRequest) -> c_int;
}

/// MPICH's `MPI_STATUS_IGNORE` and `MPI_STATUSES_IGNORE` are both `(MPI_Status *)1`.
fn status_ignore() -> *mut c_void {
    std::ptr::without_provenance_mut(1)
}

/// Run `f` `iters` times, return ns/call.
#[inline(never)]
fn ns_per_call(iters: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One `MPI_Iallreduce`+`MPI_Wait` round trip. A plain `fn`, not a closure, so the
/// A/A control case can run it from both arms without two closures capturing the
/// same buffer mutably.
fn direct_iallreduce(send: &[f64], recv: &mut [f64]) {
    let mut request: MpiRequest = 0;
    // SAFETY: send/recv outlive this call; MPI_Iallreduce issues the operation and
    // MPI_Wait blocks until it completes before this function returns.
    unsafe {
        MPI_Iallreduce(
            black_box(send.as_ptr()).cast(),
            black_box(recv.as_mut_ptr()).cast(),
            send.len() as c_int,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD,
            &mut request,
        );
        MPI_Wait(&mut request, status_ignore());
    }
}

/// Alternate `direct` and `ferrompi` across `ROUNDS` rounds — direct then ferrompi
/// on even rounds, the reverse on odd rounds — after one warm-up round of each arm,
/// and print the per-arm median ns/call and their delta.
fn compare(name: &str, iters: usize, mut direct: impl FnMut(), mut ferrompi: impl FnMut()) {
    ns_per_call(iters / 4 + 1, &mut direct);
    ns_per_call(iters / 4 + 1, &mut ferrompi);

    let mut direct_ns = Vec::with_capacity(ROUNDS);
    let mut ferrompi_ns = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            direct_ns.push(ns_per_call(iters, &mut direct));
            ferrompi_ns.push(ns_per_call(iters, &mut ferrompi));
        } else {
            ferrompi_ns.push(ns_per_call(iters, &mut ferrompi));
            direct_ns.push(ns_per_call(iters, &mut direct));
        }
    }

    let a = median(&mut direct_ns);
    let b = median(&mut ferrompi_ns);
    println!(
        "{name:<34} direct {a:>8.1} ns   ferrompi {b:>8.1} ns   delta {:>+7.1} ns",
        b - a
    );
}

fn main() {
    let requested =
        std::env::var_os("FERROMPI_BENCH_LEVEL").map(|v| v.to_string_lossy().into_owned());
    let level = match requested.as_deref() {
        None | Some("funneled") => ThreadLevel::Funneled,
        Some("multiple") => ThreadLevel::Multiple,
        Some(other) => panic!("FERROMPI_BENCH_LEVEL must be funneled or multiple, got {other}"),
    };

    let mpi = Mpi::init_thread(level).unwrap();
    if mpi.thread_level() != level {
        println!("ffi_overhead: {level:?} not provided; skipped");
        return;
    }
    let world = mpi.world();

    assert_eq!(world.size(), 1, "ffi_overhead runs at one rank");

    let version = Mpi::library_version().unwrap();
    let line = version.lines().next().unwrap_or("").trim();
    if !line.starts_with("MPICH") {
        println!("ffi_overhead: the direct arm needs MPICH's handle values; skipped on {line}");
        return;
    }
    println!("# {line}; level {level:?}; {ROUNDS} interleaved rounds per arm; median ns per call");

    // A/A direct iallreduce+wait: both arms run the direct code, on separate
    // buffers, to establish the noise floor the other cases are judged against.
    {
        let (send_a, mut recv_a) = ([1.0f64], [0.0f64]);
        let (send_b, mut recv_b) = ([1.0f64], [0.0f64]);
        compare(
            "A/A direct iallreduce+wait",
            20_000,
            || direct_iallreduce(&send_a, &mut recv_a),
            || direct_iallreduce(&send_b, &mut recv_b),
        );
    }

    // allreduce f64 sum: blocking MPI_Allreduce against world.allreduce.
    {
        let (send_d, mut recv_d) = ([1.0f64], [0.0f64]);
        let (send_f, mut recv_f) = ([1.0f64], [0.0f64]);
        compare(
            "allreduce f64 sum",
            20_000,
            || {
                // SAFETY: send_d/recv_d are live, distinct one-element f64 buffers;
                // the blocking MPI_Allreduce is done with them when it returns.
                unsafe {
                    MPI_Allreduce(
                        black_box(send_d.as_ptr()).cast(),
                        black_box(recv_d.as_mut_ptr()).cast(),
                        1,
                        MPI_DOUBLE,
                        MPI_SUM,
                        MPI_COMM_WORLD,
                    );
                }
            },
            || {
                world
                    .allreduce(black_box(&send_f), black_box(&mut recv_f), ReduceOp::Sum)
                    .unwrap();
            },
        );
    }

    // allreduce u64 bor: blocking MPI_Allreduce with MPI_BOR on u64.
    {
        let (send_d, mut recv_d) = ([1u64], [0u64]);
        let (send_f, mut recv_f) = ([1u64], [0u64]);
        compare(
            "allreduce u64 bor",
            20_000,
            || {
                // SAFETY: send_d/recv_d are live, distinct one-element u64 buffers;
                // the blocking MPI_Allreduce is done with them when it returns.
                unsafe {
                    MPI_Allreduce(
                        black_box(send_d.as_ptr()).cast(),
                        black_box(recv_d.as_mut_ptr()).cast(),
                        1,
                        MPI_UINT64_T,
                        MPI_BOR,
                        MPI_COMM_WORLD,
                    );
                }
            },
            || {
                world
                    .allreduce(
                        black_box(&send_f),
                        black_box(&mut recv_f),
                        ReduceOp::BitwiseOr,
                    )
                    .unwrap();
            },
        );
    }

    // allgatherv u8 x64: one rank contributes 64 bytes; counts/displs are
    // read-only and shared by both arms.
    {
        let send_d = [7u8; 64];
        let mut recv_d = [0u8; 64];
        let send_f = [7u8; 64];
        let mut recv_f = [0u8; 64];
        let (counts, displs) = ([64 as c_int], [0 as c_int]);
        compare(
            "allgatherv u8 x64",
            20_000,
            || {
                // SAFETY: send_d/recv_d are live, distinct 64-byte buffers; counts and
                // displs hold one entry for the one rank and outlive the call; the
                // blocking MPI_Allgatherv is done with all four when it returns.
                unsafe {
                    MPI_Allgatherv(
                        black_box(send_d.as_ptr()).cast(),
                        64,
                        MPI_UINT8_T,
                        black_box(recv_d.as_mut_ptr()).cast(),
                        counts.as_ptr(),
                        displs.as_ptr(),
                        MPI_UINT8_T,
                        MPI_COMM_WORLD,
                    );
                }
            },
            || {
                world
                    .allgatherv(black_box(&send_f), black_box(&mut recv_f), &counts, &displs)
                    .unwrap();
            },
        );
    }

    // broadcast f64: root 0 is the only rank.
    {
        let mut data_d = [1.0f64];
        let mut data_f = [1.0f64];
        compare(
            "broadcast f64",
            20_000,
            || {
                // SAFETY: data_d is a live one-element f64 buffer; the blocking
                // MPI_Bcast is done with it when it returns.
                unsafe {
                    MPI_Bcast(
                        black_box(data_d.as_mut_ptr()).cast(),
                        1,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD,
                    );
                }
            },
            || {
                world.broadcast(black_box(&mut data_f), 0).unwrap();
            },
        );
    }

    // barrier: no buffers.
    compare(
        "barrier",
        20_000,
        || {
            // SAFETY: MPI_Barrier borrows no buffers; MPI_COMM_WORLD is valid while
            // `mpi` is alive.
            unsafe {
                MPI_Barrier(MPI_COMM_WORLD);
            }
        },
        || {
            world.barrier().unwrap();
        },
    );

    // isend+irecv+wait: one self message each way.
    {
        let send_d = [1.0f64];
        let mut recv_d = [0.0f64];
        let direct = || {
            let mut rreq: MpiRequest = 0;
            let mut sreq: MpiRequest = 0;
            // SAFETY: send_d/recv_d outlive rreq/sreq; the two MPI_Wait calls below
            // complete both requests before this closure returns.
            unsafe {
                MPI_Irecv(
                    black_box(recv_d.as_mut_ptr()).cast(),
                    1,
                    MPI_DOUBLE,
                    0,
                    0,
                    MPI_COMM_WORLD,
                    &mut rreq,
                );
                MPI_Isend(
                    black_box(send_d.as_ptr()).cast(),
                    1,
                    MPI_DOUBLE,
                    0,
                    0,
                    MPI_COMM_WORLD,
                    &mut sreq,
                );
                MPI_Wait(&mut rreq, status_ignore());
                MPI_Wait(&mut sreq, status_ignore());
            }
        };

        let send_f = [1.0f64];
        let mut recv_f = [0.0f64];
        let ferrompi = || {
            let rreq = world.irecv(black_box(&mut recv_f), 0, 0).unwrap();
            let sreq = world.isend(black_box(&send_f), 0, 0).unwrap();
            rreq.wait().unwrap();
            sreq.wait().unwrap();
        };

        compare("isend+irecv+wait", 20_000, direct, ferrompi);
    }

    // 8x(isend+irecv)+waitall: 8 self messages each way, one batched wait.
    {
        let send_d = [1.0f64; BATCH];
        let mut recv_d = [0.0f64; BATCH];
        let direct = || {
            let mut reqs: [MpiRequest; 2 * BATCH] = [0; 2 * BATCH];
            let (rreqs, sreqs) = reqs.split_at_mut(BATCH);
            // SAFETY: send_d/recv_d outlive reqs; MPI_Waitall below completes every
            // request this loop posts before the closure returns.
            unsafe {
                for (i, req) in rreqs.iter_mut().enumerate() {
                    MPI_Irecv(
                        recv_d.as_mut_ptr().add(i).cast(),
                        1,
                        MPI_DOUBLE,
                        0,
                        i as c_int,
                        MPI_COMM_WORLD,
                        req,
                    );
                }
                for (i, req) in sreqs.iter_mut().enumerate() {
                    MPI_Isend(
                        send_d.as_ptr().add(i).cast(),
                        1,
                        MPI_DOUBLE,
                        0,
                        i as c_int,
                        MPI_COMM_WORLD,
                        req,
                    );
                }
                MPI_Waitall(reqs.len() as c_int, reqs.as_mut_ptr(), status_ignore());
            }
        };

        let send_f = [1.0f64; BATCH];
        let mut recv_f = [0.0f64; BATCH];
        let mut reqs_f: Vec<Request> = Vec::with_capacity(2 * BATCH);
        let ferrompi = || {
            for (i, x) in recv_f.chunks_mut(1).enumerate() {
                reqs_f.push(world.irecv(x, 0, i as i32).unwrap());
            }
            for (i, x) in send_f.chunks(1).enumerate() {
                reqs_f.push(world.isend(x, 0, i as i32).unwrap());
            }
            Request::wait_all(&mut reqs_f).unwrap();
            reqs_f.clear();
        };

        compare("8x(isend+irecv)+waitall", 5_000, direct, ferrompi);
    }

    // iallreduce+wait: direct_iallreduce against the ferrompi wrapper.
    {
        let (send_d, mut recv_d) = ([1.0f64], [0.0f64]);
        let (send_f, mut recv_f) = ([1.0f64], [0.0f64]);
        compare(
            "iallreduce+wait",
            20_000,
            || direct_iallreduce(&send_d, &mut recv_d),
            || {
                world
                    .iallreduce(black_box(&send_f), black_box(&mut recv_f), ReduceOp::Sum)
                    .unwrap()
                    .wait()
                    .unwrap();
            },
        );
    }

    // persistent start+wait: one receive/send pair to self, initialized once.
    {
        let mut recv_d = [0.0f64];
        let send_d = [1.0f64];
        let mut rreq_d: MpiRequest = 0;
        let mut sreq_d: MpiRequest = 0;
        // SAFETY: recv_d/send_d outlive rreq_d/sreq_d; both persistent requests are
        // freed with MPI_Request_free below before recv_d/send_d go out of scope.
        unsafe {
            MPI_Recv_init(
                black_box(recv_d.as_mut_ptr()).cast(),
                1,
                MPI_DOUBLE,
                0,
                0,
                MPI_COMM_WORLD,
                &mut rreq_d,
            );
            MPI_Send_init(
                black_box(send_d.as_ptr()).cast(),
                1,
                MPI_DOUBLE,
                0,
                0,
                MPI_COMM_WORLD,
                &mut sreq_d,
            );
        }

        let mut recv_f = [0.0f64];
        let send_f = [1.0f64];
        let mut recv_preq = world.recv_init(&mut recv_f, 0, 0).unwrap();
        let mut send_preq = world.send_init(&send_f, 0, 0).unwrap();

        compare(
            "persistent start+wait",
            20_000,
            || {
                // SAFETY: rreq_d/sreq_d name the persistent requests initialized
                // above; the two MPI_Wait calls complete both before this closure
                // returns, and recv_d/send_d stay alive and untouched meanwhile.
                unsafe {
                    MPI_Start(&mut rreq_d);
                    MPI_Start(&mut sreq_d);
                    MPI_Wait(&mut rreq_d, status_ignore());
                    MPI_Wait(&mut sreq_d, status_ignore());
                }
            },
            || {
                recv_preq.start().unwrap();
                send_preq.start().unwrap();
                recv_preq.wait().unwrap();
                send_preq.wait().unwrap();
            },
        );

        // SAFETY: rreq_d/sreq_d are inactive after the last MPI_Wait above;
        // MPI_Request_free releases both handles while recv_d/send_d are still in
        // scope, before the ferrompi PersistentRequests are dropped below.
        unsafe {
            MPI_Request_free(&mut rreq_d);
            MPI_Request_free(&mut sreq_d);
        }
    }

    // 8x persistent start_all+wait_all: 8 receive/send pairs to self.
    {
        let mut recv_d = [0.0f64; BATCH];
        let send_d = [1.0f64; BATCH];
        let mut reqs_d: [MpiRequest; 2 * BATCH] = [0; 2 * BATCH];
        {
            let (rreqs, sreqs) = reqs_d.split_at_mut(BATCH);
            // SAFETY: recv_d/send_d outlive reqs_d; every persistent handle this
            // loop creates is freed with MPI_Request_free below before recv_d/send_d
            // go out of scope.
            unsafe {
                for (i, req) in rreqs.iter_mut().enumerate() {
                    MPI_Recv_init(
                        recv_d.as_mut_ptr().add(i).cast(),
                        1,
                        MPI_DOUBLE,
                        0,
                        i as c_int,
                        MPI_COMM_WORLD,
                        req,
                    );
                }
                for (i, req) in sreqs.iter_mut().enumerate() {
                    MPI_Send_init(
                        send_d.as_ptr().add(i).cast(),
                        1,
                        MPI_DOUBLE,
                        0,
                        i as c_int,
                        MPI_COMM_WORLD,
                        req,
                    );
                }
            }
        }

        let mut recv_f = [0.0f64; BATCH];
        let send_f = [1.0f64; BATCH];
        let mut preqs_f: Vec<PersistentRequest> = Vec::with_capacity(2 * BATCH);
        for (i, x) in recv_f.chunks_mut(1).enumerate() {
            preqs_f.push(world.recv_init(x, 0, i as i32).unwrap());
        }
        for (i, x) in send_f.chunks(1).enumerate() {
            preqs_f.push(world.send_init(x, 0, i as i32).unwrap());
        }

        compare(
            "8x persistent start_all+wait_all",
            5_000,
            || {
                // SAFETY: reqs_d names the persistent requests initialized above;
                // MPI_Waitall below completes every one of them before this closure
                // returns.
                unsafe {
                    MPI_Startall(reqs_d.len() as c_int, reqs_d.as_mut_ptr());
                    MPI_Waitall(reqs_d.len() as c_int, reqs_d.as_mut_ptr(), status_ignore());
                }
            },
            || {
                PersistentRequest::start_all(&mut preqs_f).unwrap();
                PersistentRequest::wait_all(&mut preqs_f).unwrap();
            },
        );

        // SAFETY: every entry in reqs_d is inactive after the last MPI_Waitall
        // above; MPI_Request_free releases each persistent handle while
        // recv_d/send_d are still in scope, before the ferrompi PersistentRequests
        // are dropped below.
        unsafe {
            for req in &mut reqs_d {
                MPI_Request_free(req);
            }
        }
    }
}
