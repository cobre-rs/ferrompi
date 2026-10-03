//! Regression for RMA target bounds checking.
//!
//! Every `Win` RMA method (`put`, `get`, `accumulate`, `get_accumulate`,
//! `rput`, `rget`, `raccumulate`, `fetch_and_op`, `compare_and_swap`) must
//! reject an out-of-bounds target — a bad rank (including -1), a negative
//! displacement, an access past the target's exposed window, a negative
//! `target_count`, or a buffer length that does not match `target_count` —
//! with `Err(InvalidArgument)` or `Err(BufferSize)` naming the argument,
//! before any MPI call. Rank 0 is the origin; rank 1 is the target. Canary
//! words around the length-mismatch buffer, the `Win::create` window's
//! unchanged contents and valgrind prove no out-of-bounds write reaches memory.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_bounds
// mpi-test: np=2 valgrind skip-ok=openmpi-4

use ferrompi::{
    Communicator, Error, Mpi, PendingFetchResult, ReduceOp, Request, Win, WinFenceAssert,
};

mod common;

const CANARY: u64 = 0xC0FFEE;

const BAD_RANK: (&str, &str) = ("target_rank", "rank outside the window's communicator");
const NEGATIVE_DISP: (&str, &str) = ("target_disp", "negative displacement");
const NEGATIVE_COUNT: (&str, &str) = ("target_count", "negative count");
const PAST_WINDOW: (&str, &str) = ("target_disp", "access extends past the target's window");

fn is_rejection(err: &Error, want: (&str, &str)) -> bool {
    matches!(err, Error::InvalidArgument { arg, reason, .. } if (*arg, *reason) == want)
}

#[repr(C)]
struct Small {
    buf: [u64; 1],
    canary: [u64; 7],
}

fn check_rejected(
    ok: &mut bool,
    label: &str,
    rank: i32,
    disp: i64,
    want: (&str, &str),
    result: ferrompi::Result<()>,
) {
    if !matches!(&result, Err(e) if is_rejection(e, want)) {
        eprintln!(
            "FAIL: {label} ({rank}, {disp}) did not return Err(InvalidArgument {{ arg: {:?}, reason: {:?} }}): {result:?}",
            want.0, want.1
        );
        *ok = false;
    }
}

fn check_rejected_req<'a>(
    ok: &mut bool,
    label: &str,
    rank: i32,
    disp: i64,
    want: (&str, &str),
    result: ferrompi::Result<Request<'a>>,
    stray: &mut Vec<Request<'a>>,
) {
    match result {
        Err(e) if is_rejection(&e, want) => {}
        Err(e) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) returned unexpected error: {e}");
            *ok = false;
        }
        Ok(req) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidArgument)");
            *ok = false;
            stray.push(req);
        }
    }
}

fn check_rejected_pending(
    ok: &mut bool,
    label: &str,
    rank: i32,
    disp: i64,
    want: (&str, &str),
    result: ferrompi::Result<PendingFetchResult<u64>>,
    stray: &mut Vec<PendingFetchResult<u64>>,
) {
    match result {
        Err(e) if is_rejection(&e, want) => {}
        Err(e) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) returned unexpected error: {e}");
            *ok = false;
        }
        Ok(p) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidArgument)");
            *ok = false;
            stray.push(p);
        }
    }
}

/// Runs the seven bulk-access methods (`put`, `accumulate`, `get`,
/// `get_accumulate`, `rput`, `rget`, `raccumulate`) with a 4-element
/// buffer and `target_count` 4, and the two single-element methods
/// (`fetch_and_op`, `compare_and_swap`), against every out-of-bounds
/// `(target_rank, target_disp)` case: the six that complete with a fence in one
/// fence epoch, the three request-returning ones outside it (MPI allows
/// request-based RMA only in a passive-target epoch, and a rejected call
/// makes no MPI call). Only rank 0 issues calls; every one must return
/// `Err(InvalidArgument)` naming the rejected argument. `len` is
/// `win`'s exposed length in elements. Returns the aggregate local
/// verdict (`true` on every rank but 0, which has no cases to fail).
fn run_bounds_cases(world: &Communicator, win: &mut Win<u64>, name: &str, len: i64) -> bool {
    let size = world.size();
    let mut ok = true;
    let mut stray_requests: Vec<Request> = Vec::new();
    let mut stray_pending: Vec<PendingFetchResult<u64>> = Vec::new();
    let origin = [1u64, 2, 3, 4];
    let mut get_buf = [0u64; 4];
    let mut result = [0u64; 4];
    let bulk_cases = [
        (1, (len - 2).max(0), PAST_WINDOW),
        (1, -1, NEGATIVE_DISP),
        (size, 0, BAD_RANK),
        (-1, 0, BAD_RANK),
    ];

    win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
        if world.rank() == 0 {
            for &(rank, disp, want) in &bulk_cases {
                let label = |op: &str| format!("{name}/{op}");
                check_rejected(
                    &mut ok,
                    &label("put"),
                    rank,
                    disp,
                    want,
                    e.put(&origin, rank, disp, 4),
                );
                check_rejected(
                    &mut ok,
                    &label("accumulate"),
                    rank,
                    disp,
                    want,
                    e.accumulate(&origin, rank, disp, 4, ReduceOp::Sum),
                );
                check_rejected(
                    &mut ok,
                    &label("get"),
                    rank,
                    disp,
                    want,
                    e.get(&mut get_buf, rank, disp, 4),
                );
                check_rejected(
                    &mut ok,
                    &label("get_accumulate"),
                    rank,
                    disp,
                    want,
                    e.get_accumulate(&origin, &mut result, rank, disp, 4, ReduceOp::Sum),
                );
            }

            for &(rank, disp, want) in &[
                (1, len, PAST_WINDOW),
                (1, -1, NEGATIVE_DISP),
                (size, 0, BAD_RANK),
                (-1, 0, BAD_RANK),
            ] {
                let label = |op: &str| format!("{name}/{op}");
                check_rejected_pending(
                    &mut ok,
                    &label("fetch_and_op"),
                    rank,
                    disp,
                    want,
                    e.fetch_and_op(1u64, rank, disp, ReduceOp::Sum),
                    &mut stray_pending,
                );
                check_rejected_pending(
                    &mut ok,
                    &label("compare_and_swap"),
                    rank,
                    disp,
                    want,
                    e.compare_and_swap(2u64, 1u64, rank, disp),
                    &mut stray_pending,
                );
            }
        }
        Ok(())
    })
    .unwrap_or_else(|e| panic!("{name}: fence epoch failed: {e}"));

    if world.rank() == 0 {
        for &(rank, disp, want) in &bulk_cases {
            let label = |op: &str| format!("{name}/{op}");
            check_rejected_req(
                &mut ok,
                &label("rput"),
                rank,
                disp,
                want,
                win.rput(&origin, rank, disp, 4),
                &mut stray_requests,
            );
            check_rejected_req(
                &mut ok,
                &label("rget"),
                rank,
                disp,
                want,
                win.rget(&mut get_buf, rank, disp, 4),
                &mut stray_requests,
            );
            check_rejected_req(
                &mut ok,
                &label("raccumulate"),
                rank,
                disp,
                want,
                win.raccumulate(&origin, rank, disp, 4, ReduceOp::Sum),
                &mut stray_requests,
            );
        }
    }

    // Red-run only: an unexpected `Ok` request/pending result is completed
    // here, after the fence epoch has closed.
    for req in stray_requests {
        let _ = req.wait();
    }
    for p in stray_pending {
        // SAFETY: the fence epoch has closed.
        let _ = unsafe { p.resolve() };
    }

    // The rejected calls left nothing pending: the window still takes an epoch.
    win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |_| {
        Ok(())
    })
    .unwrap_or_else(|e| panic!("{name}: fence after the rejected calls failed: {e}"));

    ok
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_bounds requires at least 2 processes, got {size}"
    );

    let mut local_ok = true;

    // Window 1: 4-element MPI-allocated window, filled with 7 by rank 1.
    let mut win_a = Win::<u64>::allocate(&world, 4).expect("Win::allocate(4) failed");
    if rank == 1 {
        win_a.local_slice_mut().fill(7);
    }
    if !run_bounds_cases(&world, &mut win_a, "alloc4", 4) {
        local_ok = false;
    }

    // Window 2: zero-length MPI-allocated window.
    let mut win_b = Win::<u64>::allocate(&world, 0).expect("Win::allocate(0) failed");
    if !run_bounds_cases(&world, &mut win_b, "alloc0", 0) {
        local_ok = false;
    }
    drop(win_b);

    // Window 3: Win::create over an owned four-word vector, every rank
    // supplying its own. OpenMPI 4.x with `--btl=self,tcp` rejects
    // Win::create over caller-owned memory (MPI_ERR_WIN); skip gracefully
    // (see test_rma_win_create.rs).
    let mut win_c = match Win::create(&world, vec![7u64; 4]) {
        Ok(w) => Some(w),
        Err(Error::Mpi {
            class: ferrompi::MpiErrorClass::Win,
            ..
        }) => {
            common::skip(
                &world,
                "Win::create returned MPI_ERR_WIN — likely OpenMPI 4.x with a \
                 BTL that does not support one-sided over caller-owned memory \
                 (e.g., --btl=self,tcp in CI).",
            );
            None
        }
        Err(e) => panic!("Win::create failed: {e}"),
    };
    if let Some(win_c) = &mut win_c {
        if !run_bounds_cases(&world, win_c, "create", 4) {
            local_ok = false;
        }
    }

    // Length mismatches against the alloc4 window: a 1-element buffer
    // surrounded by canary words rank 0 checks after the closing fence.
    let mut small = Box::new(Small {
        buf: [0u64; 1],
        canary: [CANARY; 7],
    });
    let four = [1u64, 2, 3, 4];
    win_a.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
        if rank == 0 {
            let result = e.put(&four, 1, 0, 2);
        if !matches!(
            result,
            Err(Error::BufferSize {
                arg: "origin",
                required: 2,
                actual: 4,
                ..
            })
        ) {
            eprintln!(
                "FAIL: put target_count 2 != origin.len() 4 did not return Err(BufferSize {{ arg: \"origin\", required: 2, actual: 4 }}): {result:?}"
            );
            local_ok = false;
        }
        let result = e.put(&four, 1, 0, -4);
        if !matches!(&result, Err(e) if is_rejection(e, NEGATIVE_COUNT)) {
            eprintln!(
                "FAIL: put with negative target_count did not return Err(InvalidArgument {{ arg: \"target_count\" }}): {result:?}"
            );
            local_ok = false;
        }
        let result = e.get(&mut small.buf, 1, 0, 4);
        if !matches!(
            result,
            Err(Error::BufferSize {
                arg: "origin",
                required: 4,
                actual: 1,
                ..
            })
        ) {
            eprintln!(
                "FAIL: get of 4 elements into a 1-element origin did not return Err(BufferSize {{ arg: \"origin\", required: 4, actual: 1 }}): {result:?}"
            );
            local_ok = false;
        }
        let result = e.get_accumulate(&four, &mut small.buf, 1, 0, 4, ReduceOp::Sum);
        if !matches!(
            result,
            Err(Error::BufferSize {
                arg: "result",
                required: 4,
                actual: 1,
                ..
            })
        ) {
            eprintln!(
                "FAIL: get_accumulate with a 1-element result did not return Err(BufferSize {{ arg: \"result\", required: 4, actual: 1 }}): {result:?}"
            );
            local_ok = false;
        }
        }
        Ok(())
    })
    .expect("length-mismatch fence epoch failed");

    if rank == 0 && small.canary != [CANARY; 7] {
        eprintln!(
            "FAIL: 1-element buffer canary corrupted: {:x?}",
            small.canary
        );
        local_ok = false;
    }

    // Afterwards: rank 1 checks that the allocate window and the create
    // window still read 7. Only rank 1's window is ever a valid RMA target
    // above, so only its copy matters.
    if rank == 1 {
        if win_a.local_slice() != [7u64; 4] {
            eprintln!(
                "FAIL: alloc4 window mutated: expected [7, 7, 7, 7], got {:?}",
                win_a.local_slice()
            );
            local_ok = false;
        }
        if let Some(win_c) = &win_c {
            if win_c.local_slice() != [7u64; 4] {
                eprintln!(
                    "FAIL: create window mutated: expected [7, 7, 7, 7], got {:?}",
                    win_c.local_slice()
                );
                local_ok = false;
            }
        }
    }

    // Freeing the create window is collective; its buffer must come back
    // exactly as created.
    if let Some(win_c) = win_c {
        match win_c.into_buffer() {
            Ok(Some(v)) if v == [7u64; 4] => {}
            other => {
                eprintln!(
                    "FAIL: create window into_buffer returned {other:?}, expected Ok(Some([7, 7, 7, 7]))"
                );
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_rma_bounds");

    if rank == 0 {
        println!("PASS: test_rma_bounds");
    }
}
