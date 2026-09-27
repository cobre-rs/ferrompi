//! Regression for RMA target bounds checking.
//!
//! Every `Win` RMA method (`put`, `get`, `accumulate`, `get_accumulate`,
//! `rput`, `rget`, `raccumulate`, `fetch_and_op`, `compare_and_swap`) must
//! reject an out-of-bounds target — a bad rank (including -1), a negative
//! displacement, an access past the target's exposed window, or a buffer
//! length that does not match `target_count` — with `Err(InvalidBuffer)`
//! before any MPI call. Rank 0 is the origin; rank 1 is the target. Canary
//! words around every buffer prove no out-of-bounds write reaches memory.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_bounds
// mpi-test: np=2 valgrind skip-ok=openmpi-4

use ferrompi::{
    Communicator, Error, Mpi, PendingFetchResult, ReduceOp, Request, Win, WinFenceAssert,
};

mod common;

const CANARY: u64 = 0xC0FFEE;

#[repr(C)]
struct Frame {
    win: [u64; 4],
    canary: [u64; 8],
}

#[repr(C)]
struct Small {
    buf: [u64; 1],
    canary: [u64; 7],
}

fn check_rejected(ok: &mut bool, label: &str, rank: i32, disp: i64, result: ferrompi::Result<()>) {
    if !matches!(result, Err(Error::InvalidBuffer)) {
        eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidBuffer): {result:?}");
        *ok = false;
    }
}

fn check_rejected_req(
    ok: &mut bool,
    label: &str,
    rank: i32,
    disp: i64,
    result: ferrompi::Result<Request>,
    stray: &mut Vec<Request>,
) {
    match result {
        Err(Error::InvalidBuffer) => {}
        Err(e) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) returned unexpected error: {e}");
            *ok = false;
        }
        Ok(req) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidBuffer)");
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
    result: ferrompi::Result<PendingFetchResult<u64>>,
    stray: &mut Vec<PendingFetchResult<u64>>,
) {
    match result {
        Err(Error::InvalidBuffer) => {}
        Err(e) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) returned unexpected error: {e}");
            *ok = false;
        }
        Ok(p) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidBuffer)");
            *ok = false;
            stray.push(p);
        }
    }
}

/// Runs the seven bulk-access methods (`put`, `accumulate`, `get`,
/// `get_accumulate`, `rput`, `rget`, `raccumulate`) with a 4-element
/// buffer and `target_count` 4, and the two single-element methods
/// (`fetch_and_op`, `compare_and_swap`), against every out-of-bounds
/// `(target_rank, target_disp)` case in one fence epoch. Only rank 0
/// issues calls; every one must return `Err(InvalidBuffer)`. `len` is
/// `win`'s exposed length in elements. Returns the aggregate local
/// verdict (`true` on every rank but 0, which has no cases to fail).
fn run_bounds_cases(world: &Communicator, win: &Win<'_, u64>, name: &str, len: i64) -> bool {
    let size = world.size();
    let mut ok = true;
    let mut stray_requests: Vec<Request> = Vec::new();
    let mut stray_pending: Vec<PendingFetchResult<u64>> = Vec::new();

    win.fence(WinFenceAssert::default())
        .unwrap_or_else(|e| panic!("{name}: opening fence failed: {e}"));

    if world.rank() == 0 {
        let origin = [1u64, 2, 3, 4];
        let mut get_buf = [0u64; 4];
        let mut result = [0u64; 4];

        for &(rank, disp) in &[(1, (len - 2).max(0)), (1, -1), (size, 0), (-1, 0)] {
            let label = |op: &str| format!("{name}/{op}");
            check_rejected(
                &mut ok,
                &label("put"),
                rank,
                disp,
                win.put(&origin, rank, disp, 4),
            );
            check_rejected(
                &mut ok,
                &label("accumulate"),
                rank,
                disp,
                win.accumulate(&origin, rank, disp, 4, ReduceOp::Sum),
            );
            check_rejected(
                &mut ok,
                &label("get"),
                rank,
                disp,
                win.get(&mut get_buf, rank, disp, 4),
            );
            check_rejected(
                &mut ok,
                &label("get_accumulate"),
                rank,
                disp,
                win.get_accumulate(&origin, &mut result, rank, disp, 4, ReduceOp::Sum),
            );
            check_rejected_req(
                &mut ok,
                &label("rput"),
                rank,
                disp,
                win.rput(&origin, rank, disp, 4),
                &mut stray_requests,
            );
            check_rejected_req(
                &mut ok,
                &label("rget"),
                rank,
                disp,
                win.rget(&mut get_buf, rank, disp, 4),
                &mut stray_requests,
            );
            check_rejected_req(
                &mut ok,
                &label("raccumulate"),
                rank,
                disp,
                win.raccumulate(&origin, rank, disp, 4, ReduceOp::Sum),
                &mut stray_requests,
            );
        }

        for &(rank, disp) in &[(1, len), (1, -1), (size, 0), (-1, 0)] {
            let label = |op: &str| format!("{name}/{op}");
            check_rejected_pending(
                &mut ok,
                &label("fetch_and_op"),
                rank,
                disp,
                win.fetch_and_op(1u64, rank, disp, ReduceOp::Sum),
                &mut stray_pending,
            );
            check_rejected_pending(
                &mut ok,
                &label("compare_and_swap"),
                rank,
                disp,
                win.compare_and_swap(2u64, 1u64, rank, disp),
                &mut stray_pending,
            );
        }
    }

    win.fence(WinFenceAssert::default())
        .unwrap_or_else(|e| panic!("{name}: closing fence failed: {e}"));

    // Red-run only: an unexpected `Ok` request/pending result must not be
    // waited or dropped before the closing fence above (undefined per the
    // MPI standard); it is safe now that the epoch is closed.
    for req in stray_requests {
        let _ = req.wait();
    }
    for p in stray_pending {
        // SAFETY: the closing fence above has closed the epoch.
        let _ = unsafe { p.resolve() };
    }

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
    if !run_bounds_cases(&world, &win_a, "alloc4", 4) {
        local_ok = false;
    }

    // Window 2: zero-length MPI-allocated window.
    let win_b = Win::<u64>::allocate(&world, 0).expect("Win::allocate(0) failed");
    if !run_bounds_cases(&world, &win_b, "alloc0", 0) {
        local_ok = false;
    }
    drop(win_b);

    // Window 3: Win::create over the first four words of a boxed canary
    // frame, every rank supplying its own frame. OpenMPI 4.x with
    // `--btl=self,tcp` rejects Win::create over caller-owned memory
    // (MPI_ERR_WIN); skip gracefully (see test_rma_win_create.rs).
    let mut frame = Box::new(Frame {
        win: [7u64; 4],
        canary: [CANARY; 8],
    });
    let win_c = match Win::create(&world, &mut frame.win) {
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
    if let Some(win_c) = &win_c {
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
    win_a
        .fence(WinFenceAssert::default())
        .expect("length-mismatch opening fence failed");
    if rank == 0 {
        let four = [1u64, 2, 3, 4];
        if win_a.put(&four, 1, 0, 2).is_ok() {
            eprintln!("FAIL: put target_count 2 != origin.len() 4 accepted");
            local_ok = false;
        }
        if win_a.put(&four, 1, 0, -4).is_ok() {
            eprintln!("FAIL: put with negative target_count accepted");
            local_ok = false;
        }
        if win_a.get(&mut small.buf, 1, 0, 4).is_ok() {
            eprintln!("FAIL: get of 4 elements into a 1-element origin accepted");
            local_ok = false;
        }
        if win_a
            .get_accumulate(&four, &mut small.buf, 1, 0, 4, ReduceOp::Sum)
            .is_ok()
        {
            eprintln!("FAIL: get_accumulate with a 1-element result accepted");
            local_ok = false;
        }
    }
    win_a
        .fence(WinFenceAssert::default())
        .expect("length-mismatch closing fence failed");

    if rank == 0 && small.canary != [CANARY; 7] {
        eprintln!(
            "FAIL: 1-element buffer canary corrupted: {:x?}",
            small.canary
        );
        local_ok = false;
    }

    // Afterwards: rank 1 checks that the allocate window still reads 7 and
    // that the create window and its canary are unchanged. Only rank 1's
    // window is ever a valid RMA target above, so only its copy matters.
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
            if frame.canary != [CANARY; 8] {
                eprintln!("FAIL: create window canary corrupted: {:x?}", frame.canary);
                local_ok = false;
            }
        }
    }

    common::check(&world, local_ok, "test_rma_bounds");

    if rank == 0 {
        println!("PASS: test_rma_bounds");
    }
}
