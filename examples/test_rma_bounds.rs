//! Regression for RMA target bounds checking.
//!
//! Every RMA method (`put`, `get`, `accumulate`, `get_accumulate`, `rput`,
//! `rget`, `raccumulate`, `fetch_and_op`, `compare_and_swap`) must reject an
//! out-of-bounds target — a bad rank (including -1), a negative displacement,
//! or an access past the target's exposed window — with `Err(InvalidArgument)`
//! naming the argument, before any MPI call. The count of every method is the
//! slice length, so a transfer longer than the target's window is a past-window
//! access, and a `get_accumulate` whose `result` is shorter than its `origin`
//! returns `Err(BufferSize)`. Rank 0 is the origin; rank 1 is the target.
//! Canary words around the short `result` buffer, the `result` cells of the
//! rejected `fetch_and_op` and `compare_and_swap` calls, the `Win::create`
//! window's unchanged contents and valgrind prove no out-of-bounds write
//! reaches memory.
//!
//! A transfer must also stay inside its epoch's access set. Rank 0 opens a
//! `lock` epoch on rank 1 and a `start` epoch with the group `{1}`, and every
//! method it calls on rank 0 itself, a rank of the window's communicator with
//! room in its window, returns `Err(InvalidArgument { arg: "target_rank" })`
//! before any MPI call; a `get` from rank 1 in the same epoch succeeds, and so
//! does a flush of rank 0, which takes any rank of the window's communicator.
//! A `start` group with a member outside the window's communicator returns
//! `Err(InvalidArgument { arg: "group" })` before `MPI_Win_start`, and the
//! window still takes an epoch afterwards.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_bounds
// mpi-test: np=2 valgrind skip-ok=openmpi-4

use ferrompi::{
    Communicator, Error, LockType, Mpi, ReduceOp, Request, Win, WinFenceAssert, WinPscwAssert,
};

mod common;

const CANARY: u64 = 0xC0FFEE;

const BAD_RANK: (&str, &str) = ("target_rank", "rank outside the window's communicator");
const NEGATIVE_DISP: (&str, &str) = ("target_disp", "negative displacement");
const PAST_WINDOW: (&str, &str) = ("target_disp", "access extends past the target's window");
const OUTSIDE_ACCESS: (&str, &str) = ("target_rank", "rank outside the epoch's access set");
const FOREIGN_GROUP: (&str, &str) = ("group", "member outside the window's communicator");

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

fn check_rejected_req(
    ok: &mut bool,
    label: &str,
    rank: i32,
    disp: i64,
    want: (&str, &str),
    result: ferrompi::Result<Request<'_>>,
) {
    match result {
        Err(e) if is_rejection(&e, want) => {}
        Err(e) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) returned unexpected error: {e}");
            *ok = false;
        }
        Ok(_) => {
            eprintln!("FAIL: {label} ({rank}, {disp}) did not return Err(InvalidArgument)");
            *ok = false;
        }
    }
}

/// Runs the seven bulk-access methods (`put`, `accumulate`, `get`,
/// `get_accumulate`, `rput`, `rget`, `raccumulate`) with a 4-element buffer,
/// and the two single-element methods (`fetch_and_op`, `compare_and_swap`),
/// against every out-of-bounds `(target_rank, target_disp)` case: the six that
/// complete with a fence in one fence epoch, the three request-returning ones
/// in a `lock_all` epoch (MPI allows request-based RMA only in a passive-target
/// epoch, and a rejected call makes no MPI call). Only rank 0 issues calls;
/// every one must return `Err(InvalidArgument)` naming the rejected argument.
/// Each rejected `fetch_and_op` and `compare_and_swap` call gets its own
/// `result` cell, set to `CANARY`, which must still hold it after the epoch: a
/// rejected call writes nothing into caller storage. `len` is `win`'s exposed
/// length in elements. Returns the aggregate local verdict (`true` on every
/// rank but 0, which has no cases to fail).
fn run_bounds_cases(world: &Communicator, win: &mut Win<u64>, name: &str, len: i64) -> bool {
    let size = world.size();
    let mut ok = true;
    let origin = [1u64, 2, 3, 4];
    // Each epoch call borrows its buffer until the epoch closes, so every case
    // gets its own `get` destination and `get_accumulate` result.
    let bulk_cases = [
        (1, (len - 2).max(0), PAST_WINDOW),
        (1, -1, NEGATIVE_DISP),
        (size, 0, BAD_RANK),
        (-1, 0, BAD_RANK),
    ];
    let mut epoch_get_bufs = bulk_cases.map(|_| [0u64; 4]);
    let mut epoch_results = bulk_cases.map(|_| [0u64; 4]);
    let mut request_get_bufs = bulk_cases.map(|_| [0u64; 4]);
    let single_cases = [
        (1, len, PAST_WINDOW),
        (1, -1, NEGATIVE_DISP),
        (size, 0, BAD_RANK),
        (-1, 0, BAD_RANK),
    ];
    let (fetch_origin, cas_origin, cas_compare) = (1u64, 2u64, 1u64);
    let mut fetch_results = single_cases.map(|_| CANARY);
    let mut cas_results = single_cases.map(|_| CANARY);

    win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
        if world.rank() == 0 {
            for (&(rank, disp, want), (epoch_get_buf, epoch_result)) in bulk_cases
                .iter()
                .zip(epoch_get_bufs.iter_mut().zip(epoch_results.iter_mut()))
            {
                let label = |op: &str| format!("{name}/{op}");
                check_rejected(
                    &mut ok,
                    &label("put"),
                    rank,
                    disp,
                    want,
                    e.put(&origin, rank, disp),
                );
                check_rejected(
                    &mut ok,
                    &label("accumulate"),
                    rank,
                    disp,
                    want,
                    e.accumulate(&origin, rank, disp, ReduceOp::Sum),
                );
                check_rejected(
                    &mut ok,
                    &label("get"),
                    rank,
                    disp,
                    want,
                    e.get(epoch_get_buf, rank, disp),
                );
                check_rejected(
                    &mut ok,
                    &label("get_accumulate"),
                    rank,
                    disp,
                    want,
                    e.get_accumulate(&origin, epoch_result, rank, disp, ReduceOp::Sum),
                );
            }

            for (&(rank, disp, want), (fetch_result, cas_result)) in single_cases
                .iter()
                .zip(fetch_results.iter_mut().zip(cas_results.iter_mut()))
            {
                let label = |op: &str| format!("{name}/{op}");
                check_rejected(
                    &mut ok,
                    &label("fetch_and_op"),
                    rank,
                    disp,
                    want,
                    e.fetch_and_op(&fetch_origin, fetch_result, rank, disp, ReduceOp::Sum),
                );
                check_rejected(
                    &mut ok,
                    &label("compare_and_swap"),
                    rank,
                    disp,
                    want,
                    e.compare_and_swap(&cas_origin, &cas_compare, cas_result, rank, disp),
                );
            }
        }
        Ok(())
    })
    .unwrap_or_else(|e| panic!("{name}: fence epoch failed: {e}"));

    if world.rank() == 0 {
        win.lock_all(|e| {
            for (&(rank, disp, want), request_get_buf) in
                bulk_cases.iter().zip(request_get_bufs.iter_mut())
            {
                let label = |op: &str| format!("{name}/{op}");
                check_rejected_req(
                    &mut ok,
                    &label("rput"),
                    rank,
                    disp,
                    want,
                    e.rput(&origin, rank, disp),
                );
                check_rejected_req(
                    &mut ok,
                    &label("rget"),
                    rank,
                    disp,
                    want,
                    e.rget(request_get_buf, rank, disp),
                );
                check_rejected_req(
                    &mut ok,
                    &label("raccumulate"),
                    rank,
                    disp,
                    want,
                    e.raccumulate(&origin, rank, disp, ReduceOp::Sum),
                );
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("{name}: lock_all epoch failed: {e}"));
    }

    if world.rank() == 0 {
        if fetch_results != [CANARY; 4] {
            eprintln!(
                "FAIL: {name}/fetch_and_op: a rejected call wrote into its result cell: {fetch_results:x?}"
            );
            ok = false;
        }
        if cas_results != [CANARY; 4] {
            eprintln!(
                "FAIL: {name}/compare_and_swap: a rejected call wrote into its result cell: {cas_results:x?}"
            );
            ok = false;
        }
    }

    // The rejected calls left nothing pending: the window still takes an epoch.
    win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |_| {
        Ok(())
    })
    .unwrap_or_else(|e| panic!("{name}: fence after the rejected calls failed: {e}"));

    ok
}

/// Rank 0 opens an epoch whose access set is rank 1 alone, a `lock` on rank 1
/// and then a `start` with the group `{1}`, and calls every transfer method of
/// the epoch's handle on rank 0 itself: a rank of the window's communicator,
/// with room in its window, that is outside the access set. Each call must
/// return `Err(InvalidArgument { arg: "target_rank" })` before any MPI call
/// (the request calls only in the `lock` epoch). A `get` from rank 1, inside
/// the access set, must succeed in both epochs and read rank 1's 7s, so the
/// rejection comes from the access set. Rank 1 only exposes its window for the
/// `start` epoch. `win` is the 4-element window rank 1 filled with 7.
fn run_access_set_cases(world: &Communicator, win: &mut Win<u64>) -> bool {
    let mut ok = true;
    let rank = world.rank();
    let peers = world.group().expect("group failed");
    let (to_rank_1, to_rank_0) = (
        peers.include(&[1]).expect("include failed"),
        peers.include(&[0]).expect("include failed"),
    );

    let origin = [1u64, 2, 3, 4];
    let (fetch_origin, cas_origin, cas_compare) = (1u64, 2u64, 1u64);
    // Each epoch call borrows its buffers until the epoch closes, so every
    // call has its own `get` destination and result.
    let (mut get_buf, mut result, mut rget_buf, mut control) =
        ([0u64; 4], [0u64; 4], [0u64; 4], [0u64; 4]);
    let (mut fetch_result, mut cas_result) = (CANARY, CANARY);

    if rank == 0 {
        win.lock(LockType::Shared, 1, |e| {
            let label = |op: &str| format!("lock(1)/{op}");
            let results = [
                ("put", e.put(&origin, 0, 0)),
                ("accumulate", e.accumulate(&origin, 0, 0, ReduceOp::Sum)),
                ("get", e.get(&mut get_buf, 0, 0)),
                (
                    "get_accumulate",
                    e.get_accumulate(&origin, &mut result, 0, 0, ReduceOp::Sum),
                ),
                (
                    "fetch_and_op",
                    e.fetch_and_op(&fetch_origin, &mut fetch_result, 0, 0, ReduceOp::Sum),
                ),
                (
                    "compare_and_swap",
                    e.compare_and_swap(&cas_origin, &cas_compare, &mut cas_result, 0, 0),
                ),
            ];
            for (op, result) in results {
                check_rejected(&mut ok, &label(op), 0, 0, OUTSIDE_ACCESS, result);
            }
            check_rejected_req(
                &mut ok,
                &label("rput"),
                0,
                0,
                OUTSIDE_ACCESS,
                e.rput(&origin, 0, 0),
            );
            check_rejected_req(
                &mut ok,
                &label("rget"),
                0,
                0,
                OUTSIDE_ACCESS,
                e.rget(&mut rget_buf, 0, 0),
            );
            check_rejected_req(
                &mut ok,
                &label("raccumulate"),
                0,
                0,
                OUTSIDE_ACCESS,
                e.raccumulate(&origin, 0, 0, ReduceOp::Sum),
            );
            if let Err(err) = e.get(&mut control, 1, 0) {
                eprintln!("FAIL: lock(1)/get from the locked rank returned {err}");
                ok = false;
            }
            // A flush takes any rank of the window's communicator, in the
            // access set or not.
            if let Err(err) = e.flush(0).and_then(|()| e.flush_local(0)) {
                eprintln!("FAIL: lock(1)/flush of rank 0 returned {err}");
                ok = false;
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("access set: lock epoch failed: {e}"));
        if control != [7u64; 4] {
            eprintln!(
                "FAIL: lock(1)/get from the locked rank read {control:?}, expected [7, 7, 7, 7]"
            );
            ok = false;
        }
        if (fetch_result, cas_result) != (CANARY, CANARY) {
            eprintln!("FAIL: lock(1): a rejected call wrote into its result cell");
            ok = false;
        }
    }
    world
        .barrier()
        .expect("access set: barrier after the lock epoch failed");

    let (mut get_buf, mut result, mut control) = ([0u64; 4], [0u64; 4], [0u64; 4]);
    let (mut fetch_result, mut cas_result) = (CANARY, CANARY);
    if rank == 0 {
        win.start(&to_rank_1, WinPscwAssert::default(), |e| {
            let label = |op: &str| format!("start({{1}})/{op}");
            let results = [
                ("put", e.put(&origin, 0, 0)),
                ("accumulate", e.accumulate(&origin, 0, 0, ReduceOp::Sum)),
                ("get", e.get(&mut get_buf, 0, 0)),
                (
                    "get_accumulate",
                    e.get_accumulate(&origin, &mut result, 0, 0, ReduceOp::Sum),
                ),
                (
                    "fetch_and_op",
                    e.fetch_and_op(&fetch_origin, &mut fetch_result, 0, 0, ReduceOp::Sum),
                ),
                (
                    "compare_and_swap",
                    e.compare_and_swap(&cas_origin, &cas_compare, &mut cas_result, 0, 0),
                ),
            ];
            for (op, result) in results {
                check_rejected(&mut ok, &label(op), 0, 0, OUTSIDE_ACCESS, result);
            }
            if let Err(err) = e.get(&mut control, 1, 0) {
                eprintln!("FAIL: start({{1}})/get from the group returned {err}");
                ok = false;
            }
            Ok(())
        })
        .unwrap_or_else(|e| panic!("access set: start epoch failed: {e}"));
        if control != [7u64; 4] {
            eprintln!(
                "FAIL: start({{1}})/get from the group read {control:?}, expected [7, 7, 7, 7]"
            );
            ok = false;
        }
        if (fetch_result, cas_result) != (CANARY, CANARY) {
            eprintln!("FAIL: start({{1}}): a rejected call wrote into its result cell");
            ok = false;
        }
    } else if rank == 1 {
        win.post(&to_rank_0, WinPscwAssert::default())
            .expect("access set: post failed");
        win.wait_exposure()
            .expect("access set: wait_exposure failed");
    }
    world
        .barrier()
        .expect("access set: barrier after the start epoch failed");
    ok
}

/// Each rank builds a window on a communicator of its own, a split by rank, so
/// the world group has a member the window does not. `start` with that group
/// must return `Err(InvalidArgument { arg: "group" })` before `MPI_Win_start`,
/// with no communication, and the refused call must leave no epoch open.
fn run_foreign_group_case(world: &Communicator) -> bool {
    let mut ok = true;
    let own = world
        .split(Some(world.rank()), 0)
        .expect("split failed")
        .expect("a coloured rank has a communicator");
    let mut win = Win::<u64>::allocate(&own, 4).expect("Win::allocate failed");
    let everyone = world.group().expect("group failed");

    let result = win.start(&everyone, WinPscwAssert::default(), |_| Ok(()));
    if !matches!(&result, Err(e) if is_rejection(e, FOREIGN_GROUP)) {
        eprintln!(
            "FAIL: start with a group member outside the window's communicator did not return Err(InvalidArgument {{ arg: \"group\" }}): {result:?}"
        );
        ok = false;
    }
    if let Err(err) = win.fence(WinFenceAssert::default(), WinFenceAssert::default(), |_| {
        Ok(())
    }) {
        eprintln!("FAIL: fence after the refused start returned {err}");
        ok = false;
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
    if !run_bounds_cases(&world, &mut win_a, "alloc4", 4) {
        local_ok = false;
    }
    if !run_access_set_cases(&world, &mut win_a) {
        local_ok = false;
    }
    if !run_foreign_group_case(&world) {
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

    // The count is the slice length. A 5-element transfer into the 4-element
    // alloc4 window is a past-window access, and a get_accumulate whose result
    // is shorter than its origin is a BufferSize error. The 1-element result is
    // surrounded by canary words rank 0 checks after the closing fence.
    let mut small = Box::new(Small {
        buf: [0u64; 1],
        canary: [CANARY; 7],
    });
    let four = [1u64, 2, 3, 4];
    let five = [1u64, 2, 3, 4, 5];
    let mut five_get = [0u64; 5];
    win_a.fence(WinFenceAssert::default(), WinFenceAssert::default(), |e| {
        if rank == 0 {
            let result = e.put(&five, 1, 0);
            if !matches!(&result, Err(e) if is_rejection(e, PAST_WINDOW)) {
                eprintln!(
                    "FAIL: put of 5 elements into a 4-element window did not return Err(InvalidArgument {{ arg: \"target_disp\" }}): {result:?}"
                );
                local_ok = false;
            }
            let result = e.get(&mut five_get, 1, 0);
            if !matches!(&result, Err(e) if is_rejection(e, PAST_WINDOW)) {
                eprintln!(
                    "FAIL: get of 5 elements from a 4-element window did not return Err(InvalidArgument {{ arg: \"target_disp\" }}): {result:?}"
                );
                local_ok = false;
            }
            let result = e.accumulate(&five, 1, 0, ReduceOp::Sum);
            if !matches!(&result, Err(e) if is_rejection(e, PAST_WINDOW)) {
                eprintln!(
                    "FAIL: accumulate of 5 elements into a 4-element window did not return Err(InvalidArgument {{ arg: \"target_disp\" }}): {result:?}"
                );
                local_ok = false;
            }
            let result = e.get_accumulate(&four, &mut small.buf, 1, 0, ReduceOp::Sum);
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
