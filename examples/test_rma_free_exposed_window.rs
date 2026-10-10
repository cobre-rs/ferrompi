//! A window with an open PSCW exposure epoch is never freed: a peer may still
//! write into it until the epoch closes (MPI-4.1 §12.2.5).
//!
//! 1. Rank 0 exposes a window with `post` and rank 1 runs `start`, a `put` and
//!    the closing `complete`. `wait_exposure` closes the epoch, so `into_buffer`
//!    frees the window and returns `Ok(None)` (the window is MPI-allocated),
//!    with the peer's data in place.
//! 2. The same, with a `test_exposure` that returns `true` closing the epoch.
//! 3. Each rank exposes a window on a communicator of its own, so that a free
//!    could not block, runs the matching `start` and `complete`, and calls
//!    `into_buffer` with no `wait` yet: it returns `Error::InvalidState`
//!    without calling `MPI_Win_free` (Open MPI 5 would free the window and
//!    return `Ok`), and the window it consumed is leaked.
//! 4. With a `Win::create` window, the epoch closed by `wait_exposure` lets
//!    `into_buffer` return `Ok(Some(v))` with the peer's data. Then rank 0
//!    exposes another `Win::create` window and a `Win::allocate` window, calls
//!    `into_buffer` on the first and drops the second, both with the epoch
//!    open: both are leaked, not freed. Rank 1 puts into both, and a victim
//!    allocation of the size of the leaked buffer is unchanged, which shows
//!    that nothing frees or reuses it. Whether the puts have landed is not
//!    observed: completion at the target needs the `wait` that rank 0 never
//!    calls (MPI-4.1 §12.5.2).
//!
//! Parts 3 and 4 leave windows alive, so both ranks skip `MPI_Finalize` (rank 1
//! never frees the two windows that rank 0 left exposed, because the free is
//! collective and rank 0 never calls it). No check after the drop can use
//! `common::check`, whose allreduce is refused by then. A failed check prints
//! `FAIL:` and exits with status 2, which the runner reports as a failure;
//! status 1 would be accepted under `expect=unfinalized`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_free_exposed_window
// mpi-test: np=2 expect=unfinalized valgrind skip-ok=openmpi-4
// mpi-test-stderr: MPI_Finalize skipped: 3 window(s) still alive

use ferrompi::{Communicator, Error, Group, Mpi, MpiErrorClass, Win, WinPscwAssert};

mod common;

const SENT: u64 = 0x5555_5555_5555_5555;
const VICTIM: u64 = 0xAAAA_AAAA_AAAA_AAAA;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn peer_group(world: &Communicator, peer: i32) -> Group {
    world
        .group()
        .and_then(|g| g.include(&[peer]))
        .expect("peer group failed")
}

/// Rank 1 puts `SENT` into rank 0's window in one PSCW epoch.
fn put_to_rank0(world: &Communicator, win: &mut Win<u64>) {
    let data = [SENT; 4];
    win.start(&peer_group(world, 0), WinPscwAssert::default(), |e| {
        e.put(&data, 0, 0)
    })
    .expect("start/put/complete");
}

/// Parts 1, 2 and 4: an epoch closed by `wait_exposure` (`poll` false) or by a
/// `test_exposure` that returned `true` (`poll` true) lets `into_buffer` free
/// the window: `Ok(None)` for a `Win::allocate` window, `Ok(Some(v))` with the
/// peer's data for a `Win::create` window (`owned`).
fn closed_epoch_frees(world: &Communicator, poll: bool, owned: bool) {
    let rank = world.rank();
    let how = format!(
        "{} on a {} window",
        if poll {
            "test_exposure"
        } else {
            "wait_exposure"
        },
        if owned {
            "Win::create"
        } else {
            "Win::allocate"
        }
    );
    let mut win = if owned {
        Win::<u64>::create(world, vec![0; 4]).expect("Win::create")
    } else {
        Win::<u64>::allocate(world, 4).expect("Win::allocate")
    };
    if rank == 0 {
        win.post(&peer_group(world, 1), WinPscwAssert::default())
            .expect("post");
        if poll {
            while !win.test_exposure().expect("test_exposure") {}
        } else {
            win.wait_exposure().expect("wait_exposure");
        }
    } else {
        put_to_rank0(world, &mut win);
    }
    let landed = rank != 0 || win.local_slice() == [SENT; 4];
    let freed = match win.into_buffer() {
        Ok(Some(v)) => owned && (rank != 0 || v == [SENT; 4]),
        Ok(None) => !owned,
        Err(_) => false,
    };
    common::check(
        world,
        landed,
        &format!("{how}: the peer's put is in the window"),
    );
    common::check(
        world,
        freed,
        &format!("{how} closes the exposure epoch, so into_buffer frees the window"),
    );
}

/// Part 3: the result of `into_buffer` with the exposure epoch open and its
/// matching access epoch closed, on a window whose communicator holds this rank
/// alone. Only the `wait` is missing.
fn into_buffer_on_own_communicator(world: &Communicator) -> String {
    let own = world
        .split(Some(world.rank()), 0)
        .expect("split")
        .expect("a rank is a member of its own color");
    let mut win = Win::<u64>::allocate(&own, 4).expect("Win::allocate (own communicator)");
    let to_self = own
        .group()
        .and_then(|g| g.include(&[0]))
        .expect("group of one");
    win.post(&to_self, WinPscwAssert::default())
        .expect("post (own communicator)");
    win.start(&to_self, WinPscwAssert::default(), |_| Ok(()))
        .expect("start (own communicator)");
    format!("{:?}", win.into_buffer())
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    assert_eq!(
        world.size(),
        2,
        "test_rma_free_exposed_window requires exactly 2 processes"
    );

    closed_epoch_frees(&world, false, false);
    closed_epoch_frees(&world, true, false);

    let got = into_buffer_on_own_communicator(&world);
    common::check(
        &world,
        got.starts_with("Err(InvalidState"),
        &format!("into_buffer with an open exposure epoch returns InvalidState, got {got}"),
    );

    // Part 4. `Win::create` is refused by Open MPI 4 over caller memory.
    let mut owned = match Win::create(&world, vec![0u64; 32]) {
        Ok(win) => win,
        Err(Error::Mpi {
            class: MpiErrorClass::Win,
            ..
        }) => {
            common::skip(
                &world,
                "Win::create returned MPI_ERR_WIN — likely OpenMPI 4.x \
                 with a BTL that does not support one-sided over caller-owned \
                 memory (e.g., --btl=self,tcp in CI). Parts 1 to 3 passed.",
            );
            return;
        }
        Err(e) => panic!("Win::create failed: {e}"),
    };
    closed_epoch_frees(&world, false, true);
    let mut allocated = Win::<u64>::allocate(&world, 4).expect("Win::allocate");

    if rank == 0 {
        let to_rank1 = peer_group(&world, 1);
        owned
            .post(&to_rank1, WinPscwAssert::default())
            .expect("post (create)");
        allocated
            .post(&to_rank1, WinPscwAssert::default())
            .expect("post (allocate)");
    }
    world.barrier().expect("barrier 1");

    let mut refused = true;
    let mut victim = Vec::new();
    let mut kept = None;
    if rank == 0 {
        refused = matches!(owned.into_buffer(), Err(Error::InvalidState { .. }));
        drop(allocated);
        // The leaked buffer is not reused, so this allocation cannot overlap it.
        victim = vec![VICTIM; 32];
        world.barrier().expect("barrier 2");
        world.barrier().expect("barrier 3");
    } else {
        world.barrier().expect("barrier 2");
        put_to_rank0(&world, &mut owned);
        put_to_rank0(&world, &mut allocated);
        world.barrier().expect("barrier 3");
        kept = Some((owned, allocated));
    }

    common::check(
        &world,
        refused,
        "into_buffer with an open exposure epoch returns InvalidState (create window)",
    );
    common::check(
        &world,
        victim.iter().all(|&x| x == VICTIM),
        "no allocation reuses the leaked buffer",
    );

    // Rank 0's windows are leaked, and rank 1's two of part 4 are still alive.
    drop(mpi);
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    drop(kept);
    if rank == 0 {
        println!("test_rma_free_exposed_window: PASS");
    }
}
