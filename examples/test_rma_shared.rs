//! Integration test for `Win::allocate_shared` and the `remote` views.
//!
//! Verifies that:
//!
//! 1. The canonical shared-memory pattern, inside one `lock_all` closure: each
//!    rank stores through `PassiveEpoch::local`, `sync`, a barrier, `sync`, and
//!    then every rank's region reads back through `PassiveEpoch::remote`.
//! 2. `Win::remote(own rank)` and `Win::local` alias: a write through one reads
//!    back through the other.
//! 3. `ActiveEpoch::remote` reads every rank's region inside a fence epoch,
//!    after the stores of an earlier fence epoch.
//! 4. An RMA `put` by rank 0 into rank 1's region of the shared window, under
//!    `lock(Exclusive, 1)`, is read by rank 1 through `PassiveEpoch::local` in
//!    its own `lock(Shared, 1)`, after a barrier.
//! 5. `remote` refuses a rank outside the communicator with
//!    `InvalidArgument { arg: "rank" }`, and a window that is not shared with
//!    `NotSupported`.
//! 6. A window where rank 0 has count 0: every `remote(0)` is empty, and
//!    `remote(1)` addresses rank 1's own region.
//! 7. `PassiveEpoch::remote` of a rank outside the epoch's access set is not
//!    refused: a lock on one rank does not restrict a load/store view.
//! 8. Ranks that pass element types of different sizes to `allocate_shared` are
//!    all refused with `InvalidArgument { arg: "T" }`, before any window exists.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_rma_shared

use ferrompi::{Error, LockType, Mpi, Win, WinFenceAssert};

mod common;

const N: usize = 4;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    let node = world.split_shared().expect("split_shared failed");
    let node_rank = node.rank();
    let node_size = node.size();
    assert!(
        node_size >= 2,
        "test_rma_shared requires at least 2 processes on one node, got {node_size}"
    );
    let ranks = node_size as usize;

    // ========================================================================
    // Part 1: the canonical pattern in one lock_all closure. Each rank stores
    // rank + 1 into its four slots through its own view, `sync` completes the
    // stores, the barrier is the process synchronization, and the second `sync`
    // orders this rank's loads after every rank's stores. Nothing writes a
    // region again before the window is freed, which is collective.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        let mut seen = vec![[0u64; N]; ranks];
        let mut lens = vec![0usize; ranks];
        win.lock_all(|e| {
            e.local().copy_from_slice(&[node_rank as u64 + 1; N]);
            e.sync()?;
            node.barrier()?;
            e.sync()?;
            for r in 0..node_size {
                let view = e.remote(r)?;
                lens[r as usize] = view.len();
                view.copy_to_slice(&mut seen[r as usize]);
            }
            Ok(())
        })
        .expect("lock_all with the canonical pattern failed");

        common::check(
            &world,
            lens.iter().all(|&len| len == N),
            "every remote view has the creating rank's length",
        );
        common::check(
            &world,
            (0..ranks).all(|r| seen[r] == [r as u64 + 1; N]),
            "remote reads every rank's stores after sync, barrier and sync",
        );
    }
    if rank == 0 {
        println!("PASS: canonical pattern in one lock_all closure");
    }

    // ========================================================================
    // Part 2: remote(own rank) and local() address the same memory. Each rank
    // touches only its own region.
    // ========================================================================
    {
        let win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        let own = win.remote(node_rank).expect("remote(own rank) failed");
        let local = win.local();
        common::check(
            &world,
            own.len() == local.len(),
            "remote(own rank) and local() have one length",
        );
        own.write(1, 42);
        common::check(
            &world,
            local.read(1) == 42,
            "a write through remote(own rank) reads back through local()",
        );
        local.write(2, 43);
        common::check(
            &world,
            own.read(2) == 43,
            "a write through local() reads back through remote(own rank)",
        );
    }
    if rank == 0 {
        println!("PASS: remote(own rank) aliases local()");
    }

    // ========================================================================
    // Part 3: a fence epoch. The first epoch stores through ActiveEpoch::local,
    // and its closing fence is collective. The second epoch reads every rank's
    // region through ActiveEpoch::remote, and nothing writes in it.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        let none = WinFenceAssert::none();
        win.fence(none, none, |e| {
            e.local().copy_from_slice(&[node_rank as u64 + 1; N]);
            Ok(())
        })
        .expect("fence epoch with stores failed");

        let mut seen = vec![[0u64; N]; ranks];
        win.fence(none, none, |e| {
            for r in 0..node_size {
                e.remote(r)?.copy_to_slice(&mut seen[r as usize]);
            }
            Ok(())
        })
        .expect("fence epoch with remote reads failed");
        common::check(
            &world,
            (0..ranks).all(|r| seen[r] == [r as u64 + 1; N]),
            "ActiveEpoch::remote reads every rank's stores inside a fence epoch",
        );
    }
    if rank == 0 {
        println!("PASS: ActiveEpoch::remote");
    }

    // ========================================================================
    // Part 4: rank 0 puts into rank 1's region under an exclusive lock. The
    // unlock completes the put at the target, the barrier orders it before
    // rank 1's own shared lock, and that lock makes the update visible to rank
    // 1's loads.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        let origin = [11u64, 22, 33, 44];
        if node_rank == 0 {
            win.lock(LockType::Exclusive, 1, |e| e.put(&origin, 1, 0))
                .expect("lock(Exclusive, 1) with put failed");
        }
        node.barrier().expect("barrier after the unlock failed");

        let mut got = [0u64; N];
        if node_rank == 1 {
            win.lock(LockType::Shared, 1, |e| {
                e.local().copy_to_slice(&mut got);
                Ok(())
            })
            .expect("lock(Shared, 1) failed");
        }
        common::check(
            &world,
            node_rank != 1 || got == origin,
            "a put into a peer's region of a shared window reads back at the peer",
        );
    }
    if rank == 0 {
        println!("PASS: put into a shared window");
    }

    // ========================================================================
    // Part 5: the two refusals.
    // ========================================================================
    {
        let win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        common::check(
            &world,
            matches!(
                win.remote(node_size),
                Err(Error::InvalidArgument { arg: "rank", .. })
            ),
            "remote(node.size()) is InvalidArgument { arg: rank }",
        );
        common::check(
            &world,
            matches!(
                win.remote(-1),
                Err(Error::InvalidArgument { arg: "rank", .. })
            ),
            "remote(-1) is InvalidArgument { arg: rank }",
        );

        let plain = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        common::check(
            &world,
            matches!(plain.remote(0), Err(Error::NotSupported(_))),
            "remote on a Win::allocate window is NotSupported",
        );
    }
    if rank == 0 {
        println!("PASS: remote refusals");
    }

    // ========================================================================
    // Part 6: rank 0 has count 0. Its region is empty whatever pointer MPI
    // reports for it, and rank 1's region, which starts where the allocation
    // does, is read through remote(1) after the canonical sequence.
    // ========================================================================
    {
        let count = if node_rank == 0 { 0 } else { N };
        let mut win = Win::<u64>::allocate_shared(&node, count).expect("allocate_shared failed");
        let first = win
            .lock_all(|e| {
                if node_rank == 1 {
                    e.local().write(0, 11);
                }
                e.sync()?;
                node.barrier()?;
                e.sync()?;
                Ok(e.remote(1)?.read(0))
            })
            .expect("lock_all with an empty region failed");

        common::check(
            &world,
            win.remote(0).expect("remote(0) failed").is_empty(),
            "remote(0) is empty when rank 0 has count 0",
        );
        common::check(
            &world,
            win.remote(1).expect("remote(1) failed").len() == N,
            "remote(1) has length 4 when rank 0 has count 0",
        );
        common::check(
            &world,
            first == 11,
            "remote(1) reads rank 1's store when rank 0 has count 0",
        );
    }
    if rank == 0 {
        println!("PASS: empty region of rank 0");
    }

    // ========================================================================
    // Part 7: remote is a load/store path, not an RMA call, so an epoch's
    // access set does not restrict it.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate_shared(&node, N).expect("allocate_shared failed");
        let other = (node_rank + 1) % node_size;
        let len = win
            .lock(LockType::Shared, node_rank, |e| Ok(e.remote(other)?.len()))
            .expect("lock(Shared, own rank) with remote(other) failed");
        common::check(
            &world,
            len == N,
            "remote of a rank outside the epoch's access set is not refused",
        );
    }
    if rank == 0 {
        println!("PASS: remote is not restricted by the access set");
    }

    // ========================================================================
    // Part 8: the regions of a shared window follow one another in memory, so a
    // base is aligned for `T` only when every rank's `T` has one size. Rank 0
    // passes a 4-byte type and the others an 8-byte one: every rank must refuse,
    // before any window exists.
    // ========================================================================
    {
        let mixed = if node_rank == 0 {
            Win::<u32>::allocate_shared(&node, N).map(drop)
        } else {
            Win::<u64>::allocate_shared(&node, N).map(drop)
        };
        common::check(
            &world,
            matches!(mixed, Err(Error::InvalidArgument { arg: "T", .. })),
            "allocate_shared with element types of different sizes is InvalidArgument { arg: T }",
        );
    }
    if rank == 0 {
        println!("PASS: mixed element sizes are refused");
    }
}
// mpi-test: np=2.. release
