//! Integration test for `WinView` — the view of a window's own memory.
//!
//! Verifies that:
//!
//! 1. `Win::local` reports the window's length, and `write`/`read` and
//!    `copy_from_slice`/`copy_to_slice` round-trip.
//! 2. A put by rank 0 into rank 1's window in a fence epoch is read through
//!    `Win::local` once the closing fence returns.
//! 3. `ActiveEpoch::local` reads the same values inside a second fence epoch.
//! 4. A put under `lock(Exclusive, 1)` is read through `PassiveEpoch::local`
//!    inside rank 1's own `lock(Shared, 1)` epoch, after a barrier.
//! 5. An `f64` and a `u8` window round-trip, which covers the 8-byte and
//!    1-byte atomic accesses.
//! 6. A put under `lock(Exclusive, 1)` is read through `as_slice` inside rank
//!    1's `lock_all` epoch, published by a barrier and `sync`, and rank 1 then
//!    writes through `as_mut_slice` and reads the value back through the view.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_window_view

use ferrompi::{LockType, Mpi, Win, WinFenceAssert};

mod common;

const N: usize = 4;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_window_view requires at least 2 processes, got {size}"
    );

    let none = WinFenceAssert::none();

    // ========================================================================
    // Part 1: the view's length and its two round-trips.
    // ========================================================================
    {
        let win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let local = win.local();
        common::check(
            &world,
            local.len() == N && !local.is_empty(),
            "Win::local reports the window's length",
        );

        local.write(2, 7);
        common::check(&world, local.read(2) == 7, "write then read round-trips");

        let sent = [rank as u64 + 1, 20, 30, 40];
        local.copy_from_slice(&sent);
        let mut got = [0u64; N];
        local.copy_to_slice(&mut got);
        common::check(
            &world,
            got == sent,
            "copy_from_slice then copy_to_slice round-trips",
        );
    }
    if rank == 0 {
        println!("PASS: Win::local length and round-trips");
    }

    // ========================================================================
    // Parts 2 and 3: rank 0 puts [rank + 1; N] into rank 1's window in a fence
    // epoch. After the closing fence rank 1 reads it through `Win::local`, and
    // inside a second fence epoch through `ActiveEpoch::local`.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let origin = [rank as u64 + 1; N];
        win.fence(none, none, |e| {
            if rank == 0 {
                e.put(&origin, 1, 0)
            } else {
                Ok(())
            }
        })
        .expect("fence epoch with put failed");

        let mut got = [0u64; N];
        win.local().copy_to_slice(&mut got);
        common::check(
            &world,
            rank != 1 || got == [1; N],
            "Win::local reads the put after the closing fence",
        );
        if rank == 0 {
            println!("PASS: Win::local after a fence epoch");
        }

        let mut seen = [0u64; N];
        win.fence(none, none, |e| {
            e.local().copy_to_slice(&mut seen);
            Ok(())
        })
        .expect("second fence epoch failed");
        common::check(
            &world,
            rank != 1 || seen == [1; N],
            "ActiveEpoch::local reads the same values inside a fence epoch",
        );
        if rank == 0 {
            println!("PASS: ActiveEpoch::local");
        }
    }

    // ========================================================================
    // Part 4: rank 0 puts into rank 1 under an exclusive lock; the unlock
    // completes the put, and the barrier orders it before rank 1's own shared
    // lock, whose acquisition makes the update visible to rank 1's loads.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let origin = [11u64, 22, 33, 44];
        if rank == 0 {
            win.lock(LockType::Exclusive, 1, |e| e.put(&origin, 1, 0))
                .expect("lock(Exclusive, 1) with put failed");
        }
        world.barrier().expect("barrier after the unlock failed");

        let mut got = [0u64; N];
        if rank == 1 {
            win.lock(LockType::Shared, 1, |e| {
                e.local().copy_to_slice(&mut got);
                Ok(())
            })
            .expect("lock(Shared, 1) failed");
        }
        common::check(
            &world,
            rank != 1 || got == origin,
            "PassiveEpoch::local reads a put completed by the peer's unlock",
        );
        if rank == 0 {
            println!("PASS: PassiveEpoch::local");
        }
    }

    // ========================================================================
    // Part 5: an f64 window (8-byte atomic) and a u8 window (1-byte atomic).
    // ========================================================================
    {
        let floats = Win::<f64>::allocate(&world, N).expect("Win::allocate failed");
        let want = [1.5, -2.25, 1e300, 0.1];
        floats.local().copy_from_slice(&want);
        let mut got = [0.0f64; N];
        floats.local().copy_to_slice(&mut got);
        common::check(&world, got == want, "f64 window round-trips");

        let bytes = Win::<u8>::allocate(&world, N).expect("Win::allocate failed");
        let want = [1u8, 0x80, 0xff, 0];
        bytes.local().copy_from_slice(&want);
        let mut got = [0u8; N];
        bytes.local().copy_to_slice(&mut got);
        common::check(&world, got == want, "u8 window round-trips");
    }
    if rank == 0 {
        println!("PASS: f64 and u8 windows round-trip");
    }

    // ========================================================================
    // Part 6: the zero-copy slices. Rank 0 puts [7; N] into rank 1's window
    // under an exclusive lock, and rank 1 reads it through `as_slice` inside
    // its own `lock_all` epoch, then writes through `as_mut_slice`.
    // ========================================================================
    {
        let mut win = Win::<u64>::allocate(&world, N).expect("Win::allocate failed");
        let origin = [7u64; N];
        if rank == 0 {
            win.lock(LockType::Exclusive, 1, |e| e.put(&origin, 1, 0))
                .expect("lock(Exclusive, 1) with put failed");
        }
        world.barrier().expect("barrier after the unlock failed");

        let mut slice_read = false;
        let mut slice_written = false;
        if rank == 1 {
            win.lock_all(|e| {
                e.sync()?;
                let mut view = e.local();
                // SAFETY: rank 0's unlock completed the put at rank 1, the
                // barrier is the process synchronization, and `e.sync()` is the
                // `MPI_Win_sync` inside this passive-target epoch, so the put
                // was published before the slice exists. No other rank writes
                // before the barrier after this epoch, and rank 1's only write
                // comes after the slice's last use.
                let s = unsafe { view.as_slice() };
                slice_read = s == [7; N];
                // SAFETY: the put was published as above, `s` is dead, and this
                // is the only view or slice of the region in use. No other rank
                // accesses the region before the barrier after this epoch, and
                // this epoch makes no RMA call.
                let m = unsafe { view.as_mut_slice() };
                m.copy_from_slice(&[8; N]);
                slice_written = e.local().read(0) == 8;
                Ok(())
            })
            .expect("lock_all with the zero-copy slices failed");
        }
        world.barrier().expect("barrier after the epoch failed");

        common::check(
            &world,
            rank != 1 || slice_read,
            "as_slice reads a put published by the barrier and sync",
        );
        common::check(
            &world,
            rank != 1 || slice_written,
            "a write through as_mut_slice reads back through the view",
        );
        if rank == 0 {
            println!("PASS: as_slice and as_mut_slice");
        }
    }
}
// mpi-test: np=2..
