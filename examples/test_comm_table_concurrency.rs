//! Integration test for comm-table concurrency under MPI_THREAD_MULTIPLE.
//!
//! Spawns 4 std::thread workers on a single MPI rank; each thread calls
//! `world.duplicate()` 50 times and collects all 50 live Communicator objects
//! before appending their handles to a shared list.  All 200 communicators are
//! kept alive until every thread has finished allocating, so the handles are
//! all simultaneously resident in the table when we check for duplicates.
//! Any duplicate handle means two threads claimed the same `comm_table` slot
//! via a data race in `alloc_comm`.
//!
//! With 4 threads × 50 duplications = 200 concurrent alloc_comm calls, the
//! test races reliably without the C11 CAS fix and confirms no slot is
//! double-claimed with it.  After the assertion, the communicators are dropped,
//! which exercises concurrent ferrompi_comm_free as well.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_comm_table_concurrency
//!
//! TSan manual verification (pre-release step, not CI-gated because libmpi
//! internals trigger false positives):
//!   RUSTFLAGS="-Zsanitizer=thread" CFLAGS="-fsanitize=thread" \
//!   cargo +nightly build --target x86_64-unknown-linux-gnu --examples && \
//!   mpiexec -n 1 \
//!     ./target/x86_64-unknown-linux-gnu/debug/examples/test_comm_table_concurrency
//! Expected result: no TSan diagnostics from ferrompi C code; any reports
//! from libmpi internals are known benign.
// mpi-test: np=1

use ferrompi::{Communicator, Mpi, ThreadLevel};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

mod common;

const NUM_THREADS: usize = 4;
const DUPS_PER_THREAD: usize = 50;
const TOTAL_HANDLES: usize = NUM_THREADS * DUPS_PER_THREAD;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");
    let world = mpi.world();

    // If the MPI library cannot provide MPI_THREAD_MULTIPLE, skip gracefully.
    // Some builds (e.g. certain Cray MPT configurations) deliberately refuse it.
    if mpi.thread_level() < ThreadLevel::Multiple {
        common::skip(
            &world,
            &format!(
                "MPI provided {:?}, MPI_THREAD_MULTIPLE required; skipping test",
                mpi.thread_level()
            ),
        );
        return;
    }

    // Shared collection: each thread appends its Communicators (still live) so
    // all 200 slots are simultaneously occupied when we check for duplicates.
    let comms: Arc<Mutex<Vec<Communicator>>> =
        Arc::new(Mutex::new(Vec::with_capacity(TOTAL_HANDLES)));

    // std::thread::scope borrows `world` and `comms` for all workers.
    std::thread::scope(|s| {
        for _thread_id in 0..NUM_THREADS {
            let comms_ref = Arc::clone(&comms);
            let world_ref = &world;

            s.spawn(move || {
                // Allocate all 50 communicators before touching the shared vec
                // so that threads are racing in alloc_comm for as long as possible.
                let mut local: Vec<Communicator> = Vec::with_capacity(DUPS_PER_THREAD);
                for _ in 0..DUPS_PER_THREAD {
                    let dup = world_ref
                        .duplicate()
                        .expect("world.duplicate() must not fail");
                    local.push(dup);
                }
                // Transfer all live communicators to the shared collection.
                // The Mutex acquisition happens only once per thread, after all
                // allocations are done, keeping the hot alloc_comm loop lock-free.
                comms_ref.lock().expect("mutex poisoned").extend(local);
            });
        }
        // All threads join here (scope exit).  All 200 Communicators are now
        // alive inside `comms`.
    });

    // Check that all 200 simultaneously-live handles are distinct.
    let collected = comms.lock().expect("mutex poisoned");
    assert_eq!(
        collected.len(),
        TOTAL_HANDLES,
        "expected {TOTAL_HANDLES} communicators, got {}",
        collected.len()
    );

    let handles: Vec<i32> = collected.iter().map(|c| c.raw_handle()).collect();
    let unique: HashSet<i32> = handles.iter().copied().collect();
    assert_eq!(
        unique.len(),
        TOTAL_HANDLES,
        "duplicate comm handles detected: {} unique out of {} total — slot collision in alloc_comm",
        unique.len(),
        TOTAL_HANDLES
    );

    // Drop collected here (end of scope) — exercises concurrent ferrompi_comm_free.
    drop(collected);

    println!("OK: comm table concurrency test passed ({TOTAL_HANDLES} distinct handles)");
}
