//! Integration test for request-table concurrency under MPI_THREAD_MULTIPLE.
//!
//! Spawns 4 std::thread workers per MPI rank; each thread runs 100 iterations
//! of isend + irecv + wait, exercising concurrent alloc_request / free_request
//! calls against the C11-atomic request table implemented in csrc/ferrompi.c.
//!
//! With 2 MPI ranks and 4 threads each, this generates 400 concurrent
//! alloc_request calls per rank, which is sufficient to race without
//! the C11 CAS fix and to confirm no slot is double-claimed with it.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_request_table_concurrency
//!
//! TSan manual verification (pre-release step, not CI-gated because libmpi
//! internals trigger false positives):
//!   RUSTFLAGS="-Zsanitizer=thread" CFLAGS="-fsanitize=thread" \
//!   cargo +nightly build --target x86_64-unknown-linux-gnu --examples && \
//!   mpiexec -n 2 \
//!     ./target/x86_64-unknown-linux-gnu/debug/examples/test_request_table_concurrency
//! Expected result: no TSan diagnostics from ferrompi C code; any reports
//! from libmpi internals (allreduce, progress engine) are known benign.

use ferrompi::{Mpi, ThreadLevel};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const NUM_THREADS: usize = 4;
const ITERATIONS: usize = 100;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Multiple).expect("MPI init failed");

    // If the MPI library cannot provide MPI_THREAD_MULTIPLE, skip gracefully.
    // Some builds (e.g. certain Cray MPT configurations) deliberately refuse it.
    if mpi.thread_level() < ThreadLevel::Multiple {
        println!(
            "SKIP: MPI provided {:?}, MPI_THREAD_MULTIPLE required; skipping test",
            mpi.thread_level()
        );
        return;
    }

    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_request_table_concurrency requires at least 2 MPI ranks, got {size}"
    );

    // partner: rank 0 communicates with rank 1; all other pairs mirror this pattern.
    // With mpiexec -n 2 there are exactly ranks 0 and 1.
    let partner = if rank % 2 == 0 {
        (rank + 1) % size
    } else {
        rank - 1
    };

    // Shared error flag: set to true by any thread that encounters a failure.
    let any_error = Arc::new(AtomicBool::new(false));

    // std::thread::scope borrows `world` and `any_error` for all workers.
    // Using references rather than move captures keeps `world` alive after
    // the scope so that the barrier and allreduce below can still use it.
    std::thread::scope(|s| {
        for thread_id in 0..NUM_THREADS {
            // Clone the Arc before the closure; the closure borrows everything
            // else by reference via the scoped-thread lifetime guarantee.
            let any_error_ref = Arc::clone(&any_error);
            let world_ref = &world;

            s.spawn(move || {
                // Each (rank, thread_id) pair uses a unique tag to avoid
                // cross-thread message matching: tags are in [1000, 1000+NUM_THREADS).
                // Rank 0's thread t sends tag (1000 + t) and receives the same tag
                // from rank 1, which mirrors the same tag in its own loop.
                let tag = 1000 + thread_id as i32;

                for _iter in 0..ITERATIONS {
                    let send_buf: [u8; 4] = [rank as u8; 4];
                    let mut recv_buf: [u8; 4] = [0u8; 4];

                    // Post irecv before isend to avoid potential deadlock.
                    let recv_req = match world_ref.irecv(&mut recv_buf, partner, tag) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("rank {rank} thread {thread_id}: irecv failed: {e}");
                            any_error_ref.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    let send_req = match world_ref.isend(&send_buf, partner, tag) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("rank {rank} thread {thread_id}: isend failed: {e}");
                            any_error_ref.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    if let Err(e) = send_req.wait() {
                        eprintln!("rank {rank} thread {thread_id}: isend wait failed: {e}");
                        any_error_ref.store(true, Ordering::Relaxed);
                        return;
                    }

                    if let Err(e) = recv_req.wait() {
                        eprintln!("rank {rank} thread {thread_id}: irecv wait failed: {e}");
                        any_error_ref.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            });
        }
        // All threads join here (scope exit).
    });

    // Aggregate error state across ranks BEFORE any process exits.  If any
    // rank exited early, surviving ranks would deadlock at this allreduce —
    // so the exit-on-failure path runs after the collective, never before.
    let local_ok: i32 = if any_error.load(Ordering::Acquire) {
        0
    } else {
        1
    };
    let global_ok = world
        .allreduce_scalar(local_ok, ferrompi::ReduceOp::Min)
        .expect("allreduce_scalar failed");

    if global_ok == 0 {
        if rank == 0 {
            eprintln!("FAIL: at least one rank reported a thread error");
        }
        std::process::exit(1);
    }

    if rank == 0 {
        println!("OK: request table concurrency test passed");
    }
}
