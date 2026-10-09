//! Integration test for `Win::rput` — request-returning one-sided write.
//!
//! Verifies that:
//!
//! 1. Rank 0 locks rank 1 exclusively, calls `Win::rput` inside the closure to
//!    post a write of `[10, 20, 30, 40]` into rank 1's window, waits on the
//!    returned `Request` (local completion), then the closure ends and the
//!    unlock gives remote completion. After a barrier, rank 1 inspects its
//!    local window memory and asserts the correct values.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_rput
// mpi-test: np=2

use ferrompi::{LockType, Mpi, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_rput requires exactly 2 processes, got {size}"
    );

    let mut local_ok = true;

    // ========================================================================
    // Test 1: rank 0 rput [10, 20, 30, 40] into rank 1's window.
    //
    // Protocol:
    //   Rank 0: lock(Exclusive, rank=1, |e| rput → req.wait()) → unlock
    //   Rank 1: barrier → read local window → assert [10, 20, 30, 40]
    // ========================================================================
    {
        const N: usize = 4;
        let mut win = Win::<i32>::allocate(&world, N).expect("Win::allocate failed");

        if rank == 0 {
            let buf = [10i32, 20, 30, 40];

            // Exclusive passive-target lock on rank 1's window; the unlock when
            // the closure ends triggers remote completion — rank 1 will see the
            // write after this point.
            let locked = win.lock(LockType::Exclusive, 1, |e| {
                // Post the rput — request completes when local buffer is safe to reuse.
                let req = match e.rput(&buf, 1, 0, buf.len() as i64) {
                    Ok(r) => r,
                    Err(err) => {
                        eprintln!("rank 0: FAIL: Win::rput returned error: {err}");
                        world.abort(1);
                    }
                };

                // Wait for local buffer completion.
                if let Err(err) = req.wait() {
                    eprintln!("rank 0: FAIL: Request::wait after rput failed: {err}");
                    local_ok = false;
                }
                Ok(())
            });
            if let Err(e) = locked {
                eprintln!("rank 0: FAIL: Win::lock(Exclusive, 1) failed: {e}");
                world.abort(1);
            }
        }

        // All ranks synchronize here: rank 0's unlock has completed, so rank 1
        // can now safely read its local window memory.
        world.barrier().expect("barrier after rput failed");

        if rank == 1 {
            let expected = [10i32, 20, 30, 40];
            let local = win.local_slice();
            if local != expected {
                eprintln!(
                    "FAIL: rank 1 local window after rput: expected {expected:?}, got {local:?}"
                );
                local_ok = false;
            }
        }
    }

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && local_ok {
        println!("PASS: Win::rput with local completion");
    }

    common::check(&world, local_ok, "test_rma_rput");
}
