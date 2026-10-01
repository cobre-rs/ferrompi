//! Regression for an oversize `Win::allocate` request.
//!
//! Proves that when one rank's requested window cannot be exposed, every
//! rank rejects the creation before any window exists — not only the rank
//! whose request is too large, and regardless of which rank it is.
//!
//! 1. The oversize request (`usize::MAX / 8 + 1` `u64` elements) comes from
//!    rank 0, then from the last rank; every other rank requests 4 in both
//!    cases (they coincide at np=1). Every rank checks that its
//!    `Win::allocate` call returns `Err(Error::InvalidBuffer)` after each case.
//! 2. Every rank then creates a `Win::<u64>::allocate` of 4 elements and
//!    checks that `comm_size()` equals `world.size()`, proving a normal
//!    window still works after the rejected ones.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_oversize
// mpi-test: np=2.. timeout=30

use ferrompi::{Error, Mpi, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    let oversize_count = usize::MAX / 8 + 1;
    for oversize_rank in [0, world.size() - 1] {
        let count = if rank == oversize_rank {
            oversize_count
        } else {
            4
        };

        let result = Win::<u64>::allocate(&world, count);
        let ok = matches!(result, Err(Error::InvalidBuffer));
        if !ok {
            match &result {
                Ok(_) => eprintln!("rank {rank}: FAIL: expected Err(InvalidBuffer), got Ok"),
                Err(e) => eprintln!("rank {rank}: FAIL: expected Err(InvalidBuffer), got Err({e})"),
            }
        }
        common::check(&world, ok, "test_rma_win_oversize");
    }

    let win = Win::<u64>::allocate(&world, 4).expect("Win::allocate(4) failed after rejection");
    let ok = win.comm_size() == world.size();
    if !ok {
        eprintln!(
            "rank {rank}: FAIL: comm_size() = {}, expected {}",
            win.comm_size(),
            world.size()
        );
    }
    common::check(&world, ok, "test_rma_win_oversize");

    if rank == 0 {
        println!("PASS: test_rma_win_oversize");
    }
}
