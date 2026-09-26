//! Cancel semantics across request kinds.
//!
//! 1. Rank 1 posts an irecv for a message that never arrives, then probes
//!    with `get_status`, cancels, and waits: point-to-point cancel succeeds.
//!    Rank 0 does nothing.
//! 2. Every rank posts an `iallreduce`; `cancel()` on it must return
//!    `Err(NotSupported)`, and the still-pending request then completes
//!    normally via `wait()`.
//! 3. (with the `rma` feature) every rank `rput`s into its own window under
//!    a shared lock; `cancel()` on the returned request must return
//!    `Err(NotSupported)`, and it then completes normally via `wait()`.
// mpi-test: np=2

use ferrompi::{Error, Mpi, ReduceOp, Result};
#[cfg(feature = "rma")]
use ferrompi::{LockType, Win};

mod common;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    // Test 1: cancel of a point-to-point request.
    if rank == 1 {
        let mut buf = vec![0u8; 8];
        let mut req = world.irecv(&mut buf, 0, 99)?;
        // No message will arrive on tag 99, so get_status returns false.
        let complete = req.get_status()?;
        assert!(!complete, "get_status must report incomplete before cancel");
        req.cancel()?;
        req.wait()?;
        println!("rank 1: cancel+wait completed");
    }
    world.barrier()?;

    // Test 2: cancel of a nonblocking-collective request is refused.
    {
        let send = [1.0f64; 4];
        let mut recv = [0.0f64; 4];
        let mut req = world.iallreduce(&send, &mut recv, ReduceOp::Sum)?;
        let cancel_ok = matches!(req.cancel(), Err(Error::NotSupported(_)));
        common::check(
            &world,
            cancel_ok,
            "cancel on iallreduce returns NotSupported",
        );
        req.wait()?;
        let recv_ok = recv == [size as f64; 4];
        common::check(&world, recv_ok, "iallreduce completes after refused cancel");
    }

    // Test 3: cancel of an RMA request is refused.
    #[cfg(feature = "rma")]
    {
        let win = Win::<i32>::allocate(&world, 1)?;
        let guard = win.lock(LockType::Shared, rank)?;
        let mut req = win.rput(&[7], rank, 0, 1)?;
        let cancel_ok = matches!(req.cancel(), Err(Error::NotSupported(_)));
        common::check(&world, cancel_ok, "cancel on rput returns NotSupported");
        req.wait()?;
        drop(guard);
    }

    Ok(())
}
