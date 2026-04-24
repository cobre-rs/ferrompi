//! Rank 1 posts an irecv for a message that never arrives, then
//! probes with get_status, cancels, and waits.
//! Rank 0 does nothing and exits.

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();
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
    Ok(())
}
