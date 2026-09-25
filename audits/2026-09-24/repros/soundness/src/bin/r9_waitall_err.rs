// R9: Request::wait_all error path leaves stale MPI_Request values in the C table, then Drop re-waits them.
use ferrompi::{Mpi, Request};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    if world.rank() == 0 {
        let mut a = vec![0u8; 4];  // too small -> truncation error
        let mut b = vec![0u8; 64];
        let mut reqs = vec![world.irecv(&mut a, 1, 1)?, world.irecv(&mut b, 1, 2)?];
        world.barrier()?;
        let r = Request::wait_all(&mut reqs);
        println!("rank0: wait_all -> {:?}", r.as_ref().map(|_| ()));
        // Post new requests so MPI's freed request objects get recycled.
        let mut c = vec![0u8; 64];
        let r3 = world.irecv(&mut c, 1, 3)?;
        println!("rank0: dropping stale requests (Drop re-waits them)...");
        drop(reqs);
        println!("rank0: dropped stale requests; waiting r3");
        world.barrier()?;
        r3.wait()?;
        println!("rank0: done");
    } else {
        world.barrier()?;
        world.send(&[1u8; 16], 0, 1)?;
        world.send(&[2u8; 16], 0, 2)?;
        world.barrier()?;
        world.send(&[3u8; 16], 0, 3)?;
    }
    Ok(())
}
