use ferrompi::{Mpi, Request};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    if world.rank() == 0 {
        let mut a = vec![0u8; 4];
        let mut b = vec![0u8; 64];
        let mut reqs = vec![world.irecv(&mut a, 1, 1)?, world.irecv(&mut b, 1, 2)?];
        world.barrier()?;
        let r = Request::wait_all(&mut reqs);
        eprintln!("rank0: wait_all -> {:?}", r.as_ref().map(|_| ()));
        eprintln!("rank0: dropping stale requests (no new requests posted)...");
        drop(reqs);
        eprintln!("rank0: dropped OK");
    } else {
        world.barrier()?;
        world.send(&[1u8; 16], 0, 1)?;
        world.send(&[2u8; 16], 0, 2)?;
    }
    world.barrier()?;
    Ok(())
}
