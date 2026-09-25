// R10: wait_any passes handles of already-completed requests; a reused slot aliases another Request.
use ferrompi::{Mpi, Request};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    if world.rank() == 0 {
        let mut a = vec![0u8; 8];
        let mut b = vec![0u8; 8];
        let mut reqs = vec![world.irecv(&mut a, 1, 1)?, world.irecv(&mut b, 1, 2)?];
        println!("rank0: handles = {} {}", reqs[0].raw_handle(), reqs[1].raw_handle());
        world.barrier()?; // rank1 sends tag 1 only
        let i = Request::wait_any(&mut reqs)?;
        println!("rank0: first wait_any -> {:?}", i);
        let mut c = vec![0u8; 8];
        let other = world.irecv(&mut c, 1, 3)?; // likely reuses the freed slot
        println!("rank0: unrelated request handle = {}", other.raw_handle());
        world.barrier()?; // rank1 sends tag 3 now (tag 2 later)
        let j = Request::wait_any(&mut reqs)?; // stale handle now names `other`
        println!("rank0: second wait_any -> {:?}; other.is_completed()={}", j, other.is_completed());
        println!("rank0: other.wait() -> {:?}", other.wait());
        world.barrier()?;
        drop(reqs);
    } else {
        world.barrier()?;
        world.send(&[1u8; 8], 0, 1)?;
        world.barrier()?;
        world.send(&[3u8; 8], 0, 3)?;
        world.barrier()?;
        world.send(&[2u8; 8], 0, 2)?;
    }
    Ok(())
}
