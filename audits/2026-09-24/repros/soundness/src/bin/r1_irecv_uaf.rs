// R1: safe code — nonblocking receive buffer dropped (and reused) while MPI still owns the pointer.
use ferrompi::Mpi;
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();
    if rank == 0 {
        let req = {
            let mut buf = vec![0u8; 256];
            world.irecv(&mut buf, 1, 7)?
        }; // buf freed here while the receive is still pending
        // A new, unrelated allocation of the same size class reuses the freed chunk.
        let victim = vec![0xAAu8; 256];
        world.barrier()?;            // rank 1 sends after this
        req.wait()?;
        let corrupted = victim.iter().filter(|&&b| b != 0xAA).count();
        println!("rank0: victim bytes overwritten by MPI = {corrupted} / 256 (first={:#x})", victim[0]);
    } else {
        world.barrier()?;
        world.send(&vec![0x55u8; 256], 0, 7)?;
    }
    Ok(())
}
