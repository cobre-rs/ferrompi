// R3: safe code — counts/displs slices shorter than comm size are read out of bounds by MPI.
use ferrompi::Mpi;
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let send = [1u32; 1];
    let mut recv = vec![0u32; 4];
    // Empty (dangling) counts/displs pass the len()==len() guard; MPI reads size() ints.
    let r = world.allgatherv(&send, &mut recv, &[], &[]);
    println!("rank {}: allgatherv with empty counts returned {:?}", world.rank(), r.is_ok());
    Ok(())
}
