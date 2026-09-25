use ferrompi::Mpi;
fn main() {
    let world;
    { let mpi = Mpi::init().unwrap(); world = mpi.world(); }
    eprintln!("finalized; calling world.barrier()");
    let r = world.barrier();
    eprintln!("barrier is_err={}", r.is_err());
    eprintln!("SURVIVED");
}
