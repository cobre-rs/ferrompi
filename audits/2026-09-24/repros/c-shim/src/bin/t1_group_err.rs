use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let g = world.group().unwrap();
    eprintln!("rank {}: calling include(&[999])", world.rank());
    let r = g.include(&[999]);
    eprintln!("rank {}: include returned is_err={}", world.rank(), r.is_err());
    if let Err(e) = r { eprintln!("err: {e}"); }
    eprintln!("SURVIVED");
}
