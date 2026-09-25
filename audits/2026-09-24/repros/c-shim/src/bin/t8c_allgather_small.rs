use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let send = vec![1.0f64; 4];
    let mut recv = vec![0.0f64; 4]; // needs 4 * size
    let r = world.allgather(&send, &mut recv);
    eprintln!("rank {} allgather -> {:?}", world.rank(), r.map_err(|e| e.to_string()));
}
