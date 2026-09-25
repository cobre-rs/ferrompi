use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let send = vec![world.rank() as f64; 1];
    let mut recv = vec![0.0f64; world.size() as usize];
    // counts/displs of length 1 regardless of comm size (wrapper only checks equal lengths)
    let recvcounts = vec![1i32];
    let displs = vec![0i32];
    let r = world.gatherv(&send, &mut recv, &recvcounts, &displs, 0);
    eprintln!("rank {} gatherv -> {:?}", world.rank(), r.map_err(|e| e.to_string()));
}
