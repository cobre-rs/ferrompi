use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let send = vec![7.0f64; 64];
    let mut recv = vec![0.0f64; 1];           // far too small
    let recvcounts = [64i32];
    let displs = [0i32];
    let r = world.gatherv(&send, &mut recv, &recvcounts, &displs, 0);
    eprintln!("gatherv -> {:?}", r.map_err(|e| e.to_string()));
    eprintln!("SURVIVED");
}
