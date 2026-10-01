use ferrompi::{Mpi, ReduceOp};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let mut d = vec![world.rank() as f64 + 1.0; 3];
    let r = world.reduce_inplace(&mut d, ReduceOp::Sum, 0);
    eprintln!("rank {} reduce_inplace -> {:?} d={:?}", world.rank(), r.map_err(|e| e.to_string()), d);
}
