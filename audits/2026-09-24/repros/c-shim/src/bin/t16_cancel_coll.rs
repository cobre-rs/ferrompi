use ferrompi::{Mpi, ReduceOp};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let send = vec![1.0f64; 4];
    let mut recv = vec![0.0f64; 4];
    if world.rank() == 1 { std::thread::sleep(std::time::Duration::from_millis(500)); }
    let mut req = world.iallreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
    if world.rank() == 0 {
        let c = req.cancel();
        eprintln!("rank0 cancel(iallreduce) -> {:?}", c.map_err(|e| e.to_string()));
    }
    let w = req.wait();
    eprintln!("rank {} wait -> {:?} recv={:?}", world.rank(), w.map_err(|e| e.to_string()), recv);
}
