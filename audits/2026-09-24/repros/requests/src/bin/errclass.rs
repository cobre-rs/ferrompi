use ferrompi::{Mpi, ReduceOp};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let mut d = vec![0.0f64; 4];
    println!("bcast root=999      -> {:?}", world.broadcast(&mut d, 999).err());
    let s = [1.0f64]; let mut r = [0.0f64];
    println!("allreduce f64 BOR   -> {:?}", world.allreduce(&s, &mut r, ReduceOp::BitwiseOr).err());
    // truncation: send 4, recv into 1
    let big = [1i32; 4]; let mut small = [0i32; 1];
    let req = world.isend(&big, 0, 3).unwrap();
    println!("recv truncate       -> {:?}", world.recv(&mut small, 0, 3).err());
    let _ = req.wait();
}
