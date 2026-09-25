use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let mut d = vec![0.0f64; 4];
    match world.broadcast(&mut d, 999) { Err(e) => eprintln!("bcast root=999: {e}"), Ok(_) => eprintln!("ok?") }
    let me = world.rank();
    let mut small = vec![0i32; 1];
    let r = world.irecv(&mut small, me, 5).unwrap();
    world.send(&[1i32,2,3], me, 5).unwrap();
    match r.wait() { Err(e) => eprintln!("truncate: {e}"), Ok(_) => eprintln!("ok?") }
}
