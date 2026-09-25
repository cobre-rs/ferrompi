use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let r = world.send(&[1i32, 2, 3], -1, 0);
    eprintln!("send(dest=-1) -> {:?}", r.map_err(|e| e.to_string()));
}
