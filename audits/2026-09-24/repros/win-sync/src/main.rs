use ferrompi::{Mpi, Win};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let win = Win::<f64>::allocate(&world, 8).unwrap();
    // Verbatim from the Win::sync rustdoc: "sync is valid outside any epoch"
    match win.sync() {
        Ok(()) => println!("rank {}: bare Win::sync outside epoch -> Ok", world.rank()),
        Err(e) => println!("rank {}: bare Win::sync outside epoch -> Err: {e}", world.rank()),
    }
}
