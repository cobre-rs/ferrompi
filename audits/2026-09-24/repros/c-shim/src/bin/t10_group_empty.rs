use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let g = world.group().unwrap();
    for i in 0..3 {
        let e = g.include(&[]).unwrap();
        eprintln!("iter {i}: empty group handle {} size {:?}", e.raw_handle(), e.size());
        let d = g.difference(&g).unwrap();
        eprintln!("iter {i}: diff group handle {} size {:?}", d.raw_handle(), d.size());
    }
    eprintln!("SURVIVED");
}
