use ferrompi::Mpi;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    let mut small = vec![0i32; 1];
    let mut a = world.irecv(&mut small, me, 1).unwrap();
    eprintln!("A handle {}", a.raw_handle());
    world.send(&[1i32, 2, 3], me, 1).unwrap(); // truncation error for A
    let t = a.test();
    eprintln!("A.test() -> {:?} (A.is_completed={})", t.as_ref().map_err(|e| e.to_string()), a.is_completed());
    // New request B takes the lowest free slot
    let mut bbuf = vec![0i32; 1];
    let b = world.irecv(&mut bbuf, me, 2).unwrap();
    eprintln!("B handle {}", b.raw_handle());
    world.send(&[42i32], me, 2).unwrap();
    eprintln!("dropping A (Drop -> ferrompi_wait on A's stale handle)");
    drop(a);
    let r = b.wait();
    eprintln!("B.wait() -> {:?}; bbuf={:?}", r.map_err(|e| e.to_string()), bbuf);
    eprintln!("SURVIVED");
}
