use ferrompi::{Mpi, Request};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    let mut ra = vec![0i32; 1];
    let mut rb = vec![0i32; 1];
    let mut reqs = vec![
        world.irecv(&mut ra, me, 1).unwrap(),
        world.irecv(&mut rb, me, 2).unwrap(),
    ];
    world.send(&[1i32], me, 1).unwrap();
    world.send(&[2i32], me, 2).unwrap();
    // Classic MPI idiom: loop wait_any until None, without removing entries.
    for k in 0..3 {
        let r = Request::wait_any(&mut reqs);
        eprintln!("wait_any #{k} -> {:?}", r.as_ref().map_err(|e| e.to_string()));
        match r { Ok(None) => break, Err(_) => break, _ => {} }
    }
    eprintln!("SURVIVED");
}
