use ferrompi::{Mpi, Request};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    let mut small = vec![0i32; 1];
    let mut other = vec![0i32; 1];
    let mut reqs = vec![
        world.irecv(&mut small, me, 11).unwrap(),
        world.irecv(&mut other, me, 12).unwrap(),
    ];
    world.send(&[5i32, 6, 7, 8], me, 11).unwrap();
    let r = Request::wait_any(&mut reqs);
    eprintln!("wait_any -> {:?}", r.as_ref().map_err(|e| e.to_string()));
    let t = reqs[0].test();
    eprintln!("test on failed entry -> {:?}", t.map_err(|e| e.to_string()));
    world.send(&[1i32], me, 12).unwrap();
    let mut rest = reqs.split_off(1);
    let _ = Request::wait_all(&mut rest);
    std::mem::forget(reqs);
    eprintln!("SURVIVED");
}
