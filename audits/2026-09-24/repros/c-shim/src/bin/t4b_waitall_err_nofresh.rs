use ferrompi::{Mpi, Request};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    let mut ok_buf = vec![0i32; 4];
    let mut small = vec![0i32; 1];
    let mut reqs = vec![
        world.irecv(&mut ok_buf, me, 10).unwrap(),
        world.irecv(&mut small, me, 11).unwrap(),
    ];
    world.send(&[1i32, 2, 3, 4], me, 10).unwrap();
    world.send(&[5i32, 6, 7, 8], me, 11).unwrap();
    let r = Request::wait_all(&mut reqs);
    eprintln!("wait_all -> {:?}", r.as_ref().map_err(|e| e.to_string()));
    for q in reqs.iter_mut() {
        let t = q.test();
        eprintln!("test on handle {} after failed wait_all -> {:?}", q.raw_handle(), t.map_err(|e| e.to_string()));
    }
    eprintln!("SURVIVED");
}
