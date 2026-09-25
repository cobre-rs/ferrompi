use ferrompi::{Mpi, Request};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    let mut ok_buf = vec![0i32; 4];
    let mut small = vec![0i32; 1]; // truncation target
    let mut reqs = vec![
        world.irecv(&mut ok_buf, me, 10).unwrap(),
        world.irecv(&mut small, me, 11).unwrap(),
    ];
    eprintln!("handles {} {}", reqs[0].raw_handle(), reqs[1].raw_handle());
    world.send(&[1i32, 2, 3, 4], me, 10).unwrap();
    world.send(&[5i32, 6, 7, 8], me, 11).unwrap(); // truncates into `small`
    let r = Request::wait_all(&mut reqs);
    eprintln!("wait_all -> {:?}", r.as_ref().map_err(|e| e.to_string()));
    // Post a fresh request between; MPI may recycle the freed MPI_Request object
    let mut fresh = vec![0i32; 1];
    let f = world.irecv(&mut fresh, me, 12).unwrap();
    eprintln!("fresh handle {}", f.raw_handle());
    eprintln!("dropping reqs (Drop re-waits stale MPI_Request values)");
    drop(reqs);
    eprintln!("dropped reqs; now sending to fresh");
    world.send(&[9i32], me, 12).unwrap();
    let fr = f.wait();
    eprintln!("fresh.wait -> {:?} fresh={:?}", fr.map_err(|e| e.to_string()), fresh);
    eprintln!("SURVIVED");
}
