use ferrompi::{Mpi, Request};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let me = world.rank();
    // Part (a): call wait_any twice without removing the completed entry.
    let mut ra = vec![0i32; 1];
    let mut rb = vec![0i32; 1];
    let sa = vec![1i32]; let sb = vec![2i32];
    let mut reqs = vec![
        world.irecv(&mut ra, me, 1).unwrap(),
        world.irecv(&mut rb, me, 2).unwrap(),
    ];
    eprintln!("handles: {} {}", reqs[0].raw_handle(), reqs[1].raw_handle());
    world.send(&sa, me, 1).unwrap();
    let i = Request::wait_any(&mut reqs).unwrap();
    eprintln!("first wait_any -> {:?}", i);
    // Now post a NEW request; it will likely reuse the freed slot.
    let mut rc = vec![0i32; 1];
    let sc = vec![3i32];
    let c = world.irecv(&mut rc, me, 3).unwrap();
    eprintln!("new request C handle = {}", c.raw_handle());
    world.send(&sc, me, 3).unwrap();
    // Second wait_any on the ORIGINAL slice (stale completed entry still there).
    let r2 = Request::wait_any(&mut reqs);
    eprintln!("second wait_any -> {:?}", r2.as_ref().map_err(|e| e.to_string()));
    // Now wait on C: its slot may have been consumed by the stale wait_any.
    let rc_res = c.wait();
    eprintln!("C.wait() -> {:?}", rc_res.map_err(|e| e.to_string()));
    world.send(&sb, me, 2).unwrap();
    drop(reqs);
    eprintln!("rc={:?} rb={:?}", rc, rb);
    eprintln!("SURVIVED");
}
