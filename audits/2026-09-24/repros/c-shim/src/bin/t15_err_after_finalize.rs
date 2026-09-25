use ferrompi::Mpi;
fn main() {
    let req;
    let mut buf = Box::new([0i32; 1]);
    {
        let mpi = Mpi::init().unwrap();
        let world = mpi.world();
        let me = world.rank();
        req = world.irecv(&mut buf[..], me, 3).unwrap();
        world.send(&[7i32], me, 3).unwrap();
    } // finalize: request table swept, bits cleared
    eprintln!("finalized; req.wait()");
    let r = req.wait();
    eprintln!("wait -> {:?}", r.map_err(|e| e.to_string()));
    eprintln!("SURVIVED");
}
