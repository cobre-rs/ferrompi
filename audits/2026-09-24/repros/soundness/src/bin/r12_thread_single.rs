// R12: safe code — Communicator is Send+Sync even when MPI provided THREAD_SINGLE.
use ferrompi::{Mpi, ReduceOp};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?; // ThreadLevel::Single
    println!("provided = {:?}", mpi.thread_level());
    let world = mpi.world();
    std::thread::scope(|s| {
        for t in 0..4 {
            let w = &world;
            s.spawn(move || {
                for i in 0..20000 {
                    let peer = 1 - w.rank();
                    let send = [i as u64 + t; 16];
                    let mut recv = [0u64; 16];
                    let r = w.irecv(&mut recv, peer, t as i32).unwrap();
                    w.send(&send, peer, t as i32).unwrap();
                    r.wait().unwrap();
                    assert_eq!(recv[0], i as u64 + t, "corrupted message");
                }
            });
        }
    });
    let s = world.allreduce_scalar(1.0f64, ReduceOp::Sum)?;
    println!("rank {} done {s}", world.rank());
    Ok(())
}
