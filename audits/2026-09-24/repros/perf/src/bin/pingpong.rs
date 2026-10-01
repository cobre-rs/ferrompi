use ferrompi::Mpi;
use perfprobe::*;
use std::hint::black_box;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let n = world.size();
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let sync = || { world.barrier().unwrap(); };
    let peer = if n == 1 { 0 } else { 1 - rank };
    let mut b1 = [1.0f64]; let mut b2 = [1.0f64];
    let r = ab(reps, 20000,
        || unsafe { if rank == 0 { MPI_Send(b1.as_ptr().cast(), 1, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD); MPI_Recv(b1.as_mut_ptr().cast(), 1, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); } else { MPI_Recv(b1.as_mut_ptr().cast(), 1, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Send(b1.as_ptr().cast(), 1, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD); } },
        || { if rank == 0 { world.send(&b2, peer, 0).unwrap(); black_box(world.recv(&mut b2, peer, 0).unwrap()); } else { black_box(world.recv(&mut b2, peer, 0).unwrap()); world.send(&b2, peer, 0).unwrap(); } },
        &sync);
    if rank == 0 { println!("{:<34} raw med {:>8.1} ns | ferrompi med {:>8.1} ns | delta {:>6.1} ns ({:>4.1}%) | min raw {:>8.1} ferro {:>8.1}", "send/recv ping-pong RTT (1 f64)", r.0, r.1, r.1 - r.0, 100.0 * (r.1 - r.0) / r.0, r.2, r.3); }
    sync();
}
