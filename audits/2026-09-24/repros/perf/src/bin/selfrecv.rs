use ferrompi::Mpi;
use perfprobe::*;
use std::hint::black_box;
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    assert_eq!(world.size(), 1);
    let reps = 21; let sync = || { world.barrier().unwrap(); };
    let sb = [1.0f64]; let mut rb1 = [0.0f64]; let mut rb2 = [0.0f64];
    // isolate blocking recv wrapper (status materialization + MPI_Get_count_c)
    let r = ab(reps, 50000,
        || unsafe { let mut q = 0; MPI_Isend(sb.as_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mut q); MPI_Recv(rb1.as_mut_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Wait(&mut q, MPI_STATUS_IGNORE); },
        || unsafe { let mut q = 0; MPI_Isend(sb.as_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mut q); black_box(world.recv(&mut rb2, 0, 0).unwrap()); MPI_Wait(&mut q, MPI_STATUS_IGNORE); }, &sync);
    println!("recv wrapper: raw(STATUS_IGNORE) {:.1} ns | ferrompi recv {:.1} ns | delta {:+.1} ns", r.0, r.1, r.1 - r.0);
    let r = ab(reps, 50000,
        || unsafe { let mut q = 0; MPI_Irecv(rb1.as_mut_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mut q); MPI_Send(sb.as_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); MPI_Wait(&mut q, MPI_STATUS_IGNORE); },
        || unsafe { let mut q = 0; MPI_Irecv(rb2.as_mut_ptr().cast(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mut q); world.send(&sb, 0, 0).unwrap(); MPI_Wait(&mut q, MPI_STATUS_IGNORE); }, &sync);
    println!("send wrapper: raw {:.1} ns | ferrompi send {:.1} ns | delta {:+.1} ns", r.0, r.1, r.1 - r.0);
}
