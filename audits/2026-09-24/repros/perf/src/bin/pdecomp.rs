use ferrompi::{Mpi, PersistentRequest, ReduceOp};
use perfprobe::*;
extern "C" {
    fn ferrompi_startall(n: i64, h: *mut i64) -> i32;
    fn ferrompi_waitall(n: i64, h: *mut i64, done: *mut u8, failed_index: *mut i64) -> i32;
    fn ferrompi_start(h: i64) -> i32;
    fn ferrompi_wait(h: i64) -> i32;
}
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let rank = world.rank();
    let reps = 21;
    let sync = || {
        world.barrier().unwrap();
    };
    for &k in &[1usize, 2, 4, 8] {
        let s = vec![1.0f64; k];
        let mut r1 = vec![0.0f64; k];
        let mut r2 = vec![0.0f64; k];
        let mut r3 = vec![0.0f64; k];
        let mut q = vec![0i32; k];
        for i in 0..k {
            unsafe {
                MPI_Allreduce_init(
                    s.as_ptr().add(i).cast(),
                    r1.as_mut_ptr().add(i).cast(),
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD,
                    MPI_INFO_NULL,
                    &mut q[i],
                );
            }
        }
        let mut p2: Vec<PersistentRequest> = r2
            .chunks_mut(1)
            .enumerate()
            .map(|(i, x)| {
                world
                    .allreduce_init(&s[i..i + 1], x, ReduceOp::Sum)
                    .unwrap()
            })
            .collect();
        let mut p3: Vec<PersistentRequest> = r3
            .chunks_mut(1)
            .enumerate()
            .map(|(i, x)| {
                world
                    .allreduce_init(&s[i..i + 1], x, ReduceOp::Sum)
                    .unwrap()
            })
            .collect();
        let mut h2: Vec<i64> = p2.iter().map(|p| p.raw_handle()).collect();
        let mut done = vec![0u8; k];
        let mut failed_index: i64 = -1;
        let a = ab(
            reps,
            20000,
            || unsafe {
                MPI_Startall(k as i32, q.as_mut_ptr());
                MPI_Waitall(k as i32, q.as_mut_ptr(), MPI_STATUS_IGNORE);
            },
            || unsafe {
                ferrompi_startall(k as i64, h2.as_mut_ptr());
                ferrompi_waitall(
                    k as i64,
                    h2.as_mut_ptr(),
                    done.as_mut_ptr(),
                    &mut failed_index,
                );
            },
            &sync,
        );
        let b = ab(
            reps,
            20000,
            || unsafe {
                ferrompi_startall(k as i64, h2.as_mut_ptr());
                ferrompi_waitall(
                    k as i64,
                    h2.as_mut_ptr(),
                    done.as_mut_ptr(),
                    &mut failed_index,
                );
            },
            || {
                PersistentRequest::start_all(&mut p3).unwrap();
                PersistentRequest::wait_all(&mut p3).unwrap();
            },
            &sync,
        );
        let c = ab(
            reps,
            20000,
            || unsafe {
                for i in 0..k {
                    MPI_Start(&mut q[i]);
                }
                for i in 0..k {
                    MPI_Wait(&mut q[i], MPI_STATUS_IGNORE);
                }
            },
            || {
                for p in p3.iter_mut() {
                    p.start().unwrap();
                }
                for p in p3.iter_mut() {
                    p.wait().unwrap();
                }
            },
            &sync,
        );
        if rank == 0 {
            println!("k={k}: startall+waitall raw {:>6.1} | C shim {:>6.1} (+{:>5.1}) | Rust API {:>6.1} (+{:>5.1} over shim) || loop start/wait raw {:>6.1} Rust {:>6.1} (+{:>5.1})",
            a.0, a.1, a.1 - a.0, b.1, b.1 - b.0, c.0, c.1, c.1 - c.0);
        }
        for x in q.iter_mut() {
            unsafe {
                MPI_Request_free(x);
            }
        }
        let _ = (unsafe { ferrompi_start as usize }, unsafe {
            ferrompi_wait as usize
        });
        drop(p2);
        drop(p3);
    }
    sync();
}
