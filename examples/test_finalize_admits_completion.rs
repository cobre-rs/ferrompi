//! After `Mpi::drop` skips `MPI_Finalize` because a window is still alive and
//! a nonblocking scope is open, MPI stays initialized: `wait` completes a
//! request that was pending when the handle dropped, every other call returns
//! `Error::Finalized`, and the handles dropped afterwards are leaked without
//! an MPI call.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_finalize_admits_completion
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: MPI_Finalize skipped

use ferrompi::{Error, Mpi, Win};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let win = Win::<f64>::allocate(&world, 8).expect("Win::allocate failed");
    let group = world.group().expect("group failed");

    let data = [1.0f64, 2.0, 3.0, 4.0];
    let mut buf = [0.0f64; 4];
    ferrompi::scope(|s| {
        let rreq = world.irecv(s, &mut buf, 0, 5).expect("irecv failed");
        let sreq = world.isend(s, &data, 0, 5).expect("isend failed");

        drop(mpi);

        assert!(
            Mpi::is_finalized(),
            "Mpi::is_finalized() must be true after a skipped MPI_Finalize"
        );
        rreq.wait()
            .expect("wait on a pending receive must complete after the skip");
        sreq.wait()
            .expect("wait on a pending send must complete after the skip");
        assert!(
            matches!(world.barrier(), Err(Error::Finalized)),
            "a call that is not a completion must return Err(Finalized)"
        );
        Ok(())
    })
    .expect("scope failed");
    assert_eq!(buf, data, "the completed receive must hold the message");

    drop(group);
    drop(win);

    println!("test_finalize_admits_completion: PASS");
}
