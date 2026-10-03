//! A scope that unwinds waits for its requests before the panic leaves it.
//!
//! Rank 0 posts a receive, sends the go signal, and panics inside the scope.
//! The scope-end `MPI_Waitall` runs while the panic unwinds through the scope,
//! so it returns only once the message has arrived, and the buffer is complete
//! when `catch_unwind` returns and before the buffer can be dropped. Rank 1
//! sends the message 100 ms after the go signal, so the receive is normally
//! still pending when the unwind starts. Under valgrind a receive left pending
//! past the buffer's life would be an invalid write.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_scope_panic
// mpi-test: np=2 valgrind

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use ferrompi::Mpi;

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    if world.rank() == 0 {
        std::panic::set_hook(Box::new(|_| {}));
        let mut buf = vec![0u8; 64];
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            ferrompi::scope::<()>(|s| {
                world.irecv(s, &mut buf, 1, 4)?;
                world.send(&[0u8], 1, 9)?;
                panic!("inside the scope")
            })
        }));
        common::check(&world, outcome.is_err(), "the panic leaves the scope");
        common::check(
            &world,
            buf == [4; 64],
            "the unwind waits the pending receive",
        );
        println!("test_scope_panic: PASS");
    } else {
        let mut go = [0u8; 1];
        world
            .recv(&mut go, 0, 9)
            .expect("recv of the go signal failed");
        std::thread::sleep(Duration::from_millis(100));
        world.send(&[4u8; 64], 0, 4).expect("send failed");
        common::check(&world, true, "the panic leaves the scope");
        common::check(&world, true, "the unwind waits the pending receive");
    }
}
