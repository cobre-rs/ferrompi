//! A scope completes a request that was `mem::forget`-ten.
//!
//! Rank 0 forgets the request `irecv` returned and leaves the receive to scope
//! end. The request's slot stays registered, so the scope-end `MPI_Waitall`
//! completes the receive (MPI-4.1 §3.7.3) before the scope returns, and MPI
//! writes only into the buffer the scope borrows (MPI-4.1 §3.7.2). Rank 0 sends
//! the go signal from inside the scope and rank 1 sends the message 100 ms
//! later, so the receive is normally still pending when scope end starts
//! waiting. Under valgrind a receive that outlived its buffer would be an
//! invalid write.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_scope_forget
// mpi-test: np=2 valgrind

use std::time::Duration;

use ferrompi::Mpi;

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    if world.rank() == 0 {
        let mut buf = vec![0u8; 64];
        let result = ferrompi::scope(|s| {
            let request = world.irecv(s, &mut buf, 1, 3)?;
            #[allow(
                clippy::forget_non_drop,
                reason = "a Request has drop glue only with the rma feature; the forget is the case under test"
            )]
            std::mem::forget(request);
            world.send(&[0u8], 1, 9)?;
            Ok(())
        });
        common::check(&world, result.is_ok(), "the scope returns Ok");
        common::check(
            &world,
            buf == [3; 64],
            "scope end completes a forgotten receive",
        );
        println!("test_scope_forget: PASS");
    } else {
        let mut go = [0u8; 1];
        world
            .recv(&mut go, 0, 9)
            .expect("recv of the go signal failed");
        std::thread::sleep(Duration::from_millis(100));
        world.send(&[3u8; 64], 0, 3).expect("send failed");
        common::check(&world, true, "the scope returns Ok");
        common::check(&world, true, "scope end completes a forgotten receive");
    }
}
