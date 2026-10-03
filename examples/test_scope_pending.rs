//! A nonblocking scope returns only after every request it holds completed.
//!
//! Part 1: a scoped receive that fails when waited inside the scope (a message
//! longer than the buffer, MPI-4.1 §3.2.4) returns `Error::Mpi` with class
//! `Truncate`, and scope end issues no second wait on that slot: MPI completed
//! the request whatever the wait returned (MPI-4.1 §3.7.3). A second wait on
//! the released slot would abort the process.
//!
//! Part 2: scope end holds a truncating receive and a second, still pending
//! one. After the truncation `MPI_Waitall` reports the second entry
//! `MPI_ERR_PENDING` on both families (MPI-4.1 §3.7.5): MPICH returns once both
//! messages arrived, with that entry finished but not yet completed, and Open
//! MPI returns at the first error with it still incomplete. Either way the scope
//! waits again and returns the truncation only once the second receive is
//! filled. Rank 1 sends the truncating message 50 ms after the go signal rank 0
//! sends from inside the scope, so the truncation completes while scope end
//! waits.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_scope_pending
// mpi-test: np=2 valgrind

use std::time::Duration;

use ferrompi::{Error, Mpi, MpiErrorClass};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();

    if rank == 0 {
        let mut trunc = [0i32; 1];
        let waited = ferrompi::scope(|s| Ok(world.irecv(s, &mut trunc, 1, 3)?.wait()));
        common::check(
            &world,
            matches!(
                waited,
                Ok(Err(Error::Mpi {
                    class: MpiErrorClass::Truncate,
                    ..
                }))
            ),
            "a failing scoped wait reports the truncation and scope end issues no second wait",
        );

        let mut small = [0i32; 1];
        let mut big = [0i32; 4];
        let result = ferrompi::scope(|s| {
            world.irecv(s, &mut small, 1, 1)?;
            world.irecv(s, &mut big, 1, 2)?;
            world.send(&[0u8], 1, 9)?;
            Ok(())
        });
        // Read `big` before the first check: its collective lets MPI fill a
        // receive the scope left pending.
        let truncated = matches!(
            result,
            Err(Error::Mpi {
                class: MpiErrorClass::Truncate,
                ..
            })
        );
        let filled = big == [2; 4];
        common::check(&world, truncated, "scope end reports the truncation");
        common::check(&world, filled, "scope end waits the pending receive");
        println!("test_scope_pending: PASS");
    } else {
        world.send(&[9i32; 4], 0, 3).expect("send failed");
        common::check(
            &world,
            true,
            "a failing scoped wait reports the truncation and scope end issues no second wait",
        );

        let mut go = [0u8; 1];
        world
            .recv(&mut go, 0, 9)
            .expect("recv of the go signal failed");
        std::thread::sleep(Duration::from_millis(50));
        world.send(&[1i32; 4], 0, 1).expect("send failed");
        std::thread::sleep(Duration::from_millis(200));
        world.send(&[2i32; 4], 0, 2).expect("send failed");
        common::check(&world, true, "scope end reports the truncation");
        common::check(&world, true, "scope end waits the pending receive");
    }
}
