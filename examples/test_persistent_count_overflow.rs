//! Integration test for large-count dispatch in the persistent `*_init` shims.
//!
//! From MPI 4.0 on, `send_init` (point-to-point) and `bcast_init` (collective)
//! both dispatch the `_c` large-count variant for a count above `INT_MAX` and
//! return `Ok`. Below MPI 4.0, `send_init` returns
//! `Err(Error::Mpi { class: MpiErrorClass::Count, .. })`, while `bcast_init`
//! (stubbed out where the library lacks persistent collectives) returns
//! `Err(Error::NotSupported(_))`. On Open MPI 5, which provides persistent
//! collectives but keeps the `_c` large-count call gated on reporting MPI
//! 4.0, `bcast_init` also returns `Err(Count)` for the same reason
//! `send_init` does. Neither request is ever started, so dropping both
//! without a matching operation is legal.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_count_overflow
// mpi-test: np=2

use ferrompi::{Error, Mpi, MpiErrorClass};

mod common;

const N: usize = (1 << 32) + 16;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_count_overflow requires exactly 2 processes, got {size}"
    );

    // Zeroed allocations whose pages the shim under test never touches:
    // *_init only registers the operation, it never starts it.
    let send = vec![0u8; N];
    let mut data = vec![0u8; N];

    let p2p = world.send_init(&send, 1 - rank, 3);
    let coll = world.bcast_init(&mut data, 0);

    let ok = if common::mpi_major() >= 4 {
        p2p.is_ok() && coll.is_ok()
    } else {
        let p2p_ok = matches!(
            &p2p,
            Err(Error::Mpi {
                class: MpiErrorClass::Count,
                ..
            })
        );
        let coll_ok = if common::has_mpi4_collectives() {
            common::is_count(&coll)
        } else {
            matches!(&coll, Err(Error::NotSupported(_)))
        };
        p2p_ok && coll_ok
    };

    // Neither request was ever started, so freeing an inactive persistent
    // request here is legal.
    drop(p2p);
    drop(coll);

    common::check(&world, ok, "persistent init of 2^32+16 bytes");

    if rank == 0 {
        println!("PASS: test_persistent_count_overflow");
    }
}
