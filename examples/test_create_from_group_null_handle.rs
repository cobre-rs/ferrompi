//! Integration test for the `MPI_COMM_NULL` guard in
//! `ferrompi_comm_create_from_group` (MPI-4 path).
//!
//! 1. `Mpi::create_from_group` with a group that holds rank 0 only. Rank 0
//!    receives `Ok(Some(comm))` of size 1. Rank 1 is outside the group: MPI
//!    returns `MPI_COMM_NULL` to it, and the shim must report that as "no
//!    communicator", so the call returns `Ok(None)`. Without the guard the null
//!    handle is stored in the communicator table, and the call returns the
//!    error of the rank query on it (operation `comm_rank`).
//! 2. With the empty group every rank receives `Ok(None)`.
//!
//! Where the library lacks it (below MPI 4.0, except Open MPI 5), the shim is
//! an unconditional stub: `create_from_group` returns `Error::NotSupported`.
//! That the stub also leaves the output handle untouched is checked by the
//! `checks_that_need_mpi` unit test in `src/lib.rs`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_create_from_group_null_handle
// mpi-test: np=2

use ferrompi::{Error, Mpi};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_create_from_group_null_handle requires exactly 2 processes, got {size}"
    );

    let world_group = world.group().expect("world.group() failed");

    if !common::has_mpi4_collectives() {
        let ok = matches!(
            mpi.create_from_group(&world_group, "test_null_guard"),
            Err(Error::NotSupported(_))
        );
        common::check(
            &world,
            ok,
            "create_from_group refuses without MPI_Comm_create_from_group",
        );
        return;
    }

    // 1. A group of rank 0 alone: rank 1 is outside it.
    let rank0_group = world_group
        .include(&[0])
        .expect("group.include(&[0]) failed");
    let result = mpi.create_from_group(&rank0_group, "test_null_guard");
    let ok = match &result {
        Ok(Some(comm)) => rank == 0 && comm.size() == 1,
        Ok(None) => rank == 1,
        Err(_) => false,
    };
    if !ok {
        eprintln!(
            "rank {rank}: create_from_group returned {:?}",
            result.as_ref().map(|c| c.as_ref().map(|c| c.size()))
        );
    }
    drop(result);
    common::check(
        &world,
        ok,
        "create_from_group returns None to a rank outside the group",
    );

    // 2. The empty group.
    let nobody = world_group.include(&[]).expect("include of no rank failed");
    let result = mpi.create_from_group(&nobody, "test_null_guard_empty");
    let ok = matches!(&result, Ok(None));
    if !ok {
        eprintln!(
            "rank {rank}: create_from_group of the empty group returned {:?}",
            result.as_ref().map(|c| c.as_ref().map(|c| c.size()))
        );
    }
    drop(result);
    common::check(
        &world,
        ok,
        "create_from_group of the empty group returns None",
    );

    if rank == 0 {
        println!("PASS: test_create_from_group_null_handle");
    }
}
