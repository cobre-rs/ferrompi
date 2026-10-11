//! Integration test for the `MPI_COMM_NULL` guard in
//! `ferrompi_comm_create_from_group` (MPI-4 path).
//!
//! Calls `Mpi::create_from_group` with a group that excludes some ranks,
//! verifying that those ranks receive an `Err` from the create call itself
//! (previously the shim stored a corrupt handle because the NULL guard was
//! missing).
//!
//! Where the library lacks it (below MPI 4.0, except Open MPI 5), the shim is
//! an unconditional stub: `create_from_group` returns `Error::NotSupported`.
//! That the stub also leaves the output handle untouched is checked by the
//! `checks_that_need_mpi` unit test in `src/lib.rs`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_create_from_group_null_handle
// mpi-test: np=2

use ferrompi::{Error, Mpi, ReduceOp};

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

    // ========================================================================
    // Build a group containing only rank 0.  Rank 1 is excluded and will
    // receive MPI_COMM_NULL from MPI_Comm_create_from_group.
    // ========================================================================
    let rank0_group = world_group
        .include(&[0])
        .expect("group.include(&[0]) failed");

    let result = mpi.create_from_group(&rank0_group, "test_null_guard");

    if rank == 1 {
        // Rank 1 is not in the group; MPI_Comm_create_from_group returns
        // MPI_COMM_NULL for this rank.  The guard must refuse that in the
        // create call itself rather than store MPI_COMM_NULL in the handle
        // table.  Without the guard the caller still gets an error, but from
        // the rank query on the stored null handle (operation `comm_rank`),
        // so the operation is what tells the two apart.
        match result.err() {
            Some(
                err @ Error::Mpi {
                    operation: Some("comm_create_from_group"),
                    ..
                },
            ) => {
                println!("PASS rank {rank}: create_from_group for excluded rank returned: {err:?}");
            }
            // The capability gate above already excludes libraries without
            // MPI_Comm_create_from_group, so anything else is definitely a bug.
            other => {
                eprintln!(
                    "rank {rank}: FAIL: create_from_group for excluded rank was not refused \
                     by the create call (its error, if any: {other:?}) — MPI_COMM_NULL \
                     guard is missing or not compiled in"
                );
                // Participate in the sentinel barrier before exiting.
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        }
    } else {
        // Rank 0 is in the group; the capability gate above already excludes
        // libraries without MPI_Comm_create_from_group, so it must succeed.
        match result {
            Ok(_) => println!("PASS rank {rank}: create_from_group succeeded for member rank"),
            Err(err) => {
                eprintln!(
                    "rank {rank}: FAIL: create_from_group for member rank returned \
                     unexpected error: {err:?}"
                );
                let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
                std::process::exit(1);
            }
        }
    }

    // Sentinel barrier: all ranks must reach this point.
    world
        .allreduce_scalar(1i32, ReduceOp::Min)
        .expect("sentinel allreduce failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All create_from_group_null_handle tests passed!");
        println!("========================================");
    }
}
