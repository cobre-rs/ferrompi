//! Integration test for `Mpi::create_from_group` (MPI 4.0, or Open MPI 5).
//!
//! Exercises `MPI_Comm_create_from_group` via the
//! `ferrompi_comm_create_from_group` shim.  Verifies that:
//!
//! - On MPI 4.0, or Open MPI 5: every rank that calls `mpi.create_from_group(&g, tag)`
//!   with the same world group and the same tag receives an `Ok(Some(comm))`
//!   with `comm.size() == world.size()`.
//! - On MPI 4.0, or Open MPI 5: a string tag of 62 bytes, the longest every
//!   library must accept, creates a communicator, and one of 4096 bytes, past
//!   the limit of every library this suite runs on, is refused on every rank
//!   with `Error::InvalidArgument`.
//! - Where the library lacks it (below MPI 4.0, except Open MPI 5):
//!   `create_from_group` returns `Err(Error::NotSupported(_))` instead of
//!   calling into the (unbuilt) MPI 4.0 shim, for a tag of any length.
//!
//! All assertions are guarded by a sentinel `allreduce_scalar(Min)` before
//! any `process::exit` so that no rank exits while others are still inside
//! MPI collective calls.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_mpi_from_group
// mpi-test: np=4

use ferrompi::{Error, Mpi, ReduceOp};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 4,
        "test_mpi_from_group requires exactly 4 processes, got {size}"
    );

    // local_ok tracks whether this rank passed all its assertions.
    let mut local_ok = true;

    // ========================================================================
    // Build the world group — all ranks participate with the same tag.
    // ========================================================================
    let world_group = world.group().expect("world.group() failed");

    // ========================================================================
    // Version check — below MPI 4.0, create_from_group refuses.
    // ========================================================================
    if !common::has_mpi4_collectives() {
        // A library without the call defines no limit on the tag either, so
        // the answer is the same for a tag of any length.
        let ok = [String::from("ferrompi-test"), "t".repeat(4096)]
            .iter()
            .all(|tag| {
                matches!(
                    mpi.create_from_group(&world_group, tag),
                    Err(Error::NotSupported(op)) if op == "comm_create_from_group"
                )
            });
        common::check(
            &world,
            ok,
            "create_from_group refuses without MPI_Comm_create_from_group",
        );
        return;
    }

    // ========================================================================
    // Test 1: create_from_group returns Ok(Some(comm)) to a member.
    // ========================================================================
    let comm = match mpi.create_from_group(&world_group, "ferrompi-test") {
        Ok(Some(c)) => c,
        other => {
            eprintln!(
                "rank {rank}: FAIL: create_from_group returned {:?}",
                other.map(|c| c.map(|c| c.size()))
            );
            let _ = world.allreduce_scalar(0i32, ReduceOp::Min);
            std::process::exit(1);
        }
    };

    // ========================================================================
    // Test 2: the new communicator has the same size as COMM_WORLD.
    // ========================================================================
    let comm_size = comm.size();
    if comm_size == size {
        if rank == 0 {
            println!("PASS: Test 2 — comm.size() == world.size() == {comm_size}");
        }
    } else {
        eprintln!("rank {rank}: FAIL Test 2: comm.size() = {comm_size}, expected {size}");
        local_ok = false;
    }

    // ========================================================================
    // Test 3: allreduce on the new communicator is functional.
    // ========================================================================
    match comm.allreduce_scalar(1i32, ReduceOp::Sum) {
        Ok(result) if result == size => {
            if rank == 0 {
                println!("PASS: Test 3 — allreduce(1, Sum) == {result} (== world.size())");
            }
        }
        Ok(result) => {
            eprintln!("rank {rank}: FAIL Test 3: allreduce result = {result}, expected {size}");
            local_ok = false;
        }
        Err(e) => {
            eprintln!("rank {rank}: FAIL Test 3: allreduce failed: {e}");
            local_ok = false;
        }
    }

    // ========================================================================
    // Test 4: the length of the string tag is checked before the MPI call.
    // ========================================================================
    // 62 bytes are the longest tag every library must accept
    // (MPI-4.1 §7.4.2, p.339).
    match mpi.create_from_group(&world_group, &"t".repeat(62)) {
        Ok(Some(c)) if c.size() == size => {}
        other => {
            eprintln!(
                "rank {rank}: FAIL Test 4: a 62-byte tag returned {:?}",
                other.map(|c| c.map(|c| c.size()))
            );
            local_ok = false;
        }
    }
    // 4096 bytes are past the limit of every library this suite runs on: 256
    // in MPICH 4.2, and at most 2048 in Open MPI 5.0, however it was built.
    match mpi.create_from_group(&world_group, &"t".repeat(4096)) {
        Err(Error::InvalidArgument {
            arg: "stringtag", ..
        }) => {
            if rank == 0 {
                println!("PASS: Test 4 — a string tag past the limit is refused");
            }
        }
        other => {
            eprintln!(
                "rank {rank}: FAIL Test 4: a 4096-byte tag returned {:?}",
                other.map(|c| c.map(|c| c.size()))
            );
            local_ok = false;
        }
    }

    common::check(&world, local_ok, "test_mpi_from_group");
}
