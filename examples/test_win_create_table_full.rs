//! Regression for `Win::create` when the window table is already full.
//!
//! Fills the 256-slot window table with `Win::allocate` calls, then calls
//! `Win::create` over an owned buffer. The new window can be neither returned
//! nor freed on this rank alone, but it no longer exposes memory the caller
//! can reclaim, so the call returns `Error::ResourceExhausted`, as
//! `Win::allocate` does, instead of aborting: the window and its buffer are
//! leaked and the window counts as alive. Dropping every returned window then
//! still leaves two windows, so `Mpi` skips `MPI_Finalize`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_win_create_table_full
// mpi-test: np=1 expect=unfinalized skip-ok=openmpi-4
// mpi-test-stderr: test_win_create_table_full: calling Win::create at a full window table

use ferrompi::{Error, Mpi, MpiErrorClass, ResourceKind, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    // Probe: some Open MPI 4.x BTLs fail MPI_Win_create itself over
    // caller-owned memory, before the table is ever touched.
    match Win::create(&world, vec![0f64; 4]) {
        Ok(win) => drop(win),
        Err(Error::Mpi {
            class: MpiErrorClass::Win,
            ..
        }) => {
            common::skip(
                &world,
                "Win::create returned MPI_ERR_WIN (Open MPI 4.x TCP BTL)",
            );
            return;
        }
        Err(e) => panic!("Win::create failed: {e}"),
    }

    let mut windows = Vec::new();
    while let Ok(win) = Win::<f64>::allocate(&world, 1) {
        windows.push(win);
    }

    if world.rank() == 0 {
        eprintln!("test_win_create_table_full: calling Win::create at a full window table");
    }

    let result = Win::create(&world, vec![0f64; 4]);
    common::check(
        &world,
        matches!(
            result,
            Err(Error::ResourceExhausted {
                resource: ResourceKind::Window,
                ..
            })
        ),
        "Win::create table-full error",
    );

    drop(windows);
    drop(mpi);

    println!("test_win_create_table_full: PASS");
}
