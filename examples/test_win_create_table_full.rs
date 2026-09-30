//! Regression for `Win::create` when the window table is already full.
//!
//! Fills the 256-slot window table with `Win::allocate` calls, then calls
//! `Win::create` over a caller-owned buffer. The new window already exposes
//! that buffer to every peer once `MPI_Win_create` returns, so on a full
//! table the process must abort instead of returning an error the buffer
//! could outlive.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_win_create_table_full
// mpi-test: np=1.. expect=abort skip-ok=openmpi-4
// mpi-test-stderr: ferrompi: window table full after MPI_Win_create

use ferrompi::{Error, Mpi, MpiErrorClass, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    // Probe: some Open MPI 4.x BTLs fail MPI_Win_create itself over
    // caller-owned memory, before the table is ever touched.
    let mut probe_buf = [0f64; 4];
    match Win::create(&world, &mut probe_buf) {
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

    let mut buf = [0f64; 4];
    let result = Win::create(&world, &mut buf).map(|_| ());
    println!("FAIL: Win::create returned at a full window table: {result:?}");
    std::process::exit(1);
}
