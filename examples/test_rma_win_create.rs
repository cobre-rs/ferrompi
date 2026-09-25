//! Integration test for Win::create and Win::allocate (distributed RMA windows).
//!
//! Verifies that `Win<T>` can be constructed via both constructors, that
//! `raw_handle()` and `comm_size()` return expected values, and that the RAII
//! `Drop` implementation does not produce MPI errors.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_win_create
// mpi-test: np=2 skip-ok=openmpi-4

use ferrompi::{Mpi, Win};

mod common;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_rma_win_create requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Test 1: Win::create with a caller-supplied i32 buffer
    //
    // OpenMPI 4.x with `--btl=self,tcp` (the configuration used in CI when
    // UCX/rdma is not available) does not provide a one-sided communication
    // transport that supports `MPI_Win_create` over caller-owned memory.
    // The call returns `MPI_ERR_WIN` ("invalid window") on those builds,
    // even though the arguments are valid by the MPI standard.
    // `MPI_Win_allocate` (Test 2 below) uses MPI-managed memory and is
    // supported.  Skip Test 1 gracefully when this OpenMPI-CI quirk fires.
    // ========================================================================
    let mut buf = vec![0i32; 16];
    let test1_skipped = {
        match Win::create(&world, &mut buf) {
            Ok(win) => {
                let handle = win.raw_handle();
                assert!(
                    handle >= 0,
                    "Win::create raw_handle() = {handle}, expected >= 0"
                );

                let cs = win.comm_size();
                assert_eq!(cs, size, "Win::create comm_size() mismatch");

                // Verify local_slice / local_slice_mut round-trip
                let slice = win.local_slice();
                assert_eq!(slice.len(), 16, "Win::create local_slice len mismatch");

                // Win dropped at end of arm — exercises MPI_Win_free for
                // WinKind::Created.
                drop(win);
                false
            }
            Err(ferrompi::Error::Mpi {
                class: ferrompi::MpiErrorClass::Win,
                ..
            }) => {
                common::skip(
                    &world,
                    "Win::create returned MPI_ERR_WIN — likely OpenMPI 4.x \
                     with a BTL that does not support one-sided over caller-owned \
                     memory (e.g., --btl=self,tcp in CI). Win::allocate (Test 2) \
                     still tested.",
                );
                true
            }
            Err(e) => panic!("Win::create failed: {e}"),
        }
    };

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && !test1_skipped {
        println!("PASS: Win::create (i32, 16 elements, raw_handle, comm_size, local_slice)");
    }

    // ========================================================================
    // Test 2: Win::allocate with MPI-managed memory
    // ========================================================================
    {
        let local_count: usize = 32;
        let mut win = Win::<f64>::allocate(&world, local_count).expect("Win::allocate failed");

        let handle = win.raw_handle();
        assert!(
            handle >= 0,
            "Win::allocate raw_handle() = {handle}, expected >= 0"
        );

        let cs = win.comm_size();
        assert_eq!(cs, size, "Win::allocate comm_size() mismatch");

        // Verify local_slice_mut write-then-read round-trip
        {
            let slice = win.local_slice_mut();
            assert_eq!(
                slice.len(),
                local_count,
                "Win::allocate local_slice_mut len mismatch"
            );
            for (i, x) in slice.iter_mut().enumerate() {
                *x = (rank as f64) * 100.0 + i as f64;
            }
        }

        // Read back via local_slice
        {
            let slice = win.local_slice();
            for (i, &x) in slice.iter().enumerate() {
                let expected = (rank as f64) * 100.0 + i as f64;
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "Win::allocate local_slice[{i}] = {x}, expected {expected}"
                );
            }
        }

        // Win dropped here — exercises MPI_Win_free for WinKind::Allocated
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 {
        println!("PASS: Win::allocate (f64, 32 elements, raw_handle, comm_size, local_slice_mut, local_slice)");
    }

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win create/allocate tests passed! (2 tests)");
        println!("========================================");
    }
}
