//! Integration test for Win::create and Win::allocate (distributed RMA windows).
//!
//! Verifies that `Win<T>` can be constructed via both constructors, that
//! `raw_handle()` and `comm_size()` return expected values, and that
//! `into_buffer()` frees the window and returns the `Win::create` buffer (and
//! `None` for a `Win::allocate` window).
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
    // Test 1: Win::create over an owned i32 buffer
    //
    // OpenMPI 4.x with `--btl=self,tcp` (the configuration used in CI when
    // UCX/rdma is not available) does not provide a one-sided communication
    // transport that supports `MPI_Win_create` over caller-owned memory.
    // The call returns `MPI_ERR_WIN` ("invalid window") on those builds,
    // even though the arguments are valid by the MPI standard.
    // `MPI_Win_allocate` (Test 2 below) uses MPI-managed memory and is
    // supported.  Skip Test 1 gracefully when this OpenMPI-CI quirk fires.
    // ========================================================================
    let test1_skipped = match Win::create(&world, vec![0i32; 16]) {
        Ok(win) => {
            let handle = win.raw_handle();
            assert!(
                handle >= 0,
                "Win::create raw_handle() = {handle}, expected >= 0"
            );

            assert_eq!(win.comm_size(), size, "Win::create comm_size() mismatch");

            assert_eq!(win.local().len(), 16, "Win::create local len mismatch");

            // Write through the window, then free it: the owned buffer comes
            // back holding what was written.
            let expected: Vec<i32> = (0..16).map(|i| rank * 100 + i).collect();
            win.local().copy_from_slice(&expected);
            let buf = win
                .into_buffer()
                .expect("Win::create into_buffer failed")
                .expect("Win::create must return its buffer");
            assert_eq!(buf, expected, "Win::create into_buffer data mismatch");
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
    };

    world.barrier().expect("barrier after test 1 failed");
    if rank == 0 && !test1_skipped {
        println!("PASS: Win::create (i32, 16 elements, raw_handle, comm_size, local, into_buffer)");
    }

    // ========================================================================
    // Test 2: Win::allocate with MPI-managed memory
    // ========================================================================
    {
        let local_count: usize = 32;
        let win = Win::<f64>::allocate(&world, local_count).expect("Win::allocate failed");

        let handle = win.raw_handle();
        assert!(
            handle >= 0,
            "Win::allocate raw_handle() = {handle}, expected >= 0"
        );

        assert_eq!(win.comm_size(), size, "Win::allocate comm_size() mismatch");

        // Verify local write-then-read round-trip
        {
            let local = win.local();
            assert_eq!(local.len(), local_count, "Win::allocate local len mismatch");
            for i in 0..local_count {
                local.write(i, (rank as f64) * 100.0 + i as f64);
            }
        }

        // Read back via local
        {
            let local = win.local();
            for i in 0..local_count {
                let x = local.read(i);
                let expected = (rank as f64) * 100.0 + i as f64;
                assert!(
                    (x - expected).abs() < f64::EPSILON,
                    "Win::allocate local[{i}] = {x}, expected {expected}"
                );
            }
        }

        // Freeing a window over MPI-allocated memory returns no buffer.
        let buf = win.into_buffer().expect("Win::allocate into_buffer failed");
        assert!(buf.is_none(), "Win::allocate into_buffer must return None");
    }

    world.barrier().expect("barrier after test 2 failed");
    if rank == 0 {
        println!(
            "PASS: Win::allocate (f64, 32 elements, raw_handle, comm_size, local, into_buffer)"
        );
    }

    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All Win create/allocate tests passed! (2 tests)");
        println!("========================================");
    }
}
