//! Integration test for the `count > INT_MAX` guard in `ferrompi_waitall`.
//!
//! Calls `ferrompi_waitall` directly via a private `extern "C"` declaration
//! with a synthetic count of `i64::MAX`, bypassing the Rust-level type system
//! so that no real buffer allocation is required.  Both ranks must observe
//! `MPI_ERR_COUNT` (mapped to `MpiErrorClass::Count`) and exit 0.
//!
//! The overflow guard must fire before any `malloc` call, returning
//! `MPI_ERR_COUNT` without allocating memory.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_waitall_count_overflow
// mpi-test: np=2

use ferrompi::{Error, Mpi, MpiErrorClass};

// Raw FFI declaration for the C-side shim under test.
//
// We declare it here rather than using ferrompi::ffi (which is pub(crate)) so
// that the example binary can call it directly without requiring any API
// extension on the Rust side.
//
// SAFETY invariants for the call below:
//   - request_handles may point to a stack i64 when count is rejected before
//     the loop body; the guard returns before dereferencing the array.
//   - count = i64::MAX triggers the guard and returns MPI_ERR_COUNT before any
//     MPI function or malloc is called, so no MPI state is modified.
#[allow(dead_code)]
extern "C" {
    fn ferrompi_waitall(count: i64, request_handles: *mut i64) -> std::ffi::c_int;
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_waitall_count_overflow requires exactly 2 processes, got {size}"
    );

    // ========================================================================
    // Test: ferrompi_waitall with count > INT_MAX returns MPI_ERR_COUNT
    //
    // We use i64::MAX as the synthetic count. The C guard fires immediately
    // after the `count <= 0` check and before any malloc, so no real
    // request handle is needed.
    // ========================================================================

    // A stack i64 as a dummy request_handles pointer; the guard returns before
    // the loop body ever dereferences it.
    let mut dummy_handle: i64 = -1;

    let raw_ret = unsafe {
        // SAFETY: see the invariant comment on the extern "C" block above.
        ferrompi_waitall(
            i64::MAX, // count >> INT_MAX — must be rejected by the C guard
            std::ptr::addr_of_mut!(dummy_handle),
        )
    };

    // raw_ret == 0 means MPI_SUCCESS — the guard failed to fire.
    if raw_ret == 0 {
        eprintln!(
            "rank {rank}: FAIL: waitall with count=i64::MAX returned MPI_SUCCESS — \
             the C-side overflow guard is missing or not compiled in"
        );
        std::process::exit(1);
    }

    let err = Error::from_code(raw_ret);
    match err {
        Error::Mpi {
            class: MpiErrorClass::Count,
            ..
        } => {
            if rank == 0 {
                println!("PASS: waitall with count=i64::MAX returns MpiErrorClass::Count");
            }
        }
        other => {
            eprintln!(
                "rank {rank}: FAIL: waitall with overflow count returned \
                 unexpected error class: {other:?}"
            );
            std::process::exit(1);
        }
    }

    // Sentinel barrier: both ranks must reach this point, confirming neither
    // rank panicked or exited early.
    world.barrier().expect("sentinel barrier failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All waitall-count-overflow tests passed!");
        println!("========================================");
    }
}
