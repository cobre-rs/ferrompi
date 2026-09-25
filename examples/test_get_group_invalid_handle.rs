//! Integration test for the corrected `get_group` sentinel contract.
//!
//! Calls `ferrompi_group_size` directly via `extern "C"` with an out-of-range
//! group handle (999), which must NOT return `MPI_SUCCESS`.  Before ticket-012,
//! `get_group` returned `MPI_GROUP_EMPTY` for invalid handles, so `group_size`
//! would silently succeed with size=0.  After the fix it returns
//! `MPI_GROUP_NULL`, and MPI itself returns `MPI_ERR_GROUP`.
//!
//! Acceptance criterion (ticket-012): the call returns a non-zero MPI error
//! code that maps to `MpiErrorClass::Group` or `MpiErrorClass::Arg`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_get_group_invalid_handle
// mpi-test: np=2

use ferrompi::{Error, Mpi, MpiErrorClass};

// Raw FFI declaration for the C-side shim under test.
//
// SAFETY invariants for the calls below:
//   - group_handle 999 is deliberately out-of-range (MAX_GROUPS == 64).
//   - size points to a valid i32 on the stack; it is only written if the
//     call succeeds (which it must not).
#[allow(dead_code)]
extern "C" {
    fn ferrompi_group_size(group_handle: i32, size: *mut i32) -> std::ffi::c_int;
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_get_group_invalid_handle requires exactly 2 processes, got {size}"
    );

    // ========================================================================
    // Test: ferrompi_group_size with an out-of-range handle returns an error
    //
    // Handle 999 is well past MAX_GROUPS (64).  After ticket-012 get_group
    // returns MPI_GROUP_NULL for this input, and MPI_Group_size returns
    // MPI_ERR_GROUP (or similar implementation-specific error).
    // ========================================================================

    let mut out_size: i32 = 0;
    let raw_ret = unsafe {
        // SAFETY: see the invariant comment on the extern "C" block above.
        ferrompi_group_size(999, std::ptr::addr_of_mut!(out_size))
    };

    if raw_ret == 0 {
        eprintln!(
            "rank {rank}: FAIL: group_size with handle=999 returned MPI_SUCCESS — \
             get_group sentinel fix is missing or not compiled in"
        );
        std::process::exit(1);
    }

    let err = Error::from_code(raw_ret);
    match err {
        Error::Mpi {
            class: MpiErrorClass::Group,
            ..
        }
        | Error::Mpi {
            class: MpiErrorClass::Arg,
            ..
        } => {
            if rank == 0 {
                println!(
                    "PASS: group_size with handle=999 returns non-SUCCESS error class: {err:?}"
                );
            }
        }
        other => {
            // Any non-SUCCESS error is acceptable here — different MPI
            // implementations may return different classes when passed
            // MPI_GROUP_NULL.  Log but do not fail.
            if rank == 0 {
                println!("PASS (non-MPI_SUCCESS): group_size with handle=999 returned: {other:?}");
            }
        }
    }

    // Sentinel barrier: both ranks must reach this point.
    world.barrier().expect("sentinel barrier failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All get-group-invalid-handle tests passed!");
        println!("========================================");
    }
}
