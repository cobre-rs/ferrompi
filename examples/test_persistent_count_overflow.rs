//! Integration test for the `count > INT_MAX` guard in persistent `*_init` shims.
//!
//! Calls `ferrompi_allreduce_init` directly via a private `extern "C"` declaration
//! with a synthetic count of `i64::MAX`, bypassing the Rust-level type system so
//! that no real buffer allocation is required. Both ranks must observe
//! `MPI_ERR_COUNT` (mapped to `MpiErrorClass::Count`) and exit 0.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_persistent_count_overflow

use ferrompi::{Error, Mpi, MpiErrorClass};

mod common;

// Raw FFI declaration for the C-side shim under test.
//
// We declare it here rather than using ferrompi::ffi (which is pub(crate)) so
// that the example binary can call it directly without requiring any API
// extension on the Rust side.  This is the Option-A approach from the ticket:
// a direct-FFI call that exercises the C guard without allocating a real buffer.
//
// SAFETY invariants for the call below:
//   - sendbuf/recvbuf may be any non-null pointer when count is rejected before
//     the MPI call is made; we pass stack addresses of dummy bytes.
//   - datatype_tag 1 == DatatypeTag::F64 (MPI_DOUBLE); validated by the C shim.
//   - op 0 == ReduceOp::Sum; validated by the C shim.
//   - comm_handle is the world comm handle obtained from the MPI runtime.
//   - request_handle points to a valid i64 on the stack.
//   - count = i64::MAX triggers the guard and returns MPI_ERR_COUNT before any
//     MPI function is called, so no MPI state is modified.
#[allow(dead_code)]
extern "C" {
    fn ferrompi_allreduce_init(
        sendbuf: *const std::ffi::c_void,
        recvbuf: *mut std::ffi::c_void,
        count: i64,
        datatype_tag: i32,
        op: i32,
        comm_handle: i32,
        request_handle: *mut i64,
    ) -> std::ffi::c_int;
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size == 2,
        "test_persistent_count_overflow requires exactly 2 processes, got {size}"
    );

    // Persistent collectives are MPI 4.0+.  On older runtimes (e.g.,
    // OpenMPI 4.x reports MPI_VERSION = 3) the C-side shim is a stub
    // that always returns MPI_ERR_OTHER, so the count guard is never
    // reached.  Skip gracefully — the count guard is still in place
    // for MPI >= 4 runtimes (covered by MPICH 4.2.x in the CI matrix).
    if common::mpi_major() < 4 {
        let version = Mpi::version().expect("Mpi::version() failed");
        common::skip(
            &world,
            &format!("ferrompi_allreduce_init needs MPI 4 ({version})"),
        );
        return;
    }

    // ========================================================================
    // Test: ferrompi_allreduce_init with count > INT_MAX returns MPI_ERR_COUNT
    //
    // We use i64::MAX as the synthetic count. The C guard fires immediately
    // after the get_datatype/get_op lookups and before MPI_Allreduce_init is
    // called, so no real communication occurs and no real buffer is needed.
    // ========================================================================

    // Dummy send/recv buffers; the guard returns before touching them.
    let send_dummy: f64 = 0.0;
    let mut recv_dummy: f64 = 0.0;
    let mut request_handle: i64 = -1;

    // DATATYPE_TAG_F64 = 1 (matches DatatypeTag::F64 in src/datatype.rs)
    // OP_SUM          = 0 (matches ReduceOp::Sum in src/lib.rs)
    const DATATYPE_TAG_F64: i32 = 1;
    const OP_SUM: i32 = 0;

    let raw_ret = unsafe {
        // SAFETY: see the invariant comment on the extern "C" block above.
        ferrompi_allreduce_init(
            std::ptr::addr_of!(send_dummy).cast::<std::ffi::c_void>(),
            std::ptr::addr_of_mut!(recv_dummy).cast::<std::ffi::c_void>(),
            i64::MAX, // count >> INT_MAX — must be rejected by the C guard
            DATATYPE_TAG_F64,
            OP_SUM,
            world.raw_handle(),
            std::ptr::addr_of_mut!(request_handle),
        )
    };

    // raw_ret == 0 means MPI_SUCCESS — the guard failed to fire.
    // raw_ret != 0 — convert to ferrompi's Error to inspect the class without
    // depending on the numeric value of MPI_ERR_COUNT (varies by implementation).
    if raw_ret == 0 {
        eprintln!(
            "rank {rank}: FAIL: allreduce_init with count=i64::MAX returned MPI_SUCCESS — \
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
                println!("PASS: allreduce_init with count=i64::MAX returns MpiErrorClass::Count");
            }
        }
        other => {
            eprintln!(
                "rank {rank}: FAIL: allreduce_init with overflow count returned \
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
        println!("All persistent-count-overflow tests passed!");
        println!("========================================");
    }
}
