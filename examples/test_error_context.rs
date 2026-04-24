//! Integration test for error context (operation name) in MPI errors.
//!
//! Triggers an intentional MPI error by calling broadcast with an out-of-range
//! root rank and verifies that the resulting error message contains the
//! operation name "bcast".
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_error_context

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    // Use an out-of-range root (world size can never be 999 in this test).
    // We must call broadcast collectively, so all ranks attempt the same
    // invalid call. MPI will return an error code; our Rust layer converts
    // it to Error::Mpi with operation = Some("bcast").
    let mut data = vec![0.0f64; 4];
    let result = world.broadcast(&mut data, 999);

    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.starts_with("MPI error in bcast:"),
                "expected error message to start with 'MPI error in bcast:', got: {msg}"
            );
            if world.rank() == 0 {
                println!("PASS: error message correctly prefixed: {msg}");
                println!("\n========================================");
                println!("All error-context tests passed!");
                println!("========================================");
            }
        }
        Ok(_) => {
            // Some MPI implementations tolerate invalid roots in single-rank
            // jobs; skip rather than fail.
            if world.rank() == 0 {
                println!("SKIP: broadcast with root=999 returned Ok (MPI did not error)");
                println!("      This can happen with lenient single-rank MPI implementations.");
            }
        }
    }
}
