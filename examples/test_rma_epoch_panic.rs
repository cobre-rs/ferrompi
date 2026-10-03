//! Regression for a panic inside an open fence epoch: the closing fence is
//! collective, so one rank closing it alone could hang or mismatch the job, and
//! `Win::fence` aborts the process instead of unwinding past it.
//!
//! The panic is caught around the `fence` call, so control returns there only if
//! the abort failed to happen.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_rma_epoch_panic
// mpi-test: np=1 expect=abort
// mpi-test-stderr: a fence or PSCW epoch was still open during unwind

use std::panic::{catch_unwind, AssertUnwindSafe};

use ferrompi::{Mpi, Win, WinFenceAssert};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let mut win = Win::<f64>::allocate(&world, 1).expect("Win::allocate");

    let unwound = catch_unwind(AssertUnwindSafe(|| {
        win.fence(
            WinFenceAssert::none(),
            WinFenceAssert::none(),
            |_| -> ferrompi::Result<()> { panic!("a panic inside an open fence epoch") },
        )
    }));
    println!("FAIL: test_rma_epoch_panic: the panic unwound past fence: {unwound:?}");
    std::process::exit(1);
}
