//! Regression for calling MPI after the `Mpi` handle was dropped.
//!
//! Converts a repro that crashed the process inside the MPI runtime when a
//! collective ran after finalize. Every call below must instead return
//! `Err(Error::Finalized)` without calling MPI.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_after_finalize
// mpi-test: np=1

use ferrompi::{CustomDatatype, DatatypeTag, Error, Info, Mpi, ReduceOp, UserOp};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let dup = world.duplicate().expect("duplicate before finalize failed");
    let op = UserOp::<f64>::new(|a: &[f64], b: &mut [f64]| {
        b[0] += a[0];
    })
    .expect("UserOp::new before finalize failed");
    drop(mpi);

    assert!(
        matches!(world.barrier(), Err(Error::Finalized)),
        "barrier after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(
            world.allreduce_scalar(1.0f64, ReduceOp::Sum),
            Err(Error::Finalized)
        ),
        "allreduce_scalar after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(world.send(&[1i32], 0, 0), Err(Error::Finalized)),
        "send after finalize must return Err(Finalized)"
    );
    let mut buf = [0i32; 1];
    assert!(
        matches!(world.irecv(&mut buf, 0, 0), Err(Error::Finalized)),
        "irecv after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(world.send_init(&buf, 0, 0), Err(Error::Finalized)),
        "send_init after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(world.duplicate(), Err(Error::Finalized)),
        "duplicate after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(world.group(), Err(Error::Finalized)),
        "group after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(world.processor_name(), Err(Error::Finalized)),
        "processor_name after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(dup.barrier(), Err(Error::Finalized)),
        "barrier on a pre-finalize communicator must return Err(Finalized)"
    );
    assert!(
        matches!(Info::new(), Err(Error::Finalized)),
        "Info::new after finalize must return Err(Finalized)"
    );
    assert!(
        matches!(
            CustomDatatype::contiguous(2, DatatypeTag::F64),
            Err(Error::Finalized)
        ),
        "CustomDatatype::contiguous after finalize must return Err(Finalized)"
    );

    // More than the 16 op-table slots: each call's create fails at the
    // guarded ferrompi_op_create_user, and the rollback must release the
    // slot it allocated. If the rollback leaked slots, the table would be
    // exhausted well before the 17th call and this would observe
    // Err(ResourceExhausted) instead of Err(Finalized).
    for i in 0..17 {
        let r = UserOp::<f64>::new(|a: &[f64], b: &mut [f64]| {
            b[0] += a[0];
        });
        assert!(
            matches!(r, Err(Error::Finalized)),
            "UserOp::new after finalize must return Err(Finalized) on call {i}"
        );
    }

    // Both handles outlived `Mpi`; their `Drop` must be a silent no-op
    // rather than calling MPI after finalize.
    drop(op);
    drop(dup);

    println!("test_after_finalize: PASS");
}
