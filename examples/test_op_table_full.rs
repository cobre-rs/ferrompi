//! Regression for `UserOp::new` when the op-slot table is full.
//!
//! Creates the 16 concurrent ops the table holds, checks that a seventeenth
//! returns the table-full error, and checks that dropping one op frees a slot
//! for a new one.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_op_table_full
// mpi-test: np=1.. valgrind

use ferrompi::{Error, Mpi, ResourceKind, UserOp};

mod common;

const OP_SLOTS: usize = 16;

fn add(invec: &[f64], inoutvec: &mut [f64]) {
    inoutvec[0] += invec[0];
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();

    let mut ops = Vec::with_capacity(OP_SLOTS);
    for _ in 0..OP_SLOTS {
        let op = UserOp::<f64>::new(add);
        common::check(&world, op.is_ok(), "UserOp::new below the table limit");
        ops.push(op.expect("checked above"));
    }

    let overflow = UserOp::<f64>::new(add);
    common::check(
        &world,
        matches!(
            overflow,
            Err(Error::ResourceExhausted {
                resource: ResourceKind::Operation,
                ..
            })
        ),
        "UserOp::new table-full error",
    );

    drop(ops.pop());
    common::check(
        &world,
        UserOp::<f64>::new(add).is_ok(),
        "UserOp::new after dropping one op",
    );

    drop(ops);
    drop(mpi);

    println!("test_op_table_full: PASS");
}
