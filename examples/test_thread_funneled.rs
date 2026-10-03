//! Regression for MPI calls issued from a non-init thread below `Serialized`.
//!
//! Converts a repro where a worker thread shared `&Communicator` under
//! `ThreadLevel::Single` and corrupted messages or hung. Below `Serialized`,
//! every guarded call from a non-init thread must instead return
//! `Err(Error::ThreadLevelViolation)` without calling MPI.
//!
//! It ends with `Mpi::drop` while a worker thread is inside a nonblocking scope
//! that holds no request: the drop must wait for that scope, then really
//! finalize MPI, which the runner checks by the absence of the
//! `MPI_Finalize skipped` warning.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_thread_funneled
// mpi-test: np=1

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use ferrompi::{CustomDatatype, DatatypeTag, Error, Info, Mpi, ReduceOp, ThreadLevel, UserOp};

mod common;

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Funneled).expect("MPI init failed");
    let world = mpi.world();

    common::check(
        &world,
        mpi.thread_level() == ThreadLevel::Funneled,
        "init_thread provided Funneled",
    );

    // Every guarded call below must be rejected before touching MPI: no MPI
    // object is created or dropped on this worker thread.
    let worker_ok = std::thread::scope(|s| {
        let world_ref = &world;
        s.spawn(move || {
            let mut ok = true;
            let mut rbuf = [0i32; 1];
            let sbuf = [1i32; 1];

            ok &= matches!(world_ref.barrier(), Err(Error::ThreadLevelViolation));
            ok &= matches!(
                world_ref.allreduce_scalar(1.0f64, ReduceOp::Sum),
                Err(Error::ThreadLevelViolation)
            );
            ok &= matches!(
                world_ref.send(&[1i32], 0, 1),
                Err(Error::ThreadLevelViolation)
            );
            ok &= matches!(
                ferrompi::scope(|ms| world_ref.irecv(ms, &mut rbuf, 0, 1).map(|_| ())),
                Err(Error::ThreadLevelViolation)
            );
            ok &= matches!(
                world_ref.iprobe::<i32>(0, 1),
                Err(Error::ThreadLevelViolation)
            );
            ok &= matches!(
                world_ref.send_init(&sbuf, 0, 1),
                Err(Error::ThreadLevelViolation)
            );
            ok &= matches!(world_ref.duplicate(), Err(Error::ThreadLevelViolation));
            ok &= matches!(world_ref.group(), Err(Error::ThreadLevelViolation));
            ok &= matches!(world_ref.processor_name(), Err(Error::ThreadLevelViolation));
            ok &= matches!(Info::new(), Err(Error::ThreadLevelViolation));
            ok &= matches!(
                CustomDatatype::contiguous(2, DatatypeTag::F64),
                Err(Error::ThreadLevelViolation)
            );
            // One more than the op table's 16 slots: a rejected create must
            // not leak a slot, or the last attempt would fail differently.
            const OP_SLOTS_PLUS_ONE: usize = 17;
            for _ in 0..OP_SLOTS_PLUS_ONE {
                let r = UserOp::<f64>::new(|a: &[f64], b: &mut [f64]| {
                    b[0] += a[0];
                });
                ok &= matches!(r, Err(Error::ThreadLevelViolation));
            }

            ok
        })
        .join()
        .expect("worker thread panicked")
    });
    common::check(&world, worker_ok, "worker calls below Serialized rejected");

    // A wait() rejected by the thread check must leave `active` unchanged,
    // so a later wait() on the init thread still owns a live request.
    let mut rbuf7 = [0i32; 1];
    let sbuf7 = [7i32; 1];
    let mut recv_req = world.recv_init(&mut rbuf7, 0, 7).expect("recv_init failed");
    let mut send_req = world.send_init(&sbuf7, 0, 7).expect("send_init failed");
    recv_req.start().expect("recv start failed");
    send_req.start().expect("send start failed");

    let (mut recv_req, wrong_thread_ok) = std::thread::spawn(move || {
        let result = recv_req.wait();
        let ok = matches!(result, Err(Error::ThreadLevelViolation)) && recv_req.is_active();
        (recv_req, ok)
    })
    .join()
    .expect("wait-worker thread panicked");
    common::check(
        &world,
        wrong_thread_ok,
        "wrong-thread PersistentRequest::wait rejected, active unchanged",
    );

    recv_req.wait().expect("recv wait failed");
    send_req.wait().expect("send wait failed");
    common::check(&world, rbuf7 == sbuf7, "recv/send data matches after wait");
    common::check(
        &world,
        !recv_req.is_active() && !send_req.is_active(),
        "requests inactive after init-thread wait",
    );

    // No check after the drop can use `common::check`: its allreduce is refused
    // once `Mpi` began to drop.
    let (opened, opened_rx) = mpsc::channel();
    let worker_scope_ended = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            ferrompi::scope(|_| {
                opened.send(()).expect("init thread hung up");
                std::thread::sleep(Duration::from_millis(50));
                worker_scope_ended.store(true, Ordering::SeqCst);
                Ok(())
            })
            .expect("the worker's scope must return Ok");
        });
        opened_rx
            .recv()
            .expect("worker hung up before opening its scope");
        drop(mpi);
        assert!(
            worker_scope_ended.load(Ordering::SeqCst),
            "Mpi::drop returned before the worker's scope ended"
        );
    });

    println!("test_thread_funneled: PASS");
}
