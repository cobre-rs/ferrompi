//! A forgotten window keeps its buffer: a peer's put lands in the leaked
//! buffer, never in a later allocation that reuses its address.
//!
//! Rank 0 forgets its `Win::create` window and then allocates a victim `Vec`
//! of the same size, while rank 1 puts into the window under an exclusive lock
//! and reads the data back with a shared one. The window owns its buffer, so
//! forgetting it never frees the memory MPI still serves (MPI-4.1 §12.2.5); a
//! borrowed buffer that the caller could drop while the window lived would
//! hand the put to the victim. The victim is checked unchanged, and valgrind
//! sees no write outside an allocation.
//!
//! Both windows stay alive, so both ranks skip `MPI_Finalize`. Rank 1 does not
//! free its window, because the free is collective and rank 0 never calls it:
//! it drops `Mpi` first, and the window it drops afterwards makes no MPI call.
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_rma_forget_window
// mpi-test: np=2 expect=unfinalized valgrind skip-ok=openmpi-4
// mpi-test-stderr: MPI_Finalize skipped: 1 window(s) still alive

use ferrompi::{Error, LockType, Mpi, MpiErrorClass, Win};

mod common;

const VICTIM: u64 = 0xAAAA_AAAA_AAAA_AAAA;
const SENT: u64 = 0x5555_5555_5555_5555;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    assert_eq!(
        world.size(),
        2,
        "test_rma_forget_window requires exactly 2 processes"
    );

    // Open MPI 4 refuses `Win::create` over caller memory.
    let win = match Win::create(&world, vec![0u64; 32]) {
        Ok(win) => win,
        Err(Error::Mpi {
            class: MpiErrorClass::Win,
            ..
        }) => {
            common::skip(
                &world,
                "Win::create returned MPI_ERR_WIN — likely OpenMPI 4.x \
                 with a BTL that does not support one-sided over caller-owned \
                 memory (e.g., --btl=self,tcp in CI).",
            );
            return;
        }
        Err(e) => panic!("Win::create failed: {e}"),
    };

    let data = [SENT; 4];
    let mut back = [0u64; 4];
    let mut victim = Vec::new();
    let mut put_ok = true;
    let mut kept = None;
    if rank == 0 {
        std::mem::forget(win);
        victim = vec![VICTIM; 32];
        world.barrier().expect("barrier 1");
        world.barrier().expect("barrier 2");
    } else {
        let mut win = win;
        world.barrier().expect("barrier 1");
        put_ok = win
            .lock(LockType::Exclusive, 0, |e| e.put(&data, 0, 0))
            .is_ok()
            && win
                .lock(LockType::Shared, 0, |e| e.get(&mut back, 0, 0))
                .is_ok();
        world.barrier().expect("barrier 2");
        kept = Some(win);
    }

    common::check(&world, put_ok, "the put and the read-back returned Ok");
    common::check(
        &world,
        rank == 0 || back == data,
        "the put landed in the forgotten window's buffer",
    );
    common::check(
        &world,
        victim.iter().all(|&x| x == VICTIM),
        "the put never reached an allocation that reused the buffer",
    );

    drop(mpi);
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    drop(kept);
    if rank == 0 {
        println!("test_rma_forget_window: PASS");
    }
}
