//! Regression test: gather, allgather and scatter's blocking, nonblocking
//! and persistent variants reject an undersized buffer before any FFI call.
//!
//! One call per shape x family (nine total), each with a buffer one element
//! short of what the size-rank communicator requires. Also runs correctly
//! under `mpiexec -n 1`.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_collective_validation
// mpi-test: np=2.. valgrind

use ferrompi::{Error, Mpi, PersistentRequest, Request, Result};

mod common;

fn check_unit(result: Result<()>, rank: i32, method: &str, local_ok: &mut bool) {
    if !matches!(result, Err(Error::InvalidBuffer)) {
        eprintln!("rank {rank}: {method}: expected Err(InvalidBuffer), got {result:?}");
        *local_ok = false;
    }
}

fn check_request(result: Result<Request>, rank: i32, method: &str, local_ok: &mut bool) {
    match result {
        Err(Error::InvalidBuffer) => {}
        Err(e) => {
            eprintln!("rank {rank}: {method}: expected Err(InvalidBuffer), got Err({e:?})");
            *local_ok = false;
        }
        Ok(req) => {
            req.wait().expect("wait on unexpectedly-accepted request");
            eprintln!("rank {rank}: {method}: expected Err(InvalidBuffer), got Ok(Request)");
            *local_ok = false;
        }
    }
}

fn check_persistent(
    result: Result<PersistentRequest>,
    rank: i32,
    method: &str,
    local_ok: &mut bool,
) {
    match result {
        Err(Error::InvalidBuffer) => {}
        Err(e) => {
            eprintln!("rank {rank}: {method}: expected Err(InvalidBuffer), got Err({e:?})");
            *local_ok = false;
        }
        Ok(req) => {
            drop(req);
            eprintln!(
                "rank {rank}: {method}: expected Err(InvalidBuffer), got Ok(PersistentRequest)"
            );
            *local_ok = false;
        }
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size() as usize;

    let mut local_ok = true;

    // gather / igather / gather_init: root only, recv one element short.
    if rank == 0 {
        let send = vec![1i32; 2];

        let mut backing = vec![0i32; 2 * size];
        let result = world.gather(&send, &mut backing[..2 * size - 1], 0);
        check_unit(result, rank, "gather", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result = world.igather(&send, &mut backing[..2 * size - 1], 0);
        check_request(result, rank, "igather", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result = world.gather_init(&send, &mut backing[..2 * size - 1], 0);
        check_persistent(result, rank, "gather_init", &mut local_ok);
    }

    // allgather / iallgather / allgather_init: every rank, recv one element short.
    {
        let send = vec![1i32; 2];

        let mut backing = vec![0i32; 2 * size];
        let result = world.allgather(&send, &mut backing[..2 * size - 1]);
        check_unit(result, rank, "allgather", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result = world.iallgather(&send, &mut backing[..2 * size - 1]);
        check_request(result, rank, "iallgather", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result = world.allgather_init(&send, &mut backing[..2 * size - 1]);
        check_persistent(result, rank, "allgather_init", &mut local_ok);
    }

    // scatter / iscatter / scatter_init: root only, send one element short.
    if rank == 0 {
        let mut recv = vec![0i32; 2];

        let send_backing = vec![1i32; 2 * size];
        let result = world.scatter(&send_backing[..2 * size - 1], &mut recv, 0);
        check_unit(result, rank, "scatter", &mut local_ok);

        let send_backing = vec![1i32; 2 * size];
        let result = world.iscatter(&send_backing[..2 * size - 1], &mut recv, 0);
        check_request(result, rank, "iscatter", &mut local_ok);

        let send_backing = vec![1i32; 2 * size];
        let result = world.scatter_init(&send_backing[..2 * size - 1], &mut recv, 0);
        check_persistent(result, rank, "scatter_init", &mut local_ok);
    }

    common::check(&world, local_ok, "test_collective_validation");
    if rank == 0 {
        println!("PASS: test_collective_validation");
    }
}
