//! Regression test: gather, allgather and scatter's blocking, nonblocking
//! and persistent variants reject an undersized buffer before any FFI call;
//! gatherv, scatterv, allgatherv and alltoallv's blocking, nonblocking and
//! persistent variants reject a malformed counts/displacements pair before
//! any FFI call.
//!
//! Fixed-count: one call per shape x family (nine total), each with a buffer
//! one element short of what the size-rank communicator requires.
//!
//! Variable-count: one call per shape x family (twelve total): gatherv gets a
//! recv buffer one element short; scatterv gets sendcounts/displs one entry
//! short (empty at np=1); allgatherv gets a negative leading displacement
//! paired with a positive count; alltoallv gets a negative leading
//! recvcounts entry. Also runs correctly under `mpiexec -n 1`.
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

    // gatherv / igatherv / gatherv_init: root only, recv one element short of what
    // well-formed recvcounts/displs require.
    if rank == 0 {
        let send = vec![1i32; 2];
        let recvcounts = vec![2i32; size];
        let displs: Vec<i32> = (0..size as i32).map(|i| i * 2).collect();

        let mut backing = vec![0i32; 2 * size];
        let result = world.gatherv(&send, &mut backing[..2 * size - 1], &recvcounts, &displs, 0);
        check_unit(result, rank, "gatherv", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result = world.igatherv(&send, &mut backing[..2 * size - 1], &recvcounts, &displs, 0);
        check_request(result, rank, "igatherv", &mut local_ok);

        let mut backing = vec![0i32; 2 * size];
        let result =
            world.gatherv_init(&send, &mut backing[..2 * size - 1], &recvcounts, &displs, 0);
        check_persistent(result, rank, "gatherv_init", &mut local_ok);
    }

    // scatterv / iscatterv / scatterv_init: root only, sendcounts/displs one entry
    // short of the communicator size (empty at np=1).
    if rank == 0 {
        let send = vec![1i32; 2 * size];
        let sendcounts_full = vec![2i32; size];
        let displs_full: Vec<i32> = (0..size as i32).map(|i| i * 2).collect();
        let mut recv = vec![0i32; 2];

        let result = world.scatterv(
            &send,
            &sendcounts_full[..size - 1],
            &displs_full[..size - 1],
            &mut recv,
            0,
        );
        check_unit(result, rank, "scatterv", &mut local_ok);

        let result = world.iscatterv(
            &send,
            &mut recv,
            &sendcounts_full[..size - 1],
            &displs_full[..size - 1],
            0,
        );
        check_request(result, rank, "iscatterv", &mut local_ok);

        let result = world.scatterv_init(
            &send,
            &sendcounts_full[..size - 1],
            &displs_full[..size - 1],
            &mut recv,
            0,
        );
        check_persistent(result, rank, "scatterv_init", &mut local_ok);
    }

    // allgatherv / iallgatherv / allgatherv_init: every rank, displs[0] negative
    // paired with a positive count.
    {
        let send = vec![1i32; 1];
        let recvcounts = vec![1i32; size];
        let mut displs: Vec<i32> = (0..size as i32).collect();
        displs[0] = -1;

        let mut backing = vec![0i32; size + 1];
        let result = world.allgatherv(&send, &mut backing[1..], &recvcounts, &displs);
        check_unit(result, rank, "allgatherv", &mut local_ok);

        let mut backing = vec![0i32; size + 1];
        let result = world.iallgatherv(&send, &mut backing[1..], &recvcounts, &displs);
        check_request(result, rank, "iallgatherv", &mut local_ok);

        let mut backing = vec![0i32; size + 1];
        let result = world.allgatherv_init(&send, &mut backing[1..], &recvcounts, &displs);
        check_persistent(result, rank, "allgatherv_init", &mut local_ok);
    }

    // alltoallv / ialltoallv / alltoallv_init: every rank, recvcounts[0] negative.
    {
        let send = vec![1i32; size];
        let sendcounts = vec![1i32; size];
        let sdispls: Vec<i32> = (0..size as i32).collect();
        let mut recvcounts = vec![1i32; size];
        recvcounts[0] = -1;
        let rdispls: Vec<i32> = (0..size as i32).collect();
        let mut recv = vec![0i32; size];

        let result = world.alltoallv(
            &send,
            &sendcounts,
            &sdispls,
            &mut recv,
            &recvcounts,
            &rdispls,
        );
        check_unit(result, rank, "alltoallv", &mut local_ok);

        let result = world.ialltoallv(
            &send,
            &mut recv,
            &sendcounts,
            &sdispls,
            &recvcounts,
            &rdispls,
        );
        check_request(result, rank, "ialltoallv", &mut local_ok);

        let result = world.alltoallv_init(
            &send,
            &sendcounts,
            &sdispls,
            &mut recv,
            &recvcounts,
            &rdispls,
        );
        check_persistent(result, rank, "alltoallv_init", &mut local_ok);
    }

    common::check(&world, local_ok, "test_collective_validation");
    if rank == 0 {
        println!("PASS: test_collective_validation");
    }
}
