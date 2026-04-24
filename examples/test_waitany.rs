//! Integration test for Request::wait_any and Request::wait_some.
//!
//! Each rank posts 3 nonblocking receives and 3 nonblocking sends in a ring
//! pattern, then drives completion with wait_any (Part 1) and wait_some
//! (Part 2). Asserts that the total number of completions equals the number
//! of posted requests.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_waitany

use ferrompi::{Mpi, Request};

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_waitany requires at least 2 processes, got {size}"
    );

    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    const N: usize = 3;

    // ========================================================================
    // Part 1: Post N irecvs + N isends, drive with wait_any
    // ========================================================================
    {
        // Allocate receive buffers — must outlive the requests.
        let mut recv_bufs: Vec<Vec<f64>> = (0..N).map(|_| vec![0.0f64; 4]).collect();
        let send_bufs: Vec<Vec<f64>> = (0..N)
            .map(|i| vec![(rank * 100 + i as i32) as f64; 4])
            .collect();

        // Post all receives first, then all sends (standard deadlock-free ordering).
        let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
        for (i, buf) in recv_bufs.iter_mut().enumerate() {
            let req = world
                .irecv(buf, prev, 100 + i as i32)
                .expect("irecv failed");
            requests.push(req);
        }
        for (i, buf) in send_bufs.iter().enumerate() {
            let req = world
                .isend(buf, next, 100 + i as i32)
                .expect("isend failed");
            requests.push(req);
        }

        let total_posted = requests.len();
        let mut completions = 0usize;

        // Drive completions with wait_any; remove completed entries by swap-remove
        // so the vector shrinks naturally. We track original indices by keeping a
        // parallel index map so that wait_any indices stay valid after removals.
        //
        // Simpler approach matching the ticket spec: loop until the vec is empty,
        // using swap-remove on each returned index.
        while !requests.is_empty() {
            let idx = Request::wait_any(&mut requests)
                .expect("wait_any failed")
                .expect("wait_any returned None on non-empty active request list");
            // Remove the completed request (swap-remove preserves compactness).
            requests.swap_remove(idx);
            completions += 1;
        }

        assert_eq!(
            completions, total_posted,
            "rank {rank}: wait_any Part 1: expected {total_posted} completions, got {completions}"
        );

        // Verify received data from the previous rank.
        for (i, buf) in recv_bufs.iter().enumerate() {
            let expected_val = (prev * 100 + i as i32) as f64;
            for (j, &v) in buf.iter().enumerate() {
                assert!(
                    (v - expected_val).abs() < f64::EPSILON,
                    "rank {rank}: wait_any Part 1: recv_bufs[{i}][{j}] = {v}, expected {expected_val}"
                );
            }
        }

        println!("waitany: rank {rank} completed {completions} requests");
    }

    world.barrier().expect("barrier after Part 1 failed");

    // ========================================================================
    // Part 2: Post N isends, drive with wait_some, assert completions count
    // ========================================================================
    {
        // We only post sends here; the previous rank's Part 1 receives are gone, so
        // we pair with fresh receives on the next rank. Use a different tag range to
        // avoid message matching confusion with Part 1 traffic.
        //
        // To keep the test self-contained, post N receives AND N sends again.
        let mut recv_bufs2: Vec<Vec<f64>> = (0..N).map(|_| vec![0.0f64; 2]).collect();
        let send_bufs2: Vec<Vec<f64>> = (0..N)
            .map(|i| vec![(rank * 10 + i as i32) as f64; 2])
            .collect();

        let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
        for (i, buf) in recv_bufs2.iter_mut().enumerate() {
            let req = world
                .irecv(buf, prev, 200 + i as i32)
                .expect("irecv Part 2 failed");
            requests.push(req);
        }
        for (i, buf) in send_bufs2.iter().enumerate() {
            let req = world
                .isend(buf, next, 200 + i as i32)
                .expect("isend Part 2 failed");
            requests.push(req);
        }

        let total_posted = requests.len();
        let mut all_completed_indices: Vec<usize> = Vec::new();

        // Drive with wait_some; accumulate all returned indices, then remove.
        while !requests.is_empty() {
            let batch = Request::wait_some(&mut requests).expect("wait_some failed");
            // wait_some returning empty on a non-empty active list is an error.
            assert!(
                !batch.is_empty(),
                "rank {rank}: wait_some returned empty on non-empty active request list"
            );
            // Sort descending so swap-removes do not invalidate earlier indices.
            let mut sorted = batch.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            for idx in sorted {
                requests.swap_remove(idx);
            }
            all_completed_indices.extend(batch);
        }

        assert_eq!(
            all_completed_indices.len(),
            total_posted,
            "rank {rank}: wait_some Part 2: expected {total_posted} completions, got {}",
            all_completed_indices.len()
        );

        if rank == 0 {
            println!(
                "waitany: rank {rank} completed {} requests via wait_some",
                all_completed_indices.len()
            );
        }
    }

    world.barrier().expect("barrier after Part 2 failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All wait_any / wait_some tests passed!");
        println!("========================================");
    }
}
