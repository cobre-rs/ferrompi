//! Integration test for Request::wait_any, wait_some and test_some.
//!
//! Each rank posts 3 nonblocking receives and 3 nonblocking sends in a ring
//! pattern, then drives completion with wait_any (Part 1), wait_some
//! (Part 2), test_any (Part 3) and test_some (Part 4); Part 5 drives 70 of
//! each with wait_some, past the stack scratch. Asserts that the total number
//! of completions equals the number of posted requests, and that the statuses
//! of the some-calls report the ring source and no error.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_waitany
// mpi-test: np=4

use ferrompi::{Mpi, Request, Source, Status, Tag};
use std::time::Duration;

mod common;

/// True iff `status` is a completed ring request that did not fail: a receive
/// of `len` elements from `prev`, or a send, which reports the empty status.
fn is_ring_status(status: &Status, prev: i32, len: usize) -> bool {
    let recv = status.source == Source::Rank(prev) && status.count == Some(len);
    let send = status.source == Source::Any && status.tag == Tag::Any && status.count == Some(0);
    status.error.is_none() && (recv || send)
}

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

        let (completions, total_posted) = ferrompi::scope(|sc| {
            // Post all receives first, then all sends (standard deadlock-free ordering).
            let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
            for (i, buf) in recv_bufs.iter_mut().enumerate() {
                let req = world
                    .irecv(sc, buf, prev, 100 + i as i32)
                    .expect("irecv failed");
                requests.push(req);
            }
            for (i, buf) in send_bufs.iter().enumerate() {
                let req = world
                    .isend(sc, buf, next, 100 + i as i32)
                    .expect("isend failed");
                requests.push(req);
            }

            let total_posted = requests.len();
            let mut completions = 0usize;

            // Loop until the vec is empty: wait_any returns an index into the
            // current vec, then swap_remove removes that entry.
            while !requests.is_empty() {
                let idx = Request::wait_any(&mut requests)
                    .expect("wait_any failed")
                    .map(|(i, _)| i)
                    .expect("wait_any returned None on non-empty active request list");
                // Remove the completed request (swap-remove preserves compactness).
                requests.swap_remove(idx);
                completions += 1;
            }

            Ok((completions, total_posted))
        })
        .expect("scope failed");

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

        let (total_posted, all_completed_indices, statuses_ok) = ferrompi::scope(|sc| {
            let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
            for (i, buf) in recv_bufs2.iter_mut().enumerate() {
                let req = world
                    .irecv(sc, buf, prev, 200 + i as i32)
                    .expect("irecv Part 2 failed");
                requests.push(req);
            }
            for (i, buf) in send_bufs2.iter().enumerate() {
                let req = world
                    .isend(sc, buf, next, 200 + i as i32)
                    .expect("isend Part 2 failed");
                requests.push(req);
            }

            let total_posted = requests.len();
            let mut all_completed_indices: Vec<usize> = Vec::new();
            let mut statuses_ok = true;

            // Drive with wait_some; accumulate all returned indices, then remove.
            while !requests.is_empty() {
                let batch = Request::wait_some(&mut requests).expect("wait_some failed");
                // wait_some returning empty on a non-empty active list is an error.
                assert!(
                    !batch.is_empty(),
                    "rank {rank}: wait_some returned empty on non-empty active request list"
                );
                statuses_ok &= batch.iter().all(|(_, s)| is_ring_status(s, prev, 2));
                let batch: Vec<usize> = batch.iter().map(|&(i, _)| i).collect();
                // Sort descending so swap-removes do not invalidate earlier indices.
                let mut sorted = batch.clone();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                for idx in sorted {
                    requests.swap_remove(idx);
                }
                all_completed_indices.extend(batch);
            }

            Ok((total_posted, all_completed_indices, statuses_ok))
        })
        .expect("scope failed");

        assert_eq!(
            all_completed_indices.len(),
            total_posted,
            "rank {rank}: wait_some Part 2: expected {total_posted} completions, got {}",
            all_completed_indices.len()
        );
        common::check(
            &world,
            statuses_ok,
            "part 2: wait_some statuses report the ring source and no error",
        );

        if rank == 0 {
            println!(
                "waitany: rank {rank} completed {} requests via wait_some",
                all_completed_indices.len()
            );
        }
    }

    world.barrier().expect("barrier after Part 2 failed");

    // ========================================================================
    // Part 3: Post N irecvs + N isends, drive with test_any polling loop
    // ========================================================================
    {
        let mut recv_bufs3: Vec<Vec<f64>> = (0..N).map(|_| vec![0.0f64; 4]).collect();
        let send_bufs3: Vec<Vec<f64>> = (0..N)
            .map(|i| vec![(rank * 1000 + i as i32) as f64; 4])
            .collect();

        let completions = ferrompi::scope(|sc| {
            let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
            for (i, buf) in recv_bufs3.iter_mut().enumerate() {
                let req = world
                    .irecv(sc, buf, prev, 300 + i as i32)
                    .expect("irecv Part 3 failed");
                requests.push(req);
            }
            for (i, buf) in send_bufs3.iter().enumerate() {
                let req = world
                    .isend(sc, buf, next, 300 + i as i32)
                    .expect("isend Part 3 failed");
                requests.push(req);
            }

            let mut completions = 0usize;

            while !requests.is_empty() {
                match Request::test_any(&mut requests) {
                    Ok(Some((idx, _))) => {
                        requests.swap_remove(idx);
                        completions += 1;
                    }
                    Ok(None) => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => panic!("test_any failed: {e}"),
                }
            }

            Ok(completions)
        })
        .expect("scope failed");

        assert_eq!(
            completions,
            N * 2,
            "rank {rank}: test_any Part 3: expected {} completions, got {completions}",
            N * 2
        );

        // Verify received data from the previous rank.
        for (i, buf) in recv_bufs3.iter().enumerate() {
            let expected_val = (prev * 1000 + i as i32) as f64;
            for (j, &v) in buf.iter().enumerate() {
                assert!(
                    (v - expected_val).abs() < f64::EPSILON,
                    "rank {rank}: test_any Part 3: recv_bufs3[{i}][{j}] = {v}, expected {expected_val}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: test_any polling completed {completions} requests");
        }
    }

    world.barrier().expect("barrier after Part 3 failed");

    // ========================================================================
    // Part 4: Post N irecvs + N isends, drive with test_some polling loop
    // ========================================================================
    {
        let mut recv_bufs4: Vec<Vec<f64>> = (0..N).map(|_| vec![0.0f64; 4]).collect();
        let send_bufs4: Vec<Vec<f64>> = (0..N)
            .map(|i| vec![(rank * 1000 + i as i32) as f64; 4])
            .collect();

        let (completions, statuses_ok) = ferrompi::scope(|sc| {
            let mut requests: Vec<Request> = Vec::with_capacity(N * 2);
            for (i, buf) in recv_bufs4.iter_mut().enumerate() {
                let req = world
                    .irecv(sc, buf, prev, 400 + i as i32)
                    .expect("irecv Part 4 failed");
                requests.push(req);
            }
            for (i, buf) in send_bufs4.iter().enumerate() {
                let req = world
                    .isend(sc, buf, next, 400 + i as i32)
                    .expect("isend Part 4 failed");
                requests.push(req);
            }

            let mut completions = 0usize;
            let mut statuses_ok = true;

            while !requests.is_empty() {
                match Request::test_some(&mut requests) {
                    Ok(batch) if !batch.is_empty() => {
                        statuses_ok &= batch.iter().all(|(_, s)| is_ring_status(s, prev, 4));
                        let mut sorted: Vec<usize> = batch.iter().map(|&(i, _)| i).collect();
                        sorted.sort_unstable_by(|a, b| b.cmp(a));
                        completions += sorted.len();
                        for idx in sorted {
                            requests.swap_remove(idx);
                        }
                    }
                    Ok(_) => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => panic!("test_some failed: {e}"),
                }
            }

            Ok((completions, statuses_ok))
        })
        .expect("scope failed");

        assert_eq!(
            completions,
            N * 2,
            "rank {rank}: test_some Part 4: expected {} completions, got {completions}",
            N * 2
        );
        common::check(
            &world,
            statuses_ok,
            "part 4: test_some statuses report the ring source and no error",
        );

        // Verify received data from the previous rank.
        for (i, buf) in recv_bufs4.iter().enumerate() {
            let expected_val = (prev * 1000 + i as i32) as f64;
            for (j, &v) in buf.iter().enumerate() {
                assert!(
                    (v - expected_val).abs() < f64::EPSILON,
                    "rank {rank}: test_some Part 4: recv_bufs4[{i}][{j}] = {v}, expected {expected_val}"
                );
            }
        }

        if rank == 0 {
            println!("PASS: test_some polling completed {completions} requests");
        }
    }

    world.barrier().expect("barrier after Part 4 failed");

    // ========================================================================
    // Part 5: more requests than the stack scratch holds, drive with wait_some
    // ========================================================================
    {
        const BIG: usize = 70;
        let mut recv_bufs5: Vec<Vec<f64>> = (0..BIG).map(|_| vec![0.0f64; 1]).collect();
        let send_bufs5: Vec<Vec<f64>> = (0..BIG)
            .map(|i| vec![(rank * 1000 + i as i32) as f64; 1])
            .collect();

        let (completions, statuses_ok) = ferrompi::scope(|sc| {
            let mut requests: Vec<Request> = Vec::with_capacity(BIG * 2);
            for (i, buf) in recv_bufs5.iter_mut().enumerate() {
                let req = world
                    .irecv(sc, buf, prev, 500 + i as i32)
                    .expect("irecv Part 5 failed");
                requests.push(req);
            }
            for (i, buf) in send_bufs5.iter().enumerate() {
                let req = world
                    .isend(sc, buf, next, 500 + i as i32)
                    .expect("isend Part 5 failed");
                requests.push(req);
            }

            let mut completions = 0usize;
            let mut statuses_ok = true;

            while !requests.is_empty() {
                let batch = Request::wait_some(&mut requests).expect("wait_some Part 5 failed");
                assert!(
                    !batch.is_empty(),
                    "rank {rank}: wait_some Part 5 returned empty on non-empty active request list"
                );
                statuses_ok &= batch.iter().all(|(_, s)| is_ring_status(s, prev, 1));
                let mut sorted: Vec<usize> = batch.iter().map(|&(i, _)| i).collect();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                completions += sorted.len();
                for idx in sorted {
                    requests.swap_remove(idx);
                }
            }

            Ok((completions, statuses_ok))
        })
        .expect("scope failed");

        assert_eq!(
            completions,
            BIG * 2,
            "rank {rank}: wait_some Part 5: expected {} completions, got {completions}",
            BIG * 2
        );
        common::check(
            &world,
            statuses_ok,
            "part 5: wait_some over the stack scratch reports the ring source and no error",
        );

        for (i, buf) in recv_bufs5.iter().enumerate() {
            let expected_val = (prev * 1000 + i as i32) as f64;
            assert!(
                (buf[0] - expected_val).abs() < f64::EPSILON,
                "rank {rank}: wait_some Part 5: recv_bufs5[{i}] = {}, expected {expected_val}",
                buf[0]
            );
        }

        if rank == 0 {
            println!("PASS: wait_some over the stack scratch completed {completions} requests");
        }
    }

    world.barrier().expect("barrier after Part 5 failed");

    if rank == 0 {
        println!("\n========================================");
        println!("All wait_any / wait_some tests passed!");
        println!("========================================");
    }
}
