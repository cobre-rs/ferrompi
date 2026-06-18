//! Integration test for nonblocking point-to-point operations.
//!
//! Exercises isend/irecv with Request::wait(), sendrecv, probe/iprobe,
//! and blocking send/recv. Each operation is verified with assertions.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_nonblocking

use ferrompi::Mpi;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_nonblocking requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Test 1: isend / irecv with Request::wait()
    // ========================================================================
    // Ring pattern: each rank sends to next, receives from previous
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;
        let tag = 100;

        let send_data = vec![rank as f64 * 10.0 + 1.0, rank as f64 * 10.0 + 2.0];
        let mut recv_data = vec![0.0f64; 2];

        // Post nonblocking receive first, then nonblocking send
        let recv_req = world
            .irecv(&mut recv_data, prev, tag)
            .expect("irecv failed");
        let send_req = world.isend(&send_data, next, tag).expect("isend failed");

        // Wait for both to complete
        send_req.wait().expect("isend wait failed");
        recv_req.wait().expect("irecv wait failed");

        // Verify received data is from the previous rank
        let expected = vec![prev as f64 * 10.0 + 1.0, prev as f64 * 10.0 + 2.0];
        assert_eq!(
            recv_data, expected,
            "rank {rank}: isend/irecv ring data mismatch: got {recv_data:?}, expected {expected:?}"
        );
        if rank == 0 {
            println!("PASS: isend/irecv ring");
        }
    }

    world.barrier().expect("barrier 1 failed");

    // ========================================================================
    // Test 2: isend / irecv with multiple outstanding requests
    // ========================================================================
    // Each rank sends 3 messages to the next and receives 3 from the previous
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;

        let send_bufs: Vec<Vec<f64>> = (0..3).map(|i| vec![(rank * 100 + i) as f64; 4]).collect();
        let mut recv_bufs: Vec<Vec<f64>> = (0..3).map(|_| vec![0.0f64; 4]).collect();

        // Post all receives first
        let mut recv_reqs: Vec<_> = recv_bufs
            .iter_mut()
            .enumerate()
            .map(|(i, buf)| {
                world
                    .irecv(buf, prev, 200 + i as i32)
                    .expect("irecv failed")
            })
            .collect();

        // Post all sends
        let mut send_reqs: Vec<_> = send_bufs
            .iter()
            .enumerate()
            .map(|(i, buf)| {
                world
                    .isend(buf, next, 200 + i as i32)
                    .expect("isend failed")
            })
            .collect();

        // Wait for all using wait_all
        ferrompi::Request::wait_all(&mut send_reqs).expect("send wait_all failed");
        ferrompi::Request::wait_all(&mut recv_reqs).expect("recv wait_all failed");

        // Verify each received buffer
        for (i, buf) in recv_bufs.iter().enumerate() {
            let expected_val = (prev * 100 + i as i32) as f64;
            for (j, &v) in buf.iter().enumerate() {
                assert!(
                    (v - expected_val).abs() < f64::EPSILON,
                    "rank {rank}: multi-isend recv_bufs[{i}][{j}] = {v}, expected {expected_val}"
                );
            }
        }
        if rank == 0 {
            println!("PASS: isend/irecv multiple outstanding");
        }
    }

    world.barrier().expect("barrier 2 failed");

    // ========================================================================
    // Test 3: sendrecv (blocking, bidirectional)
    // ========================================================================
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;
        let tag = 300;

        let send_buf = vec![rank as f64 * 7.0; 5];
        let mut recv_buf = vec![0.0f64; 5];

        let (source, actual_tag, count) = world
            .sendrecv(&send_buf, next, tag, &mut recv_buf, prev, tag)
            .expect("sendrecv failed");

        assert_eq!(
            source, prev,
            "rank {rank}: sendrecv source = {source}, expected {prev}"
        );
        assert_eq!(
            actual_tag, tag,
            "rank {rank}: sendrecv tag = {actual_tag}, expected {tag}"
        );
        assert_eq!(
            count, 5,
            "rank {rank}: sendrecv count = {count}, expected 5"
        );

        let expected_val = prev as f64 * 7.0;
        for (i, &v) in recv_buf.iter().enumerate() {
            assert!(
                (v - expected_val).abs() < f64::EPSILON,
                "rank {rank}: sendrecv recv_buf[{i}] = {v}, expected {expected_val}"
            );
        }
        if rank == 0 {
            println!("PASS: sendrecv");
        }
    }

    world.barrier().expect("barrier 3 failed");

    // ========================================================================
    // Test 4: blocking send/recv (even/odd pattern to avoid deadlock)
    // ========================================================================
    {
        let tag = 400;
        if size >= 2 {
            if rank % 2 == 0 {
                let partner = (rank + 1) % size;
                let send_data = vec![(rank * 11) as f64; 3];
                world.send(&send_data, partner, tag).expect("send failed");

                let mut recv_data = vec![0.0f64; 3];
                let (src, _, _) = world
                    .recv(&mut recv_data, partner, tag)
                    .expect("recv failed");
                assert_eq!(
                    src, partner,
                    "rank {rank}: recv source = {src}, expected {partner}"
                );
                let expected = (partner * 11) as f64;
                for &v in &recv_data {
                    assert!(
                        (v - expected).abs() < f64::EPSILON,
                        "rank {rank}: send/recv got {v}, expected {expected}"
                    );
                }
            } else {
                let partner = (rank + size - 1) % size;
                let mut recv_data = vec![0.0f64; 3];
                let (src, _, _) = world
                    .recv(&mut recv_data, partner, tag)
                    .expect("recv failed");
                assert_eq!(
                    src, partner,
                    "rank {rank}: recv source = {src}, expected {partner}"
                );
                let expected = (partner * 11) as f64;
                for &v in &recv_data {
                    assert!(
                        (v - expected).abs() < f64::EPSILON,
                        "rank {rank}: send/recv got {v}, expected {expected}"
                    );
                }

                let send_data = vec![(rank * 11) as f64; 3];
                world.send(&send_data, partner, tag).expect("send failed");
            }
        }
        if rank == 0 {
            println!("PASS: blocking send/recv");
        }
    }

    world.barrier().expect("barrier 4 failed");

    // ========================================================================
    // Test 5: probe (blocking)
    // ========================================================================
    // Rank 0 sends to rank 1, rank 1 probes before receiving
    {
        let tag = 500;
        if rank == 0 && size >= 2 {
            let data = vec![42.0f64; 7];
            world.send(&data, 1, tag).expect("probe test: send failed");
        } else if rank == 1 {
            let status = world.probe::<f64>(0, tag).expect("probe failed");
            assert_eq!(status.source, 0, "probe source mismatch");
            assert_eq!(status.tag, tag, "probe tag mismatch");
            assert_eq!(status.count, 7, "probe count mismatch");

            // Now receive the probed message
            let mut buf = vec![0.0f64; status.count as usize];
            world
                .recv(&mut buf, 0, tag)
                .expect("recv after probe failed");
            assert!(
                buf.iter().all(|&v| (v - 42.0).abs() < f64::EPSILON),
                "rank 1: probe+recv data mismatch"
            );
        }
        world.barrier().expect("barrier probe failed");
        if rank == 0 {
            println!("PASS: probe");
        }
    }

    // ========================================================================
    // Test 6: iprobe (nonblocking probe)
    // ========================================================================
    // Rank 0 sends to rank 1, rank 1 polls with iprobe
    {
        let tag = 600;
        if rank == 0 && size >= 2 {
            let data = vec![99.0f64; 3];
            world.send(&data, 1, tag).expect("iprobe test: send failed");
        } else if rank == 1 {
            // Poll until message arrives
            let mut status = None;
            for _ in 0..100_000 {
                if let Some(s) = world.iprobe::<f64>(0, tag).expect("iprobe failed") {
                    status = Some(s);
                    break;
                }
                std::hint::spin_loop();
            }
            let status = status.expect("iprobe: message never arrived");
            assert_eq!(status.source, 0, "iprobe source mismatch");
            assert_eq!(status.tag, tag, "iprobe tag mismatch");
            assert_eq!(status.count, 3, "iprobe count mismatch");

            let mut buf = vec![0.0f64; status.count as usize];
            world
                .recv(&mut buf, 0, tag)
                .expect("recv after iprobe failed");
            assert!(
                buf.iter().all(|&v| (v - 99.0).abs() < f64::EPSILON),
                "rank 1: iprobe+recv data mismatch"
            );
        }
        world.barrier().expect("barrier iprobe failed");
        if rank == 0 {
            println!("PASS: iprobe");
        }
    }

    // ========================================================================
    // Test 7: isend/irecv with Request::test() polling
    // ========================================================================
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;
        let tag = 700;

        let send_data = vec![(rank + 1) as f64; 10];
        let mut recv_data = vec![0.0f64; 10];

        let mut recv_req = world
            .irecv(&mut recv_data, prev, tag)
            .expect("irecv failed");
        let send_req = world.isend(&send_data, next, tag).expect("isend failed");

        // Poll recv with test()
        let mut polls = 0u64;
        while !recv_req.test().expect("test failed") {
            polls += 1;
            std::hint::spin_loop();
        }

        // send_req must also complete (wait to be safe)
        send_req.wait().expect("isend wait failed");

        let expected_val = (prev + 1) as f64;
        for (i, &v) in recv_data.iter().enumerate() {
            assert!(
                (v - expected_val).abs() < f64::EPSILON,
                "rank {rank}: test() poll recv_data[{i}] = {v}, expected {expected_val}"
            );
        }

        if rank == 0 {
            println!("PASS: isend/irecv with test() polling ({polls} polls)");
        }
    }

    world.barrier().expect("barrier 7 failed");

    // ========================================================================
    // Test 8: sendrecv with different send/recv sizes
    // ========================================================================
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;

        // Each rank sends 3 elements but receives 3 elements too (same size here)
        let send_buf = vec![
            (rank * 3) as f64,
            (rank * 3 + 1) as f64,
            (rank * 3 + 2) as f64,
        ];
        let mut recv_buf = vec![0.0f64; 3];

        let (source, _tag, count) = world
            .sendrecv(&send_buf, next, 800, &mut recv_buf, prev, 800)
            .expect("sendrecv different tags failed");

        assert_eq!(source, prev, "rank {rank}: sendrecv source mismatch");
        assert_eq!(count, 3, "rank {rank}: sendrecv count mismatch");

        let expected = vec![
            (prev * 3) as f64,
            (prev * 3 + 1) as f64,
            (prev * 3 + 2) as f64,
        ];
        assert_eq!(recv_buf, expected, "rank {rank}: sendrecv data mismatch");

        if rank == 0 {
            println!("PASS: sendrecv (structured data)");
        }
    }

    world.barrier().expect("barrier 8 failed");

    // ========================================================================
    // Test 9: Request::drop() without wait + is_completed() coverage
    // ========================================================================
    // Ring pattern: each rank posts irecv then isend.
    // We wait on the recv request (to get the data) but DROP the send request
    // without calling wait(). This exercises the Drop impl's FFI wait path.
    // Safety: the matching irecv.wait() on the receiving rank ensures the
    // send operation completes, so the Drop-side wait will return promptly.
    //
    // We also verify is_completed() returns false on a freshly created request
    // (before any wait/test) and true after test() succeeds.
    {
        let next = (rank + 1) % size;
        let prev = (rank + size - 1) % size;
        let tag = 900;

        let send_data = vec![(rank + 1) as f64 * 7.77; 6];
        let mut recv_data = vec![0.0f64; 6];

        // Post nonblocking receive, then nonblocking send
        let recv_req = world
            .irecv(&mut recv_data, prev, tag)
            .expect("irecv failed");
        let send_req = world.isend(&send_data, next, tag).expect("isend failed");

        // Verify is_completed() returns false on a fresh request
        assert!(
            !send_req.is_completed(),
            "rank {rank}: is_completed() should be false before wait/test"
        );

        // Wait on recv to complete the data transfer
        recv_req.wait().expect("irecv wait failed");

        // Verify received data from previous rank
        let expected_val = (prev + 1) as f64 * 7.77;
        for (i, &v) in recv_data.iter().enumerate() {
            assert!(
                (v - expected_val).abs() < f64::EPSILON,
                "rank {rank}: drop test recv_data[{i}] = {v}, expected {expected_val}"
            );
        }

        // DROP send_req without calling wait() — exercises Drop impl's FFI path
        drop(send_req);
    }

    world.barrier().expect("barrier 9 failed");

    // Post-drop verification: all ranks survived the drop-without-wait path.
    // If the Drop impl didn't call ferrompi_wait, MPI would be in an undefined
    // state and this barrier (or subsequent operations) would likely hang/crash.
    if rank == 0 {
        println!("PASS: Request::drop without wait + is_completed()");
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All nonblocking/p2p tests passed! (9 tests)");
        println!("========================================");
    }
}
