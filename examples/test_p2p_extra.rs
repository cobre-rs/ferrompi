//! Multi-type point-to-point tests for coverage.
//!
//! Exercises send/recv, sendrecv, isend/irecv with multiple data types
//! (i32, f32, u8, u32, u64, i64) to cover generic monomorphizations.
//! Also tests probe and iprobe with an additional type (i32) beyond the
//! f64 coverage in test_nonblocking.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_p2p_extra

use ferrompi::Mpi;

/// Helper trait for test values — provides a canonical test value
/// derived from an MPI rank, so we can verify data integrity.
trait TestValue: Copy + PartialEq + std::fmt::Debug {
    fn from_rank(rank: i32) -> Self;
    fn from_rank_indexed(rank: i32, index: usize) -> Self;
    fn type_name() -> &'static str;
}

impl TestValue for i32 {
    fn from_rank(rank: i32) -> Self {
        rank * 11 + 3
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        rank * 100 + index as i32
    }
    fn type_name() -> &'static str {
        "i32"
    }
}

impl TestValue for f32 {
    fn from_rank(rank: i32) -> Self {
        rank as f32 * 7.5 + 1.0
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        rank as f32 * 100.0 + index as f32
    }
    fn type_name() -> &'static str {
        "f32"
    }
}

impl TestValue for u8 {
    fn from_rank(rank: i32) -> Self {
        (rank * 13 + 5) as u8
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        ((rank * 10 + index as i32) % 256) as u8
    }
    fn type_name() -> &'static str {
        "u8"
    }
}

impl TestValue for u32 {
    fn from_rank(rank: i32) -> Self {
        (rank * 17 + 7) as u32
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        (rank * 1000 + index as i32) as u32
    }
    fn type_name() -> &'static str {
        "u32"
    }
}

impl TestValue for u64 {
    fn from_rank(rank: i32) -> Self {
        (rank as u64) * 1_000_001 + 42
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        (rank as u64) * 10_000 + index as u64
    }
    fn type_name() -> &'static str {
        "u64"
    }
}

impl TestValue for i64 {
    fn from_rank(rank: i32) -> Self {
        (rank as i64) * -500_003 + 99
    }
    fn from_rank_indexed(rank: i32, index: usize) -> Self {
        (rank as i64) * 10_000 + index as i64
    }
    fn type_name() -> &'static str {
        "i64"
    }
}

/// Verify that each element in `recv_data` matches the expected value
/// from `from_rank_indexed(partner, i)`.
fn verify_data<T: TestValue>(recv_data: &[T], partner: i32, rank: i32, op_name: &str) {
    for (i, val) in recv_data.iter().enumerate() {
        assert_eq!(
            *val,
            T::from_rank_indexed(partner, i),
            "{}: {op_name} data mismatch at index {i} on rank {rank}",
            T::type_name()
        );
    }
}

/// Test 1: Blocking send/recv with even/odd pairing to avoid deadlock.
///
/// Even ranks send first then receive; odd ranks receive first then send.
/// Uses a partner of (rank ± 1) so rank 0 pairs with rank 1, etc.
fn test_send_recv<T: ferrompi::MpiDatatype + TestValue>(
    world: &ferrompi::Communicator,
    rank: i32,
    size: i32,
    tag_base: i32,
) {
    let tag = tag_base;
    let buf_len = 4;

    let partner = if rank % 2 == 0 { rank + 1 } else { rank - 1 };
    if partner >= size {
        return;
    }

    if rank % 2 == 0 {
        // Even rank: send first, then receive
        let send_data: Vec<T> = (0..buf_len)
            .map(|i| T::from_rank_indexed(rank, i))
            .collect();
        world
            .send(&send_data, partner, tag)
            .expect("send failed in send/recv test");

        let mut recv_data = vec![T::from_rank(0); buf_len];
        let (src, actual_tag, count) = world
            .recv(&mut recv_data, partner, tag)
            .expect("recv failed in send/recv test");

        assert_eq!(
            src,
            partner,
            "{}: send/recv source mismatch: got {src}, expected {partner}",
            T::type_name()
        );
        assert_eq!(
            actual_tag,
            tag,
            "{}: send/recv tag mismatch",
            T::type_name()
        );
        assert_eq!(
            count,
            buf_len as i64,
            "{}: send/recv count mismatch",
            T::type_name()
        );
        verify_data(&recv_data, partner, rank, "send/recv");
    } else {
        // Odd rank: receive first, then send
        let mut recv_data = vec![T::from_rank(0); buf_len];
        let (src, actual_tag, count) = world
            .recv(&mut recv_data, partner, tag)
            .expect("recv failed in send/recv test");

        assert_eq!(
            src,
            partner,
            "{}: send/recv source mismatch: got {src}, expected {partner}",
            T::type_name()
        );
        assert_eq!(
            actual_tag,
            tag,
            "{}: send/recv tag mismatch",
            T::type_name()
        );
        assert_eq!(
            count,
            buf_len as i64,
            "{}: send/recv count mismatch",
            T::type_name()
        );
        verify_data(&recv_data, partner, rank, "send/recv");

        let send_data: Vec<T> = (0..buf_len)
            .map(|i| T::from_rank_indexed(rank, i))
            .collect();
        world
            .send(&send_data, partner, tag)
            .expect("send failed in send/recv test");
    }
}

/// Test 2: Bidirectional sendrecv in a ring pattern.
///
/// Each rank sends to next and receives from previous simultaneously.
fn test_sendrecv<T: ferrompi::MpiDatatype + TestValue>(
    world: &ferrompi::Communicator,
    rank: i32,
    size: i32,
    tag_base: i32,
) {
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    let tag = tag_base;
    let buf_len = 5;

    let send_buf: Vec<T> = (0..buf_len)
        .map(|i| T::from_rank_indexed(rank, i))
        .collect();
    let mut recv_buf = vec![T::from_rank(0); buf_len];

    let (source, actual_tag, count) = world
        .sendrecv(&send_buf, next, tag, &mut recv_buf, prev, tag)
        .expect("sendrecv failed");

    assert_eq!(
        source,
        prev,
        "{}: sendrecv source mismatch: got {source}, expected {prev}",
        T::type_name()
    );
    assert_eq!(actual_tag, tag, "{}: sendrecv tag mismatch", T::type_name());
    assert_eq!(
        count,
        buf_len as i64,
        "{}: sendrecv count mismatch",
        T::type_name()
    );
    verify_data(&recv_buf, prev, rank, "sendrecv");
}

/// Test 3: Nonblocking isend/irecv with wait, ring pattern.
///
/// Each rank posts irecv from previous, then isend to next, then waits both.
fn test_isend_irecv<T: ferrompi::MpiDatatype + TestValue>(
    world: &ferrompi::Communicator,
    rank: i32,
    size: i32,
    tag_base: i32,
) {
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    let tag = tag_base;
    let buf_len = 6;

    let send_data: Vec<T> = (0..buf_len)
        .map(|i| T::from_rank_indexed(rank, i))
        .collect();
    let mut recv_data = vec![T::from_rank(0); buf_len];

    // Post nonblocking receive first, then nonblocking send
    let recv_req = world
        .irecv(&mut recv_data, prev, tag)
        .expect("irecv failed");
    let send_req = world.isend(&send_data, next, tag).expect("isend failed");

    // Wait for both to complete
    send_req.wait().expect("isend wait failed");
    recv_req.wait().expect("irecv wait failed");

    // Verify received data is from the previous rank
    verify_data(&recv_data, prev, rank, "isend/irecv");
}

/// Run all three P2P tests for a given type, with barriers between them.
/// Tags are offset by `tag_offset` to avoid collisions between types.
fn run_type_tests<T: ferrompi::MpiDatatype + TestValue>(
    world: &ferrompi::Communicator,
    rank: i32,
    size: i32,
    tag_offset: i32,
    test_counter: &mut i32,
) {
    // Test: blocking send/recv
    test_send_recv::<T>(world, rank, size, tag_offset + 10);
    world.barrier().expect("barrier after send/recv failed");
    *test_counter += 1;
    if rank == 0 {
        println!("PASS: send/recv <{}>", T::type_name());
    }

    // Test: sendrecv
    test_sendrecv::<T>(world, rank, size, tag_offset + 20);
    world.barrier().expect("barrier after sendrecv failed");
    *test_counter += 1;
    if rank == 0 {
        println!("PASS: sendrecv <{}>", T::type_name());
    }

    // Test: isend/irecv
    test_isend_irecv::<T>(world, rank, size, tag_offset + 30);
    world.barrier().expect("barrier after isend/irecv failed");
    *test_counter += 1;
    if rank == 0 {
        println!("PASS: isend/irecv <{}>", T::type_name());
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_p2p_extra requires at least 2 processes, got {size}"
    );

    let mut test_count: i32 = 0;

    // ========================================================================
    // Multi-type P2P tests
    // ========================================================================
    // Each type gets a unique tag_offset range (100 apart) to avoid collisions.

    // === i32 === (tag_offset 1000)
    run_type_tests::<i32>(&world, rank, size, 1000, &mut test_count);

    // === f32 === (tag_offset 1100)
    run_type_tests::<f32>(&world, rank, size, 1100, &mut test_count);

    // === u8 === (tag_offset 1200)
    run_type_tests::<u8>(&world, rank, size, 1200, &mut test_count);

    // === u32 === (tag_offset 1300)
    run_type_tests::<u32>(&world, rank, size, 1300, &mut test_count);

    // === u64 === (tag_offset 1400)
    run_type_tests::<u64>(&world, rank, size, 1400, &mut test_count);

    // === i64 === (tag_offset 1500)
    run_type_tests::<i64>(&world, rank, size, 1500, &mut test_count);

    // ========================================================================
    // Test: probe::<i32> (blocking)
    // ========================================================================
    // Rank 0 sends i32 data to rank 1, rank 1 probes before receiving.
    {
        let tag = 2000;
        if rank == 0 && size >= 2 {
            let data = vec![42i32; 7];
            world.send(&data, 1, tag).expect("probe test: send failed");
        } else if rank == 1 {
            let status = world.probe::<i32>(0, tag).expect("probe::<i32> failed");
            assert_eq!(status.source, 0, "probe::<i32> source mismatch");
            assert_eq!(status.tag, tag, "probe::<i32> tag mismatch");
            assert_eq!(status.count, 7, "probe::<i32> count mismatch");

            // Receive the probed message
            let mut buf = vec![0i32; status.count as usize];
            world
                .recv(&mut buf, 0, tag)
                .expect("recv after probe::<i32> failed");
            assert!(
                buf.iter().all(|&v| v == 42),
                "probe::<i32> + recv data mismatch"
            );
        }
        world.barrier().expect("barrier after probe::<i32> failed");
        test_count += 1;
        if rank == 0 {
            println!("PASS: probe::<i32>");
        }
    }

    // ========================================================================
    // Test: iprobe::<i32> (nonblocking probe)
    // ========================================================================
    // Rank 0 sends i32 data to rank 1, rank 1 polls with iprobe.
    {
        let tag = 2100;
        if rank == 0 && size >= 2 {
            let data = vec![99i32; 3];
            world.send(&data, 1, tag).expect("iprobe test: send failed");
        } else if rank == 1 {
            // Poll until message arrives
            let mut status = None;
            for _ in 0..100_000 {
                if let Some(s) = world.iprobe::<i32>(0, tag).expect("iprobe::<i32> failed") {
                    status = Some(s);
                    break;
                }
                std::hint::spin_loop();
            }
            let status = status.expect("iprobe::<i32>: message never arrived");
            assert_eq!(status.source, 0, "iprobe::<i32> source mismatch");
            assert_eq!(status.tag, tag, "iprobe::<i32> tag mismatch");
            assert_eq!(status.count, 3, "iprobe::<i32> count mismatch");

            let mut buf = vec![0i32; status.count as usize];
            world
                .recv(&mut buf, 0, tag)
                .expect("recv after iprobe::<i32> failed");
            assert!(
                buf.iter().all(|&v| v == 99),
                "iprobe::<i32> + recv data mismatch"
            );
        }
        world.barrier().expect("barrier after iprobe::<i32> failed");
        test_count += 1;
        if rank == 0 {
            println!("PASS: iprobe::<i32>");
        }
    }

    // ========================================================================
    // Final barrier and summary
    // ========================================================================
    world.barrier().expect("final barrier failed");
    if rank == 0 {
        println!("\n========================================");
        println!("All multi-type P2P tests passed! ({test_count} tests)");
        println!("========================================");
    }
}
