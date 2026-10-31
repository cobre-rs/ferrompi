//! A reproducible compensated-sum `UserOp` through the persistent reductions.
//!
//! The op is commutative and carries one compensated `f32` sum as a single
//! `u64` element, `(sum.to_bits() << 32) | comp.to_bits()`, so it stays pointwise
//! per element as MPI requires: two adjacent `f64` elements would break
//! under MPI's element-wise segmentation. The closure combines each pair
//! `(s1, c1)`, `(s2, c2)` with `TwoSum(s1, s2) -> (s, e)`, `c = c1 + c2 + e`,
//! then `FastTwoSum(s, c)`.
//!
//! Part 1: every handle to the op is dropped right after an `allreduce_init`,
//! so only the request keeps it alive, and the request runs 10 iterations.
//! Before each start, rank 0 writes `2^24`, the last rank `-2^24` and every
//! other rank `1.0`, each times `2^(iter % 4)`, as `(x, 0.0)`, four identical
//! elements. The exact total is `(size - 2)` times that scale. A plain `f32`
//! sum loses the ones next to `2^24`; here a power-of-two scale multiplies
//! every partial total exactly, so every partial total is held exactly by its
//! `(sum, comp)` pair, any reduction order MPI chooses gives the same bits as a
//! serial rank-ordered fold of the same combine, and `sum + comp` equals the
//! exact total bit for bit.
//!
//! Part 2: a plain wrapping-add `UserOp`, dropped right after the inits,
//! through `reduce_init` (with an empty `recv` at the non-root ranks),
//! `scan_init`, `exscan_init` and `reduce_scatter_block_init`, started together
//! with one `start_all` per iteration.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_user_op_persistent
// mpi-test: np=2.. valgrind skip-ok=openmpi-4

use ferrompi::{Mpi, PersistentRequest, ReduceOp, UserOp};

mod common;

fn pack(sum: f32, comp: f32) -> u64 {
    (u64::from(sum.to_bits()) << 32) | u64::from(comp.to_bits())
}

fn unpack(value: u64) -> (f32, f32) {
    (
        f32::from_bits((value >> 32) as u32),
        f32::from_bits(value as u32),
    )
}

fn combine(a: u64, b: u64) -> u64 {
    let (s1, c1) = unpack(a);
    let (s2, c2) = unpack(b);
    let s = s1 + s2;
    let bb = s - s1;
    let e = (s1 - (s - bb)) + (s2 - bb);
    let c = c1 + c2 + e;
    let hi = s + c;
    pack(hi, c - (hi - s))
}

fn contribution(rank: i32, size: i32, scale: f32) -> f32 {
    scale
        * if rank == 0 {
            16_777_216.0
        } else if rank == size - 1 {
            -16_777_216.0
        } else {
            1.0
        }
}

fn total(value: u64) -> f32 {
    let (sum, comp) = unpack(value);
    sum + comp
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_user_op_persistent requires at least 2 processes, got {size}"
    );

    if !common::has_mpi4_collectives() {
        common::skip(&world, "persistent collectives need MPI 4.0 or Open MPI 5");
        return;
    }

    // ========================================================================
    // Part 1: compensated sum through allreduce_init
    // ========================================================================
    let compensated: UserOp<u64> = UserOp::new(|invec: &[u64], inoutvec: &mut [u64]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = combine(*x, *y);
        }
    })
    .expect("UserOp::new failed");

    let mut sum_req = world
        .allreduce_init(vec![0u64; 4], vec![0u64; 4], &compensated)
        .expect("allreduce_init with the compensated-sum op failed");
    // The request keeps the op alive: no handle is left to the caller.
    drop(compensated);

    let mut plain_req = world
        .allreduce_init(vec![0.0f32; 4], vec![0.0f32; 4], ReduceOp::Sum)
        .expect("plain f32 allreduce_init failed");

    for iter in 0..10 {
        let scale = (1u32 << (iter % 4)) as f32;
        sum_req
            .send_buffer_mut()
            .expect("send_buffer_mut")
            .fill(pack(contribution(rank, size, scale), 0.0));
        plain_req
            .send_buffer_mut()
            .expect("send_buffer_mut")
            .fill(contribution(rank, size, scale));
        sum_req.start().expect("compensated start failed");
        plain_req.start().expect("plain start failed");
        sum_req.wait().expect("compensated wait failed");
        plain_req.wait().expect("plain wait failed");

        let serial = (1..size).fold(pack(contribution(0, size, scale), 0.0), |acc, r| {
            combine(acc, pack(contribution(r, size, scale), 0.0))
        });
        let exact = (size - 2) as f32 * scale;
        let recv = sum_req.buffer().expect("buffer");
        common::check(
            &world,
            recv.iter().all(|&value| value == serial),
            &format!("iteration {iter}: the compensated-sum allreduce_init equals the serial rank-ordered fold"),
        );
        common::check(
            &world,
            total(serial).to_bits() == exact.to_bits()
                && recv.iter().all(|&v| total(v).to_bits() == exact.to_bits()),
            &format!("iteration {iter}: sum + comp of the compensated-sum allreduce_init equals the exact total bit for bit"),
        );
        if rank == 0 {
            println!(
                "iteration {iter}: compensated sum {}, plain f32 Sum {}, exact {exact}",
                total(recv[0]),
                plain_req.buffer().expect("buffer")[0]
            );
        }
    }

    // ========================================================================
    // Part 2: a plain user op through the other four reductions
    // ========================================================================
    let add: UserOp<u64> = UserOp::new(|invec: &[u64], inoutvec: &mut [u64]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = y.wrapping_add(*x);
        }
    })
    .expect("UserOp::new failed");

    let ranks = size as usize;
    let mine = rank as u64 + 1;
    let reduce_recv = if rank == 0 { vec![0u64; 3] } else { Vec::new() };
    let mut requests = [
        world
            .reduce_init(vec![mine; 3], reduce_recv, &add, 0)
            .expect("reduce_init with a user op failed"),
        world
            .scan_init(vec![mine; 3], vec![0u64; 3], &add)
            .expect("scan_init with a user op failed"),
        world
            .exscan_init(vec![mine; 3], vec![0u64; 3], &add)
            .expect("exscan_init with a user op failed"),
        world
            .reduce_scatter_block_init(vec![mine; 3 * ranks], vec![0u64; 3], &add)
            .expect("reduce_scatter_block_init with a user op failed"),
    ];
    drop(add);

    let me = rank as u64;
    for iter in 0..3u64 {
        let factor = iter + 1;
        for request in requests.iter_mut() {
            request
                .send_buffer_mut()
                .expect("send_buffer_mut")
                .fill(mine * factor);
        }
        PersistentRequest::start_all(&mut requests).expect("start_all failed");
        PersistentRequest::wait_all(&mut requests).expect("wait_all failed");

        let all = (ranks * (ranks + 1) / 2) as u64 * factor;
        let [reduce, scan, exscan, scatter] = &requests;
        common::check(
            &world,
            rank != 0 || reduce.buffer().expect("buffer") == [all; 3],
            "reduce_init with a user op sums at the root",
        );
        common::check(
            &world,
            scan.buffer().expect("buffer") == [(me + 1) * (me + 2) / 2 * factor; 3],
            "scan_init with a user op is the inclusive prefix sum",
        );
        common::check(
            &world,
            rank == 0 || exscan.buffer().expect("buffer") == [me * (me + 1) / 2 * factor; 3],
            "exscan_init with a user op is the exclusive prefix sum",
        );
        common::check(
            &world,
            scatter.buffer().expect("buffer") == [all; 3],
            "reduce_scatter_block_init with a user op sums every block",
        );
    }

    if rank == 0 {
        println!("PASS: test_user_op_persistent");
    }
}
