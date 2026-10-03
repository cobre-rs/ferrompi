//! A reproducible compensated-sum `UserOp` through the nonblocking reductions.
//!
//! The op is commutative and carries one compensated `f32` sum as a single
//! `u64` element, `(sum.to_bits() << 32) | comp.to_bits()`, so it stays pointwise
//! per element as MPI requires: two adjacent `f64` elements would break
//! under MPI's element-wise segmentation. The closure combines each pair
//! `(s1, c1)`, `(s2, c2)` with `TwoSum(s1, s2) -> (s, e)`, `c = c1 + c2 + e`,
//! then `FastTwoSum(s, c)`.
//!
//! Part 1: rank 0 contributes `2^24`, the last rank `-2^24`, and every other
//! rank `1.0`, as `(x, 0.0)`, four identical elements each. The exact total is
//! `size - 2`. A plain `f32` sum loses the ones next to `2^24`; here every
//! partial total is held exactly by its `(sum, comp)` pair, so any reduction
//! order MPI chooses gives the same bits as a serial rank-ordered fold of the
//! same combine, and `sum + comp` equals the exact total bit for bit.
//!
//! Part 2: a plain wrapping-add `UserOp` through `ireduce` (with an empty
//! `recv` at the non-root ranks), `iscan`, `iexscan` and
//! `ireduce_scatter_block`, all pending in one scope.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_user_op_nonblocking
// mpi-test: np=2.. valgrind

use ferrompi::{Mpi, ReduceOp, Request, UserOp};

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

fn contribution(rank: i32, size: i32) -> f32 {
    if rank == 0 {
        16_777_216.0
    } else if rank == size - 1 {
        -16_777_216.0
    } else {
        1.0
    }
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    assert!(
        size >= 2,
        "test_user_op_nonblocking requires at least 2 processes, got {size}"
    );

    // ========================================================================
    // Part 1: compensated sum through iallreduce
    // ========================================================================
    let compensated: UserOp<u64> = UserOp::new(|invec: &[u64], inoutvec: &mut [u64]| {
        for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
            *y = combine(*x, *y);
        }
    })
    .expect("UserOp::new failed");

    let send = [pack(contribution(rank, size), 0.0); 4];
    let mut recv = [0u64; 4];
    ferrompi::scope(|s| {
        world
            .iallreduce(s, &send, &mut recv, &compensated)?
            .wait()?;
        Ok(())
    })
    .expect("iallreduce with the compensated-sum op failed");

    let serial = (1..size).fold(pack(contribution(0, size), 0.0), |acc, r| {
        combine(acc, pack(contribution(r, size), 0.0))
    });
    let exact = (size - 2) as f32;
    let total = |value: u64| {
        let (sum, comp) = unpack(value);
        sum + comp
    };
    common::check(
        &world,
        recv.iter().all(|&value| value == serial),
        "iallreduce with the compensated-sum op equals the serial rank-ordered fold",
    );
    common::check(
        &world,
        total(serial).to_bits() == exact.to_bits()
            && recv.iter().all(|&v| total(v).to_bits() == exact.to_bits()),
        "sum + comp of the compensated-sum iallreduce equals the exact total bit for bit",
    );

    let plain_send = [contribution(rank, size); 4];
    let mut plain_recv = [0.0f32; 4];
    world
        .allreduce(&plain_send, &mut plain_recv, ReduceOp::Sum)
        .expect("plain f32 allreduce failed");
    if rank == 0 {
        println!(
            "compensated sum {}, plain f32 Sum {}, exact {exact}",
            total(recv[0]),
            plain_recv[0]
        );
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
    let mine = [rank as u64 + 1; 3];
    let block_send = vec![rank as u64 + 1; 3 * ranks];
    let mut reduced = if rank == 0 { vec![0u64; 3] } else { Vec::new() };
    let mut scanned = [0u64; 3];
    let mut exscanned = [0u64; 3];
    let mut scattered = [0u64; 3];
    ferrompi::scope(|s| {
        let reduce = world.ireduce(s, &mine, &mut reduced, &add, 0)?;
        let scan = world.iscan(s, &mine, &mut scanned, &add)?;
        let exscan = world.iexscan(s, &mine, &mut exscanned, &add)?;
        let scatter = world.ireduce_scatter_block(s, &block_send, &mut scattered, &add)?;
        Request::wait_all(&mut [reduce, scan, exscan, scatter])?;
        Ok(())
    })
    .expect("nonblocking reductions with a user op failed");

    let total = (ranks * (ranks + 1) / 2) as u64;
    let me = rank as u64;
    common::check(
        &world,
        rank != 0 || reduced == [total; 3],
        "ireduce with a user op sums at the root",
    );
    common::check(
        &world,
        scanned == [(me + 1) * (me + 2) / 2; 3],
        "iscan with a user op is the inclusive prefix sum",
    );
    common::check(
        &world,
        rank == 0 || exscanned == [me * (me + 1) / 2; 3],
        "iexscan with a user op is the exclusive prefix sum",
    );
    common::check(
        &world,
        scattered == [total; 3],
        "ireduce_scatter_block with a user op sums every block",
    );

    if rank == 0 {
        println!("PASS: test_user_op_nonblocking");
    }
}
