//! The MPI communication pattern of an SDDP solver.
//!
//! One long-lived context holds the MPI state. The program then broadcasts
//! run parameters, exchanges variable-length serialized cuts, reduces bounds
//! and convergence flags, and reads scenario data from node-shared memory. The
//! example has no LP solve and no I/O; it shows the communication only.
//!
//! Build: `cargo build --features rma --example sddp_pattern`
//! Run:   `mpiexec -n 4 ./target/debug/examples/sddp_pattern`

use ferrompi::{Communicator, Mpi, ReduceOp, ThreadLevel, Win};

/// Scenario values each rank stores in the node-shared window.
const COUNT: usize = 6;
/// Discount rate, convergence tolerance, iteration limit, forward-pass scenarios.
const RUN_PARAMS: [f64; 4] = [0.95, 1e-4, 100.0, 20.0];

/// A solver keeps one boxed context for the whole run. Handles carry no borrow
/// of `Mpi`, so the context can own all of them.
///
/// Fields drop in declaration order, so the window and the communicators are
/// freed before `mpi` finalizes MPI. `mpi` must stay the last field.
struct Ctx {
    win: Win<f64>,
    node: Communicator,
    world: Communicator,
    mpi: Mpi,
}

/// A fixed function of `(rank, k)`: terms of 2^60 that cancel, among terms near
/// 1. The magnitudes span more than 2^53, so the order of a sum changes its bits.
fn value(rank: i32, k: usize) -> f64 {
    match (rank as usize + k) % 3 {
        0 => 1.0 + 0.25 * f64::from(rank) + 0.125 * k as f64,
        1 => 2f64.powi(60),
        _ => -(2f64.powi(60)),
    }
}

/// The serialized cuts of `rank`: `rank + 1` values, little-endian.
fn cut_bytes(rank: i32) -> Vec<u8> {
    (0..=rank as usize)
        .flat_map(|k| value(rank, k).to_le_bytes())
        .collect()
}

/// Neumaier's compensated sum, in the order of `xs`.
fn neumaier(xs: &[f64]) -> f64 {
    let (mut sum, mut err) = (0.0, 0.0);
    for &x in xs {
        let t = sum + x;
        err += if sum.abs() >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
    }
    sum + err
}

// A boxed error lets the integer conversions below propagate with `?` under
// their own types; every ferrompi call propagates the same way.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start-up: only the main thread calls MPI, which is `Funneled`.
    let mpi = Mpi::init_thread(ThreadLevel::Funneled)?;
    if mpi.thread_level() < ThreadLevel::Funneled {
        println!(
            "sddp_pattern: MPI provided {:?}, Funneled is needed",
            mpi.thread_level()
        );
        return Ok(());
    }
    let world = mpi.world();
    // The processes of one node share memory.
    let node = world.split_shared()?;
    let win = Win::<f64>::allocate_shared(&node, COUNT)?;
    let mut ctx: Box<Ctx> = Box::new(Ctx {
        win,
        node,
        world,
        mpi,
    });
    let rank = ctx.world.rank();
    let size = ctx.world.size();

    // The job's layout: how many processes run on which hosts.
    let topo = ctx.world.topology(&ctx.mpi)?;
    assert_eq!(topo.size(), size);
    assert_eq!(
        topo.hosts().iter().map(|h| h.ranks.len()).sum::<usize>(),
        size as usize
    );
    if rank == 0 {
        println!("{topo}");
    }

    // Run parameters: rank 0 holds them, every rank receives them.
    let mut params = if rank == 0 { RUN_PARAMS } else { [0.0; 4] };
    ctx.world.broadcast(&mut params, 0)?;
    assert_eq!(params, RUN_PARAMS);

    // The forward pass shares its trial points and the backward pass its cuts
    // as serialized bytes of a different length on each rank. The byte counts
    // go first, then the bytes, packed in rank order.
    let mine = cut_bytes(rank);
    let mut counts = vec![0i32; size as usize];
    ctx.world
        .allgather(&[i32::try_from(mine.len())?], &mut counts)?;
    let mut displs = Vec::with_capacity(counts.len());
    let mut total = 0i32;
    for &count in &counts {
        displs.push(total);
        total = total
            .checked_add(count)
            .ok_or("the exchanged cuts exceed i32::MAX bytes")?;
    }
    let mut recv = vec![0u8; usize::try_from(total)?];
    ctx.world.allgatherv(&mine, &mut recv, &counts, &displs)?;
    // Every rank regenerates what all ranks sent, so it can check what it got.
    let expected: Vec<u8> = (0..size).flat_map(cut_bytes).collect();
    assert_eq!(recv, expected);
    let cuts = recv
        .chunks_exact(8)
        .map(|b| b.try_into().map(f64::from_le_bytes))
        .collect::<Result<Vec<f64>, _>>()?;

    // A bound summed by `allreduce` is summed in MPI's order, so its last bits
    // may change between runs. The gathered values are in rank order on every
    // rank: summing them there with a compensated sum is bit-identical on every
    // rank and to a serial sum.
    let oracle: Vec<f64> = (0..size)
        .flat_map(|r| (0..=r as usize).map(move |k| value(r, k)))
        .collect();
    let bound = neumaier(&cuts).to_bits();
    assert_eq!(bound, neumaier(&oracle).to_bits());
    assert_eq!(
        ctx.world.allreduce_scalar(bound, ReduceOp::Min)?,
        ctx.world.allreduce_scalar(bound, ReduceOp::Max)?
    );

    // Statistics of the forward pass: the sum, the lowest and the highest cost.
    let cost = f64::from(rank + 1);
    let n = f64::from(size);
    assert_eq!(
        ctx.world.allreduce_scalar(cost, ReduceOp::Sum)?,
        n * (n + 1.0) / 2.0
    );
    assert_eq!(ctx.world.allreduce_scalar(cost, ReduceOp::Min)?, 1.0);
    assert_eq!(ctx.world.allreduce_scalar(cost, ReduceOp::Max)?, n);

    // Convergence test: each rank sets bit `rank % 64`, so with up to 64
    // ranks the OR is the set of ranks that have converged.
    let flags = ctx
        .world
        .allreduce_scalar(1u64 << (rank % 64), ReduceOp::BitwiseOr)?;
    assert_eq!(
        flags,
        if size >= 64 {
            u64::MAX
        } else {
            (1u64 << size) - 1
        }
    );

    // Scenario data: every rank of the node writes its part once, then every
    // rank reads every part. Store, `sync`, barrier, `sync`, then load.
    let node_rank = ctx.node.rank();
    let node_size = ctx.node.size();
    ctx.win.lock_all(|e| {
        let part: Vec<f64> = (0..COUNT).map(|k| value(node_rank, k)).collect();
        e.local().copy_from_slice(&part);
        e.sync()?;
        ctx.node.barrier()?;
        e.sync()?;
        for r in 0..node_size {
            let want: Vec<f64> = (0..COUNT).map(|k| value(r, k)).collect();
            let view = e.remote(r)?;
            let mut copy = [0.0; COUNT];
            view.copy_to_slice(&mut copy);
            assert_eq!(copy[..], want[..]);
            // SAFETY: no process writes this region while the slice lives: all
            // stores came before the barrier, and the loop only loads. The
            // writer's `MPI_Win_sync` after its stores, the barrier, and this
            // rank's `MPI_Win_sync` after the barrier published them here.
            let seen = unsafe { view.as_slice() };
            assert_eq!(seen, &want[..]);
        }
        Ok(())
    })?;

    ctx.world.barrier()?;
    if rank == 0 {
        println!("sddp_pattern: OK");
    }
    // `ctx` drops here: the window, then the communicators, then `MPI_Finalize`.
    Ok(())
}
// mpi-test: np=2.. release
