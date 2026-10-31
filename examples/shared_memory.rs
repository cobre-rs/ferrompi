use ferrompi::{Mpi, Result, Win, WinFenceAssert};

const COUNT: usize = 10;

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    // A communicator of the processes that share memory, that is, the processes
    // on one physical node.
    let node = world.split_shared()?;
    let rank = node.rank();
    let size = node.size();
    println!(
        "Global rank {}, local rank {rank} of {size} on node",
        world.rank()
    );

    // Each process allocates 10 f64 elements of shared memory. Every process
    // can load and store the region of every other process.
    let mut win = Win::<f64>::allocate_shared(&node, COUNT)?;

    // Phase 1: each rank fills its own region with a recognizable pattern, in a
    // fence epoch:
    //   rank 0: [0, 1, 2, ..., 9]
    //   rank 1: [10, 11, 12, ..., 19]
    //   and so on.
    // The stores are visible to the loads of the next fence epoch.
    let none = WinFenceAssert::none();
    win.fence(none, none, |e| {
        let mut pattern = [0.0f64; COUNT];
        for (i, x) in pattern.iter_mut().enumerate() {
            *x = (rank as usize * COUNT + i) as f64;
        }
        e.local().copy_from_slice(&pattern);
        Ok(())
    })?;

    // Phase 2: a fence epoch (active target). Every rank reads rank 0's region
    // through the epoch's view. The closing fence asserts `no_succeed`, that it
    // opens no epoch, and the barrier makes sure every rank left the fence epoch
    // before any rank takes a lock.
    let rank0 = win.fence(none, WinFenceAssert::no_succeed(), |e| {
        let mut seen = [0.0f64; COUNT];
        e.remote(0)?.copy_to_slice(&mut seen);
        Ok(seen)
    })?;
    node.barrier()?;
    assert_eq!(rank0[..3], [0.0, 1.0, 2.0]);
    println!("Rank {rank}: rank 0's data[0..3] = {:?}", &rank0[..3]);

    // Phase 3: a passive-target epoch. Inside one `lock_all` closure, each rank
    // adds its rank plus one to slot 0 of its own region and publishes the
    // store: store, `sync`, barrier, `sync`, load. Then it reads its
    // neighbour's slot 0.
    let neighbour = (rank + 1) % size;
    let seen = win.lock_all(|e| {
        let own = e.local();
        own.write(0, own.read(0) + (rank + 1) as f64);
        e.sync()?;
        node.barrier()?;
        e.sync()?;
        Ok(e.remote(neighbour)?.read(0))
    })?;
    assert_eq!(
        seen,
        (neighbour as usize * COUNT) as f64 + (neighbour + 1) as f64
    );
    println!("Rank {rank}: neighbour {neighbour}'s slot 0 = {seen}");

    // The window is freed on drop, which is collective; the node communicator is
    // freed on drop; MPI is finalized when `mpi` is dropped.
    Ok(())
}
// mpi-test: np=2.. release
