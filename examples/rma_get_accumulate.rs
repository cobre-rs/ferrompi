use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut win = Win::<i32>::allocate(&world, 4)?;
    let origin = [1i32, 2, 3, 4];
    let mut result = [0i32; 4];

    // Rank 1 initialises its window
    if world.rank() == 1 {
        win.local().copy_from_slice(&[10, 20, 30, 40]);
    }

    // `origin` and `result` are declared before the epoch: MPI uses them
    // until the closing fence returns
    win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
        // Rank 0 atomically fetches rank 1's values and adds [1, 2, 3, 4]
        if world.rank() == 0 {
            e.get_accumulate(&origin, &mut result, 1, 0, ReduceOp::Sum)?;
        }
        Ok(())
    })?;

    // The epoch is closed: rank 0 holds the values from before the update,
    // and rank 1's window holds the sums
    if world.rank() == 0 {
        assert_eq!(result, [10, 20, 30, 40]);
    }
    if world.rank() == 1 {
        let mut seen = [0i32; 4];
        win.local().copy_to_slice(&mut seen);
        assert_eq!(seen, [11, 22, 33, 44]);
    }
    Ok(())
}
// mpi-test: np=2 release
