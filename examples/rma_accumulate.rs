use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut win = Win::<f64>::allocate(&world, 4)?;
    let buf = [10.0f64, 20.0, 30.0, 40.0];

    // Rank 1 initialises its window
    if world.rank() == 1 {
        win.local_slice_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }

    // `buf` is declared before the epoch: MPI may read it until the
    // closing fence returns
    win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
        // Rank 0 adds [10, 20, 30, 40] to rank 1's window
        if world.rank() == 0 {
            e.accumulate(&buf, 1, 0, ReduceOp::Sum)?;
        }
        Ok(())
    })?;

    // The epoch is closed: rank 1's window holds the sums
    if world.rank() == 1 {
        assert_eq!(win.local_slice(), [11.0, 22.0, 33.0, 44.0]);
    }
    Ok(())
}
// mpi-test: np=2
