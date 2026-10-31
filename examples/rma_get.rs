use ferrompi::{Mpi, Win, WinFenceAssert};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut win = Win::<f64>::allocate(&world, 4)?;
    let mut buf = [0.0f64; 4];

    // Rank 1 initialises its window
    if world.rank() == 1 {
        win.local().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }

    // `buf` is declared before the epoch: MPI may write it until the
    // closing fence returns
    win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
        // Rank 0 reads four elements from rank 1's window at displacement 0
        if world.rank() == 0 {
            e.get(&mut buf, 1, 0)?;
        }
        Ok(())
    })?;

    // The epoch is closed: rank 0's buffer holds the data
    if world.rank() == 0 {
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);
    }
    Ok(())
}
// mpi-test: np=2 release
