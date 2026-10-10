use ferrompi::{Mpi, Win, WinFenceAssert};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut win = Win::<f64>::allocate(&world, 4)?;
    let buf = [1.0f64, 2.0, 3.0, 4.0];

    // `buf` is declared before the epoch: MPI may read it until the
    // closing fence returns
    win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
        // Rank 0 puts four elements into rank 1's window at displacement 0
        if world.rank() == 0 {
            e.put(&buf, 1, 0)?;
        }
        Ok(())
    })?;

    // The epoch is closed: rank 1's window holds the data
    if world.rank() == 1 {
        let mut seen = [0.0f64; 4];
        win.local().copy_to_slice(&mut seen);
        assert_eq!(seen, buf);
    }
    Ok(())
}
// mpi-test: np=2 release
