// R5: safe code — windows are not lifetime-bound to Mpi; finalize frees their memory.
use ferrompi::{Mpi, SharedWindow};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let node = mpi.world().split_shared()?;
    let mut win = SharedWindow::<f64>::allocate(&node, 1 << 16)?;
    win.local_slice_mut()[0] = 42.0;
    drop(node);
    drop(mpi); // ferrompi_finalize -> MPI_Win_free on every live window -> shm unmapped
    eprintln!("finalized; reading win.local_slice()[0] ...");
    let v = win.local_slice()[0];
    eprintln!("read {v}");
    Ok(())
}
