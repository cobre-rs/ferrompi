use ferrompi::{Mpi, Win};
fn main() {
    let win: Win<'static, f64>;
    {
        let mpi = Mpi::init().unwrap();
        let world = mpi.world();
        let mut w = Win::<f64>::allocate(&world, 1024).unwrap();
        w.local_slice_mut().fill(1.5);
        win = w;
        // mpi dropped here -> ferrompi_finalize -> MPI_Win_free on the live window
    }
    let s: f64 = win.local_slice().iter().sum();
    eprintln!("sum after finalize = {s}");
    eprintln!("SURVIVED");
}
