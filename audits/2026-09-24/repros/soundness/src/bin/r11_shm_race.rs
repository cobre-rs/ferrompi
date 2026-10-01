// R11: safe code — &[T] from remote_slice aliases memory another process mutates -> data race / miscompile.
use ferrompi::{Mpi, SharedWindow};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let node = mpi.world().split_shared()?;
    let mut win = SharedWindow::<u64>::allocate(&node, 1)?;
    win.local_slice_mut()[0] = 0;
    win.fence()?;
    if node.rank() == 0 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        win.local_slice_mut()[0] = 1;
        eprintln!("rank0: flag set");
        std::thread::sleep(std::time::Duration::from_secs(3));
        eprintln!("rank0: exiting");
        std::process::exit(0);
    } else {
        let r = win.remote_slice(0)?;
        let mut spins: u64 = 0;
        while r[0] == 0 { spins = spins.wrapping_add(1); }
        eprintln!("rank1: observed flag after {spins} spins");
        std::process::exit(0);
    }
}
