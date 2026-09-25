// R7: does MPI validate target_disp/target_count against the remote window size?
use ferrompi::{Mpi, Win, WinFenceAssert};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let win = Win::<u64>::allocate(&world, 4)?;
    win.fence(WinFenceAssert::default())?;
    if world.rank() == 0 {
        let buf = [0x4141414141414141u64; 64];
        let r = win.put(&buf, 1, 2, 64);
        println!("put 64 elems at disp 2 into 4-elem window -> {:?}", r.as_ref().map(|_| ()));
        let mut small = [0u64; 1];
        let r2 = win.get(&mut small, 1, 0, 4);
        println!("get 4 target elems into 1-elem origin -> {:?}", r2.as_ref().map(|_| ()));
    }
    win.fence(WinFenceAssert::default())?;
    println!("rank {} survived", world.rank());
    Ok(())
}
