// R8: safe code — mem::forget(Win::create) ends the buffer borrow while the window stays exposed.
use ferrompi::{LockType, Mpi, Win};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    if world.rank() == 0 {
        let mut buf = vec![0u64; 32];
        let win = Win::create(&world, &mut buf)?;
        std::mem::forget(win);
        drop(buf);                                   // allocation freed; window still points at it
        let victim = vec![0xAAAA_AAAA_AAAA_AAAAu64; 32]; // reuses the chunk
        world.barrier()?;  // rank 1 puts during/after this
        world.barrier()?;
        println!("rank0: victim[0..4] = {:x?}", &victim[..4]);
        std::process::exit(0); // skip finalize (leaked window would make MPI_Win_free collective mismatch)
    } else {
        let mut dummy = vec![0u64; 32];
        let win = Win::create(&world, &mut dummy)?;
        world.barrier()?;
        {
            let _g = win.lock(LockType::Exclusive, 0)?;
            win.put(&[0x5555_5555_5555_5555u64; 4], 0, 0, 4)?;
        }
        world.barrier()?;
        std::mem::forget(win);
        std::process::exit(0);
    }
}
