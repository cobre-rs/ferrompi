// R7b: safe put/get with target_count/target_disp beyond the window (remote OOB write) and
// target_count > origin.len() (local OOB write). No validation in ferrompi; MPICH accepts it.
use ferrompi::{Mpi, Win, WinFenceAssert};
#[repr(C)]
struct Frame { win: [u64; 4], canary: [u64; 12] }
#[repr(C)]
struct Small { origin: [u64; 1], canary: [u64; 7] }
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut f = Box::new(Frame { win: [7; 4], canary: [0xC0FFEE; 12] });
    let mut s = Box::new(Small { origin: [0; 1], canary: [0xC0FFEE; 7] });
    {
        let win = Win::create(&world, &mut f.win)?;
        win.fence(WinFenceAssert::default())?;
        if world.rank() == 0 {
            let buf = [0x4141_4141_4141_4141u64; 8];
            win.put(&buf, 1, 2, 8)?;                 // disp 2 + 8 elems into a 4-elem window
            win.get(&mut s.origin, 1, 0, 4)?;        // 4 target elems into a 1-elem origin
        }
        win.fence(WinFenceAssert::default())?;
    }
    if world.rank() == 1 {
        println!("rank1: window-adjacent canary = {:x?}", &f.canary[..8]);
    } else {
        println!("rank0: origin-adjacent canary = {:x?}", &s.canary[..4]);
    }
    Ok(())
}
