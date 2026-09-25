// R2: safe code — collectives do not validate recv.len() against send.len()*size.
use ferrompi::Mpi;
#[repr(C)]
struct Frame { recv: [u32; 1], canary: [u32; 15] }
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut f = Frame { recv: [0; 1], canary: [0xC0FFEE; 15] };
    let send = [0xDEADu32; 8];
    world.allgather(&send, &mut f.recv)?;           // recv slice has 1 element, MPI writes 8*size
    println!("allgather: canary = {:x?}", &f.canary[..8]);
    let mut g = Frame { recv: [0; 1], canary: [0xC0FFEE; 15] };
    world.gather(&send, &mut g.recv, 0)?;
    println!("gather:    canary = {:x?}", &g.canary[..8]);
    // gatherv: counts/displs never checked against recv.len()
    let mut h = Frame { recv: [0; 1], canary: [0xC0FFEE; 15] };
    let counts = vec![8i32; world.size() as usize];
    let displs: Vec<i32> = (0..world.size()).map(|r| r * 8).collect();
    world.gatherv(&send, &mut h.recv, &counts, &displs, 0)?;
    println!("gatherv:   canary = {:x?}", &h.canary[..8]);
    Ok(())
}
