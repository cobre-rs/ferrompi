// R6: safe code — dropping a PendingFetchResult before the epoch closes frees the result box
// that MPI writes into at the closing fence.
use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut win = Win::<i64>::allocate(&world, 1)?;
    if world.rank() == 1 { win.local_slice_mut()[0] = 0x1111_1111_1111_1111; }
    win.fence(WinFenceAssert::default())?;
    let mut victim_val = 0;
    let mut victim_addr = 0usize;
    let mut victim: Option<Box<i64>> = None;
    if world.rank() == 0 {
        let _ = win.fetch_and_op(1, 1, 0, ReduceOp::Sum)?; // "fire and forget" atomic increment
        let b = Box::new(0x2222_2222_2222_2222i64);       // reuses the freed result chunk
        victim_addr = &*b as *const i64 as usize;
        victim = Some(b);
    }
    win.fence(WinFenceAssert::default())?; // MPI completes fetch_and_op here
    if let Some(b) = victim { victim_val = *b; }
    if world.rank() == 0 {
        println!("rank0: unrelated Box<i64> at {victim_addr:#x} now holds {victim_val:#x} (expected 0x2222222222222222)");
    }
    // Same for get(): buffer dropped before the closing fence.
    win.fence(WinFenceAssert::default())?;
    let mut v2: Option<Vec<i64>> = None;
    if world.rank() == 0 {
        { let mut tmp = vec![0i64; 1]; win.get(&mut tmp, 1, 0, 1)?; }
        v2 = Some(vec![0x3333_3333_3333_3333i64; 1]);
    }
    win.fence(WinFenceAssert::default())?;
    if let Some(v) = v2 { println!("rank0: unrelated Vec after get() completes = {:#x}", v[0]); }
    Ok(())
}
