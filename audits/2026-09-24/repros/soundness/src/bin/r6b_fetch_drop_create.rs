// R6b: as R6 but over a Win::create window (MPICH defers the op to the closing fence).
use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut mem = vec![0i64; 1];
    if world.rank() == 1 { mem[0] = 0x1111_1111_1111_1111; }
    let win = Win::create(&world, &mut mem)?;
    win.fence(WinFenceAssert::default())?;
    let mut victim: Option<Box<i64>> = None;
    if world.rank() == 0 {
        let _ = win.fetch_and_op(1, 1, 0, ReduceOp::Sum)?; // natural "fire-and-forget" increment
        victim = Some(Box::new(0x2222_2222_2222_2222i64)); // reuses the freed result chunk
    }
    win.fence(WinFenceAssert::default())?;
    if let Some(b) = victim {
        println!("rank0: unrelated Box<i64> now holds {:#x} (expected 0x2222222222222222)", *b);
    }
    let mut v2: Option<Vec<i64>> = None;
    if world.rank() == 0 {
        { let mut tmp = vec![0i64; 1]; win.get(&mut tmp, 1, 0, 1)?; } // buffer dropped mid-epoch
        v2 = Some(vec![0x3333_3333_3333_3333i64; 1]);
    }
    win.fence(WinFenceAssert::default())?;
    if let Some(v) = v2 { println!("rank0: unrelated Vec after get() completes = {:#x} (expected 0x3333333333333333)", v[0]); }
    Ok(())
}
