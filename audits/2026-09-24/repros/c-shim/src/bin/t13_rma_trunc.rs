use ferrompi::{Mpi, Win, WinFenceAssert};
fn main() {
    let mpi = Mpi::init().unwrap();
    let world = mpi.world();
    let win = Win::<f64>::allocate(&world, 8).unwrap();
    win.fence(WinFenceAssert::default()).unwrap();
    let buf = [3.0f64; 4];
    let bogus: i64 = (1i64 << 32) + 4; // not representable as int
    let r = win.put(&buf, 0, 0, bogus);
    eprintln!("put(target_count={bogus}) -> {:?}", r.map_err(|e| e.to_string()));
    win.fence(WinFenceAssert::default()).unwrap();
    eprintln!("window = {:?}", win.local_slice());
}
