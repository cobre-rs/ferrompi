// R4: safe code — send/recv_custom with unbounded T and unchecked datatype extent.
use ferrompi::{CustomDatatype, DatatypeTag, Mpi};
#[repr(C)]
struct Frame { recv: [u8; 1], canary: [u8; 63] }
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    // Case A: datatype extent (32 bytes) > size_of::<u8>() (1 byte).
    let dt = CustomDatatype::contiguous(32, DatatypeTag::U8)?;
    let src = [0x41u8; 32];
    let req = world.isend_custom(&src[..1], &dt, 0, 1)?; // count=1 element of dt (reads 32 bytes from a 1-byte slice)
    let mut f = Frame { recv: [0; 1], canary: [0; 63] };
    let st = world.recv_custom(&mut f.recv, &dt, 0, 1)?;
    req.wait()?;
    println!("A: status.count={} canary[0..31] = {:x?}", st.count, &f.canary[..31]);
    // Case B: arbitrary T (Box<u64>) receives raw bytes -> invalid pointer in safe code.
    let dt8 = CustomDatatype::contiguous(8, DatatypeTag::U8)?;
    let bytes = [0x41u8; 8];
    let req = world.isend_custom(&bytes[..1], &dt8, 0, 2)?; // 1 element of dt8 = 8 bytes
    let mut boxes: [Box<u64>; 1] = [Box::new(5)];
    world.recv_custom(&mut boxes, &dt8, 0, 2)?;
    req.wait()?;
    println!("B: about to deref Box<u64> filled by MPI...");
    println!("B: value = {}", *boxes[0]);
    Ok(())
}
