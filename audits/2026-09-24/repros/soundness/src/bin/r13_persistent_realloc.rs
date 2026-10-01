// R13: safe code — persistent request outlives its buffer's allocation (Vec reallocation).
use ferrompi::Mpi;
fn main() -> Result<(), ferrompi::Error> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let mut data = vec![0u64; 4];
    if world.rank() == 0 { data.fill(0x5555_5555_5555_5555); }
    let mut p = world.recv_init(&mut data, 1 - world.rank(), 9)?; // both ranks: recv from peer
    data.reserve(4096);                              // reallocates; old chunk freed
    let victim = vec![0xAAAA_AAAA_AAAA_AAAAu64; 4];  // reuses the freed chunk
    let send = [0x5151_5151_5151_5151u64; 4];
    p.start()?;
    world.send(&send, 1 - world.rank(), 9)?;
    p.wait()?;
    println!("rank {}: victim = {:x?}; data[0] = {:x}", world.rank(), victim, data[0]);
    Ok(())
}
