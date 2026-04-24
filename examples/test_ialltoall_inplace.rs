//! Integration test for ialltoall_inplace.
//!
//! Each rank r pre-writes data[s] = r*10 + s into slot s (the payload destined for
//! rank s). After ialltoall_inplace, every rank asserts data[s] == s*10 + r (the data
//! received FROM rank s, which is rank s's contribution destined for rank r).
//!
//! Run with: mpiexec -n 4 cargo run --example test_ialltoall_inplace

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(size, 4, "This test requires exactly 4 MPI processes");

    // Each rank r pre-writes slot s with the value it wants to send to rank s.
    // data[s] = r*10 + s  means "from rank r, to rank s: value r*10+s"
    let mut data: Vec<i32> = (0..size).map(|s| rank * 10 + s).collect();

    let req = world.ialltoall_inplace(&mut data)?;
    req.wait()?;

    // After the call: data[s] should contain what rank s sent to rank r.
    // Rank s sent to rank r: s*10 + r.
    for s in 0..size {
        let expected = s * 10 + rank;
        assert_eq!(
            data[s as usize], expected,
            "ialltoall_inplace: rank {} slot {} expected {} but got {}",
            rank, s, expected, data[s as usize]
        );
    }

    if rank == 0 {
        println!("ialltoall_inplace: rank 0 result {:?}", data);
    }

    world.barrier()?;

    Ok(())
}
