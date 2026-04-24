//! Integration test for igather_inplace.
//!
//! Each rank r pre-writes r*10 into its in-place slot at offset r of the shared
//! buffer. Only rank 0 (root) calls igather_inplace; non-root ranks call the
//! regular igather to contribute their value.
//!
//! After the call, rank 0 asserts the buffer equals [0, 10, 20, 30].
//!
//! Run with: mpiexec -n 4 cargo run --example test_igather_inplace

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(size, 4, "This test requires exactly 4 MPI processes");

    // recvcount = 1 element per rank; full buffer length = size.
    // Each rank's slot is at offset rank * recvcount = rank.
    if rank == 0 {
        // Root: allocate full buffer and pre-write own slot.
        let mut data = vec![0i32; size as usize];
        data[rank as usize] = rank * 10; // slot 0 = 0

        let req = world.igather_inplace(&mut data, 0)?;
        req.wait()?;

        let expected = vec![0i32, 10, 20, 30];
        assert_eq!(
            data, expected,
            "igather_inplace: rank 0 expected {:?} but got {:?}",
            expected, data
        );
        println!("igather_inplace: {:?}", data);
    } else {
        // Non-root: use regular igather with a single-element send buffer.
        let send = vec![rank * 10];
        // recv is ignored at non-root.
        let mut recv = vec![0i32; 0];
        let req = world.igather(&send, &mut recv, 0)?;
        req.wait()?;
    }

    world.barrier()?;

    Ok(())
}
