//! Integration test for scatter_inplace.
//!
//! Root (rank 0) pre-populates a 4-element buffer [0, 10, 20, 30]. After
//! scatter_inplace(root=0), rank 0 retains its own slot (data[0] == 0) and each
//! non-root rank receives a single-element buffer matching its slot value.
//!
//! Run with: mpiexec -n 4 cargo run --example test_scatter_inplace

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(size, 4, "This test requires exactly 4 MPI processes");

    if rank == 0 {
        // Root: full send buffer with one slot per rank.
        // Slot r contains the value to scatter to rank r.
        let mut data = vec![0i32, 10, 20, 30];

        world.scatter_inplace(&mut data, 0)?;

        // Root's own slot (index 0) must be retained unchanged.
        assert_eq!(
            data[0], 0,
            "scatter_inplace: rank 0 expected data[0]==0 but got {}",
            data[0]
        );
        println!("scatter_inplace: rank 0 retained slot data[0]={}", data[0]);
    } else {
        // Non-root: single-element receive buffer; scatter fills it with the
        // value root placed in slot `rank`.
        let mut data = vec![0i32; 1];

        world.scatter_inplace(&mut data, 0)?;

        let expected = rank * 10;
        assert_eq!(
            data[0], expected,
            "scatter_inplace: rank {} expected data[0]=={} but got {}",
            rank, expected, data[0]
        );
        println!("scatter_inplace: rank {} received {}", rank, data[0]);
    }

    world.barrier()?;

    Ok(())
}
