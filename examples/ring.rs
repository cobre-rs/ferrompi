//! Ring communication example - point-to-point communication.
//!
//! Each process sends data to the next process in a ring pattern.
//!
//! Run with: mpiexec -n 4 cargo run --example ring
// mpi-test: np=2..

use ferrompi::{Mpi, Result};

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    if size < 2 {
        if rank == 0 {
            eprintln!("This example requires at least 2 processes");
        }
        return Ok(());
    }

    // Calculate neighbors in the ring
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    // Data to send
    let send_data = vec![rank as f64 * 100.0 + 1.0, rank as f64 * 100.0 + 2.0];
    let mut recv_data = vec![0.0; 2];

    println!("Rank {}: sending {:?} to rank {}", rank, send_data, next);

    // Even ranks send first, then receive
    // Odd ranks receive first, then send
    // This avoids deadlock
    if rank % 2 == 0 {
        world.send(&send_data, next, 0)?;
        let (source, tag, count) = world.recv(&mut recv_data, prev, 0)?;
        println!(
            "Rank {}: received {:?} from rank {} (tag={}, count={})",
            rank, recv_data, source, tag, count
        );
    } else {
        let (source, tag, count) = world.recv(&mut recv_data, prev, 0)?;
        println!(
            "Rank {}: received {:?} from rank {} (tag={}, count={})",
            rank, recv_data, source, tag, count
        );
        world.send(&send_data, next, 0)?;
    }

    // Verify we got the right data
    let expected = vec![prev as f64 * 100.0 + 1.0, prev as f64 * 100.0 + 2.0];
    assert_eq!(recv_data, expected, "Data mismatch!");

    world.barrier()?;

    if rank == 0 {
        println!("\nRing communication test passed!");
    }

    Ok(())
}
