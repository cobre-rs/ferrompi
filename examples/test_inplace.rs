//! Folds the twelve historical in-place-collective examples into one binary:
//! `test_{gather,allgather,scatter,alltoall}_inplace`, their `i*`
//! nonblocking forms, and their `*_init_inplace` persistent forms. Each
//! function below keeps the original file's assertion, generalized from
//! `size == 4` to any `world.size() >= 2`:
//!
//! | Function         | Family      | Assertion (inventory row) |
//! |---|---|---|
//! | `gather`         | blocking    | root: `data == [r*10 for r in 0..size]`; non-root sends `[rank*10]` via `gather` |
//! | `gather_all_inplace` | blocking | as `gather`, but non-root also calls `gather_inplace`, with its block `[rank*10]` |
//! | `allgather`      | blocking    | every rank: `data == [r*10 ...]` |
//! | `scatter`        | blocking    | root keeps `data[0] == 0`; rank r receives `r*10` |
//! | `alltoall`       | blocking    | every rank, every slot s: `data[s] == s*10 + rank` |
//! | `igather`        | nonblocking | as `gather`, via `igather_inplace` / `igather` + `wait` |
//! | `igather_all_inplace` | nonblocking | as `gather_all_inplace`, via `igather_inplace` + `wait` on every rank |
//! | `iallgather`     | nonblocking | as `allgather`, via `iallgather_inplace` + `wait` |
//! | `iscatter`       | nonblocking | as `scatter`, via `iscatter_inplace` + `wait` |
//! | `ialltoall`      | nonblocking | as `alltoall`, via `ialltoall_inplace` + `wait` |
//! | `gather_init`    | persistent  | as `gather`, 3 iterations of `start`/`wait`, root re-seeds its slot each time |
//! | `gather_init_all_inplace` | persistent | as `gather_all_inplace`, via `gather_init_inplace` on every rank, 3 iterations |
//! | `allgather_init` | persistent  | as `allgather`, 3 iterations, re-seed |
//! | `alltoall_init`  | persistent  | as `alltoall`, 3 iterations, re-seed |
//! | `scatter_init`   | persistent  | as `scatter`, 3 iterations, root re-seeds all slots; skipped on MPICH 4.2.x |
//!
//! The persistent functions run only where the library provides
//! persistent collectives (`common::has_mpi4_collectives()`); otherwise,
//! each asserts `Error::NotSupported` instead of running. `scatter_init`
//! additionally carries its own narrow MPICH 4.2.x skip for a known
//! `MPI_Scatter_init` + `MPI_IN_PLACE` deadlock.
//!
//! The three `*_all_inplace` functions have no historical file: every rank
//! calls the in-place gather, with a non-root rank's `data` as its own block.
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_inplace
// mpi-test: np=2.. skip-ok=mpich-4.2

use ferrompi::{Communicator, Error, Mpi, Result};

mod common;

const ITERATIONS: usize = 3;

/// `test_gather_inplace.rs`. Root calls `gather_inplace`; non-root ranks call
/// the out-of-place `gather` with an empty receive slice.
fn gather(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    if rank == 0 {
        let mut data = vec![0i32; size as usize];
        world.gather_inplace(&mut data, 0)?;
        let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
        assert_eq!(
            data, expected,
            "gather: rank 0 expected {expected:?} but got {data:?}"
        );
    } else {
        let send = vec![rank * 10];
        let mut recv: Vec<i32> = vec![];
        world.gather(&send, &mut recv, 0)?;
    }
    Ok(())
}

/// Every rank calls `gather_inplace`: root with a slot per rank, a non-root
/// rank with its one-element block.
fn gather_all_inplace(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data = if rank == 0 {
        vec![0i32; size as usize]
    } else {
        vec![rank * 10]
    };
    world.gather_inplace(&mut data, 0)?;
    if rank == 0 {
        let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
        assert_eq!(
            data, expected,
            "gather_all_inplace: rank 0 expected {expected:?} but got {data:?}"
        );
    }
    Ok(())
}

/// `test_allgather_inplace.rs`.
fn allgather(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data = vec![0i32; size as usize];
    data[rank as usize] = rank * 10;
    world.allgather_inplace(&mut data)?;
    let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
    assert_eq!(
        data, expected,
        "allgather: rank {rank} expected {expected:?} but got {data:?}"
    );
    Ok(())
}

/// `test_scatter_inplace.rs`.
fn scatter(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    if rank == 0 {
        let mut data: Vec<i32> = (0..size).map(|r| r * 10).collect();
        world.scatter_inplace(&mut data, 0)?;
        assert_eq!(
            data[0], 0,
            "scatter: rank 0 expected data[0]==0 but got {}",
            data[0]
        );
    } else {
        let mut data = vec![0i32; 1];
        world.scatter_inplace(&mut data, 0)?;
        let expected = rank * 10;
        assert_eq!(
            data[0], expected,
            "scatter: rank {rank} expected data[0]=={expected} but got {}",
            data[0]
        );
    }
    Ok(())
}

/// `test_alltoall_inplace.rs`.
fn alltoall(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data: Vec<i32> = (0..size).map(|s| rank * 10 + s).collect();
    world.alltoall_inplace(&mut data)?;
    for s in 0..size {
        let expected = s * 10 + rank;
        assert_eq!(
            data[s as usize], expected,
            "alltoall: rank {rank} slot {s} expected {expected} but got {}",
            data[s as usize]
        );
    }
    Ok(())
}

/// `test_igather_inplace.rs`.
fn igather(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    if rank == 0 {
        let mut data = vec![0i32; size as usize];
        ferrompi::scope(|s| world.igather_inplace(s, &mut data, 0)?.wait())?;
        let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
        assert_eq!(
            data, expected,
            "igather: rank 0 expected {expected:?} but got {data:?}"
        );
    } else {
        let send = vec![rank * 10];
        let mut recv: Vec<i32> = vec![];
        ferrompi::scope(|s| world.igather(s, &send, &mut recv, 0)?.wait())?;
    }
    Ok(())
}

/// Every rank calls `igather_inplace`: root with a slot per rank, a non-root
/// rank with its one-element block.
fn igather_all_inplace(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data = if rank == 0 {
        vec![0i32; size as usize]
    } else {
        vec![rank * 10]
    };
    ferrompi::scope(|s| world.igather_inplace(s, &mut data, 0)?.wait())?;
    if rank == 0 {
        let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
        assert_eq!(
            data, expected,
            "igather_all_inplace: rank 0 expected {expected:?} but got {data:?}"
        );
    }
    Ok(())
}

/// `test_iallgather_inplace.rs`.
fn iallgather(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data = vec![0i32; size as usize];
    data[rank as usize] = rank * 10;
    ferrompi::scope(|s| world.iallgather_inplace(s, &mut data)?.wait())?;
    let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
    assert_eq!(
        data, expected,
        "iallgather: rank {rank} expected {expected:?} but got {data:?}"
    );
    Ok(())
}

/// `test_iscatter_inplace.rs`.
fn iscatter(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    if rank == 0 {
        let mut data: Vec<i32> = (0..size).map(|r| r * 10).collect();
        ferrompi::scope(|s| world.iscatter_inplace(s, &mut data, 0)?.wait())?;
        assert_eq!(
            data[0], 0,
            "iscatter: rank 0 expected data[0]==0 but got {}",
            data[0]
        );
    } else {
        let mut data = vec![0i32; 1];
        ferrompi::scope(|s| world.iscatter_inplace(s, &mut data, 0)?.wait())?;
        let expected = rank * 10;
        assert_eq!(
            data[0], expected,
            "iscatter: rank {rank} expected data[0]=={expected} but got {}",
            data[0]
        );
    }
    Ok(())
}

/// `test_ialltoall_inplace.rs`.
fn ialltoall(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data: Vec<i32> = (0..size).map(|s| rank * 10 + s).collect();
    ferrompi::scope(|s| world.ialltoall_inplace(s, &mut data)?.wait())?;
    for s in 0..size {
        let expected = s * 10 + rank;
        assert_eq!(
            data[s as usize], expected,
            "ialltoall: rank {rank} slot {s} expected {expected} but got {}",
            data[s as usize]
        );
    }
    Ok(())
}

/// `test_gather_init_inplace.rs`. Non-root ranks use the out-of-place
/// `gather_init`, exactly as the original file does.
fn gather_init(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();

    if rank == 0 {
        let mut data = vec![0i32; size as usize];
        let mut req = world.gather_init_inplace(&mut data, 0)?;
        for iter in 0..ITERATIONS {
            data[0] = 0; // re-seed: a persistent request borrows nothing.
            req.start()?;
            req.wait()?;
            assert_eq!(
                data, expected,
                "gather_init iter {iter}: rank 0 expected {expected:?} but got {data:?}"
            );
        }
    } else {
        let send = vec![rank * 10];
        let mut recv: Vec<i32> = vec![];
        let mut req = world.gather_init(&send, &mut recv, 0)?;
        for _ in 0..ITERATIONS {
            req.start()?;
            req.wait()?;
        }
    }
    Ok(())
}

/// Every rank calls `gather_init_inplace`: root with a slot per rank, a
/// non-root rank with its one-element block, kept alive across every
/// `start`/`wait`.
fn gather_init_all_inplace(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
    let mut data = if rank == 0 {
        vec![0i32; size as usize]
    } else {
        vec![rank * 10]
    };
    let mut req = world.gather_init_inplace(&mut data, 0)?;
    for iter in 0..ITERATIONS {
        if rank == 0 {
            data.fill(0); // re-seed: a slot left by the previous start must not pass.
        }
        req.start()?;
        req.wait()?;
        if rank == 0 {
            assert_eq!(
                data, expected,
                "gather_init_all_inplace iter {iter}: rank 0 expected {expected:?} but got {data:?}"
            );
        }
    }
    Ok(())
}

/// `test_allgather_init_inplace.rs`.
fn allgather_init(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let expected: Vec<i32> = (0..size).map(|r| r * 10).collect();
    let mut data = vec![0i32; size as usize];
    data[rank as usize] = rank * 10;
    let mut req = world.allgather_init_inplace(&mut data)?;
    for iter in 0..ITERATIONS {
        data[rank as usize] = rank * 10; // re-seed before each start.
        req.start()?;
        req.wait()?;
        assert_eq!(
            data, expected,
            "allgather_init iter {iter}: rank {rank} expected {expected:?} but got {data:?}"
        );
    }
    Ok(())
}

/// `test_alltoall_init_inplace.rs`.
fn alltoall_init(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size();
    let mut data: Vec<i32> = (0..size).map(|s| rank * 10 + s).collect();
    let mut req = world.alltoall_init_inplace(&mut data)?;
    for iter in 0..ITERATIONS {
        for s in 0..size {
            data[s as usize] = rank * 10 + s; // re-seed before each start.
        }
        req.start()?;
        req.wait()?;
        for s in 0..size {
            let expected = s * 10 + rank;
            assert_eq!(
                data[s as usize], expected,
                "alltoall_init iter {iter}: rank {rank} slot {s} expected {expected} but got {}",
                data[s as usize]
            );
        }
    }
    Ok(())
}

/// `test_scatter_init_inplace.rs`. Narrow skip for the known MPICH 4.2.x
/// `MPI_Scatter_init` + `MPI_IN_PLACE` deadlock — only this function is
/// skipped, not the other three persistent families.
fn scatter_init(world: &Communicator) -> Result<()> {
    if let Ok(version) = Mpi::library_version() {
        // Whitespace-tolerant match on any MPICH 4.2.x. Fixed in MPICH 4.3+;
        // absent on OpenMPI.
        let mpich_42 = version.contains("MPICH")
            && version
                .lines()
                .any(|line| line.contains("Version:") && line.contains("4.2."));
        if mpich_42 {
            common::skip(
                world,
                "MPICH 4.2.x MPI_Scatter_init + MPI_IN_PLACE deadlock",
            );
            return Ok(());
        }
    }

    let rank = world.rank();
    let size = world.size();
    let seed: Vec<i32> = (0..size).map(|r| r * 10).collect();

    if rank == 0 {
        let mut data = seed.clone();
        let mut req = world.scatter_init_inplace(&mut data, 0)?;
        for iter in 0..ITERATIONS {
            data.copy_from_slice(&seed); // re-seed the full buffer.
            req.start()?;
            req.wait()?;
            assert_eq!(
                data[0], 0,
                "scatter_init iter {iter}: rank 0 expected data[0]==0 but got {}",
                data[0]
            );
        }
    } else {
        let mut data = vec![0i32; 1];
        let mut req = world.scatter_init_inplace(&mut data, 0)?;
        for iter in 0..ITERATIONS {
            req.start()?;
            req.wait()?;
            let expected = rank * 10;
            assert_eq!(
                data[0], expected,
                "scatter_init iter {iter}: rank {rank} expected data[0]=={expected} but got {}",
                data[0]
            );
        }
    }
    Ok(())
}

/// Where the library lacks persistent collectives (below MPI 4.0, except
/// Open MPI 5), asserts that the four persistent in-place collectives
/// refuse with `Error::NotSupported` naming themselves, instead of running.
/// Each reaches the FFI shim on every rank, so each is exercised on every
/// rank.
fn assert_persistent_inplace_not_supported(world: &Communicator) -> Result<()> {
    let rank = world.rank();
    let size = world.size() as usize;

    let mut data = vec![0i32; if rank == 0 { size } else { 1 }];
    let gather_ok = matches!(
        world.gather_init_inplace(&mut data, 0),
        Err(Error::NotSupported(op)) if op == "gather_init_inplace"
    );
    common::check(
        world,
        gather_ok,
        "gather_init_inplace refuses without persistent collectives",
    );

    let scatter_ok = matches!(
        world.scatter_init_inplace(&mut data, 0),
        Err(Error::NotSupported(op)) if op == "scatter_init_inplace"
    );
    common::check(
        world,
        scatter_ok,
        "scatter_init_inplace refuses without persistent collectives",
    );

    let mut data = vec![0i32; size];
    let ok = matches!(
        world.allgather_init_inplace(&mut data),
        Err(Error::NotSupported(op)) if op == "allgather_init_inplace"
    );
    common::check(
        world,
        ok,
        "allgather_init_inplace refuses without persistent collectives",
    );

    let mut data = vec![0i32; size];
    let ok = matches!(
        world.alltoall_init_inplace(&mut data),
        Err(Error::NotSupported(op)) if op == "alltoall_init_inplace"
    );
    common::check(
        world,
        ok,
        "alltoall_init_inplace refuses without persistent collectives",
    );

    Ok(())
}

fn main() -> Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let size = world.size();
    assert!(
        size >= 2,
        "test_inplace requires at least 2 MPI processes, got {size}"
    );

    gather(&world)?;
    gather_all_inplace(&world)?;
    allgather(&world)?;
    scatter(&world)?;
    alltoall(&world)?;

    igather(&world)?;
    igather_all_inplace(&world)?;
    iallgather(&world)?;
    iscatter(&world)?;
    ialltoall(&world)?;

    if common::has_mpi4_collectives() {
        gather_init(&world)?;
        gather_init_all_inplace(&world)?;
        allgather_init(&world)?;
        alltoall_init(&world)?;
        scatter_init(&world)?;
    } else {
        assert_persistent_inplace_not_supported(&world)?;
    }

    world.barrier()?;

    if world.rank() == 0 {
        println!("PASS: test_inplace");
    }

    Ok(())
}
