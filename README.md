# FerroMPI

**Safe, generic Rust bindings for MPI, with MPI 4.0 persistent collectives and large counts when the library provides them.**

[![Crates.io](https://img.shields.io/crates/v/ferrompi.svg)](https://crates.io/crates/ferrompi)
[![Documentation](https://docs.rs/ferrompi/badge.svg)](https://docs.rs/ferrompi)
[![License](https://img.shields.io/crates/l/ferrompi.svg)](#license)
[![CI](https://github.com/cobre-rs/ferrompi/actions/workflows/test.yml/badge.svg)](https://github.com/cobre-rs/ferrompi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cobre-rs/ferrompi/branch/main/graph/badge.svg)](https://codecov.io/gh/cobre-rs/ferrompi)
[![Security](https://github.com/cobre-rs/ferrompi/actions/workflows/security.yml/badge.svg)](https://github.com/cobre-rs/ferrompi/actions/workflows/security.yml)

## Highlights

- Generic over `MpiDatatype`: one API for `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.
- Fallible operations return `Result` with structured MPI error context, not panics.
- Persistent collectives and counts above `i32::MAX` when the library reports MPI 4.0.
- RMA and shared-memory windows with RAII lock guards (feature: `rma`).
- Hybrid MPI + threads: `Communicator` is `Send + Sync`, and a call from a thread the requested `ThreadLevel` does not permit is rejected at run time.
- On MPICH at 2 ranks, a persistent allreduce was 20–58 % faster per call than `iallreduce` up to 4 KiB, at most 8 % faster at 32 KiB, and no faster from 256 KiB ([benches](benches/README.md)).

## Requirements

- **Rust 1.85** or newer (MSRV).
- **Linux**, the only CI-tested platform. macOS builds but is not CI-tested and lacks `LongDoubleInt`/`LongInt`. Windows is not supported.
- An MPI 3.1 library. CI tests MPICH 4.2, Open MPI 4.1 and Open MPI 5.0.
- Persistent collectives, `create_from_group` and counts above `i32::MAX` need a library that reports MPI 4.0 — currently MPICH 4.x (both Open MPI versions report 3.1).

**Ubuntu/Debian:**

```bash
sudo apt install mpich libmpich-dev
```

**macOS:**

```bash
brew install mpich
```

See [docs/mpi-compatibility.md](docs/mpi-compatibility.md) for build selection, platform details, and known issues.

## Quick start

```toml
[dependencies]
ferrompi = "0.6"
```

Enable `rma` for shared-memory windows, or `numa` for the SLURM job-topology helpers (implies `rma`):

```toml
[dependencies]
ferrompi = { version = "0.6", features = ["rma"] }
```

```rust,no_run
use ferrompi::{Mpi, ReduceOp};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();
    let size = world.size();

    println!("Hello from rank {} of {}", rank, size);

    // Generic all-reduce — works with any MpiDatatype
    let sum = world.allreduce_scalar(rank as f64, ReduceOp::Sum)?;
    println!("Rank {}: sum = {}", rank, sum);

    Ok(())
}
```

```bash
cargo build --release
mpiexec -n 4 ./target/release/my_program
```

`build.rs` uses `MPI_PKG_CONFIG`, `MPICC` or `CRAY_MPICH_DIR` when one is set, and otherwise searches pkg-config, `mpicc` and common install prefixes; see [docs/mpi-compatibility.md](docs/mpi-compatibility.md) for the full detection order. No rpath is embedded, so the MPI library must be discoverable at run time: set `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on macOS) or load your cluster's MPI module — same doc.

## Documentation

- [docs.rs](https://docs.rs/ferrompi) — API reference, built with all features; feature-gated items carry a badge.
- [examples/](examples/) — runnable examples for collectives, RMA windows, and hybrid MPI+threads.
- [Migrating from rsmpi](docs/migrating-from-rsmpi.md)
- [MPI compatibility](docs/mpi-compatibility.md) — build selection, platforms, running, and known issues.
- [Architecture](docs/architecture.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and run tests.

## Acknowledgments

- [rsmpi](https://github.com/rsmpi/rsmpi) - Comprehensive MPI bindings for Rust
