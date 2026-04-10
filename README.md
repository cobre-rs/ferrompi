# FerroMPI

**Safe, generic Rust bindings for MPI 4.x with persistent collectives support.**

[![Crates.io](https://img.shields.io/crates/v/ferrompi.svg)](https://crates.io/crates/ferrompi)
[![Documentation](https://docs.rs/ferrompi/badge.svg)](https://docs.rs/ferrompi)
[![License](https://img.shields.io/crates/l/ferrompi.svg)](LICENSE)
[![CI](https://github.com/cobre-rs/ferrompi/actions/workflows/test.yml/badge.svg)](https://github.com/cobre-rs/ferrompi/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/cobre-rs/ferrompi/branch/main/graph/badge.svg)](https://codecov.io/gh/cobre-rs/ferrompi)
[![Security](https://github.com/cobre-rs/ferrompi/actions/workflows/security.yml/badge.svg)](https://github.com/cobre-rs/ferrompi/actions/workflows/security.yml)

FerroMPI provides safe, generic Rust bindings to MPI through a thin C wrapper layer, enabling access to MPI 4.0+ features like **persistent collectives** that are not available in other Rust MPI bindings. All communication operations are generic over `MpiDatatype`, supporting `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`.

## Features

- 🚀 **MPI 4.0+ support**: Persistent collectives, large-count operations
- 🪶 **Lightweight**: Minimal C wrapper (~2400 lines), focused API
- 🔒 **Safe**: Rust-idiomatic API with proper error handling and RAII
- 🔧 **Flexible**: Works with MPICH, OpenMPI, Intel MPI, and Cray MPI
- ⚡ **Fast**: Zero-cost abstractions, direct FFI calls
- 🧬 **Generic**: Type-safe API for all supported MPI datatypes
- 🧵 **Thread-safe**: `Communicator` is `Send + Sync` for hybrid MPI+threads programs
- 🪟 **Shared memory**: RMA windows with RAII lock guards (feature: `rma`)
- 📊 **SLURM integration**: Job topology helpers (feature: `numa`)

## Why FerroMPI?

| Feature                | FerroMPI         | rsmpi                  |
| ---------------------- | ---------------- | ---------------------- |
| MPI Version            | 4.1              | 3.1                    |
| Persistent Collectives | ✅               | ❌                     |
| Large Count (>2³¹)     | ✅               | ❌                     |
| Generic API            | ✅               | ✅                     |
| Shared Memory Windows  | ✅               | ❌                     |
| Thread Safety          | `Send + Sync`    | `!Send`                |
| API Style              | Minimal, focused | Comprehensive          |
| C Wrapper              | ~2400 lines      | None (direct bindings) |

FerroMPI is ideal for:

- Iterative algorithms benefiting from persistent collectives (10-30% speedup)
- Applications with large data transfers (>2GB)
- Hybrid MPI+threads programs (OpenMP, Rayon, `std::thread`)
- Intra-node shared memory communication
- Users who want a simple, focused MPI API

## Supported Types

All communication operations are generic over `MpiDatatype`:

| Rust Type | MPI Equivalent |
| --------- | -------------- |
| `f32`     | `MPI_FLOAT`    |
| `f64`     | `MPI_DOUBLE`   |
| `i32`     | `MPI_INT32_T`  |
| `i64`     | `MPI_INT64_T`  |
| `u8`      | `MPI_UINT8_T`  |
| `u32`     | `MPI_UINT32_T` |
| `u64`     | `MPI_UINT64_T` |

## Feature Flags

| Feature | Description                                        | Dependencies |
| ------- | -------------------------------------------------- | ------------ |
| `rma`   | RMA shared memory window operations                | —            |
| `numa`  | NUMA-aware shared memory windows and SLURM helpers | `rma`        |

Enable features in your `Cargo.toml`:

```toml
[dependencies]
ferrompi = { version = "0.2", features = ["rma"] }
```

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ferrompi = "0.2"
```

### Requirements

- **Rust 1.74+**
- **MPICH 4.0+** (recommended) or **OpenMPI 5.0+**

**Ubuntu/Debian:**

```bash
sudo apt install mpich libmpich-dev
```

**macOS:**

```bash
brew install mpich
```

### Hello World

```rust
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

## Examples

### Blocking Collectives

```rust
use ferrompi::{Mpi, ReduceOp};

let mpi = Mpi::init()?;
let world = mpi.world();

// Broadcast (generic — works with f64, i32, u8, etc.)
let mut data = vec![0.0f64; 100];
if world.rank() == 0 {
    data.fill(42.0);
}
world.broadcast(&mut data, 0)?;

// All-reduce
let send = vec![1.0f64; 100];
let mut recv = vec![0.0f64; 100];
world.allreduce(&send, &mut recv, ReduceOp::Sum)?;

// Gather
let my_data = vec![world.rank() as f64];
let mut gathered = vec![0.0f64; world.size() as usize];
world.gather(&my_data, &mut gathered, 0)?;

// Works with integers too!
let mut int_data = vec![0i32; 100];
world.broadcast(&mut int_data, 0)?;
```

### Nonblocking Collectives

```rust
use ferrompi::{Mpi, ReduceOp, Request};

let mpi = Mpi::init()?;
let world = mpi.world();

let send = vec![1.0f64; 1000];
let mut recv = vec![0.0f64; 1000];

// Start nonblocking operation
let request = world.iallreduce(&send, &mut recv, ReduceOp::Sum)?;

// Do other work while communication proceeds...
expensive_computation();

// Wait for completion
request.wait()?;
// recv now contains the result
```

### Persistent Collectives (MPI 4.0+)

```rust
use ferrompi::{Mpi, ReduceOp};

let mpi = Mpi::init()?;
let world = mpi.world();

// Buffer used for all iterations
let mut data = vec![0.0f64; 1000];

// Initialize ONCE
let mut persistent = world.bcast_init(&mut data, 0)?;

// Use MANY times — amortizes setup cost!
for iter in 0..10000 {
    if world.rank() == 0 {
        data.fill(iter as f64);
    }

    persistent.start()?;
    persistent.wait()?;

    // data contains broadcast result on all ranks
}
// Cleanup on drop
```

### Point-to-Point Communication

```rust
use ferrompi::Mpi;

let mpi = Mpi::init()?;
let world = mpi.world();

if world.rank() == 0 {
    let data = vec![1.0f64, 2.0, 3.0];
    world.send(&data, 1, 0)?;
} else if world.rank() == 1 {
    let mut buf = vec![0.0f64; 3];
    let (source, tag, count) = world.recv(&mut buf, 0, 0)?;
    println!("Received {:?} from rank {}", buf, source);
}
```

### Available Examples

Run examples with `mpiexec`:

```bash
cargo build --release --examples
cargo build --release --examples --features rma

# Core examples
mpiexec -n 4 ./target/release/examples/hello_world
mpiexec -n 4 ./target/release/examples/ring
mpiexec -n 4 ./target/release/examples/allreduce
mpiexec -n 4 ./target/release/examples/nonblocking
mpiexec -n 4 ./target/release/examples/persistent_bcast
mpiexec -n 4 ./target/release/examples/pi_monte_carlo

# Communicator management
mpiexec -n 4 ./target/release/examples/comm_split

# Scan and variable-length collectives
mpiexec -n 4 ./target/release/examples/scan
mpiexec -n 4 ./target/release/examples/gatherv

# Shared memory (requires --features rma)
mpiexec -n 4 ./target/release/examples/shared_memory

# Hybrid MPI+threads
mpiexec -n 2 ./target/release/examples/hybrid_openmp
```

| Example            | Description                                  | Feature |
| ------------------ | -------------------------------------------- | ------- |
| `hello_world`      | Basic MPI initialization and rank/size query | —       |
| `ring`             | Point-to-point ring communication pattern    | —       |
| `allreduce`        | Blocking and nonblocking allreduce           | —       |
| `nonblocking`      | Nonblocking collective operations            | —       |
| `persistent_bcast` | Persistent broadcast (MPI 4.0+)              | —       |
| `pi_monte_carlo`   | Monte Carlo Pi estimation with reduce        | —       |
| `comm_split`       | Communicator splitting and management        | —       |
| `scan`             | Prefix scan and exclusive scan operations    | —       |
| `gatherv`          | Variable-length gather (gatherv)             | —       |
| `shared_memory`    | Shared memory windows with RAII lock guards  | `rma`   |
| `hybrid_openmp`    | Hybrid MPI + threads with thread-level init  | —       |

## API Reference

### Core Types

| Type                | Description                            |
| ------------------- | -------------------------------------- |
| `Mpi`               | MPI environment handle (init/finalize) |
| `Communicator`      | MPI communicator wrapper               |
| `Request`           | Nonblocking operation handle           |
| `PersistentRequest` | Persistent operation handle (MPI 4.0+) |
| `MpiDatatype`       | Trait for types usable in MPI ops      |
| `Status`            | Message status (source, tag, count)    |
| `Info`              | MPI_Info object with RAII              |
| `SharedWindow<T>`   | Shared memory window (feature: `rma`)  |
| `LockGuard`         | RAII window lock (feature: `rma`)      |
| `LockAllGuard`      | RAII window lock-all (feature: `rma`)  |

### Collective Operations

| Operation            | Blocking               | Nonblocking             | Persistent                  |
| -------------------- | ---------------------- | ----------------------- | --------------------------- |
| Broadcast            | `broadcast`            | `ibroadcast`            | `bcast_init`                |
| Reduce               | `reduce`               | `ireduce`               | `reduce_init`               |
| Allreduce            | `allreduce`            | `iallreduce`            | `allreduce_init`            |
| Gather               | `gather`               | `igather`               | `gather_init`               |
| Allgather            | `allgather`            | `iallgather`            | `allgather_init`            |
| Scatter              | `scatter`              | `iscatter`              | `scatter_init`              |
| Alltoall             | `alltoall`             | `ialltoall`             | `alltoall_init`             |
| Scan                 | `scan`                 | `iscan`                 | `scan_init`                 |
| Exscan               | `exscan`               | `iexscan`               | `exscan_init`               |
| Reduce-scatter-block | `reduce_scatter_block` | `ireduce_scatter_block` | `reduce_scatter_block_init` |
| Barrier              | `barrier`              | `ibarrier`              | —                           |

Additional scalar and in-place variants:

| Variant                  | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `reduce_scalar`          | Reduce a single value (returns scalar on root)    |
| `reduce_inplace`         | In-place reduce (root's buffer is both send/recv) |
| `allreduce_scalar`       | Allreduce a single value (returns scalar)         |
| `allreduce_inplace`      | In-place allreduce                                |
| `allreduce_init_inplace` | Persistent in-place allreduce                     |
| `scan_scalar`            | Prefix scan on a single value                     |
| `exscan_scalar`          | Exclusive prefix scan on a single value           |

Variable-length (V-variant) collectives:

| Operation  | Blocking     | Nonblocking   | Persistent        |
| ---------- | ------------ | ------------- | ----------------- |
| Gatherv    | `gatherv`    | `igatherv`    | `gatherv_init`    |
| Scatterv   | `scatterv`   | `iscatterv`   | `scatterv_init`   |
| Allgatherv | `allgatherv` | `iallgatherv` | `allgatherv_init` |
| Alltoallv  | `alltoallv`  | `ialltoallv`  | `alltoallv_init`  |

### Point-to-Point Operations

| Operation  | Description                                   |
| ---------- | --------------------------------------------- |
| `send`     | Blocking send                                 |
| `recv`     | Blocking receive (returns source, tag, count) |
| `isend`    | Nonblocking send (returns `Request`)          |
| `irecv`    | Nonblocking receive (returns `Request`)       |
| `sendrecv` | Simultaneous send and receive                 |
| `probe`    | Blocking probe (returns `Status`)             |
| `iprobe`   | Nonblocking probe (returns `Option<Status>`)  |

### Reduction Operations

```rust
pub enum ReduceOp {
    Sum,   // MPI_SUM
    Max,   // MPI_MAX
    Min,   // MPI_MIN
    Prod,  // MPI_PROD
}
```

## Thread Safety

`Communicator` is `Send + Sync`, enabling hybrid MPI + threads programs where MPI handles inter-node communication and threads (via `std::thread`, Rayon, or OpenMP) handle intra-node parallelism.

The thread-safety guarantee depends on the level requested at initialization:

| Thread Level | Who can call MPI | Use case                        |
| ------------ | ---------------- | ------------------------------- |
| `Single`     | Main thread only | Pure MPI, no threads            |
| `Funneled`   | Main thread only | Threads compute, main calls MPI |
| `Serialized` | Any thread       | User serializes MPI calls       |
| `Multiple`   | Any thread       | Full concurrent MPI access      |

```rust
use ferrompi::{Mpi, ThreadLevel, ReduceOp};

// Request funneled support for hybrid MPI + threads
let mpi = Mpi::init_thread(ThreadLevel::Funneled)?;
assert!(mpi.thread_level() >= ThreadLevel::Funneled);

let world = mpi.world();
// Worker threads compute locally, main thread calls MPI
let local = 42.0_f64;
let global = world.allreduce_scalar(local, ReduceOp::Sum)?;
```

See `examples/hybrid_openmp.rs` for a complete hybrid MPI + threads pattern.

## SLURM Configuration

The `numa` feature flag enables the `slurm` module with helpers for reading SLURM job topology at runtime. These functions return `None` when not running under SLURM.

```toml
[dependencies]
ferrompi = { version = "0.2", features = ["numa"] }
```

| Function          | SLURM Variable          | Description                     |
| ----------------- | ----------------------- | ------------------------------- |
| `is_slurm_job()`  | `SLURM_JOB_ID`          | Check if running under SLURM    |
| `job_id()`        | `SLURM_JOB_ID`          | Unique job identifier           |
| `local_rank()`    | `SLURM_LOCALID`         | Task ID relative to this node   |
| `local_size()`    | `SLURM_NTASKS_PER_NODE` | Number of tasks on this node    |
| `num_nodes()`     | `SLURM_NNODES`          | Total number of allocated nodes |
| `cpus_per_task()` | `SLURM_CPUS_PER_TASK`   | CPUs allocated per task         |
| `node_name()`     | `SLURM_NODENAME`        | Name of this compute node       |
| `node_list()`     | `SLURM_NODELIST`        | Compact list of allocated nodes |

Example SLURM batch script for hybrid MPI + threads:

```bash
#!/bin/bash
#SBATCH --ntasks-per-node=4        # MPI ranks per node
#SBATCH --cpus-per-task=8          # threads per rank
#SBATCH --bind-to core
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./target/release/my_program
```

## RMA / Shared Memory Windows

The `rma` feature flag enables `SharedWindow<T>`, a safe wrapper around `MPI_Win_allocate_shared` with RAII lifecycle management. Shared memory windows allow processes on the same node to directly access each other's memory without message passing.

```toml
[dependencies]
ferrompi = { version = "0.2", features = ["rma"] }
```

```rust
use ferrompi::{Mpi, SharedWindow, LockType};

let mpi = Mpi::init()?;
let world = mpi.world();
let node = world.split_shared()?;

// Each process allocates 100 f64s in shared memory
let mut win = SharedWindow::<f64>::allocate(&node, 100)?;

// Write to local portion
{
    let local = win.local_slice_mut();
    for (i, x) in local.iter_mut().enumerate() {
        *x = (node.rank() * 100 + i as i32) as f64;
    }
}

// Fence synchronization — all processes participate
win.fence()?;

// Read from any rank's memory (zero-copy!)
let remote = win.remote_slice(0)?;
println!("Rank 0's first value: {}", remote[0]);
```

Synchronization modes:

- **Active target** (`fence`): Bulk-synchronous, all processes participate
- **Passive target** (`lock` / `lock_all`): Fine-grained one-sided access with RAII guards

See `examples/shared_memory.rs` for a complete shared memory example.

## Running Tests

```bash
# Unit tests (no MPI required)
cargo test
cargo test --features numa

# MPI integration tests (requires mpiexec)
./tests/run_mpi_tests.sh               # Default features
./tests/run_mpi_tests.sh rma           # With RMA/shared memory tests
./tests/run_mpi_tests.sh numa          # With NUMA features (implies rma)
MPI_NP=8 ./tests/run_mpi_tests.sh      # Custom process count

# Build and run individual examples
cargo build --release --examples
mpiexec -n 4 ./target/release/examples/hello_world
```

## Configuration

### Environment Variables

| Variable         | Description           | Example                     |
| ---------------- | --------------------- | --------------------------- |
| `MPI_PKG_CONFIG` | pkg-config name       | `mpich`, `ompi`             |
| `MPICC`          | MPI compiler wrapper  | `/opt/mpich/bin/mpicc`      |
| `CRAY_MPICH_DIR` | Cray MPI installation | `/opt/cray/pe/mpich/8.1.25` |

### Build Configuration

FerroMPI automatically detects MPI installations via:

1. `MPI_PKG_CONFIG` environment variable
2. pkg-config (`mpich`, `ompi`, `mpi`)
3. `mpicc -show` output
4. `CRAY_MPICH_DIR` (for Cray systems)
5. Common installation paths

## Troubleshooting

### "Could not find MPI installation"

```bash
# Check if MPI is installed
which mpiexec
mpiexec --version

# Set pkg-config name explicitly
export MPI_PKG_CONFIG=mpich
cargo build
```

### "Persistent collectives not available"

Persistent collectives require MPI 4.0+. Check your MPI version:

```bash
mpiexec --version
# MPICH Version: 4.2.0  ✓
# Open MPI 5.0.0        ✓
# MPICH Version: 3.4.2  ✗ (too old)
```

### macOS linking issues

```bash
export DYLD_LIBRARY_PATH=$(brew --prefix mpich)/lib:$DYLD_LIBRARY_PATH
```

## Architecture

```
┌─────────────────────────┐
│    Rust Application     │
├─────────────────────────┤
│  ferrompi (Safe Rust)   │
├─────────────────────────┤
│     ffi.rs (bindings)   │
├─────────────────────────┤
│   ferrompi.c (C layer)  │  ← ~2400 lines
├─────────────────────────┤
│   MPICH / OpenMPI       │
└─────────────────────────┘
```

The C layer provides:

- Handle tables for MPI opaque objects (256 comms, 16384 requests, 256 windows, 64 infos)
- Automatic large-count operation selection
- Request management
- Graceful degradation for MPI <4.0

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Contributing

Contributions welcome! Please ensure:

- All examples pass with `mpiexec -n 4`
- New features include tests and documentation
- Code follows Rust style guidelines (`cargo fmt`, `cargo clippy`)

## Acknowledgments

FerroMPI was inspired by:

- [rsmpi](https://github.com/rsmpi/rsmpi) - Comprehensive MPI bindings for Rust
- The MPI Forum for the excellent MPI 4.0 specification
