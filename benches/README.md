# ferrompi Benchmarks

This directory contains [Criterion](https://github.com/bheisler/criterion.rs) benchmarks
for ferrompi. Because every benchmark may call MPI collective operations, all benches
**must** be launched via `mpiexec` — running `cargo bench` directly will hang or abort.

## Prerequisites

- MPICH 4.0+ or Open MPI 5.0+ installed and available as `mpiexec` in `PATH`.
- Rust toolchain (stable).

## Running the benchmarks

```
mpiexec -n 2 cargo bench --bench bench_smoke
```

Replace `bench_smoke` with the name of the benchmark binary you want to run
(e.g. `allreduce_roundtrip`, `persistent_vs_iallreduce`, `ffi_overhead`).

To run a quick measurement with fewer samples (useful during development):

```
mpiexec -n 2 cargo bench --bench bench_smoke -- --quick
```

To run all registered bench binaries in one pass:

```
mpiexec -n 2 cargo bench
```

### Open MPI note

Open MPI may refuse to oversubscribe cores on a single-host machine.
If you see `There are not enough slots available`, add `--oversubscribe`:

```
mpiexec --oversubscribe -n 2 cargo bench --bench bench_smoke
```

## Output directories

| Rank                   | Location                       |
| ---------------------- | ------------------------------ |
| Rank 0 (authoritative) | `target/criterion/`            |
| Rank N (N > 0)         | `/tmp/ferrompi-bench-rank<N>/` |

Only rank 0 produces the full Criterion HTML report. Output from other ranks is
written to `/tmp/` so it does not interfere with rank 0's results and is not
committed to version control.

To inspect the HTML report after a run:

```
xdg-open target/criterion/report/index.html   # Linux
open target/criterion/report/index.html        # macOS
```

Individual benchmark reports are at `target/criterion/<bench-name>/report/index.html`,
for example `target/criterion/noop/report/index.html`.

## Benchmark list

| Binary                     | Measures                                                  |
| -------------------------- | --------------------------------------------------------- |
| `bench_smoke`              | No-op (harness correctness / compile check)               |
| `allreduce_roundtrip`      | `allreduce` latency/throughput for f64                    |
| `persistent_vs_iallreduce` | 100-iteration persistent allreduce vs iallreduce at 1 MiB |
| `ffi_overhead`             | Fixed per-call FFI cost: field reads vs MPI collectives   |

### Allreduce roundtrip

Measures `world.allreduce(&send, &mut recv, ReduceOp::Sum)` for three `f64`
buffer sizes:

| Size label | Elements  | Bytes  |
| ---------- | --------- | ------ |
| 2          | 2         | 16 B   |
| 131072     | 131 072   | 1 MiB  |
| 2097152    | 2 097 152 | 16 MiB |

Run the benchmark:

```
mpiexec -n 2 cargo bench --bench allreduce_roundtrip
```

For a quick measurement with fewer samples (useful during development):

```
mpiexec -n 2 cargo bench --bench allreduce_roundtrip -- --quick
```

Criterion reports land in `target/criterion/allreduce_f64/`. Each of the three
size points produces a subdirectory (e.g. `target/criterion/allreduce_f64/2/`,
`target/criterion/allreduce_f64/131072/`,
`target/criterion/allreduce_f64/2097152/`) containing `report/index.html` and
`new/estimates.json`.

The bench asserts `world.size() >= 2` at startup and panics with a clear
message if launched with fewer than two ranks.

**Measurement caveat.** Each `b.iter` sample issues one 16 B `u64` sentinel
`allreduce` before the measured `f64` `allreduce`, so non-root ranks stay in
lockstep with rank 0's Criterion driver. At `n = 2` (16 B) the sentinel adds
roughly 50% to the reported time — read that size point as a relative-trend
indicator, not absolute single-call latency. At 1 MiB and 16 MiB the sentinel
is < 0.01% of the data payload and effectively invisible.

### Persistent vs iallreduce

Measures the cost of 100 consecutive `allreduce` operations at 1 MiB (131 072
`f64` elements) using two strategies side by side:

| Benchmark    | Strategy                                                      |
| ------------ | ------------------------------------------------------------- |
| `persistent` | One `allreduce_init` outside the loop; 100 × `start` + `wait` |
| `iallreduce` | 100 × `iallreduce` + `wait` (fresh `Request` each iteration)  |

Run the benchmark:

```
mpiexec -n 2 cargo bench --bench persistent_vs_iallreduce
```

For a quick measurement with fewer samples (useful during development):

```
mpiexec -n 2 cargo bench --bench persistent_vs_iallreduce -- --quick
```

Criterion reports land in `target/criterion/iterative_allreduce_1mib_100x/`.
Two subdirectories are produced:

- `target/criterion/iterative_allreduce_1mib_100x/persistent/`
- `target/criterion/iterative_allreduce_1mib_100x/iallreduce/`

Each contains `report/index.html` and `new/estimates.json`.
A side-by-side comparison is available at
`target/criterion/iterative_allreduce_1mib_100x/report/index.html`.

Per MPICH 4.2.3, the expected outcome is that `persistent` runs faster than
`iallreduce` by roughly 10-30% for iterative reduction workloads, because
persistent collectives amortize MPI setup cost across many iterations. The
exact speedup is MPI-implementation-dependent and hardware-dependent. This
benchmark is not a pass/fail gate — the reported numbers are inspected by a
human to validate (or refute) the README claim.

The bench asserts `world.size() >= 2` at startup and panics with a clear
message if launched with fewer than two ranks.

### FFI overhead

Measures the fixed per-call cost at the ferrompi/MPI FFI boundary for five
representative operations:

| Bench             | What it measures                                         |
| ----------------- | -------------------------------------------------------- |
| `rank_cached`     | `world.rank()` — cached field read, zero FFI calls       |
| `size_cached`     | `world.size()` — cached field read, zero FFI calls       |
| `barrier`         | `MPI_Barrier` — one FFI round-trip per call              |
| `broadcast_1elem` | `MPI_Bcast` on 1 × f64 — smallest possible bcast payload |
| `allreduce_1elem` | `MPI_Allreduce` on 1 × f64 — smallest possible payload   |

`rank_cached` and `size_cached` are expected to be nanosecond-scale because
they are field reads that never enter the MPI library (epic-02 ticket-007
cached `rank` and `size` as `pub(crate) i32` fields on `Communicator`).
`barrier` and the single-element collectives expose the fixed FFI + MPI
dispatch cost independent of payload size.

Ticket-013 uses the numbers produced by this benchmark to decide which FFI
trampolines warrant `#[inline]`.

Run the benchmark:

```
mpiexec -n 2 cargo bench --bench ffi_overhead
```

For a quick measurement with fewer samples (useful during development):

```
mpiexec -n 2 cargo bench --bench ffi_overhead -- --quick
```

Criterion reports land in `target/criterion/ffi_overhead/`. Five subdirectories
are produced:

- `target/criterion/ffi_overhead/rank_cached/`
- `target/criterion/ffi_overhead/size_cached/`
- `target/criterion/ffi_overhead/barrier/`
- `target/criterion/ffi_overhead/broadcast_1elem/`
- `target/criterion/ffi_overhead/allreduce_1elem/`

Each contains `report/index.html` and `new/estimates.json`.

The bench asserts `world.size() >= 2` at startup and panics with a clear
message if launched with fewer than two ranks.

**Measurement caveat.** `rank_cached` and `size_cached` are pure local reads
and their numbers are accurate. The three collective benches (`barrier`,
`broadcast_1elem`, `allreduce_1elem`) each prepend a coordinating 8 B
sentinel `allreduce` inside `b.iter` so non-root ranks stay in lockstep with
rank 0's Criterion driver. Because the sentinel's payload is comparable to
the measured operations, each sample reports roughly `latency(sentinel) +
latency(target_op)` — the `allreduce_1elem` number in particular is close to
**twice** the bare single-call cost. Use these numbers as _relative_ trend
indicators between the three collective points (e.g. ticket-013 inlining
decisions) rather than absolute single-op latencies.

## Design notes

- `criterion_main!` is intentionally **not** used. That macro defines its own
  `fn main` which calls `Criterion::default()` before `Mpi::init()` can run.
  Collective operations in subsequent benchmarks would then deadlock on non-root
  ranks because MPI was never initialized on them.
- Criterion's `rayon` feature is disabled (`default-features = false`). Rayon
  worker threads calling MPI without `MPI_THREAD_MULTIPLE` will abort.
- The shared helper `benches/common/mod.rs` is re-exported via `mod common;`
  in each bench file. It handles MPI initialization and `CRITERION_HOME`
  redirection for non-root ranks.
