# ferrompi Benchmarks

This directory holds three [Criterion](https://github.com/bheisler/criterion.rs)-adjacent
benchmarks for ferrompi's MPI paths. They measure ferrompi on MPICH, are not packaged with
the crate, and are not run by CI.

## Requirements

- An MPI implementation as `mpiexec` in `PATH`:
  - `persistent_vs_iallreduce` needs persistent collectives (MPICH 4.x or Open MPI 5).
    On Open MPI 4.1 the bench exits with an `allreduce_init needs persistent collectives`
    error.
  - `ffi_overhead`'s direct-MPI arm needs MPICH; on any other library it prints a skip
    line and exits 0.
  - `allreduce_roundtrip` runs on any MPI implementation.
- A stable Rust toolchain.

## Build and run

Build the bench binaries first, then launch the printed executable under `mpiexec`:

```
cargo bench --no-run
```

```
mpiexec -n 2 target/release/deps/allreduce_roundtrip-<hash> --bench [--quick] [--noplot]
mpiexec -n 2 target/release/deps/persistent_vs_iallreduce-<hash> --bench [--quick] [--noplot]
mpiexec -n 1 target/release/deps/ffi_overhead-<hash>
```

`ffi_overhead` can also run as a singleton with `cargo bench --bench ffi_overhead`.

Launching `cargo bench` itself under `mpiexec` is **wrong**: each rank would start its own
`cargo` process (and its own Criterion driver) instead of two ranks of one MPI job running
the same binary, so `cargo bench` must never be the command handed to `mpiexec`.

If Open MPI refuses to oversubscribe cores on a single-host machine, add `--oversubscribe`
to the binary invocation:

```
mpiexec --oversubscribe -n 2 target/release/deps/allreduce_roundtrip-<hash> --bench
```

Only rank 0 constructs a `Criterion` instance and writes output, under
`target/criterion/<group>/`; non-root ranks run a mirror loop and produce no output.

## Bench table

| Binary                     | Measures                                                         | Ranks |
| --------------------------- | ----------------------------------------------------------------- | ----- |
| `allreduce_roundtrip`      | `allreduce` latency/throughput for `f64`, three sizes             | 2+    |
| `persistent_vs_iallreduce` | 100-iteration persistent allreduce vs. `iallreduce`, seven sizes  | 2+    |
| `ffi_overhead`             | Fixed per-call FFI cost, ferrompi vs. direct `MPI_*`               | 1     |

## `allreduce_roundtrip`

Group `allreduce_f64` measures `world.allreduce(&send, &mut recv, ReduceOp::Sum)` for
three `f64` buffer sizes: 2 elements (16 B), 131 072 elements (1 MiB), and 2 097 152
elements (16 MiB).

**Command caveat.** Each `b.iter` sample issues one 16 B `u64` sentinel allreduce (via
`common::lead`) before the measured `f64` allreduce, to keep non-root ranks in lockstep
with rank 0's Criterion driver. At the 16 B size point the sentinel is comparable to the
measured call, so read that point as a relative trend, not an absolute latency.

## `persistent_vs_iallreduce`

Group `iterative_allreduce_100x` sweeps seven `f64` sizes — 8, 64, 512, 4 096, 32 768,
262 144 and 1 048 576 bytes (8 B to 256 KiB in steps of 8x, then 1 MiB) — and compares 100
consecutive iterations of two strategies per size: `persistent` (one `allreduce_init`
outside the loop, then 100x `start`+`wait`) against `iallreduce` (100x fresh
`iallreduce`+`wait`). This produces 14 benchmark ids:
`iterative_allreduce_100x/persistent/<bytes>` and
`iterative_allreduce_100x/iallreduce/<bytes>`.

**SETUP ordering.** Rank 0 sends a `SETUP` command before each size's `allreduce_init`,
and every rank re-initializes its persistent request only in response to that command, so
every `allreduce_init` call is issued in the same collective order on every rank.
MPI-4.1 §7.13 requires this: initialization calls for persistent collective operations are
nonlocal and follow the existing collective-operation ordering rules.

**Measured results.** np 2, 3 full runs each, local MPICH 4.2.3 (`ch4:ofi`, shm) and an
apt MPICH 4.2.1 + UCX container (`UCX_TLS=self,sm,tcp`). ns per allreduce is Criterion's
middle `time:` estimate divided by 100. "Faster by" is reported only when all three
persistent runs are below all three iallreduce runs; otherwise "none".

| Size | MPICH 4.2.3 ch4:ofi (shm): persistent / iallreduce | faster by | MPICH 4.2.1 + UCX (`self,sm,tcp`) | faster by |
|---|---|---|---|---|
| 8 B | 305 / 499 ns | 39 % | 239 / 564 ns | 58 % |
| 64 B | 323 / 507 ns | 36 % | 247 / 572 ns | 57 % |
| 512 B | 403 / 586 ns | 31 % | 369 / 711 ns | 48 % |
| 4 KiB | 1.14 / 1.43 µs | 20 % | 1.08 / 1.51 µs | 28 % |
| 32 KiB | 5.94 / 6.17 µs | none | 6.10 / 6.60 µs | 8 % |
| 256 KiB | 28.21 / 28.47 µs | none | 26.49 / 27.59 µs | none |
| 1 MiB | 148.75 / 148.61 µs | none | 113.23 / 114.39 µs | none |

On MPICH at 2 ranks, a persistent allreduce was 20–58 % faster per call than
`iallreduce` up to 4 KiB, at most 8 % faster at 32 KiB, and no faster from 256 KiB. The
table above shows the 8 % separation at 32 KiB holds only over UCX; the local `ch4:ofi`
run shows no separation at that size.

The bench also runs on Open MPI 5, but its regime there is unmeasured: the table above
was measured on MPICH only. Open MPI 4.1 lacks the MPI 4.0 `MPI_*_init` entry points, so
ferrompi does not enable persistent collectives there.

This benchmark is not a pass/fail gate — the reported numbers are inspected by a human.

## `ffi_overhead`

A plain, `harness = false` program run at one rank. It alternates 21 interleaved rounds
of a direct `MPI_*` call and the matching ferrompi call per case (ABBA order: direct then
ferrompi on even rounds, the reverse on odd rounds), after a warm-up round of each arm,
and reports each arm's median ns/call plus their delta. The first case,
`A/A direct iallreduce+wait`, runs the direct call on both arms to give the noise floor
the other cases' deltas are judged against.

**Six cases:** `A/A direct iallreduce+wait` (noise floor), `isend+irecv+wait`,
`8x(isend+irecv)+waitall`, `iallreduce+wait`, `persistent start+wait`, and
`8x persistent start_all+wait_all`.

**Output.** A header line, `# <library line>; 21 interleaved rounds per arm; median ns
per call`, then one line per case:
`<case> direct X ns   ferrompi Y ns   delta ±Z ns`.

**MPICH-only direct arm.** The direct arm declares MPICH's integer handle values and
calls only MPI-1/MPI-3 symbols, so the binary links on every MPI implementation. At run
time it checks `Mpi::library_version()`; on any library other than MPICH it prints
`ffi_overhead: the direct arm needs MPICH's handle values; skipped on <line>` and exits 0.

**Measured results.** `mpiexec -n 1`, 7 runs. Values are the median delta, ferrompi minus
direct, in ns per call:

| Case | MPICH 4.2.3 ch4:ofi | MPICH 4.2.1 + UCX |
|---|---|---|
| A/A iallreduce+wait (noise floor) | −0.1 | +0.2 |
| isend+irecv+wait | +37.5 | +37.5 |
| 8x(isend+irecv)+waitall | +387.9 | +402.1 |
| iallreduce+wait | +12.2 | +15.5 |
| persistent start+wait | +10.1 | +6.4 |
| 8x persistent start_all+wait_all | +113.6 | +103.6 |

The noise floor ranged from 3.5 ns locally to 0.9 ns under UCX, and every other case's
minimum delta cleared its floor on both. On Open MPI 4.1.6 and 5.0.7 the bench prints the
skip line.

## Design notes

- `criterion_main!` is intentionally **not** used. That macro defines its own `fn main`
  which calls `Criterion::default()` before `Mpi::init()` can run. Collective operations
  in subsequent benchmarks would then deadlock on non-root ranks because MPI was never
  initialized on them.
- Criterion's `rayon` feature is disabled (`default-features = false`). Rayon worker
  threads calling MPI without `MPI_THREAD_MULTIPLE` will abort.
- `benches/common` provides `init_mpi_for_bench` and the `lead`/`follow`/`STOP` command
  protocol: rank 0 sends a command via `lead` before each measured call, and non-root
  ranks run `follow`'s loop to stay in step with rank 0's Criterion driver until `STOP`.
