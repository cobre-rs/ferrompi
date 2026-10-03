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

**Seventeen cases** (13 at `funneled`, see Thread level), in run order:
`A/A direct iallreduce+wait` (noise floor), then the blocking collectives
`allreduce f64 sum`, `allreduce u64 bor`, `allgatherv u8 x64`, `broadcast f64` and
`barrier`, then `isend+irecv+wait`, `8x(isend+irecv)+waitall`, `iallreduce+wait`,
`persistent start+wait`, and `8x persistent start_all+wait_all`, then the six group-query
cases `group size T=1`, `group rank T=1`, `group size T=4`, `group size T=8`,
`group rank T=4` and `group rank T=8`.

The group-query cases compare `MPI_Group_size` / `MPI_Group_rank` with `Group::size` /
`Group::rank` on one shared group, 20 000 calls per thread. `T=n` is the thread count:
each round runs one arm on `n` scoped threads released together by a barrier, and the
round's value is the median of the threads' ns/call. The `T=4` and `T=8` cases run only at
`multiple`: at `funneled` a thread other than the initializing one cannot call MPI through
ferrompi. The `T=8` cases are meant to run pinned, for example under `taskset -c 0-7`.

Y and Z, the thread-level overhead figures, are read from the group cases:

- Y is the T=1 delta at multiple minus the same case's multiple delta recorded for 0.6.0, with the sum of both sessions' A/A deltas as tolerance; it is never a same-session multiple-minus-funneled difference.
- Z is the T=8 ferrompi per-call cost against T=1.

**Thread level.** The environment variable `FERROMPI_BENCH_LEVEL` selects the level MPI
is initialized with: `funneled` (the default when unset) or `multiple`. Any other value
panics with `FERROMPI_BENCH_LEVEL must be funneled or multiple, got <value>`. If the
library grants a different level than requested, the bench prints
`ffi_overhead: <level> not provided; skipped` and exits 0. The eleven original cases and
the two `T=1` group cases run at either level, 13 case lines at `funneled`; `multiple`
adds the four `T=4` and `T=8` group cases, 17 case lines.

```
FERROMPI_BENCH_LEVEL=multiple mpiexec -n 1 target/release/deps/ffi_overhead-<hash>
```

**Output.** A header line, `# <library line>; level <Level>; 21 interleaved rounds per
arm; median ns per call`, then one line per case:
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

**Blocking and thread-level results.** The 0.6.0 library on local MPICH 4.2.3 (`ch4:ofi`),
run as a singleton pinned with `taskset -c 0-7`: three runs per level, taken alternately at
`funneled` and `multiple`. Values are the median of the three runs' deltas, ferrompi minus
direct, in ns per call, for the 13 cases at `funneled` and the 17 at `multiple`; `n/a` marks
a case the level does not run.

| Case | funneled | multiple |
|---|---|---|
| A/A direct iallreduce+wait | +0.3 | −0.8 |
| allreduce f64 sum | +1.7 | +1.4 |
| allreduce u64 bor | +1.5 | +1.4 |
| allgatherv u8 x64 | +3.6 | +3.4 |
| broadcast f64 | +0.8 | +1.0 |
| barrier | +1.2 | +1.2 |
| isend+irecv+wait | +36.9 | +38.8 |
| 8x(isend+irecv)+waitall | +359.0 | +374.4 |
| iallreduce+wait | +12.2 | +12.3 |
| persistent start+wait | +8.6 | +5.6 |
| 8x persistent start_all+wait_all | +135.7 | +124.9 |
| group size T=1 | +2.0 | +1.8 |
| group rank T=1 | +2.0 | +2.0 |
| group size T=4 | n/a | +1.8 |
| group size T=8 | n/a | +3.3 |
| group rank T=4 | n/a | +2.0 |
| group rank T=8 | n/a | +3.0 |

The `A/A direct iallreduce+wait` delta of the three runs was +0.4, +0.2 and +0.3 ns at
`funneled` and −0.1, −0.8 and −1.3 ns at `multiple`. A run whose A/A delta exceeded 2 ns in
absolute value would have been re-run; none was. On the measuring machine (an i7-12700KF)
CPUs 0-7 are four physical cores with two hardware threads each, so the `T=8` cases run two
threads per core.

### 0.7 in-flight accounting and scopes

The in-flight counter and the nonblocking scope, measured on 2026-10-03 against three
builds in one session: local MPICH 4.2.3 (`ch4:ofi`), a singleton, on an i7-12700KF.

- Z is the 0.6.0 behaviour (`2802bf9`).
- E is the build before the counter and the scope landed (`030b3a5`).
- H is the build measured (`6e834be`).

Each set runs the binaries as Z H H Z four times, then E H H E twice, so it holds 8 slots
of Z, 12 of H and 4 of E, and the medians below are over those slots. Values are the
median of each binary's deltas, ferrompi minus direct, in ns per call. A slot whose
`A/A direct iallreduce+wait` delta exceeded 2 ns in absolute value would have been
re-run; none was.

The budgets:

- B: a blocking collective at `funneled` costs at most 1 ns more than at 0.6.0. Those arms
  hold no scope, so B also shows the in-flight count stays off the blocking path.
- X: a nonblocking scope costs at most 5 ns per request, with no heap allocation per
  iteration while 64 or fewer requests are in flight (`examples/test_scope_alloc.rs`).
- Y: one enter/exit pair of the in-flight counter at `multiple`, uncontended, costs at
  most 15 ns. It is the `multiple` delta at H minus the `multiple` delta of the 0.6.0
  reference, never a `multiple` minus `funneled` difference in one session.
- Z: the same call at 8 threads costs at most 30 ns per call, twice Y.

A budget is met when the increase is at most the budget plus the A/A allowance. For B and
X the allowance is the largest |A/A| of the set, since both sides are measured together.
For Y and Z it is the sum of the largest |A/A| of each side's slots in the set. The
`group rank` reference is E, not Z: `Group::rank` returns an `Option<i32>` and costs
1.4-1.7 ns more than at 0.6.0.

| Budget | Case | Level, pinning | Reference | Reference Δ | Head Δ | Increase | Limit | Verdict |
|---|---|---|---|---|---|---|---|---|
| B | allreduce f64 sum | funneled, `taskset -c 0-7` | Z | +1.3 | +1.9 | +0.6 | ≤ 1 + 1.8 = 2.8 | met |
| B | allreduce u64 bor | funneled, `taskset -c 0-7` | Z | +1.3 | +2.1 | +0.8 | ≤ 1 + 1.8 = 2.8 | met |
| B | allgatherv u8 x64 | funneled, `taskset -c 0-7` | Z | +3.7 | +4.7 | +1.0 | ≤ 1 + 1.8 = 2.8 | met |
| B | broadcast f64 | funneled, `taskset -c 0-7` | Z | +1.0 | +1.2 | +0.2 | ≤ 1 + 1.8 = 2.8 | met |
| B | barrier | funneled, `taskset -c 0-7` | Z | +1.2 | +1.05 | −0.15 | ≤ 1 + 1.8 = 2.8 | met |
| X | 8x(isend+irecv)+waitall | funneled, `taskset -c 0-7` | E | +362.95 | +384.95 | +22.0 (1.38 per request over 16) | ≤ 16 × 5 + 1.8 = 81.8 | met |
| X | iallreduce+wait | funneled, `taskset -c 0-7` | E | +12.35 | +16.55 | +4.2 | ≤ 5 + 1.8 = 6.8 | met |
| Y | group size T=1 | multiple, `taskset -c 0-7` | Z | +1.9 | +10.7 | +8.8 | ≤ 15 + 0.9 + 0.5 = 16.4 | met |
| Y | group rank T=1 | multiple, `taskset -c 0-7` | E | +3.9 | +13.6 | +9.7 | ≤ 15 + 0.1 + 0.5 = 15.6 | met |
| Z | group size T=8 | multiple, `taskset -c 0,2,4,6,8,10,12,14` | Z | +1.9 | +9.0 | +7.1 | ≤ 30 + 1.6 + 1.1 = 32.7 | met |
| Z | group rank T=8 | multiple, `taskset -c 0,2,4,6,8,10,12,14` | E | +4.1 | +12.3 | +8.2 | ≤ 30 + 0.9 + 1.1 = 32.0 | met |
| context | isend+irecv+wait | funneled, `taskset -c 0-7` | E | +36.15 | +40.4 | +4.25 (2.13 per request over 2) | none | n/a |
| context | persistent start+wait | funneled, `taskset -c 0-7` | E | +9.6 | +10.5 | +0.9 | none | n/a |
| context | 8x persistent start_all+wait_all | funneled, `taskset -c 0-7` | E | +127.65 | +116.6 | −11.05 | none | n/a |
| context | persistent start+wait | multiple, `taskset -c 0-7` | E | +5.4 | +50.8 | +45.4 | none | n/a |
| context | 8x persistent start_all+wait_all | multiple, `taskset -c 0-7` | E | +137.1 | +129.9 | −7.2 | none | n/a |

The context rows have no limit: no budget covers the persistent requests or the
two-request arm.

The largest |A/A| per binary, for Z, E and H, was 0.4, 1.2 and 1.8 ns in the `funneled`
set, 0.9, 0.1 and 0.5 ns in the `multiple` set, and 1.6, 0.9 and 1.1 ns in the set pinned
to one thread per physical core.

The Z rows run pinned to `taskset -c 0,2,4,6,8,10,12,14`, one thread per physical core of
the measuring machine. There the direct call's T=8/T=1 ratio is 1.08 for `group size` and
1.18 for `group rank` (medians over the 24 slots; H alone gives 1.08 and 1.15), so the
call qualifies for the 8-thread budget. Under `taskset -c 0-7`, which puts two threads on
each of four cores, the same ratios are 1.64 and 1.77. The T=8/T=1 ratio of ferrompi at H
is 0.90 for `group size` and 0.96 for `group rank`.

`examples/test_scope_alloc.rs` counts the heap allocations Rust code makes, not MPI's own,
with a counting global allocator. After 10 warm-up iterations, 1000 scopes of 64 requests
each (32 self receives and 32 self sends, completed by `Request::wait_all`) allocate
nothing; one scope of 65 requests does allocate, which marks the 64-slot boundary.

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
