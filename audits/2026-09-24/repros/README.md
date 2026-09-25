# Repro programs: 2026-09-24 assessment evidence

These programs are the evidence behind the findings in `../findings/`. Each one shows a defect in ferrompi v0.5.0 (commit `755497b`), or confirms that a suspected problem is not real (rows marked `VER-*`).

They are meant to become regression tests during the 0.5.x plan. Once a finding is fixed and has its own test in `examples/` or `src/`, delete the matching repro.

> **Warning.** Many of these programs deliberately corrupt the heap, write outside buffer bounds, hang, or abort.
> - Run them only by hand, one at a time.
> - Wrap anything that can hang in `timeout`.
> - Never run them in CI as-is.

## Prerequisites

- **MPI:** results were collected with MPICH 4.2.3 installed at `/opt/mpich` (ch4:ofi, one node, x86_64 Fedora).
- **Library path:** `build.rs` does not embed an RPATH, so export the MPI paths first:
  ```bash
  export PATH=/opt/mpich/bin:$PATH
  export LD_LIBRARY_PATH=/opt/mpich/lib:$LD_LIBRARY_PATH
  ```
- **Tools:** `valgrind` for the rows that say so, and `clang` for the cross-target layout probe.
- **Perf crate:** `perf/` hard-codes MPICH's integer handle values in `perf/src/lib.rs`, so it only works with MPICH.

## Building

Every crate is a standalone Cargo package with its own empty `[workspace]`. Each depends on the ferrompi working tree through a relative path (`../../../..`), so it always builds against the current code.

```bash
cd audits/2026-09-24/repros/<crate>      # soundness | c-shim | requests | win-sync | perf
cargo build --release
mpiexec -n <N> target/release/<program>
```

- `soundness`, `c-shim`, `win-sync` and `perf` enable `ferrompi/rma`. `requests` uses default features.
- Some fixes will make a repro stop compiling on purpose. The SND-01, SND-02, SND-05 and SND-08 (case B) repros rely on APIs that the fix removes. A compile error is then the expected "after fix" result, and the repro should be replaced by a `compile_fail` doctest.

Building the C programs:

```bash
mpicc c-shim/c/errhandler_group_incl.c -o eh   && mpiexec -n 1 ./eh
mpicc c-shim/c/pair_type_extents.c    -o ext  && mpiexec -n 1 ./ext
mpicc c-shim/c/gfree.c                -o gfree && mpiexec -n 1 ./gfree
clang --target=aarch64-apple-darwin -c c-shim/c/ldi.c -o ldi.o && llvm-nm -S ldi.o   # symbol sizes = sizeof/_Alignof
cc -std=gnu11 -O2 -pthread perf/c/request_table_bench.c -o tab          # current bitmap table
cc -std=gnu11 -O2 -pthread -DPLAIN perf/c/request_table_bench.c -o tabp # plain stores (single-thread levels)
cc -std=gnu11 -O2 -pthread -DDENSE perf/c/request_table_bench.c -o tabd # pre-v0.4 dense CAS table
taskset -c 0-7 ./tab 1   # arg = thread count; pin to performance cores
```

Building the non_exhaustive cast check:

```bash
rustc --edition 2021 --crate-type lib --crate-name lib_ne api-checks/lib_ne.rs --out-dir /tmp/ne
rustc --edition 2021 api-checks/main_ne.rs --extern lib_ne=/tmp/ne/liblib_ne.rlib -o /tmp/ne/main_ne && /tmp/ne/main_ne
```

## Program index

Finding IDs refer to `../findings/`. In the "How to run" column, `np` is the `mpiexec -n` value, and "valgrind" means the finding was confirmed by running `mpiexec -n <np> valgrind ./prog`.

### soundness/: safe code that causes undefined behaviour

| Program | Finding | What it does | How to run | Observed on v0.5.0 | Expected after fix |
|---|---|---|---|---|---|
| `soundness/src/bin/r1_irecv_uaf.rs` | SND-01 | `irecv` into a `Vec` that is dropped while the receive is pending, then allocates a same-size victim `Vec` that reuses the freed memory | np=2 | `victim bytes overwritten by MPI = 256 / 256 (first=0x55)`: MPI writes into the reused allocation | Does not compile: the request or scope keeps the buffer borrowed |
| `soundness/src/bin/r2_allgather_overflow.rs` | SND-06, SND-07 | `allgather`, `gather` and `gatherv` into a 1-element `recv` placed inside a `#[repr(C)]` frame with canary words | np=1 | Canary words `0xc0ffee` overwritten with `0xdead` for all three calls | Each call returns `Err(InvalidBuffer…)` before reaching FFI; canaries intact |
| `soundness/src/bin/r3_vcoll_short_counts.rs` | SND-07 | `allgatherv` with empty `counts`/`displs`; this passes the only check (`len()==len()`) | np=2 | SIGSEGV: MPI reads `size()` entries from dangling slices | `Err` (the count and displacement arrays must have one entry per rank) |
| `soundness/src/bin/r4_custom_dt.rs` | SND-08 | Case A: a `contiguous(32,U8)` datatype sent from and received into 1-byte slices. Case B: `recv_custom` into `[Box<u64>;1]` | np=1 | A: 31 canary bytes overwritten with `0x41`. B: `free(): invalid pointer`, SIGABRT (gdb: `Box<u64>::drop`) | A: `Err` (datatype extent ≠ `size_of::<T>()`). B: does not compile (`T` bounded to plain-data types) |
| `soundness/src/bin/r5_use_after_finalize.rs` | SND-10 | `SharedWindow` outlives `drop(mpi)`, then reads `local_slice()[0]` | np=1 | SIGSEGV in `local_slice` (gdb): finalize freed the window's memory | No read of freed memory (a defined error or loud abort, per the 0.5.x lifecycle design) |
| `soundness/src/bin/r6_fetch_drop.rs` | SND-04 | Drops a `PendingFetchResult` before the closing fence on a `Win::allocate` window, plus a `get` into a buffer dropped mid-epoch | np=2 | **Did not manifest**: MPICH completes allocate-window RMA eagerly. Kept as the negative control for r6b | Same (no corruption) |
| `soundness/src/bin/r6b_fetch_drop_create.rs` | SND-04 | Same as r6, over a `Win::create` window (MPICH defers the operation to the fence) | np=2 | `unrelated Vec after get() completes = 0x1111111111111112 (expected 0x3333333333333333)`: the fetched value lands in freed and reused memory | Does not compile, or the result is kept alive until the epoch closes |
| `soundness/src/bin/r7_rma_range.rs` | SND-09 | Checks whether MPICH itself validates `target_disp`/`target_count` against the window size | np=2 | `put` of 64 elements at displacement 2 into a 4-element window returns `Ok`; `get` of 4 target elements into a 1-element origin returns `Ok`. MPICH validates nothing | `Err` from ferrompi |
| `soundness/src/bin/r7b_rma_range_canary.rs` | SND-09 | Same out-of-range `put`/`get` into `Win::create` memory bordered by canaries | np=2 | rank 1: 6 canary words after the window overwritten with `0x4141414141414141`, a remote out-of-bounds write | `Err` before FFI; canaries intact |
| `soundness/src/bin/r8_forget_win.rs` | SND-05 | `mem::forget(Win::create(&mut buf))`, `drop(buf)`, victim `Vec` reuses the memory, then rank 1 does a locked `put` | np=2 | `victim[0..4] = 0x5555555555555555…`: the remote `put` writes into the freed and reused allocation | Does not compile, or `Win::create` owns the buffer so forgetting it leaks the buffer instead of freeing it |
| `soundness/src/bin/r9_waitall_err.rs` | COR-03 (also COR-02, COR-01) | One truncating `irecv` makes `wait_all` fail; a new `irecv` is posted; then the old requests are dropped | np=2 | `evidence/r9.out`: `wait_all -> Err(class: Intern, code: 17 …)` (17 is `IN_STATUS` on MPICH, see COR-01). The drop then hangs and is killed with SIGTERM | `wait_all` error writes the request state back; drop does nothing; `r3.wait()` returns `Ok`; prints `done` |
| `soundness/src/bin/r9b_waitall_err_nor3.rs` | COR-03 (control) | Same as r9 without posting a new request | np=2 | `evidence/r9b.out`: `dropped OK`. This isolates reuse of freed MPI request objects as the trigger for r9 | Unchanged, with class `InStatus` |
| `soundness/src/bin/r10_waitany_aba.rs` | COR-02 | `wait_any` twice on the same slice, with an unrelated `irecv` posted in between (it takes the freed slot) | np=2 | The second `wait_any` returns `Some(0)` and completes and frees `other`; `other.wait()` gives `Err "Invalid MPI_Request"` while `other.is_completed()==false` | The second `wait_any` ignores the completed entry; `other.wait()` returns `Ok` |
| `soundness/src/bin/r11_shm_race.rs` | SND-13 | Rank 1 spins on `remote_slice(0)[0]` (a `&[u64]`) while rank 0 sets the flag through its own slice | np=2 (same node) | `evidence/r11.out`: rank 0 printed `flag set`/`exiting` 3 s apart; rank 1 never printed `observed flag`. The release build hoisted the load out of the loop: the disassembly is a single `cmpq $0x0,(%rcx)` followed by a jump to itself, so the loop never ends. The miscompilation is observed, not just theoretical | No plain `&[T]` over memory another process writes; the access API forces an atomic or volatile read, and the loop sees the flag |
| `soundness/src/bin/r12_thread_single.rs` | SND-11 | `Mpi::init()` (Single), then 4 scoped threads share `&world` and run `irecv`+`send`+`wait` 20k times | np=2 | `evidence/r12.{1,2,3}.out`: 3 of 3 runs failed. Two hit `corrupted message` asserts (left 1415/right 1417, left 1/right 2) and one hung; all were killed | Calling from another thread below `Multiple` (or `Serialized` without serialization) is rejected (`Err`/panic, or not allowed at compile time) |
| `soundness/src/bin/r12b_thread_multiple.rs` | SND-11 (control) | The same program initialised with `ThreadLevel::Multiple` | np=2 | `evidence/r12b.{1,2,3}.out`: 3 of 3 pass (`rank N done 2`) | Unchanged |
| `soundness/src/bin/r13_persistent_realloc.rs` | SND-02 | `recv_init(&mut data)`, then `data.reserve(4096)` reallocates the buffer, then `start`/`send`/`wait` | np=2 | Heap corruption: SIGSEGV inside `MPI_Finalize` (gdb: `unlink_chunk`/`_int_malloc` under `Mpi::drop`) | Does not compile: the persistent request owns its buffer |

### c-shim/: C-layer correctness (`src/bin/t*.rs`) and C baselines (`c/`)

| Program | Finding | What it does | How to run | Observed on v0.5.0 | Expected after fix |
|---|---|---|---|---|---|
| `c-shim/src/bin/t1_group_err.rs` | COR-06 | `group.include(&[999])`: a group error with no communicator involved | np=1 | MPICH 4.2.3 returns `Err` and the program survives, because MPICH falls back to the `COMM_WORLD` error handler. Under MPI-4 rules (`MPI_COMM_SELF`) it would abort; that is not verified here (no Open MPI 5 available) | `Err` on every implementation (`MPI_ERRORS_RETURN` set on `MPI_COMM_SELF`) |
| `c-shim/src/bin/t2_userop_after_finalize.rs` | COR-07 | `struct App { mpi, op: UserOp }`; the `Mpi` field drops first | np=1 | `Attempting to use an MPI routine (internal_Op_free) … after finalizing MPICH`, exit 1, `SURVIVED` never printed | Prints `SURVIVED` (op table swept at finalize) |
| `c-shim/src/bin/t3_waitany_stale.rs` | COR-02 | `wait_any`, then a new `irecv` C (it reuses slot 0), then `wait_any` again on the original slice | np=1 | The second `wait_any` acts on C through the stale handle; `C.wait()` gives `Err "Invalid MPI_Request"` | Once fixed, the second `wait_any` waits on `b`, whose send is posted later, so **this program deadlocks by design**. To make it a regression test, post `send(&sb…)` before the second call; then expect `Some(1)` and `C.wait()` → `Ok` |
| `c-shim/src/bin/t3b_waitany_null.rs` | COR-02 | The standard MPI idiom: loop `wait_any` until `None` without removing completed entries | np=1 | The first call returns `Some(i)`. The next call, with the completed entry still in the slice, returns `Err "Invalid MPI_Request"`; `Ok(None)` is never reached | `Some`, `Some`, then `Ok(None)` |
| `c-shim/src/bin/t4_waitall_err.rs` | COR-03, COR-04, COR-01 | `wait_all` with one truncating receive, then a new `irecv`, then drop of the old requests | np=1 | `evidence/t4.log`: `wait_all -> Err("… See the MPI_ERROR field in MPI_Status … (class=ERR_INTERN, code=17)")`, `fresh handle 2`, then it hangs at `dropping reqs` | The error names the truncation and has the correct class; drop does nothing; `fresh.wait` returns `Ok`; prints `SURVIVED` |
| `c-shim/src/bin/t4b_waitall_err_nofresh.rs` | COR-03 | After the failed `wait_all`, calls `test()` on each request | np=1 | MPICH `INTERNAL ERROR: unexpected value in case statement`: `test` runs on already-freed MPI request objects | `test` reports the requests as completed; prints `SURVIVED` |
| `c-shim/src/bin/t4c_waitany_err.rs` | COR-03 | `wait_any` returns a truncation error; then `test()` on the failed entry | np=1 | Same failure class as t4b: the table keeps a freed MPI request | The failed entry is marked completed; prints `SURVIVED` |
| `c-shim/src/bin/t5_reinit.rs` | COR-07 | `Mpi::init()` again after the first `Mpi` was dropped | np=1 | MPICH abort: `Cannot call MPI_INIT or MPI_INIT_THREAD more than once` (exit 15) | `re-init is_err=true`, `SURVIVED` |
| `c-shim/src/bin/t6_world_after_finalize.rs` | COR-07 | `world.barrier()` after `Mpi` was dropped | np=1 | The process exits inside MPICH (MPI call after finalize) | `Err`, `SURVIVED` |
| `c-shim/src/bin/t7_errclass.rs` | COR-01 | `broadcast` with root 999, and a truncating receive | np=1 | `class=ERR_REQUEST` for the invalid root; `class=ERR_UNKNOWN` for the truncation | `ERR_ROOT`, `ERR_TRUNCATE` |
| `c-shim/src/bin/t8_gatherv_overflow.rs` | SND-07 | `gatherv` of 64 elements into a 1-element `recv` | np=1, valgrind | valgrind: `Invalid write of size 8 … 0 bytes after a block of size 8` in `ferrompi_gatherv` | `Err` before FFI; valgrind clean |
| `c-shim/src/bin/t8b_gatherv_short_counts.rs` | SND-07 | `gatherv` with counts/displs of length 1 on 2 ranks | np=2, valgrind | valgrind: invalid read past the counts array | `Err` (array length must equal the communicator size) |
| `c-shim/src/bin/t8c_allgather_small.rs` | SND-06 | `allgather` with `recv.len() == send.len()` on 2 ranks | np=2, valgrind | valgrind: invalid write | `Err` (`recv.len() ≥ send.len()*size` required) |
| `c-shim/src/bin/t9_test_err_aba.rs` | COR-02 | `A.test()` fails with truncation, so C frees slot 0 while `A.is_completed()==false`. B then takes slot 0, and `drop(A)` waits on it | np=1 | `drop(A)` consumes B's request; `B.wait()` gives `Err "Invalid MPI_Request"` | `B.wait()` returns `Ok`, `bbuf=[42]` |
| `c-shim/src/bin/t10_group_empty.rs` | VER (GROUP_EMPTY slot) | Creates and drops empty groups (`include(&[])`, `difference(self)`) 3 times | np=1 | Works and prints `SURVIVED`; slot 0 (`MPI_GROUP_EMPTY`) is never handed out or freed | Unchanged |
| `c-shim/src/bin/t11_win_after_finalize.rs` | SND-10 | A `Win<'static,f64>` from `allocate` outlives `Mpi`, then its slice is summed | np=1, valgrind | valgrind: `Invalid read … inside a block of size 8,192 free'd by PMPI_Win_free ← ferrompi_finalize (ferrompi.c:758)` | No invalid read |
| `c-shim/src/bin/t12_reduce_inplace.rs` | VER (in-place semantics) | `reduce_inplace` Sum to root 0 | np=2 | Correct result on MPICH; the in-place alias check passes | Unchanged |
| `c-shim/src/bin/t13_rma_trunc.rs` | COR-05, SND-12 | `put` with `target_count = 2^32+4` into a `Win::allocate` window, then prints the window | np=1 | `put -> Ok` but only 4 elements are transferred, because `(int)` truncates the count. Untouched elements show uninitialised memory (e.g. `1.012e-320`, `2e-323`) | `put` → `Err` (MPI_ERR_COUNT, or `_c` dispatch); the window starts zero-filled |
| `c-shim/src/bin/t14_send_neg1.rs` | COR-09 | `send(dest = -1)` | np=1 | Returns `Ok` and does nothing: `-1 == MPI_PROC_NULL` on MPICH. On Open MPI the same call is an invalid-rank error | Portable, explicit semantics (typed ProcNull/Any, normalised in C) |
| `c-shim/src/bin/t15_err_after_finalize.rs` | VER (late request drop, error string after finalize) | An `irecv` request outlives `Mpi`; `wait()` is called after finalize | np=1 | Returns `Err` with a readable message and prints `SURVIVED`. The table sweep makes late request use harmless, and `MPI_Error_class`/`Error_string` work after finalize on MPICH | Unchanged, or a clearer "finalized" error once COR-07 lands |
| `c-shim/src/bin/t16_cancel_coll.rs` | COR-10 | `cancel()` on an `iallreduce` request | np=2 | rank 0: `Err "Attempt to cancel an unknown type of request"` from MPICH | ferrompi refuses cancel on non-point-to-point requests with a typed error, without calling MPI |
| `c-shim/c/errhandler_group_incl.c` | COR-06 (C baseline) | Plain C: `MPI_ERRORS_RETURN` on `COMM_WORLD` only, then `MPI_Group_incl` with rank 999 and `MPI_Type_contiguous(-1)` | `mpicc`; np=1 | Output not archived. Written to show that MPICH 4.2.3 returns the error code here (consistent with t1) | Reference only |
| `c-shim/c/pair_type_extents.c` | VER (pair layouts), COR-11 baseline | Prints lb, extent and size of the MPI value+index pair types (`MPI_FLOAT_INT` … `MPI_LONG_DOUBLE_INT`) | `mpicc`; np=1 | x86_64 Linux MPICH: FLOAT_INT 8, DOUBLE_INT 16, LONG_INT 16, 2INT 8, SHORT_INT 8, LONG_DOUBLE_INT 32. All match the Rust `#[repr(C)]` sizes in `src/datatype.rs` | Unchanged on Linux |
| `c-shim/c/ldi.c` | COR-11 | Compile-only layout probe: symbol sizes equal `sizeof`/`_Alignof` of `{long double; int}` and `{long; int}` | `clang --target=aarch64-apple-darwin -c` (also `x86_64-pc-windows-msvc`), then `llvm-nm -S` | aarch64-apple-darwin: `{long double;int}` is 16 bytes, 8-byte aligned, versus Rust `LongDoubleInt` at 32 bytes, 16-byte aligned. MSVC: `long` is 4 bytes, so `LongInt` differs too | Per-target Rust layouts, or C `_Static_assert` size checks |
| `c-shim/c/gfree.c` | VER | `MPI_Group_free(MPI_GROUP_EMPTY)` in plain C | `mpicc`; np=1 | MPICH accepts it (rc=0) | Reference only |

### requests/: request-slot reuse and error classes (default features)

| Program | Finding | What it does | How to run | Observed on v0.5.0 | Expected after fix |
|---|---|---|---|---|---|
| `requests/src/main.rs` (bin `ferrompi-audit-requests`) | COR-02, COR-01 | Two self-requests completed with `test()`; two new requests reuse slots 0 and 1; then `wait_all` on the old slice; plus the standard `wait_any` loop | np=1 | `old handles: 0 1`, `new handles: 0 1`, `wait_all(old) -> Ok(())`. Then `new_recv.wait()` and `new_send.wait()` both give `Err(Mpi { class: Pending, code: 19, "Invalid MPI_Request" })`: the new requests were silently completed and freed, and code 19 (`MPI_ERR_REQUEST` on MPICH) is decoded as `Pending`. `wait_any #1 -> Some(0)`, `wait_any #2 -> Err(… "Invalid MPI_Request")` | `wait_all(old)` does nothing; both new waits return `Ok`; `wait_any #2` returns `Some(1)` |
| `requests/src/bin/errclass.rs` | COR-01 | Invalid-root `broadcast`, `allreduce` of `f64` with `BitwiseOr`, and a truncating `recv` | np=1 | Classes decoded as `Request`, `Group` and `Unknown` (MPICH values 7, 9, 14) | `Root`, `Op`, `Truncate` |

### win-sync/: `Win::sync` rustdoc example

| Program | Finding | What it does | How to run | Observed on v0.5.0 | Expected after fix |
|---|---|---|---|---|---|
| `win-sync/src/main.rs` | Docs finding: `Win::sync` rustdoc (see `../findings/`) | Runs the rustdoc example as written: a bare `win.sync()` outside any epoch | np=1 or 2 | `bare Win::sync outside epoch -> Err: MPI error in win_sync: Wrong synchronization of RMA calls` | The rustdoc is corrected (or `sync` moves onto the lock guards), and this call is documented as an error |

Note: the scratchpad crate this came from was named `reinit`, but it contains only the `Win::sync` check. The re-initialisation repro is `c-shim/src/bin/t5_reinit.rs`.

### perf/: overhead measurements (MPICH only; release profile uses thin LTO, cgu=1)

All binaries run interleaved A/B rounds, timing raw MPI (called directly through `extern "C"` from Rust) against ferrompi. Reported values are medians. `REPS`, `ITERS` and `K` are read from the environment where a program supports them. Run at np=1 to isolate software cost, and at np=2 for intranode latency. At np=2, deltas under about 30 ns are noise.

| Program | Finding | What it measures | How to run | Observed on v0.5.0 (i7-12700KF) | Expected after fix |
|---|---|---|---|---|---|
| `perf/src/bin/overhead.rs` | PRF-01, PRF-04; VER blocking paths | Raw vs ferrompi for `allreduce`(1, 64), `allreduce_scalar`, `barrier`, `bcast`, `iallreduce+wait`, persistent `start+wait`, 8×(isend+irecv)+waitall, `fetch_and_op`+flush, `put`+flush, `SharedWindow::remote_slice` | np=1 and np=2 | Blocking paths about +1 ns. **8×(isend+irecv)+waitall: +230 ns (+58%) at np=1, +396 ns (+33%) at np=2.** `iallreduce+wait` +5.8 ns. Persistent `start+wait` +2.7 ns. `put`+flush +4 ns. `remote_slice` 6 ns. At 1 f64, np=2: persistent 298 ns vs `iallreduce` 491 ns (PRF-04 small-message regime) | Nonblocking request overhead near 0 once handles are stored by value; blocking unchanged |
| `perf/src/bin/sweep.rs` | VER (blocking path) | `allreduce` counts 1–512: raw vs C shim vs Rust API | np=1 | C shim +0.9 ns; Rust about +0 ns over the shim | Unchanged. Supports the decision not to pursue cross-language LTO |
| `perf/src/bin/reqdecomp.rs` | PRF-01 | Splits the nonblocking overhead: raw → C shim → Rust API, for `waitall` and per-request `wait` (`K` = requests per side) | np=1 (`K=8`) | C shim +13.9 ns/request over raw; Rust +0.4 ns/request over the shim | Shim share about 0 |
| `perf/src/bin/batch.rs` | PRF-01, PRF-05 | `waitany` drain of 16 or 64 requests, `test_some` polling of 32, persistent `start_all+wait_all` (2 requests), `allreduce`(64) | np=1 and np=2 | `waitany` drain +14.4 ns/request; `test_some` polling +15.3 ns/request; `start_all+wait_all` (2): raw 92–96 ns, ferrompi +20–24 ns | Request overhead gone; about 8 ns saved on `start_all`/`wait_all` |
| `perf/src/bin/pdecomp.rs` | PRF-05 | Persistent `start_all+wait_all` for k=1,2,4,8: raw vs C shim vs Rust; and a loop of single `start`/`wait` | np=1 | k=2: C shim +10 ns, Rust +14 ns. Single `start`/`wait` +2.7–3.3 ns (already optimal) | Rust share reduced (no 512-byte `memset`) |
| `perf/src/bin/memsetcost.rs` | PRF-05 | Handle scratch buffer: zeroed `[0i64;64]` vs `MaybeUninit`. No MPI | plain `./memsetcost` | About 4 ns per call saved with `MaybeUninit` | Scratch buffer is `MaybeUninit` |
| `perf/src/bin/mtreq.rs` | PRF-02 | `THREAD_MULTIPLE`, one duplicated communicator per thread, K self send/recv then `waitall`, for T threads | np=1 | ferrompi adds +13 to +40 ns/request at 1–4 threads and +5 ns at 8, against raw 58–438 ns/request (MPICH's own lock dominates) | Table contention gone |
| `perf/src/bin/pingpong.rs` | VER | `send`/`recv` round trip, 1 f64 | np=2 | Within noise | Unchanged |
| `perf/src/bin/selfrecv.rs` | VER | Blocking `recv` wrapper (status plus `MPI_Get_count_c`) and `send` wrapper, isolated | np=1 (asserts size==1) | `recv` +5 ns (about 1% of an intranode round trip) | Unchanged |
| `perf/src/bin/aa.rs` | methodology control | A/A (identical arms) and A/B `allreduce`(64) | np=1 and np=2 | A/A within ±2 ns at np=1. At np=2, A/B swung by up to ±30 ns and sometimes flipped sign | — |
| `perf/c/request_table_bench.c` | PRF-01, PRF-02 | Verbatim copy of ferrompi's request-table alloc/free (`MPI_Request=int`), timed on T pinned threads. `-DPLAIN` uses relaxed plain stores; `-DDENSE` is the pre-v0.4 dense CAS table (ADR-0002 Option A) | `taskset -c <P-cores> ./tab T` | Alloc+free pair, 1 thread: bitmap 14.8 ns (16–19 ns in other runs), plain 4–5 ns, dense 5.2 ns. T=2/4/8: bitmap 91/169/473 ns vs dense 70/142/261 ns | Obsolete once requests are stored by value |

### api-checks/: `#[non_exhaustive]` cast check

| Program | Finding | What it does | How to run | Observed | Expected |
|---|---|---|---|---|---|
| `api-checks/lib_ne.rs` + `api-checks/main_ne.rs` | ARC-02 | A downstream crate casts a `#[non_exhaustive] #[repr(i32)]` enum from another crate with `as i32` | two `rustc` invocations (see Building) | Compiles and prints `0` on rustc 1.95. Adding `#[non_exhaustive]` to `DatatypeTag`/`ReduceOp` does not break numeric casts; only the documented discriminant "contract" is a lock-in | Reference only |

### abi/: MPI 5 standard-ABI compile and link check

| File | Finding | Purpose | How to run |
|---|---|---|---|
| `abi/mpiabi-stub.pc.in` | MPI 5 ABI readiness (see `../findings/`) | pkg-config template for linking ferrompi against the MPI Forum reference ABI stubs | Steps below |

The steps below are reconstructed from the ABI reviewer's report; the exact command lines were not archived.

1. `git clone https://github.com/mpi-forum/mpi-abi-stubs`, then `cc -shared -fPIC -I. mpilib.c -o libmpi_abi.so`.
2. `cc -std=c11 -fsyntax-only -Wall -Wextra -Wpedantic -I<stubs> csrc/ferrompi.c`. Observed: **zero diagnostics**. The same result was obtained against MPICH v5.0.1's `mpi_abi.h`.
3. Copy the template to `<pcdir>/mpiabi-stub.pc`, replacing `@PREFIX@` with the stubs directory. Then run `PKG_CONFIG_PATH=<pcdir> MPI_PKG_CONFIG=mpiabi-stub cargo build --locked --features rma --examples` from the repo root. Observed: builds and links against `libmpi_abi.so`.

Runtime was not tested, because the stubs call `abort()` in every non-trivial function.

Third-party references, deliberately not copied here:
- https://github.com/mpi-forum/mpi-abi-stubs
- https://github.com/pmodels/mpich/blob/v5.0.1/src/binding/abi/mpi_abi.h
- https://github.com/pmodels/mpich/blob/v4.3.0/src/binding/abi/mpi_abi.h (a pre-standard draft: `MPI_ERRORS_RETURN`/`ABORT` are swapped and the thread levels differ)
- https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf

## tools/

| File | Purpose |
|---|---|
| `tools/ffi_cmp.py` | Compares every `pub fn` in `src/ffi.rs` with its C definition in `csrc/ferrompi.c` (arity, argument widths, return type). Run `python3 tools/ffi_cmp.py`. It resolves the repo root relative to itself. Result on v0.5.0: `checked 172 bad 0` |
| `tools/run_mpi_tests_patched.sh` | `tests/run_mpi_tests.sh` with one change: binary paths honour `${CARGO_TARGET_DIR:-./target}` (2 lines). Used to produce the np=1/3/4/5/8 logs in `evidence/` (INF-01, INF-10). Usage: `MPI_NP=<n> MPI_TEST_TIMEOUT=60 tools/run_mpi_tests_patched.sh [rma]` from the repo root |
| `tools/decoy-mpich.pc.in` | INF-04 probe-precedence repro. Replace `@PREFIX@` with an empty directory that has `include/` and `lib/`, then run `PKG_CONFIG_PATH=<dir> MPICC=/opt/mpich/bin/mpicc cargo build`. Observed: `build.rs` prints `Found MPI via pkg-config: mpich`, so `MPICC` is ignored. The C compile then picked up a stray `/usr/local/include/mpi.h` and failed with `fatal error: mpi_proto.h`. Expected after fix: an explicit `MPICC` takes precedence, and an explicit variable that fails to resolve is a hard error |

## evidence/

| File | What it shows |
|---|---|
| `unmodified_run.log` | INF-01: the unmodified `tests/run_mpi_tests.sh` run with `CARGO_TARGET_DIR` set. Every binary is "not found", yet it reports `Results: 0 passed, 0 failed, 57 skipped (57 total)` / `All tests passed!` and exits 0 |
| `mpi_default.log` | Patched runner, default features, np=4: 57/57 pass |
| `mpi_rma.log` | Patched runner, `rma`, np=4: 73/73 pass |
| `mpi_rma_np1.log` | `rma`, np=1: 65/73. 8 tests exit 101 on their "needs ≥ 2 ranks" asserts (`test_collectives`, `test_blocking_extra`, `test_nonblocking`, `test_p2p_extra`, `test_nonblocking_collectives`, `test_persistent`, `test_errhandler_returns`, `test_rma_window`) |
| `mpi_rma_np3.log`, `mpi_rma_np5.log` | INF-10: 71/73. `test_nonblocking` and `test_p2p_extra` hit the 60 s timeout (even/odd pairing deadlock at odd np) |
| `mpi_rma_np8.log` | `rma`, np=8: 73/73 pass |
| `r9.out`, `r9b.out` | Output of `soundness` r9 (hang, then SIGTERM) and r9b (`dropped OK`) |
| `r11.out` | Output of `soundness` r11: rank 1 never observes the flag |
| `r12.1.out`–`r12.3.out` | Output of `soundness` r12 (`ThreadLevel::Single`): corrupted-message panics or hang in 3 of 3 runs |
| `r12b.1.out`–`r12b.3.out` | Output of `soundness` r12b (`ThreadLevel::Multiple`): 3 of 3 pass |
| `t4.log` | Output of `c-shim` t4: `wait_all` error mis-decoded as `ERR_INTERN` (code 17 = `IN_STATUS`), then a hang on drop |
| `bt.out` | VER: last iteration of a 300× stress run (`--test-threads=2`) of the `buffer_attach`/`buffer_detach` unit tests in `src/lib.rs`. 4 passed, 0 failures across all 300 iterations; the shared `ATTACHED_BUFFER` static does not race in practice |

Not archived:
- Criterion output from `cargo bench` (`ffi_overhead` and the 1 MiB `persistent_vs_iallreduce` runs behind PRF-03 and PRF-04). The PRF-04 figure at 1 MiB, 2 ranks, 100 iterations was persistent 14.77 ms vs `iallreduce` 14.68 ms.
- The rustc 1.74 MSRV check, which was run against a scratch copy of the repo.
