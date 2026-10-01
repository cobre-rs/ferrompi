# 05 — Performance: what ferrompi adds on top of MPI

## Method

- Interleaved A/B in the same binary: ferrompi call vs raw `MPI_*` via `extern "C"` from Rust (MPICH handles are ints); a third arm calls the C shim directly to split C vs Rust cost. Harness: [`../repros/perf/`](../repros/README.md) (`perfprobe` crate + `tab.c` table microbench).
- Hardware: i7-12700KF; MPICH 4.2.3 ch4:ofi (intranode shared memory); release build (thin LTO, `codegen-units = 1`).
- **1-rank** runs isolate software cost reliably. **2-rank** numbers are real intranode latency but noisy: separate A/B runs swung up to ±30 ns and sometimes flipped sign; A/A control within ±2 ns. Figures come from quiet reruns (load ≈ 1). Treat small 2-rank deltas as noise.

## Findings

### PRF-01 — Request-table alloc/free costs ~14 ns per nonblocking request, even single-threaded

- **Severity:** major · **Verified:** measured + disassembly · **Target:** 0.7 durable (ARC-01); optional 0.5.x mitigation (see fix 1)
- **Locations:** `csrc/ferrompi.c:296-319` (`alloc_request`), `:341-349` (`free_request`), `:3567-3631` (`wait`/`waitall`; waitall re-validates every handle in a second pass). Reached by every `isend/irecv`, every `i*` collective, `rput/rget/raccumulate`, every completion API.
- **Root cause:** `atomic_fetch_or_explicit(…, acq_rel)` whose return value is used compiles to a `lock cmpxchg` retry loop; every free is a `lock and`. Two locked RMWs per request regardless of thread level.
- **Measurements:**

  | Case | Raw MPI | ferrompi | Added |
  |---|---|---|---|
  | 8×(irecv+isend)+waitall, 1 rank | 397 ns | 627 ns | +230 ns (**+58%**) |
  | 8×(irecv+isend)+waitall, 2 ranks | 1184 ns | 1580 ns | +396 ns (**+33%**, ~25 ns/request on a 74 ns/request baseline) |
  | iallreduce+wait, 1 rank | — | — | +5.8 ns |
  | waitany drain of 32, 1 rank | — | — | +14.4 ns/request |
  | test_some polling, 1 rank | — | — | +15.3 ns/request |

  Split (1 rank): C shim +13.9 ns/request; Rust layer +0.4 ns/request. Isolated table code pinned to a P-core: **16–19 ns** per alloc+free (current bitmap) vs **4–5 ns** plain-store variant vs **5.2 ns** for the pre-v0.4.x dense design — the bitmap rewrite made the single-thread path ~3× slower.
- **Impact:** intranode small-message nonblocking p2p (halo exchange) loses 25–35% of message rate; nonblocking collectives ~1% (5.8 ns vs 491 ns iallreduce at 2 ranks); inter-node < 2%.
- **Fix options:** (1) cheap: record the provided thread level in C; below `MPI_THREAD_MULTIPLE` use relaxed load + plain store (saves ~11–14 ns/request); (2) durable: no request table — `MPI_Request` stored by value in `Request`/`PersistentRequest` (ARC-01).
- **Acceptance:** ≤ 2 ns/request overhead at 1 rank in the new ffi bench (PRF-03).

### PRF-02 — The occupancy bitmap concentrates contention; ADR and code comment claim the opposite

- **Severity:** minor · **Verified:** measured (microbench) + `nm` · **Target:** 0.5.x docs correction; removed by ARC-01
- **Locations:** `csrc/ferrompi.c:26-32` (comment "less false sharing, F2-002"), `:311-312` (unconditional hint store on every alloc); `docs/adr/0002-handle-tables.md:408-425` ("~64× less metadata to false-share").
- **Evidence:** `next_request_hint` (0x74780) and `request_bits` (0x747a0) share one 64-byte line; all threads start at the same hint word and RMW the same 64-bit word (true sharing + CAS retries). Per-thread alloc+free pinned to distinct P-cores:

  | Threads | Bitmap (current) | Dense (pre-v0.4.x) |
  |---|---|---|
  | 1 | 14.8 ns | 5.2 ns |
  | 2 | 91 ns | 70 ns |
  | 4 | 169 ns | 142 ns |
  | 8 | 473 ns | 261 ns |

  Real `MPI_THREAD_MULTIPLE` (1 rank, self messages): ferrompi +13…+40 ns/request at 1–4 threads, +5 ns at 8, vs raw 58–438 ns/request — MPICH's own lock dominates today (`MPIR_CVAR_CH4_NUM_VCIS=8` did not help self messages).
- **Fix direction:** if the table survives until 0.7: per-thread starting word, store the hint only when it changes, correct ADR/comment.

### PRF-03 — `benches/ffi_overhead.rs` cannot measure the overhead it is named for

- **Severity:** minor (methodology) · **Verified:** measured · **Target:** 0.5.x
- **Locations:** `benches/ffi_overhead.rs:36-88` (sentinel `allreduce` inside every `b.iter`, no raw-MPI control arm; `rank_cached`/`size_cached` time a field read, 220–360 ps); `benches/README.md:166-172` ("Ticket-013 uses the numbers … to decide which FFI trampolines warrant `#[inline]`"), `:204-209`.
- **Evidence:** bench reports barrier 777 ns / broadcast 635 ns / allreduce 745 ns (±30 ns); interleaved measurement of the same ops: ~288 / 98 / 232 ns. Real blocking overhead ≈ 0.9 ns (C) + ~0 ns (Rust), ~30× below the bench's resolution; the path with real overhead (PRF-01) is not benchmarked at all.
- **Fix direction:** interleaved A/B vs direct `MPI_*`, 1 rank, covering isend/irecv/wait/waitall, iallreduce, persistent start/start_all. Replace (not add to) the current bench.

### PRF-04 — "Persistent is 10–30% faster than iallreduce" is refuted as stated; the bench measures the wrong regime

- **Severity:** minor (docs + bench) · **Verified:** measured · **Target:** 0.5.x
- **Locations:** `benches/README.md:143-146`; `benches/persistent_vs_iallreduce.rs:16` (N = 131072 f64 = 1 MiB only); also `README.md:43`, `docs/migrating-from-rsmpi.md:394-395`.
- **Evidence:** 1 MiB, 2 ranks, 100 iterations: persistent 14.77 ms vs iallreduce 14.68 ms (0.6% slower, noise). 1 f64, 2 ranks: persistent start+wait 298 ns vs iallreduce+wait 491 ns (**39% faster**). The benefit exists only in the small-message regime SDDP uses.
- **Fix direction:** sweep 8 B → 1 MiB; state the measured claim with its regime.

### PRF-05 — `start_all`/`wait_all` zero a 512-byte scratch array per call

- **Severity:** nit · **Verified:** measured + disassembly · **Target:** 0.5.x (optional)
- **Locations:** `src/persistent.rs:56-68`, `src/request.rs:17-42` (`[0i64; 64]` → `memset` 512 B; `with_index_buf` adds 256 B); C copy-in/copy-out `csrc/ferrompi.c:3594-3670`.
- **Evidence:** 2 persistent requests, 1 rank: raw 95.7 ns, C +10 ns, Rust +14 ns; zeroing ≈ 4 ns/call (8 ns per start_all+wait_all). Individual `start()`/`wait()` add only 2.7–3.3 ns (already optimal).
- **Fix direction:** `MaybeUninit` scratch; ~8 ns/iteration. Low priority; subsumed by ARC-01.

### PRF-06 — Small dead or allocating work off the hot path

- **Severity:** nit · **Target:** 0.5.x (with BLT-15)
- `src/request.rs:222-229, 305-312`: `wait_some`/`test_some` allocate a `Vec<usize>` on every completion (dominated by PRF-01 today).
- `csrc/ferrompi.c:4805-4821, 4833`: `UserOp` trampoline computes a datatype tag (up to 14 compares) that Rust ignores (`_dt_tag`, `src/op.rs:167`) — dead work per reduction callback (BLT-15).
- `src/topology.rs:122-123`: allgathers 256 B per rank → 256·P bytes per rank (~25 MB at P = 100k). One-shot diagnostic; only a concern at extreme scale.

### PRF-07 — `[profile.release]` in a library does not reach downstream users; its comment says it does

- **Severity:** minor (misleading) · **Verified:** Cargo semantics · **Target:** 0.5.x
- **Locations:** `Cargo.toml:303-318` ("… measures the same configuration downstream users get from `--release`"); CHANGELOG 0.5.0 "Added".
- **Fix direction:** keep only if it helps this repo's own benches, with a one-line accurate comment; otherwise delete.

## Verified efficient — do NOT optimise

| Path | Measured overhead | Note |
|---|---|---|
| Blocking collectives / send | +0.4…+1.3 ns at 1 rank (C +0.9, Rust ~0) | Rust wrapper → length compare + call + `test eax`, `#[cold]` error path; C ≈ 20 instructions ending in tail `jmp MPI_Allreduce@plt`. Cross-language LTO could recover ≤ 1 ns (< 0.5% of 230 ns intranode allreduce) — the decision not to pursue it (`Cargo.toml:302-316`) is correct |
| Scalar helpers (`allreduce_scalar` …) | +0.4 ns | stack arrays, no allocation |
| Persistent `start()`/`wait()` | +2.7 ns (1% at 2 ranks) | no allocation; one bounds check + acquire load (`mov` on x86) + tail `jmp MPI_Start` |
| Large-count dispatch | one `cmp $0x7fffffff; jg` | no allocation |
| V-collectives | 0 conversions | `&[i32]` passes straight through as `int32_t*` |
| RMA put/get/accumulate | put+flush +4 ns (2.4%) at 1 rank | window lookup + two switch lookups |
| `fetch_and_op`/`compare_and_swap` Box allocations | not measurable (−0.9 ns vs 677 ns op) | leave as is |
| `SharedWindow::remote_slice` | 6 ns/call | callers can hoist |
| Blocking `recv` + `MPI_Get_count_c` | +5 ns (~1% of RTT) | |
| Batch completion scratch | stack ≤ 64 | O(N) copies vanish at N = 128 (−0.2%) |
