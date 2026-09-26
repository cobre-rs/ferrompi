# 01 — Soundness: safe Rust can cause undefined behaviour

Every finding here is reachable from **100% safe Rust** (no `unsafe` block in the
calling program). Repro programs live in [`../repros/`](../repros/README.md);
"observed" is the behaviour on v0.5.0 with MPICH 4.2.3.

Severity legend: **critical** = safe code → UB / memory corruption / data race.

---

### SND-01 — Nonblocking `Request` is not tied to its buffer

- **Severity:** critical · **Verified:** repro · **Target:** 0.6 (see ARC-06)
- **Locations:** `src/request.rs:102` (`struct Request { handle: i64, completed: bool }` — no lifetime, no owned buffer); `src/comm/p2p.rs:112` (`isend`), `:156` (`irecv`), `:511` (`isend_custom`), `:571` (`irecv_custom`); every `i*` collective in `src/comm/nonblocking.rs` and `src/comm/v_collective.rs`.
- **Defect:** the buffer borrow ends when the constructor returns, but MPI keeps a raw pointer until completion. The rustdoc (`src/request.rs:50-59`) states the invariant in prose on a *safe* fn; nothing enforces it.
- **Evidence:** `repros/soundness/src/bin/r1_irecv_uaf.rs` — `irecv` into a `Vec` that goes out of scope, a new `Vec` reuses the chunk, peer sends: **256/256 bytes of the unrelated allocation overwritten by MPI**. `std::mem::forget(req)` additionally lets the write outlive the buffer with no wait at all.
- **Fix direction (accepted, D-1):** closure-scoped API for nonblocking requests (scope end waits all; a scope cannot be forgotten). Lifetime-only `Request<'buf>` is **not** sufficient — `mem::forget` ends the borrow (see SND-05).
- **Acceptance:** the r1 program no longer compiles (buffer escapes scope) or is rewritten against the scope API and shows no corruption; a `compile_fail` doctest covers "drop buffer while request outstanding".

### SND-02 — `PersistentRequest` is not tied to its buffer

- **Severity:** critical · **Verified:** repro · **Target:** 0.6 (ARC-06)
- **Locations:** `src/persistent.rs:83` (no lifetime/ownership); every `*_init` in `src/comm/persistent.rs` (`send_init` :42 … `alltoall_init_inplace` :1120).
- **Defect:** pointer captured once at `*_init` and reused on every `start()`; any reallocation of the source `Vec` silently retargets MPI at freed memory. The module doc example (`src/persistent.rs:9-41`) mutates `data` between init and start — legal MPI, but it compiles only because nothing borrows `data`.
- **Evidence:** `repros/soundness/src/bin/r13_persistent_realloc.rs` — `recv_init(&mut data)`, `data.reserve(4096)` (realloc frees old chunk), victim `Vec` reuses it, `start/send/wait` → **SIGSEGV inside `MPI_Finalize`** (`unlink_chunk`/`_int_malloc`: heap metadata corrupted).
- **Fix direction (accepted, D-1):** `PersistentRequest` takes **ownership** of its buffer (`Vec<T>`/`Box<[T]>`) and exposes `buffer()`/`buffer_mut()` only while inactive; `into_buffer()` returns it. `mem::forget` then only leaks the buffer (safe).
- **Acceptance:** r13 cannot be expressed; SDDP-style loop (init once, mutate between iterations, start/wait) still zero-allocation per iteration (bench PRF-04 regime).

### SND-03 — RMA origin buffers are not tied to the access epoch; four rustdoc examples are themselves UB

- **Severity:** critical · **Verified:** reading · **Target:** 0.6 (ARC-06)
- **Locations:** `src/window.rs:1586` (`put`), `:1776` (`get`), `:1974` (`accumulate`), `:2201` (`get_accumulate`), and `rput`/`rget`/`raccumulate`. Rustdoc examples that drop the origin before the closing fence: `src/window.rs:1579`, `:1768`, `:1966`, `:2191-2192`. The "Safety Contract" sections at `:1549-1563` document the rule on safe fns.
- **Defect:** MPI may read/write the origin buffer until the epoch closes (fence / unlock / complete / flush); the borrow ends at return.
- **Evidence:** the four examples are `no_run`, so CI never executes them; the integration test `examples/test_rma_get.rs:61` scopes its buffer correctly, showing the examples are wrong, not the tests.
- **Fix direction (accepted, D-1):** closure-scoped epochs (`win.fence_epoch(|ep| ep.put(&buf, …))`, same for lock/lock_all/PSCW) so origin borrows must outlive the scope, which always closes the epoch.
- **Acceptance:** the four rustdoc examples rewritten against the scope API and run (not `no_run`) in the MPI suite.

### SND-04 — `PendingFetchResult` dropped before the epoch closes → MPI writes into freed heap

- **Severity:** critical · **Verified:** repro · **Target:** 0.6 (ARC-06)
- **Locations:** `src/window.rs:741` (`PendingFetchResult`), `:773` (`unsafe fn resolve`), `:2318` (`fetch_and_op`), compare-and-swap path.
- **Defect:** the boxed result buffers are only safe while the value is kept alive until the closing synchronisation; `let _ = win.fetch_and_op(…)` frees the box immediately.
- **Evidence:** `repros/soundness/src/bin/r6b_fetch_drop_create.rs` (window from `Win::create`) — victim `Vec<i64>` reusing the chunk reads **`0x1111111111111112`** (the remote fetched value) after the fence. `r6_fetch_drop.rs` (`Win::allocate`) did not manifest because MPICH completed eagerly — same bug, timing-dependent.
- **Fix direction:** fold into the epoch scope (SND-03): pending results are owned by the epoch object and resolved when the scope closes.
- **Acceptance:** r6b cannot be expressed; results obtainable only after epoch close.

### SND-05 — `mem::forget(Win::create(...))` leaves MPI aliasing a released buffer

- **Severity:** critical · **Verified:** repro · **Target:** 0.6 (ARC-06)
- **Locations:** `src/window.rs:900` (`Win::create(buf: &'a mut [T])`), `Drop` at `:2516`.
- **Defect:** `Win<'a, T>` does carry the buffer lifetime, but `forget` skips `MPI_Win_free` and ends the borrow; remote ranks keep writing into memory the caller has freed. This also refutes "a lifetime parameter fixes it" for SND-01/02.
- **Evidence:** `repros/soundness/src/bin/r8_forget_win.rs` — `forget(win)`, `drop(buf)`, victim reuses the chunk, rank 1 `put`s: **victim[0..4] = `0x5555555555555555`**.
- **Fix direction (accepted, D-1):** `Win::create` takes ownership of the buffer (`Vec<T>`/`Box<[T]>`), mirroring `Win::allocate` which owns MPI-allocated memory.
- **Acceptance:** r8 cannot be expressed.

### SND-06 — `gather` / `allgather` / `scatter` never validate buffer sizes (heap overflow)

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x
- **Locations (9 methods):** blocking `src/comm/blocking.rs:689` (`gather`), `:719` (`allgather`), `:985` (`scatter`); nonblocking `src/comm/nonblocking.rs:163` (`igather`), `:203` (`iallgather`), `:237` (`iscatter`); persistent `src/comm/persistent.rs:515` (`gather_init`), `:568` (`scatter_init`), `:619` (`allgather_init`). C shims forward counts verbatim (e.g. `csrc/ferrompi.c:1451-1473`).
- **Defect:** gather/allgather pass `send.len()` as the per-rank *receive* count, so MPI writes `send.len() * size` elements into `recv` regardless of `recv.len()`; scatter passes `recv.len()`, so root reads `recv.len() * size` elements from `send`. Sibling methods *do* check (`alltoall` `blocking.rs:1030`, `reduce_scatter_block` `:1083`, every `*_inplace`) — validation drift (ARC-10). `allgather_init` docs (`persistent.rs:603`) state the requirement but do not check it.
- **Evidence:** `repros/soundness/src/bin/r2_allgather_overflow.rs` — 8-element send into a 1-element `recv` field of a `#[repr(C)]` struct: **7 canary words overwritten** for `allgather`, `gather`, `gatherv`; `repros/c-shim/src/bin/t8c_allgather_small.rs` under valgrind: "Invalid write of size 8".
- **Fix direction:** validate `recv.len() >= send.len() * size` (gather at root, allgather everywhere) and `send.len() >= recv.len() * size` (scatter at root) → `Error::InvalidBuffer` before any FFI call; one shared private validator used by all three families (ARC-10).
- **Acceptance:** r2 / t8c return `Err(InvalidBuffer)` on every rank that holds a bad buffer; new integration test per family; valgrind clean.

### SND-07 — V-collectives do not validate `counts`/`displs` against communicator size or buffer length

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `src/comm/v_collective.rs:62` (`gatherv`), `:130` (`scatterv`), `:197` (`allgatherv`), `:265` (`alltoallv`), and the `i*`/`*_init` variants at `:613`, `:676`, `:738`, `:807`; false SAFETY comment at `:200-204` ("ensures the arrays are long enough"); C shims `csrc/ferrompi.c:1567-1621`, `2118-2220`, `2950-3080` take the arrays with **no length**.
- **Defect:** only `counts.len() == displs.len()` is checked. MPI reads exactly `comm.size()` entries from each array and writes wherever `displs[i]` points; negative counts are passed through.
- **Evidence:** `repros/soundness/src/bin/r3_vcoll_short_counts.rs` — `allgatherv(&send, &mut recv, &[], &[])` on 2 ranks → **SIGSEGV**; `repros/c-shim/src/bin/t8_gatherv_overflow.rs` — valgrind "Invalid write of size 8 … 0 bytes after a block of size 8"; `t8b_gatherv_short_counts.rs` — invalid read past the counts array.
- **Fix direction:** require `counts.len() == size` and `displs.len() == size` (each array that MPI reads on this rank), every `count >= 0`, and `max(displs[i] + counts[i]) <= buf.len()` computed in `i64` without overflow; only on ranks where MPI reads the argument (root for gatherv/scatterv). Correct the SAFETY comment.
- **Acceptance:** r3/t8/t8b return `Err(InvalidBuffer)`; valgrind clean; tests for empty, short, negative, overlapping-past-end cases.

### SND-08 — `*_custom` point-to-point: unbounded `T` and unchecked datatype extent

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x (soundness fix; bound per D-7)
- **Locations:** `src/comm/p2p.rs:389` (`send_custom<T>`), `:445` (`recv_custom<T>`), `:511` (`isend_custom<T>`), `:571` (`irecv_custom<T>`); false doc claims at `:362-365`, `:415-417` ("mismatch → well-defined `MPI_ERR_TRUNCATE` … not memory unsafety"); `src/datatype_builder.rs:84-87` (`CustomDatatype` does not record size/extent).
- **Defect:** (a) count passed is `buf.len()` elements *of the derived datatype*, whose extent is never compared to `size_of::<T>()` → MPI reads/writes past the slice when extent > size_of::<T>(); (b) `T` has **no bound at all**, so raw wire bytes land in types with invalid bit patterns (`Box`, `&T`, `bool`, enums).
- **Evidence:** `repros/soundness/src/bin/r4_custom_dt.rs` — (A) `contiguous(32, U8)` + `recv_custom(&mut recv[..1])` → **31 canary bytes overwritten with 0x41**; (B) `recv_custom(&mut [Box::new(5u64)])` → Box pointer becomes `0x4141414141414141`, drop → **`free(): invalid pointer`, SIGABRT**.
- **Fix direction:** record `size`/`extent` at commit (`MPI_Type_size`, `MPI_Type_get_extent`) inside `CustomDatatype`; reject when `extent != size_of::<T>()` (or require `buf.len() * size_of::<T>() >= count * extent`); bound `T` so only plain-old-data types are accepted. **Decided (D-7, 2026-09-24):** new public `unsafe trait PlainData: Copy + 'static`, implemented by users for their `#[repr(C)]` types and blanket-implemented for all `MpiDatatype` types, plus the size/extent check.
- **Acceptance:** r4 (A) returns `Err(InvalidBuffer)`; r4 (B) no longer compiles; doc claims corrected.

### SND-09 — RMA target range and origin count are not validated (remote out-of-bounds write)

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x (per D-11)
- **Locations:** `src/window.rs:1586` (`put`), `:1776` (`get`), `:1974` (`accumulate`), `:2201` (`get_accumulate`) + `rput/rget/raccumulate`; C `csrc/ferrompi.c:4079-4212` forwards verbatim.
- **Defect:** neither ferrompi nor MPICH checks that `origin.len()` matches `target_count` or that `target_disp + target_count` fits the target rank's exposed window.
- **Evidence:** `repros/soundness/src/bin/r7b_rma_range_canary.rs` — `put(&[u64; 8], rank 1, disp 2, count 8)` into a 4-element window: **6 words past the window overwritten** on the remote process.
- **Fix direction:** reject `origin.len() != target_count` locally (the count is redundant anyway — ARC-17); validate `target_disp + target_count <= local_len` when the window is symmetric and known (`Win::allocate`, `SharedWindow` via `shared_query`); **decided (D-11, 2026-09-24):** window creation (already collective) allgathers each rank's exposed length once (8·P bytes per window), so every put/get/accumulate is bounds-checked against the target rank, for `Win::create` and `Win::allocate` alike.
- **Acceptance:** r7b returns `Err`; remote memory untouched (canaries intact).

### SND-10 — Window memory used after `MPI_Finalize` (dangling `local_slice`)

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x (per D-8)
- **Locations:** `csrc/ferrompi.c:754-761` (finalize sweep `MPI_Win_free`s every live window); `src/window.rs:414`, `:457` (`SharedWindow::local_slice`), `:980-1022`, `:1035` (`Win`); `src/lib.rs:700-710` (`Mpi::drop`).
- **Defect:** `Win`/`SharedWindow` are not tied to `Mpi`; dropping `Mpi` first frees/unmaps window memory while the window still hands out `&[T]` into it.
- **Evidence:** `repros/soundness/src/bin/r5_use_after_finalize.rs` — `drop(mpi)`, `win.local_slice()[0]` → **SIGSEGV**; `repros/c-shim/src/bin/t11_win_after_finalize.rs` — valgrind "Invalid read … inside a block of size 8,192 free'd by PMPI_Win_free ← ferrompi_finalize (ferrompi.c:758)".
- **Fix direction — decided (D-8, 2026-09-24):** finalize does not free windows that are still alive; their memory is leaked and a warning is written to stderr (MPI requires windows to be freed before finalize, so only erroneous programs are affected, and freed memory is never exposed). Rejected: accessors panicking after finalize; refcount-deferred finalize. **Extended (council A1):** Open MPI 4.1/5.0 free and unmap live windows inside `MPI_Finalize` itself, so `MPI_Finalize` is skipped (stderr warning) whenever an MPI-allocated window (`Win::allocate`, `SharedWindow`) is still alive at `Mpi` drop.
- **Acceptance:** r5/t11 no longer read freed memory (valgrind clean).

### SND-11 — `Communicator` is `Send + Sync` regardless of the provided thread level

- **Severity:** critical · **Verified:** repro · **Target:** 0.5.x runtime check (accepted, D-3); type-level encoding deferred
- **Locations:** `src/comm/mod.rs:79-80` (`unsafe impl Send/Sync`), its SAFETY comment `:67-78` ("users must serialize"); `src/lib.rs:377-379` (`Mpi::init()` defaults to `Single`); same pattern for `Group` (`src/group.rs:119-120`), `CustomDatatype` (`src/datatype_builder.rs:96-97`), auto-`Send+Sync` `Request`/`PersistentRequest`.
- **Defect:** safe code can share `&Communicator` across threads and call MPI concurrently under `Single`/`Funneled`/`Serialized` — an erroneous MPI program that races inside the MPI library. The only operation the types restrict (`Mpi: !Send`) is finalize.
- **Evidence:** `repros/soundness/src/bin/r12_thread_single.rs` — `Mpi::init()` (provided `Single`), 4 scoped threads doing `irecv/send/wait` on `&world`: **`corrupted message` assertion (e.g. 1415 vs 1417) and hangs**, reproducible across runs; control `r12b_thread_multiple.rs` (`Multiple`) passes 3/3.
- **Fix direction (accepted):** record the init thread id and provided level in the C/Rust layer; below `Serialized`, any MPI entry point called from a non-init thread returns an error (new `Error` variant, e.g. `ThreadLevelViolation`). **Decided (D-9, 2026-09-24):** `Serialized` overlap detection (atomic in-call flag) is compiled only under `debug_assertions`; release builds pay nothing.
- **Acceptance:** r12 returns `Err` on worker threads instead of corrupting; r12b unchanged; overhead on the single-thread fast path measured (< 1 ns target).

### SND-12 — Uninitialised window memory exposed as `&[T]`

- **Severity:** major (UB by Rust rules; every `MpiDatatype` is POD so no invalid values, but reading uninit memory is UB) · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:3871-3892` (`MPI_Win_allocate`), `:3919-3942` (`MPI_Win_allocate_shared`); `src/window.rs:414`, `:457`, `:980-1022`, `:1035`.
- **Evidence:** `repros/c-shim/src/bin/t13_rma_trunc.rs` prints fresh `Win<f64>` elements like `1.012e-320`, `2e-323`.
- **Fix direction:** zero the local segment in C right after allocation (shared windows: each rank zeroes its own segment, then `MPI_Win_sync` + barrier before returning — decide in plan whether the constructor already synchronises).
- **Acceptance:** fresh windows read as zero; valgrind (`--track-origins`) reports no uninitialised reads.

### SND-13 — `SharedWindow` slices alias memory other processes write concurrently (observed miscompile)

- **Severity:** critical · **Verified:** repro + disassembly · **Target:** 0.6 (with ARC-13); 0.5.x docs warning
- **Locations:** `src/window.rs:457-548` (`local_slice`, `local_slice_mut`, `remote_slice`).
- **Defect:** `remote_slice(&self) -> &[T]` promises immutability for the borrow's lifetime, but other ranks write that memory between/within epochs. The optimiser is entitled to assume the slice does not change.
- **Evidence:** `repros/soundness/src/bin/r11_shm_race.rs`, output `repros/evidence/r11.out` — rank 1 polls a flag through the slice and **never observes rank 0's write** during the 3 s wait: the release binary compiles the loop to a single `cmpq $0x0,(%rcx)` followed by a `jmp` to itself (load hoisted out of the loop → infinite spin).
- **Fix direction:** scope slices to an epoch guard (reads only after the synchronising call), or return a view type built on raw pointers with volatile/atomic element accessors (`read(i)`, `write(i, v)`), decided together with the `SharedWindow`→`Win` merge (ARC-13). Until then, document that polling through the slice is unsupported.
- **Acceptance:** r11 terminates with the flag observed.

### SND-14 — `UserOp` fat-pointer transmute relies on unspecified layout

- **Severity:** minor · **Verified:** reading · **Target:** 0.5.x (internal; part of BLT-15)
- **Locations:** `src/op.rs:172-185`, `:253`, `:411` (`Box<dyn Fn>` split into `[*mut (); 2]` and transmuted back).
- **Defect:** Rust does not guarantee the `(data, vtable)` order/layout of trait-object pointers.
- **Fix direction:** store a thin pointer: `Box<Box<dyn Fn…>>` → `*mut c_void` (one pointer across FFI, no transmute); also removes the second registry (BLT-15).
- **Acceptance:** no `transmute` in `op.rs`; `test_user_op` passes under Miri-free review + MPI suite.

### SND-15 — `fetch_and_op`/`compare_and_swap` result pointer derived from a shared borrow

- **Severity:** minor · **Verified:** reading · **Target:** 0.5.x
- **Locations:** `src/window.rs:2329` (`fetch_and_op`), `:2481` (`compare_and_swap`) — both `(result_box.as_ref() as *const MaybeUninit<T> as *mut MaybeUninit<T>)`.
- **Defect:** the result pointer MPI writes through is obtained by casting away constness from `result_box.as_ref()`, a shared reference. A write through a pointer derived from a shared borrow is undefined behaviour in Rust, independent of whether the write happens to work in practice.
- **Fix direction:** `PendingFetchResult.result` becomes a raw-owned `NonNull<MaybeUninit<T>>` obtained from `Box::into_raw`, never derived from a reference; that pointer, unchanged, is what MPI receives. A `Drop` impl frees the allocation exactly once, on the `resolve` path, the drop-without-resolve path, and the FFI error path.
- **Acceptance:** no cast from a shared borrow feeds a write pointer in `window.rs`; a non-MPI unit test constructs `PendingFetchResult` values, resolves one and drops another unresolved, with no double free; `PendingFetchResult`'s `Send`/`Sync` auto traits are unchanged.

### SND-16 — `Mpi` drop racing a concurrent guarded call at `Serialized`/`Multiple` reaches MPI after finalize

- **Severity:** major · **Verified:** reading · **Target:** 0.6
- **Locations:** `src/rt.rs:61` (`finalize`'s `STATE` store to `FINALIZED`), `:77` (`enter`'s `STATE` load), `:97` (`drop_guard`'s `STATE` load).
- **Defect:** at `Serialized`/`Multiple`, a worker thread's guarded call (through `enter` or `drop_guard`) can read `Active` from `STATE` just before the init thread's `Mpi::drop` stores `FINALIZED`, then reach MPI just after `ferrompi_finalize` runs. No ordering on `STATE` closes this window; it is currently a caller contract (`Mpi` must not be dropped while another thread is inside an MPI call) that `ferrompi` does not enforce.
- **Fix direction:** in-flight call tracking in `rt::enter`/its matching exit (an epoch or counter), with `finalize` waiting for it to reach zero before calling `MPI_Finalize`; needs an ADR-033 amendment and an R29 re-measurement of the fast-path overhead under `Multiple`.
- **Acceptance:** a concurrent-finalize stress repro no longer reaches MPI after `MPI_Finalize` returns, at both `Serialized` and `Multiple`.
