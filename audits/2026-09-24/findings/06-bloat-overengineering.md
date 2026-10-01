# 06 — AI bloat and overengineering

The maintainer treats bloat and overengineering as **critical defects**. This
lens estimates removable/collapsible text and flags speculative machinery.
Estimates are lines of code/text; "risk" is what is load-bearing.

**Total:** ~6,600 lines (~14% of ~45.6k tracked lines: 31.8k non-example + 13.8k examples).

BLT numbers are non-contiguous on purpose: items owned by another lens were
given that lens's ID and are listed in "Tracked elsewhere" at the end.

Target legend: **0.5.x†** = non-breaking, eligible for 0.5.x (in 0.5.x scope per D-6);
**0.6** = touches public API; **→ X** = tracked under another ID.

| Category | Lines | IDs |
|---|---:|---|
| Test theater | ~1,050 | BLT-01, BLT-26, BLT-32 |
| Comment bloat | ~1,150 | BLT-02, BLT-08, BLT-21 |
| Example / test / bench / CI sprawl | ~1,450 | BLT-04, BLT-06, BLT-09, BLT-10, BLT-16, (INF-16) |
| Overengineering / speculative / dead | ~1,240 | BLT-15, BLT-23, BLT-25, BLT-27, BLT-29, (ARC-12, ARC-13, ARC-14, ARC-17, PRF-07) |
| Dead C stubs / C duplication | ~855 | BLT-03, BLT-14, BLT-20 |
| Doc bloat / staleness | ~930 | BLT-11, (DOC-05, DOC-07, DOC-11, DOC-12) |

---

### BLT-01 — 76 of 181 `#[test]` fns in `src/` catch no regression; 16 "compile witness" fns

- **Target:** 0.5.x† · **~1,000 lines** · **Risk:** low
- **Where (no-value tests):** `src/window.rs:2790-3114` (28 of 32: `lock_type_*`×3 / `win_kind_*`×3 derived-trait tests, 11 `*_signature_compiles`, 3 `*_forget_does_not_drop`, 8 `win_{fence,pscw}_assert_{default_is_none,none_constructor_is_zero,debug_is_implemented,copy_and_clone}`); `src/lib.rs:716-989` (10 of 15); `src/error.rs:428-757` (7 of 16); `src/datatype.rs:424-683` (9 of 13); `src/group.rs:580-752` (5 tests + 7 witness fns + a `const _`); `src/status.rs:34-63` (2/2); `src/op.rs:502-512` (1/1); `src/comm/mod.rs:157-210` (3/4); `src/comm/persistent.rs` (5 `*_signature_compiles`); `src/comm/p2p.rs:600-645`, `src/comm/mgmt.rs:280-293` (5 witness fns); `raw_handle_returns_*` / `new_request_is_*` in `request.rs`, `persistent.rs`, `datatype_builder.rs`; `src/comm/blocking.rs:1292` (`allreduce_indexed_error_tag_renders_correctly` builds an `Error::Mpi` literal, never calls the method).
- **Evidence:** verbatim duplicates `reduce_op_repr_values` (`lib.rs:764`) = `reduce_op_all_variants_match_c_switch` (`:833`, touches no C), `replace_noop_discriminants` (`:860`) is a subset; `DatatypeTag::Byte == 13` asserted in 3 tests; names promising unchecked behaviour — `supports_create_from_group_does_not_cache_probe_failures` is `let _: fn() -> bool = …` ("verified by the acceptance grep"), `indexed_and_primitive_traits_are_disjoint` never checks disjointness, `from_code_with_op_sets_operation_field` never calls `from_code_with_op`; `buffer_attach_invalid_buffer_variant_exists` (`lib.rs:875`) = `assert!(matches!(Error::InvalidBuffer, Error::InvalidBuffer))`. Compile witnesses duplicate what ~78 examples and `no_run` doctests already compile in CI.
- **Keep:** one merged discriminant test per `#[repr(i32)]` enum shared with C (`ThreadLevel`, `ReduceOp`, `DatatypeTag`, `GroupComparison`, `SplitType`) until ARC-01/ARC-03 remove the contract; `indexed_datatype_struct_layouts`, `rank_range_repr_and_size` (ABI layout); buffer attach/detach state tests; `create_from_group_null_byte_in_tag`; sentinel mapping + `from_code(0)`; the 41 `*_mismatched_*_returns_invalid_buffer` / `*_inplace_nonroot_*` tests (real per-function checks); slurm/info validation tests; `WinFenceAssert`/`WinPscwAssert` OR tests.

### BLT-02 — Private C header restates every signature in Doxygen

- **Target:** 0.5.x† · **~800 lines** · **Risk:** low
- **Where:** `csrc/ferrompi.h` (2,171 lines, ~1,532 comment lines, 174 `/** */` blocks, 477 `@param`, 115 identical `@return MPI error code`); typical block `ferrompi.h:604-616`; inconsistent with one-line comments at `:676-722`.
- **Why removable:** the header is consumed only by `ferrompi.c` and mirrored by hand in `src/ffi.rs`; there is no external C API.
- **Keep:** non-obvious ABI contracts (−1 for `MPI_COMM_NULL`/`MPI_UNDEFINED`, `MPI_ERR_OTHER` stubs on MPI < 4, `is_root` semantics, sentinel codes, buffer-lifetime notes).

### BLT-03 — Every `MPI_VERSION < 3` branch is unreachable

- **Target:** 0.5.x† · **~455 lines** (C ~260, examples ~180, Rust ~15) · **Risk:** none
- **Where:** C stubs `csrc/ferrompi.c:2387-2425` (persistent p2p), `:2499-2520` (buffer attach/bsend), `:4238-4436` (all RMA stubs); Rust fallback comments `src/window.rs:99-101, 108-110, 225-229, 236-238`; probes in 12 examples (e.g. `examples/test_rma_put.rs:22-37`, `if major < 3 { … SKIP … }`).
- **Evidence:** the file calls MPI-3 functions **without** guards — `MPI_Comm_split_type` (`:879`), `MPI_Iallreduce` (`:1683`), `MPI_Ibarrier` (`:1859`), `MPI_Get_library_version` (`:4444`) — so it cannot compile against MPI < 3. The gating is also wrong: `MPI_Send_init`/`MPI_Buffer_attach` are MPI-1.1 (section header `:2427` says "(MPI 1.1+)" then gates `>= 3`).
- **Keep:** the `MPI_VERSION >= 4` stubs (`:3114-3259`) — live, Open MPI 4 is MPI 3.1.

### BLT-04 — 12 near-identical in-place example binaries

- **Target:** 0.5.x† · **~450 lines** · **Risk:** low (keep every assertion)
- **Where:** `examples/test_{gather,allgather,scatter,alltoall}_inplace.rs`, `test_i{…}_inplace.rs`, `test_{…}_init_inplace.rs` (713 lines total). `diff test_gather_inplace.rs test_igather_inplace.rs` differs only in `world.igather_inplace(...)?; req.wait()?`. `examples/test_collectives.rs` already folds 21+ ops into one binary.
- **Fix:** one `test_inplace.rs` (~250 lines); each binary also costs a link + an `mpiexec` launch.

### BLT-06 — Example scaffolding duplicated in 26 files

- **Target:** 0.5.x† · **~390 lines** · **Risk:** low
- **Where:** 26 `examples/test_*.rs` repeat a `local_ok` flag + `allreduce_scalar(local_ok as i32, ReduceOp::Min)` sentinel + `println!("====…All … tests passed!")` banner (274 `local_ok` references), e.g. `test_rma_put.rs:117-136`. `benches/common/mod.rs` exists; examples have no shared module.
- **Fix:** `examples/common/mod.rs` (`#[path]` include) with `check(world, ok, name)`; tests should fail the process (non-zero exit) on any rank's failure.

### BLT-08 — Boilerplate SAFETY comments + repeated FFI marshalling; 59 `unsafe` blocks have none

- **Target:** 0.5.x† · **~300 lines** · **Risk:** low
- **Where:** 173 SAFETY blocks (620 lines) in `src/`; "T::TAG matches T's MPI datatype per ADR-0003" ×33, "…outlive this blocking call" ×26; history-narrating SAFETY at `src/window.rs:2342-2343, 2492-2493` ("invariants are unchanged from the previous implementation"); marshalling triplet `.cast::<std::ffi::c_void>()` ×134, `T::TAG as i32` ×89, `len() as i64` ×88. Meanwhile 59 of 232 `unsafe {}` blocks lack SAFETY (e.g. `src/lib.rs:402, 441, 453, 467, 482, 489, 704`) — violates the maintainer's Rust standard.
- **Fix:** one private helper `fn buf<T: MpiDatatype>(s: &[T]) -> (*const c_void, i64, i32)` (+ `_mut`) carrying the single invariant; a real SAFETY comment on every remaining `unsafe` block (consider `#![deny(clippy::undocumented_unsafe_blocks)]`).
- **Keep:** distinct invariants (aliasing in `get_accumulate`, `PendingFetchResult` boxing, `op.rs` pointer handling).

### BLT-09 — Three hand-maintained test runners that drifted

- **Target:** 0.5.x† · **~200 lines** · **Risk:** low
- **Where:** `tests/run_mpi_tests.sh` (288 lines, runs 73 examples), `tests/run_mpi_coverage.sh` (258 lines, runs **11**), `test.sh` (17 lines, self-declared deprecated). `run_test()`, colour setup and box header copied verbatim; `gatherv`, `pi_monte_carlo`, `scan`, `topology` built but run by neither.
- **Fix:** one runner globbing `examples/test_*.rs` (np from a header comment), shared by coverage; delete `test.sh`. Correctness defects of the runner: INF-01, INF-09.

### BLT-10 — 42 of 58 `[[example]]` entries in `Cargo.toml` are redundant

- **Target:** 0.5.x† · **~185 lines** · **Risk:** none
- **Where:** `Cargo.toml:17-263`. 42 entries lack `required-features` (auto-discovered); 20 other examples are not listed and build fine; the 16 `rma` entries need only `name` + `required-features`.

### BLT-11 — Same tables/indexes repeated across 3–5 documentation surfaces; marketing tone

- **Target:** 0.5.x† · **~250 lines** · **Risk:** none
- **Where:** thread-level table (`src/lib.rs:99-104`, `README.md:357-364`, `docs/architecture.md:123-128`); feature table (`lib.rs:50-53`, `README.md:63-68`); SLURM sbatch script (`lib.rs:159-167`, `README.md:403-410`); ADR index ×5 (`lib.rs:178-187`, `src/doc.rs`, `docs/README.md:20-36`, `docs/architecture.md:404-428`, the ADR files); README "Features" (14-24) vs "Why FerroMPI?" (26-47); "⚡ Fast: Zero-cost abstractions".
- **Fix:** one canonical place each (rustdoc for API facts, README for pitch + quick start). Factual errors in these tables: DOC-08, DOC-09.

### BLT-14 — 15 C in-place shims differ only by `MPI_IN_PLACE`; dead `is_root` parameter

- **Target:** 0.5.x† (private ABI) · **~300 lines** · **Risk:** low; optional collapse
- **Where:** `ferrompi_{allreduce,reduce,gather,allgather,scatter,alltoall}_inplace` + `i*` + `*_init` (15 distinct, 20 with stubs). `ferrompi_allreduce_inplace` (`csrc/ferrompi.c:1310-1321`) = `ferrompi_allreduce` with `MPI_IN_PLACE`. `ferrompi_gather_inplace`'s `is_root` is always `1` from Rust (`src/comm/blocking.rs:788`) yet C branches on it (`:1330-1333`).
- **Fix:** C maps a NULL send pointer to `MPI_IN_PLACE`; at minimum delete the dead `is_root`.

### BLT-15 — `UserOp` keeps its closure in two registries and runs a dead lookup per reduction

- **Target:** 0.5.x† (internal) · **~120 lines** · **Risk:** low
- **Where:** C `csrc/ferrompi.c:4805-4821` (`ferrompi_tag_from_mpi_dt`, 14-way compare on every callback — verified by main session at `:4833`), `:4770`, `:4830-4837`; Rust `src/op.rs:51-137` (`REGISTRY`), `:413-418`, `:167` (`_dt_tag` ignored). Same fat pointer stored in C (`op_closure_data/vtbl`) and in Rust `REGISTRY` (only for drop/rollback). Stale comments: `op.rs:61-73` ("OnceLock is used" — it isn't), `:413` (trampoline loads that don't exist). Transmute to `[*mut (); 2]` → SND-14.
- **Fix:** one registry holding a thin `*mut c_void` (`Box<Box<dyn Fn…>>`); delete the tag lookup.
- **Keep:** the 16 per-slot trampolines (`MPI_User_function` has no user-data argument).

### BLT-16 — Benches that measure nothing actionable; duplicated bench protocol

- **Target:** 0.5.x† · **~150 lines + ~60 README** · **Risk:** none
- **Where:** `benches/bench_smoke.rs` times `black_box(1u64 + 1)`; `benches/common/mod.rs:51-56` `rank_zero_only` is `#[allow(dead_code)]` with no callers; sentinel/follower/stop protocol (~50 lines) copied into three benches; no CI job runs benches. Methodology defects: PRF-03, PRF-04.

### BLT-20 — Six copies of the slot-claim loop in C

- **Target:** superseded by ARC-01 (0.7); collapse only if tables survive · **~70 lines**
- **Where:** `alloc_comm` `csrc/ferrompi.c:263`, `alloc_win` `:385`, `alloc_info` `:432`, `alloc_group` `:471`, `alloc_datatype` `:526`, `alloc_op_slot` `:4776` — identical modulo names and slot-0 skipping.

### BLT-21 — Process artifacts, stale line numbers and expired promises in comments

- **Target:** 0.5.x† · **~45 lines** · **Risk:** none
- **Where:** 15 hits in `src/`+`csrc/` (grep `epic|ticket|F2-0|PERF-0`): `src/lib.rs:331,339` "(Epic 7)", `:873` "deferred to epic-06 follow-up", `:951` "verified by the acceptance grep"; `src/window.rs:859,1067` "tickets 053–056"; `src/datatype.rs:168` "that is Epic 6" (shipped); `src/comm/mgmt.rs:55` "a future epic"; `csrc/ferrompi.h:179` "(ticket-050)"; `csrc/ferrompi.c:30,32` F2-002/F2-006, `:631` "pre-ticket-008", `:940` "epic-02 invariant"; `src/persistent.rs:49`, `src/request.rs:15,32` PERF-02/03. 16 more in `examples/`+`benches/` (e.g. `test_get_group_invalid_handle.rs:4,9,41` "ticket-012"; `test_rma_win_flush_sync.rs:5`). Stale line refs: `csrc/ferrompi.c:129-136` cites ~4686/4568/4699/4726 (now 4910/4792/4923/4970); `:498-500` cites ~723/~743/3216-3338 (now 899/919/3326-3530); `src/request.rs:474` "line 63". Expired: `src/request.rs:77-79, 426-427` "planned for v0.5" (crate is 0.5.0); `csrc/ferrompi.c:929-934` "ADR-0005 for the v0.5 follow-up" (ADR-0005 is about `MPI_Op_create`).

### BLT-23 — `Group::undefined()` makes an unsafe FFI call to a C function returning literal −1

- **Target:** 0.5.x† · **~35 lines** · **Risk:** none (keep `Group::undefined()` as `const fn` → −1)
- **Where:** `src/group.rs:179-205`, `csrc/ferrompi.c:3319-3324` (`int32_t ferrompi_mpi_undefined(void) { return -1; }`), `ferrompi.h:1715`, ffi decl; `Communicator::UNDEFINED` already `pub const = -1`.

### BLT-25 — Small dead C branches

- **Target:** 0.5.x† · **~30 lines**
- `install_errors_return_win` (`csrc/ferrompi.c:617-650`) always returns `MPI_SUCCESS`; its 20-line comment describes unreachable error returns; the window constructors' failure branches (`:3880-3884, 3905-3909, 3930-3934`) are dead. `ferrompi_comm_create_from_group` passes `MPI_ERRORS_RETURN` to MPI and then installs it again "defensively" (`:936-941`).

### BLT-26 — `const _` Send/Sync assertions next to the `unsafe impl` they assert

- **Target:** 0.5.x† · **~27 lines**
- `src/comm/mod.rs:147-155` (after `unsafe impl` at `:79-80`), `src/group.rs:585-593`, `src/datatype_builder.rs:375-383` — an explicit `unsafe impl` cannot fail them.

### BLT-27 — Dead `ferrompi_init`, hidden by a module-wide `#![allow(dead_code)]`

- **Target:** 0.5.x† · **~25 lines**
- `csrc/ferrompi.c:693-708`, `ferrompi.h:74-78`, `src/ffi.rs:24` — the only one of 172 externs with no caller (`Mpi::init` → `init_thread(Single)`). Remove `#![allow(dead_code)]` at `src/ffi.rs:5` so the compiler flags future dead externs.

### BLT-29 — `with_handles` / `HANDLE_STACK_CAP` implemented twice

- **Target:** 0.5.x† (or removed by ARC-01) · **~20 lines**
- `src/request.rs:11-29` and `src/persistent.rs:51-69` identical modulo element type.

### BLT-32 — `ReduceOp` `compile_fail` doctest assembled from 14 `#[cfg_attr(…, doc = …)]` lines

- **Target:** moot once ARC-02 makes `Replace`/`NoOp` unconditional (0.6); delete then · **~16 lines**
- `src/lib.rs:272-289` only proves `#[cfg(feature = "rma")]` works.

### BLT-34 — Test modules use `use super::*`

- **Target:** 0.5.x† (with BLT-01 cleanup) · **Risk:** none
- 11 files (`src/error.rs`, `topology.rs`, `datatype.rs`, `status.rs`, `info.rs`, `op.rs`, `window.rs`, `slurm.rs`, `request.rs`, `persistent.rs`, `lib.rs`) — the maintainer's Rust standard asks for explicit imports in test modules.

### Tracked elsewhere (listed so the bloat total is complete)

| Bloat item | ~Lines | Tracked as |
|---|---:|---|
| `Info` has no consumer | 560 | ARC-12 (D-4) |
| `SharedWindow` duplicates `Win`; 4 guard types | 350 | ARC-13 |
| `WinKind` dead field + enum | 30 | ARC-13 |
| `numa` feature with no NUMA code | 10 | ARC-14 |
| RMA redundant `target_count`, duplicate tags | 60 | ARC-17 |
| `[profile.release]` comment/claim | 17 | PRF-07 |
| MPICH hotfix copy-pasted ×4 in CI | 75 | INF-16 |
| `build.rs` dead outputs | 15 | INF-05 |
| CONTRIBUTING template filler | 150 | DOC-11 |
| ADRs as plans/changelogs | 220 | DOC-05 |
| Compatibility matrix mostly "?" | 120 | DOC-07 |
| CHANGELOG restating rustdoc + internal IDs | 150 | DOC-12 |
| False statements in docs | 40 | DOC-01, INF-02 |

## Looks bloated but is justified — keep

- The 16 C op trampolines (`MPI_User_function` has no user-data argument).
- The public blocking / `i*` / `*_init` method trios (macro generation would hurt rustdoc; share *validation* instead — ARC-10).
- Every `MpiErrorClass` variant (faithful mapping; the *decoding* is the bug, COR-01).
- The four sealed traits (each gates a distinct safety property).
- The `MPI_VERSION >= 4` stubs (Open MPI 4 = MPI 3.1).
- `compile_fail` doctests in `src/op.rs:36` (`Send + Sync` bound) and `src/datatype.rs:352` (seal).
- `TopologyInfo` / `slurm.rs` — small, self-contained; remove only if no downstream crate uses them.
- Note: the bloat reviewer also listed "hand-written `ffi.rs` + handle tables" as justified; overridden for the tables by ARC-01 (the hand-written FFI itself remains justified until the Rust-native ABI backend).
