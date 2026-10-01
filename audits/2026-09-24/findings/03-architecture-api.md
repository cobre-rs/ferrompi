# 03 — Architecture and public API

Structural issues. Each carries a **reversibility** tag: *one-way* (public API,
semver, format) or *two-way* (internal). MPI 5 ABI support is a stated future
direction (see [04-mpi5-abi](04-mpi5-abi.md)); one-way doors are judged against it.

---

### ARC-01 — The C handle tables are the root cause of several defects

- **Severity:** major · **Reversibility:** two-way (internal) but large · **Target:** 0.7 (accepted, D-2), before the ABI backend
- **Locations:** `csrc/ferrompi.c:21-153` (7 tables + caps: `MAX_REQUESTS 16384`, `MAX_COMMS 256`, `MAX_WINDOWS 256`, `MAX_DATATYPES/GROUPS/INFOS 64`, `MAX_OPS 16`), `:226-560` (alloc/get/free per table), `:710-778` (finalize sweep); `src/error.rs:31-68, 263-273` (`ResourceKind`, `ResourceExhausted`); `docs/adr/0001-why-c-wrapper.md:125-130, 434-442`; `docs/adr/0002-handle-tables.md`.
- **What the tables cause:**
  - COR-02 handle ABA (slot reuse without generation);
  - PRF-01/PRF-02: ~14 ns and two locked RMWs per nonblocking request, contention concentrated on one cache line;
  - fixed caps that downstream cannot configure (16 user ops, 64 datatypes/groups/infos);
  - the finalize sweep and its erroneous frees (COR-08, SND-10);
  - public-API leakage (ARC-03).
- **Why they exist (ADR-0001 driver 1):** "a single ferrompi build links against MPICH, Open MPI and Cray without recompilation". **False:** `build.rs` compiles the shim against one `mpi.h`; switching MPI requires a rebuild either way. What the tables really provide is a uniform integer handle type on the Rust side — obtainable without tables.
- **Direction (accepted):** store MPI handles **by value in Rust** as opaque 8-byte values (`_Static_assert(sizeof(MPI_Comm) <= 8)` etc. — MPICH ints, Open MPI pointers and MPI-5 ABI pointers all fit); the C shim becomes stateless (memcpy in/out); late drops handled by the global finalized flag (COR-07). Removes ABA, caps, sweep, table cost, and makes the Rust-native ABI backend an internal swap.
- **Note:** the bloat reviewer recommended keeping the tables (ADR-0001 rationale); the evidence above (false driver + ABA + perf + ABI) overrides it.
- **Acceptance:** no `*_table` in `ferrompi.c`; `ResourceKind` table variants removed or deprecated; PRF-01 overhead < 2 ns/request; COR-02 repros impossible by construction.

### ARC-02 — Non-additive `rma` feature and no `#[non_exhaustive]` anywhere

- **Severity:** major · **Reversibility:** one-way (cheapest now, in 0.x) · **Target:** 0.6
- **Locations:** `src/lib.rs:290-347` (`#[cfg(feature = "rma")] Replace = 12, NoOp = 13` on `ReduceOp`; the C switch already accepts 12/13 unconditionally); `docs/architecture.md:267-273`; zero `#[non_exhaustive]` in `src/` (grep).
- **Defect:** enabling `rma` anywhere in a dependency graph adds variants → any downstream exhaustive `match` on `ReduceOp` stops compiling (feature unification). Every public enum/pub-field struct is a breaking-change trap: `Error`, `MpiErrorClass`, `ResourceKind`, `ReduceOp`, `DatatypeTag`, `SplitType` (1 variant; MPI-4 adds `HW_GUIDED`), `ThreadLevel`, `WinKind`, `Status`, `StructField`, `RankRange`, `HostEntry`, `SlurmInfo`.
- **Fix direction:** make `Replace`/`NoOp` unconditional; add `#[non_exhaustive]` to the enums above and to pub-field structs (give `StructField` a constructor). Leave `GroupComparison` and `LockType` exhaustive (MPI defines them closed). Verified: `#[non_exhaustive]` still allows cross-crate `as i32` casts on rustc 1.95 (`repros/api-checks/`).
- **Acceptance:** `cargo build --features rma` and default build expose identical enum shapes; a downstream crate with an exhaustive match gets the non-exhaustive lint instead of a break.

### ARC-03 — The C-shim representation leaks into the public API

- **Severity:** major (conflicts with MPI-5 ABI direction) · **Reversibility:** one-way · **Target:** 0.6 (deprecations can start in 0.5.x)
- **Locations:** `raw_handle()` on 8 types — `src/comm/mod.rs:117-120` (i32), `src/request.rs:116-119` (i64), `src/persistent.rs:97-100` (i64), `src/group.rs:174-177`, `src/info.rs:194-199` (-1 = null), `src/datatype_builder.rs:346-351`, `src/window.rs:666-672`, `:1062-1070` (docs claim "raw MPI window handle … for custom FFI calls"); `DatatypeTag` `#[repr(i32)]` declared a "semver contract" (`src/datatype.rs:67-110`, `docs/architecture.md:257-265`, ADR-0003); public `MpiDatatype::TAG`; `StructField.basetype: DatatypeTag` (pub field, `src/datatype_builder.rs:56-63`); public `Error::from_code/check/check_with_op` that understand shim-only sentinels `-7001..-7007` (`src/error.rs:31-37, 300-425`); `Communicator::UNDEFINED`.
- **Defect:** `raw_handle()` returns an index into a private table (the `ffi` module is private, no `from_raw` exists) — meaningless outside the crate and reusable by another object after slot recycling. Freezing discriminant values forbids the zero-cost ABI encoding (discriminant = ABI handle value, e.g. `F64 = 532`, `Sum = 33`).
- **Fix direction:** deprecate/`#[doc(hidden)]` the eight `raw_handle`s (examples use a hidden accessor); make `from_code`/`check*` `pub(crate)`; declare discriminants unspecified (internal `fn tag()`); do **not** add a native-handle interop API before an ABI backend exists (it would expose the implementation-specific type — the very thing ADR-0001 avoided).

### ARC-04 — Handles are not tied to `Mpi`'s lifetime

- **Severity:** major · **Reversibility:** runtime guard two-way (0.5.x); lifetime parameter one-way (decide with D-1) · **Target:** 0.5.x guards (COR-07, SND-10), 0.6 lifetime decision
- **Locations:** `src/lib.rs:433-435` (`Mpi::world(&self) -> Communicator` owned, no lifetime), `:700-710`; `docs/architecture.md:225-227` (claims "the borrow checker enforces lifetime containment" — it does not).
- **Defect:** any handle can outlive `Mpi`; consequences are SND-10 (memory) and COR-07 (aborts). The finalize sweep masks some cases (late comm/group/info/datatype/request drops become no-ops — VER-04).
- **Fix direction:** 0.5.x: global finalized flag + guards; 0.6: evaluate `Communicator<'mpi>` together with the buffer-safety scopes (D-1), which naturally borrow from `Mpi`.

### ARC-05 — Thread-safety model is inconsistent

- **Severity:** major · **Reversibility:** runtime check two-way; typestate one-way · **Target:** 0.5.x runtime check (accepted, D-3); typestate deferred
- **Locations:** `src/lib.rs:114-116, 364-369` (`Mpi` `!Send` "to pin finalize"), `src/comm/mod.rs:79-80`, `src/group.rs:119-120`, `src/datatype_builder.rs:96-97`, auto-`Send+Sync` `Request`/`PersistentRequest`; crate docs recommend `Funneled` for hybrid code while handing out `Sync` communicators.
- **Defect:** the only thing the types restrict is finalize, the least dangerous operation; concurrent calls under `Single/Funneled/Serialized` compile fine (SND-11).
- **Fix direction:** see SND-11.

### ARC-06 — Buffer-safety model (umbrella for SND-01..05)

- **Severity:** critical · **Reversibility:** one-way · **Target:** 0.6 (accepted, D-1)
- **Locations:** see SND-01..SND-05; `docs/adr/0004-persistent-collective-approach.md:227-283` (Option C rejected — its reasoning conflates lifetime `'a` with type `T` and claims one `'a` cannot cover two borrows; `Win<'a, T>` at `src/window.rs:844-866` already disproves "lifetimes are unworkable"); `docs/migrating-from-rsmpi.md:232-234, 307-308` (claims the compiler protects in-flight buffers).
- **Decision (accepted 2026-09-24):**
  - `PersistentRequest` and `Win::create` **own** their buffers; access only while inactive; `into_*` returns them.
  - Nonblocking `Request` and RMA epochs use **closure scopes** (a scope cannot be forgotten; scope end completes everything).
  - Rejected: lifetime-only handles (defeated by `mem::forget`, SND-05); `unsafe fn` constructors (honest but pushes UB burden onto every call site).
- **Follow-ups:** supersede ADR-0004 with a new ADR; rewrite the migration-guide section; `PendingFetchResult` folds into the epoch scope.

### ARC-07 — Reduction ops and datatypes are not parameters (method explosion, capability gaps)

- **Severity:** major · **Reversibility:** one-way · **Target:** 0.6
- **Locations:** `src/comm/blocking.rs:326-351` (`allreduce_with_op` — the **only** place a `UserOp` works), `:400-428` (`allreduce_indexed`), `:475-516` (`allreduce_bytes`); `src/comm/p2p.rs:389-595` (`CustomDatatype` only in p2p); `src/lib.rs:292-347` (`ReduceOp` mixes collective ops, accumulate-only `Replace/NoOp`, pair-only `MaxLoc/MinLoc`).
- **Defect:** no user op in reduce/scan/iallreduce/allreduce_init/accumulate — a reproducible compensated-sum `UserOp` cannot be used in the persistent/nonblocking allreduce that an SDDP loop needs; no custom datatype in any collective. Validity checked only in some methods: `allreduce_indexed` rejects non-MAXLOC ops, but `allreduce/iallreduce/allreduce_init` accept `MaxLoc` on `f64` and fail at runtime with `MPI_ERR_OP`.
- **Fix direction:** accept an `Op<'a, T>` parameter (`From<ReduceOp>`, `From<&UserOp<T>>`) in every reducing collective, and a datatype parameter in collectives; validate op/type compatibility in one place. Note: adding user ops to nonblocking/persistent paths makes the `MPI_Op_free`-before-closure-drop order unsafe (see VER-09) — the op must outlive every pending operation (borrow from the scope).

### ARC-08 — Error model misleads and loses context

- **Severity:** major · **Reversibility:** one-way · **Target:** 0.6
- **Locations:** `src/error.rs:250-277`; `InvalidOp` returned at `src/comm/blocking.rs:771`, `src/comm/nonblocking.rs:500`, `src/comm/persistent.rs:908` (non-root calling `gather_inplace`), `src/lib.rs:631,683` (buffer already/not attached), `src/datatype_builder.rs:114` (indexed base type) — while its message always says "Invalid reduction operation for this method"; `InvalidBuffer` without argument/expected/actual at ~40 sites; `Internal(String)` for user misuse at `src/persistent.rs:118,178` (double `start`), `src/lib.rs:563` (nul in tag), `src/info.rs:125,130,163`; `src/op.rs:352-357` hand-builds `Error::Mpi{class: Other, code: -7004}` bypassing `ResourceExhausted`; `src/datatype_builder.rs` docs promise class `Other` on table-full but code returns `ResourceExhausted`; `src/lib.rs:406-411` stores the MPI error code in the class field (`Raw(ret)`) on init failure.
- **Fix direction:** `#[non_exhaustive]` structured variants — e.g. `InvalidArgument { arg: &'static str, reason }`, `BufferSize { arg, expected, actual }`, `NotRoot { root, rank }`, `InvalidState`, `Finalized`, `ThreadLevelViolation`; route op-table exhaustion through `from_code` (0.5.x internal fix possible).

### ARC-09 — `Status` is always discarded

- **Severity:** major · **Reversibility:** one-way · **Target:** 0.6
- **Locations:** every completion shim passes `MPI_STATUS_IGNORE`/`MPI_STATUSES_IGNORE` (`csrc/ferrompi.c:3567-3860`); `Request::wait(self) -> Result<()>` (`src/request.rs:134`); blocking `recv` returns an `(i32, i32, i64)` tuple while `recv_custom`/`probe` return `Status`.
- **Defect:** a wildcard `irecv` can never learn its source/tag/count — master/worker and dynamic load-balancing (asynchronous SDDP variants) cannot be written.
- **Fix direction:** `wait`/`wait_any`/`test` return `Status` (or `Option<Status>`); unify `recv`/`sendrecv` on `Status`; add `error` to `Status` (needs ARC-02 first).

### ARC-10 — Copy-pasted blocking / nonblocking / persistent bodies drifted

- **Severity:** major (root cause of SND-06) · **Reversibility:** two-way · **Target:** 0.5.x (validators, needed by SND-06/07), rest opportunistic
- **Locations:** scatter-in-place argument tuple written 3× (`src/comm/blocking.rs:876`, `src/comm/nonblocking.rs:609`, `src/comm/persistent.rs:1040`); `size == 0 || data.len() % size != 0` written 12×; ~55 near-identical "FFI call → wrap handle" bodies; `size == 0` guards only in `persistent.rs:788,846`.
- **Fix direction:** keep the public trios (rustdoc/ergonomics — macro-generating them would hurt docs), but route each collective through one private validated argument builder (validate once → pointers + counts) shared by the three families.

### ARC-11 — Capability gaps that matter to the target users

- **Severity:** minor · **Reversibility:** additive · **Target:** 0.6+ (as needed)
- **Items:** `iallreduce_inplace` / `ireduce_inplace` (blocking and persistent in-place allreduce exist; hot SDDP pattern); `reduce_init_inplace`; `barrier_init` (docs claim "all 15 `_init` variants", there are 14); `mprobe`/`mrecv` (probe-then-recv races under `MPI_THREAD_MULTIPLE`, `src/comm/p2p.rs:274-357`); `MPI_COMM_TYPE_HW_GUIDED` split (+ `mpi_hw_resource_type` info) for NUMA/socket-level communicators; `MPI_Comm_create_group`.

### ARC-12 — `Info` is public but nothing accepts it

- **Severity:** minor · **Reversibility:** one-way (public type) · **Target:** 0.6 — **open decision D-4** (wire vs delete)
- **Locations:** `src/info.rs` (283 lines), `src/ffi.rs:865-878`, `csrc/ferrompi.c:3265-3313` + info table; window constructors hard-code `-1 // MPI_INFO_NULL` (`src/window.rs:429, 919, 997`); `split_type` passes `MPI_INFO_NULL` (`csrc/ferrompi.c:868-897`); `src/lib.rs:80-81` claims Info is used by constructors; `examples/test_info.rs`.
- **Decision rule (accepted):** wire `Option<&Info>` into window constructors and `split_type` **if** the SDDP stack needs `alloc_shared_noncontig` or hardware-guided splits; otherwise delete (~560 LOC incl. C table, error plumbing, example).

### ARC-13 — `SharedWindow` duplicates `Win`; `WinKind` is dead

- **Severity:** minor · **Reversibility:** one-way · **Target:** 0.6 (accepted, D-5: merge with one deprecation cycle)
- **Locations:** `src/window.rs:309-696` (`LockType`, `SharedWindow`) vs `:844-1405` (`Win`); guards `:2532-2647` (`LockGuard`, `LockAllGuard`) vs `:2649-2788` (`WinLockGuard`, `WinLockAllGuard`); duplicated `LockType` → FFI mapping (`:618-621` and `:1354-1357`); module example duplicated (`:25-49` and `:350-374`); `WinKind` `:787-817` stored in an `#[allow(dead_code)]` field (`:856-862`) "for future introspection"; `SharedWindow::fence()` takes no assert while `Win::fence(WinFenceAssert)` does; `SharedWindow` has no put/get/accumulate though MPI allows them on shared windows.
- **Fix direction:** `Win::allocate_shared` + `Win::shared_query`/remote view replaces `SharedWindow`; two guard types; delete `WinKind`; slice semantics per SND-13.

### ARC-14 — The `numa` feature contains no NUMA code and needlessly implies `rma`

- **Severity:** minor · **Reversibility:** one-way (feature name) · **Target:** 0.6
- **Locations:** `Cargo.toml` (`numa = ["rma"]`); `src/slurm.rs` (only `std::env` reads); `src/topology.rs` (`SlurmInfo`); docs claim "NUMA-aware shared memory windows" and `libhwloc-dev` (`README.md:24,68`, `src/lib.rs:53,85-87`, `docs/mpi-compatibility.md:424`); CI installs `libhwloc-dev` for nothing.
- **Fix direction:** rename to `slurm` without the `rma` implication (keep `numa` as a deprecated alias for one release), or ungate the ~60 lines of zero-dependency code.

### ARC-15 — Naming and coverage asymmetries

- **Severity:** minor · **Reversibility:** one-way (renames) · **Target:** 0.6
- **Items:** `broadcast` / `ibroadcast` / `bcast_init`; `Communicator::create_from_group` wraps `MPI_Comm_create` (`csrc/ferrompi.c:899`) while `Mpi::create_from_group` wraps `MPI_Comm_create_from_group` — same name, different collective semantics; `gather_inplace` errors on non-root (`src/comm/blocking.rs:770`) while `scatter_inplace`/`reduce_inplace` accept non-root; `reduce*` requires `recv.len() == send.len()` on non-root ranks where MPI ignores `recv`; `LockGuard` vs `WinLockGuard`.

### ARC-16 — Magic `-1` sentinels; no typed source/tag

- **Severity:** minor · **Reversibility:** one-way · **Target:** 0.6
- **Locations:** `csrc/ferrompi.c:846, 1000-1001, 1125-1126`; `src/comm/p2p.rs:47,137,181`; `src/comm/mod.rs:87`.
- **Defect:** `-1` means ANY_SOURCE (source), ANY_TAG (tag), UNDEFINED (split color); only UNDEFINED is named. See COR-09 for the PROC_NULL portability bug. Under the MPI-5 ABI `ANY_TAG = -2`, `PROC_NULL = -3`, `UNDEFINED = -32766` — translation stays necessary.
- **Fix direction:** `Source::{Any, ProcNull, Rank(i32)}`, `Tag::{Any, Value(i32)}` (or named constants normalised in C).

### ARC-17 — RMA API takes a redundant `target_count` and duplicate datatype tags

- **Severity:** minor · **Reversibility:** one-way · **Target:** 0.6
- **Locations:** `Win::{put, rput, get, rget, accumulate, raccumulate, get_accumulate}` (`src/window.rs:1586-2241`); C shims `csrc/ferrompi.c:4079-4210` take separate origin/target tags that Rust always passes as the same `T::TAG` (3× for `get_accumulate`).
- **Fix direction:** derive the count from the slice (shorter transfers = sub-slice); single tag across the FFI. Removes the mismatch class behind SND-09 (0.5.x validates instead).

### ARC-18 — Open MPI 5 capabilities unused (added 2026-09-24 during 0.5.x planning)

- **Severity:** minor · **Reversibility:** two-way · **Target:** 0.6 (API milestone, ships 0.7.0)
- **Source:** design council (hpc-parallel-computing-specialist, build-deploy, CI briefs in `plans/ferrompi-0.5.x-hardening/design/council/`).
- **Defect:** Open MPI 5.0.x reports `MPI_VERSION` 3.1 and has no `_c` API, but implements the persistent collectives and `MPI_Comm_create_from_group`. The shim gates these on `MPI_VERSION >= 4`, so Open MPI 5 users get `NotSupported` stubs.
- **Fix direction:** per-feature availability gating (e.g. `MPI_VERSION >= 4 || OMPI_MAJOR_VERSION >= 5`) or configure-style link probes in `build.rs`, replacing the version probe in tests with a capability query; needs an Open MPI 5 CI leg (added in the 0.5.x plan) to verify.
