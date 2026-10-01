# 02 — Correctness bugs (MPI semantics, C shim, lifecycle)

Wrong results, misreported errors, crashes or hangs — not necessarily memory-UB
from safe code (those are in [01-soundness](01-soundness.md)).

---

### COR-01 — Error classes are decoded with Open MPI's numbering; wrong on MPICH-family

- **Severity:** major · **Verified:** repro (re-verified by main session) · **Target:** 0.5.x
- **Locations:** `src/error.rs:170-215` (`MpiErrorClass::from_raw` hard-codes 7→Request … 19→Pending, doc claims "fixed values per the MPI spec"); `src/error.rs:12-21` + `csrc/ferrompi.c:4990-4992` (only FILE/INFO/WIN are queried from C); masking tests `examples/test_errhandler_returns.rs:43-54`, `examples/test_get_group_invalid_handle.rs:77-84` (catch-all arms print PASS for any class).
- **Defect:** the MPI standard fixes only `MPI_SUCCESS = 0`. MPICH (`/opt/mpich/include/mpi.h:593-622`) uses ROOT=7, GROUP=8, OP=9, TOPOLOGY=10, DIMS=11, ARG=12, UNKNOWN=13, TRUNCATE=14, OTHER=15, INTERN=16, IN_STATUS=17, PENDING=18, REQUEST=19. 13 of 19 classes decode wrong on MPICH, Cray MPICH, Intel MPI, MVAPICH. Under the MPI-5 ABI the values match Open MPI's except PENDING=18 / IN_STATUS=19 (swapped vs the hard-coded table) — so the table is wrong there too.
- **Evidence (`repros/requests/src/bin/errclass.rs`, `repros/c-shim/src/bin/t7_errclass.rs`):**
  | Error provoked | MPICH class | ferrompi reports |
  |---|---|---|
  | `broadcast` root=999 | MPI_ERR_ROOT (7) | `Request` |
  | `allreduce` f64 with `BitwiseOr` | MPI_ERR_OP (9) | `Group` |
  | `recv` into too-small buffer | MPI_ERR_TRUNCATE (14) | `Unknown` |
  | stale request handle | MPI_ERR_REQUEST (19) | `Pending` |
  | `wait_all` with a truncating recv | MPI_ERR_IN_STATUS (17) | `Intern` |
- **Fix direction:** classify in C against the symbolic `MPI_ERR_*` constants (same pattern as `ferrompi_group_compare`, `csrc/ferrompi.c:3513-3525`) and return ferrompi-stable class codes; keep `MpiErrorClass::from_raw` semantics = "implementation class value" by comparing against C-exported constants, or deprecate it (public fn). Tighten the two masking tests to assert the exact class.
- **Acceptance:** errclass/t7 print `Root`, `Op`, `Truncate`, `Request`, `InStatus`; `test_errhandler_returns` fails if the class is wrong; same assertions pass on Open MPI in CI.

### COR-02 — Stale request handles act on unrelated requests (handle ABA)

- **Severity:** major · **Verified:** repro (re-verified by main session) · **Target:** 0.5.x
- **Locations:** `src/request.rs:17-29` (`with_handles` copies **all** handles, including `completed` ones), `:173-321` (`wait_any/wait_some/test_any/test_some`), `:387-411` (`wait_all`); `csrc/ferrompi.c:296-319` (`alloc_request` reuses the lowest free bit immediately), `:341-349` (`free_request`), `:3567-3592` (`ferrompi_wait/test` free the slot even when Rust keeps the request "active" on error); `docs/adr/0002-handle-tables.md:84` ("No ABA problem" — true only for allocation).
- **Defect:** handles are bare slot indices with no generation. After completion, a `Request`'s slot is freed and handed to the next nonblocking op; passing the old handle again (the documented "removing it from the vector is the caller's responsibility" contract, or the MPI `Waitany` loop idiom) targets whichever request owns the slot now. The documented `Ok(None)` ("all requests were MPI_REQUEST_NULL") is unreachable because completed entries are never passed as null.
- **Evidence:**
  - `repros/requests/src/main.rs`: two completed requests (handles 0,1); two new unrelated requests get handles 0,1; `Request::wait_all(&mut old)` → `Ok(())` **after completing and freeing the new requests**; `new_recv.wait()` / `new_send.wait()` → `Err(… "Invalid MPI_Request")`. Second `wait_any` on a slice still holding the completed entry → `Err(… waitany)`.
  - `repros/soundness/src/bin/r10_waitany_aba.rs`, `repros/c-shim/src/bin/t3_waitany_stale.rs`, `t3b_waitany_null.rs`: same via `wait_any`.
  - `repros/c-shim/src/bin/t9_test_err_aba.rs`: `a.test()` returns a truncation error and C frees slot 0 while Rust keeps `a` active; new `irecv` B takes slot 0; `drop(a)` waits on and consumes B; `B.wait()` → "Invalid MPI_Request". `cancel()` on a stale handle would cancel someone else's receive.
- **Fix direction:** (1) pass completed entries as a null sentinel (`-1`) that C maps to `MPI_REQUEST_NULL`, making `Ok(None)` reachable and the Waitany loop idiom work; (2) encode a generation in the 64-bit handle (`gen << 32 | idx`) validated on every lookup, so any remaining stale use fails with `MPI_ERR_REQUEST` instead of aliasing; (3) keep Rust's `completed` flag consistent with C on every error path. The durable fix is ARC-01 (no table).
- **Acceptance:** all four repros: new requests unaffected; second `wait_any` returns `Ok(None)` when everything is completed; add a regression example that loops `wait_any` without removing entries. Note: `t3_waitany_stale.rs` must be adapted before reuse as a regression test — after the fix its second `wait_any` correctly waits on `b`, whose send is posted later, so move `send(&sb…)` before that call (see repros README).

### COR-03 — No request write-back on error in `waitall/waitany/waitsome/testany/testsome`

- **Severity:** major (use-after-free of MPI request objects inside MPI) · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:3614-3627` (waitall), `:3714-3726` (waitany), `:3755-3774` (waitsome), `:3796-3811` (testany), `:3840-3859` (testsome); `src/request.rs:397-411` ("on error leave requests active so Drop re-waits"); persistent `wait_all` has the opposite policy.
- **Defect:** on error MPI has already completed and freed some requests (set to `MPI_REQUEST_NULL` in the local array), but the shim copies the array back only on success, so the table keeps freed `MPI_Request` values and Rust re-waits them in `Drop`.
- **Evidence:** `repros/c-shim/src/bin/t4b_waitall_err_nofresh.rs` — one ok + one truncated `irecv`, `wait_all` → `ERR_IN_STATUS`; `test()` afterwards → MPICH **"INTERNAL ERROR: unexpected value in case statement"**; `t4_waitall_err.rs` — with a new `irecv` posted first, dropping the old requests **hangs forever** (recycled handle aliases the new request); `t4c_waitany_err.rs` — same for `wait_any`; `repros/soundness/src/bin/r9_waitall_err.rs` — abnormal termination (SIGTERM after timeout), control `r9b` survives.
- **Fix direction:** always copy `reqs[i]` back and free slots that became `MPI_REQUEST_NULL`, regardless of return code; report per-handle completion to Rust (e.g. an out array of flags) so Rust marks exactly those `completed`; unify `Request::wait_all` and `PersistentRequest::wait_all` error policy.
- **Acceptance:** t4/t4b/t4c/r9 terminate cleanly; no MPICH internal error; a request that completed with error is never waited again.

### COR-04 — `MPI_STATUSES_IGNORE` discards the per-request error on `MPI_ERR_IN_STATUS`

- **Severity:** minor · **Verified:** repro (`t4_waitall_err.rs` shows only "See the MPI_ERROR field in MPI_Status") · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:3614`, `:3755`, `:3840` (and waitany/testany with `MPI_STATUS_IGNORE`).
- **Fix direction:** pass a statuses array (stack for ≤ 64, heap above, as today for requests) and, on `MPI_ERR_IN_STATUS`, return the first failing request's `MPI_ERROR` (plus its index) so `Error::Mpi` carries the real cause. Full `Status` return is ARC-09 (0.6).
- **Acceptance:** t4 reports `Truncate` with the failing index.

### COR-05 — Counts above `INT_MAX` are silently truncated with `(int)`

- **Severity:** critical (silent data corruption) · **Verified:** repro (RMA) + reading · **Target:** 0.5.x
- **Locations:** RMA `csrc/ferrompi.c:4088-4090`, `4103-4105`, `4125-4127`, `4164-4166`, `4209-4212` (Rust only checks `i64::try_from`, `src/window.rs:1593`); v-collective scalar counts `:1576`, `:1591`, `:1604`, `:2129`, `:2156`, `:2181` (no `_c` path even on MPI 4); every `#else` branch of the `#if MPI_VERSION >= 4 … _c … #endif return MPI_X(…(int)count…)` pattern (e.g. `:981`, `:1010`, `:1051`, `:1089`, `:1137`) — i.e. **every blocking/nonblocking call on MPI 3.x builds, including Open MPI 4.x used in CI**.
- **Defect:** three coexisting policies: `_c` dispatch (MPI 4 blocking/nonblocking), `MPI_ERR_COUNT` (persistent, waitall), silent truncation (everything else).
- **Evidence:** `repros/c-shim/src/bin/t13_rma_trunc.rs` — `win.put(&[3.0; 4], 0, 0, (1 << 32) + 4)` returns **`Ok` and transfers 4 elements**. On Open MPI 4.1, `send` of 2³²+10 bytes would send 10 bytes and return `Ok` (reading).
- **Fix direction:** one C helper per call family: use the `_c` variant when available, otherwise return `MPI_ERR_COUNT`; never cast unchecked. Add `MPI_Put_c`/`MPI_Get_c`/`MPI_Accumulate_c`/`MPI_Gatherv_c`… on MPI 4. The v-collective count arrays remain `&[i32]` in 0.5.x (type change is ARC / 0.6+).
- **Acceptance:** t13 returns `Err(Count)`; grep shows no unchecked `(int)` count cast in `ferrompi.c`; opt-in large-count integration test (INF-11) passes.

### COR-06 — `MPI_ERRORS_RETURN` is not installed on `MPI_COMM_SELF`

- **Severity:** major on MPI-4-conformant libraries · **Verified:** reading against the standard (MPI-4.0+ raises errors with no associated object on `MPI_COMM_SELF`; MPI 5.0 text ~lines 41091-41096 of the report); **not reproducible on MPICH 4.2.3**, which still falls back to `MPI_COMM_WORLD` (`repros/c-shim/src/bin/t1_group_err.rs` returned `Err` there) · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:674`, `:701` (errhandler set on `MPI_COMM_WORLD` only).
- **Defect:** group constructors, type constructors, info, op create/free, buffer attach/detach, and calls on a `MPI_COMM_NULL` handle would abort the job on Open MPI 5 instead of returning the documented `Err`.
- **Fix direction:** `MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN)` in both init paths.
- **Acceptance:** t1 returns `Err` on MPICH and (once INF-20 adds it) Open MPI 5.

### COR-07 — Finalize / re-init lifecycle: aborts instead of errors; `UserOp` dropped after finalize kills the process

- **Severity:** major · **Verified:** repro · **Target:** 0.5.x
- **Locations:** `src/lib.rs:700-710` (`Mpi::drop` resets `MPI_INITIALIZED` to false, re-allowing `MPI_Init_thread`); `csrc/ferrompi.c:656-708` (no `MPI_Finalized` check), `:226-235` (`get_comm(0)` always returns `MPI_COMM_WORLD`), `:710-774` (finalize sweep skips `op_table`), `:4923-4946` (`ferrompi_op_free`); `src/op.rs:476-493`.
- **Evidence:**
  - `repros/c-shim/src/bin/t5_reinit.rs` — second `Mpi::init()` after drop → MPICH fatal **"Cannot call MPI_INIT or MPI_INIT_THREAD more than once"** (exit 15) instead of `Err`.
  - `repros/c-shim/src/bin/t6_world_after_finalize.rs` — `world.barrier()` after `drop(mpi)` → process exits.
  - `repros/c-shim/src/bin/t2_userop_after_finalize.rs` — `struct App { mpi: Mpi, op: UserOp<f64> }` drops fields in order: **"Attempting to use an MPI routine (internal_Op_free) … after finalizing MPICH"**, exit 1.
- **Fix direction:** a process-global finalized flag (C side, checked at every entry point → `MPI_ERR_OTHER`/new `Error::Finalized`); `Mpi::init` after finalize → `Err`; finalize sweeps the op table (drop closures, reset slots); `ferrompi_op_free` early-returns after finalize. Document the lifecycle (DOC-14).
- **Acceptance:** t5/t6/t2 return `Err` or no-op; no MPI call after finalize.

### COR-08 — Finalize sweep frees active requests and collectively frees leaked windows

- **Severity:** minor · **Verified:** reading (standard: `MPI_Request_free` on active nonblocking-collective/RMA/persistent-collective requests is erroneous; `MPI_Win_free` is collective) · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:715-722` (requests), `:755-761` (windows).
- **Defect:** a `Request` from `iallreduce` that outlives `Mpi` triggers an erroneous `MPI_Request_free`; ranks that leaked *different* windows deadlock inside the sweep.
- **Fix direction:** free only inactive requests; leave active ones (report to stderr in debug); do not free live windows from finalize (D-8: leak + stderr warning).

### COR-09 — `-1` means `MPI_PROC_NULL` on MPICH but `MPI_ANY_SOURCE` on Open MPI; no portable `PROC_NULL`

- **Severity:** minor · **Verified:** repro (`repros/c-shim/src/bin/t14_send_neg1.rs`: `send(dest = -1)` returns `Ok` as a silent no-op on MPICH; Open MPI returns `MPI_ERR_RANK` by reading) · **Target:** 0.6 (typed sentinels, ARC-16); 0.5.x may reject negative `dest` other than a named constant
- **Locations:** `csrc/ferrompi.c:965-982`, `1030-1063`, `1103-1140`, `4079+` (dest/target passed raw); `:1000-1001`, `:1125-1126` (source `-1` → `MPI_ANY_SOURCE`); `src/comm/p2p.rs:47,137,181`.
- **Defect:** non-periodic halo exchanges cannot use `PROC_NULL` portably; receiving from `PROC_NULL` is impossible.
- **Fix direction:** explicit ferrompi sentinels mapped in C (e.g. `ANY_SOURCE = -1`, `PROC_NULL = -2`, `ANY_TAG = -1`), typed `Source`/`Tag` in 0.6.

### COR-10 — `Request::cancel` is allowed on nonblocking-collective and RMA requests

- **Severity:** minor · **Verified:** repro (`repros/c-shim/src/bin/t16_cancel_coll.rs`: `iallreduce(...).cancel()` → MPICH "Attempt to cancel an unknown type of request") · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:3694-3698`; `src/request.rs:367-374`; the shim's own comment `:357-363` states cancel is erroneous for these kinds.
- **Fix direction:** record the request kind (p2p / collective / RMA) in the table (or in Rust `Request`) and return an error for non-p2p kinds without calling MPI.

### COR-11 — `LongDoubleInt` / `LongInt` layouts wrong on macOS arm64 / Windows

- **Severity:** minor · **Verified:** reading + clang layout check (`repros/c-shim/ldi.c`) · **Target:** 0.5.x
- **Locations:** `src/datatype.rs:285-313` (`LongDoubleInt` = `[u8;16]` + `align(16)` = 32 bytes; comment ":301 over-aligns harmlessly" is false); `csrc/ferrompi.c:594-599`.
- **Defect:** on Apple arm64 `long double` is 8 bytes → C `{long double; int}` is 16 bytes; on MSVC `long` is 4 bytes → `LongInt` mismatches. MPI then reduces with the wrong stride. (Linux x86_64/aarch64/ppc64le layouts verified correct — see VER-06.)
- **Fix direction:** `_Static_assert(sizeof(struct{long double a; int b;}) == 32)` etc. in C so a mismatched platform fails to build, plus `cfg`-specific Rust layouts where needed; or gate the types to verified targets.

### COR-12 — Pre-MPI-4 persistent/`create_from_group` stubs return `MPI_ERR_OTHER`, never `Error::NotSupported`

- **Severity:** minor · **Verified:** reading · **Target:** 0.5.x
- **Locations:** stubs `csrc/ferrompi.c:948`, `3114-3259` (return `MPI_ERR_OTHER`; `MPI_ERR_UNSUPPORTED_OPERATION` appears nowhere); `Error::NotSupported` constructed only at `src/lib.rs:565`; three contradicting docs: `docs/mpi-compatibility.md:117-119,224-228,308-313` (`MPI_ERR_UNSUPPORTED_OPERATION` → `NotSupported`), `docs/adr/0004-persistent-collective-approach.md:443-444` (nonexistent `MpiErrorClass::UnsupportedOperation`), `src/error.rs:259-261`.
- **Defect:** user code matching `Err(Error::NotSupported(_))` never fires on Open MPI 4.
- **Fix direction:** stubs return a dedicated sentinel (like the `-700x` family) that `Error::from_code` maps to `NotSupported(op)`; fix all three docs.

### COR-13 — `Win::sync` rustdoc says no epoch is needed; the example fails on MPICH

- **Severity:** minor · **Verified:** repro (`repros/win-sync/`: the rustdoc example → `Err: MPI error in win_sync: Wrong synchronization of RMA calls`) · **Target:** 0.5.x (docs) / 0.6 (move `sync` onto guards)
- **Locations:** `src/window.rs:1496-1517`; contradicting `docs/mpi-compatibility.md:201-208,275-279` (advises calling sync "via a `WinLockGuard`", which has no `sync`); only test `examples/test_rma_win_flush_sync.rs:113` (inside `lock_all`).
- **Fix direction:** correct rustdoc (passive-target epoch required); in 0.6 expose `sync` on lock guards only.

### COR-14 — `MPI_UNDEFINED` from `MPI_Get_count[_c]` leaks raw; its return code is ignored

- **Severity:** nit · **Verified:** reading · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:1017-1019`, `1146-1148`, `1179-1181`, `1210-1212`, `4662-4664`.
- **Fix direction:** check the return code; normalise `MPI_UNDEFINED` to `-1` (as elsewhere) or surface an error; document in `Status`.

### COR-15 — `ferrompi_op_set_closure` lacks a slot bounds check; `ferrompi_op_create_user` does not check `op_used`

- **Severity:** nit (only reachable via crate-internal callers today) · **Verified:** reading · **Target:** 0.5.x (or removed by BLT-15 redesign)
- **Locations:** `csrc/ferrompi.c:4902-4905`, `4910-4918`.

### COR-16 — `ferrompi_type_create_struct` may read uninitialised stack arrays when `count == 0`

- **Severity:** nit (MPI ignores the arrays at count 0; Rust rejects empty field lists) · **Verified:** compiler `-Wmaybe-uninitialized` · **Target:** 0.5.x
- **Locations:** `csrc/ferrompi.c:4535-4560`.
- **Fix direction:** early-return for `count <= 0`; builds warning-free under `-Wall -Wextra`.
