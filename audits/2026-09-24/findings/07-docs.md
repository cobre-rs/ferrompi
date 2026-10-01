# 07 — Documentation accuracy

Only real mismatches (claim vs code/behaviour) and missing essentials. Pure
duplication/tone is in [06-bloat](06-bloat-overengineering.md) (BLT-11).

---

### DOC-01 — `docs/architecture.md` contains false statements

- **Severity:** major (contributors act on it; also published via `ferrompi::doc::architecture`) · **Target:** 0.5.x
- **Items:**
  - `:396-402` "`thiserror` was removed … `[dependencies]` is currently empty" — `Cargo.toml:266` has `thiserror = "2"`; `src/error.rs:45,91,231` derive it; CHANGELOG says the crate *migrated to* thiserror v2.
  - `:61-69, 96-98, 111-117, 142-150` handle-table section: shows `atomic_int request_used[16384]` and plain `int win_used/info_used…`; "Open hardening work" says win/info/comm are non-atomic. Code: all seven tables atomic (`csrc/ferrompi.c:81-143`), request table is a `_Atomic uint64_t` bitmap. Links `plans/ferrompi-gap-closure/…` (gitignored, not in repo).
  - `:25` "4629 LOC" (file has 4992).
  - `:139` communicator "valid to use from any thread, provided … at least `MPI_THREAD_FUNNELED`" — under FUNNELED only the main thread may call MPI; contradicts `src/lib.rs:101-102`, `README.md:361-362`, and `:126` of the same file.
  - `:306-310` "every shim … `if (comm == MPI_COMM_NULL) return MPI_ERR_COMM`" — present in 7 of 86 `get_comm` shims.
  - `:353-356` cites `debug_assert_eq!(T::TAG as i32, mapped_tag(*dt))` in UserOp trampolines — does not exist.
  - `:225-227` "the borrow checker enforces lifetime containment" — it does not (ARC-04).
  - `:257-265` discriminant "semver contract" — to be retracted (ARC-03).

### DOC-02 — ADR-0001 driver 1 is false → see ABI-05

### DOC-03 — ADR-0002 claims that do not hold

- **Severity:** minor · **Target:** 0.5.x (amendment note) / superseded by the ARC-01 ADR
- `docs/adr/0002-handle-tables.md:84` "No ABA problem" (true only for allocation; handles are reused — COR-02); `:408-425` "~64× less metadata to false-share" (measured opposite — PRF-02); `:365-377` mandates a TSan CI step or a mandatory manual pre-release step in `CONTRIBUTING.md`/`docs/testing.md` — neither exists (INF-19); `:379-436` "Out of Scope (at time of ticket-023)" / "Update (F2-002, F2-003, F2-006)" sections turn an "immutable" ADR (per `docs/README.md`) into a changelog.

### DOC-04 — ADR-0004's rejection of lifetime-bound requests is mis-argued; references a nonexistent variant

- **Severity:** minor · **Target:** superseded by the D-1 ADR (0.6); 0.5.x note
- `docs/adr/0004-persistent-collective-approach.md:227-283` conflates lifetime `'a` with type `T` and claims one `'a` cannot cover two borrows (`Win<'a, T>` disproves "unworkable"); `:440-449` references `MpiErrorClass::UnsupportedOperation` (does not exist; see COR-12).

### DOC-05 — ADR-0005 Decision 7 describes the design it says was rejected; plan/changelog sections

- **Severity:** major (misleading) · **Target:** 0.5.x
- `docs/adr/0005-mpi-op-create.md:310-356` (Decision 7) specifies a trampoline typed over `T` with a `debug_assert_eq!` tag check and **rejects** "type-erased `Box<dyn Fn(...)>`"; the implementation is exactly that (`type ByteClosure = Box<dyn Fn(&[u8], &mut [u8]) …>`, `src/op.rs:59`; tag ignored `_dt_tag`, `:167`). `:358-483` "Ticket-037 will implement exactly…", `:530-551` "Open Questions Deferred" are plan text (~220 lines of bloat).
- **Fix:** correct Decision 7 to what shipped (or record the reversal); delete plan sections.

### DOC-06 — `docs/migrating-from-rsmpi.md` names APIs that do not exist and makes a false safety claim

- **Severity:** major (self-described "authoritative conversion reference") · **Target:** 0.5.x
- Nonexistent: `comm.ssend`, `comm.bsend`, `comm.rsend` (only `*_init` forms exist); `world.clone()` (`Communicator` has no `Clone`, `src/comm/mod.rs:61`); a "`DatatypeBuilder` API"; "implement `BytePermutable` via `unsafe impl`" (sealed). Lines `:109-111, 475, 478-480, 516-517, 552, 569`.
- Drop table says `Request` → `MPI_Request_free`; code (`src/request.rs:428-437`) and the guide's own `:309-317` say `MPI_Wait`.
- `:232-234, 307-308` claim the compiler protects in-flight buffers (it does not — SND-01/02).
- `ferrompi = "0.2"`.
- Snippets are `rust,ignore`, so none of this is compiled.
- **Fix:** correct names; un-ignore the ferrompi-side blocks (they are included via `src/doc.rs`, so they become doctests).

### DOC-07 — `docs/mpi-compatibility.md` inaccuracies

- **Severity:** minor · **Target:** 0.5.x
- `:117-119, 224-228, 308-313` `MPI_ERR_UNSUPPORTED_OPERATION` → `NotSupported` (COR-12).
- Matrix lists `barrier_init` ✓ and `ibcast` (real method `ibroadcast`) (`:90, 99, 127, 304`).
- MPICH 3.x ✓ marks "based on MPI standard conformance" (`:92`) vs legend "✓ = confirmed by CI or a verified user report"; 132 "?" cells (Open MPI 5, Cray).
- Build docs `:395-399, 421, 428-436`: `MPICC` said to override when the wrong wrapper is found (ignored if any `.pc` matches); Cray `CRAY_MPICH_DIR` said to be checked first and "all other detection skipped" (it is checked 4th); "Set `MPICC=$(which cc)`" (Cray `cc` has no `-show`) — see INF-04.
- `:168` `test_rsend_init` (now `test_persistent_rsend`).
- `:201-208, 275-279` `Win::sync` advice (COR-13).
- `:424` `numa` needs hwloc (ARC-14).

### DOC-08 — `README.md` inaccuracies

- **Severity:** minor · **Target:** 0.5.x
- License badge links `LICENSE` (files are `LICENSE-MIT`/`LICENSE-APACHE`) (`:7`).
- Quick start `ferrompi = "0.2"` (`:85`).
- `ReduceOp` listing shows 4 of 12–14 variants (`:344-351`); "Core Types" omits `Win`, `Group`, `UserOp`, `CustomDatatype` (`:278-291`).
- `#SBATCH --bind-to core` is an mpirun option, not sbatch (`:407`; also `src/lib.rs:164`).
- "Unit tests (no MPI required)" (`:456-458`) — `build.rs` panics without MPI.
- "Requires MPICH 4.0+ or OpenMPI 5.0+" / "Works with MPICH, OpenMPI, Intel MPI, and Cray MPI" (`:19, 91`; also `CONTRIBUTING.md:19`, `tests/run_mpi_tests.sh:20`, `benches/README.md:9`) vs CI: MPICH 4.2 + Open MPI **4.x** only.
- "10–30% speedup" (`:43`) unsupported (PRF-04); "Built and tested against the MPI 4.1 standard" vs Open MPI 4 = MPI 3.1.
- Omits `LD_LIBRARY_PATH` on Linux (`:492-521`; only macOS `DYLD_LIBRARY_PATH`) even though `build.rs:43-47` deliberately drops rpath — `hello_world` fails with `libmpi.so.12: cannot open shared object file` (reproduced by docs reviewer).
- Handle-table list omits group/datatype/op tables (`:541`).

### DOC-09 — Crate-level rustdoc (`src/lib.rs`) inaccuracies

- **Severity:** minor · **Target:** 0.5.x
- `:61` "All 15 `_init` variants" — 14 distinct persistent collectives (no `barrier_init`) + 5 in-place forms (`README.md:33` too).
- `:80-81` `Info` "for runtime hint passing to communicator, window, and operation constructors" — nothing accepts it (ARC-12).
- `:109-110` comment "Request serialized thread support" above code requesting `Funneled`.
- `:53, 85-87` `numa` = "NUMA-aware windows" (ARC-14).
- `:352, 376, 395-399` "There can only be one instance … at a time" / "Returns an error if MPI is already initialized" — re-init after drop aborts (COR-07).
- `:11-13` large-count note is accurate; keep.

### DOC-10 — Stale or wrong item-level rustdoc

- **Severity:** minor · **Target:** 0.5.x
- `src/request.rs:77-79, 426-427` "planned for v0.5" (we are 0.5.0).
- `src/group.rs` (`Communicator::group` doc) "`MPI_Comm_create_group` is a future epic"; `src/comm/mgmt.rs:55`.
- `src/datatype.rs:166-168` "that is Epic 6" (`CustomDatatype` exists).
- `src/window.rs:666-672, 1062-1070` `raw_handle` "raw MPI window handle … for custom FFI calls" (it is a table index — ARC-03).
- `src/datatype_builder.rs:144-247` promises `Error::Mpi` class `Other` on a full table; code returns `ResourceExhausted` (ARC-08).
- `src/datatype.rs:296-301` `LongDoubleInt` "over-aligns harmlessly" (COR-11).
- `src/comm/p2p.rs:362-365, 415-417` custom-datatype "well-defined `MPI_ERR_TRUNCATE` … not memory unsafety" (SND-08).
- `src/comm/v_collective.rs:200-204` SAFETY comment (SND-07).
- `src/window.rs:1496-1517` `Win::sync` (COR-13); `:1579, 1768, 1966, 2191` RMA examples (SND-03).
- `src/op.rs:61-73, 413` stale comments (BLT-15).
- `src/error.rs:170-200` "fixed values per the MPI spec" (COR-01); `:259-261` `NotSupported` (COR-12).

### DOC-11 — `CONTRIBUTING.md` is template filler and states false policies

- **Severity:** minor · **Target:** 0.5.x
- `:107-275` fictitious `ferrompi::add` example and `FerrompiError` enum; a `#[should_panic]` template next to "don't use `unwrap()` in library code"; generic "Do's ✅ / Don'ts ❌", "Recognition"; mandates `# Arguments`/`# Returns`/`# Examples` on every public fn (source of 80 `# Arguments`, 154 `# Example` sections — bloat).
- `:59` "Library code passes `clippy::pedantic`" — the documented command fails at `build.rs:60` (`items_after_statements`); the library has 331 pedantic warnings.
- `:88-92` "run `cargo bench`" — `benches/README.md:4-5` says that hangs/aborts (benches need `mpiexec`, size ≥ 2).
- `:96-101` "security audit on all PRs" — `security.yml` runs only for PRs to `main`.
- `:239` "test MSRV compatibility" — no job does (INF-02).
- `:18` MSRV 1.74 (INF-02).

### DOC-12 — CHANGELOG restates rustdoc and leaks internal IDs; release notes are empty

- **Severity:** minor · **Target:** 0.5.x
- `CHANGELOG.md:218-462` (0.4.0 lists every `Group::`/`CustomDatatype::` method); 14 internal IDs (`F2-004`, `PERF-01`, `epic-06`). Published GitHub release bodies are empty (INF-14).

### DOC-13 — `benches/README.md` claims

- **Severity:** minor · **Target:** 0.5.x — `:143-148` persistent 10–30% (PRF-04); `:166-172` "Ticket-013 uses the numbers…" (PRF-03); `:167, 172` ticket IDs.

### DOC-14 — Missing documentation users need

- **Severity:** minor · **Target:** 0.5.x
- Lifecycle: `Mpi` drop = finalize; what happens to live handles; no re-init; errors inside `MPI_Init` are fatal before the error handler is installed.
- Error reporting: which errors return `Err` vs abort (errhandler scope incl. `MPI_COMM_SELF`, COR-06).
- Linux runtime library path (`LD_LIBRARY_PATH` / module system) given the deliberate no-rpath build.
- Building on Open MPI (pkg-config name `ompi`; `mpicc --showme` vs `-show`).

### DOC-15 — C comments that are wrong

- **Severity:** nit · **Target:** 0.5.x
- `csrc/ferrompi.c:321-323` claims the acquire load makes "the request_table value written by the allocating thread" visible — same wrong claim ADR F2-003 corrected elsewhere; the write is sequenced after the claiming RMW.
- `csrc/ferrompi.h:1082` waitany/waitsome "handle array updated in place" — C never writes it.
- `csrc/ferrompi.c:26-32` false-sharing claim (PRF-02); stale line refs and process artifacts (BLT-21).
