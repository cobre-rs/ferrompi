# 04 — MPI 5.0 standard-ABI readiness

Supporting the MPI-5 standard ABI is a **stated future direction** (maintainer,
2026-09-24). This file records the verified facts (as of 2026-09-24), where
ferrompi couples to implementation-specific ABI, and the accepted migration path.

## Verified facts

Primary sources: MPI 5.0 report ch. 20 + Annex A.1.1 (June 2025,
<https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf>); Forum reference header
<https://github.com/mpi-forum/mpi-abi-stubs> (spot-checked by the main session:
`MPI_Comm` typedef, `MPI_COMM_WORLD`, `MPI_DOUBLE`, `MPI_SUM`, `MPI_Status`, error
classes, `MPI_THREAD_MULTIPLE`, `MPI_ANY_TAG`, `MPI_PROC_NULL`, `MPI_ABI_VERSION`,
`MPI_Comm_toint/fromint`).

| Topic | Fact |
|---|---|
| Header / library | header still `mpi.h`; library **must** be `libmpi_abi` (`libmpi_abi.so`) and the app's only direct MPI dependency; mixing ABIs forbidden (§20.2.1) |
| Handles | pointers to incomplete structs: `typedef struct MPI_ABI_Comm* MPI_Comm;` (also Datatype, Errhandler, File, Group, Info, Message, Op, Request, Session, Win) |
| Predefined handles | integer constants cast to the handle type; 1–4095 reserved, 0 never valid; datatypes 512–1023, ops 32–63, others 256–511. `MPI_COMM_NULL` 256, `MPI_COMM_WORLD` 257 (0x101), `MPI_COMM_SELF` 258, `MPI_GROUP_EMPTY` 265, `MPI_REQUEST_NULL` 384, `MPI_ERRORS_RETURN` 323 |
| Datatypes | `MPI_DOUBLE` 532 (0x214), `MPI_FLOAT` 528, `MPI_INT32_T` 592, `MPI_UINT32_T` 593, `MPI_INT64_T` 600, `MPI_UINT64_T` 601, `MPI_UINT8_T` 577, `MPI_BYTE` 583, `MPI_DOUBLE_INT` 553, `MPI_2INT` 555 |
| Ops | `MPI_SUM` 33 (0x21), `MPI_MIN` 34, `MPI_MAX` 35, `MPI_PROD` 36, `MPI_BAND/BOR/BXOR` 40/41/42, `MPI_LAND/LOR/LXOR` 48/49/50, `MPI_MINLOC/MAXLOC` 56/57, `MPI_REPLACE` 60, `MPI_NO_OP` 61 |
| `MPI_Status` | exactly `{ int MPI_SOURCE; int MPI_TAG; int MPI_ERROR; int MPI_internal[5]; }` |
| Integer types | `MPI_Aint` = `intptr_t`; `MPI_Offset`, `MPI_Count` = `int64_t` |
| Error classes | fixed: e.g. `MPI_ERR_REQUEST` 7, `MPI_ERR_ROOT` 8, `MPI_ERR_TRUNCATE` 15, `MPI_ERR_PENDING` 18, `MPI_ERR_IN_STATUS` 19, `MPI_ERR_FILE` 30, `MPI_ERR_INFO` 34, `MPI_ERR_WIN` 56, `MPI_ERR_ABI` 62 |
| Sentinels | `MPI_ANY_SOURCE` −1, `MPI_ANY_TAG` −2, `MPI_PROC_NULL` −3, `MPI_ROOT` −4, `MPI_UNDEFINED` −32766 |
| Thread levels | 0 / 1024 / 2048 / 4096 |
| Interop | `MPI_*_c2f/f2c` are **not** in the ABI; replaced by `int MPI_Comm_toint(MPI_Comm)` / `MPI_Comm MPI_Comm_fromint(int)` (and per handle type) |
| Detection | compile time `MPI_ABI_VERSION` 1 / `MPI_ABI_SUBVERSION` 0; run time `MPI_Abi_get_version`, `MPI_Abi_get_info` |
| User ops | still only `MPI_Op_create[_c]` — no user-data pointer (the 16 trampolines remain necessary) |

**Implementations (as of 2026-09-24):**
- **MPICH 4.3.0** (Feb 2025): pre-ratification *draft* ABI — declares `MPI_ABI_VERSION` but is **binary-incompatible** with the ratified ABI (`MPI_ERRORS_RETURN`/`MPI_ERRORS_ABORT` swapped, thread levels 1/2/7, `MPI_SEEK_*` reordered). Detection must require `MPI_ABI_VERSION` **and** `MPI_VERSION >= 5`.
- **MPICH 5.0.x**: "full support for the MPI-5 standard including the new MPI ABI specification" (release notes, verified). Opt-in: `--enable-mpi-abi` builds `libmpi_abi` alongside `libmpi`; `mpicc_abi` / `mpicc -mpi-abi` adds `-DMPI_ABI` and links `-lmpi_abi`; `mpi.h` switches to the ABI header only under `#ifdef MPI_ABI`; no ABI pkg-config file. MPICH main (unreleased, PR #7949) makes `--enable-mpi-abi` ABI-only so plain `mpicc` defines `MPI_ABI` (unverified end state).
- **Open MPI 6.0**: ABI via `--enable-standard-abi`, `mpicc_abi`, pkg-config `ompi-forum-abi[-c]`; **only rc1** exists.
- **Wrappers:** `mpi_abi_wrapper` (MPI-5 `mpi.h` + `libmpi_abi.so` over any MPI); Mukautuva (pre-standard MUK ABI); MPItrampoline (own ABI); traMPI. Intel MPI / Cray MPICH / MVAPICH ABI support: **unverified**.
- **Local /opt/mpich 4.2.3:** no `mpi_abi.h`, `libmpi_abi.so`, or `mpicc_abi`.

## Empirical readiness (ABI reviewer, 2026-09-24)

- `csrc/ferrompi.c` passes `cc -std=c11 -fsyntax-only -Wall -Wextra -Wpedantic` with **zero diagnostics** against both MPICH v5.0.1 `mpi_abi.h` and the Forum stubs `mpi.h`.
- All 172 MPI symbols the shim references are exported by a `libmpi_abi.so` built from the Forum's `mpilib.c`.
- The full crate (`--features rma --examples`) **builds and links** against that stub library via `PKG_CONFIG_PATH=… MPI_PKG_CONFIG=mpiabi-stub`.
- **Runtime untested** (the stubs `abort()` in every non-trivial call).

## Coupling points

| Location | What | Leaks into public API | Goes away under ABI |
|---|---|---|---|
| `csrc/ferrompi.c:80-153, 226-560` | 7 handle tables + caps + CAS | indirectly (`ResourceKind`, `raw_handle`) | yes (ARC-01) |
| `csrc/ferrompi.c:563-603` (`get_op`, `get_datatype`, 93 call sites) | tag → handle switch | **yes** — `DatatypeTag`/`ReduceOp` discriminant "semver contract" | yes (handles are constants) |
| `csrc/ferrompi.c:656-691` | thread level 0..3 ↔ `MPI_THREAD_*` | no (`ThreadLevel` is ferrompi's own) | becomes a constant table |
| `csrc/ferrompi.c:3972, 4072`; `src/window.rs:89-160` | `MPI_MODE_*` queried at runtime | no | yes (fixed values) |
| `src/error.rs:170-215` | hard-coded error classes | **yes** (`pub fn from_raw`, exhaustive enum) | must be fixed regardless (COR-01) |
| `csrc/ferrompi.c:3319`; `src/group.rs:179-205`; `src/comm/mod.rs:87` | `MPI_UNDEFINED` → −1 | ferrompi convention | no (ABI value −32766 → translate) |
| receive/probe shims (`:1000-1001` …) | −1 → `MPI_ANY_SOURCE/TAG` | ferrompi convention | no (`ANY_TAG` = −2) |
| 54 `#if MPI_VERSION`, 48 `_c` calls, 74 `INT_MAX` checks | large-count dispatch | no | version guards yes; dispatch/validation logic stays |
| `csrc/ferrompi.c:613-654` | `MPI_ERRORS_RETURN` install | no | no (real logic) |
| `csrc/ferrompi.c:4752-4860`; `src/op.rs:160,233` | 16 op trampolines | no | no (no user-data op create in MPI 5.0); in Rust, `extern "C" fn tramp<const N: usize>` would generate them |
| public `raw_handle()` ×8 | table indices | **yes** | semantics impossible without tables (ARC-03) |
| `src/datatype_builder.rs:56-63` | `StructField.basetype: DatatypeTag` pub field | **yes** | — |
| `src/status.rs:24-32` | `Status {source, tag, count}` | pub fields, exhaustive (does **not** mirror `MPI_Status` — good) | adding `error` later is breaking without `#[non_exhaustive]` |
| `csrc/ferrompi.h` / no `c2f`/`toint` | no interop with C/Fortran MPI libraries (PETSc, HDF5-MPI) | — | trivial under ABI (same `MPI_Comm` type everywhere) |
| `build.rs:138-194` | build-system coupling | — | ABI-01..03 |

## Findings

### ABI-01 — `build.rs` drops `-D` flags: `mpicc_abi` would compile the native header but link `libmpi_abi.so`

- **Severity:** critical (latent; silent ABI mismatch = UB) · **Verified:** reading (MPICH `mpicc.sh.in:193-196, 275-285` adds `-DMPI_ABI`) · **Target:** 0.5.x
- **Locations:** `build.rs:168-194` (`parse_mpicc_show` keeps only `-I`/`-L`/`-l`), `:138-149` (`try_pkg_config` ignores the pkg-config crate's `defines`).
- **Fix direction:** pass `-D` flags from `mpicc -show` and pkg-config `defines` to `cc::Build::define`.
- **Acceptance:** building with `MPICC=mpicc_abi` (or a fake wrapper printing `-DMPI_ABI`) compiles the shim with `MPI_ABI` defined.

### ABI-02 — No ABI detection probe; MPICH 4.3's draft ABI would be accepted silently

- **Severity:** major · **Verified:** reading (header diff v4.3.0 vs v5.0.1) · **Target:** 0.5.x (reject draft) / 0.7 (full ABI build)
- **Fix direction:** compile probe in `build.rs`: if `MPI_ABI_VERSION` is defined and `MPI_VERSION < 5` → hard build error explaining the draft ABI; if `MPI_ABI_VERSION` and `MPI_VERSION >= 5` → emit `cargo:rustc-cfg=ferrompi_mpi_abi` (with `rustc-check-cfg`). The draft ABI swaps `MPI_ERRORS_RETURN`/`ABORT`, which would make `install_errors_return` install *abort*.

### ABI-03 — Build-selection env vars are not tracked for rebuilds

- **Severity:** major · **Target:** 0.5.x · Same defect as INF-03 (`MPICC`, `MPI_PKG_CONFIG`, `CRAY_MPICH_DIR`, `PATH`); the ABI/native switch would use exactly these knobs.

### ABI-04 — Public-API one-way doors that block a table-free / ABI backend

- **Severity:** major · **Target:** 0.6 · Covered by ARC-02 (`#[non_exhaustive]`, additive features) and ARC-03 (`raw_handle`, discriminant contract, public `from_code`). Also: retract the discriminant "semver contract" in `docs/architecture.md:257-265` and ADR-0003 so "discriminant = ABI handle value" stays available as a non-breaking change.

### ABI-05 — ADR-0001 misstates what the ABI would add

- **Severity:** minor (docs) · **Target:** 0.5.x (amend ADR-0001) + new ADR-0006 "MPI-5 ABI direction"
- **Locations:** `docs/adr/0001-why-c-wrapper.md:125-130` (driver 1 "without recompilation"), `:434-442` (tables "eliminate this friction entirely").
- **Fix direction:** the tables give *source* portability only; *binary* portability is exactly what the MPI-5 ABI adds.

### ABI-06 — No interop API (deliberately deferred)

- **Severity:** info · **Target:** after an ABI backend exists
- Do **not** add `as_raw() -> MPI_Comm` now: it would put the implementation-specific type in the public API. Under the ABI backend, expose `MPI_Comm_toint`/`fromint`-based or pointer-typed interop.

### ABI-07 — Optional CI job: compile + link against the Forum ABI stubs

- **Severity:** info · **Target:** 0.5.x or 0.7
- Three commands (build stubs `libmpi_abi.so` from `mpi-abi-stubs`, `.pc` file, `cargo build --features rma --examples` with `MPI_PKG_CONFIG`) keep the shim ABI-clean.

## Accepted migration path (D-10)

1. **0.5.x:** ABI-01, ABI-02 (reject draft), ABI-03, COR-01 (symbolic error classes), ADR-0001 amendment.
2. **0.6:** protect the one-way doors (ARC-02, ARC-03).
3. **0.7:** by-value handles (ARC-01), then "C shim compiled against the ABI `mpi.h`, auto-detected" (~100 LOC, reversible) — already delivers build-once-run-on-any-ABI-MPI because the shim is linked statically and the binary's only MPI dependency is `libmpi_abi.so`.
4. **Later, on a concrete requirement** (C-library interop, dropping the C toolchain): Rust-native ABI backend (~1–1.5k LOC declarations + ~2.5–3.5k LOC ported shim logic); after ARC-01 it is an internal change.
5. **ABI-only major version:** only after (4) exists and ABI MPIs ship by default (MPICH, Open MPI 6 final, vendor stacks).
