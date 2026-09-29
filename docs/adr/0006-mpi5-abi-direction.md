# ADR-0006: MPI 5.0 Standard ABI Direction

**Status:** Accepted — 2026-09-24

## Context

The MPI Forum ratified a standard binary ABI in MPI 5.0 (June 2025,
<https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf>). Conforming
implementations keep the `mpi.h` header but build and export a separate
library, `libmpi_abi`, which must be the application's only direct MPI
dependency. Handles become pointers to incomplete struct types, and
predefined handles and constants take fixed values across every conforming
implementation. A compiler defines `MPI_ABI_VERSION` when the ABI header is
in effect.

ferrompi's C shim is compiled against one selected MPI implementation's
`mpi.h` and linked against that implementation's library (ADR-0001, amended
2026-09-24). The shim's handle tables give one Rust API that compiles
against every implementation's headers — source portability — not a single
binary that runs against any conforming MPI library. Binary portability is
what a ratified-ABI build adds.

MPICH 4.3's `-DMPI_ABI` build defines `MPI_ABI_VERSION` but predates
ratification: its error-handler and thread-level constants differ from the
ratified ABI, so installing `MPI_ERRORS_RETURN` under that draft header
installs an aborting handler instead. A build that trusts `MPI_ABI_VERSION`
alone cannot tell the draft header from the ratified one.

## Decision

1. Binary portability comes from compiling the unchanged C shim against a
   ratified MPI 5.0 ABI `mpi.h` and linking `libmpi_abi`. The build script
   already builds this way when the MPI compiler wrapper is an ABI wrapper
   (for example `MPICC=<prefix>/bin/mpicc_abi`), and it forwards that
   wrapper's `-D` defines to the C compiler. ferrompi adds no ABI-specific
   `cfg`, Cargo feature, or Rust code path.
2. The draft ABI is rejected at compile time: the shim's build stops with a
   preprocessor error when `MPI_ABI_VERSION` is defined and `MPI_VERSION`
   is below 5.
3. ferrompi supports the ratified-ABI build with run-time testing once it
   stores MPI handles by value instead of in the shim's handle tables.
4. A Rust-native ABI backend — ABI struct and constant declarations written
   directly in Rust, with no C shim — waits for a concrete requirement,
   such as interoperability with another MPI-using C library or dropping
   the C toolchain from the build.

## Consequences

- The ratified-ABI build compiles and links but is not tested at run time
  until 0.8.0. CI's "MPI ABI stubs" job builds the crate and its examples
  against the MPI Forum's reference ABI stubs through their `mpicc_abi`
  wrapper; the stubs abort inside every non-trivial call, so nothing runs.
- The `DatatypeTag` and `ReduceOp` discriminants are internal (ADR-0003,
  amended 2026-09-24), which leaves room to encode them as ABI handle
  values later.
- The public API exposes no implementation handle type; interoperability
  with other MPI-using libraries waits for the by-value handle
  representation in Decision 3.
- `docs/mpi-compatibility.md` is the reference for selecting an ABI build.

## Alternatives

- Rejected: a build-script ABI probe emitting a `cfg` — no ferrompi code
  path differs between a native header and an ABI header.
- Rejected: accepting the draft ABI — its constants would install an
  aborting error handler.
- Rejected: a Rust-native ABI backend now — thousands of lines of handle
  and constant declarations for no stated requirement.
- Rejected: an ABI-only release — not while every ABI implementation is an
  opt-in build.
