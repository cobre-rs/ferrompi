# ferrompi Documentation

This directory contains long-form documentation artifacts for ferrompi. Each
file is also embedded into the rustdoc HTML via `#[doc = include_str!(...)]`
modules under `ferrompi::doc::*`, so the same content is navigable from the
[published rustdoc](https://cobre-rs.github.io/ferrompi/ferrompi/doc/index.html).

> **Navigation note for rustdoc users:** Prose in these files contains relative
> links (e.g., `adr/0002-handle-tables.md`) that resolve correctly on GitHub
> but do not resolve inside rustdoc, where each document is rendered as a flat
> module page. When reading these docs via rustdoc, navigate between documents
> using the `ferrompi::doc` module index rather than clicking prose links.

> **Mermaid diagrams:** `architecture.md` contains Mermaid diagram syntax.
> Diagrams render on GitHub. In rustdoc they appear as plain fenced code blocks
> because rustdoc does not execute the Mermaid JS renderer.

---

## Long-form guides

- [Architecture](architecture.md) — Six-layer stack, handle tables, thread-safety model, C layer scope, FFI/ABI invariants, and generic `MpiDatatype` design. Start here if you are contributing to ferrompi internals.
- [Migrating from rsmpi](migrating-from-rsmpi.md) — Function-for-function API mapping, migration cookbook examples, unsupported features, and API ergonomic differences for developers coming from rsmpi.
- [MPI implementation compatibility](mpi-compatibility.md) — ferrompi's reference for selecting, building against, and running under an MPI library: the CI-tested feature table (MPICH 4.2.1, Open MPI 4.1.6, Open MPI 5.0.7), known implementation issues, and how to report new compatibility data.

## Architecture Decision Records

ADRs record the major design choices made during ferrompi's development. An
accepted ADR's decision text does not change: corrections are appended as
dated amendments and marked where the corrected text sits, and a changed
decision gets a new ADR that supersedes the old one.

- [ADR-0001: Why a C wrapper](adr/0001-why-c-wrapper.md) — Why ferrompi uses a hand-written C shim layer rather than `bindgen`-generated bindings: one Rust API across MPI implementations (source portability), `_c` large-count version gating, and the op-trampoline infrastructure.
- [ADR-0002: Handle tables](adr/0002-handle-tables.md) — Concurrency strategy for the request table under `MPI_THREAD_MULTIPLE`. Justifies C11 atomic `compare_exchange_strong` over pthread mutex and a lock-free Treiber stack.
- [ADR-0003: Generic MPI datatype](adr/0003-generic-mpi-datatype.md) — Design of the sealed-trait type family (`MpiDatatype`, `MpiIndexedDatatype`, `BytePermutable`, `AtomicMpiDatatype`).
- [ADR-0004: Persistent collective approach](adr/0004-persistent-collective-approach.md) — `PersistentRequest` lifecycle and buffer-lifetime invariants: the `*_init` / `start` / `wait` design and `Drop` behaviour.
- [ADR-0005: MPI Op create](adr/0005-mpi-op-create.md) — Seven decisions covering `MPI_Op_create` safety: closure storage, `Send + Sync + 'static` bounds, `MPI_Op_free`-before-slot-release drop ordering, default commutativity, per-slot baked-index C trampolines, and `catch_unwind + abort` panic handling.
- [ADR-0006: MPI 5.0 ABI direction](adr/0006-mpi5-abi-direction.md) — How ferrompi reaches binary portability through the MPI 5.0 standard ABI, and why the draft ABI is rejected.
