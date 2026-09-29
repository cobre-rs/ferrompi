# ADR-0003: Sealed Generic MpiDatatype Trait Family

**Status:** Accepted — 2026-05-17; amended 2026-09-24
**Date:** 2026-05-17
**Deciders:** Rogerio Alves

---

## Context

Every MPI communication operation — broadcast, send, receive, reduce, scatter,
gather, and their variants — requires three pieces of information about a buffer:
a pointer to the first element, an element count, and an `MPI_Datatype` handle
that tells the MPI library how to interpret each element (e.g., `MPI_FLOAT`,
`MPI_INT32_T`). In a Rust API, this means that every generic function over a
buffer type `T` must have a mechanism to map `T` to the corresponding
`MPI_Datatype` handle before crossing the FFI boundary.

ferrompi exposes every collective and point-to-point call in terms of Rust
slices: `broadcast(&mut [T], root)`, `send(&[T], dest, tag)`, and so on. The
generic parameter `T` appears in every public API signature. There must be a
trait that bounds `T` and provides the `MPI_Datatype` mapping with no overhead
per call.

### Why this is non-trivial

The problem is not simply associating a constant with each type. The deeper
constraint is that ferrompi's C layer (`csrc/ferrompi.c` and `csrc/ferrompi.h`)
resolves the `MPI_Datatype` handle through a `switch` statement on a
`repr(i32)` tag integer passed from Rust. The C-side `FERROMPI_*` defines and
the `switch` cases form a closed enumeration: they are written by hand, compiled
into the shared library, and cannot be extended at runtime by an external crate.
Any Rust design that allows external crates to add new buffer types to the
generic API must therefore either (a) send the type information through a
mechanism other than the tag switch — which adds a runtime dispatch cost or a
vtable — or (b) break the C layer's closed enumeration guarantee.

A second complication is that not all operations are valid for all primitive
types. The MPI standard distinguishes:

- General communication types: `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`.
- MAXLOC/MINLOC paired types: six predefined struct types (`MPI_FLOAT_INT`,
  `MPI_DOUBLE_INT`, etc.) that carry both a value and a rank index.
- Byte-level bitwise reduction types: types whose entire memory representation
  consists of meaningful bytes with no padding, suitable for element-wise
  `BitwiseOr`/`BitwiseAnd`/`BitwiseXor` via `MPI_BYTE`.
- Atomic RMA types: the MPI specification does not require implementations to
  support floating-point `MPI_Compare_and_swap`; only integer and byte types are
  portable for atomic RMA operations.

A single trait cannot express all four distinctions without subtyping. The
design must support a family of traits with independent implementation sets and
explicit exclusions.

rsmpi, the most widely used prior Rust MPI binding, solves the datatype mapping
problem with an `Equivalence` trait that external crates may implement for their
own types. This makes the type set open and extensible. ferrompi's constraints
differ: the C layer's `DatatypeTag` enumeration is closed by construction, and
any type that implements `MpiDatatype` must have a corresponding `FERROMPI_*`
constant in `csrc/ferrompi.h` and a matching `case` in every C-side switch.
Adding an rsmpi-style open trait would either require rewriting the C dispatch
layer or accepting a vtable lookup on every buffer element.

---

## Decision Drivers

1. **Zero-overhead dispatch.** The `MPI_Datatype` handle lookup must cost nothing
   per element. Every collective call transfers potentially millions of elements;
   adding a function pointer dereference or a branch per element is unacceptable.
   The tag lookup must resolve to a compile-time constant so the compiler can
   inline it and the C-side switch reduces to a single indirect jump at the call
   site, not inside the element loop.

2. **Closed type set.** ferrompi supports exactly the primitive types documented
   in `README.md` plus the custom datatype mechanism for user-defined
   structures. The C-side `DatatypeTag` enumeration is not extensible at runtime:
   new variants require source changes in `csrc/ferrompi.h`, the `switch` bodies
   in `csrc/ferrompi.c`, and the Rust `DatatypeTag` enum in `src/datatype.rs`,
   followed by a recompile. External crates must not be able to add types to the
   set by implementing a trait, because such types would have no corresponding C
   `case` to dispatch to.

3. **Compile-time rejection of misuse.** Passing a `bool`, `String`, or
   `Vec<f64>` where an `&[T: MpiDatatype]` is required must produce a compile
   error, not a runtime `MPI_ERR_TYPE`. The programmer must learn about the type
   restriction at the earliest possible moment — during compilation — rather than
   at the MPI call site during program execution, where a diagnostic would require
   cross-referencing an MPI error code with a ferrompi API.

4. **Subtype constraints for restricted operations.** Some MPI operations accept
   only a subset of the general communication types. RMA atomic operations
   (`MPI_Compare_and_swap`, `MPI_Fetch_and_op`) do not require MPI implementations
   to support floating-point types. MAXLOC/MINLOC reductions operate exclusively
   on predefined paired value+index structures, not on scalar primitives. The
   design must represent these distinctions as separate traits with their own
   implementation sets, so that a method like `Win::compare_and_swap` can carry a
   distinct bound (`T: AtomicMpiDatatype`) that excludes `f32` and `f64` at
   compile time.

5. **No `Drop` cost for primitive types.** `T: MpiDatatype` for primitive types
   must not introduce any `Drop` implementation. The primitives `f32`, `f64`,
   `i32`, `i64`, `u8`, `u32`, and `u64` are already `Copy`; the trait bound
   must not prevent the compiler from treating buffers as plain stack or heap
   data with no destructor overhead. The `MpiDatatype` supertrait bound includes
   `Copy`, which structurally prevents implementing `Drop`.

---

## Options Considered

### Option A (chosen): Sealed trait family with `const TAG: DatatypeTag`

Each trait in the family is sealed via a `pub(crate) mod sealed` pattern: a
private `Sealed` marker trait is placed in a crate-internal module, and the
public trait has `sealed::Sealed` as a supertrait. Because the marker trait is
not publicly accessible, external crates cannot implement the supertrait bound
and therefore cannot implement the public trait. The implementation in
`src/datatype.rs` defines four independent sealed modules (`sealed`,
`sealed_indexed`, `sealed_byte`, `sealed_atomic`) so that the four public traits
remain disjoint — a type that implements `MpiDatatype` does not automatically
satisfy `MpiIndexedDatatype`, and vice versa.

The `MpiDatatype` trait carries a single associated constant:

```rust,ignore
pub trait MpiDatatype: sealed::Sealed + Copy + Send + 'static {
    const TAG: DatatypeTag;
}
```

`DatatypeTag` is a `#[repr(i32)]` enum whose discriminants match the
`FERROMPI_*` defines in `csrc/ferrompi.h`. The constant `TAG` is resolved at
compile time; the C layer receives it as an `i32` argument and dispatches
through a `switch`. No allocation, no vtable, no branch per element.

Subtype traits compose naturally via supertrait bounds. `AtomicMpiDatatype` does
not redeclare `TAG`; it inherits the `MpiDatatype` bound implicitly through the
requirement that any type satisfying `AtomicMpiDatatype` must also satisfy
`MpiDatatype`. Each subtype trait has its own sealed module and its own
`compile_fail` doctest verifying that disallowed types (e.g., `f64` for
`AtomicMpiDatatype`) produce a compile error.

Pros:

- The `TAG` constant is resolved at compile time; zero per-element cost in
  production builds and in debug builds.
- The closed type set is enforced structurally: the `sealed::Sealed` supertrait
  cannot be implemented outside `src/datatype.rs`. No runtime guard is needed.
- Subtype traits share the same dispatch mechanism without duplicating impls.
  `AtomicMpiDatatype` impls are five lines; they do not repeat the tag values.
- The `compile_fail` doctest pattern catches regressions in sealing immediately;
  if a future refactor accidentally exposes the sealed module, doctests fail.
- `Copy + Send + 'static` supertrait bounds prevent `Drop` impls and restrict
  buffer element types to values safe to pass across threads and process
  boundaries.

Cons:

- External crates cannot add new primitive types to the MPI dispatch table
  without modifying ferrompi's source. This is intentional — the C layer
  cannot be extended at runtime — but it means ferrompi cannot be used as a
  generic MPI backend for arbitrary numeric tower types (e.g., `half::f16`,
  `num_complex::Complex<f64>`). User-defined types route through `CustomDatatype`
  builders instead (see Consequences).

### Option B (rejected): Open `MpiDatatype` trait, rsmpi-style

Under this design, `MpiDatatype` carries no sealed supertrait. Any external
crate can implement it:

```rust,ignore
// External crate
impl ferrompi::MpiDatatype for MyMatrix { ... }
```

Pros: extensibility matches the Rust trait system's intended use; symmetry with
`Send`/`Sync`; compatible with numeric tower crates that define their own float
or complex types.

Cons:

- The C-side `DatatypeTag` enumeration is not extensible at runtime. A user type
  that implements `MpiDatatype` has no corresponding `FERROMPI_*` define and no
  matching `switch` case in `csrc/ferrompi.c`. The C layer would either have to
  use a fallback path (e.g., a dynamic lookup through a Rust-side registry) or
  crash with `MPI_ERR_TYPE`. Both outcomes violate decision driver 1 (zero
  overhead) and decision driver 3 (compile-time rejection).
- Implementing a sound open variant would require replacing the tag-switch
  dispatch in C with a vtable or a function-pointer callback into Rust for every
  datatype resolution. This is architecturally incompatible with the design
  established in ADR-0001 (hand-written C wrapper layer).
- rsmpi's `Equivalence` trait is appropriate for rsmpi's architecture, which
  uses `bindgen`-generated bindings and resolves `MPI_Datatype` handles through
  the MPI type system itself. ferrompi's C layer does not use `MPI_Type_create_*`
  for primitives; it maps Rust types to predefined MPI constants directly.

### Option C (rejected): Runtime tagged enum

Under this design, no trait carries type information. Callers pass a separate
`DataKind { F32, F64, I32, ... }` parameter alongside the buffer:

```rust,ignore
world.broadcast(buf: &mut [u8], kind: DataKind, root: i32) -> Result<()>
```

Pros: no trait machinery; no const fields; simpler to explain to new
contributors.

Cons:

- Every call site must supply `DataKind` explicitly, and the compiler cannot
  verify that the buffer element type matches the declared kind. A call
  `broadcast(f32_buffer, DataKind::F64, 0)` compiles but produces incorrect
  MPI behavior at runtime — the `MPI_Datatype` handle does not match the buffer
  layout. This is the exact class of error that decision driver 3 requires to be
  caught at compile time.
- The API ergonomics are significantly worse than idiomatic Rust. Existing
  Rust MPI libraries and the broader numeric computing ecosystem in Rust use
  generic functions over type-parameterized buffers, not stringly or enum-typed
  kind tags.
- Type-erased APIs prevent the compiler from eliminating dead branches for
  unused types. The sealed-trait design allows monomorphization; each concrete
  `T` produces a distinct, optimized code path.

---

## Decision

**Option A — sealed trait family with `const TAG: DatatypeTag` — is chosen.**

The decision is grounded directly in the five decision drivers:

- **Driver 1 (zero overhead):** `T::TAG` is a compile-time constant. The C
  layer's `switch` on the tag integer is evaluated once per call, not per
  element. Monomorphization of generic functions over `T: MpiDatatype` produces
  one code path per concrete type with no vtable and no dynamic dispatch.

- **Driver 2 (closed type set):** The `pub(crate) mod sealed` pattern makes it
  structurally impossible for external crates to implement `MpiDatatype`. The
  invariant does not rely on documentation or convention; it is enforced by the
  Rust privacy system.

- **Driver 3 (compile-time rejection):** A call site that passes a `bool` or
  `String` where `T: MpiDatatype` is required fails to compile because neither
  type is in `sealed::Sealed`. The compiler error appears at the generic bound,
  not at runtime.

- **Driver 4 (subtype constraints):** `AtomicMpiDatatype`, `MpiIndexedDatatype`,
  and `BytePermutable` are independent sealed traits with their own
  implementation lists. Methods that require a stricter type set carry the
  narrower bound. The compiler enforces the restriction; no runtime check is
  needed.

- **Driver 5 (no `Drop` cost):** The `Copy` supertrait bound on `MpiDatatype`
  prevents any type with a `Drop` impl from satisfying the bound. The seven
  primitive types are `Copy` by definition; no wrapper or destructor is
  introduced.

Option B is rejected because an open trait cannot be reconciled with the C
layer's closed `DatatypeTag` switch without adding runtime overhead or replacing
the dispatch architecture entirely.

Option C is rejected because runtime kind tags shift type mismatches from
compile time to runtime, undermining the primary safety guarantee of a
typed Rust MPI API.

---

## Consequences

**Sealing is structural, not conventional.** The four private modules
(`sealed`, `sealed_indexed`, `sealed_byte`, `sealed_atomic`) in `src/datatype.rs`
make each trait's implementation set a crate-internal invariant. Adding a new
implementation requires modifying `src/datatype.rs` directly; there is no
mechanism by which a downstream crate could extend any of the four traits.
Attempting to implement `MpiDatatype` in an external crate produces a compiler
error on the `sealed::Sealed` supertrait, not a linker error or a runtime
failure.

**Subtype trait composition.** `AtomicMpiDatatype`, `MpiIndexedDatatype`, and
`BytePermutable` are each sealed by their own independent module. This means the
four trait families are disjoint by construction: a type that satisfies
`MpiDatatype` does not implicitly satisfy `MpiIndexedDatatype`. The disjoint
sealing prevents accidental cross-use — for example, passing a `FloatInt`
struct (which is `MpiIndexedDatatype`) where a plain `MpiDatatype` is required
will fail at the call site because `FloatInt` is not in `sealed::Sealed`.

**`AtomicMpiDatatype` excludes `f32` and `f64`.** Per the MPI specification,
`MPI_Compare_and_swap` is defined only for integer and byte types. Floating-point
CAS is not portable across MPI implementations. `AtomicMpiDatatype` is
implemented for `i32`, `i64`, `u32`, `u64`, and `u8` only; `f32` and `f64` are
excluded. This is encoded structurally: the macro that generates `AtomicMpiDatatype`
impls in `src/datatype.rs` is invoked only for integer types. A method
`Win::compare_and_swap` carrying a `T: AtomicMpiDatatype` bound rejects
`Win<f64>` at compile time. The `compile_fail` doctest in `src/datatype.rs`
verifies this rejection.

**`AtomicMpiDatatype` is gated on the `rma` feature.** The trait and its impls
are wrapped in `#[cfg(feature = "rma")]`. Projects that do not enable RMA
operations do not pay the compile time for the trait or its impls, and the
`AtomicMpiDatatype` name is not visible in their API surface.

**`compile_fail` doctests verify sealing.** Each of the four traits has a
`compile_fail` doctest in `src/datatype.rs` that attempts to implement the trait
for a disallowed type and confirms the result does not compile. These doctests
act as regression tests for the sealing invariant. A future refactor that
accidentally exposes a sealed module will cause `cargo test --doc` to fail.

**Adding a new primitive type requires three coordinated changes.** The closed
type set is a commitment: adding a type to `MpiDatatype` is not a local change.
It requires (1) a new `DatatypeTag` variant in `src/datatype.rs` and a
corresponding `FERROMPI_*` define in `csrc/ferrompi.h`, (2) a new `impl MpiDatatype`
block in `src/datatype.rs` via the `impl_mpi_datatype!` macro, and (3) a new
`case` in every C-side `switch` on the tag value in `csrc/ferrompi.c`. All
three changes must land together; a partial update (new Rust impl without a new
C case) produces `MPI_ERR_TYPE` at runtime for the new type. The coordination
requirement is a deliberate friction point that discourages ad-hoc additions to
the type set.

**User-defined non-primitive types are supported through `CustomDatatype` builders.**
The sealed-trait design does not prevent ferrompi from operating on
user-defined structs, padded layouts, or non-primitive numeric types. The
`CustomDatatype` API (`src/datatype_builder.rs`) provides
constructors for contiguous blocks, strided vectors, and heterogeneous structs
via `MPI_Type_contiguous`, `MPI_Type_vector`, and `MPI_Type_create_struct`. A
`CustomDatatype` produces a committed `MPI_Datatype` handle at runtime and is
passed to the `*_custom` family of collective methods. The two mechanisms are
intentional and complementary: `MpiDatatype` gives zero-overhead dispatch for
the seven well-known primitives; `CustomDatatype` gives full MPI derived-type
power for everything else. They share the same C dispatch path through
`get_datatype` (for `DatatypeTag` values) and `get_datatype_committed` (for
`CustomDatatype` handles).

---

## Alternatives Considered

**Macro-generated impls for all numeric primitives.** A procedural macro could
generate `MpiDatatype` impls for every type in a specified list, including types
like `usize`, `isize`, `u128`, and `i128`. This was considered and rejected
because the MPI standard does not define portable equivalents for these types.
`usize` maps to `MPI_AINT` on 64-bit platforms but has no fixed-size MPI
counterpart that works uniformly across 32-bit and 64-bit systems. `u128` has
no predefined MPI equivalent at all. Generating impls for these types would
require either runtime size detection (violating driver 1) or platform-conditional
impls that produce different `DatatypeTag` values on different architectures,
which breaks the ABI stability invariant of `DatatypeTag`.

> **Amended 2026-09-24 — Discriminant values are internal.** See [Amendments](#amendments).

**Attribute-macro opt-in on user types.** A `#[derive(MpiDatatype)]` or
`#[ferrompi::mpi_type(f64)]` attribute macro could allow user types to opt in
to the trait for a specific base MPI type. This is rejected for the same reason
as Option B: a user-declared impl would require a corresponding C `case` that
does not exist. The attribute could theoretically generate a C extension through
a build script, but this would couple user crates to ferrompi's build
infrastructure and break the clean separation between the Rust trait layer and
the C dispatch layer. `CustomDatatype` builders address the same need without
modifying the closed `DatatypeTag` enumeration.

---

## Amendments

### 2026-09-24 — Discriminant values are internal

The Alternatives section above called the `DatatypeTag` discriminant values an ABI stability invariant, and `docs/architecture.md` presented the `DatatypeTag` and `ReduceOp` discriminants as a stability promise. That promise is withdrawn. The discriminant values are an internal contract between the Rust enums and the C shim's `FERROMPI_*` defines; any release may change them.

No discriminant value changed in 0.6.0. Code names the enum variant, never its integer value. Keeping the values internal leaves room to encode them as MPI 5.0 standard-ABI handle values.

---

## Status

Accepted — 2026-05-17; amended 2026-09-24.
