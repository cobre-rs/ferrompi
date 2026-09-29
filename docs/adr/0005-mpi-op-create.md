# ADR-0005: MPI_Op_create Safety Model — Closure Storage, C Trampoline, and Thread-Safety Contract

**Status:** Accepted — 2026-05-17; amended 2026-09-24
**Date:** 2026-05-17
**Deciders:** Rogerio Alves

---

## Context

User-defined reduction operations use `MPI_Op_create`. This is the
only ferrompi API surface where the MPI library calls back into Rust: the C
function pointer registered with `MPI_Op_create` has signature
`void (*)(void*, void*, int*, MPI_Datatype*)` and is invoked by MPI on
potentially any thread under `MPI_THREAD_MULTIPLE`.

Every existing FFI call in ferrompi flows Rust → C. `MPI_Op_create` inverts
that flow: C → Rust. This inversion creates three constraints with no
precedent in prior epics:

1. **Closure lifetime.** The Rust closure passed by the caller has an opaque
   type and may hold borrowed or owned state. MPI holds the `MPI_Op` handle —
   and therefore may invoke the function pointer — until `MPI_Op_free` is
   called. The closure storage must outlive the C-side handle with no
   possibility of a dangling pointer.

2. **Thread-safety.** Under `MPI_THREAD_MULTIPLE` (the thread level exposed by
   `Mpi::init_thread(ThreadLevel::Multiple)`), multiple threads within the
   same MPI process may invoke the same reduction operation concurrently (e.g.,
   concurrent `MPI_Allreduce` calls on different communicators can both
   dispatch user-defined ops). The closure must be callable from multiple
   threads simultaneously; it cannot hold non-`Sync` state.

3. **Panic barrier.** A Rust `panic!` unwinds through Rust stack frames using
   the Rust unwinding ABI. Crossing an FFI boundary while unwinding is
   undefined behaviour; the C/MPI layer has no mechanism to propagate or
   observe a Rust panic. The trampoline must intercept any panic before it
   reaches the C frame.

None of these problems arise in any existing ferrompi code path. This ADR
resolves all three, plus four related design questions (commutativity flag,
trampoline dispatch, datatype contract, and the drop ordering that avoids
use-after-free), before implementation begins.

### Related existing patterns

- `csrc/ferrompi.c` lines 41-71: fixed-size static slot tables for
  `request_table`, `win_table`, `info_table`, `group_table`, and
  `datatype_table`. Each table has a parallel `int *_used[]` occupancy array
  and an `alloc_*`/`free_*`/`get_*` function triple.
- `docs/adr/0002-handle-tables.md`: C11-atomic slot-claim strategy for the
  request table under `MPI_THREAD_MULTIPLE`. The same pattern is referenced
  here for the op table.
- `src/datatype.rs` lines 126-136: `MpiDatatype` sealed trait with `const TAG:
DatatypeTag` that maps Rust types to C-side `FERROMPI_*` tag integers. The
  same `MpiDatatype` bound is used in the closure type constraint.

---

## Decision

Seven decisions are recorded below. Each section names the chosen mechanism
and explicitly rejects the alternatives.

---

### Decision 1: Closure Storage — Per-op static slot table in `csrc/ferrompi.c`

**Decision.** Closure pointers are stored in a fixed-size static array
`op_closure_table` in `csrc/ferrompi.c`, parallel to the existing `win_table`
and `info_table` arrays. The table holds `void*` fat-pointer pairs (data
pointer + vtable pointer) cast from `*mut dyn Fn(...)`. Slot allocation and
release follow the existing `alloc_*`/`free_*` pattern; occupancy is tracked
by a parallel `atomic_int op_used[MAX_OPS]` array using the C11 CAS strategy
established in ADR-0002.

> **Amended 2026-09-24 — Closure registry in Rust.** See [Amendments](#amendments).

**Rationale.** The slot table is the only closure-storage mechanism that
satisfies both lifetime safety and clean deterministic reclamation. The Rust
side boxes the closure into a `Box<dyn Fn(&[T], &mut [T])>` and stores the raw
pointer in the table; the box is reconstructed from the raw pointer in `Drop`
and dropped there — no memory is leaked. The C trampoline for slot `N`
reconstructs the fat pointer from `op_closure_table[N]` and calls it. The
table is entirely within the C static segment, which lives for the process
lifetime, so the slot pointer cannot dangle before the Rust `UserOp` drops it.
Using `atomic_int` for occupancy (as in the request table) makes
`alloc_op`/`free_op` safe under `MPI_THREAD_MULTIPLE` concurrent creation of
distinct user ops.

**Rejected: `Box::leak` into `'static`.** `Box::leak` converts a `Box<T>` to
a `&'static mut T`, which is syntactically `'static` but is never reclaimed.
Leaking is acceptable for true singletons (e.g., a global registry), but
`UserOp` is an owned, droppable value. A user-facing API that leaks memory on
every `UserOp::new()` call, with no reclamation path short of process exit, is
not acceptable. There is no mechanism by which `Drop for UserOp` could
de-leak a `Box::leak`'d allocation.

**Rejected: Thread-local registry.** A `thread_local!` registry would store
the closure on the creating thread's local storage and expose it only on that
thread. Under `MPI_THREAD_MULTIPLE`, MPI may invoke the reduction function on
any thread — not necessarily the thread that called `MPI_Op_create`. Accessing
a thread-local from a different thread is unsound; the closure pointer would
point into TLS storage that is inaccessible (or destroyed) on the calling
thread.

---

### Decision 2: `Send + Sync + 'static` Bound on the Closure

**Decision.** The closure type parameter `F` on `UserOp<T>` must satisfy
`F: Fn(&[T], &mut [T]) + Send + Sync + 'static`. All three bounds are
mandatory.

**Rationale.** `Send` is required because the closure is moved into a slot in
global static storage (`op_closure_table`), which is shared across all threads.
Moving a non-`Send` type into shared storage would allow it to be accessed from
a thread other than the one that created it, violating the `Send` contract.
`Sync` is required because, under `MPI_THREAD_MULTIPLE`, the MPI runtime may
call the trampoline concurrently from multiple threads for the same `MPI_Op`
(e.g., two threads calling `MPI_Allreduce` on different communicators with the
same user-defined op). A non-`Sync` closure type would allow data races on
the closure's captured state. `'static` is required because the closure is
stored in a raw pointer in the C static table; any borrow shorter than `'static`
in the captured state could be invalidated while the C handle is still live,
producing a dangling reference. Together, these three bounds are the minimum
necessary and sufficient set: they are the same constraints imposed by
`std::thread::spawn`.

**Ruled-out captures.** The combined bound rules out several capture patterns
that would otherwise be syntactically valid:

| Capture type         | Violates  | Reason                                               |
| -------------------- | --------- | ---------------------------------------------------- |
| `Rc<T>`              | `Send`    | `Rc` is not thread-safe (non-atomic reference count) |
| `Cell<T>`            | `Sync`    | `Cell` allows unsynchronized interior mutation       |
| `RefCell<T>`         | `Sync`    | `RefCell` borrow checks are not thread-safe          |
| `*const T`           | `Send`    | Raw pointers are not `Send` by default               |
| `&'a T` (non-static) | `'static` | Borrowed data may not outlive the `MPI_Op` handle    |

Callers who need shared mutable state inside the closure must use
`Arc<Mutex<T>>` or `Arc<RwLock<T>>`, which are `Send + Sync + 'static`.

---

### Decision 3: Lifetime Relationship and Drop Ordering

**Decision.** The closure storage always outlives the C-side `MPI_Op`.
`Drop for UserOp<T>` calls `MPI_Op_free` first and then releases the slot.
The ordering is: (1) `MPI_Op_free`, (2) `free_op_slot` (which reconstructs
the `Box` from the raw pointer and drops it).

> **Amended 2026-09-24 — Drop after finalize and on another thread.** See [Amendments](#amendments).

**Rationale.** The invariant "MPI_Op is freed before the closure slot is
released" is the key safety property. After `MPI_Op_free` returns, the MPI
library guarantees it will not invoke the user function pointer again. Only
after that guarantee is established may the closure storage be freed. If the
slot were released first — reconstructing and dropping the `Box` — a
concurrent MPI operation that still holds a reference to the `MPI_Op` could
invoke the trampoline against a freed function pointer, which is a
use-after-free.

**MPI_Op_free first** also means the trampoline can never be called with a
dangling closure pointer: the window between "MPI_Op_free returns" and "Box
is dropped" is a period during which neither the MPI library nor any Rust code
holds a live reference to the closure. The drop then runs in the conventional
Rust sense — no raw pointers are read after the `Box` is reconstructed.

The invariant is encoded structurally: `Drop for UserOp<T>` is the only code
that calls either `ferrompi_op_free` (the C wrapper for `MPI_Op_free`) or
`ferrompi_op_free_slot`. There is no public `UserOp::free()` method, and the
handle integer is not exposed to callers, so the ordering cannot be violated
from outside the module.

**Rejected: Drop the Box first.** Dropping the closure storage before calling
`MPI_Op_free` is a use-after-free if any in-flight collective operation is
still dispatching the user function. MPI provides no mechanism to query whether
an op is currently executing; the only safe signal is the return of
`MPI_Op_free`.

---

### Decision 4: Commutativity Flag — Default `commute = true`

**Decision.** `UserOp::new` does not expose a `commute` parameter. The
`commute` argument to `MPI_Op_create` is always set to `1` (true). Users who
require a non-commutative reduction must explicitly use a different
constructor: `UserOp::new_noncommutative`.

**Rationale.** The `commute` parameter is an optimizer hint: an `MPI_Op` marked
commutative allows MPI implementations to reorder operands for better
performance (e.g., binary tree reductions on hardware with non-uniform topology
can shuffle partial results for better load balance). Most user-defined
reductions encountered in scientific computing — sums, maxima, custom norms,
element-wise functions — are commutative. Defaulting to `true` produces the
correct result for the common case and allows MPI to apply its full set of
reordering optimizations. Operators that are genuinely non-commutative (e.g.,
matrix multiplication, string concatenation) are uncommon and require explicit
opt-out rather than a pervasive `commute: bool` parameter cluttering the common
call site.

Providing `UserOp::new_noncommutative` makes the non-commutative path
discoverable without making it the default. The Rust type system cannot express
commutativity statically, so the runtime flag must exist; it is better to push
it to a separate constructor name where its presence is self-documenting.

**Rejected: Expose as a parameter on `UserOp::new`.** A `UserOp::new(f, commute)`
signature forces every caller to supply a commutativity flag even when the
semantics of their operation are obviously commutative. API surface complexity
increases for no benefit in the common case. The flag is also easy to get wrong
silently: passing `false` for a commutative operation is always correct but
foregoes the optimization hint; passing `true` for a non-commutative operation
produces incorrect results. Defaulting to `true` via the common constructor
fails safe.

---

### Decision 5: Trampoline Dispatch — Slot Index Baked into Per-Slot C Functions

**Decision.** The C layer provides `MAX_OPS` distinct trampoline functions with
the naming pattern `ferrompi_user_op_trampoline_N` where `N` is the decimal
slot index (0 through `MAX_OPS - 1`). Each trampoline loads the closure pointer
from `op_closure_table[N]` using the slot index baked into its own function
body, then invokes the closure. The array of function pointers
`ferrompi_user_op_trampolines[MAX_OPS]` is exposed to Rust so that
`alloc_op_slot(N)` can pass `ferrompi_user_op_trampolines[N]` to
`MPI_Op_create` as the user function pointer.

> **Amended 2026-09-24 — Closure registry in Rust.** See [Amendments](#amendments).

**Rationale.** `MPI_Op_create` accepts a plain C function pointer; it provides
no `void* user_data` argument alongside the function pointer. The only way for
a single trampoline to locate its associated closure without a side channel is
to encode the slot index into the function itself — either by having `N`
distinct functions each hard-coded to their slot, or by writing self-modifying
code (unacceptable). A compile-time loop using a C macro (`FERROMPI_DEFINE_OP_TRAMPOLINE(N)`)
generates all `N` functions from a single template body, keeping the
implementation DRY while producing distinct function pointers. Each trampoline
body is four statements: load the fat pointer from `op_closure_table[N]`,
wrap the invocation in a `catch_unwind` (see Decision 6 via the `abort_on_panic`
C callback), reconstruct the `&[T]` and `&mut [T]` slices from `invec`,
`inoutvec`, and `*len`, and call the Rust closure through the fat pointer.

**Rejected: Single trampoline with thread-local slot lookup.** If a single
`ferrompi_user_op_trampoline(void*, void*, int*, MPI_Datatype*)` function were
used, it would need to locate which slot's closure to invoke. The only
candidate for this side channel is thread-local storage: the calling thread
stores the slot index in TLS before MPI dispatches the function pointer. This
is fundamentally unsafe under `MPI_THREAD_MULTIPLE`: MPI is free to call the
user function on a thread-pool thread that is entirely under MPI's control, not
the application's. TLS set by the application thread before the `MPI_Allreduce`
call is not observable on the MPI-internal thread that executes the reduction.
The slot lookup would read uninitialized TLS, producing arbitrary behaviour.

**Rejected: Per-slot generated function pointers (JIT/mmap).** Generating
function code at runtime (writing machine code to an `mmap(PROT_EXEC)` region)
would allow a single template to be instantiated with any slot index. This
approach introduces JIT complexity, requires platform-specific code generation,
and raises significant security concerns (W^X policy violations on hardened
systems). It is entirely disproportionate to the problem; compile-time macro
generation of `MAX_OPS` static functions has zero runtime overhead, zero
platform dependencies, and is trivially auditable.

---

### Decision 6: Panic Handling — Abort the Process

**Decision.** The trampoline wraps the Rust closure invocation in
`std::panic::catch_unwind`. If `catch_unwind` returns `Err(_)` (a panic was
caught), the trampoline calls `std::process::abort()` immediately. It does not
log, does not set a sticky flag, and does not leave `inoutvec` in the partially
modified state produced by the panicking closure.

**Rationale.** The MPI user-function signature has no error return path: it is
`void (*)(void*, void*, int*, MPI_Datatype*)`. A panic that is not caught
before the trampoline returns to the C/MPI caller would unwind through C stack
frames using the Rust unwinding ABI — this is undefined behaviour. The three
possible recovery strategies — abort, continue with unmodified `inoutvec`, or
set a sticky error flag — have the following properties:

- **Abort**: Immediate, deterministic, no possibility of corrupted collective
  state. Collective operations involve all ranks; a panicking rank that returns
  a garbage `inoutvec` to MPI would allow the collective to silently complete
  with incorrect results on all ranks. Abort prevents this by ensuring that the
  failure mode is loud and traceable.
- **Continue with unmodified `inoutvec`**: The partial reduction (closure
  executed through some range, then panicked) leaves `inoutvec` in a state that
  may be partially correct, completely incorrect, or internally inconsistent
  depending on where the panic occurred. Returning this to MPI and allowing the
  collective to continue produces a collective result that is wrong in a way
  that is invisible to all participants. Silent data corruption in a reduction
  result is strictly worse than a process abort for HPC use cases.
- **Sticky error flag**: There is no MPI mechanism by which the user function
  can signal an error that causes `MPI_Allreduce` to return a non-`MPI_SUCCESS`
  code to the caller. The sticky flag would be observed only on the next
  ferrompi call on the same thread — at that point the collective has already
  completed on all other ranks with a corrupted partial result, and the error
  signal is too late to be actionable.

Process abort is therefore the only choice that prevents silent data corruption.
Panicking inside a reduction closure is an application-level programming error;
the right response is to fail loudly.

`catch_unwind` is required even when the intent is to abort, because without it
the panic unwinds through C frames before the process terminates, which is
undefined behaviour. `catch_unwind` ensures Rust's unwinding is contained
within Rust frames; the `abort` call inside the catch block terminates the
process cleanly.

---

### Decision 7: Datatype Contract — Generic `UserOp<T: MpiDatatype>` with Debug-Mode Tag Assertion

**Decision.** `UserOp<T>` is generic over `T: MpiDatatype`. The closure type
is `F: Fn(&[T], &mut [T]) + Send + Sync + 'static`. The trampoline reconstructs
the `invec` and `inoutvec` pointers as `*const T` and `*mut T` respectively,
using `*len` as the element count to construct `&[T]` and `&mut [T]` slices.
A `debug_assert_eq!` inside the trampoline verifies that `T::TAG as i32` equals
the `MPI_Datatype` argument passed by MPI; in release builds the assert is
elided. Mismatches between the declared type `T` and the actual MPI datatype
passed at runtime are programming errors detectable in debug mode.

> **Amended 2026-09-24 — Closure registry in Rust.** See [Amendments](#amendments).

**Rationale.** Genericity over `T: MpiDatatype` is the correct level of
abstraction: it lets the compiler enforce type consistency at `UserOp::new` time
(the caller declares what element type the op works on), carries the type
information through to the trampoline (the trampoline knows how to compute
`std::mem::size_of::<T>()` for any slice arithmetic), and reuses the existing
`MpiDatatype` sealed-trait infrastructure from `src/datatype.rs`. The `const
TAG: DatatypeTag` associated constant provides a compile-time integer that the
trampoline can compare against the runtime `MPI_Datatype` argument for sanity
checking.

The tag check is debug-only because: (a) in a correct program, the MPI
datatype passed at reduction time always matches the `UserOp<T>` type
parameter — MPI dispatches the op with the same datatype that was used when
registering it; (b) the check requires converting an opaque `MPI_Datatype`
handle to a `FERROMPI_*` tag integer, which involves a runtime table lookup in
the C layer; and (c) in release builds the tag check would add a branch on
every element-wise call inside a reduction, which may execute many millions of
times. Debug mode is the appropriate place for a correctness assertion with
a measurable hot-path cost.

**Rejected: Untyped `UserOp` with raw `*mut c_void` closure.** An untyped API
that passes `(invec: *mut c_void, inoutvec: *mut c_void, len: i32)` to the
closure shifts the burden of safe slice reconstruction onto the caller. Every
caller must cast to the correct type, compute the byte length, and handle
misalignment — the same operations that `UserOp<T>` handles correctly and once
internally. An untyped API also makes it impossible to statically enforce that
the `MPI_Op` is registered with a matching MPI datatype.

**Rejected: Type-erased `Box<dyn Fn(...)>` with no static type parameter.**
A type-erased trait object for the closure would prevent the trampoline from
knowing `T` at the call site and would require the caller to transmit the
element type through a separate runtime parameter. This loses the static
guarantee that the slice type matches the registered datatype and forces all
alignment and size computations to be done dynamically.

---

## Alternatives Considered

### A: `Box::leak` into `'static` for closure storage

Under this approach, `UserOp::new` would call `Box::leak(Box::new(f))` to
produce a `&'static F` and store the raw pointer as the trampoline's user data.
Because `Box::leak` is not a standard `MPI_Op_create` parameter, the pointer
would still need to be encoded via one of the trampoline dispatch mechanisms
(see Decision 5). The critical failure is in `Drop`: `Box::leak` does not
provide a mechanism to reclaim the allocation. The pointer stored in the C
table could be cast back to `Box<F>` and dropped by transmuting the `&'static`
reference, but this is unsound — it violates the uniqueness contract of `&mut T`
and requires additional unsafe with no advantage over the slot table approach.
In practice, every `UserOp` allocation would permanently leak the closure
memory. For long-running MPI programs that create many user-defined ops (e.g.,
a solver that registers new operations each iteration), this leak is
operationally unacceptable. Rejected because it leaks memory with no
reclamation path.

### B: Thread-local registry for closure storage

Under this approach, a `thread_local!` `HashMap<i32, Box<dyn Fn(...)>>` would
store closures keyed by slot index. `UserOp::new` would insert into the TLS map
on the creating thread; the trampoline would look up the closure in the TLS map
on the executing thread. This approach fails under `MPI_THREAD_MULTIPLE` because
MPI may invoke the reduction function on an MPI-internal thread-pool thread that
has no TLS entry for the registered op. The lookup would return `None`, and
the trampoline would have no closure to call. There is no portable, standard way
to propagate Rust TLS entries to threads not created by the application. Even if
TLS sharing were possible via raw pointer casts across thread boundaries, doing
so would require `unsafe` that violates TLS lifetime guarantees. Rejected because
TLS is per-thread and MPI user functions are called on arbitrary threads.

### C: Single dispatch trampoline via MPI attribute key

MPI provides `MPI_Op_set_attr` and `MPI_Op_get_attr` for attaching opaque
`void*` attributes to an `MPI_Op` handle. A single trampoline could retrieve
its closure pointer via `MPI_Op_get_attr`. This approach is clean in principle
but fails for two reasons: (a) `MPI_Op_get_attr` is not part of MPI 3.1 or MPI
4.x — the attribute interface for ops is not standardised in the same way as
communicator or window attributes; and (b) even if implemented by a specific
MPI library, it would not be portable across MPICH, Open MPI, and Cray MPT.
Rejected due to non-standard MPI interface dependency.

---

## Amendments

### 2026-09-24 — Closure registry in Rust

- The shipped design differs from Decisions 1, 5 and 7. The closure registry lives in
  Rust (`src/op.rs`): an array of `AtomicPtr` slots, each holding a thin pointer to a
  boxed byte-level closure. A slot is published with a release store before
  `MPI_Op_create` and read with an acquire load.
- C keeps the op table and the per-slot trampolines, and trampoline `N` passes only `N`
  to `rust_user_op_invoke`, which runs the closure inside `catch_unwind` and aborts on
  panic.
- `UserOp<T>` stays generic over `T: MpiDatatype`: `UserOp::new` wraps the typed closure
  in a byte-level adapter that rebuilds `&[T]`/`&mut [T]` from MPI's element count.
  Decision 7's rejection of a type-erased closure is reversed for storage only.
- There is no datatype-tag assertion: the trampoline ignores MPI's datatype argument,
  and the only path that applies a `UserOp<T>`, `allreduce_with_op`, passes `T`'s own
  datatype.
- Decision 3's order is kept: `MPI_Op_free`, then the Rust callback that drops the
  closure, then the slot.

### 2026-09-24 — Drop after finalize and on another thread

- `Drop for UserOp` first calls the lifecycle guard.
- `Mpi`'s drop frees every live user op through the same routine as `Drop` before
  `MPI_Finalize`, so a `UserOp` dropped afterwards makes no MPI call.
- If `MPI_Finalize` is skipped because a window is still alive, that sweep does not run
  and the closures stay allocated until the process exits.
- Below `ThreadLevel::Serialized`, a `UserOp` dropped on a thread other than the one
  that initialised MPI aborts the process.

---

## Status

Accepted — 2026-05-17; amended 2026-09-24.
