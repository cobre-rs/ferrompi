# ADR-0004: PersistentRequest Lifecycle and Buffer-Lifetime Invariants

**Status:** Accepted — 2026-05-17; amended 2026-09-24
**Date:** 2026-05-17
**Deciders:** Rogerio Alves

---

## Context

### What persistent operations are

MPI has supported persistent point-to-point operations since MPI 1.1: `MPI_Send_init`,
`MPI_Recv_init`, and their mode variants (`MPI_Bsend_init`, `MPI_Rsend_init`,
`MPI_Ssend_init`). MPI 4.0 extended the same concept to collectives: `MPI_Bcast_init`,
`MPI_Allreduce_init`, `MPI_Gather_init`, and eleven more variants. All of these
operations share the same abstraction: initialization is decoupled from execution. The
caller invokes `*_init` once to register a communication pattern, then calls
`MPI_Start` and `MPI_Wait` on successive iterations to execute it.

The `MPI_Start` / `MPI_Wait` cycle may repeat without limit. Between two `MPI_Start`
calls the request is "inactive"; between `MPI_Start` and `MPI_Wait` it is "active".
`MPI_Request_free` deallocates the request and must only be called when the request is
inactive.

### Why persistent operations matter for iterative algorithms

HPC iterative solvers — stochastic dynamic programming, conjugate gradient,
domain-decomposition methods — execute the same collective communication pattern with
identical buffer shapes on every iteration, often thousands of times. The overhead of a
one-shot blocking collective is dominated by initialization work: argument validation,
algorithm dispatch, routing topology computation, and MPI progress-engine setup. For an
iterative algorithm these costs are paid identically on every call even though neither
the buffer shape nor the participating ranks change.

Persistent operations shift the initialization cost to a one-time `*_init` call.
Subsequent iterations pay only the marginal data-movement cost. In practice this
produces the 10–30% per-iteration speedup quoted in the ferrompi README for
representative MPI 4.0–capable runtimes.

> **Amended 2026-09-24 — Measured persistent speed-up.** See [Amendments](#amendments).

### The buffer-lifetime problem

`MPI_Send_init` and all `*_init` variants capture the buffer address at registration
time. The MPI specification requires that the buffer remain valid (not deallocated, not
moved) for the lifetime of the request — specifically, it must remain valid from
`MPI_Start` through `MPI_Wait` on every iteration for as long as the request exists.
Freeing or reallocating the buffer while the request is alive, even between two
`MPI_Start` / `MPI_Wait` cycles, is a violation of the MPI specification and produces
undefined behavior.

This constraint is unlike typical MPI send/receive buffers, which only need to outlive a
single `MPI_Wait`. A persistent request outlives many `MPI_Wait` calls; the buffer must
outlive the entire `PersistentRequest` value, not just the current active window.

In Rust, this invariant is either enforced at compile time via a lifetime parameter on
the request type, or it must be enforced by documentation and programmer discipline.
Deciding which mechanism to use is one of the two central questions this ADR answers.

### The active-state problem

MPI 4.0 §10.2 specifies that calling `MPI_Start` on an already-active request, or
calling `MPI_Request_free` on an active request, is erroneous and produces undefined
behavior at the MPI level. Neither the C API nor the Rust API can recover from this
condition through error codes alone, because by the time the error is detected, the MPI
progress engine may already have been corrupted.

The ferrompi Rust API must either prevent this condition at compile time (by making it
unrepresentable via the type system) or detect it at runtime and respond with a clear
failure before the MPI call is issued.

### Scope: persistent collectives and persistent P2P share one design

The buffer-lifetime problem and the active-state problem are identical for all
persistent operations regardless of whether the underlying MPI call is a collective or a
point-to-point send or receive. The decision made here applies uniformly to:

- All 11 collective `_init` variants (`bcast_init`,
  `allreduce_init`, `reduce_init`, `gather_init`, `scatter_init`, `allgather_init`,
  `alltoall_init`, `scan_init`, `exscan_init`, `reduce_scatter_block_init`,
  `barrier_init`).
- The four V-variants (`gatherv_init`, `scatterv_init`, `allgatherv_init`,
  `alltoallv_init`).
- In-place variants (`allreduce_init_inplace`, `gather_init_inplace`,
  `scatter_init_inplace`, `allgather_init_inplace`, `alltoall_init_inplace`).
- The five P2P `_init` variants (`send_init`, `recv_init`,
  `bsend_init`, `rsend_init`, `ssend_init`).

> **Amended 2026-09-24 — Constructor set and MPI version gates.** See [Amendments](#amendments).

This ADR therefore covers both persistent collectives and persistent P2P, which share the
lifecycle design and the `PersistentRequest` type. Splitting into two
ADRs would document the same design twice; one ADR covers both.

---

## Decision Drivers

1. **Buffer-lifetime safety.** The buffer passed to any `*_init` constructor must
   remain valid for the entire lifetime of the `PersistentRequest`. The design must
   make it as hard as possible to violate this requirement, ideally preventing violations
   at compile time or, if that is not feasible, documenting the invariant prominently in
   SAFETY comments on every constructor.

2. **Active-state enforcement.** `start()` must not be callable when the request is
   already active. If the type system cannot prevent this, a runtime check that returns
   an error (or panics) before the invalid `MPI_Start` is issued is required. The check
   must fire before the MPI call, not after.

3. **Drop safety.** Dropping a `PersistentRequest` must not leak MPI resources and
   must not invoke undefined behavior. If the request happens to be active at drop time
   (for example, because the enclosing scope panicked), the `Drop` implementation must
   handle this case safely — either by calling `MPI_Wait` before `MPI_Request_free`, or
   by documenting why the active case cannot arise.

4. **API symmetry with nonblocking collectives.** The persistent API should be
   recognizable to callers who already use ferrompi's nonblocking (`isend`, `irecv`,
   `iallreduce`) surface. In particular, `start()` and `wait()` should mirror the
   `Request::wait()` pattern so that callers moving between nonblocking and persistent
   forms can transfer their mental model with minimal friction.

5. **No overhead per `start`/`wait` beyond the MPI call.** The hot iteration loop
   runs `start()` and `wait()` thousands of times. Any bookkeeping added by the Rust
   wrapper must be bounded and fast — a single boolean read/write per call is acceptable;
   a heap allocation or a mutex acquisition is not.

---

## Options Considered

### Option A (chosen): Opaque `PersistentRequest` with handle indirection and runtime active-state check

`PersistentRequest` is a plain struct holding an `i64` slot index into the C-side
`request_table` and an `active: bool` flag:

```rust,ignore
pub struct PersistentRequest {
    handle: i64,
    active: bool,
}
```

Every `*_init` constructor allocates a request slot, stores the `MPI_Request` in
`request_table[handle]`, and returns a `PersistentRequest { handle, active: false }`.
The buffer pointer captured by MPI lives in the C side of the `request_table`; Rust
holds only the opaque integer index.

`start()` checks `self.active`, returns `Err(Error::Internal(...))` if already active,
then calls `ferrompi_start(self.handle)` and sets `self.active = true`. `wait()` calls
`ferrompi_wait(self.handle)` and sets `self.active = false`. `Drop` checks `self.active`
and calls `ferrompi_wait(self.handle)` if the request is active before calling
`ferrompi_request_free(self.handle)`.

The buffer-lifetime invariant is enforced by documentation: each constructor carries a
`// SAFETY:` comment stating that the buffer must remain valid for the lifetime of the
returned `PersistentRequest`. There is no Rust lifetime parameter tying the buffer to
the request.

**Pros:**

- Simple, self-contained type with no generics. Callers store `PersistentRequest` in any
  context without wrestling with lifetime propagation.
- The boolean active-state check is a single register read followed by a conditional
  branch — negligible overhead on the hot path.
- `Drop` handling of the active case (wait-then-free) is safe, deterministic, and matches
  the behavior of `Request` drop elsewhere in ferrompi.
- All 20+ constructor variants (collectives, V-variants, in-place, P2P) return the
  same unparameterized type, keeping the API surface uniform.
- The handle-indirection design reuses the existing `request_table` infrastructure
  established in ADR-0002 and already exercised by the nonblocking collective layer.

**Cons:**

- The buffer-lifetime invariant is not enforced by the Rust type system; a caller who
  frees the buffer while the request is alive will compile without error and produce
  undefined behavior at the MPI level at runtime.
- The active-state check is a runtime `Err` return rather than a compile-time
  prevention. Correct iterative code never triggers it, but it remains a runtime
  failure path.

### Option B (rejected): Type-state `PersistentRequest` / `ActivePersistentRequest`

Under the type-state pattern, `start` consumes the inactive variant and produces an
active variant; `wait` consumes the active variant and produces the inactive one:

```rust,ignore
impl PersistentRequest {
    pub fn start(self) -> Result<ActivePersistentRequest> { ... }
}

impl ActivePersistentRequest {
    pub fn wait(self) -> Result<PersistentRequest> { ... }
}
```

This is a defensible design used in other Rust APIs (the `typestate` crate, session
types, builder patterns). It provides compile-time proof that `start` is never called on
an already-active request, and that `wait` is always called before the next `start`.

**Why it was rejected:**

The ergonomic cost in iterative loops is substantial. Every iteration requires rebinding
two variables:

```rust,ignore
let req = req.start()?;
let req = req.wait()?;
```

This pattern forces the request to move out of any collection (`Vec<PersistentRequest>`)
on each iteration and back in after `wait`, which is incompatible with storing requests
alongside other per-rank state. It is also incompatible with `start_all` and `wait_all`
on a slice of requests — a slice cannot move its elements out one at a time and replace
them with a different type.

More importantly, the type-state pattern does not help with the buffer-lifetime problem,
which is the more dangerous invariant. A caller who stores a `PersistentRequest` longer
than the buffer does not benefit from type-state tracking of the start/wait cycle. The
key safety property that type-state would buy — preventing double-start — is also caught
by the runtime check in Option A before the MPI call is issued. In practice the runtime
check fires only on programming errors that would also be caught immediately in
development or in tests; the probability that it fires silently in production without a
prior failure during testing is low.

Given the ergonomic cost, the incompatibility with batch operations (`start_all`,
`wait_all`), and the limited safety advantage over the runtime check, the type-state
approach is not the right trade-off here.

### Option C (rejected): Lifetime-parameterized `PersistentRequest<'a, T>`

Under this option, `PersistentRequest<'a, T>` holds a `PhantomData<(&'a [T], &'a mut [T])>`
to tie the request lifetime to the buffer:

```rust,ignore
pub struct PersistentRequest<'a, T: MpiDatatype> {
    handle: i64,
    active: bool,
    _marker: std::marker::PhantomData<(&'a [T], &'a mut [T])>,
}
```

Each constructor's signature would carry the lifetime:

```rust,ignore
pub fn allreduce_init<'a, T: MpiDatatype>(
    &self,
    send: &'a [T],
    recv: &'a mut [T],
    op: ReduceOp,
) -> Result<PersistentRequest<'a, T>> { ... }
```

The compiler would then refuse to compile code where the buffer is freed while a
`PersistentRequest<'a, T>` that borrows it is still in scope. This is the strongest
possible enforcement of the buffer-lifetime invariant.

**Why it was rejected:**

The type parameters proliferate across the entire call stack. Any struct that holds a
`PersistentRequest<'a, T>` must itself become generic over `'a` and `T`. In a solver
that manages 20 persistent operations across different communicators and element types,
this produces deep lifetime nesting that the borrow checker cannot simplify, requiring
explicit lifetime annotations at every level.

More fundamentally, the buffer-ownership picture is more complex than a single `'a`
parameter captures. For in-place variants the send and receive buffers alias the same
memory, which requires `&'a mut [T]` for both, but the MPI call internally treats the
single buffer as both send and receive source. For collective variants like
`allreduce_init` there are two disjoint buffers with potentially different call-site
owners. A single `'a` parameter cannot capture this without either overly restricting the
caller or introducing multiple lifetime parameters per constructor. The resulting
signatures become harder to use than the invariant they protect is dangerous.

> **Amended 2026-09-24 — Lifetime-bound requests.** See [Amendments](#amendments).

The practical safety improvement over well-documented SAFETY comments on every
constructor is also limited: callers who read and follow the SAFETY documentation write
correct code; callers who do not will violate either the lifetime-parameterized or the
documented variant. The SAFETY-comment approach imposes zero ergonomic cost on callers
who never make the mistake.

Finally, `PersistentRequest` is the return type of 20+ constructor functions. Introducing
a generic lifetime parameter into the return type would require updating all call sites
across the codebase and would complicate the handle-table design that already exists for
nonblocking requests.

---

## Decision

**Option A is chosen.** `PersistentRequest` is an opaque handle type with no lifetime
or type parameters. The active-state invariant is enforced at runtime: `start()` returns
`Err(Error::Internal("Request is already active"))` if called on a request for which
`start()` has already been called without an intervening `wait()`. The buffer-lifetime
invariant is enforced by SAFETY documentation on each constructor.

Rationale against the decision drivers:

- **Driver 1 (buffer-lifetime safety)**: The invariant is documented in a `// SAFETY:`
  comment on every `*_init` constructor. While this does not prevent a violation at
  compile time, it makes the requirement explicit and auditable. Callers who follow the
  SAFETY contract write correct code. The type-parameterized alternative (Option C) was
  rejected because its ergonomic cost outweighs the marginal safety benefit in this
  codebase's use patterns.

- **Driver 2 (active-state enforcement)**: `start()` reads `self.active` before issuing
  any MPI call and returns `Err` immediately if the request is active. The MPI call is
  never issued on an invalid state. The runtime check satisfies the requirement that the
  error fires before `MPI_Start`, not after.

- **Driver 3 (Drop safety)**: `Drop` calls `ferrompi_wait(self.handle)` if
  `self.active` is true before calling `ferrompi_request_free(self.handle)`. This is the
  safe behavior: it waits for the in-flight operation to complete before freeing the
  request, which is the only behavior permitted by the MPI specification. The wait-then-
  free sequence matches the established drop behavior for `Request` in the nonblocking
  collective layer.

- **Driver 4 (API symmetry)**: `PersistentRequest::start()` and
  `PersistentRequest::wait()` mirror the `Request::wait()` call pattern from the
  nonblocking collective layer. Callers already familiar with `iallreduce(...).wait()` can
  directly map to `allreduce_init(...)?` / `req.start()?` / `req.wait()?`. The single-
  type design (no state-machine variants) means that callers do not need to track two
  different request types.

- **Driver 5 (no overhead per start/wait)**: Each `start()` call performs one boolean
  read, one `Err` check, one FFI call, and one boolean write. Each `wait()` call performs
  one boolean read, one FFI call, and one boolean write. There are no heap allocations,
  no locks, and no virtual dispatch on the hot path.

---

## Consequences

### The single `PersistentRequest` type covers all 20+ persistent operations

`PersistentRequest` is the return type of every constructor variant across the entire
persistent operation surface: the 11 collective `_init` functions, the 4 V-variants, the
5 in-place variants, and the 5 P2P `_init` functions (`send_init`,
`recv_init`, `bsend_init`, `rsend_init`, `ssend_init`). Because the type carries no
buffer type parameter, the same type is used uniformly regardless of whether the
underlying operation is a broadcast over `f64` or a synchronous-mode send of `i32`.

> **Amended 2026-09-24 — Constructor set and MPI version gates.** See [Amendments](#amendments).

### Persistent P2P shares `PersistentRequest` with no special casing

The five P2P variants (`send_init`, `recv_init`, `bsend_init`, `rsend_init`,
`ssend_init`) return `PersistentRequest` with the same lifecycle semantics as any
collective `_init` variant. The only difference between P2P and collective constructors
is the MPI version requirement: persistent P2P is available since MPI 1.1 (its shims have
no version gate), whereas persistent collectives require MPI 4.0
(gated via `#if MPI_VERSION >= 4`). At the Rust level both families look identical.

> **Amended 2026-09-24 — Constructor set and MPI version gates.** See [Amendments](#amendments).

### Runtime active-state check: `start()` returns `Err`, not `panic!`

Calling `start()` on a `PersistentRequest` that is already active returns
`Err(Error::Internal("Request is already active"))`. The implementation returns an error
value rather than panicking. The distinction matters: returning `Err` is recoverable in
principle and is composable with `?`-based error propagation. In correct iterative code
this error is never triggered; the check exists as a safety net. The rustdoc for
`start()` documents this explicitly.

`start_all()` performs the same check across a slice of requests: if any request in the
slice is already active it returns `Err` before issuing `MPI_Startall` for any of them.

### Drop behavior: wait before free

`Drop for PersistentRequest` calls `ferrompi_wait(self.handle)` when `self.active` is
`true`, and then calls `ferrompi_request_free(self.handle)` unconditionally. This
two-step sequence ensures that:

1. If the owning scope panics while a request is active, the drop glue calls `MPI_Wait`
   before freeing the request handle, avoiding an `MPI_Request_free`-on-active-request
   undefined-behavior violation.
2. The `MPI_Request` slot in the C-side `request_table` is always returned by
   `MPI_Request_free` before the slot is freed, preventing a slot leak.

The `ferrompi_wait` call in `Drop` ignores its return value (errors during drop are
not propagatable). This is an accepted trade-off: if `MPI_Wait` fails inside `Drop`, the
`MPI_Request_free` still runs, ensuring the slot is released. The failure is effectively
swallowed, which is consistent with Rust's convention that `Drop` must not panic.

> **Amended 2026-09-24 — Drop after finalize and on another thread.** See [Amendments](#amendments).

### Drop behavior for nonblocking `Request`

`Drop for Request` calls `ferrompi_wait(self.handle)` (which invokes `MPI_Wait`) when
`self.completed == false`. The call blocks until the peer posts the matching send or
receive operation.

**If the peer is unreachable — for example, because an error path returned early and the
peer is waiting on a different message — `MPI_Wait` will block indefinitely, deadlocking
the calling process.** This is not a bug in ferrompi; it is the expected MPI behavior
when a request is not matched.

The rationale for blocking over alternatives:

- **Blocking is preferred to leaking the handle.** Returning from `Drop` without calling
  `MPI_Wait` would leave the request slot occupied in the C-side request table, exhausting
  the table capacity after enough leaked drops.
- **Blocking is preferred to `MPI_Cancel`.** `MPI_Cancel` is unreliable for send
  operations: the MPI standard permits an implementation to complete the send anyway,
  ignoring the cancel request. A cancel-then-wait sequence would not reliably prevent the
  peer from receiving unexpected data, and the error path handling becomes more complex
  (see `src/request.rs` for the cancel comment).

The future evolution path: a `MPI_Cancel`-then-`MPI_Wait`-with-timeout approach is being
considered to make `Drop` non-blocking on error paths. This is out of scope for v0.4.1
and planned for v0.5.

> **Amended 2026-09-24 — Drop after finalize and on another thread.** See [Amendments](#amendments).

### Buffer-lifetime invariant: documented in SAFETY comments

Each `*_init` constructor carries a `// SAFETY:` comment documenting that the caller
must not modify the buffer while the request is active and that the buffer must remain
valid for the entire lifetime of the returned `PersistentRequest`. For send-style
operations (`send_init`, `bsend_init`, `rsend_init`, `ssend_init`, collective senders)
the buffer is `&[T]` — the borrow prevents concurrent mutation on the calling thread but
does not prevent the buffer's backing allocation from being freed if the slice is derived
from a `Vec` whose owner drops it while the `PersistentRequest` is still alive. For
receive-style operations (`recv_init`) and collective operations that write into a
separate receive buffer, the buffer is `&mut [T]` — the exclusive borrow prevents any
other code from observing the buffer while the reference is live, but the same lifetime
escape concern applies.

The invariant is: **the allocation backing the buffer slice must outlive the
`PersistentRequest`**. This is a rule that applies across the entire duration of the
persistent request's lifetime, including inactive periods between `start()`/`wait()`
cycles.

### V-variant buffers: counts and displacements arrays are also captured

For `gatherv_init`, `scatterv_init`, `allgatherv_init`, and `alltoallv_init`, MPI
captures pointers to both the data buffer and the `counts` and `displacements` integer
arrays at `MPI_Gatherv_init` time. The same buffer-lifetime invariant applies to all
three arrays: they must all remain valid for the lifetime of the `PersistentRequest`.
This is documented in the SAFETY comments of those constructors.

### In-place variants: single mutable borrow covers both roles

For in-place collective variants (`allreduce_init_inplace`, `gather_init_inplace`,
`scatter_init_inplace`, `allgather_init_inplace`, `alltoall_init_inplace`), a single
`&mut [T]` buffer serves as both send and receive. The exclusive mutable borrow prevents
any other code from reading or writing the buffer while the reference is in scope, which
is the strongest local protection possible. The constructor passes `MPI_IN_PLACE` as the
send buffer in the underlying C shim; the data buffer pointer is captured once. The
same lifetime rules apply.

### Persistent collectives require MPI 4.0; persistent P2P requires MPI 1.1

Collective `_init` constructors are compiled with `#if MPI_VERSION >= 4` in the C
shims. On an MPI older than 4.0 the shims return a ferrompi sentinel that surfaces as
`Err(Error::NotSupported(_))` naming the operation. The persistent P2P shims have no
version gate, and no runtime version probe is required.

> **Amended 2026-09-24 — Constructor set and MPI version gates.** See [Amendments](#amendments).
> **Amended 2026-09-29 — Open MPI 5 capability gate.** See [Amendments](#amendments).

---

## Alternatives Considered

### Closure-based API

A session-style API of the form:

```rust,ignore
world.persistent(|s| {
    let req = s.bcast_init(&mut data, 0)?;
    req.start()?;
    req.wait()?;
    Ok(())
})?;
```

would scope the request lifetime to the closure, making it structurally impossible to
hold a `PersistentRequest` longer than the closure's `&mut data` borrow. This is
appealing in principle but does not capture buffer lifetime correctly: the closure
receives `&mut data` by value capture, so `data` must already be borrowed for the
closure's duration by the outer scope — the scoping enforcement is pushed to the closure
boundary rather than to the request type. It also prevents the caller from interleaving
other work between `start()` and `wait()`, which is the primary use case for persistent
operations in iterative solvers (post-start computation while communication is in flight).

### Macro-generated typed constructors per variant

Generating a distinct newtype wrapper for each `*_init` variant
(`BcastRequest`, `AllreduceRequest<T>`, `GatherRequest<T>`, etc.) would allow each
type to carry the precise borrow signature of its buffer arguments. The cost is an
explosion in API surface: 20+ types, each with its own `start`, `wait`, `Drop`, and
`start_all` / `wait_all` integration. The implementation complexity scales linearly with
the number of collective variants, and any future addition to the persistent operation
surface (e.g., `reduce_scatter_init` in a future MPI version) requires a new type.
A single `PersistentRequest` that serves all variants is strictly preferable.

---

## Amendments

### 2026-09-24 — Lifetime-bound requests

Option C's second argument does not hold. One lifetime parameter can bound several
borrows in one signature (`&'a [T]` and `&'a mut [T]`), as `Win<'a, T>` bounds
`Win::create`'s buffer, and `'a` limits how long the borrows last, not what they point
to. A lifetime alone would still not make the API sound, because `std::mem::forget` on a
request ends the borrow while MPI keeps using the buffer. The buffer-safety model of this
ADR is superseded in 0.7.0 by a new ADR on buffer ownership.

### 2026-09-24 — Drop after finalize and on another thread

- Every `Drop` described here first calls ferrompi's lifecycle guard.
- After the `Mpi` handle is dropped, `Drop` for `Request` and `PersistentRequest` makes
  no MPI call, including when `MPI_Finalize` was skipped because a window was still
  alive.
- Before `MPI_Finalize`, `Mpi`'s drop frees every inactive persistent request and leaves
  active requests to MPI; debug builds print their count.
- Below `ThreadLevel::Serialized`, a request dropped on a thread other than the one that
  initialised MPI aborts the process with a message naming the type and the thread,
  since it can neither call MPI there nor return an error.
- The cancel-with-timeout change considered for 0.5 was not made: `Drop` still waits.

### 2026-09-24 — Constructor set and MPI version gates

- There are ten persistent collective constructors; the persistent barrier constructor
  listed in the scope was never added.
- On an MPI older than 4.0 the collective constructors return `Err(Error::NotSupported(_))`
  naming the operation. The paragraph that named a nonexistent error class was corrected
  in place.
- The persistent point-to-point shims have no version gate.

### 2026-09-24 — Measured persistent speed-up

The 10–30% per-iteration figure above was not measured. On MPICH at 2 ranks, a
persistent allreduce was 20–58 % faster per call than `iallreduce` up to 4 KiB, at most
8 % faster at 32 KiB, and no faster from 256 KiB.

### 2026-09-29 — Open MPI 5 capability gate

The collective constructors are compiled when `mpi.h` reports `MPI_VERSION >= 4` or the
library is Open MPI 5 or later, which implements them while reporting MPI 3.1. Their
large-count forms stay gated on `MPI_VERSION >= 4`. Elsewhere the constructors return
`Err(Error::NotSupported(_))` as before.

---

## Status

Accepted — 2026-05-17; amended 2026-09-24, 2026-09-29. The implementation lives in `src/persistent.rs` (the `PersistentRequest`
type and its `Drop` impl) and `src/comm/persistent.rs` (all `*_init` constructor methods
on `Communicator`).
