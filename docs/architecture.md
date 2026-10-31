# ferrompi Architecture

## Overview

This document is the canonical reference for ferrompi's internal architecture.
Its audience is contributors investigating implementation details, not end users
seeking usage guidance (which lives in `README.md`). It explains why a
hand-written C wrapper exists, how handle tables are organised and made
thread-safe, which invariants must be preserved when adding new MPI entry
points, and how the sealed-trait type system enforces datatype safety across
the Rust/C boundary. The decisions behind this layout are recorded as ADRs,
indexed in `docs/README.md`.

## Layer Diagram

ferrompi sits between application Rust code and an underlying MPI runtime
through a deliberate six-layer stack. Each layer has a clearly bounded
responsibility; crossing a layer boundary requires following the invariants
described in the sections below.

```mermaid
graph TD
    A["Rust application code\n(user crate, examples/)"]
    B["Public ferrompi API\nsrc/ (all modules except ffi.rs)"]
    C["FFI declarations\n(src/ffi.rs — extern \"C\" blocks)"]
    D["C wrapper layer\n(csrc/ferrompi.c\ncsrc/ferrompi.h)"]
    E["MPI implementation\n(MPICH / Open MPI / Cray MPT)"]
    F["MPI runtime\n(process manager, network fabric, RDMA HW)"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

| Layer                    | Responsibility                                                                                                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Rust application         | Calls ferrompi's public API; owns all live data.                                                               |
| Public ferrompi API      | Type safety, RAII drop semantics, `Result` error mapping, sealed-trait enforcement.                            |
| `src/ffi.rs`             | Extern declarations; `guarded_extern!` wraps most of them with the lifecycle guard in `src/rt.rs` — see "Lifecycle guard" for what it leaves out and why. |
| `csrc/ferrompi.c` / `.h` | Handle tables, large-count branching, `MPI_UNDEFINED` normalisation, op trampolines, runtime constant queries. |
| MPI implementation       | Collective algorithms, point-to-point transport, window coherence.                                             |
| MPI runtime              | Process launch, rank assignment, network fabric, memory registration.                                          |

The split between the C layer and the public Rust API is deliberate and
explained in full in `adr/0001-why-c-wrapper.md`. The short answer: several MPI
idioms (opaque handle tables, `_c` large-count branching, implementation-defined
constant values) require C11 constructs that cannot be expressed portably in
Rust FFI without replicating the C layer anyway.

## Handle Tables

The C layer in `csrc/ferrompi.c` owns six fixed-size static tables that map
compact integer handles — passed over the FFI boundary as `int32_t` or
`int64_t` — to the opaque MPI handle types that the MPI library returns from
its constructor functions. MPI handles (`MPI_Comm`, `MPI_Request`, etc.) are
not stable integers; their values are implementation-defined and may change
across library versions. Storing them in a table and exposing only the table
index to Rust insulates the Rust layer from this instability.

The `comm_table`, `win_table`, `group_table`, `datatype_table` and `op_table`
each use an `atomic_int` used-flag per slot, claimed by a compare-and-swap in
the matching `alloc_*` function. The `request_table` instead tracks occupancy
with a bitmap of 64-bit words (see "Request table" below). Slot 0 of
`comm_table` is reserved for `MPI_COMM_WORLD` and excluded from allocation.
Handle 0 of `group_table` stands for `MPI_GROUP_EMPTY`: no slot holds it, and
allocation never returns 0.

Five of the six tables have both allocation and free helpers
(`alloc_*`/`free_*`, `free_datatype_slot` and `free_op_slot` for the last
two) that scan from a cached hint and reset a freed slot to its null sentinel
(`group_table` to `MPI_GROUP_NULL`); `comm_table` has only `alloc_comm`, and
its free is inlined in `ferrompi_comm_free`.

### Request table

The `request_table` is by far the most active table: every non-blocking send,
receive, and collective operation allocates a slot. Its occupancy is a bitmap
of 64-bit words (`request_bits`) rather than one atomic flag per slot:
allocation finds a free bit within a word with a hardware find-first-zero
instead of probing each slot individually.

`alloc_request` claims a free bit with an `acq_rel` `fetch_or`. `free_request`
clears it with a `release` `fetch_and`. `request_slot` resolves a handle back
to a slot with an `acquire` load of the same bitmap word — the acquire/release
pairing ensures a thread that observes a bit set also observes the
`request_table` entry the allocating thread wrote.

A request handle is `(generation << 32) | slot`: each slot also carries a
31-bit generation counter that `free_request` bumps every time the slot is
freed. A handle kept past its request's free therefore fails the generation
check in `request_slot` instead of resolving to whichever request now occupies
the reused slot.

Each slot also carries a persistent/active state byte (`request_state`)
marking whether the slot holds a registered persistent request and, if so,
whether it is currently active. The finalize sweep in `ferrompi_finalize`
reads this state to free any inactive persistent request still registered
when the `Mpi` handle is dropped.

The full analysis, including rejection of a pthread-mutex approach and a
lock-free Treiber stack, is in `adr/0002-handle-tables.md`.

### Drop guards

`Group`'s `Drop` calls `ferrompi_group_free` only when `handle > 0`, so
handle 0 (`MPI_GROUP_EMPTY`) is never freed. `CustomDatatype`'s `Drop` calls
`ferrompi_type_free` only when `handle >= 0`. `OpRegistration`'s `Drop`
(the last holder of a `UserOp`'s op) carries no handle guard — a registration
always holds a valid slot — and runs `ferrompi_op_free` unconditionally once
the lifecycle guard (`rt::drop_guard`, see "Lifecycle guard" below) permits
it; see "Drop ordering for `UserOp`".

## Thread-Safety Model

`ThreadLevel` names MPI's four thread levels; its rustdoc states what each one
allows.

`ThreadLevel` is selected at init time via `Mpi::init_thread(level)`. The
requested level is advisory — the MPI implementation may grant a lower level —
and the level ferrompi actually enforces is the one it granted:
`Mpi::thread_level()` reports it, and the lifecycle guard in `src/rt.rs`
enforces it on every guarded call.

Under `ThreadLevel::Single`/`ThreadLevel::Funneled`, only the thread that
called `Mpi::init_thread` may call MPI. A guarded call from any other thread
returns `Err(Error::ThreadLevelViolation)` without calling MPI, and an
MPI-calling `Drop` on another thread aborts the process instead (see
"Lifecycle guard"). At `ThreadLevel::Serialized`, the caller must serialise
its own MPI calls; debug builds flag two such calls overlapping. At
`ThreadLevel::Multiple`, any thread may call MPI concurrently.

### `Communicator` and `Group` are `Send + Sync`

`Communicator` and `Group` both implement `Send` and `Sync`. Each wraps an
integer index into a handle table, which carries no thread affinity, so
handing either type to another thread is always safe — it is the lifecycle
guard above, not this impl, that decides which thread may actually call MPI
through it.

### Synchronisation in the C layer

The per-table atomics described in "Handle Tables" are the main
synchronisation primitive in the C wrapper layer. The one mutex is
`group_mutex`, which serialises the local MPI group calls that can write an
object two threads share. The `GROUP_CALL` macro takes it only when MPI
granted `Serialized` or `Multiple` (below that the Rust guard admits one
thread), holds it across one local group call at a time, never across a
collective or blocking call, and no other lock is taken in the file. The
calls that create a communicator from a group are collective, so they run
outside it. The C layer adds no condition variables or barriers. At
`ThreadLevel::Multiple`, every other correctness guarantee comes from the MPI
implementation's own thread-safety contract: MPI guarantees that concurrent
calls to its own functions on the same communicator are safe once it reports
`MPI_THREAD_MULTIPLE`.

### Lifecycle guard

`src/rt.rs` holds the process-wide lifecycle state (`STATE: AtomicU8`),
`Uninit -> Initializing -> Active(level) -> Finalizing -> Finalized`, and a
thread-local flag recording which thread called `Mpi::init_thread`; the
module doc of `src/rt.rs` is the reference for the states. This section
describes the mechanism. What a caller sees, and must do, is stated in the
`# Lifecycle` section of `Mpi`'s rustdoc.

The `guarded_extern!` macro in `src/ffi.rs` wraps every extern that returns
`c_int`, except calls with no init/finalize precondition (the init call
itself, lifecycle and version queries, error-message lookup), op-table
bookkeeping that makes no MPI call, and calls made only from a `Drop` impl.
Anything whose return type isn't `c_int` — `ferrompi_wtime`'s `c_double`,
the RMA mode-value getters' `void` — sits outside the macro regardless of
whether it calls MPI. A wrapped call goes through `rt::enter` before
forwarding, and holds the `InFlight` token it returns until the call
returns. `rt::enter` rejects the call, without touching MPI, in `Uninit`; in
`Initializing` on every thread but the initializing one; in `Finalizing` and
`Finalized`; and at `Single`/`Funneled` on any thread but the init thread. At
`Serialized`/`Multiple` the token is counted in a per-thread shard of
`IN_FLIGHT`; below that it is empty. The externs after the macro's trailing
marker complete, test or cancel a pending request, receive a matched message
or close an open epoch; they take `rt::enter_completion`, which also admits
`Finalizing`.

Every MPI-calling `Drop` impl calls `rt::drop_guard` in `rt::enter`'s place,
since a `Drop` cannot return `Err`: it skips the MPI call silently once
finalizing, and aborts the process on a wrong-thread drop below
`Serialized`. A `Request` has no `Drop`: its scope completes it, and the
scope holds a `ScopeToken` for its whole extent. An active
`PersistentRequest` and a matched, unreceived `Message` each hold a
`HandleToken`.

`Mpi::drop` finalizes in two steps. `rt::begin_finalize` stores `Finalizing`,
which refuses new calls, frees and scope opens, then waits for the scopes
open on other threads and, at `Serialized`/`Multiple`, for every counted call
to return (the first drain). `finalize_skip_reason` in `src/lib.rs` then
decides whether `MPI_Finalize` must be skipped: for a scope open on the
dropping thread (`rt::scopes_on_this_thread`), a live `HandleToken`
(`rt::live_handle_tokens`, which finalizing never waits for, because the
thread that holds one may itself be waiting for the finalizer), or a live RMA
window (`LIVE_WINDOWS` in `src/window.rs`). On a skip the state stays
`Finalizing` for the rest of the process and MPI stays initialized, with a
stderr warning. Otherwise `rt::end_finalize` stores `Finalized`, waits again
(the second drain) for the completion calls admitted in between, and
`ferrompi_finalize` runs. Because the state is already `Finalized`, a handle
whose drop is nested inside the finalize sweep — a request or op the sweep
itself frees — makes no MPI call. `ferrompi_finalize` frees every inactive
persistent request, live user op, group, datatype and communicator still
registered (slot 0, `MPI_COMM_WORLD`, is skipped).

### `UserOp` closure thread-safety contract

`UserOp<T>` wraps a user-supplied Rust closure that MPI invokes during
reduction operations. The closure lives in the Rust-side registry
`REGISTRY: [AtomicPtr<ByteClosure>; MAX_OPS]` in `src/op.rs`, as a thin
pointer to a boxed, type-erased closure. It is published with a release
store before `MPI_Op_create` is called, and the trampoline loads it with an
acquire load. Each of the 16 C trampoline functions,
`ferrompi_user_op_trampoline_0` through `ferrompi_user_op_trampoline_15`,
passes only its own baked-in slot number to `rust_user_op_invoke`, the
`extern "C"` entry point that performs that acquire load — so it may be
called from any thread under `MPI_THREAD_MULTIPLE`, including an
MPI-internal thread-pool thread the application did not create.

The closure must satisfy `F: Fn(&[T], &mut [T]) + Send + Sync + 'static`. All
three bounds are mandatory and enforced at compile time (see
`adr/0005-mpi-op-create.md` Decision 2):

- `Send` — the closure is moved into the static registry, accessible from any thread.
- `Sync` — concurrent invocations of the same `MPI_Op` (e.g., two concurrent `allreduce` calls on different communicators) must not produce data races on the closure's captured state.
- `'static` — the closure is held until the op is freed, which may be much later than the call that created it; any borrow shorter than `'static` could be invalidated while the `MPI_Op` handle is still live.

Captures using `Rc<T>`, `Cell<T>`, `RefCell<T>`, raw pointers, or non-`'static`
references are rejected at compile time. Callers requiring shared mutable
closure state must use `Arc<Mutex<T>>` or `Arc<RwLock<T>>`.

### Drop ordering for `UserOp`

`UserOp<T>` holds an `Arc<OpRegistration>`. The registration drops when its
last holder does: the `UserOp` handles plus every persistent request
initialised with the op. Its `Drop` runs `ferrompi_op_free`, guarded only by
`rt::drop_guard`. This ordering is mandatory: `ferrompi_op_free` runs
`MPI_Op_free` first — MPI-4.1 §2.5.1 says it marks the op for deallocation
and lets pending operations that use it complete normally, which is why the
registration waits for its last holder — then drops the Rust-boxed closure,
then reclaims the slot. Dropping the closure before freeing the op would be
a use-after-free if an operation were still dispatching the trampoline. The
finalize sweep in `ferrompi_finalize` frees any op still live when `Mpi` is
dropped through this same `ferrompi_op_free` path, preserving the ordering
there too. The full rationale is in `adr/0005-mpi-op-create.md` Decision 3,
the handle-lifetime model in `adr/0007-buffer-safety-and-handle-lifetimes.md`,
and the caller's rules in `UserOp`'s `# Dropping` rustdoc.

## C Layer Scope

The list below distinguishes what belongs in C from what belongs in Rust.

### What goes in C (`csrc/ferrompi.c`)

- **Handle tables** — `comm_table`, `request_table`, `win_table`,
  `group_table`, `datatype_table`, `op_table` (see "Handle Tables"); MPI
  opaque handles cannot be stored in Rust without copying the allocation
  strategy.
- **Large-count branching** — every scalar-count shim (e.g. `ferrompi_send`)
  uses the classic MPI call when the count fits `int`. Above `INT_MAX`, it
  uses the `_c` variant, which accepts `MPI_Count`, when `MPI_VERSION >= 4`,
  and returns `MPI_ERR_COUNT` otherwise. The v-collective shims (`gatherv`,
  `scatterv`, `allgatherv`, `alltoallv`, and their nonblocking/persistent
  forms) return `MPI_ERR_COUNT` above `INT_MAX` on every MPI version, since
  their count arrays stay `int32_t`. Every reduction shim returns
  `MPI_ERR_COUNT` above `INT_MAX` for a user op code, because a user op is a
  classic `MPI_User_function` with an `int` length. This branching requires C
  preprocessor guards that would be unreadable as inline Rust assembly or
  build-script code generation.
- **`install_errors_return`** — called on each newly-created communicator
  handle to set `MPI_ERRORS_RETURN`, converting MPI aborts into
  `Err(Error::Mpi { .. })`; `ferrompi_comm_create_from_group` instead passes
  `MPI_ERRORS_RETURN` to MPI directly.
- **Op trampolines** — 16 distinct C functions, `ferrompi_user_op_trampoline_0`
  through `ferrompi_user_op_trampoline_15`, generated by a preprocessor macro
  (see "`UserOp` closure thread-safety contract"; rationale in
  `adr/0005-mpi-op-create.md` Decision 5).
- **Runtime implementation-defined constants** — `MPI_MODE_NOSTORE`,
  `MPI_MODE_NOPUT`, `MPI_MODE_NOPRECEDE`, `MPI_MODE_NOSUCCEED`, and the
  analogous PSCW asserts (see "`MPI_MODE_*` constants queried at runtime").
- **MPI-4 capability gating** — `FERROMPI_HAVE_MPI4_COLLECTIVES` is defined
  at compile time from `mpi.h`'s version macros; when unset, the
  MPI-4-only persistent-collective and `comm_create_from_group` shims
  return `FERROMPI_ERR_NOT_SUPPORTED` instead of calling MPI.
- **Error-class index mapping** — `ferrompi_error_class_index` compares an
  `MPI_ERR_*` value against the linked library's own constants and returns a
  stable index in `MpiErrorClass`'s declaration order.
- **Batch-completion bookkeeping** — the batch shims track per-request
  completion in a `done[]` array. `ferrompi_waitall` reports the first
  failing request through an out-parameter; `ferrompi_waitany`/
  `ferrompi_testany` report the completed `index`; `ferrompi_waitsome`/
  `ferrompi_testsome` fill one `ferrompi_status` per completed request, with
  `error` set to that request's MPI error class.
- **Window zeroing** — `zero_own_segment` zeroes each rank's own segment of a
  freshly allocated RMA window before returning the handle to Rust.
- **ULFM abort on a pending failure** — `abort_if_pending_after_failure`
  aborts the process, on an MPI built with ULFM's `PROC_FAILED_PENDING`
  error class, if a completion path reports it (a wildcard receive left
  pending by a process failure), rather than return while MPI still owns
  the buffer.

### What stays in Rust (`src/`)

- **Type safety** — the sealed-trait families `MpiDatatype`,
  `AtomicMpiDatatype`, `MpiIndexedDatatype`, and `BytePermutable` (see
  "Generic-over-`MpiDatatype` Design") ensure that only valid Rust types reach
  MPI entry points. The C layer accepts raw integers and cannot enforce this.
- **RAII drop semantics** — `Communicator`, `Group`, `PersistentRequest`,
  `Message`, `CustomDatatype`, `Win` and `UserOp` (through its
  `Arc<OpRegistration>`) implement `Drop`, which releases the underlying
  resource. A `Request` has no `Drop`: the scope that created it completes it
  before returning. `PersistentRequest`, `Message`, `Win` and `UserOp` list
  their drop outcomes in a `# Dropping` section of their rustdoc; the other
  three free their handle behind `rt::drop_guard` (see "Drop guards"). See
  "Lifecycle guard" above for how a drop behaves relative to `Mpi`'s own
  lifetime, and `adr/0007-buffer-safety-and-handle-lifetimes.md` for the
  ownership model.
- **`Error` mapping** — `Error::check_with_op`/`Error::from_code_with_op`
  map an FFI return code to the matching `Error` variant: `Error::Mpi {
  class, code, message, operation }` for a genuine MPI failure, and the
  appropriate non-`Mpi` variant (`Error::Finalized`,
  `Error::ThreadLevelViolation`, `Error::NotSupported`,
  `Error::ResourceExhausted`) for a Rust- or C-layer sentinel code (see
  "`Error::check_with_op` and operation tags" for the `<tag>` rule, its
  named exception, and why both are crate-private).
- **`catch_unwind + abort` panic fence** — the `rust_user_op_invoke`
  `extern "C"` entry point in `src/op.rs` wraps every closure invocation in
  `std::panic::catch_unwind`. If the closure panics, the process aborts
  immediately. Panicking across the FFI boundary is undefined behaviour; silent
  data corruption in a collective result is worse than a loud process abort for
  HPC use cases. See `adr/0005-mpi-op-create.md` Decision 6.
- **Epochs are closures** — `Win::fence` and `Win::start` run a closure that
  receives an `ActiveEpoch`; `Win::lock` and `Win::lock_all` run one that
  receives a `PassiveEpoch`. The handle carries the transfers (`put`, `get`,
  `accumulate`, `get_accumulate`, `fetch_and_op`, `compare_and_swap`) and, on
  `PassiveEpoch`, the request-based transfers (`rput`, `rget`, `raccumulate`),
  the flushes (`flush`, `flush_all`, `flush_local`, `flush_local_all`) and
  `sync`. The closing call runs when the closure returns, so a safe caller
  cannot leave an epoch open. A panic in a `fence` or `start` closure aborts the
  process, because the closing call is collective or synchronizing; one in a
  `lock` or `lock_all` closure completes the pending requests, unlocks, and
  keeps unwinding. See `adr/0007-buffer-safety-and-handle-lifetimes.md`
  (Decisions 4 and 5); the views the handles return are in
  `adr/0008-window-memory-access-model.md`.

## FFI / ABI Invariants

The following invariants must be preserved by every new MPI entry point added
to ferrompi.

### `#[repr(i32)]` enums with explicit discriminants

`ReduceOp`, `ThreadLevel`, `SplitType` and `DatatypeTag` carry `#[repr(i32)]`
with explicit `= N` discriminants and cross the FFI boundary raw (cast with
`op as i32` in `Builtin::code`, `tag_of::<T>() as i32`, and so on).
`DatatypeTag` is crate-private: it has no public path, and the crate reads it
through `tag_of::<T>()`. The discriminant values are an
internal contract between the Rust enums and the C shim; any release may
change them. A few other cross-FFI integer contracts rely on a sync
comment: the sentinel codes carry one on both sides; the leaked-window
marker, op-slot count and error-class order on one side only.

### Unconditional C switch

Every case in the C switches that decode op and datatype tags is
unconditional, except `FERROMPI_LONG_INT` and `FERROMPI_LONG_DOUBLE_INT`,
which compile only when `FERROMPI_LONG_PAIRS_VERIFIED` is defined — the same
target predicate (Linux on `x86_64`, `aarch64`, or little-endian
`powerpc64`) that gates the Rust `LongInt`/`LongDoubleInt` types, backed by a
`_Static_assert` on the pair layout. The codes only the `rma` calls send,
12 (`AccumulateOp::REPLACE`, `FetchOp::REPLACE`) and 13 (`FetchOp::NO_OP`),
are unconditional too: the Rust layer may legally produce any tag value an
enum can represent, including codes only sent when a Cargo feature is
enabled, so the C layer must accept all of them.

### `MPI_UNDEFINED` normalised to `-1`

The MPI standard does not fix the value of `MPI_UNDEFINED`, so every shim
that can return it maps it to `-1` before returning to Rust (e.g.
`ferrompi_group_translate_ranks`), and a shim that takes `-1` where MPI takes
`MPI_UNDEFINED` (`ferrompi_comm_split`'s color) maps it the other way. The Rust
layer maps `-1` to `None` uniformly, regardless of what value the linked MPI
library defines: `Status::count`, `Group::translate_ranks`, and
`Communicator::split(None, ..)`.

### Entry-point pattern

Every new MPI entry point follows this pattern in order:

1. C declaration in `csrc/ferrompi.h`.
2. C implementation in `csrc/ferrompi.c`, following the large-count policy
   above and installing `MPI_ERRORS_RETURN` on any communicator it creates
   (directly, or via `install_errors_return`).
3. `extern "C"` declaration inside the `guarded_extern!` block in
   `src/ffi.rs`, unless it falls into one of the categories "Lifecycle
   guard" names.
4. Safe Rust wrapper in the appropriate `src/` module: collective arguments
   are checked by the validator functions (`check_rank_slots`,
   `check_same_len`, `rank_block` in `src/comm/mod.rs`; `check_v_args` in
   `src/comm/v_collective.rs`), and the FFI return code is checked with
   `Error::check_with_op`.
5. An `examples/test_*.rs` integration example carrying a `// mpi-test:`
   directive, which `tests/run_mpi_tests.sh` discovers automatically; an
   `rma`-only example also needs a `[[example]]` entry in `Cargo.toml` with
   `required-features = ["rma"]`.

Some `examples/test_*.rs` files declare shim externs directly instead of
going through `src/ffi.rs`; grep `examples/` for `extern "C"` when changing a
shim signature.

Omitting any layer is a scope violation.

### `Error::check_with_op` and operation tags

`Error::check_with_op` and `Error::from_code_with_op` are `pub(crate)`: only
the crate's own wrappers call them, and `Error::check` no longer exists. A new
entry point maps its FFI return code through `Error::check_with_op` (or
`Error::from_code_with_op` directly, when the non-zero case is already
known). `<tag>` is the C shim name with the `ferrompi_` prefix stripped
(e.g. `"comm_dup"` for `ferrompi_comm_dup`), except where a Rust
method shares its shim with a sibling method on the same type — the
in-place collective variants, `allreduce_indexed`, and `allreduce_bytes` all
call the same shim as their plain collective — where the tag is the Rust
method name instead. `UserOp`'s constructors are a known exception: every
failure path uses `"op_create"`, and a full op table is reported as
`Error::ResourceExhausted` with resource `Operation`.

### `install_errors_return` on comm-creating shims

Every shim that creates a new communicator (or may return one from MPI)
calls `install_errors_return(newcomm)` before returning the handle to Rust,
setting `MPI_ERRORS_RETURN` as the error handler so MPI aborts become
return codes. `ferrompi_comm_create_from_group` is the one exception: it
passes `MPI_ERRORS_RETURN` directly to `MPI_Comm_create_from_group` instead.
Windows get the same treatment through `install_errors_return_win`, which is
best effort — the handler stays `MPI_ERRORS_ARE_FATAL` on Open MPI 4 when
the call is rejected on a fresh window.

### `MPI_MODE_*` constants queried at runtime

RMA fence and PSCW assert constants (`MPI_MODE_NOSTORE`, `MPI_MODE_NOPUT`,
`MPI_MODE_NOPRECEDE`, `MPI_MODE_NOSUCCEED`, and their PSCW equivalents) are
never hardcoded in Rust. They are queried once via C shims at first use and
cached via `OnceLock<[i32; N]>` in `src/window.rs`. This is required because
their values are implementation-defined.

## Generic-over-`MpiDatatype` Design

All ferrompi communication APIs are generic over the element type `T`. Four
sealed-trait families in `src/datatype.rs` define which Rust types are valid
for which MPI operations.

| Trait                 | Sealed module         | Types                                                                                        | Operations                                                         |
| ---------------------- | --------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `MpiDatatype`          | `mod sealed`           | `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, `u64`                                                 | All point-to-point, collectives, RMA                                  |
| `AtomicMpiDatatype`    | `mod sealed_atomic`    | `i32`, `i64`, `u32`, `u64`, `u8`                                                               | `compare_and_swap` on epoch handles only (feature `rma`)              |
| `MpiIndexedDatatype`   | `mod sealed_indexed`   | `FloatInt`, `DoubleInt`, `Int2`, `ShortInt`, and (Linux `x86_64`/`aarch64`/little-endian `powerpc64` only) `LongInt`/`LongDoubleInt` | `allreduce_indexed` (MaxLoc/MinLoc)                                   |
| `BytePermutable`       | `mod sealed_byte`      | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `[T; N]`                                 | `allreduce_bytes` (bitwise ops only)                                  |

Each trait uses a private `mod sealed` module containing a `Sealed` marker
trait. External crates cannot implement `sealed::Sealed` and therefore cannot
add new `MpiDatatype` (or any other) implementations. This prevents misuse at
compile time rather than at runtime.

`AtomicMpiDatatype` bounds only `compare_and_swap` on `ActiveEpoch` and
`PassiveEpoch`; their `fetch_and_op` and `accumulate` accept any
`MpiDatatype`, including `f32`/`f64`. MPI defines `MPI_Compare_and_swap` only
for integer, logical and byte datatypes — admitting a floating-point type
there would silently produce undefined behaviour on most MPI implementations —
while `MPI_Fetch_and_op` carries no such restriction. A `compile_fail` doctest
on `AtomicMpiDatatype` in `src/datatype.rs` verifies that `compare_and_swap`
is not found on an `f64` epoch handle.

`unsafe trait PlainData` is an open trait: it is blanket-implemented for every
`MpiDatatype` and for `[T; N]` of any `PlainData` element, and it bounds the
element type of the custom-datatype point-to-point methods (`send_custom`,
`recv_custom`, `isend_custom`, `irecv_custom`).

`MpiDatatype` and `MpiIndexedDatatype` each carry a sealed `tag` method, whose
argument is the private `sealed::Token`, so no bound outside the crate can read
it. The crate reads the tag with `tag_of::<T>()` (`pair_tag_of::<T>()` for a
pair type), letting C identify the element type via the `#[repr(i32)]`
discriminant. `AtomicMpiDatatype`/`BytePermutable` are marker traits with no
tag: the former reuses `MpiDatatype`'s tag through `tag_of::<T>()`;
`allreduce_bytes<T: BytePermutable>` always passes the fixed
`DatatypeTag::Byte` instead.

The authoritative design record for this trait family is
`adr/0003-generic-mpi-datatype.md`.

## Error Handling Model

All fallible ferrompi operations return `Result<T, Error>`. `Error` is
`#[non_exhaustive]`; its variants and their fields are documented on the
`Error` item rustdoc, which is the canonical home for that detail — this file
describes the mechanism, not the variant list.

### `Error::check_with_op` pattern

```rust,ignore
Error::check_with_op(ret, "allreduce")?;
```

This is the standard idiom for FFI calls. It returns `Ok(())` when
`ret == MPI_SUCCESS` and constructs `Err(Error::Mpi { operation: Some("allreduce"), .. })`
for a genuine MPI failure — or the matching non-`Mpi` variant for a Rust- or
C-layer sentinel code (see "Lifecycle guard" for `Error::Finalized` and
`Error::ThreadLevelViolation`, and "`Error::check_with_op` and operation
tags" for the `<tag>` rule).

### `thiserror`-derived `Display`

`Error` and `MpiErrorClass` derive `thiserror::Error`. The `[dependencies]`
table in `Cargo.toml` lists `thiserror = "2"`.
