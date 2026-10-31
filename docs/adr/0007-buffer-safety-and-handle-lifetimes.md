# ADR-0007: Buffer Safety and Handle Lifetimes

**Status:** Accepted — 2026-10-29
**Date:** 2026-10-29
**Deciders:** Rogerio Alves
**Supersedes:** the buffer-lifetime invariants of [ADR-0004](0004-persistent-collective-approach.md) and its argument against lifetime-bound requests

## Context

MPI reads and writes the memory a nonblocking call names until the request completes
(MPI-4.1 §3.7). Rust's borrow checker knows nothing about that. In 0.6 safe code could
leave a buffer in use by MPI in four ways:

- forget a nonblocking `Request`: `std::mem::forget` ends the buffer's borrow while MPI
  keeps writing to the buffer;
- borrow a persistent request's buffer for a lifetime that `std::mem::forget` could end;
- borrow a window's buffer through `Win<'a, T>`;
- forget an RMA lock guard, so the closing `MPI_Win_unlock` never ran.

A borrow alone cannot make a nonblocking buffer sound, because a borrow ends when the
borrowing value is forgotten, not when MPI stops using the memory. Soundness needs an
owner that outlives the operation, or a scope that cannot be left before the operation
completes.

[ADR-0004](0004-persistent-collective-approach.md) rejected lifetime-bound persistent
requests (its Option C) with two arguments: the lifetime parameter spreads through every
type that holds a request, and a single lifetime cannot describe a collective with
separate send and receive buffers. The second argument does not hold. One lifetime can
bound several borrows in one signature (`&'a [T]` and `&'a mut [T]`), and a lifetime
limits how long a borrow lasts, not what it points to. The first argument concerns
handles that live as long as the program, which is a real cost, but not one that a
scope-shaped borrow has.

The 0.6 audit proposed a lifetime parameter on every handle to tie it to `Mpi`. A
long-lived struct that owns `Mpi` together with its communicators and windows compiles
only if handles carry no borrow of `Mpi`, and that struct is the shape of a solver
(`examples/sddp_pattern.rs`).

## Decision

1. **Nonblocking operations run inside a scope.** `ferrompi::scope(|s| ...)` is shaped
   like `std::thread::scope`: `Scope<'s, 'env>` is invariant in its lifetimes. The
   constructors stay `Communicator` methods, take `s` first and borrow their buffers for
   `'s`. `Request<'s>` has no `Drop` and is not `Send`. `Request::wait` returns `Status`
   and `Request::test` returns `Option<Status>`. The scope returns, or continues
   unwinding, only when no request is pending: its end runs `MPI_Waitall` on every live
   request, and waits again on the requests MPI reports as `MPI_ERR_PENDING`. If it
   cannot establish that every request completed, it prints a message and aborts the
   process, because the `'env` borrows cannot be leaked. The guarantee: a buffer lent to
   a nonblocking call stays borrowed until its request completes, and forgetting a
   `Request` does not end the borrow early because the scope, not the request, completes
   it.

2. **A persistent request owns its buffers.** `PersistentRequest<T>` holds one or two
   `Vec<T>` and gives access to them only while it is inactive: the accessors return
   `Error::InvalidState` while it is active. `into_buffer` returns the buffer, waiting
   first if the request is active; a `PersistentRequest` dropped while active waits
   before `MPI_Request_free`. It holds an in-flight token from `start` to completion
   and stays `Send`. When it is dropped after a skipped `MPI_Finalize`, it leaks the
   request handle, and while active its buffers too. `start_all` and `wait_all` take
   requests of one `T`. The guarantee: the memory MPI uses cannot be freed or aliased
   while the request is active, and no lifetime appears on the type.

3. **A window owns its buffer.** `Win<T>` has no lifetime. `Win::create` takes a
   `Vec<T>`, and the collective `into_buffer` returns it. The buffer is released only
   after a successful `MPI_Win_free`, and is leaked whenever the free is skipped or
   fails. The guarantee: no safe code can free a window's memory while MPI can still
   address it.

4. **Epochs are closures.** `Win::fence`, `Win::start` (the PSCW access epoch),
   `Win::lock` and `Win::lock_all` take a closure over `&mut Win` and always issue the
   closing call. The epoch handles, `ActiveEpoch` and `PassiveEpoch`, carry the scope's
   invariant lifetime, so a transfer's buffers outlive the closing call and must be
   declared before the epoch call, and a view cannot leave its epoch's closure. Only the
   passive-target handle carries `flush`, `flush_local`, `sync` and request-based RMA,
   and its requests complete before the unlock. The guarantee: an epoch cannot be left
   open by forgetting a guard, because no guard exists.

5. **Panics during unwind follow the closing call.** A nonblocking scope waits for its
   outstanding requests while unwinding, as `std::thread::scope` does. A fence or
   `start` epoch still open during a panic aborts the process, because its closing call
   is collective or synchronizing and a panicking rank cannot take part in it alone. A
   `lock` or `lock_all` closure unwinds after the unlock. All of this relies on
   unwinding: under `panic = "abort"` the process stops at the panic itself.

6. **Handles stay `'static`, and run-time accounting covers the rest.** `Communicator`,
   `Group`, `Win`, `PersistentRequest`, `CustomDatatype` and `UserOp` carry no `'mpi`
   lifetime. Only the buffer-safety scopes carry lifetimes. Every method that calls MPI
   passes a lifecycle guard: `Error::Finalized` after `Mpi` is dropped, and
   `Error::ThreadLevelViolation` on a thread the thread level does not allow. A live
   nonblocking scope, an active persistent request and a matched but unreceived
   `Message` each hold an in-flight token. `Mpi`'s drop moves to a finalizing state that
   refuses new calls but admits the completion and closing calls, waits for every
   in-flight call, and skips `MPI_Finalize` while a token or a live window remains. The
   guarantee: a handle that outlives `Mpi` returns `Error::Finalized` instead of
   calling MPI, and finalization cannot run under an operation in flight.

## Consequences

- **The compiler checks:** a buffer is borrowed until its request completes; a
  transfer's buffers outlive the epoch's closing call; a view cannot leave its epoch's
  closure; a handle never borrows `Mpi`, so owning structs compile.
- **Run time checks:** `Error::Finalized` after `Mpi` is dropped,
  `Error::ThreadLevelViolation` on a thread the level does not allow,
  `Error::InvalidState` for a persistent request's buffer while it is active, and the
  skipped `MPI_Finalize` while a token or a window remains. A skipped `MPI_Finalize`
  leaves the process to exit without finalizing MPI; stderr carries a warning.
- **What stays the caller's:** the zero-copy slice contracts of window memory, which
  [ADR-0008](0008-window-memory-access-model.md) records; the RMA assertions `NOPUT`,
  `NOSTORE` and `NOCHECK` (MPI-4.1 §12.5.5); and a peer that blocks in a collective free
  or a synchronizing close when a rank leaks its window or stops calling MPI.
- **Costs:** one in-flight token per scope, per active persistent request and per
  matched message; a scope keeps a 64-slot inline registry before it spills to the heap.
  `benches/README.md` holds the measured tables.
- **Migration:** nonblocking code moves into a scope closure, persistent code reads its
  buffers through the request, and window code takes a `Vec<T>`. The migration guide
  lists each change.
- **ADR-0004** stands for the `*_init` / `start` / `wait` design. Its buffer-lifetime
  invariants and `Drop` description are those of 0.6.

## Alternatives considered

- **Option A: a lifetime on every handle** (`Communicator<'mpi>`, `Win<'mpi, T>`).
  Rejected: handles would borrow `Mpi`, so a long-lived struct that owns `Mpi` and its
  communicators and windows would not compile, and the parameter would spread through
  every type that holds a handle. Run-time guards and in-flight accounting give the same
  protection against use after finalize.
- **Option B: `unsafe` request constructors.** Rejected: they leave the obligation with
  the caller of each constructor and keep every later call on the request safe-looking
  while it depends on that promise. The scope moves the obligation into a single place
  that the compiler checks.
- **Option C: RAII guards for requests and epochs, the 0.6 design.** Rejected: a guard
  can be forgotten, and `std::mem::forget` leaves the buffer in use or the epoch open
  without any signal. Safe code must not be able to cause that.
- **Option D: run-time checks alone, without a scope.** Rejected: a check can refuse a
  call, but it cannot stop a buffer from being freed while MPI writes to it. Only
  ownership or a borrow that outlasts the operation can.
- **Option E: lifetime-bound persistent requests, ADR-0004's Option C.** Not taken:
  owning the buffers gives the same guarantee without a lifetime on the type, and
  `std::mem::forget` on a borrowed request would still end the borrow.

## Status

Accepted — 2026-10-29. The implementation lives in `src/scope.rs`, `src/request.rs`,
`src/persistent.rs` and `src/window.rs`.
