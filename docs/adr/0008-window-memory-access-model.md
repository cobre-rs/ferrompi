# ADR-0008: Window Memory Access Model

**Status:** Accepted — 2026-10-29
**Date:** 2026-10-29
**Deciders:** Rogerio Alves

## Context

Window memory can change while a reference to it lives: a peer's put or accumulate
writes it, a peer stores into it through a shared mapping, and a fence lands remote
writes in it. A safe `&[T]` or `&mut [T]` over that memory promises the compiler that
nothing else writes it, so a release build can hoist a load out of a polling loop and
never see the peer's write. The 0.6 accessors did exactly that: `Win::local_slice`,
`Win::local_slice_mut` and the guards of `SharedWindow`.

GitHub issue #28 proposed per-region borrow guards on shared windows. It raised the
whole-window borrow problem, and its point that a remote zero-copy view has to be
`unsafe`, because the library cannot verify that the owner stays read-only, is part of
this design. ADR-0007 (`docs/adr/0007-buffer-safety-and-handle-lifetimes.md`) records the other
side of windows, who owns the buffer and how epochs close.

Two facts from the verification stacks shape the decisions. `MPI_Win_allocate_shared`,
`MPI_Win_allocate` and `MPI_Win_create` report `MPI_WIN_UNIFIED` on MPICH 4.2.3, MPICH
4.2.1 with UCX, Open MPI 4.1.6 and Open MPI 5.0.7, at 2 and 4 ranks. And Open MPI 4.1.6
aligns the first shared segment to 8 bytes only (8 mod 16 in every run), which is below
the "any predefined datatype" bound of MPI-4.1 §9.2 (p.447) but enough for every window
element, whose alignment is at most 8.

## Decision

1. **A view on every window.** `local()` exists on every `Win` and on both epoch
   handles, `ActiveEpoch` and `PassiveEpoch`. `remote(rank)` exists on windows made by
   `Win::allocate_shared` and on their epoch handles; on any other window it returns
   `Error::NotSupported`, and no marker type distinguishes the two. Both return
   `WinView`, which is neither `Send` nor `Sync` and claims no exclusivity: `write`
   takes `&self`, and views of the local and a remote region coexist. The 0.6
   `Win::local_slice` and `Win::local_slice_mut` are gone, and so is `SharedWindow`.

2. **Element access is one `Relaxed` atomic.** A load or store goes through
   `AtomicUN::from_ptr`, so a concurrent write by a peer, by MPI or by another view
   never makes an access undefined behaviour. Elements are the `MpiDatatype` set. A
   compile-time assertion fails the build unless `size_of::<T>()` is 1, 2, 4 or 8,
   `align_of::<T>()` is at least the atomic's alignment, and `cfg(target_has_atomic)`
   holds for that width. There is no volatile fallback. The bulk copies,
   `copy_to_slice` and `copy_from_slice`, go element by element: no tearing between
   views, and the copy as a whole is not atomic. The view panics on an out-of-range
   index or a length mismatch, as slices do. It is the one ferrompi API that panics on
   caller input; every other call returns an `Error`.

3. **Zero-copy slices are `unsafe` and carry a written contract.** `WinView::as_slice`
   and `WinView::as_mut_slice` serve a caller that knows its phase structure. The
   rustdoc of `src/window.rs` is the normative text; the `# Safety` paragraphs below
   are quoted from it verbatim. For `as_slice`:

> No process writes the region by store, put or accumulate while the slice lives, and the writes it must observe were published before it was created, either by a fence or `MPI_Win_wait` at this process that completed them, or by process synchronization after they completed, then `MPI_Win_sync` at this process inside a passive-target epoch.
>
> A store through any view of the region counts, this view's `write` and `copy_from_slice` included: the borrow checker does not stop them while the slice lives.

   For `as_mut_slice`:

> No process writes the region by store, put or accumulate while the slice lives, and the writes it must observe were published before it was created, either by a fence or `MPI_Win_wait` at this process that completed them, or by process synchronization after they completed, then `MPI_Win_sync` at this process inside a passive-target epoch. No other access to the region, read or write, happens while the slice lives: by any process, by MPI on this process's behalf, or by any other view or slice in this process, including `remote(own rank)`, which aliases `local()`.
>
> The reasons are those of `as_slice`, and a `&mut [T]` must also be unique. Writes through the slice itself, including by an MPI call it is lent to, are the one exception to the first sentence. `&mut self` enforces uniqueness only against the view it is taken from: another view of the region, or an RMA call that targets this process, is not stopped by it.

4. **Publication is MPI synchronization.** What a reader of a peer's region may rely on
   is quoted from the rustdoc of `Win::remote`:

> A store, put or accumulate by another process is guaranteed to be visible to this process's loads of the region only after MPI synchronization; on coherent hardware it may be seen earlier. `MPI_Win_wait` at this process never publishes a peer's region: it completes the exposure epoch of this process's own region. A fence publishes what was written before the previous fence. That fence completes puts and accumulates at the region's owner, and makes stores to shared memory visible, before it returns there (MPI-4.1 §12.5.1, p.591; §12.7, rule 2, p.606). The next fence synchronizes unless it carries `MPI_MODE_NOPRECEDE` (p.592), so it returns at this process only after the owner has entered it. Otherwise a reader needs the passive-target route: the writer's `sync` after its stores, or its unlock or flush after its puts and accumulates, then process synchronization, then `sync` at this process inside a passive-target epoch (MPI-4.1 §12.5.4, p.603; the advice to users after rule U5, MPI-4.1 §12.7, p.609; Example 12.23, p.622). The same holds for a slice of the view, [`WinView::as_slice`].

   Publishing through a lock on one's own rank (the "self-lock route") is deliberately
   outside the contract. Its routes are a fence, `MPI_Win_wait` at this process, or
   process synchronization followed by `MPI_Win_sync` inside a passive-target epoch.

5. **Shared windows are ordinary windows.** `Win::allocate_shared` makes a `Win<T>`, so
   epochs, views and transfers are one API for every window kind.
   - It queries every rank's base and length once, at creation, with
     `MPI_Win_shared_query`, so `remote` makes no MPI call.
   - Every rank must pass an element type of one size. A mismatch is refused on every
     rank before any window exists, with `Error::InvalidArgument` and `arg: "T"`.
   - `remote` returns `Error::NotSupported` unless MPI reports `MPI_WIN_UNIFIED`,
     because MPI defines no semantics for direct access to shared window memory in the
     separate model (MPI-4.1 §12.2.3, p.558). `local` works in either model.
   - MPI aligns the first non-empty segment for any predefined datatype and places the
     others after it (MPI-4.1 §9.2, p.447; §12.2.3, p.555-556). Open MPI 4.1.6 meets
     only the 8-byte floor of Context, which suffices because the element assertion of
     item 2 caps element alignment at 8.

6. **A publish-then-read-only window is deferred.** A collective freeze after which
   the window yields a safe `&[T]` would add a read-only phase without the `unsafe`
   contract. It is not built: no consumer needs it, and adding it later breaks nothing.

## Consequences

- Polling a flag is safe: each poll is an atomic load, and the compiler cannot hoist it.
- Indexing and `Deref` are lost on the safe path. The bulk copies and the `unsafe`
  slices replace them. The window is stored in long-lived state, and a view or slice is
  derived per phase: a view is a pointer and a length, with no MPI call.
- What a load returns follows MPI's memory model (MPI-4.1 §12.7). In the unified model
  a load that races a put or accumulate is valid and returns an implementation-dependent
  value, so poll for a change from zero to nonzero, not for a relative magnitude. In the
  separate model this process's window synchronization calls would access the region on
  its behalf; every verification stack reports the unified model, so `remote` refuses
  the separate one.
- A process that only polls makes no MPI progress: a peer's put or accumulate may not
  reach it until it calls MPI (MPI-4.1 §12.7.3, p.615-616).
- The borrow checker enforces little of aliasing: a slice borrows the view it comes
  from, and the view borrows its window or epoch handle, so a slice cannot outlive
  either, and an epoch call, which borrows the window mutably, cannot run while it
  lives. `as_mut_slice` also stops every other use of its own view. Another view of
  the region, other processes, and RMA calls that target this process are covered
  only by the written contract of item 3.

## Alternatives considered

- **Option A: safe slice guards over window memory** (issue #28's `SharedArray` and
  `SharedArrayMut`). Rejected: a safe `&[T]` or `&mut [T]` over memory that another
  process can write is undefined behaviour the moment a peer writes, whatever the
  in-process borrow state. The local region has the same problem, so it is not a
  shared-window-only fix.
- **Option B: per-region `RefCell` tracking**, also from issue #28. Rejected: a token in
  one process cannot exclude writes by another process, and the view claims no
  exclusivity, so no in-process claim is left to police. A per-access flag would cost
  every access and still leave the cross-process case to a written contract.
- **Option C: volatile access.** Rejected: volatile loads and stores are not atomic, so
  a racing write can tear an element and the access is still undefined behaviour in
  Rust. A fallback for elements without a matching atomic would hide that gap, so the
  build fails there instead.
- **Option D: a lazy per-rank `MPI_Win_shared_query` cache.** Rejected: it would make
  `remote` an MPI call on first use. Creation is already collective over a small
  node-local communicator, and the eager query costs 16 bytes per rank.
- **Option E: a marker type for shared windows.** Rejected: a second window type
  splits epochs, views and transfers into two APIs. `remote` on a window that is not
  shared returns `Error::NotSupported` instead.

[`WinView::as_slice`]: https://docs.rs/ferrompi/latest/ferrompi/struct.WinView.html#method.as_slice
