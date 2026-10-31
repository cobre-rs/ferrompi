# Migrating from ferrompi 0.6 to 0.7

This guide lists every change that breaks ferrompi 0.6 code, section by section, with the 0.6 code and its 0.7 form. The CHANGELOG's migration notes point here. ADR-0007 (`adr/0007-buffer-safety-and-handle-lifetimes.md`) and ADR-0008 (`adr/0008-window-memory-access-model.md`) give the reasons for the buffer, handle and window changes.

## Errors

`Error::InvalidBuffer` and `Error::InvalidOp` are removed. Three variants replace them, and each names the argument or state that failed:

| 0.6 code that returned `InvalidBuffer`, `InvalidOp` or `Internal`                                                                                    | 0.7 error                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| A collective buffer of the wrong length                                                                                                              | `BufferSize { arg, required, actual }`; `InvalidArgument` if not a multiple of the size  |
| A variable-count collective with a counts or displacements array of the wrong length, or a block past the buffer                                     | `BufferSize`                                                                             |
| A negative count or displacement                                                                                                                     | `InvalidArgument`                                                                        |
| A `*_custom` method with a datatype that does not fit `T`                                                                                            | `InvalidArgument { arg: "datatype", .. }`; `reason` tells a wrong extent from stray data |
| An RMA target or window-size check (`target_rank`, `target_disp`, `target_count`, `origin`, `result`, `count`)                                       | `InvalidArgument` or `BufferSize`                                                        |
| `buffer_attach`, `buffer_detach`, `PersistentRequest::start`, `start_all`                                                                            | `InvalidState`                                                                           |
| An empty or oversized attach buffer, a NUL in the tag of `create_from_group`, a wrong op for `allreduce_bytes`                                       | `InvalidArgument`                                                                        |
| A bitwise or logical op on `f32` or `f64` (was `Mpi` with class `Op`, or a hang or abort at the RMA target on MPICH), now raised before any MPI call | `InvalidArgument { arg: "op", .. }`                                                      |

`Error` gains the three variants, and its struct variants are `#[non_exhaustive]`: a pattern on `Error::Mpi { .. }` or `Error::ResourceExhausted { .. }` needs a trailing `..`.

```rust,ignore
match world.allreduce(&send, &mut recv, ReduceOp::Sum) {
    Err(Error::InvalidBuffer) => eprintln!("buffer length"),
    Err(Error::InvalidOp) => eprintln!("op"),
    Err(Error::Mpi { class, code, message, operation }) => eprintln!("{class:?} {code}"),
    _ => {}
}
```

```rust,no_run
# use ferrompi::{Error, Mpi, ReduceOp};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let send = [1.0f64; 4];
# let mut recv = [0.0f64; 3];
match world.allreduce(&send, &mut recv, ReduceOp::Sum) {
    Err(Error::BufferSize { arg, required, actual, .. }) => eprintln!("{arg}: {actual} of {required}"),
    Err(Error::InvalidArgument { arg, reason, .. }) => eprintln!("{arg}: {reason}"),
    Err(Error::InvalidState { reason, .. }) => eprintln!("{reason}"),
    Err(Error::Mpi { class, code, .. }) => eprintln!("{class:?} {code}"),
    _ => {}
}
# Ok(())
# }
```

Four constructors no longer exist outside the crate: `Error::from_code`, `Error::from_code_with_op`, `Error::check` and `Error::check_with_op` interpreted return codes of the crate's own C layer, which no code outside the crate can produce. `MpiErrorClass::from_raw` mapped a class number of the linked MPI library, which no ferrompi call returns. Match on the `Error` a call returns instead, and read the `class` field of `Error::Mpi`.

## Sources, destinations and tags

Receives and probes take `impl Into<Source>` and `impl Into<Tag>`; sends take `dest: impl Into<Source>`. `Source` is `Any`, `ProcNull` or `Rank(r)`, and `Tag` is `Any` or `Value(t)`. An `i32` still converts, so `world.recv(&mut buf, 0, 7)` compiles unchanged. A negative rank or tag, or `Source::Any` as a destination, returns `Error::InvalidArgument` before any MPI call. A 0.6 `-1` destination, which meant `MPI_PROC_NULL` on MPICH, becomes `Source::ProcNull`.

`Communicator::split` takes `color: Option<i32>`, where `None` opts out, and `Group::rank` returns `Result<Option<i32>>`, where `None` marks a non-member. `Communicator::UNDEFINED` and `Group::undefined()` are removed. A negative `Some` color returns `Error::InvalidArgument`.

```rust,ignore
world.recv(&mut buf, -1, -1)?;
world.send(&buf, -1, 0)?;
let half = world.split(world.rank() % 2, 0)?;
let none = world.split(Communicator::UNDEFINED, 0)?;
let rank = group.rank()?;
```

```rust,no_run
# use ferrompi::{Mpi, Source, Tag};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let mut buf = [0.0f64; 4];
world.recv(&mut buf, Source::Any, Tag::Any)?;
world.send(&buf, Source::ProcNull, 0)?;
let half = world.split(Some(world.rank() % 2), 0)?;
let none = world.split(None, 0)?;
let rank: Option<i32> = world.group()?.rank()?;
# Ok(())
# }
```

## Receive status

`recv` and `sendrecv` return a `Status` instead of a `(source, tag, count)` tuple. `Status` is `#[non_exhaustive]` with typed fields: `source: Source`, `tag: Tag`, `count: Option<usize>` (`None` when the message is not a whole number of elements) and `error: Option<MpiErrorClass>`. A receive from `Source::ProcNull` reports `Source::ProcNull`, `Tag::Any` and `Some(0)` on every MPI.

```rust,ignore
let (source, tag, count) = world.recv(&mut buf, -1, -1)?;
if count < 0 {
    eprintln!("partial element");
}
```

```rust,no_run
# use ferrompi::{Mpi, Source, Tag};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let mut buf = [0.0f64; 4];
let st = world.recv(&mut buf, Source::Any, Tag::Any)?;
if st.count.is_none() {
    eprintln!("partial element");
}
if st.source == Source::Rank(0) && st.tag == Tag::Value(7) {
    eprintln!("from rank 0, tag 7");
}
# Ok(())
# }
```

## Reduction operations

`ReduceOp::MaxLoc` and `ReduceOp::MinLoc` are replaced by `CollectiveOp::MAX_LOC` and `CollectiveOp::MIN_LOC`, which exist only for the pair types. `allreduce_indexed` takes `impl Into<CollectiveOp<T>>`, so any other op on a pair type, or `MAX_LOC` on a primitive type, is a compile error instead of a run-time error.

The blocking reductions (`reduce`, `allreduce`, `scan`, `exscan`, `reduce_scatter_block` and their scalar and in-place forms) take `op: impl Into<CollectiveOp<T>>`, which accepts a `ReduceOp`, a `&UserOp<T>` or a `CollectiveOp`. A `ReduceOp` argument compiles unchanged, and `allreduce_with_op` is removed in favour of `allreduce`. A user op above `i32::MAX` elements returns `Error::Mpi` with class `Count` on every MPI version.

`ReduceOp::Replace` and `ReduceOp::NoOp` are replaced by `AccumulateOp::REPLACE`, `FetchOp::REPLACE` and `FetchOp::NO_OP`, present with and without the `rma` feature. `accumulate` and `raccumulate` take `impl Into<AccumulateOp>` and `get_accumulate` and `fetch_and_op` take `impl Into<FetchOp>`, so a `ReduceOp` argument still compiles, but a user op in an accumulate, `NO_OP` in `accumulate`, or `REPLACE` in a collective is a compile error. The calls are in [RMA epochs](#rma-epochs).

```rust,ignore
world.allreduce_indexed(&send, &mut recv, ReduceOp::MaxLoc)?;
world.allreduce_with_op(&send, &mut recv, &op)?;
let replace = ReduceOp::Replace;
let no_op = ReduceOp::NoOp;
```

```rust,no_run
# use ferrompi::{AccumulateOp, CollectiveOp, DoubleInt, FetchOp, Mpi, UserOp};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let send = [DoubleInt { value: 1.0, index: 0 }];
# let mut recv = [DoubleInt { value: 0.0, index: 0 }];
world.allreduce_indexed(&send, &mut recv, CollectiveOp::MAX_LOC)?;
let op = UserOp::<f64>::new(|a, b| b.iter_mut().zip(a).for_each(|(y, x)| *y += x))?;
world.allreduce(&[1.0f64], &mut [0.0f64], &op)?;
let (replace, fetch_replace, no_op) = (AccumulateOp::REPLACE, FetchOp::REPLACE, FetchOp::NO_OP);
# Ok(())
# }
```

## Derived datatypes

`CustomDatatype::contiguous`, `CustomDatatype::vector` and `StructField::new` take their base type as a type parameter. `DatatypeTag` and the `TAG` constants of `MpiDatatype` and `MpiIndexedDatatype` are removed, and `StructField`'s fields are private. A pair type as a base type is a compile error (it was `Error::InvalidOp`), and `MPI_BYTE` is no longer a base type: use `u8` where `DatatypeTag::Byte` was the base type.

```rust,ignore
let row = CustomDatatype::contiguous(10, DatatypeTag::F64)?;
let col = CustomDatatype::vector(4, 1, 10, DatatypeTag::F64)?;
let field = StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 };
```

```rust,no_run
# use ferrompi::{CustomDatatype, Mpi, StructField};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
let row = CustomDatatype::contiguous::<f64>(10)?;
let col = CustomDatatype::vector::<f64>(4, 1, 10)?;
let field = StructField::new::<i32>(1, 8);
let record = CustomDatatype::create_struct(&[field])?;
# Ok(())
# }
```

## Nonblocking requests and scopes

`Request` has a lifetime (`Request<'s>`) and is no longer `Send`: keep each request on the thread that created it. Every nonblocking call takes the scope `s` first, inside `ferrompi::scope(|s| ...)`, and borrows its buffers (and datatype, user op, and the v-forms' count and displacement arrays) until `scope` returns. Receive buffers are readable again after the scope. This covers `isend`, `irecv`, `isend_custom`, `irecv_custom`, `ibarrier`, the nonblocking reductions (which also take `op: impl Into<CollectiveOp<T>>`), and every other nonblocking collective.

`Request::wait` returns `Status`, `test` returns `Option<Status>`, and `wait_any` and `test_any` return `Option<(usize, Status)>`. A send, collective or RMA request reports the empty status (`Source::Any`, `Tag::Any`, `count: Some(0)`). `wait_some` and `test_some` return `Vec<(usize, Status)>`, and a failed request in the batch reports its error in `Status.error` instead of failing the whole call.

`iscatterv` and `ialltoallv` take their parameters in the order of `scatterv` and `alltoallv`, which is MPI's: `recv` moves behind the send displacements. A call left in the old order passes a shared slice where `recv` is expected and does not compile.

```rust,ignore
let req = world.isend(&data, 1, 0)?;
let mut incoming = world.irecv(&mut buf, 0, 0)?;
let barrier = world.ibarrier()?;
while !incoming.test()? {}
let first = Request::wait_any(&mut reqs)?;
let done = Request::test_some(&mut reqs)?;
world.iscatterv(&send, &mut recv, &counts, &displs, 0)?.wait()?;
```

```rust,no_run
# use ferrompi::{Mpi, Request};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let data = [1.0f64; 4];
# let mut buf = [0.0f64; 4];
# let (send, mut recv) = (vec![0.0f64; world.size() as usize], [0.0f64; 1]);
# let (counts, displs) = (vec![1; world.size() as usize], (0..world.size()).collect::<Vec<_>>());
ferrompi::scope(|s| {
    let req = world.isend(s, &data, 1, 0)?;
    let mut incoming = world.irecv(s, &mut buf, 0, 0)?;
    let barrier = world.ibarrier(s)?;
    while incoming.test()?.is_none() {}
    let mut reqs = [req, barrier];
    let first: Option<usize> = Request::wait_any(&mut reqs)?.map(|(i, _)| i);
    let done: Vec<usize> = Request::test_some(&mut reqs)?.iter().map(|&(i, _)| i).collect();
    world.iscatterv(s, &send, &counts, &displs, &mut recv, 0)?.wait()?;
    Ok(())
})?;
# Ok(())
# }
```

## Persistent requests

The persistent point-to-point requests (`send_init`, `bsend_init`, `rsend_init`, `ssend_init`, `recv_init`) and the persistent collectives (including the reductions, which also take `op: impl Into<CollectiveOp<T>>`) own their buffers. They take `Vec<T>` by value and return `PersistentRequest<T>`. `buffer()` and `buffer_mut()` reach the buffer while the request is inactive and return `Error::InvalidState` while it is active. `into_buffer()` returns the buffer, and `into_buffers()` returns both buffers of a two-buffer request. `send_buffer_mut()` reaches the send buffer of such a request while it is inactive, and returns `Error::InvalidState` while it is active. A `&UserOp` passed to a persistent reduction stays alive inside the request until the request is freed. The v-forms copy their count and displacement arrays, which the caller may reuse. `start_all` and `wait_all` take requests of one element type: split a mixed batch by type.

`bcast_init` is renamed `broadcast_init`. `PersistentRequest::wait` returns `Status` and `test` returns `Option<Status>`, as `Request`'s do; a persistent receive reports the matched source, tag and count on every start, and a persistent send, collective or inactive request reports the empty status.

```rust,ignore
let mut data = vec![0.0f64; 8];
let mut req = world.bcast_init(&mut data, 0)?;
req.start()?;
while !req.test()? {}
req.wait()?;
println!("{data:?}");
```

```rust,no_run
# use ferrompi::{Mpi, PersistentRequest, ReduceOp};
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
let mut req = world.broadcast_init(vec![0.0f64; 8], 0)?;
req.start()?;
while req.test()?.is_none() {}
req.wait()?;
println!("{:?}", req.buffer()?);

let mut sums = world.allreduce_init(vec![1.0f64; 8], vec![0.0; 8], ReduceOp::Sum)?;
PersistentRequest::start_all(std::slice::from_mut(&mut sums))?;
PersistentRequest::wait_all(std::slice::from_mut(&mut sums))?;
let totals: Vec<f64> = sums.into_buffer()?;
# Ok(())
# }
```

## RMA epochs

Everything here needs the `rma` feature. A window owns its buffer: `Win<T>` has no lifetime parameter, `Win::create` takes a `Vec<T>`, and `Win::into_buffer` (collective) frees the window and returns the buffer (`None` for `Win::allocate` and `Win::allocate_shared`). A window dropped or forgotten while MPI may still use its buffer leaks the buffer instead of freeing it (ADR-0007).

Every access epoch is a closure that receives an epoch handle, and the crate runs the closing call (fence, `MPI_Win_complete` or unlock) whether the closure returns `Ok` or `Err`; if the closure panics, `Win::lock` and `Win::lock_all` unlock and continue unwinding, while `Win::fence` and `Win::start` abort the process before any closing call.

- `Win::fence(open, close, |e| ..)` and `Win::start(group, assert, |e| ..)` pass an `ActiveEpoch`. `Win::complete` is removed. `fence` rejects `no_succeed()` on the opening fence and `no_precede()` on the closing one with `Error::InvalidArgument`.
- `Win::lock(kind, rank, |e| ..)` and `Win::lock_all(|e| ..)` pass a `PassiveEpoch`. `flush`, `flush_all`, `flush_local`, `flush_local_all` and `sync` are `PassiveEpoch` methods; `WinLockGuard`, `WinLockAllGuard` and the `Win` versions of the flushes and `sync` are removed.
- `put`, `get`, `accumulate`, `get_accumulate`, `fetch_and_op` and `compare_and_swap` are epoch-handle methods. They borrow their buffers until the epoch closes and take their count from the slice, so the `target_count` argument is gone: pass a sub-slice for a shorter transfer. `get_accumulate` needs `result` as long as `origin`. Declare the buffers before the closure.
- `fetch_and_op` and `compare_and_swap` take `origin: &T` (and `compare: &T`) and write into `result: &mut T`, then return `Result<()>`. `PendingFetchResult` is removed: read the result after the closure returns.
- `rput`, `rget` and `raccumulate` exist only on `PassiveEpoch`, and return a `Request` that the epoch completes before it unlocks. `Win::rput`, `Win::rget` and `Win::raccumulate` are removed.
- A transfer's target must be in its epoch's access set: any rank but the locked one is refused in `Win::lock`, and any rank outside the `group` of `Win::start`. A fence or `lock_all` epoch covers every rank, and `flush` and `flush_local` still take any rank of the window's communicator. The refusal is `Error::InvalidArgument` with `arg: "target_rank"`, before any MPI call. `Win::start` also returns `InvalidArgument` with `arg: "group"` for a member outside the window's communicator. Open the epoch on the ranks the closure targets.

```rust,ignore
let mut buf = vec![0.0f64; 16];
let mut win = Win::create(&world, &mut buf)?;
win.fence(WinFenceAssert::none())?;
win.put(&origin, 1, 0, 4)?;
win.fence(WinFenceAssert::none())?;
win.start(&group, WinPscwAssert::none())?;
win.accumulate(&origin, 1, 0, 4, ReduceOp::Sum)?;
win.complete()?;

let guard = win.lock(LockType::Exclusive, 1)?;
let old = guard.fetch_and_op(1, 1, 0, ReduceOp::Sum)?;
let swapped = guard.compare_and_swap(5, 0, 1, 1)?;
let req = guard.rput(&origin, 1, 0, 4)?;
guard.flush(1)?;
let old = old.wait()?;
drop(guard);
```

```rust,no_run
# #[cfg(feature = "rma")]
# fn main() -> ferrompi::Result<()> {
# use ferrompi::{FetchOp, LockType, Mpi, AccumulateOp, Win, WinFenceAssert, WinPscwAssert};
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let group = world.group()?;
let mut win = Win::create(&world, vec![0.0f64; 16])?;
let origin = [1.0f64; 4];
let none = WinFenceAssert::none();
win.fence(none, none, |e| e.put(&origin, 1, 0))?;
win.start(&group, WinPscwAssert::none(), |e| {
    e.accumulate(&origin, 1, 0, AccumulateOp::REPLACE)
})?;

let mut old = 0.0f64;
let mut seen = 0.0f64;
win.lock(LockType::Exclusive, 1, |e| {
    e.fetch_and_op(&1.0, &mut old, 1, 0, FetchOp::NO_OP)?;
    let req = e.rput(&origin, 1, 0)?;
    e.flush(1)?;
    req.wait()?;
    Ok(())
})?;
let buffer: Option<Vec<f64>> = win.into_buffer()?;
# Ok(())
# }
# #[cfg(not(feature = "rma"))]
# fn main() {}
```

## Window memory and shared windows

`Win::local_slice` and `Win::local_slice_mut` are removed. `win.local()`, and `e.local()` on both epoch handles, return a `WinView` whose `read`, `write`, `copy_to_slice` and `copy_from_slice` are Relaxed atomic accesses that stay sound while peers write the window. `WinView::as_slice` and `as_mut_slice` (`unsafe`) give a zero-copy slice under the synchronization contract their documentation states.

`SharedWindow`, `LockGuard` and `LockAllGuard` are removed. `Win::allocate_shared` creates a shared-memory window as an ordinary `Win`, and `remote(r)` replaces `remote_slice(r)`. Close a fence epoch with `WinFenceAssert::no_succeed()` and a barrier before the next lock.

```rust,ignore
let mut win = SharedWindow::<f64>::allocate(&node, 10)?;
win.local_slice_mut()[0] = 1.0;
win.fence()?;
let first = win.remote_slice(0)?[0];
let _guard = win.lock_all()?;
```

```rust,no_run
# #[cfg(feature = "rma")]
# fn main() -> ferrompi::Result<()> {
# use ferrompi::{Mpi, Win, WinFenceAssert};
# let mpi = Mpi::init()?;
# let node = mpi.world().split_shared()?;
let mut win = Win::<f64>::allocate_shared(&node, 10)?;
let none = WinFenceAssert::none();
win.fence(none, WinFenceAssert::no_succeed(), |e| {
    e.local().write(0, 1.0);
    Ok(())
})?;
node.barrier()?;
let first = win.lock_all(|e| Ok(e.remote(0)?.read(0)))?;
# Ok(())
# }
# #[cfg(not(feature = "rma"))]
# fn main() {}
```

## Non-exhaustive enums

`MpiErrorClass`, `ResourceKind`, `ReduceOp` and `SplitType` are `#[non_exhaustive]`, and so are the structs `HostEntry` and `SlurmInfo`. `ThreadLevel`, `GroupComparison`, `Source`, `Tag` and `LockType` stay exhaustive, and `RankRange` keeps its three public fields, since MPI fixes them. Add a wildcard arm to a `match` on one of the four enums. A pattern on `HostEntry` or `SlurmInfo` needs a trailing `..`, and neither can be built with a struct literal.

## Communicators from groups

`Communicator::create_from_group` is renamed `Communicator::create`, after the `MPI_Comm_create` it wraps; an error from it names the operation `comm_create`. `Mpi::create_from_group` keeps its name, and returns `Result<Option<Communicator>>`: a rank outside the group, and every rank with the empty group, gets `Ok(None)`. It used to get `Error::Mpi` with class `Other`.

```rust,ignore
let sub = world.create_from_group(&group)?;
let comm = mpi.create_from_group(&group, "app.workers")?;
comm.barrier()?;
```

```rust,no_run
# use ferrompi::Mpi;
# fn main() -> ferrompi::Result<()> {
# let mpi = Mpi::init()?;
# let world = mpi.world();
# let group = world.group()?;
let sub = world.create(&group)?;
if let Some(comm) = mpi.create_from_group(&group, "app.workers")? {
    comm.barrier()?;
}
# Ok(())
# }
```

## Removed without replacement

- `raw_handle()` on `Communicator`, `Group`, `Request`, `PersistentRequest`, `CustomDatatype` and `Win`. The value was an index into a private table, not an MPI handle. Delete the call, and keep your own identifier to tell two objects apart.
- `WinFenceAssert::bits()` and `WinPscwAssert::bits()`. They returned the library's `MPI_MODE_*` values, which differ between MPI implementations. Delete the call, and combine the named constructors with `|`.
- `Info` and `ResourceKind::Info`. No ferrompi call took an `Info`, so no MPI hint could be set through it. Delete its uses and any match arm that names `ResourceKind::Info`. Every call with an MPI info argument passes `MPI_INFO_NULL`, as it always did.
- The debug-build note `MPI_Finalize leaves N active request(s) unfreed`. No request can be left pending at finalize any more, so there is nothing to replace it with.

## The slurm feature

The `numa` feature is renamed `slurm`, after what it gates: the `slurm` module and `SlurmInfo`. `slurm` does not imply `rma`. `numa` stays for 0.7 as an alias for `slurm` plus `rma`, its 0.6 meaning, and is removed in 0.8.0.

```toml
# 0.6
ferrompi = { version = "0.6", features = ["numa"] }
```

```toml
# 0.7: add "rma" only if the crate uses a window
ferrompi = { version = "0.7", features = ["slurm", "rma"] }
```

## Changes that still compile

These compile unchanged and behave differently. Check each.

- **`-1` sources and tags.** A `-1` rank or tag converts from `i32`, and now returns `Error::InvalidArgument` before any MPI call. Search for them and use `Source::Any`, `Tag::Any` or `Source::ProcNull`.
- **Root-only length check.** `reduce`, `ireduce` and `reduce_init` check `recv`'s length only at the root. Another rank may pass an empty buffer, as for `gather` and `scatter`.
- **After a skipped `MPI_Finalize`** (a window still alive when `Mpi` is dropped), `wait` and `test` on a pending request still complete it and `cancel` still reaches it, where they returned `Error::Finalized`. Every other call still returns `Error::Finalized`. Dropping `Mpi` inside a `scope` on the same thread also skips `MPI_Finalize`, with a warning.
- **MPI error text** is no longer asked from MPI while `MPI_Finalize` runs or after it returned, or on a thread the thread level does not allow to call MPI. The error has class `Raw` and the message `MPI error code N`.
- **Dropping a request** does not wait for it. Its `scope`, or the passive epoch that returned it from `rput`, `rget` or `raccumulate`, completes it.
- **Dropping `Mpi` with an active `PersistentRequest`** skips `MPI_Finalize`, on whatever thread the request is, with a stderr warning. The request can still be completed afterwards, and dropping it then makes no MPI call.
- **`Win::create` on a full window table** returns `Error::ResourceExhausted` instead of aborting. The window and its buffer are leaked, and `Mpi` then skips `MPI_Finalize`.
