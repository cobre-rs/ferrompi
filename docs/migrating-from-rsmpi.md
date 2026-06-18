# Migrating from rsmpi to ferrompi

## Overview

This guide is for developers who already know rsmpi and want to move a project to
ferrompi. It is not a tutorial: it assumes familiarity with rsmpi's trait-based API
and MPI concepts in general. The document maps rsmpi expressions to ferrompi
equivalents, highlights the three categories of difference that affect every
migration (naming, type ergonomics, and feature scope), and lists features that
rsmpi supports but ferrompi deliberately does not provide. Read the summary table
first to decide whether the trade-offs are right for your project, then use the
function-for-function mapping as a conversion reference.

---

## 1. Quick Comparison

The table below extends the feature matrix in `README.md` with the dimensions that
matter most during migration. See `docs/architecture.md` for the rationale behind
each design choice.

| Dimension              | rsmpi 0.8                                    | ferrompi 0.5                                               |
| ---------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| MPI standard           | 3.1                                          | 4.1                                                        |
| Persistent collectives | No                                           | Yes (`_init` / `start` / `wait`)                           |
| Large-count (>2³¹)     | No                                           | Yes                                                        |
| Thread safety          | `!Send` — communicators cannot cross threads | `Send + Sync` — hybrid MPI+threads                         |
| Generic API style      | Trait objects (`Buffer`, `BufferMut`)        | Sealed trait (`MpiDatatype`)                               |
| Buffer wrapping        | `.buffer()` / `.buffer_mut()` required       | `&[T]` / `&mut [T]` directly                               |
| Custom datatypes       | `Equivalence` derive macro                   | `CustomDatatype` builder + `BytePermutable`                |
| Custom reductions      | `UserOperation` trait                        | `UserOp<T: MpiDatatype>` (16-slot registry)                |
| Error handling         | Panics on MPI errors by default              | `Result<T, Error>` on every call                           |
| RAII cleanup           | Manual or partial                            | Full — `Drop` frees groups, comms, windows, datatypes, ops |
| Shared memory windows  | No                                           | Yes (`Win<T>` with `rma` feature)                          |
| Intercommunicators     | Yes                                          | No                                                         |
| Dynamic processes      | Yes (`spawn`, `accept`, `open_port`)         | No                                                         |
| MPI I/O                | Yes (`MPI_File_*`)                           | No                                                         |

ferrompi is deliberately narrower than rsmpi. The trade-off is a smaller, more
consistent API in exchange for dropping features that few HPC applications use.

---

## 2. Function-for-Function Mapping

### 2a. Collectives

In rsmpi, collectives are split across two traits: `Root` (operations where one
rank is special — broadcast, gather, scatter, reduce) and `CommunicatorCollectives`
(symmetric operations — allreduce, allgather, alltoall, scan, barrier). The caller
must first obtain a `Process` handle via `comm.process_at_rank(root)` and call the
method on that handle. In ferrompi all collectives are inherent methods on
`Communicator` with an explicit `root: i32` parameter where needed.

Nonblocking rsmpi methods are prefixed `immediate_`; ferrompi uses `i` prefix.
Persistent collectives do not exist in rsmpi; the right column shows `_init` methods
(MPI 4.0+).

| rsmpi expression                                                    | ferrompi expression                                     | Notes                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------ |
| `comm.process_at_rank(r).broadcast_into(&mut buf)`                  | `comm.broadcast(&mut buf, r)?`                          | Root and non-root use the same call                    |
| `comm.process_at_rank(r).immediate_broadcast_into(scope, &mut buf)` | `comm.ibroadcast(&mut buf, r)?`                         | Returns `Request`; no scope needed                     |
| `(none — not available in rsmpi)`                                   | `comm.bcast_init(&mut buf, r)?`                         | Persistent; call `.start()` + `.wait()` each iteration |
| `comm.process_at_rank(r).reduce_into(&send, op)`                    | `comm.reduce(&send, &mut recv, op, r)?`                 | Non-root `recv` may be empty slice                     |
| `comm.process_at_rank(r).reduce_into_root(&send, &mut recv, op)`    | `comm.reduce(&send, &mut recv, op, r)?`                 | Same call for root and non-root                        |
| `comm.process_at_rank(r).immediate_reduce_into(&send, op)`          | `comm.ireduce(&send, &mut recv, op, r)?`                | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.reduce_init(&send, &mut recv, op, r)?`            | Persistent                                             |
| `comm.all_reduce_into(&send, &mut recv, op)`                        | `comm.allreduce(&send, &mut recv, op)?`                 |                                                        |
| `comm.all_reduce_into(&send, &mut recv, op)` (scalar)               | `comm.allreduce_scalar(value, op)?`                     | Convenience: no buffers; returns `T`                   |
| `comm.immediate_all_reduce_into(&send, &mut recv, op)`              | `comm.iallreduce(&send, &mut recv, op)?`                | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.allreduce_init(&send, &mut recv, op)?`            | Persistent                                             |
| `comm.process_at_rank(r).gather_into(&send)`                        | `comm.gather(&send, &mut recv, r)?`                     | `recv` ignored on non-root                             |
| `comm.process_at_rank(r).gather_into_root(&send, &mut recv)`        | `comm.gather(&send, &mut recv, r)?`                     | Same call for root and non-root                        |
| `comm.process_at_rank(r).immediate_gather_into(&send)`              | `comm.igather(&send, &mut recv, r)?`                    | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.gather_init(&send, &mut recv, r)?`                | Persistent                                             |
| `comm.all_gather_into(&send, &mut recv)`                            | `comm.allgather(&send, &mut recv)?`                     |                                                        |
| `comm.immediate_all_gather_into(&send, &mut recv)`                  | `comm.iallgather(&send, &mut recv)?`                    | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.allgather_init(&send, &mut recv)?`                | Persistent                                             |
| `comm.process_at_rank(r).scatter_into(&mut recv)`                   | `comm.scatter(&send, &mut recv, r)?`                    | `send` ignored on non-root                             |
| `comm.process_at_rank(r).scatter_into_root(&send, &mut recv)`       | `comm.scatter(&send, &mut recv, r)?`                    | Same call for root and non-root                        |
| `comm.process_at_rank(r).immediate_scatter_into(&mut recv)`         | `comm.iscatter(&send, &mut recv, r)?`                   | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.scatter_init(&send, &mut recv, r)?`               | Persistent                                             |
| `comm.all_to_all_into(&send, &mut recv)`                            | `comm.alltoall(&send, &mut recv)?`                      |                                                        |
| `comm.immediate_all_to_all_into(&send, &mut recv)`                  | `comm.ialltoall(&send, &mut recv)?`                     | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.alltoall_init(&send, &mut recv)?`                 | Persistent                                             |
| `comm.scan_into(&send, &mut recv, op)`                              | `comm.scan(&send, &mut recv, op)?`                      | Inclusive prefix scan                                  |
| `comm.immediate_scan_into(&send, &mut recv, op)`                    | `comm.iscan(&send, &mut recv, op)?`                     | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.scan_init(&send, &mut recv, op)?`                 | Persistent                                             |
| `comm.exclusive_scan_into(&send, &mut recv, op)`                    | `comm.exscan(&send, &mut recv, op)?`                    | Exclusive prefix scan                                  |
| `comm.immediate_exclusive_scan_into(&send, &mut recv, op)`          | `comm.iexscan(&send, &mut recv, op)?`                   | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.exscan_init(&send, &mut recv, op)?`               | Persistent                                             |
| `comm.reduce_scatter_block_into(&send, &mut recv, op)`              | `comm.reduce_scatter_block(&send, &mut recv, op)?`      | Equal-block variant                                    |
| `comm.immediate_reduce_scatter_block_into(&send, &mut recv, op)`    | `comm.ireduce_scatter_block(&send, &mut recv, op)?`     | Returns `Request`                                      |
| `(none — not available in rsmpi)`                                   | `comm.reduce_scatter_block_init(&send, &mut recv, op)?` | Persistent                                             |
| `comm.barrier()`                                                    | `comm.barrier()?`                                       | Returns `Result<()>` in ferrompi                       |
| `comm.immediate_barrier()`                                          | `comm.ibarrier()?`                                      | Returns `Request`                                      |

### 2b. Point-to-Point

In rsmpi, send and receive are called on `Process` handles obtained from
`comm.process_at_rank(dest)` or `comm.any_process()`. In ferrompi, `dest`,
`source`, and `tag` are plain `i32` parameters on `Communicator`. Use
`MPI_ANY_SOURCE` (`-1`) and `MPI_ANY_TAG` (`-1`) as constants.

| rsmpi expression                                                 | ferrompi expression                                       | Notes                                             |
| ---------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------- |
| `comm.process_at_rank(dest).send(&buf)`                          | `comm.send(&buf, dest, tag)?`                             | Tag is explicit in ferrompi                       |
| `comm.process_at_rank(dest).send_with_tag(&buf, tag)`            | `comm.send(&buf, dest, tag)?`                             |                                                   |
| `comm.process_at_rank(dest).synchronous_send(&buf)`              | `comm.ssend(&buf, dest, tag)?`                            | Synchronous mode                                  |
| `comm.process_at_rank(dest).buffered_send(&buf)`                 | `comm.bsend(&buf, dest, tag)?`                            | Requires buffer attached via `Mpi::buffer_attach` |
| `comm.process_at_rank(dest).ready_send(&buf)`                    | `comm.rsend(&buf, dest, tag)?`                            | Receiver must have posted recv first              |
| `comm.any_process().receive_into(&mut buf)`                      | `comm.recv(&mut buf, -1, -1)?`                            | Returns `(source, tag, count)` tuple              |
| `comm.process_at_rank(src).receive_into_with_tag(&mut buf, tag)` | `comm.recv(&mut buf, src, tag)?`                          |                                                   |
| `comm.process_at_rank(dest).immediate_send(&buf)`                | `comm.isend(&buf, dest, tag)?`                            | Returns `Request`                                 |
| `comm.process_at_rank(src).immediate_receive_into(&mut buf)`     | `comm.irecv(&mut buf, src, tag)?`                         | Returns `Request`                                 |
| `comm.send_receive(sendbuf, dest, recvbuf, src)`                 | `comm.sendrecv(&send, dest, stag, &mut recv, src, rtag)?` |                                                   |
| `comm.process_at_rank(src).probe()`                              | `comm.probe::<T>(src, tag)?`                              | Returns `Status`; blocks                          |
| `comm.process_at_rank(src).immediate_probe()`                    | `comm.iprobe::<T>(src, tag)?`                             | Returns `Option<Status>`                          |

### 2c. Groups and Communicators

In rsmpi, groups are obtained via `comm.group()` and set operations are free
functions or methods on the `Group` type. ferrompi uses the same structure but
exposes everything as inherent methods on `Group` and `Communicator`. All
group-creating calls return owned `Group` values that free their handle on drop.

| rsmpi expression                             | ferrompi expression                       | Notes                                        |
| -------------------------------------------- | ----------------------------------------- | -------------------------------------------- |
| `comm.rank()`                                | `comm.rank()`                             | ferrompi returns `i32` directly (not `Rank`) |
| `comm.size()`                                | `comm.size()`                             | Returns `i32`                                |
| `comm.group()`                               | `comm.group()?`                           | Returns `Result<Group>`                      |
| `group.size()`                               | `group.size()?`                           | Returns `Result<i32>`                        |
| `group.rank()`                               | `group.rank()?`                           | Returns `Result<i32>`                        |
| `mpi::topology::Group::union(&a, &b)`        | `a.union(&b)?`                            | Returns new `Group`                          |
| `mpi::topology::Group::intersection(&a, &b)` | `a.intersection(&b)?`                     | Returns new `Group`                          |
| `mpi::topology::Group::difference(&a, &b)`   | `a.difference(&b)?`                       | Returns new `Group`                          |
| `group.include(&ranks)`                      | `group.include(&ranks)?`                  | Subgroup from rank list                      |
| `group.exclude(&ranks)`                      | `group.exclude(&ranks)?`                  | Subgroup excluding ranks                     |
| `group.translate_rank(rank, &other)`         | `group.translate_ranks(&[rank], &other)?` | Returns `Vec<Option<i32>>`                   |
| `comm.duplicate()`                           | `comm.duplicate()?`                       | Returns `Result<Communicator>`               |
| `comm.split_by_color(color)`                 | `comm.split(color, key)?`                 | Returns `Result<Option<Communicator>>`       |
| `comm.split_by_color_with_key(color, key)`   | `comm.split(color, key)?`                 | Same call                                    |
| `comm.split_shared(key)`                     | `comm.split_shared()?`                    | Shared-memory subcommunicator                |
| `comm.split_by_subgroup_collective(&group)`  | `comm.create_from_group(&group)?`         | Returns `Option<Communicator>`               |
| `Mpi::create_from_group(group, tag)`         | `mpi.create_from_group(&group, tag)?`     | Called on `Mpi` handle in ferrompi           |

---

## 3. Migration Cookbook

The three examples below show the most common migration patterns. Each pair shows
the rsmpi version on the left and the ferrompi equivalent on the right.

### 3a. Hello World with Allreduce

The classic rsmpi README example: initialise, get the world communicator, do an
all-reduce.

<table>
<tr>
<td><strong>rsmpi</strong></td>
<td><strong>ferrompi</strong></td>
</tr>
<tr>
<td>

```rust,ignore
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();

    // all_reduce_into requires Buffer impls
    // and a process-derived operation object
    let local = rank as f64;
    let mut global = 0.0f64;
    world.all_reduce_into(
        &local,
        &mut global,
        &mpi::collective::SystemOperation::sum(),
    );

    if rank == 0 {
        println!("sum = {}", global);
    }
    // universe dropped → MPI_Finalize
}
```

</td>
<td>

```rust
use ferrompi::{Mpi, ReduceOp};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();

    let rank = world.rank();

    // allreduce_scalar takes a plain value,
    // no buffer wrapper needed
    let local = rank as f64;
    let global = world.allreduce_scalar(
        local,
        ReduceOp::Sum,
    )?;

    if rank == 0 {
        println!("sum = {}", global);
    }
    // mpi dropped → MPI_Finalize
    Ok(())
}
```

</td>
</tr>
</table>

Key differences: ferrompi propagates errors via `?` instead of panicking; the
`allreduce_scalar` convenience method eliminates the send/receive buffer pair for
scalar reductions; `ReduceOp::Sum` is a simple enum variant rather than a trait
object.

### 3b. Nonblocking Send and Receive with Wait

rsmpi's `WaitGuard` wraps a `Request` and calls `MPI_Wait` on drop, making wait
implicit. ferrompi's `Request::wait` is explicit and consumes the request by value,
giving the compiler a clear lifetime for the in-flight buffer.

<table>
<tr>
<td><strong>rsmpi</strong></td>
<td><strong>ferrompi</strong></td>
</tr>
<tr>
<td>

```rust,ignore
use mpi::traits::*;
use mpi::request::WaitGuard;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    if rank == 0 {
        let data = vec![1.0f64, 2.0, 3.0];
        // WaitGuard forces MPI_Wait on drop
        let _guard = WaitGuard::from(
            world.process_at_rank(1)
                 .immediate_send(&data[..]),
        );
        // _guard dropped here → implicit wait
    } else {
        let mut buf = vec![0.0f64; 3];
        // Scope-based lifetime on the request
        mpi::request::scope(|scope| {
            let req = world
                .any_process()
                .immediate_receive_into(scope, &mut buf[..]);
            req.wait_without_status();
        });
        println!("rank 1 received {:?}", buf);
    }
}
```

</td>
<td>

```rust,no_run
use ferrompi::Mpi;

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();

    if rank == 0 {
        let data = vec![1.0f64, 2.0, 3.0];
        // isend returns a Request; wait is explicit
        let req = world.isend(&data, 1, 0)?;
        // Do other work here if desired ...
        req.wait()?; // consumes the Request
    } else {
        let mut buf = vec![0.0f64; 3];
        // No scope wrapper needed
        let req = world.irecv(&mut buf, 0, 0)?;
        req.wait()?;
        println!("rank 1 received {:?}", buf);
    }
    Ok(())
}
```

</td>
</tr>
</table>

Key differences: ferrompi does not use lifetime scopes or guard wrappers for
nonblocking operations — `Request::wait()` takes ownership and the buffer is safe
to read afterwards. A `Request` dropped without calling `wait()` will **block in Drop**
until the operation completes — the handle is not leaked, but the
implicit wait may be surprising in latency-sensitive code, and can
deadlock if the peer never matches the send/recv. For control flow
that bypasses `wait()` (e.g., an early-return `?` on an error path,
or a panic unwind), prefer explicit `wait()` or `test()` to keep
failure modes observable. A future ferrompi version is expected to
attempt `MPI_Cancel` from `Drop` to make this safer; until then,
assume `Drop` blocks.

### 3c. Persistent Broadcast Loop

rsmpi has no persistent collectives. The closest equivalent is a nonblocking
`immediate_broadcast_into` inside a loop. ferrompi's `bcast_init` initialises the
operation once and reuses it across thousands of iterations, amortising the setup
overhead. This is the primary use case that motivates ferrompi's existence.

<table>
<tr>
<td><strong>rsmpi (no persistent — uses nonblocking loop)</strong></td>
<td><strong>ferrompi (persistent collective)</strong></td>
</tr>
<tr>
<td>

```rust,ignore
use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let mut data = vec![0.0f64; 1000];

    for iter in 0..10_000 {
        if rank == 0 {
            data.fill(iter as f64);
        }
        // Re-enters MPI scheduler every iteration;
        // no amortisation of setup cost
        mpi::request::scope(|scope| {
            let req = world
                .process_at_rank(0)
                .immediate_broadcast_into(scope, &mut data[..]);
            req.wait_without_status();
        });
        // use data ...
    }
}
```

</td>
<td>

```rust,no_run
use ferrompi::Mpi;

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let rank = world.rank();
    let mut data = vec![0.0f64; 1000];

    // Initialise ONCE — MPI_Bcast_init called here
    let mut bcast = world.bcast_init(&mut data, 0)?;

    for iter in 0..10_000 {
        if rank == 0 {
            data.fill(iter as f64);
        }
        // start/wait reuse the initialised request
        bcast.start()?;
        bcast.wait()?;
        // use data ...
    }
    // bcast dropped → MPI_Request_free
    Ok(())
}
```

</td>
</tr>
</table>

Key differences: `bcast_init` calls `MPI_Bcast_init` once; each `start`/`wait` pair
calls `MPI_Start`/`MPI_Wait` without reinitialising the request. On MPICH 4.2 and
OpenMPI 5.0 this reduces per-iteration latency by 10–30% for small messages. The
`PersistentRequest` is freed automatically when it goes out of scope.

---

## 4. Features Not Supported in ferrompi

The following rsmpi features are absent from ferrompi. For each, a brief reason is
given. Features marked "out of scope for v0.x" may be added in a future epic if user
demand materialises and a corresponding ticket is opened on the issue tracker.

### Dynamic Process Management

`MPI_Comm_spawn`, `MPI_Comm_spawn_multiple`, `MPI_Comm_accept`,
`MPI_Comm_connect`, `MPI_Open_port`, and `MPI_Close_port` are not exposed in
ferrompi. The C wrapper (`csrc/ferrompi.c`) does not include shims for these
functions. Reason: dynamic process management requires intercommunicators (see
below), which are also absent. The two features are co-dependent. Out of scope for
v0.x.

### MPI I/O (File Operations)

The `MPI_File_*` family — `MPI_File_open`, `MPI_File_read`, `MPI_File_write`,
`MPI_File_seek`, `MPI_File_close`, and all collective I/O variants — is not exposed.
Reason: parallel file I/O is a large, self-contained feature area that overlaps with
HDF5 and NetCDF tooling. Most ferrompi users target in-memory computation. Out of
scope for v0.x.

### Intercommunicators

`MPI_Intercomm_create` and `MPI_Intercomm_merge` are not exposed. ferrompi only
supports intracommunicators — communicators where all processes are members of a
single group. Reason: intercommunicators are required for dynamic process
management and are rarely used in tightly-coupled scientific computations. The
ferrompi type system does not have an `InterCommunicator` type. Out of scope for
v0.x.

### Topology Communicators — Partial Support

rsmpi exposes `MPI_Cart_create`, `MPI_Graph_create`, and their associated query
functions through the `CartesianCommunicator` and related types. ferrompi exposes
`Communicator::topology` and `Communicator::split_type` but does NOT expose
Cartesian or graph topology constructors (`MPI_Cart_create`, `MPI_Cart_coords`,
`MPI_Cart_shift`, `MPI_Graph_create`, `MPI_Dist_graph_create`). What ferrompi does
provide via `src/topology.rs` is:

- `Communicator::topology` — collective query returning `TopologyInfo` (rank-to-host
  mapping, MPI library version, SLURM metadata)
- `Communicator::split_type` — wraps `MPI_Comm_split_type` (e.g. `MPI_COMM_TYPE_SHARED`)
- `Communicator::split_shared` — convenience wrapper for shared-memory subcommunicators

Cartesian and graph topology constructors are out of scope for v0.x.

### Non-Blocking I/O

Non-blocking file operations (`MPI_File_iread`, `MPI_File_iwrite`) are not exposed,
as MPI I/O itself is absent. This category is listed for completeness.

---

## 5. API Ergonomic Differences

These are the non-obvious differences a migrant will encounter, beyond the function
renaming captured in the mapping tables.

### No `Equivalence` Trait — Use `MpiDatatype` or `CustomDatatype`

rsmpi's `Equivalence` derive macro allows arbitrary structs to be used in MPI
operations:

```rust,ignore
// rsmpi
#[derive(Equivalence)]
struct Particle { x: f64, y: f64, mass: f32 }
```

ferrompi uses the sealed `MpiDatatype` trait, which is implemented only for
primitive types: `f32`, `f64`, `i32`, `i64`, `u8`, `u32`, and `u64`. Custom
compound types have two routes:

- **`CustomDatatype`**: Use the `DatatypeBuilder` API (epic 6) to describe the
  layout of a struct and commit it to a `CustomDatatype`. Pass it to `send_custom`,
  `recv_custom`, `isend_custom`, `irecv_custom`.
- **`BytePermutable`**: For types that are safe to transmit as raw bytes with no
  padding. Implement this sealed trait (gated by an `unsafe impl`) and use
  `allreduce_bytes`.

Neither path is as ergonomic as a derive macro. The limitation is intentional: the
sealed trait guarantees that every type used in a generic MPI call has a known,
stable ABI without runtime inspection.

### No Buffer Wrappers

rsmpi requires callers to call `.buffer()` on slices before passing them to most
collective and point-to-point methods:

```rust,ignore
// rsmpi — .buffer() and .buffer_mut() are required
world.all_reduce_into(&send.buffer(), &mut recv.buffer_mut(), &op);
```

ferrompi takes `&[T]` and `&mut [T]` directly. No wrapping is needed:

```rust,ignore
// ferrompi — plain slice references
world.allreduce(&send, &mut recv, op)?;
```

### Thread Safety — `Send + Sync` vs `!Send`

rsmpi communicators are `!Send` and `!Sync`. They cannot be moved across thread
boundaries, making hybrid MPI+threads programs awkward. ferrompi's `Communicator`
is `Send + Sync`, enabling patterns such as:

```rust,ignore
use ferrompi::{Mpi, ReduceOp};
use std::thread;

let mpi = Mpi::init_thread(ferrompi::ThreadLevel::Multiple)?;
let world = mpi.world();

// Communicator can be cloned and sent to a thread
let world2 = world.clone();
thread::spawn(move || {
    let _ = world2.allreduce_scalar(1.0f64, ReduceOp::Sum);
});
```

The thread-safety guarantee is conditional on the MPI runtime supporting the
requested thread level. Use `Mpi::init_thread(ThreadLevel::Multiple)` for full
concurrent access. See `README.md` and `docs/architecture.md` for the thread-level
matrix.

### Error Handling — `Result` vs Panic

rsmpi panics on MPI errors by default. ferrompi returns `Result<T, ferrompi::Error>`
on every fallible call. `ferrompi::Error::Mpi` carries four fields:

```rust,ignore
Error::Mpi { class, code, message, operation }
```

where `class` is a `MpiErrorClass` enum, `code` is the raw MPI error integer,
`message` is the string from `MPI_Error_string`, and `operation` is the ferrompi
function name that failed (e.g., `"allreduce"`). This means that migrated code must
handle errors explicitly — typically with `?` in `fn main() -> ferrompi::Result<()>`
— instead of relying on process termination.

### RAII Drop — No Manual Cleanup Needed

rsmpi requires explicit calls to free some objects. ferrompi implements `Drop` on
all handle-owning types:

| Type                | What `Drop` calls                              |
| ------------------- | ---------------------------------------------- |
| `Communicator`      | `MPI_Comm_free`                                |
| `Group`             | `MPI_Group_free`                               |
| `Request`           | `MPI_Request_free` (if not consumed by `wait`) |
| `PersistentRequest` | `MPI_Request_free`                             |
| `Win<T>`            | `MPI_Win_free`                                 |
| `CustomDatatype`    | `MPI_Type_free`                                |
| `UserOp<T>`         | `MPI_Op_free`                                  |

Migrated code should remove any manual teardown calls — they will double-free if
left in place.

---

## 6. Migration Checklist

Use this checklist when porting a crate from rsmpi to ferrompi. The function
mapping in [Section 2](#2-function-for-function-mapping) is the authoritative
conversion reference.

- [ ] Add `ferrompi = "0.2"` to `Cargo.toml`; remove `mpi` dependency.
- [ ] Add `features = ["rma"]` if using shared memory windows; add `features =
["numa"]` for SLURM helpers.
- [ ] Replace `use mpi::traits::*` with `use ferrompi::{Mpi, ReduceOp}` (and
      other types as needed).
- [ ] Replace `mpi::initialize()` with `Mpi::init()?`; change `fn main()` to
      return `ferrompi::Result<()>`.
- [ ] Replace `universe.world()` with `mpi.world()`. The `universe`/`Mpi` binding
      must remain in scope for the lifetime of the program (dropping it calls
      `MPI_Finalize`).
- [ ] Replace `comm.process_at_rank(r).broadcast_into(...)` and similar
      root-taking calls using the Collectives table in
      [Section 2a](#2a-collectives).
- [ ] Replace `comm.all_reduce_into(...)` and symmetric collectives using the same
      table.
- [ ] Replace `comm.process_at_rank(dest).send(...)` and point-to-point calls
      using the P2P table in [Section 2b](#2b-point-to-point).
- [ ] Note: `world.barrier()` now returns `Result<()>` — add `?`.
- [ ] Replace `WaitGuard` wrapping with explicit `request.wait()?`. Verify that
      the buffer is not accessed between the `isend`/`irecv` call and the
      `wait()` call.
- [ ] Replace `Equivalence`-derived types with `CustomDatatype` or
      `BytePermutable` (see [Section 5](#5-api-ergonomic-differences)).
- [ ] Remove manual `MPI_*_free` calls — ferrompi `Drop` handles cleanup.
- [ ] If the project used `spawn`, `Intercomm`, or `MPI_File_*`, check the
      [Section 4](#4-features-not-supported-in-ferrompi) list before starting the
      migration.
- [ ] Run `cargo clippy` and address any type errors; the sealed `MpiDatatype`
      constraint will surface incompatible types at compile time.
