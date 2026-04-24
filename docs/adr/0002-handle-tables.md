# ADR 0002: Handle-Table Concurrency Strategy (Request Table)

## Context

ferrompi's C wrapper layer in `csrc/ferrompi.c` maintains four fixed-size static
tables for MPI handles:

| Table           | Capacity | Element type  | Occupancy tracking        |
| --------------- | -------- | ------------- | ------------------------- |
| `comm_table`    | 256      | `MPI_Comm`    | `MPI_COMM_NULL` sentinel  |
| `request_table` | 16384    | `MPI_Request` | `int request_used[16384]` |
| `win_table`     | 256      | `MPI_Win`     | `int win_used[256]`       |
| `info_table`    | 64       | `MPI_Info`    | `int info_used[64]`       |

The `request_table` is by far the most active: every non-blocking send, receive,
and collective operation allocates a slot via `alloc_request` and later frees it
via `free_request`. At line 99-111, `alloc_request` performs a scan starting from
`next_request_hint`, finds the first slot where `request_used[idx] == 0`, writes
`request_table[idx] = req`, and sets `request_used[idx] = 1`. `free_request` at
lines 122-127 simply writes `request_used[handle] = 0` after nulling the slot.
`get_request_ptr` at lines 114-119 reads `request_used[handle]` as a validity
guard before returning a pointer into the table.

ferrompi exposes `ThreadLevel::Multiple`, which maps to `MPI_THREAD_MULTIPLE`.
Under this thread level, multiple threads within the same process may call any
ferrompi wrapper function concurrently — including `ferrompi_isend`,
`ferrompi_irecv`, and the wait/test family. Every one of these calls reaches
`alloc_request` or `free_request` or `get_request_ptr`. The read-modify-write
on `request_used[idx]` in `alloc_request` is not protected by any synchronization
primitive: two threads may simultaneously observe `request_used[idx] == 0`, both
execute the CAS-equivalent sequence non-atomically, both claim the same slot, and
the second `request_table[idx] = req` silently clobbers the first — a lost
request with no error signal. ThreadSanitizer would flag this as a data race.

The `comm_table` allocation at lines 87-96 has the same structural race via its
`MPI_COMM_NULL` sentinel, and `win_table`/`info_table` share the same `int used[]`
pattern. **The scope of this ADR and its implementing ticket (ticket-023) is the
request table only.** The comm/win/info tables are addressed in later epics per
the master plan's Confirmed Decision 3.

Current state (verified 2026-04-24):

- No `<stdatomic.h>` include, no `_Atomic` qualifier, no `atomic_*` call in
  either `csrc/ferrompi.c` or `csrc/ferrompi.h`.
- No `<pthread.h>` include and no pthread symbol in the wrapper layer.
- `build.rs` passes no explicit `-std=` flag to the C compiler; GCC and Clang
  both default to at least C11 feature availability on all CI targets
  (MPICH/GCC, Open MPI/GCC, Cray/GCC backend).

## Decision Drivers

1. **Correctness under `MPI_THREAD_MULTIPLE`** — the data race on `request_used`
   must be eliminated; this is a must-have.
2. **Minimum review surface** — the ferrompi C layer is intentionally small; every
   C line is a potential security-audit target. New code must be easy to verify
   correct by inspection.
3. **Zero external dependencies** — no new linker requirements, no new `build.rs`
   flags, no compiler-version workarounds specific to Cray MPT.
4. **Read-path latency on `get_request_ptr`** — this function is called from every
   `wait`, `test`, and `cancel` invocation; serializing it under a global lock
   is unacceptable for multi-threaded producer workloads.
5. **Portability** — the solution must compile and run correctly under MPICH's
   `mpicc` (GCC), Open MPI's `mpicc` (GCC), and Cray's `cc` (GCC or Cray-ICC
   backend), which are the three compiler environments present in ferrompi's CI.

## Options Considered

### Option A: C11 atomics with CAS

Replace `int request_used[MAX_REQUESTS]` with
`atomic_int request_used[MAX_REQUESTS]` (from `<stdatomic.h>`). Replace the
non-atomic read-modify-write in `alloc_request` with a
`atomic_compare_exchange_strong_explicit` call using `memory_order_acq_rel` on
success; replace the plain store in `free_request` with
`atomic_store_explicit(..., memory_order_release)`; replace the plain load in
`get_request_ptr` with `atomic_load_explicit(..., memory_order_acquire)`.
Additionally change `next_request_hint` to `atomic_int` and update it with
`memory_order_relaxed` stores — the hint is advisory, not correctness-critical.

Pros:

- Minimal diff: approximately 20 lines changed in a single file, no new data
  structures introduced.
- No ABA problem: slots are indexed integers, not pointer-linked nodes. Two
  threads racing on the same index produce a deterministic outcome — exactly one
  CAS succeeds and claims the slot; the loser moves to the next index.
- The `get_request_ptr` read path reduces to a single acquire atomic load — no
  lock acquisition, no memory bus arbitration beyond cache-line coherence.
- `<stdatomic.h>` is part of the C11 standard library. No new dependency, no
  `build.rs` flag, no Cray workaround required.
- O(1) amortized allocation via the hint, O(N) worst case when the table is nearly
  full. With `MAX_REQUESTS = 16384` and typical in-flight request counts in the
  low hundreds, even a complete linear scan is measured in microseconds.

Cons:

- Requires GCC >= 4.9 or Clang >= 3.3 for lock-free `atomic_int` on x86-64.
  Both compilers have met this bar for over a decade; all three CI compiler
  environments satisfy it.
- `memory_order_acq_rel` on the CAS and `memory_order_acquire` on the load are
  stronger than `relaxed`, which adds a minor overhead on weakly-ordered
  architectures. x86-64 (TSO) makes this overhead negligible; the correctness
  guarantee is universal.

### Option B: Single pthread mutex

Introduce `static pthread_mutex_t request_table_mutex = PTHREAD_MUTEX_INITIALIZER`
and acquire it at the top of `alloc_request`, `free_request`, and `get_request_ptr`.
Requires `#include <pthread.h>`.

Pros:

- Trivially correct: the mutex makes all table operations sequentially consistent
  by construction. No atomic memory-order reasoning required.

Cons:

- Fails decision driver 4: `get_request_ptr` is on the hot path for every wait
  and test operation. Under a multi-threaded producer workload — the exact
  scenario motivating this hardening — every thread contends for the same mutex
  on every MPI call, serializing what would otherwise be independent operations
  on distinct slots.
- Fails decision driver 3: `<pthread.h>` is a new dependency for the wrapper
  layer. Although MPICH and Open MPI transitively link `-lpthread`, Cray MPT
  environments do not guarantee the same symbol availability without explicit
  `build.rs` changes.

### Option C: Lock-free free-list (Treiber stack)

Maintain an `atomic_int free_head` pointing to the top of a LIFO stack of free
slot indices. Each free slot stores the index of the next free slot in the
`request_used` array (repurposed as a next-pointer). `alloc_request` pops from
the head with a CAS; `free_request` pushes with a CAS.

To mitigate the ABA problem — where a thread reads `head`, another thread pops
and then pushes the same node before the first thread's CAS, leaving `head`
numerically unchanged but the list semantically mutated — a generational counter
must be packed alongside the index. This requires either a double-wide CAS (not
universally available) or a 64-bit word with a 32-bit index and a 32-bit
generation counter.

Pros:

- True O(1) allocation regardless of table occupancy.
- Lock-free under contention: no thread is ever blocked waiting for another.

Cons:

- Fails decision driver 2: ABA handling requires manual bit-packing of a
  generational counter into a 64-bit atomic word, or reliance on a
  double-wide CAS. Either path adds substantial code complexity and a
  correctness argument that cannot be verified by inspection alone.
- Fails decision driver 5: Cray's older ICC-derived toolchains have incomplete
  support for `_Atomic` on compound types and double-wide atomics. Packing a
  generational counter requires either compiler-specific intrinsics or a struct
  with `_Atomic` qualifier, both of which are unreliable on these targets.
- The O(1) advantage over Option A is not relevant to ferrompi's workload:
  in-flight request counts are in the low hundreds, making even a linear scan
  of `MAX_REQUESTS = 16384` amortized O(1) in practice.

## Decision

**Option A (C11 atomics with CAS) is chosen.**

Rationale against the decision drivers:

- Driver 1: The `atomic_compare_exchange_strong_explicit` CAS in `alloc_request`
  is the textbook-correct mechanism for slot-claim races. Exactly one thread
  succeeds per index per CAS; losers advance to the next slot. No slot is ever
  double-claimed.
- Driver 2: The total diff is approximately 20 lines in `csrc/ferrompi.c` — type
  changes on two variables, three modified function bodies, and one additional
  `#include`. There are no new data structures, no new call patterns, and no
  algorithmic changes. The correctness argument fits in a paragraph.
- Driver 3: `<stdatomic.h>` is part of ISO C11. No `build.rs` change, no linker
  flag, and no compiler-version guard is required on any of the three CI targets.
- Driver 4: The read path in `get_request_ptr` becomes a single
  `atomic_load_explicit(..., memory_order_acquire)` — no mutex acquisition, no
  global serialization. Under `MPI_THREAD_MULTIPLE` with multiple concurrent
  waiters on distinct requests, each thread reads its own slot without
  contending with others.
- Driver 5: GCC >= 4.9 and Clang >= 3.3 have provided lock-free `atomic_int`
  on x86-64 for over a decade. All three CI environments (MPICH/GCC, Open
  MPI/GCC, Cray/GCC) satisfy this requirement.

Option B is rejected because its global mutex serializes the hot `get_request_ptr`
path (driver 4) and introduces a pthread dependency (driver 3).

Option C is rejected because ABA handling introduces code complexity that cannot
be verified by inspection (driver 2) and relies on wide-atomic support that is
incomplete on Cray's older toolchains (driver 5). Its O(1) allocation advantage
provides no measurable benefit at ferrompi's target workload scale.

## Consequences

The following eight steps constitute the complete implementation specification for
ticket-023. Every file path, line number, and function name refers to
`csrc/ferrompi.c` as it exists at the time this ADR was accepted (commit 06b4044).

**Step 1 — Add `<stdatomic.h>`**

Insert `#include <stdatomic.h>` directly after `#include <stdlib.h>` at line 10.
No other header or `build.rs` change is required.

**Step 2 — Change `request_used` type**

At line 39, change:

```c
static int request_used[MAX_REQUESTS];  // 1 if slot is in use
```

to:

```c
static atomic_int request_used[MAX_REQUESTS];  // 1 if slot is in use
```

**Step 3 — Change `next_request_hint` type**

At line 40, change:

```c
static int next_request_hint = 0;
```

to:

```c
static atomic_int next_request_hint = 0;
```

**Step 4 — Update `init_tables`**

In `init_tables` (lines 55-76), replace the request-table initialisation loop:

```c
for (int i = 0; i < MAX_REQUESTS; i++) {
    request_table[i] = MPI_REQUEST_NULL;
    request_used[i] = 0;
}
```

with:

```c
for (int i = 0; i < MAX_REQUESTS; i++) {
    request_table[i] = MPI_REQUEST_NULL;
    atomic_init(&request_used[i], 0);
}
atomic_init(&next_request_hint, 0);
```

`atomic_init` is the correct way to initialise a statically-allocated `atomic_int`
that was not given a constant initialiser (required because `atomic_int` is not
necessarily assignment-compatible with plain `int` on all C11 implementations).

**Step 5 — Rewrite `alloc_request`**

Replace `alloc_request` (lines 99-111) with:

```c
static int64_t alloc_request(MPI_Request req) {
    int hint = atomic_load_explicit(&next_request_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_REQUESTS; i++) {
        int idx = (hint + i) % MAX_REQUESTS;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &request_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            request_table[idx] = req;
            atomic_store_explicit(&next_request_hint,
                (idx + 1) % MAX_REQUESTS, memory_order_relaxed);
            return (int64_t)idx;
        }
    }
    return -1;  /* No space */
}
```

Key memory-ordering decisions:

- The CAS uses `memory_order_acq_rel` on success so that the subsequent
  `request_table[idx] = req` store is visible to any thread that later
  acquires the slot through `get_request_ptr`.
- The hint load and store are `memory_order_relaxed` because the hint is
  advisory: an inaccurate hint never produces an incorrect result, only
  a slightly longer scan.

**Step 6 — Rewrite `free_request`**

Replace the body of `free_request` (lines 122-127) with:

```c
static void free_request(int64_t handle) {
    if (handle >= 0 && handle < MAX_REQUESTS) {
        request_table[handle] = MPI_REQUEST_NULL;
        atomic_store_explicit(&request_used[handle], 0, memory_order_release);
    }
}
```

The `request_table[handle] = MPI_REQUEST_NULL` plain store before the release
store is safe: the release on `request_used` establishes the happens-before
edge. Any thread that subsequently acquires the slot via `get_request_ptr`'s
acquire load on `request_used` will observe the null value in `request_table`.

**Step 7 — Rewrite `get_request_ptr`**

Replace the guard in `get_request_ptr` (lines 114-119):

```c
static MPI_Request* get_request_ptr(int64_t handle) {
    if (handle < 0 || handle >= MAX_REQUESTS ||
            atomic_load_explicit(&request_used[handle],
                                 memory_order_acquire) == 0) {
        return NULL;
    }
    return &request_table[handle];
}
```

The acquire load pairs with the release store in `free_request`, ensuring that a
thread observing `request_used[handle] != 0` also observes the `request_table`
value written by the allocating thread.

**Step 8 — Update `ferrompi_finalize`**

In `ferrompi_finalize` (lines 307-342), replace the request-cleanup loop
(lines 309-314):

```c
for (int i = 0; i < MAX_REQUESTS; i++) {
    if (request_used[i] && request_table[i] != MPI_REQUEST_NULL) {
        MPI_Request_free(&request_table[i]);
    }
    request_used[i] = 0;
}
```

with:

```c
for (int i = 0; i < MAX_REQUESTS; i++) {
    if (atomic_load_explicit(&request_used[i], memory_order_acquire) &&
            request_table[i] != MPI_REQUEST_NULL) {
        MPI_Request_free(&request_table[i]);
    }
    atomic_store_explicit(&request_used[i], 0, memory_order_release);
}
```

Finalize is called after `MPI_Finalize` is complete and no concurrent MPI
operations can be in flight, so the acquire/release here is defensive rather
than strictly necessary — but it keeps the access pattern consistent with the
rest of the implementation and avoids triggering TSan's finalizer checks.

### Follow-up: TSan Verification

Ticket-023 must add a ThreadSanitizer verification step. The preferred form is a
new script or target in `tests/` (e.g. `tests/run_tsan_tests.sh`) that compiles
`csrc/ferrompi.c` with `-fsanitize=thread` and runs a multi-threaded non-blocking
send/receive test that exercises concurrent `alloc_request` and `free_request`
calls, exiting 0 with no TSan diagnostics.

If TSan is not practical in the CI environment (e.g., Cray's system MPI
libraries are not instrumented and produce false positives), ticket-023 must
document this in a code comment adjacent to the atomic declarations and register
the multi-threaded isend scenario as a mandatory manual pre-release step in
`CONTRIBUTING.md` or a dedicated `docs/testing.md` entry.

### Out of Scope

The `comm_table`, `win_table`, and `info_table` have the same structural race
under `MPI_THREAD_MULTIPLE`. They are excluded from this ADR and from ticket-023
per the master plan's Confirmed Decision 3. A follow-up ADR will address them in
a later epic if operational evidence or a TSan run identifies them as practical
risks.

## Status

Accepted — 2026-04-24. Implemented by ticket-023.
