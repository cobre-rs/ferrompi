/* the implementation of ferrompi.h; see the header for the interface contract. */

#include "ferrompi.h"
#include <mpi.h>

#if defined(MPI_ABI_VERSION) && MPI_VERSION < 5
#error "ferrompi does not support the draft MPI ABI (MPI_ABI_VERSION with MPI_VERSION < 5, as in MPICH 4.3 built with -DMPI_ABI); build against the native MPI headers or an MPI 5.0 ABI implementation"
#endif

#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>

// Every int32_t* to int* cast below (count, displacement and rank-array
// arguments passed to the underlying MPI call, and the rank-range triples
// passed as int (*)[3]) is safe: int is at least 32 bits on all MPI
// platforms, and MPI does not write through these IN arguments.

/* ============================================================
 * Internal State and Handle Management
 * ============================================================ */

// Maximum number of concurrent communicators (excluding COMM_WORLD)
#define MAX_COMMS 256

// Maximum number of concurrent requests
#define MAX_REQUESTS 16384

// The request table tracks occupancy with a bitmap of 64-bit words (one bit
// per slot) rather than a dense array of one atomic_int per slot: allocation
// skips 64 taken slots per scanned word via a hardware find-first-zero
// rather than probing each slot. MAX_REQUESTS must stay a multiple of 64 so
// every word is full.
//
// Measured: every thread starts its scan at the same hint word and
// read-modify-writes the same 64-bit word, so the bitmap concentrates
// contention on one cache line, and a single-thread alloc+free pair costs
// about 3x the dense per-slot table it replaced.
#define REQUEST_BITS_WORDS (MAX_REQUESTS / 64)
_Static_assert(MAX_REQUESTS % 64 == 0,
               "MAX_REQUESTS must be a multiple of 64 for the occupancy bitmap");

// Element count at or below which the batch wait/test/start helpers
// (ferrompi_waitall/startall/waitany/waitsome/testany/testsome) use an
// on-stack MPI_Request / index / status scratch buffer instead of malloc,
// eliminating per-call heap traffic on the hot completion path. Batches
// larger than this fall back to a heap allocation. 64 covers the
// overwhelming majority of real request sets (halo exchange, ping-pong,
// neighbour collectives) while keeping the worst-case stack footprint small
// (64 * sizeof(MPI_Request) plus, for waitall, waitsome and testsome,
// 64 * sizeof(MPI_Status), and for waitsome/testsome, 64 * sizeof(int)).
#define FERROMPI_REQ_STACK 64

// Open MPI 5 implements the MPI 4.0 persistent collectives and
// MPI_Comm_create_from_group while its mpi.h still reports MPI_VERSION 3.
// The _c large-count calls stay gated on MPI_VERSION >= 4: Open MPI 5 has
// none.
#if MPI_VERSION >= 4 || (defined(OMPI_MAJOR_VERSION) && OMPI_MAJOR_VERSION >= 5)
#define FERROMPI_HAVE_MPI4_COLLECTIVES 1
#endif

// Maximum number of concurrent MPI_Info objects
#define MAX_INFOS 64

// Maximum number of concurrent MPI_Group objects
#define MAX_GROUPS 64

// Maximum number of concurrent custom (derived) MPI_Datatype objects
#define MAX_DATATYPES 64

// Maximum number of concurrent RMA windows
#define MAX_WINDOWS 256

// Communicator table (index 0 is always COMM_WORLD)
// comm_used uses C11 atomics to eliminate data races under MPI_THREAD_MULTIPLE.
// alloc_comm uses a CAS loop (skipping slot 0, which is permanently MPI_COMM_WORLD);
// ferrompi_comm_free uses atomic_store (release); get_comm uses atomic_load (acquire)
// for slots other than 0.  Mirrors the win_used/alloc_win pattern.  See
// docs/adr/0002-handle-tables.md for the full rationale.
static MPI_Comm comm_table[MAX_COMMS];
static atomic_int comm_used[MAX_COMMS];  // 1 if slot is in use
static atomic_int next_comm_hint;

// Request table
// Occupancy is a bitmap of 64-bit words (request_bits): bit b of word w marks
// slot w*64+b as in use.  A request handle is (generation << 32) | slot: the
// per-slot generation in request_gen is bumped by free_request each time a
// slot is freed, so a handle left over from a slot's prior occupant fails the
// generation check in request_slot instead of aliasing whatever now occupies
// the reused slot.  alloc_request claims a free bit with an acq_rel fetch_or,
// free_request clears it with a release fetch_and, and request_slot tests it
// with an acquire load.  next_request_hint is an advisory *word* index
// (0..REQUEST_BITS_WORDS-1) so scans start near the last success.  See
// docs/adr/0002-handle-tables.md for the full rationale.
static MPI_Request request_table[MAX_REQUESTS];
static _Atomic(uint64_t) request_bits[REQUEST_BITS_WORDS];  // bit set => slot in use
// Per-slot generation counter, masked to 31 bits so (generation << 32) | slot
// never sets the sign bit of the int64_t handle. Initialised by
// init_tables; bumped only by free_request, before its release fetch_and.
// Read by alloc_request (to mint the next handle) and request_slot (to
// validate one).
static _Atomic(uint32_t) request_gen[MAX_REQUESTS];
#define REQUEST_GEN_MASK 0x7fffffffu
static atomic_int next_request_hint;  // advisory start word for the next scan

// Per-slot request-kind/activity state, written only by the owning thread
// (as request_table itself is): REQUEST_PERSISTENT marks a slot registered
// by a persistent initiator; REQUEST_ACTIVE marks a persistent request
// active from a successful MPI_Start, or from being passed to
// MPI_Startall (whatever that call returns, since a partial failure can
// leave some of them already started), until MPI reports it complete. A
// registered nonblocking (non-persistent) request is always active and
// carries no REQUEST_ACTIVE bit of its own. Plain bytes, not atomics:
// distinct slots are distinct objects, and ferrompi_finalize — the one
// reader outside the owning thread's own calls — relies on the caller
// having ordered every other thread's request calls before Mpi is
// dropped (joined or equivalent); it does not itself order against them.
static uint8_t request_state[MAX_REQUESTS];
#define REQUEST_PERSISTENT 1
#define REQUEST_ACTIVE     2

// Window table
// win_used uses C11 atomics to eliminate data races under MPI_THREAD_MULTIPLE.
// alloc_win uses a CAS loop; free_win uses atomic_store (release); readers use
// atomic_load (acquire).  Mirrors the comm_used/alloc_comm pattern.  (Unlike
// the request table, the small fixed tables — windows, comms, datatypes, ops,
// groups, infos — keep the dense per-slot used-flag scan.)
static MPI_Win win_table[MAX_WINDOWS];
static atomic_int win_used[MAX_WINDOWS];  // 1 if slot is in use
static atomic_int next_win_hint;

// Info table
// info_used uses C11 atomics for the same reason as win_used above.
static MPI_Info info_table[MAX_INFOS];
static atomic_int info_used[MAX_INFOS];  // 1 if slot is in use
static atomic_int next_info_hint;

// Group table (slot 0 reserved for MPI_GROUP_EMPTY, lazily populated)
// group_used uses C11 atomics for the same reason as win_used above.
static MPI_Group group_table[MAX_GROUPS];
static atomic_int group_used[MAX_GROUPS];  // 1 if slot is in use
static atomic_int next_group_hint;

// Custom (derived) datatype table
// datatype_used uses C11 atomics for the same reason as win_used above.
static MPI_Datatype datatype_table[MAX_DATATYPES];
static atomic_int datatype_used[MAX_DATATYPES];  // 1 if slot is in use
static atomic_int next_datatype_hint;

// Maximum number of concurrent user-defined MPI_Op objects
#define MAX_OPS 16

// User-defined op table.  _Atomic(MPI_Op) provides happens-before ordering for
// the MPI_Op payload across the four access sites:
//   - WRITER ferrompi_op_create_user:
//       atomic_store_explicit(release) after MPI_Op_create.
//   - WRITER free_op_slot:
//       atomic_store_explicit(release) of MPI_OP_NULL.
//   - WRITER ferrompi_op_free (also called per-slot by the ferrompi_finalize
//     sweep):
//       stages through a local; loads (acquire), invokes MPI_Op_free
//       on the local, stores (release) the result back.
//   - READER ferrompi_allreduce_user_op:
//       atomic_load_explicit(acquire) before the MPI_Allreduce call.
// The acquire/release pairing guarantees that any thread observing
// op_used[h] == 1 (via the existing op_used acquire protocol) and then
// loading op_table[h] sees either the valid post-create MPI_Op or
// MPI_OP_NULL (post-free), never a torn value.
static _Atomic(MPI_Op) op_table[MAX_OPS];
static atomic_int op_used[MAX_OPS];  // 1 if slot is in use
static atomic_int next_op_hint;

// Initialization guard.  CAS-based: exactly one thread performs the
// atomic_init loops below; concurrent callers see the CAS fail and return
// immediately.  Memory-ordering rationale: C11 §6.7.4 guarantees static
// storage duration objects are zero-initialized at program load, so the
// atomic_init loop only restates that zero state.  Losing threads return
// after observing tables_initialized == 1 and rely on the pre-load static
// zero-init as the baseline — no synchronizes-with edge is required for
// the loop body's stores because the values being written are already the
// observable state.  If any future revision ever writes a non-zero value
// in the init loop, a release fence (e.g. atomic_thread_fence(release))
// must be inserted before the CAS, or the non-zero init must move into
// each call site of get_*.
static atomic_int tables_initialized;

static void init_tables(void) {
    int expected = 0;
    if (!atomic_compare_exchange_strong_explicit(
            &tables_initialized, &expected, 1,
            memory_order_acq_rel, memory_order_acquire)) {
        return;  /* another thread already initialized */
    }

    // Note: MPI_COMM_WORLD must be set AFTER MPI_Init is called
    // comm_table[0] will be set in ferrompi_init_thread; comm_used[0] is
    // set to 1 there as well, immediately after MPI_COMM_WORLD is assigned.
    for (int i = 0; i < MAX_COMMS; i++) {
        comm_table[i] = MPI_COMM_NULL;
        atomic_init(&comm_used[i], 0);
    }
    atomic_init(&next_comm_hint, 1);  // Start scanning from slot 1 (slot 0 is COMM_WORLD)
    for (int i = 0; i < MAX_REQUESTS; i++) {
        request_table[i] = MPI_REQUEST_NULL;
        atomic_init(&request_gen[i], (uint32_t)0);
    }
    for (int w = 0; w < REQUEST_BITS_WORDS; w++) {
        atomic_init(&request_bits[w], (uint64_t)0);
    }
    atomic_init(&next_request_hint, 0);
    for (int i = 0; i < MAX_WINDOWS; i++) {
        win_table[i] = MPI_WIN_NULL;
        atomic_init(&win_used[i], 0);
    }
    atomic_init(&next_win_hint, 0);
    for (int i = 0; i < MAX_INFOS; i++) {
        info_table[i] = MPI_INFO_NULL;
        atomic_init(&info_used[i], 0);
    }
    atomic_init(&next_info_hint, 0);
    for (int i = 0; i < MAX_GROUPS; i++) {
        group_table[i] = MPI_GROUP_EMPTY;
        atomic_init(&group_used[i], 0);
    }
    atomic_init(&next_group_hint, 0);
    for (int i = 0; i < MAX_DATATYPES; i++) {
        datatype_table[i] = MPI_DATATYPE_NULL;
        atomic_init(&datatype_used[i], 0);
    }
    atomic_init(&next_datatype_hint, 0);
    for (int i = 0; i < MAX_OPS; i++) {
        atomic_init(&op_table[i], MPI_OP_NULL);
        atomic_init(&op_used[i], 0);
    }
    atomic_init(&next_op_hint, 0);
    /* No trailing tables_initialized = 1; — the CAS above already set it. */
}

// Get MPI_Comm from handle (thread-safe: acquire load pairs with the release
// store in ferrompi_comm_free).  Slot 0 (MPI_COMM_WORLD) is always valid
// post-init and does not require a comm_used check.
static MPI_Comm get_comm(int32_t handle) {
    if (handle < 0 || handle >= MAX_COMMS) {
        return MPI_COMM_NULL;
    }
    if (handle != 0 &&
            !atomic_load_explicit(&comm_used[handle], memory_order_acquire)) {
        return MPI_COMM_NULL;
    }
    return comm_table[handle];
}

// Allocate a communicator handle (thread-safe via C11 CAS, mirrors alloc_win).
// Slot 0 is permanently MPI_COMM_WORLD; we skip it explicitly so the CAS can
// never claim it.  The hint is advisory: an inaccurate hint only lengthens the
// scan, never produces an incorrect result.  acq_rel on the CAS prevents
// reordering of the slot-claim with later operations on the same thread.
//
// The subsequent comm_table[idx] write is sequenced AFTER the CAS's release,
// not before — so a thread reading comm_table[idx] after acquire-loading
// comm_used[idx] only synchronizes-with the CAS store of `1`, not the table
// payload write.  Under this implementation, callers must ensure
// happens-before is established between the allocation site and any
// cross-thread read of the handle, via an external synchronization
// mechanism (Mutex unlock/lock, channel send/recv, Arc clone+drop, etc.).
// MPI itself requires the user to coordinate communicator usage between
// threads under MPI_THREAD_MULTIPLE, so this is a strictly weaker
// constraint than the MPI standard already imposes.
//
// On TSO architectures (x86, x86_64), every plain store has implicit
// release semantics, so the gap is invisible.  On weakly-ordered
// architectures (ARM64, POWER), the gap is observable in principle but
// is closed by the external synchronization that MPI usage already
// requires.  A fully-paired pattern (write table first, then release-
// store comm_used) is used in ferrompi_comm_free for the deallocation
// side; allocation could be migrated to that pattern in a future
// revision to remove the contract-via-MPI-usage dependency.  See the
// `free` paths in this file for the canonical paired pattern.
static int32_t alloc_comm(MPI_Comm comm) {
    int hint = atomic_load_explicit(&next_comm_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_COMMS; i++) {
        int idx = (hint + i) % MAX_COMMS;
        if (idx == 0) continue;  // slot 0 is MPI_COMM_WORLD; never reuse
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &comm_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            comm_table[idx] = comm;
            atomic_store_explicit(&next_comm_hint,
                (idx + 1) % MAX_COMMS, memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Allocate a request handle (thread-safe, lock-free via the C11 occupancy
// bitmap).  Scans words from the advisory hint; within a non-full word it
// finds the lowest free slot with a hardware count-trailing-zeros and claims
// it with an acq_rel fetch_or.  fetch_or is idempotent, so a lost race (the
// returned word already had our chosen bit set) simply means another thread
// took that slot — we pick the next free bit and retry, never corrupting
// occupancy.  The hint is advisory: a stale hint only lengthens the scan.
//
// As in the prior CAS design, the subsequent request_table[idx] write is
// sequenced AFTER the claiming fetch_or, not ordered before it by the acq_rel
// release.  Handles are used same-thread (caller writes the handle, then later
// reads it via get_request_ptr); a cross-thread handle transfer must establish
// its own happens-before via the transfer mechanism (channel, Arc, etc.).
// Within that same-thread/transfer-aware contract the implementation is
// correct on x86, ARM64, and POWER.  See docs/adr/0002-handle-tables.md.
static int64_t alloc_request(MPI_Request req, int persistent) {
    unsigned hint = (unsigned)atomic_load_explicit(&next_request_hint,
                                                    memory_order_relaxed);
    for (int w = 0; w < REQUEST_BITS_WORDS; w++) {
        unsigned widx = (hint + (unsigned)w) % REQUEST_BITS_WORDS;
        uint64_t cur = atomic_load_explicit(&request_bits[widx],
                                            memory_order_relaxed);
        while (cur != UINT64_MAX) {
            int bit = __builtin_ctzll(~cur);  // lowest free slot in this word
            uint64_t mask = (uint64_t)1 << bit;
            uint64_t old = atomic_fetch_or_explicit(&request_bits[widx], mask,
                                                    memory_order_acq_rel);
            if ((old & mask) == 0) {
                int64_t idx = (int64_t)widx * 64 + bit;
                request_table[idx] = req;
                request_state[idx] = persistent ? REQUEST_PERSISTENT : 0;
                // Relaxed: sequenced after the acq_rel fetch_or above, whose
                // acquire component already synchronizes-with the release
                // fetch_and of whichever free_request last vacated this slot,
                // so this read observes that free's generation bump without
                // needing its own ordering.
                uint32_t gen = atomic_load_explicit(&request_gen[idx],
                                                    memory_order_relaxed);
                atomic_store_explicit(&next_request_hint, (int)widx,
                                      memory_order_relaxed);
                return ((int64_t)gen << 32) | idx;
            }
            cur = old;  // lost the race for that bit; retry the next free one
        }
    }
    return -1;  /* No space */
}

// Resolve a handle to its slot index (thread-safe: the acquire load of the
// occupancy bit pairs with the release fetch_and in free_request, so this
// holds for any handle whose free happens-before this lookup, whether on the
// same thread or transferred to it — see alloc_request). Returns -1 for a
// handle that is negative, out of range, names a currently-free slot, or
// carries a generation that does not match the slot's current occupant (a
// stale handle from before the slot was last freed and reused).
static int64_t request_slot(int64_t handle) {
    if (handle < 0) {
        return -1;
    }
    uint64_t slot = (uint64_t)handle & 0xffffffffu;
    if (slot >= (uint64_t)MAX_REQUESTS) {
        return -1;
    }
    unsigned widx = (unsigned)(slot / 64);
    uint64_t mask = (uint64_t)1 << (slot % 64);
    if ((atomic_load_explicit(&request_bits[widx], memory_order_acquire)
            & mask) == 0) {
        return -1;
    }
    // Relaxed: sequenced after the acquire load above, whose happens-before
    // already covers this read (see request_gen's declaration comment).
    uint32_t gen = atomic_load_explicit(&request_gen[slot], memory_order_relaxed);
    if (gen != (uint32_t)((uint64_t)handle >> 32)) {
        return -1;
    }
    return (int64_t)slot;
}

// Get MPI_Request pointer from handle. Thin wrapper over request_slot's
// generation-checked lookup.
static MPI_Request* get_request_ptr(int64_t handle) {
    int64_t slot = request_slot(handle);
    if (slot < 0) {
        return NULL;
    }
    return &request_table[slot];
}

// Free a request handle (thread-safe: the plain store to request_table and
// the generation bump both happen-before the release fetch_and that clears
// the occupancy bit, pairing with the acquire load in request_slot so any
// subsequent acquirer observes the null value and the bumped generation).
static void free_request(int64_t handle) {
    int64_t slot = request_slot(handle);
    if (slot < 0) {
        return;
    }
    request_table[slot] = MPI_REQUEST_NULL;
    // Relaxed: sequenced-before the release fetch_and below, whose release
    // covers every write (atomic or not) sequenced before it in this thread,
    // same as the plain request_table store above — no stronger order needed.
    uint32_t gen = atomic_load_explicit(&request_gen[slot], memory_order_relaxed);
    atomic_store_explicit(&request_gen[slot], (gen + 1) & REQUEST_GEN_MASK,
                          memory_order_relaxed);
    unsigned widx = (unsigned)(slot / 64);
    uint64_t mask = (uint64_t)1 << (slot % 64);
    atomic_fetch_and_explicit(&request_bits[widx], ~mask,
                              memory_order_release);
}

// Clear a persistent request's REQUEST_ACTIVE bit on completion. A stale
// or already-freed handle resolves to no slot, so this is a no-op.
static void mark_inactive(int64_t handle) {
    int64_t slot = request_slot(handle);
    if (slot < 0) {
        return;
    }
    request_state[slot] &= (uint8_t)~REQUEST_ACTIVE;
}

#if defined(MPIX_ERR_PROC_FAILED_PENDING)
#define FERROMPI_PROC_FAILED_PENDING MPIX_ERR_PROC_FAILED_PENDING
#elif defined(MPI_ERR_PROC_FAILED_PENDING)
#define FERROMPI_PROC_FAILED_PENDING MPI_ERR_PROC_FAILED_PENDING
#endif

/* Under a fault-tolerant MPI, a receive from any source can fail with
 * PROC_FAILED_PENDING while MPI still holds it pending and owns its
 * buffer. No call ferrompi exposes can complete it, so reporting it done,
 * or returning while the Rust side considers it done, would hand the
 * buffer back to the program while MPI can still write into it. */
static void abort_if_pending_after_failure(int err) {
#ifdef FERROMPI_PROC_FAILED_PENDING
    int cls;
    if (err != MPI_SUCCESS && MPI_Error_class(err, &cls) == MPI_SUCCESS
            && cls == FERROMPI_PROC_FAILED_PENDING) {
        fputs("ferrompi: receive pending after a process failure "
              "(MPI_ERR_PROC_FAILED_PENDING); fault-tolerant MPI is not "
              "supported\n", stderr);
        abort();
    }
#else
    (void)err;
#endif
}

/* Tear down an ACTIVE MPI request that could not be registered in the request
 * table (table full).  This is the failure path of every nonblocking initiator
 * (MPI_Isend/Irecv, the nonblocking collectives, and the RMA R-variants
 * Rput/Rget/Raccumulate): MPI has already started the operation against the
 * user buffer, but there is no free slot to hand a handle back to Rust.
 *
 * MPI_Request_free must NOT be used here: per MPI-3 §3.7.3 it does not cancel
 * an active operation, so the transfer would continue against the user buffer
 * after the Rust wrapper returns Err and releases the buffer borrow — a
 * use-after-free / data race.  MPI_Cancel is also unusable as a general teardown
 * here: it is erroneous for nonblocking collectives (MPI-3 §5.12) and is not
 * defined for RMA request handles, so it cannot be applied uniformly across
 * every active-request call site.
 *
 * MPI_Wait is the one teardown that is valid for every active request kind: it
 * drives the operation to local completion, after which MPI no longer touches
 * the buffer, so it is safe for the caller to drop or reuse it once the wrapper
 * returns the error.  Waiting here matches the crate's existing Drop-waits
 * philosophy (ADR-0004): blocking on completion is preferred over leaving an
 * operation in flight against a buffer the borrow checker believes is free.
 * This path is only reached at request-table saturation, an exceptional case.
 * Under a fault-tolerant MPI that wait can fail with PROC_FAILED_PENDING and
 * leave a wildcard receive pending; like every completion call, this path
 * then ends the process instead of returning while MPI owns the buffer.
 *
 * NOTE: persistent (*_init) initiators do NOT use this helper.  MPI_*_init
 * produces an INACTIVE request with no transfer in flight, for which
 * MPI_Request_free is the correct release; those sites are intentionally left
 * calling MPI_Request_free. */
static void complete_unregistered_request(MPI_Request* req) {
    abort_if_pending_after_failure(MPI_Wait(req, MPI_STATUS_IGNORE));
}

// Allocate a window handle (thread-safe via C11 CAS).
// The hint is advisory: an inaccurate hint only lengthens the scan, never
// produces an incorrect result.  acq_rel on the CAS prevents reordering of the
// slot-claim with later operations on the same thread.
static int32_t alloc_win(MPI_Win win) {
    int hint = atomic_load_explicit(&next_win_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_WINDOWS; i++) {
        int idx = (hint + i) % MAX_WINDOWS;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &win_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            win_table[idx] = win;
            atomic_store_explicit(&next_win_hint,
                (idx + 1) % MAX_WINDOWS, memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Win from handle (thread-safe: acquire load pairs with the release
// store in free_win).
static MPI_Win get_win(int32_t handle) {
    if (handle < 0 || handle >= MAX_WINDOWS ||
            !atomic_load_explicit(&win_used[handle], memory_order_acquire)) {
        return MPI_WIN_NULL;
    }
    return win_table[handle];
}

// Get MPI_Win pointer from handle (for operations that modify the win).
// Thread-safe via the same acquire/release protocol as get_win.
static MPI_Win* get_win_ptr(int32_t handle) {
    if (handle < 0 || handle >= MAX_WINDOWS ||
            !atomic_load_explicit(&win_used[handle], memory_order_acquire)) {
        return NULL;
    }
    return &win_table[handle];
}

// Free a window handle (thread-safe: plain store to win_table happens-before
// the release store to win_used, pairing with the acquire load in get_win).
static void free_win(int32_t handle) {
    if (handle >= 0 && handle < MAX_WINDOWS) {
        win_table[handle] = MPI_WIN_NULL;
        atomic_store_explicit(&win_used[handle], 0, memory_order_release);
    }
}

// Allocate an info handle (thread-safe via C11 CAS, mirrors alloc_win).
static int32_t alloc_info(MPI_Info info) {
    int hint = atomic_load_explicit(&next_info_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_INFOS; i++) {
        int idx = (hint + i) % MAX_INFOS;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &info_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            info_table[idx] = info;
            atomic_store_explicit(&next_info_hint,
                (idx + 1) % MAX_INFOS, memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Get MPI_Info from handle (thread-safe: acquire load pairs with release store
// in free_info).
static MPI_Info get_info(int32_t handle) {
    if (handle < 0 || handle >= MAX_INFOS ||
            !atomic_load_explicit(&info_used[handle], memory_order_acquire)) {
        return MPI_INFO_NULL;
    }
    return info_table[handle];
}

// Free an info handle (thread-safe: release store to info_used).
static void free_info(int32_t handle) {
    if (handle >= 0 && handle < MAX_INFOS) {
        info_table[handle] = MPI_INFO_NULL;
        atomic_store_explicit(&info_used[handle], 0, memory_order_release);
    }
}

// Allocate a group handle (thread-safe via C11 CAS, mirrors alloc_win).
// Slot 0 is reserved for MPI_GROUP_EMPTY and must NOT be allocated via this
// path; alloc_group starts scanning from hint > 0 by construction, but the
// hint wraps around, so we explicitly skip slot 0.
static int32_t alloc_group(MPI_Group group) {
    int hint = atomic_load_explicit(&next_group_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_GROUPS - 1; i++) {
        int idx = (hint + i) % (MAX_GROUPS - 1) + 1;  // slots 1..MAX_GROUPS-1
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &group_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            group_table[idx] = group;
            atomic_store_explicit(&next_group_hint,
                idx % (MAX_GROUPS - 1), memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

/* Get MPI_Group from handle (thread-safe: acquire load).
 *
 * Sentinel contract (depended on by all callers in this file):
 *   handle == 0                → MPI_GROUP_EMPTY (the reserved lazy slot)
 *   handle out of range        → MPI_GROUP_NULL  (invalid handle error)
 *   handle in range, unused    → MPI_GROUP_NULL  (handle was freed)
 *   handle in range, used      → group_table[handle]
 *
 * The MPI_GROUP_NULL sentinel pairs with the
 *   `if (g == MPI_GROUP_NULL) return MPI_ERR_ARG;`
 * guards at ferrompi_comm_create_from_group_parent,
 * ferrompi_comm_create_from_group, and the group-operation shims.
 * Callers without that guard (group_incl, group_excl, group_size,
 * group_rank) correctly defer to MPI's own MPI_ERR_GROUP for invalid
 * handles. */
static MPI_Group get_group(int32_t handle) {
    // Slot 0: lazily return MPI_GROUP_EMPTY (always valid post-init)
    if (handle == 0) {
        return MPI_GROUP_EMPTY;
    }
    if (handle < 0 || handle >= MAX_GROUPS) {
        return MPI_GROUP_NULL;
    }
    if (!atomic_load_explicit(&group_used[handle], memory_order_acquire)) {
        return MPI_GROUP_NULL;
    }
    return group_table[handle];
}

// Free a group handle (thread-safe: release store; does NOT call MPI_Group_free).
static void free_group(int32_t handle) {
    if (handle > 0 && handle < MAX_GROUPS) {
        group_table[handle] = MPI_GROUP_EMPTY;
        atomic_store_explicit(&group_used[handle], 0, memory_order_release);
    }
}

// Allocate a custom datatype handle (thread-safe via C11 CAS, mirrors alloc_win).
static int32_t alloc_datatype(MPI_Datatype dtype) {
    int hint = atomic_load_explicit(&next_datatype_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_DATATYPES; i++) {
        int idx = (hint + i) % MAX_DATATYPES;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &datatype_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            datatype_table[idx] = dtype;
            atomic_store_explicit(&next_datatype_hint,
                (idx + 1) % MAX_DATATYPES, memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  // No space
}

// Get a committed MPI_Datatype from a custom-datatype handle (thread-safe: acquire load).
// Named get_datatype_committed to avoid collision with the predefined-tag
// helper get_datatype(int32_t tag) defined below.
static MPI_Datatype get_datatype_committed(int32_t handle) {
    if (handle < 0 || handle >= MAX_DATATYPES ||
            !atomic_load_explicit(&datatype_used[handle], memory_order_acquire)) {
        return MPI_DATATYPE_NULL;
    }
    return datatype_table[handle];
}

// Free a custom datatype handle slot (thread-safe: release store;
// does NOT call MPI_Type_free; callers do that).
static void free_datatype_slot(int32_t handle) {
    if (handle >= 0 && handle < MAX_DATATYPES) {
        datatype_table[handle] = MPI_DATATYPE_NULL;
        atomic_store_explicit(&datatype_used[handle], 0, memory_order_release);
    }
}

// Map operation code to MPI_Op
static MPI_Op get_op(int32_t op) {
    switch (op) {
        case 0: return MPI_SUM;
        case 1: return MPI_MAX;
        case 2: return MPI_MIN;
        case 3: return MPI_PROD;
        case 4: return MPI_BOR;
        case 5: return MPI_BAND;
        case 6: return MPI_BXOR;
        case 7: return MPI_LOR;
        case 8: return MPI_LAND;
        case 9: return MPI_LXOR;
        case 10: return MPI_MAXLOC;
        case 11: return MPI_MINLOC;
        case 12: return MPI_REPLACE;
        case 13: return MPI_NO_OP;
        default: return MPI_OP_NULL;
    }
}

// MPI_LONG_INT / MPI_LONG_DOUBLE_INT are exposed only where their C layout is
// verified against the Rust LongInt / LongDoubleInt structs.
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__) || (defined(__powerpc64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define FERROMPI_LONG_PAIRS_VERIFIED 1
_Static_assert(sizeof(struct { long v; int i; }) == 16 &&
               _Alignof(struct { long v; int i; }) == 8,
               "MPI_LONG_INT pair layout differs from the Rust LongInt");
_Static_assert(sizeof(struct { long double v; int i; }) == 32 &&
               _Alignof(struct { long double v; int i; }) == 16,
               "MPI_LONG_DOUBLE_INT pair layout differs from the Rust LongDoubleInt");
#endif

// Map datatype tag to MPI_Datatype
static MPI_Datatype get_datatype(int32_t tag) {
    switch (tag) {
        case FERROMPI_F32:             return MPI_FLOAT;
        case FERROMPI_F64:             return MPI_DOUBLE;
        case FERROMPI_I32:             return MPI_INT32_T;
        case FERROMPI_I64:             return MPI_INT64_T;
        case FERROMPI_U8:              return MPI_UINT8_T;
        case FERROMPI_U32:             return MPI_UINT32_T;
        case FERROMPI_U64:             return MPI_UINT64_T;
        case FERROMPI_FLOAT_INT:       return MPI_FLOAT_INT;
        case FERROMPI_DOUBLE_INT:      return MPI_DOUBLE_INT;
#ifdef FERROMPI_LONG_PAIRS_VERIFIED
        case FERROMPI_LONG_INT:        return MPI_LONG_INT;
#endif
        case FERROMPI_2INT:            return MPI_2INT;
        case FERROMPI_SHORT_INT:       return MPI_SHORT_INT;
#ifdef FERROMPI_LONG_PAIRS_VERIFIED
        case FERROMPI_LONG_DOUBLE_INT: return MPI_LONG_DOUBLE_INT;
#endif
        case FERROMPI_BYTE:            return MPI_BYTE;
        default:                       return MPI_DATATYPE_NULL;
    }
}

/* ============================================================
 * Error Handler Installation
 * ============================================================ */

/* Install MPI_ERRORS_RETURN on the given communicator so that MPI
 * errors are returned as codes to ferrompi rather than aborting the
 * process via MPI_ERRORS_ARE_FATAL. Called from every site that
 * creates or adopts an MPI_Comm before handing it to Rust, and also on
 * MPI_COMM_SELF at init, since MPI 4 delivers errors that are not
 * associated with any object (e.g. group construction) through
 * MPI_COMM_SELF's handler. */
static int install_errors_return(MPI_Comm comm) {
    return MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
}

/* Install MPI_ERRORS_RETURN on a window, best-effort: Open MPI 4.x can
 * reject MPI_Win_set_errhandler on a fresh window, in which case the
 * window keeps MPI_ERRORS_ARE_FATAL and a warning is printed. */
static void install_errors_return_win(MPI_Win win) {
    int ret = MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN);
    if (ret != MPI_SUCCESS) {
        /* Note: cannot use Rust's logging from C; stderr is the
         * portable fallback. */
        fprintf(stderr,
                "ferrompi: warning: MPI_Win_set_errhandler returned %d on a "
                "freshly-created window.  The window will use the MPI default "
                "error handler (MPI_ERRORS_ARE_FATAL).  Subsequent RMA errors "
                "on this window will abort the process rather than return as "
                "Result::Err.  This is a known OpenMPI 4.x quirk.\n",
                ret);
    }
}

/* ============================================================
 * Initialization and Finalization
 * ============================================================ */

int ferrompi_init_thread(int required, int* provided) {
    init_tables();
    int mpi_required;
    switch (required) {
        case 0: mpi_required = MPI_THREAD_SINGLE; break;
        case 1: mpi_required = MPI_THREAD_FUNNELED; break;
        case 2: mpi_required = MPI_THREAD_SERIALIZED; break;
        case 3: mpi_required = MPI_THREAD_MULTIPLE; break;
        default: mpi_required = MPI_THREAD_SINGLE; break;
    }
    
    int mpi_provided;
    int ret = MPI_Init_thread(NULL, NULL, mpi_required, &mpi_provided);
    
    // Initialize COMM_WORLD after MPI_Init and install the error handler
    if (ret == MPI_SUCCESS) {
        comm_table[0] = MPI_COMM_WORLD;
        atomic_store_explicit(&comm_used[0], 1, memory_order_release);
        int eh_ret = install_errors_return(MPI_COMM_WORLD);
        if (eh_ret == MPI_SUCCESS) {
            eh_ret = install_errors_return(MPI_COMM_SELF);
        }
        if (eh_ret != MPI_SUCCESS) {
            /* MPI is initialized and cannot be initialized again, and
             * MPI_Finalize is collective while this failure is local:
             * end the job as MPI's default error handler would. MPI_Abort
             * is only required to make a "best attempt"; it can return
             * (measured: MPICH before the launcher kills the other ranks),
             * so abort() guarantees this rank never falls through. */
            MPI_Abort(MPI_COMM_WORLD, eh_ret);
            abort();
        }
    }

    if (provided) {
        switch (mpi_provided) {
            case MPI_THREAD_SINGLE: *provided = 0; break;
            case MPI_THREAD_FUNNELED: *provided = 1; break;
            case MPI_THREAD_SERIALIZED: *provided = 2; break;
            case MPI_THREAD_MULTIPLE: *provided = 3; break;
            default: *provided = 0; break;
        }
    }
    
    return ret;
}

int ferrompi_finalize(int32_t* active_requests) {
    // Clean up requests first (they may reference communicators).
    // Acquire/release here is defensive: MPI_Finalize is called after all
    // concurrent MPI operations are complete, but the consistent access
    // pattern avoids spurious TSan warnings in the finalizer check.
    //
    // Only an inactive persistent request is freed here. Freeing an active
    // point-to-point request, persistent included, is legal, but this table
    // does not distinguish point-to-point from collective requests: freeing
    // a nonblocking-collective request or an active persistent collective is
    // erroneous (Open MPI returns MPI_ERR_REQUEST). Leaving a request
    // pending at MPI_Finalize is itself non-conforming but tolerated by
    // MPICH and Open MPI, which progress it inside MPI_Finalize, so the
    // sweep takes that as the lesser error: every other occupied slot is
    // left for MPI to tear down and counted instead, so the caller can
    // report it.
    int32_t active = 0;
    for (int i = 0; i < MAX_REQUESTS; i++) {
        unsigned widx = (unsigned)(i / 64);
        uint64_t mask = (uint64_t)1 << (i % 64);
        if ((atomic_load_explicit(&request_bits[widx], memory_order_acquire) & mask) &&
                request_table[i] != MPI_REQUEST_NULL) {
            if (request_state[i] == REQUEST_PERSISTENT) {
                MPI_Request_free(&request_table[i]);
            } else {
                active++;
            }
        }
    }
    for (int w = 0; w < REQUEST_BITS_WORDS; w++) {
        atomic_store_explicit(&request_bits[w], (uint64_t)0, memory_order_release);
    }
    *active_requests = active;

    // Free every live user op: frees the MPI op and drops the Rust closure
    // through the same path as UserOp's Drop; a failed MPI_Op_free keeps both.
    for (int i = 0; i < MAX_OPS; i++) {
        if (atomic_load_explicit(&op_used[i], memory_order_acquire)) {
            ferrompi_op_free(i);
        }
    }

    // Clean up any remaining group objects (skip slot 0, which is MPI_GROUP_EMPTY)
    for (int i = 1; i < MAX_GROUPS; i++) {
        if (atomic_load_explicit(&group_used[i], memory_order_acquire) &&
                group_table[i] != MPI_GROUP_EMPTY) {
            MPI_Group_free(&group_table[i]);
        }
        atomic_store_explicit(&group_used[i], 0, memory_order_release);
    }

    // Clean up any remaining info objects
    for (int i = 0; i < MAX_INFOS; i++) {
        if (atomic_load_explicit(&info_used[i], memory_order_acquire) &&
                info_table[i] != MPI_INFO_NULL) {
            MPI_Info_free(&info_table[i]);
        }
        atomic_store_explicit(&info_used[i], 0, memory_order_release);
    }

    // Clean up any remaining custom datatypes
    for (int i = 0; i < MAX_DATATYPES; i++) {
        if (atomic_load_explicit(&datatype_used[i], memory_order_acquire) &&
                datatype_table[i] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&datatype_table[i]);
        }
        atomic_store_explicit(&datatype_used[i], 0, memory_order_release);
    }

    // Mpi skips MPI_Finalize while any window is alive, so this point is
    // never reached with a live window; just reset the table.
    for (int i = 0; i < MAX_WINDOWS; i++) {
        atomic_store_explicit(&win_used[i], 0, memory_order_release);
    }

    // Clean up any remaining communicators (skip slot 0, which is MPI_COMM_WORLD)
    for (int i = 1; i < MAX_COMMS; i++) {
        if (atomic_load_explicit(&comm_used[i], memory_order_acquire)) {
            MPI_Comm_free(&comm_table[i]);
            comm_table[i] = MPI_COMM_NULL;
            atomic_store_explicit(&comm_used[i], 0, memory_order_release);
        }
    }
    
    return MPI_Finalize();
}

int ferrompi_initialized(int* flag) {
    return MPI_Initialized(flag);
}

int ferrompi_finalized(int* flag) {
    return MPI_Finalized(flag);
}

/* ============================================================
 * Communicator Operations
 * ============================================================ */

int ferrompi_comm_rank(int32_t comm_handle, int32_t* rank) {
    MPI_Comm comm = get_comm(comm_handle);
    int r;
    int ret = MPI_Comm_rank(comm, &r);
    *rank = (int32_t)r;
    return ret;
}

int ferrompi_comm_size(int32_t comm_handle, int32_t* size) {
    MPI_Comm comm = get_comm(comm_handle);
    int s;
    int ret = MPI_Comm_size(comm, &s);
    *size = (int32_t)s;
    return ret;
}

/* A shim below that cannot hand a new communicator to Rust (MPI_ERRORS_RETURN
 * cannot be installed on it, or the handle table is full) returns the error
 * without freeing it: MPI created it on every rank, MPI_Comm_free is
 * collective, and freeing it on this rank alone can hang or mismatch
 * collectives. It stays allocated until MPI_Finalize. */
int ferrompi_comm_dup(int32_t comm_handle, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Comm newcomm;
    int ret = MPI_Comm_dup(comm, &newcomm);
    if (ret == MPI_SUCCESS) {
        int eh_ret = install_errors_return(newcomm);
        if (eh_ret != MPI_SUCCESS) {
            return eh_ret;
        }
        *newcomm_handle = alloc_comm(newcomm);
        if (*newcomm_handle < 0) {
            return FERROMPI_ERR_COMMS_FULL;
        }
    }
    return ret;
}

int ferrompi_comm_free(int32_t comm_handle) {
    if (comm_handle == 0) {
        return MPI_ERR_COMM;  // Cannot free COMM_WORLD
    }
    if (comm_handle < 0 || comm_handle >= MAX_COMMS) {
        return MPI_ERR_COMM;
    }
    if (!atomic_load_explicit(&comm_used[comm_handle], memory_order_acquire)) {
        return MPI_SUCCESS;  // Already freed
    }
    MPI_Comm comm = comm_table[comm_handle];
    int ret = MPI_Comm_free(&comm);
    comm_table[comm_handle] = MPI_COMM_NULL;
    atomic_store_explicit(&comm_used[comm_handle], 0, memory_order_release);
    return ret;
}

int ferrompi_comm_split(int32_t comm_handle, int32_t color, int32_t key, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    int mpi_color = (color == -1) ? MPI_UNDEFINED : color;
    MPI_Comm newcomm;
    int ret = MPI_Comm_split(comm, mpi_color, key, &newcomm);
    if (ret == MPI_SUCCESS) {
        if (newcomm == MPI_COMM_NULL) {
            *newcomm_handle = -1;  // Process opted out
        } else {
            int eh_ret = install_errors_return(newcomm);
            if (eh_ret != MPI_SUCCESS) {
                return eh_ret;
            }
            *newcomm_handle = alloc_comm(newcomm);
            if (*newcomm_handle < 0) {
                return FERROMPI_ERR_COMMS_FULL;
            }
        }
    }
    return ret;
}

int ferrompi_comm_split_type(int32_t comm_handle, int32_t split_type, int32_t key, int32_t* newcomm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    int mpi_split_type;
    switch (split_type) {
        case FERROMPI_COMM_TYPE_SHARED:
            mpi_split_type = MPI_COMM_TYPE_SHARED;
            break;
        default:
            return MPI_ERR_ARG;
    }
    MPI_Comm newcomm;
    int ret = MPI_Comm_split_type(comm, mpi_split_type, key, MPI_INFO_NULL, &newcomm);
    if (ret == MPI_SUCCESS) {
        if (newcomm == MPI_COMM_NULL) {
            *newcomm_handle = -1;
        } else {
            int eh_ret = install_errors_return(newcomm);
            if (eh_ret != MPI_SUCCESS) {
                return eh_ret;
            }
            *newcomm_handle = alloc_comm(newcomm);
            if (*newcomm_handle < 0) {
                return FERROMPI_ERR_COMMS_FULL;
            }
        }
    }
    return ret;
}

int ferrompi_comm_create_from_group_parent(int32_t comm_h,
                                           int32_t group_h,
                                           int32_t* out_h) {
    MPI_Comm parent = get_comm(comm_h);
    MPI_Group g = get_group(group_h);
    if (parent == MPI_COMM_NULL || g == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Comm new_comm;
    int ret = MPI_Comm_create(parent, g, &new_comm);
    if (ret != MPI_SUCCESS) return ret;
    if (new_comm == MPI_COMM_NULL) {
        *out_h = -1;  /* Caller is not in the group */
        return MPI_SUCCESS;
    }
    int eh_ret = install_errors_return(new_comm);
    if (eh_ret != MPI_SUCCESS) return eh_ret;
    *out_h = alloc_comm(new_comm);
    if (*out_h < 0) return FERROMPI_ERR_COMMS_FULL;
    return MPI_SUCCESS;
}

int ferrompi_comm_create_from_group(int32_t group_h,
                                    const char* stringtag,
                                    int32_t* out_h) {
#ifdef FERROMPI_HAVE_MPI4_COLLECTIVES
    MPI_Group g = get_group(group_h);
    if (g == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Comm new_comm;
    int ret = MPI_Comm_create_from_group(g, stringtag, MPI_INFO_NULL,
                                         MPI_ERRORS_RETURN, &new_comm);
    if (ret != MPI_SUCCESS) return ret;
    /* MPI_COMM_NULL is returned on ranks outside the group */
    if (new_comm == MPI_COMM_NULL) {
        return MPI_ERR_OTHER;
    }
    *out_h = alloc_comm(new_comm);
    if (*out_h < 0) return FERROMPI_ERR_COMMS_FULL;
    return MPI_SUCCESS;
#else
    (void)group_h; (void)stringtag; (void)out_h;
    return FERROMPI_ERR_NOT_SUPPORTED;  /* needs MPI 4.0, or Open MPI 5 */
#endif
}

/* ============================================================
 * Synchronization
 * ============================================================ */

int ferrompi_barrier(int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Barrier(comm);
}

/* ============================================================
 * Generic Point-to-Point Communication
 * ============================================================ */

// Element count of a received message, -1 when MPI reports MPI_UNDEFINED.
static int get_count64(const MPI_Status* status, MPI_Datatype dt, int64_t* count) {
#if MPI_VERSION >= 4
    MPI_Count cnt = MPI_UNDEFINED;
    int ret = MPI_Get_count_c(status, dt, &cnt);
#else
    int cnt = MPI_UNDEFINED;
    int ret = MPI_Get_count(status, dt, &cnt);
#endif
    *count = (cnt == MPI_UNDEFINED) ? -1 : (int64_t)cnt;
    return ret;
}

static int send_typed(const void* buf, int64_t count, MPI_Datatype dt,
                       int32_t dest, int32_t tag, int32_t comm_handle) {
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Comm comm = get_comm(comm_handle);
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Send_c(buf, (MPI_Count)count, dt, dest, tag, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Send(buf, (int)count, dt, dest, tag, comm);
}

static int recv_typed(void* buf, int64_t count, MPI_Datatype dt,
                       int32_t source, int32_t tag, int32_t comm_handle,
                       int32_t* actual_source, int32_t* actual_tag,
                       int64_t* actual_count) {
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Status status;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    int ret;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Recv_c(buf, (MPI_Count)count, dt, mpi_source, mpi_tag, comm, &status);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Recv(buf, (int)count, dt, mpi_source, mpi_tag, comm, &status);
    }

    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
        ret = get_count64(&status, dt, actual_count);
    }

    return ret;
}

static int isend_typed(const void* buf, int64_t count, MPI_Datatype dt,
                        int32_t dest, int32_t tag, int32_t comm_handle,
                        int64_t* request_handle) {
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Isend_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Isend(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

static int irecv_typed(void* buf, int64_t count, MPI_Datatype dt,
                        int32_t source, int32_t tag, int32_t comm_handle,
                        int64_t* request_handle) {
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    int ret;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Irecv_c(buf, (MPI_Count)count, dt, mpi_source, mpi_tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Irecv(buf, (int)count, dt, mpi_source, mpi_tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_send(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle
) {
    return send_typed(buf, count, get_datatype(datatype_tag), dest, tag, comm_handle);
}

int ferrompi_recv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    return recv_typed(buf, count, get_datatype(datatype_tag), source, tag, comm_handle,
                       actual_source, actual_tag, actual_count);
}

int ferrompi_isend(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    return isend_typed(buf, count, get_datatype(datatype_tag), dest, tag, comm_handle, request_handle);
}

int ferrompi_irecv(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    return irecv_typed(buf, count, get_datatype(datatype_tag), source, tag, comm_handle, request_handle);
}

int ferrompi_sendrecv(
    const void* sendbuf,
    int64_t sendcount,
    int32_t send_datatype_tag,
    int32_t dest,
    int32_t sendtag,
    void* recvbuf,
    int64_t recvcount,
    int32_t recv_datatype_tag,
    int32_t source,
    int32_t recvtag,
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype send_dt = get_datatype(send_datatype_tag);
    MPI_Datatype recv_dt = get_datatype(recv_datatype_tag);
    if (send_dt == MPI_DATATYPE_NULL || recv_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Status status;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_recvtag = (recvtag == -1) ? MPI_ANY_TAG : recvtag;

    int ret;
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Sendrecv_c(sendbuf, (MPI_Count)sendcount, send_dt, dest, sendtag,
                             recvbuf, (MPI_Count)recvcount, recv_dt, mpi_source, mpi_recvtag,
                             comm, &status);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Sendrecv(sendbuf, (int)sendcount, send_dt, dest, sendtag,
                           recvbuf, (int)recvcount, recv_dt, mpi_source, mpi_recvtag,
                           comm, &status);
    }

    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
        ret = get_count64(&status, recv_dt, actual_count);
    }

    return ret;
}

/* ============================================================
 * Message Probing
 * ============================================================ */

int ferrompi_probe(int32_t source, int32_t tag, int32_t comm_handle,
                   int32_t* actual_source, int32_t* actual_tag,
                   int64_t* count, int32_t datatype_tag) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int ret = MPI_Probe(mpi_source, mpi_tag, comm, &status);
    if (ret == MPI_SUCCESS) {
        *actual_source = status.MPI_SOURCE;
        *actual_tag = status.MPI_TAG;
        ret = get_count64(&status, dt, count);
    }
    return ret;
}

int ferrompi_iprobe(int32_t source, int32_t tag, int32_t comm_handle,
                    int32_t* flag, int32_t* actual_source, int32_t* actual_tag,
                    int64_t* count, int32_t datatype_tag) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    MPI_Status status;
    int f;
    int ret = MPI_Iprobe(mpi_source, mpi_tag, comm, &f, &status);
    if (ret == MPI_SUCCESS) {
        *flag = f;
        if (f) {
            *actual_source = status.MPI_SOURCE;
            *actual_tag = status.MPI_TAG;
            ret = get_count64(&status, dt, count);
        }
    }
    return ret;
}

/* ============================================================
 * Generic Collective Operations - Blocking
 * ============================================================ */

int ferrompi_bcast(void* buf, int64_t count, int32_t datatype_tag, int32_t root, int32_t comm_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Bcast_c(buf, (MPI_Count)count, dt, root, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Bcast(buf, (int)count, dt, root, comm);
}

int ferrompi_reduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Reduce_c(sb, recvbuf, (MPI_Count)count, dt, mpi_op, root, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Reduce(sb, recvbuf, (int)count, dt, mpi_op, root, comm);
}

int ferrompi_allreduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Allreduce_c(sb, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Allreduce(sb, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_scan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Scan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Scan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_exscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Exscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Exscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm);
}

int ferrompi_gather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Gather_c(sb, (MPI_Count)sendcount, dt,
                           recvbuf, (MPI_Count)recvcount, dt,
                           root, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Gather(sb, (int)sendcount, dt,
                      recvbuf, (int)recvcount, dt,
                      root, comm);
}

int ferrompi_allgather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Allgather_c(sb, (MPI_Count)sendcount, dt,
                               recvbuf, (MPI_Count)recvcount, dt,
                               comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Allgather(sb, (int)sendcount, dt,
                         recvbuf, (int)recvcount, dt,
                         comm);
}

int ferrompi_scatter(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    void* rb = recvbuf ? recvbuf : MPI_IN_PLACE;
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Scatter_c(sendbuf, (MPI_Count)sendcount, dt,
                            rb, (MPI_Count)recvcount, dt,
                            root, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Scatter(sendbuf, (int)sendcount, dt,
                       rb, (int)recvcount, dt,
                       root, comm);
}

int ferrompi_alltoall(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Alltoall_c(sb, (MPI_Count)sendcount, dt,
                              recvbuf, (MPI_Count)recvcount, dt, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Alltoall(sb, (int)sendcount, dt,
                        recvbuf, (int)recvcount, dt, comm);
}

int ferrompi_reduce_scatter_block(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    MPI_Op mpi_op = get_op(op);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Reduce_scatter_block_c(sendbuf, recvbuf, (MPI_Count)recvcount, dt, mpi_op, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Reduce_scatter_block(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm);
}

/* ============================================================
 * Generic V-Collectives (variable-count)
 * ============================================================ */

int ferrompi_gatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    return MPI_Gatherv(sendbuf, (int)sendcount, dt,
                       recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                       root, comm);
}

int ferrompi_scatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (recvcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    return MPI_Scatterv(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                        recvbuf, (int)recvcount, dt,
                        root, comm);
}

int ferrompi_allgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    return MPI_Allgatherv(sendbuf, (int)sendcount, dt,
                          recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                          comm);
}

int ferrompi_alltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    return MPI_Alltoallv(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                         recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                         comm);
}

/* ============================================================
 * Generic Collective Operations - Nonblocking
 * ============================================================ */

int ferrompi_ibcast(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;
    
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Ibcast_c(buf, (MPI_Count)count, dt, root, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Ibcast(buf, (int)count, dt, root, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    
    return ret;
}

int ferrompi_iallreduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;
    
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Iallreduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Iallreduce(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }
    
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    
    return ret;
}

int ferrompi_ireduce(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Ireduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, root, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Ireduce(sendbuf, recvbuf, (int)count, dt, mpi_op, root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_igather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Igather_c(sb, (MPI_Count)sendcount, dt,
                            recvbuf, (MPI_Count)recvcount, dt,
                            root, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Igather(sb, (int)sendcount, dt,
                          recvbuf, (int)recvcount, dt,
                          root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iallgather(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Iallgather_c(sb, (MPI_Count)sendcount, dt,
                                recvbuf, (MPI_Count)recvcount, dt,
                                comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Iallgather(sb, (int)sendcount, dt,
                             recvbuf, (int)recvcount, dt,
                             comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iscatter(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    void* rb = recvbuf ? recvbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Iscatter_c(sendbuf, (MPI_Count)sendcount, dt,
                             rb, (MPI_Count)recvcount, dt,
                             root, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Iscatter(sendbuf, (int)sendcount, dt,
                           rb, (int)recvcount, dt,
                           root, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_ibarrier(
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Request req;
    int ret = MPI_Ibarrier(comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Iscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Iscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iexscan(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Iexscan_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Iexscan(sendbuf, recvbuf, (int)count, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_ialltoall(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Ialltoall_c(sb, (MPI_Count)sendcount, dt,
                               recvbuf, (MPI_Count)recvcount, dt, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Ialltoall(sb, (int)sendcount, dt,
                            recvbuf, (int)recvcount, dt, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_igatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t root, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;
    int ret = MPI_Igatherv(sendbuf, (int)sendcount, dt,
                           recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                           root, comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iscatterv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
    void* recvbuf, int64_t recvcount,
    int32_t datatype_tag, int32_t root, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (recvcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;
    int ret = MPI_Iscatterv(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                            recvbuf, (int)recvcount, dt,
                            root, comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_iallgatherv(
    const void* sendbuf, int64_t sendcount,
    void* recvbuf, const int32_t* recvcounts, const int32_t* displs,
    int32_t datatype_tag, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;
    int ret = MPI_Iallgatherv(sendbuf, (int)sendcount, dt,
                              recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                              comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_ialltoallv(
    const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
    void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
    int32_t datatype_tag, int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret = MPI_Ialltoallv(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                             recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                             comm, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_ireduce_scatter_block(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Ireduce_scatter_block_c(sendbuf, recvbuf, (MPI_Count)recvcount, dt, mpi_op, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Ireduce_scatter_block(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

/* ============================================================
 * Persistent Point-to-Point (MPI 1.1+)
 * ============================================================ */

int ferrompi_send_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Send_init_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Send_init(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_recv_init(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int mpi_source = (source == -1) ? MPI_ANY_SOURCE : source;
    int mpi_tag = (tag == -1) ? MPI_ANY_TAG : tag;

    int ret;
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Recv_init_c(buf, (MPI_Count)count, dt, mpi_source, mpi_tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Recv_init(buf, (int)count, dt, mpi_source, mpi_tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_rsend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Rsend_init_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Rsend_init(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_ssend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Ssend_init_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Ssend_init(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

/* ============================================================
 * Buffered Send Buffer Management and Persistent Buffered Send (MPI 1.1+)
 * ============================================================ */

/* size is cast to int; the Rust caller rejects sizes above INT_MAX. */
int ferrompi_buffer_attach(void* buffer, int64_t size) {
    return MPI_Buffer_attach(buffer, (int)size);
}

int ferrompi_buffer_detach(void** buffer, int64_t* size) {
    int int_size = 0;
    int ret = MPI_Buffer_detach(buffer, &int_size);
    if (ret == MPI_SUCCESS) {
        *size = (int64_t)int_size;
    }
    return ret;
}

int ferrompi_bsend_init(
    const void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Bsend_init_c(buf, (MPI_Count)count, dt, dest, tag, comm, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Bsend_init(buf, (int)count, dt, dest, tag, comm, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

/* ============================================================
 * Generic Persistent Collectives (MPI 4.0, or Open MPI 5)
 * ============================================================ */

#ifdef FERROMPI_HAVE_MPI4_COLLECTIVES

int ferrompi_bcast_init(
    void* buf,
    int64_t count,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Bcast_init_c(buf, (MPI_Count)count, dt, root, comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Bcast_init(buf, (int)count, dt, root, comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    
    return ret;
}

int ferrompi_allreduce_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Allreduce_init_c(sb, recvbuf, (MPI_Count)count, dt,
                                    mpi_op, comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Allreduce_init(sb, recvbuf, (int)count, dt,
                                  mpi_op, comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_gather_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Gather_init_c(sb, (MPI_Count)sendcount, dt,
                                 recvbuf, (MPI_Count)recvcount, dt,
                                 root, comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Gather_init(sb, (int)sendcount, dt,
                              recvbuf, (int)recvcount, dt,
                              root, comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    
    return ret;
}

int ferrompi_reduce_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Reduce_init_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, root, comm,
                                 MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Reduce_init(sendbuf, recvbuf, (int)count, dt, mpi_op, root, comm,
                              MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_scatter_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    void* rb = recvbuf ? recvbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Scatter_init_c(sendbuf, (MPI_Count)sendcount, dt,
                                  rb, (MPI_Count)recvcount, dt,
                                  root, comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Scatter_init(sendbuf, (int)sendcount, dt,
                               rb, (int)recvcount, dt,
                               root, comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_allgather_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Allgather_init_c(sb, (MPI_Count)sendcount, dt,
                                    recvbuf, (MPI_Count)recvcount, dt,
                                    comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Allgather_init(sb, (int)sendcount, dt,
                                 recvbuf, (int)recvcount, dt,
                                 comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_scan_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Scan_init_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm,
                               MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Scan_init(sendbuf, recvbuf, (int)count, dt, mpi_op, comm,
                            MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_exscan_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Exscan_init_c(sendbuf, recvbuf, (MPI_Count)count, dt, mpi_op, comm,
                                 MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Exscan_init(sendbuf, recvbuf, (int)count, dt, mpi_op, comm,
                              MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_alltoall_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    const void* sb = sendbuf ? sendbuf : MPI_IN_PLACE;
    MPI_Request req;
    int ret;

    if (sendcount > INT_MAX || recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Alltoall_init_c(sb, (MPI_Count)sendcount, dt,
                                   recvbuf, (MPI_Count)recvcount, dt,
                                   comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Alltoall_init(sb, (int)sendcount, dt,
                                recvbuf, (int)recvcount, dt,
                                comm, MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_gatherv_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* displs,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;

    int ret = MPI_Gatherv_init(sendbuf, (int)sendcount, dt,
                               recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                               root, comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_scatterv_init(
    const void* sendbuf,
    const int32_t* sendcounts,
    const int32_t* displs,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t root,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (recvcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;

    int ret = MPI_Scatterv_init(sendbuf, (const int*)sendcounts, (const int*)displs, dt,
                                recvbuf, (int)recvcount, dt,
                                root, comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_allgatherv_init(
    const void* sendbuf,
    int64_t sendcount,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* displs,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (sendcount > INT_MAX) {
        return MPI_ERR_COUNT;
    }
    MPI_Request req;

    int ret = MPI_Allgatherv_init(sendbuf, (int)sendcount, dt,
                                  recvbuf, (const int*)recvcounts, (const int*)displs, dt,
                                  comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_alltoallv_init(
    const void* sendbuf,
    const int32_t* sendcounts,
    const int32_t* sdispls,
    void* recvbuf,
    const int32_t* recvcounts,
    const int32_t* rdispls,
    int32_t datatype_tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;

    int ret = MPI_Alltoallv_init(sendbuf, (const int*)sendcounts, (const int*)sdispls, dt,
                                 recvbuf, (const int*)recvcounts, (const int*)rdispls, dt,
                                 comm, MPI_INFO_NULL, &req);

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

int ferrompi_reduce_scatter_block_init(
    const void* sendbuf,
    void* recvbuf,
    int64_t recvcount,
    int32_t datatype_tag,
    int32_t op,
    int32_t comm_handle,
    int64_t* request_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op);
    MPI_Request req;
    int ret;

    if (recvcount > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Reduce_scatter_block_init_c(sendbuf, recvbuf, (MPI_Count)recvcount, dt, mpi_op,
                                               comm, MPI_INFO_NULL, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Reduce_scatter_block_init(sendbuf, recvbuf, (int)recvcount, dt, mpi_op, comm,
                                            MPI_INFO_NULL, &req);
    }

    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 1);
        if (*request_handle < 0) {
            MPI_Request_free(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }

    return ret;
}

#else /* !FERROMPI_HAVE_MPI4_COLLECTIVES */

int ferrompi_bcast_init(void* buf, int64_t count, int32_t datatype_tag, int32_t root,
                        int32_t comm_handle, int64_t* request_handle) {
    (void)buf; (void)count; (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_allreduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                            int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag; (void)op; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_gather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                         int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_reduce_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                         int32_t op, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_scatter_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                          int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_allgather_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                            int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_scan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                       int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_exscan_init(const void* sendbuf, void* recvbuf, int64_t count, int32_t datatype_tag,
                         int32_t op, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_alltoall_init(const void* sendbuf, int64_t sendcount, void* recvbuf, int64_t recvcount,
                           int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_gatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf,
                          const int32_t* recvcounts, const int32_t* displs,
                          int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcounts; (void)displs;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_scatterv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* displs,
                           void* recvbuf, int64_t recvcount,
                           int32_t datatype_tag, int32_t root, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcounts; (void)displs; (void)recvbuf; (void)recvcount;
    (void)datatype_tag; (void)root; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_allgatherv_init(const void* sendbuf, int64_t sendcount, void* recvbuf,
                             const int32_t* recvcounts, const int32_t* displs,
                             int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcount; (void)recvbuf; (void)recvcounts; (void)displs;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_alltoallv_init(const void* sendbuf, const int32_t* sendcounts, const int32_t* sdispls,
                            void* recvbuf, const int32_t* recvcounts, const int32_t* rdispls,
                            int32_t datatype_tag, int32_t comm_handle, int64_t* request_handle) {
    (void)sendbuf; (void)sendcounts; (void)sdispls; (void)recvbuf; (void)recvcounts; (void)rdispls;
    (void)datatype_tag; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

int ferrompi_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int64_t recvcount,
                                       int32_t datatype_tag, int32_t op, int32_t comm_handle,
                                       int64_t* request_handle) {
    (void)sendbuf; (void)recvbuf; (void)recvcount; (void)datatype_tag;
    (void)op; (void)comm_handle; (void)request_handle;
    return FERROMPI_ERR_NOT_SUPPORTED;
}

#endif /* FERROMPI_HAVE_MPI4_COLLECTIVES */

/* ============================================================
 * Info Object Operations
 * ============================================================ */

int ferrompi_info_create(int32_t* info_handle) {
    MPI_Info info;
    int ret = MPI_Info_create(&info);
    if (ret == MPI_SUCCESS) {
        *info_handle = alloc_info(info);
        if (*info_handle < 0) {
            MPI_Info_free(&info);
            return FERROMPI_ERR_INFOS_FULL;
        }
    }
    return ret;
}

int ferrompi_info_free(int32_t info_handle) {
    if (info_handle < 0 || info_handle >= MAX_INFOS ||
            !atomic_load_explicit(&info_used[info_handle],
                                  memory_order_acquire)) {
        return MPI_SUCCESS;
    }
    int ret = MPI_Info_free(&info_table[info_handle]);
    free_info(info_handle);
    return ret;
}

int ferrompi_info_set(int32_t info_handle, const char* key, const char* value) {
    MPI_Info info = get_info(info_handle);
    if (info == MPI_INFO_NULL) return MPI_ERR_INFO;
    return MPI_Info_set(info, key, value);
}

int ferrompi_info_get(int32_t info_handle, const char* key, char* value, int32_t* valuelen, int32_t* flag) {
    MPI_Info info = get_info(info_handle);
    if (info == MPI_INFO_NULL) return MPI_ERR_INFO;
    int f;
#if MPI_VERSION >= 4
    int ret = MPI_Info_get_string(info, key, valuelen, value, &f);
#else
    /* MPI 3.x fallback: MPI_Info_get uses (info, key, valuelen, value, &flag) */
    /* where valuelen is input max length, value is output buffer */
    int vlen = *valuelen - 1;  /* MPI_Info_get expects max value length excluding null */
    if (vlen < 0) vlen = 0;
    int ret = MPI_Info_get(info, key, vlen, value, &f);
    if (f) {
        *valuelen = (int32_t)strlen(value);
    }
#endif
    *flag = f;
    return ret;
}

/* ============================================================
 * Group Operations
 * ============================================================ */

int ferrompi_comm_group(int32_t comm_handle, int32_t* group_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    MPI_Group g;
    int ret = MPI_Comm_group(comm, &g);
    if (ret == MPI_SUCCESS) {
        *group_handle = alloc_group(g);
        if (*group_handle < 0) {
            MPI_Group_free(&g);
            return FERROMPI_ERR_GROUPS_FULL;
        }
    }
    return ret;
}

int ferrompi_group_incl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle) {
    if (n < 0) {
        return MPI_ERR_ARG;
    }
    MPI_Group group = get_group(group_handle);
    MPI_Group newgroup;
    int ret = MPI_Group_incl(group, (int)n, (const int*)ranks, &newgroup);
    if (ret == MPI_SUCCESS) {
        *newgroup_handle = alloc_group(newgroup);
        if (*newgroup_handle < 0) {
            MPI_Group_free(&newgroup);
            return FERROMPI_ERR_GROUPS_FULL;
        }
    }
    return ret;
}

int ferrompi_group_excl(int32_t group_handle, int32_t n, const int32_t* ranks, int32_t* newgroup_handle) {
    if (n < 0) {
        return MPI_ERR_ARG;
    }
    MPI_Group group = get_group(group_handle);
    MPI_Group newgroup;
    int ret = MPI_Group_excl(group, (int)n, (const int*)ranks, &newgroup);
    if (ret == MPI_SUCCESS) {
        *newgroup_handle = alloc_group(newgroup);
        if (*newgroup_handle < 0) {
            MPI_Group_free(&newgroup);
            return FERROMPI_ERR_GROUPS_FULL;
        }
    }
    return ret;
}

int ferrompi_group_free(int32_t group_handle) {
    // Slot 0 is reserved for MPI_GROUP_EMPTY — never free it.
    if (group_handle <= 0) {
        return MPI_SUCCESS;
    }
    if (group_handle >= MAX_GROUPS ||
            !atomic_load_explicit(&group_used[group_handle],
                                  memory_order_acquire)) {
        return MPI_SUCCESS;
    }
    int ret = MPI_Group_free(&group_table[group_handle]);
    free_group(group_handle);
    return ret;
}

int ferrompi_group_size(int32_t group_handle, int32_t* size) {
    MPI_Group group = get_group(group_handle);
    int s;
    int ret = MPI_Group_size(group, &s);
    if (ret == MPI_SUCCESS) {
        *size = (int32_t)s;
    }
    return ret;
}

int ferrompi_group_rank(int32_t group_handle, int32_t* rank) {
    MPI_Group group = get_group(group_handle);
    int r;
    int ret = MPI_Group_rank(group, &r);
    if (ret == MPI_SUCCESS) {
        // Normalize MPI_UNDEFINED to -1 for portable Rust-side comparison.
        *rank = (r == MPI_UNDEFINED) ? -1 : (int32_t)r;
    }
    return ret;
}

int ferrompi_group_union(int32_t g1_h, int32_t g2_h, int32_t* out_h) {
    MPI_Group g1 = get_group(g1_h);
    MPI_Group g2 = get_group(g2_h);
    if (g1 == MPI_GROUP_NULL || g2 == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Group new_grp;
    int ret = MPI_Group_union(g1, g2, &new_grp);
    if (ret == MPI_SUCCESS) {
        *out_h = alloc_group(new_grp);
        if (*out_h < 0) { MPI_Group_free(&new_grp); return FERROMPI_ERR_GROUPS_FULL; }
    }
    return ret;
}

int ferrompi_group_intersection(int32_t g1_h, int32_t g2_h, int32_t* out_h) {
    MPI_Group g1 = get_group(g1_h);
    MPI_Group g2 = get_group(g2_h);
    if (g1 == MPI_GROUP_NULL || g2 == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Group new_grp;
    int ret = MPI_Group_intersection(g1, g2, &new_grp);
    if (ret == MPI_SUCCESS) {
        *out_h = alloc_group(new_grp);
        if (*out_h < 0) { MPI_Group_free(&new_grp); return FERROMPI_ERR_GROUPS_FULL; }
    }
    return ret;
}

int ferrompi_group_difference(int32_t g1_h, int32_t g2_h, int32_t* out_h) {
    MPI_Group g1 = get_group(g1_h);
    MPI_Group g2 = get_group(g2_h);
    if (g1 == MPI_GROUP_NULL || g2 == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Group new_grp;
    int ret = MPI_Group_difference(g1, g2, &new_grp);
    if (ret == MPI_SUCCESS) {
        *out_h = alloc_group(new_grp);
        if (*out_h < 0) { MPI_Group_free(&new_grp); return FERROMPI_ERR_GROUPS_FULL; }
    }
    return ret;
}

int ferrompi_group_range_incl(int32_t g_h, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* out_h) {
    if (n < 0) return MPI_ERR_ARG;
    MPI_Group g = get_group(g_h);
    if (g == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Group new_grp;
    int ret = MPI_Group_range_incl(g, n, (int (*)[3])ranges_flat, &new_grp);
    if (ret == MPI_SUCCESS) {
        *out_h = alloc_group(new_grp);
        if (*out_h < 0) { MPI_Group_free(&new_grp); return FERROMPI_ERR_GROUPS_FULL; }
    }
    return ret;
}

int ferrompi_group_range_excl(int32_t g_h, int32_t n,
                               const int32_t* ranges_flat,
                               int32_t* out_h) {
    if (n < 0) return MPI_ERR_ARG;
    MPI_Group g = get_group(g_h);
    if (g == MPI_GROUP_NULL) return MPI_ERR_ARG;
    MPI_Group new_grp;
    int ret = MPI_Group_range_excl(g, n, (int (*)[3])ranges_flat, &new_grp);
    if (ret == MPI_SUCCESS) {
        *out_h = alloc_group(new_grp);
        if (*out_h < 0) { MPI_Group_free(&new_grp); return FERROMPI_ERR_GROUPS_FULL; }
    }
    return ret;
}

int ferrompi_group_compare(int32_t g1_h, int32_t g2_h, int32_t* result) {
    MPI_Group g1 = get_group(g1_h);
    MPI_Group g2 = get_group(g2_h);
    if (g1 == MPI_GROUP_NULL || g2 == MPI_GROUP_NULL) return MPI_ERR_ARG;
    int mpi_result;
    int ret = MPI_Group_compare(g1, g2, &mpi_result);
    if (ret != MPI_SUCCESS) return ret;
    if (mpi_result == MPI_IDENT)        *result = 0;
    else if (mpi_result == MPI_SIMILAR) *result = 1;
    else if (mpi_result == MPI_UNEQUAL) *result = 2;
    else                                return MPI_ERR_INTERN;
    return MPI_SUCCESS;
}

int ferrompi_group_translate_ranks(int32_t g1_h, int32_t n,
                                   const int32_t* ranks1,
                                   int32_t g2_h, int32_t* ranks2) {
    if (n < 0) return MPI_ERR_ARG;
    MPI_Group g1 = get_group(g1_h);
    MPI_Group g2 = get_group(g2_h);
    if (g1 == MPI_GROUP_NULL || g2 == MPI_GROUP_NULL) return MPI_ERR_ARG;
    int ret = MPI_Group_translate_ranks(g1, n, ranks1, g2, ranks2);
    if (ret != MPI_SUCCESS) return ret;
    /* Normalize MPI_UNDEFINED to -1 so the Rust side has a single
     * portable sentinel. The MPI standard does not fix MPI_UNDEFINED's
     * value across implementations, so this shim always translates it
     * to -1, as the other shims in this file do. */
    for (int32_t i = 0; i < n; i++) {
        if (ranks2[i] == MPI_UNDEFINED) ranks2[i] = -1;
    }
    return MPI_SUCCESS;
}

/* ============================================================
 * Error Information
 * ============================================================ */

_Static_assert(MPI_MAX_ERROR_STRING <= 512, "MPI_MAX_ERROR_STRING exceeds the 512-byte buffer of Error::from_code");
int ferrompi_error_info(int code, int32_t* error_class, char* message, int32_t* msg_len) {
    int cls;
    int ret = MPI_Error_class(code, &cls);
    if (ret != MPI_SUCCESS) return ret;
    *error_class = (int32_t)cls;

    int len;
    ret = MPI_Error_string(code, message, &len);
    if (ret != MPI_SUCCESS) return ret;
    *msg_len = (int32_t)len;
    return MPI_SUCCESS;
}

/* Error class VALUES are not fixed by the MPI standard (only MPI_SUCCESS = 0
 * is); MPICH-derived and Open MPI libraries number the rest differently.
 * This compares `error_class` against the linked library's own MPI_ERR_*
 * constants and returns a ferrompi-stable index in MpiErrorClass's Rust
 * declaration order, or -1 if unrecognized. It makes no MPI call. */
int32_t ferrompi_error_class_index(int error_class) {
    if (error_class == MPI_SUCCESS) return 0;
    if (error_class == MPI_ERR_BUFFER) return 1;
    if (error_class == MPI_ERR_COUNT) return 2;
    if (error_class == MPI_ERR_TYPE) return 3;
    if (error_class == MPI_ERR_TAG) return 4;
    if (error_class == MPI_ERR_COMM) return 5;
    if (error_class == MPI_ERR_RANK) return 6;
    if (error_class == MPI_ERR_REQUEST) return 7;
    if (error_class == MPI_ERR_ROOT) return 8;
    if (error_class == MPI_ERR_GROUP) return 9;
    if (error_class == MPI_ERR_OP) return 10;
    if (error_class == MPI_ERR_TOPOLOGY) return 11;
    if (error_class == MPI_ERR_DIMS) return 12;
    if (error_class == MPI_ERR_ARG) return 13;
    if (error_class == MPI_ERR_UNKNOWN) return 14;
    if (error_class == MPI_ERR_TRUNCATE) return 15;
    if (error_class == MPI_ERR_OTHER) return 16;
    if (error_class == MPI_ERR_INTERN) return 17;
    if (error_class == MPI_ERR_IN_STATUS) return 18;
    if (error_class == MPI_ERR_PENDING) return 19;
    if (error_class == MPI_ERR_WIN) return 20;
    if (error_class == MPI_ERR_INFO) return 21;
    if (error_class == MPI_ERR_FILE) return 22;
    return -1;
}

/* ============================================================
 * Request Management
 * ============================================================ */

// Copies each request's post-call MPI_Request value back into its handle's
// request-table slot, whatever the batch call's return code, and frees the
// slot when MPI nulled it there, marking done[i]. The caller has already set
// done[i] for the requests the call itself reported complete — this is what
// covers persistent requests, since MPI leaves those inactive rather than
// null on completion; for one of those, this also clears its REQUEST_ACTIVE
// bit. A handle that does not resolve through get_request_ptr (the -1
// sentinel, or a stale/bad handle) is skipped.
static void write_back(int64_t count, const int64_t* handles,
                        const MPI_Request* reqs, uint8_t* done) {
    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(handles[i]);
        if (!req) continue;
        *req = reqs[i];
        if (*req == MPI_REQUEST_NULL) {
            free_request(handles[i]);
            done[i] = 1;
        } else if (done[i]) {
            mark_inactive(handles[i]);
        }
    }
}

int ferrompi_wait(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int ret = MPI_Wait(req, MPI_STATUS_IGNORE);
    abort_if_pending_after_failure(ret);
    // MPI_Wait completed the request whatever it returned: a nonblocking
    // request is freed; a persistent one is inactive, or freed too by Open
    // MPI when it failed.
    if (*req == MPI_REQUEST_NULL) {
        free_request(request_handle);
    } else {
        mark_inactive(request_handle);
    }
    return ret;
}

int ferrompi_test(int64_t request_handle, int32_t* flag) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_ERR_REQUEST;
    }
    int f = 0;
    int ret = MPI_Test(req, &f, MPI_STATUS_IGNORE);
    abort_if_pending_after_failure(ret);
    // MPI frees a nonblocking request that completes whether or not MPI_Test
    // itself reports an error (e.g. a truncated receive still nulls the
    // request), so free the slot and report completion on that condition
    // rather than on ret == MPI_SUCCESS. A persistent request MPI completed
    // with an error is inactive too, so flag reports f whatever ret is.
    if (*req == MPI_REQUEST_NULL) {
        free_request(request_handle);
        *flag = 1;
    } else {
        *flag = f;
        if (f) {
            mark_inactive(request_handle);
        }
    }
    return ret;
}

int ferrompi_waitall(int64_t count, const int64_t* request_handles, uint8_t* done,
                      int64_t* failed_index) {
    *failed_index = -1;
    if (count <= 0) return MPI_SUCCESS;
    if (count > INT_MAX) return MPI_ERR_COUNT;

    // Stack scratch for small batches; heap fallback only above the threshold.
    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;

    MPI_Status stack_sts[FERROMPI_REQ_STACK];
    MPI_Status* sts = (count <= FERROMPI_REQ_STACK)
        ? stack_sts
        : (MPI_Status*)malloc((size_t)count * sizeof(MPI_Status));
    if (!sts) {
        if (reqs != stack_reqs) free(reqs);
        return MPI_ERR_NO_MEM;
    }

    for (int64_t i = 0; i < count; i++) {
        done[i] = 0;
        // Insurance: MPI_ERR_PENDING if a library leaves this status unfilled.
        sts[i].MPI_ERROR = MPI_ERR_PENDING;
        if (request_handles[i] == -1) {
            reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            if (sts != stack_sts) free(sts);
            if (reqs != stack_reqs) free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }

    int ret = MPI_Waitall((int)count, reqs, sts);
    abort_if_pending_after_failure(ret);
    if (ret == MPI_ERR_IN_STATUS) {
        for (int64_t i = 0; i < count; i++) {
            abort_if_pending_after_failure(sts[i].MPI_ERROR);
        }
    }

    // Whatever ret is, mark done[i] for every request MPI completed: all of
    // them on success, or on MPI_ERR_IN_STATUS the ones whose own status
    // error is not MPI_ERR_PENDING (MPICH stops at the first failure and
    // reports the rest pending; Open MPI completes them all).
    for (int64_t i = 0; i < count; i++) {
        done[i] = (ret == MPI_SUCCESS)
            || (ret == MPI_ERR_IN_STATUS && sts[i].MPI_ERROR != MPI_ERR_PENDING);
    }
    write_back(count, request_handles, reqs, done);

    // Report the first request whose own status carries the real error, so
    // the caller sees that request's class/code instead of the opaque
    // MPI_ERR_IN_STATUS wrapper. Open MPI can also return MPI_SUCCESS for
    // persistent requests that had already finished and leave a failed
    // request's error only in its status; a conforming MPI leaves the
    // MPI_ERR_PENDING pre-fill or writes MPI_SUCCESS there, so the scan
    // cannot misfire on success.
    if (ret == MPI_ERR_IN_STATUS || ret == MPI_SUCCESS) {
        for (int64_t i = 0; i < count; i++) {
            if (sts[i].MPI_ERROR != MPI_SUCCESS && sts[i].MPI_ERROR != MPI_ERR_PENDING) {
                *failed_index = i;
                ret = sts[i].MPI_ERROR;
                break;
            }
        }
    }

    if (sts != stack_sts) free(sts);
    if (reqs != stack_reqs) free(reqs);
    return ret;
}

int ferrompi_start(int64_t request_handle) {
    int64_t slot = request_slot(request_handle);
    if (slot < 0) return MPI_ERR_REQUEST;
    MPI_Request* req = &request_table[slot];
    int ret = MPI_Start(req);
    if (ret == MPI_SUCCESS) {
        request_state[slot] |= REQUEST_ACTIVE;
    }
    return ret;
}

int ferrompi_startall(int64_t count, const int64_t* request_handles, uint8_t* started) {
    if (count <= 0) return MPI_SUCCESS;
    if (count > INT_MAX) return MPI_ERR_COUNT;

    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;

    for (int64_t i = 0; i < count; i++) {
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            if (reqs != stack_reqs) free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }

    int ret = MPI_Startall((int)count, reqs);

    // MPI_Startall may have started any subset before failing (it is MPI_Start
    // on each request, in some order), so every request it was given counts as
    // started: a later wait on one that never started returns at once.
    for (int64_t i = 0; i < count; i++) {
        int64_t slot = request_slot(request_handles[i]);
        if (slot < 0) continue;
        MPI_Request* req = &request_table[slot];
        *req = reqs[i];
        request_state[slot] |= REQUEST_ACTIVE;
        started[i] = 1;
    }

    if (reqs != stack_reqs) free(reqs);
    return ret;
}

int ferrompi_request_free(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) {
        return MPI_SUCCESS;  // Already freed
    }
    if (*req != MPI_REQUEST_NULL) {
        int ret = MPI_Request_free(req);
        if (ret != MPI_SUCCESS) return ret;
    }
    free_request(request_handle);
    return MPI_SUCCESS;
}

int ferrompi_request_get_status(int64_t request_handle, int32_t* flag) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) return MPI_ERR_REQUEST;
    int f;
    int ret = MPI_Request_get_status(*req, &f, MPI_STATUS_IGNORE);
    *flag = (int32_t)f;
    return ret;
}

int ferrompi_cancel(int64_t request_handle) {
    MPI_Request* req = get_request_ptr(request_handle);
    if (!req) return MPI_ERR_REQUEST;
    return MPI_Cancel(req);
}

int ferrompi_waitany(int64_t count, const int64_t* request_handles,
                     int32_t* index, uint8_t* done) {
    *index = -1;
    if (count <= 0) { return MPI_SUCCESS; }
    if (count > INT_MAX) return MPI_ERR_COUNT;
    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;
    for (int64_t i = 0; i < count; i++) {
        done[i] = 0;
        if (request_handles[i] == -1) {
            reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) { if (reqs != stack_reqs) free(reqs); return MPI_ERR_REQUEST; }
        reqs[i] = *req;
    }
    int idx = MPI_UNDEFINED;
    int ret = MPI_Waitany((int)count, reqs, &idx, MPI_STATUS_IGNORE);
    abort_if_pending_after_failure(ret);
    if (idx != MPI_UNDEFINED) {
        done[idx] = 1;
    }
    *index = (idx == MPI_UNDEFINED) ? -1 : (int32_t)idx;
    write_back(count, request_handles, reqs, done);
    if (reqs != stack_reqs) free(reqs);
    return ret;
}

int ferrompi_waitsome(int64_t count, const int64_t* request_handles,
                      int64_t* outcount, int32_t* indices, uint8_t* done,
                      int64_t* failed_index) {
    *failed_index = -1;
    if (count <= 0) { *outcount = -1; return MPI_SUCCESS; }
    if (count > INT_MAX) return MPI_ERR_COUNT;
    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    int stack_idx[FERROMPI_REQ_STACK];
    MPI_Status stack_sts[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;
    int* tmp_indices = (count <= FERROMPI_REQ_STACK)
        ? stack_idx
        : (int*)malloc((size_t)count * sizeof(int));
    if (!tmp_indices) { if (reqs != stack_reqs) free(reqs); return MPI_ERR_NO_MEM; }
    MPI_Status* sts = (count <= FERROMPI_REQ_STACK)
        ? stack_sts
        : (MPI_Status*)malloc((size_t)count * sizeof(MPI_Status));
    if (!sts) {
        if (tmp_indices != stack_idx) free(tmp_indices);
        if (reqs != stack_reqs) free(reqs);
        return MPI_ERR_NO_MEM;
    }
    for (int64_t i = 0; i < count; i++) {
        done[i] = 0;
        if (request_handles[i] == -1) {
            reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            if (sts != stack_sts) free(sts);
            if (tmp_indices != stack_idx) free(tmp_indices);
            if (reqs != stack_reqs) free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }
    int out = MPI_UNDEFINED;
    int ret = MPI_Waitsome((int)count, reqs, &out, tmp_indices, sts);
    abort_if_pending_after_failure(ret);
    if (ret == MPI_ERR_IN_STATUS) {
        for (int i = 0; i < out; i++) {
            abort_if_pending_after_failure(sts[i].MPI_ERROR);
        }
    }
    if (out == MPI_UNDEFINED) {
        *outcount = -1;
    } else {
        *outcount = (int64_t)out;
        for (int i = 0; i < out; i++) {
            indices[i] = (int32_t)tmp_indices[i];
            done[tmp_indices[i]] = 1;
        }
    }
    write_back(count, request_handles, reqs, done);

    // statuses[k] belongs to request indices[k] (completion order, not
    // caller order); report the caller's index of the first real failure.
    if (ret == MPI_ERR_IN_STATUS) {
        for (int i = 0; i < out; i++) {
            if (sts[i].MPI_ERROR != MPI_SUCCESS && sts[i].MPI_ERROR != MPI_ERR_PENDING) {
                *failed_index = tmp_indices[i];
                ret = sts[i].MPI_ERROR;
                break;
            }
        }
    }

    if (sts != stack_sts) free(sts);
    if (tmp_indices != stack_idx) free(tmp_indices);
    if (reqs != stack_reqs) free(reqs);
    return ret;
}

int ferrompi_testany(int64_t count, const int64_t* request_handles,
                     int32_t* index, int32_t* flag, uint8_t* done) {
    *index = -1;
    if (count <= 0) { *flag = 1; return MPI_SUCCESS; }
    if (count > INT_MAX) return MPI_ERR_COUNT;
    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;
    for (int64_t i = 0; i < count; i++) {
        done[i] = 0;
        if (request_handles[i] == -1) {
            reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) { if (reqs != stack_reqs) free(reqs); return MPI_ERR_REQUEST; }
        reqs[i] = *req;
    }
    int idx = MPI_UNDEFINED;
    int f = 0;
    int ret = MPI_Testany((int)count, reqs, &idx, &f, MPI_STATUS_IGNORE);
    abort_if_pending_after_failure(ret);
    if (f && idx != MPI_UNDEFINED) {
        done[idx] = 1;
    }
    *flag = (int32_t)f;
    *index = (idx == MPI_UNDEFINED) ? -1 : (int32_t)idx;
    write_back(count, request_handles, reqs, done);
    if (reqs != stack_reqs) free(reqs);
    return ret;
}

int ferrompi_testsome(int64_t count, const int64_t* request_handles,
                      int64_t* outcount, int32_t* indices, uint8_t* done,
                      int64_t* failed_index) {
    *failed_index = -1;
    if (count <= 0) { *outcount = -1; return MPI_SUCCESS; }
    if (count > INT_MAX) return MPI_ERR_COUNT;
    MPI_Request stack_reqs[FERROMPI_REQ_STACK];
    int stack_idx[FERROMPI_REQ_STACK];
    MPI_Status stack_sts[FERROMPI_REQ_STACK];
    MPI_Request* reqs = (count <= FERROMPI_REQ_STACK)
        ? stack_reqs
        : (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!reqs) return MPI_ERR_NO_MEM;
    int* tmp_indices = (count <= FERROMPI_REQ_STACK)
        ? stack_idx
        : (int*)malloc((size_t)count * sizeof(int));
    if (!tmp_indices) { if (reqs != stack_reqs) free(reqs); return MPI_ERR_NO_MEM; }
    MPI_Status* sts = (count <= FERROMPI_REQ_STACK)
        ? stack_sts
        : (MPI_Status*)malloc((size_t)count * sizeof(MPI_Status));
    if (!sts) {
        if (tmp_indices != stack_idx) free(tmp_indices);
        if (reqs != stack_reqs) free(reqs);
        return MPI_ERR_NO_MEM;
    }
    for (int64_t i = 0; i < count; i++) {
        done[i] = 0;
        if (request_handles[i] == -1) {
            reqs[i] = MPI_REQUEST_NULL;
            continue;
        }
        MPI_Request* req = get_request_ptr(request_handles[i]);
        if (!req) {
            if (sts != stack_sts) free(sts);
            if (tmp_indices != stack_idx) free(tmp_indices);
            if (reqs != stack_reqs) free(reqs);
            return MPI_ERR_REQUEST;
        }
        reqs[i] = *req;
    }
    int out = MPI_UNDEFINED;
    int ret = MPI_Testsome((int)count, reqs, &out, tmp_indices, sts);
    abort_if_pending_after_failure(ret);
    if (ret == MPI_ERR_IN_STATUS) {
        for (int i = 0; i < out; i++) {
            abort_if_pending_after_failure(sts[i].MPI_ERROR);
        }
    }
    if (out == MPI_UNDEFINED) {
        *outcount = -1;
    } else {
        *outcount = (int64_t)out;
        for (int i = 0; i < out; i++) {
            indices[i] = (int32_t)tmp_indices[i];
            done[tmp_indices[i]] = 1;
        }
    }
    write_back(count, request_handles, reqs, done);

    // statuses[k] belongs to request indices[k] (completion order, not
    // caller order); report the caller's index of the first real failure.
    if (ret == MPI_ERR_IN_STATUS) {
        for (int i = 0; i < out; i++) {
            if (sts[i].MPI_ERROR != MPI_SUCCESS && sts[i].MPI_ERROR != MPI_ERR_PENDING) {
                *failed_index = tmp_indices[i];
                ret = sts[i].MPI_ERROR;
                break;
            }
        }
    }

    if (sts != stack_sts) free(sts);
    if (tmp_indices != stack_idx) free(tmp_indices);
    if (reqs != stack_reqs) free(reqs);
    return ret;
}

/* ============================================================
 * RMA Window Operations (MPI 3.0+)
 * ============================================================ */

/* Zeroes the calling rank's own segment of a freshly allocated window so
 * that MPI_Win_allocate[_shared] never hands out uninitialised memory. The
 * memset itself is not an MPI call and needs no lock to be valid C, but it
 * must run inside a passive-target epoch: without the lock/sync bracket a
 * peer racing ahead of this rank could read the segment before the memset
 * is visible to it under RMA's relaxed consistency. MPI_MODE_NOCHECK skips
 * lock negotiation: valid because no process holds or will attempt a
 * conflicting lock on this window, since every peer takes only a shared
 * lock_all here and none can leave before the closing barrier below.
 *
 * The middle barrier keeps a fast peer from reading this rank's segment
 * until the memset above has completed. The closing barrier keeps every
 * rank inside the constructor until every peer has left this zeroing
 * epoch, so a post, fence, or lock issued right after construction returns
 * is always legal MPI: no rank can still be mid-epoch on the window.
 *
 * Both barriers are always reached, whether or not the lock/sync calls
 * succeed: only the calls that depend on a failed lock (sync, unlock) are
 * skipped, and the first non-success code is returned only after both
 * barriers. On a failure the window is never freed: MPI_Win_free is
 * collective, and a peer whose own zeroing succeeded still holds a live
 * window that only its own teardown can free. The caller must not call
 * MPI_Win_free here; it reports the leak through FERROMPI_WIN_LEAKED so the
 * Rust side counts the window as alive. */
static int zero_own_segment(MPI_Win win, MPI_Comm comm, void* base, MPI_Aint size) {
    int ret = MPI_SUCCESS;

    int first = MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    int locked = (first == MPI_SUCCESS);
    if (ret == MPI_SUCCESS) ret = first;

    if (size > 0 && base != NULL) memset(base, 0, (size_t)size);

    if (locked) {
        int r = MPI_Win_sync(win);
        if (ret == MPI_SUCCESS) ret = r;
    }

    {
        int r = MPI_Barrier(comm);
        if (ret == MPI_SUCCESS) ret = r;
    }

    if (locked) {
        int r = MPI_Win_sync(win);
        if (ret == MPI_SUCCESS) ret = r;
        r = MPI_Win_unlock_all(win);
        if (ret == MPI_SUCCESS) ret = r;
    }

    {
        int r = MPI_Barrier(comm);
        if (ret == MPI_SUCCESS) ret = r;
    }

    return ret;
}

int ferrompi_win_allocate_shared(int64_t size, int32_t disp_unit, int32_t info_handle,
                                  int32_t comm_handle, void** baseptr, int32_t* win_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    MPI_Info info = (info_handle < 0) ? MPI_INFO_NULL : get_info(info_handle);
    MPI_Win win;
    int ret = MPI_Win_allocate_shared((MPI_Aint)size, disp_unit, info, comm, baseptr, &win);
    if (ret == MPI_SUCCESS) {
        install_errors_return_win(win);
        ret = zero_own_segment(win, comm, *baseptr, (MPI_Aint)size);
        if (ret != MPI_SUCCESS) {
            *win_handle = FERROMPI_WIN_LEAKED;
            return ret;
        }
        *win_handle = alloc_win(win);
        if (*win_handle < 0) {
            *win_handle = FERROMPI_WIN_LEAKED;
            return FERROMPI_ERR_WINDOWS_FULL;
        }
    }
    return ret;
}

int ferrompi_win_create(void* base, int64_t size, int32_t disp_unit, int32_t info_handle,
                         int32_t comm_handle, int32_t* win_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    MPI_Info info = (info_handle < 0) ? MPI_INFO_NULL : get_info(info_handle);
    MPI_Win win;
    int ret = MPI_Win_create(base, (MPI_Aint)size, disp_unit, info, comm, &win);
    if (ret == MPI_SUCCESS) {
        install_errors_return_win(win);
        *win_handle = alloc_win(win);
        if (*win_handle < 0) {
            /* The window exposes the caller's buffer, which the caller gets
             * back once this returns an error while peers can still reach
             * it: it can be neither leaked nor freed on this rank alone
             * (MPI_Win_free is collective). MPI_Abort is only required to
             * make a "best attempt"; it can return (measured: MPICH before
             * the launcher kills the other ranks), so this must not fall
             * through to a return that hands the buffer back to Rust. */
            fputs("ferrompi: window table full after MPI_Win_create; aborting "
                  "because the window exposes the caller's buffer\n", stderr);
            MPI_Abort(comm, MPI_ERR_OTHER);
            abort();
        }
    }
    return ret;
}

int ferrompi_win_allocate(int64_t size, int32_t disp_unit, int32_t info_handle,
                           int32_t comm_handle, void** baseptr, int32_t* win_handle) {
    MPI_Comm comm = get_comm(comm_handle);
    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    MPI_Info info = (info_handle < 0) ? MPI_INFO_NULL : get_info(info_handle);
    MPI_Win win;
    int ret = MPI_Win_allocate((MPI_Aint)size, disp_unit, info, comm, baseptr, &win);
    if (ret == MPI_SUCCESS) {
        install_errors_return_win(win);
        ret = zero_own_segment(win, comm, *baseptr, (MPI_Aint)size);
        if (ret != MPI_SUCCESS) {
            *win_handle = FERROMPI_WIN_LEAKED;
            return ret;
        }
        *win_handle = alloc_win(win);
        if (*win_handle < 0) {
            *win_handle = FERROMPI_WIN_LEAKED;
            return FERROMPI_ERR_WINDOWS_FULL;
        }
    }
    return ret;
}

int ferrompi_win_shared_query(int32_t win_handle, int32_t rank,
                               int64_t* size, int32_t* disp_unit, void** baseptr) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Aint sz;
    int du;
    int ret = MPI_Win_shared_query(win, rank, &sz, &du, baseptr);
    if (ret == MPI_SUCCESS) {
        *size = (int64_t)sz;
        *disp_unit = (int32_t)du;
    }
    return ret;
}

int ferrompi_win_free(int32_t win_handle) {
    MPI_Win* winp = get_win_ptr(win_handle);
    if (!winp) return MPI_SUCCESS;
    int ret = MPI_Win_free(winp);
    free_win(win_handle);
    return ret;
}

int ferrompi_win_fence(int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_fence(assert_val, win);
}

void ferrompi_win_fence_mode_values(int32_t* out) {
    out[0] = MPI_MODE_NOSTORE;
    out[1] = MPI_MODE_NOPUT;
    out[2] = MPI_MODE_NOPRECEDE;
    out[3] = MPI_MODE_NOSUCCEED;
}

int ferrompi_win_lock(int32_t lock_type, int32_t rank, int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    int mpi_lock = (lock_type == FERROMPI_LOCK_SHARED) ? MPI_LOCK_SHARED : MPI_LOCK_EXCLUSIVE;
    return MPI_Win_lock(mpi_lock, rank, assert_val, win);
}

int ferrompi_win_unlock(int32_t rank, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_unlock(rank, win);
}

int ferrompi_win_lock_all(int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_lock_all(assert_val, win);
}

int ferrompi_win_unlock_all(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_unlock_all(win);
}

int ferrompi_win_flush(int32_t rank, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush(rank, win);
}

int ferrompi_win_flush_all(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush_all(win);
}

int ferrompi_win_flush_local(int32_t rank, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush_local(rank, win);
}

int ferrompi_win_flush_local_all(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_flush_local_all(win);
}

int ferrompi_win_sync(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_sync(win);
}

int ferrompi_win_post(int32_t group_handle, int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Group group = get_group(group_handle);
    if (group == MPI_GROUP_NULL) return MPI_ERR_GROUP;
    return MPI_Win_post(group, assert_val, win);
}

int ferrompi_win_start(int32_t group_handle, int32_t assert_val, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Group group = get_group(group_handle);
    if (group == MPI_GROUP_NULL) return MPI_ERR_GROUP;
    return MPI_Win_start(group, assert_val, win);
}

int ferrompi_win_complete(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_complete(win);
}

int ferrompi_win_wait(int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    return MPI_Win_wait(win);
}

int ferrompi_win_test(int32_t win_handle, int32_t* flag) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    int mpi_flag = 0;
    int ret = MPI_Win_test(win, &mpi_flag);
    *flag = (int32_t)mpi_flag;
    return ret;
}

void ferrompi_win_pscw_mode_values(int32_t* out) {
    out[0] = MPI_MODE_NOCHECK;
    out[1] = MPI_MODE_NOSTORE;
    out[2] = MPI_MODE_NOPUT;
}

int ferrompi_put(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Put_c(origin, (MPI_Count)origin_count, origin_dt,
                         target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                         target_dt, win);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Put(origin, (int)origin_count, origin_dt,
                   target_rank, (MPI_Aint)target_disp, (int)target_count,
                   target_dt, win);
}

int ferrompi_rput(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Rput_c(origin, (MPI_Count)origin_count, origin_dt,
                         target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                         target_dt, win, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Rput(origin, (int)origin_count, origin_dt,
                       target_rank, (MPI_Aint)target_disp, (int)target_count,
                       target_dt, win, &req);
    }
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    return ret;
}

int ferrompi_get(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                 int32_t target_rank, int64_t target_disp, int64_t target_count,
                 int32_t target_dt_tag, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Get_c(origin, (MPI_Count)origin_count, origin_dt,
                         target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                         target_dt, win);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Get(origin, (int)origin_count, origin_dt,
                   target_rank, (MPI_Aint)target_disp, (int)target_count,
                   target_dt, win);
}

int ferrompi_rget(void* origin, int64_t origin_count, int32_t origin_dt_tag,
                  int32_t target_rank, int64_t target_disp, int64_t target_count,
                  int32_t target_dt_tag, int32_t win_handle, int64_t* request_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Request req;
    int ret;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Rget_c(origin, (MPI_Count)origin_count, origin_dt,
                         target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                         target_dt, win, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Rget(origin, (int)origin_count, origin_dt,
                       target_rank, (MPI_Aint)target_disp, (int)target_count,
                       target_dt, win, &req);
    }
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    return ret;
}

int ferrompi_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                        int32_t target_rank, int64_t target_disp, int64_t target_count,
                        int32_t target_dt_tag, int32_t op_tag, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op_tag);
    if (mpi_op == MPI_OP_NULL) return MPI_ERR_OP;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Accumulate_c(origin, (MPI_Count)origin_count, origin_dt,
                                target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                                target_dt, mpi_op, win);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Accumulate(origin, (int)origin_count, origin_dt,
                          target_rank, (MPI_Aint)target_disp, (int)target_count,
                          target_dt, mpi_op, win);
}

int ferrompi_raccumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                         int32_t target_rank, int64_t target_disp, int64_t target_count,
                         int32_t target_dt_tag, int32_t op_tag, int32_t win_handle,
                         int64_t* request_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op_tag);
    if (mpi_op == MPI_OP_NULL) return MPI_ERR_OP;
    MPI_Request req;
    int ret;
    if (origin_count > INT_MAX || target_count > INT_MAX) {
#if MPI_VERSION >= 4
        ret = MPI_Raccumulate_c(origin, (MPI_Count)origin_count, origin_dt,
                                target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                                target_dt, mpi_op, win, &req);
#else
        return MPI_ERR_COUNT;
#endif
    } else {
        ret = MPI_Raccumulate(origin, (int)origin_count, origin_dt,
                              target_rank, (MPI_Aint)target_disp, (int)target_count,
                              target_dt, mpi_op, win, &req);
    }
    if (ret == MPI_SUCCESS) {
        *request_handle = alloc_request(req, 0);
        if (*request_handle < 0) {
            complete_unregistered_request(&req);
            return FERROMPI_ERR_REQUESTS_FULL;
        }
    }
    return ret;
}

int ferrompi_get_accumulate(const void* origin, int64_t origin_count, int32_t origin_dt_tag,
                            void* result, int64_t result_count, int32_t result_dt_tag,
                            int32_t target_rank, int64_t target_disp, int64_t target_count,
                            int32_t target_dt_tag, int32_t op_tag, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype origin_dt = get_datatype(origin_dt_tag);
    if (origin_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype result_dt = get_datatype(result_dt_tag);
    if (result_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype target_dt = get_datatype(target_dt_tag);
    if (target_dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op_tag);
    if (mpi_op == MPI_OP_NULL) return MPI_ERR_OP;
    if (origin_count > INT_MAX || target_count > INT_MAX || result_count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Get_accumulate_c(origin, (MPI_Count)origin_count, origin_dt,
                                    result, (MPI_Count)result_count, result_dt,
                                    target_rank, (MPI_Aint)target_disp, (MPI_Count)target_count,
                                    target_dt, mpi_op, win);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Get_accumulate(origin, (int)origin_count, origin_dt,
                              result, (int)result_count, result_dt,
                              target_rank, (MPI_Aint)target_disp, (int)target_count,
                              target_dt, mpi_op, win);
}

int ferrompi_fetch_and_op(const void* origin, void* result, int32_t dt_tag,
                          int32_t target_rank, int64_t target_disp,
                          int32_t op_tag, int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype dt = get_datatype(dt_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Op mpi_op = get_op(op_tag);
    if (mpi_op == MPI_OP_NULL) return MPI_ERR_OP;
    return MPI_Fetch_and_op(origin, result, dt, target_rank, (MPI_Aint)target_disp, mpi_op, win);
}

int ferrompi_compare_and_swap(const void* origin, const void* compare, void* result,
                               int32_t dt_tag, int32_t target_rank, int64_t target_disp,
                               int32_t win_handle) {
    MPI_Win win = get_win(win_handle);
    if (win == MPI_WIN_NULL) return MPI_ERR_WIN;
    MPI_Datatype dt = get_datatype(dt_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    return MPI_Compare_and_swap(origin, compare, result, dt, target_rank,
                                (MPI_Aint)target_disp, win);
}

/* ============================================================
 * Utility Functions
 * ============================================================ */

_Static_assert(MPI_MAX_LIBRARY_VERSION_STRING <= 8192, "MPI_MAX_LIBRARY_VERSION_STRING exceeds the 8192-byte buffer of Mpi::library_version");
int ferrompi_get_library_version(char* buf, int32_t* len) {
    int l = 0;
    int ret = MPI_Get_library_version(buf, &l);
    if (ret == MPI_SUCCESS) {
        *len = l;
    } else {
        *len = 0;
    }
    return ret;
}

int ferrompi_get_version(char* version, int32_t* len) {
    int version_num, subversion_num;
    int ret = MPI_Get_version(&version_num, &subversion_num);
    if (ret == MPI_SUCCESS) {
        int n = snprintf(version, 256, "MPI %d.%d", version_num, subversion_num);
        /* Clamp: snprintf can return a value > buffer size per C standard */
        *len = (n < 0) ? 0 : (n > 255 ? 255 : n);
    }
    return ret;
}

_Static_assert(MPI_MAX_PROCESSOR_NAME <= 256, "MPI_MAX_PROCESSOR_NAME exceeds the 256-byte buffer of Communicator::processor_name");
int ferrompi_get_processor_name(char* name, int32_t* len) {
    int l = 0;
    int ret = MPI_Get_processor_name(name, &l);
    if (ret == MPI_SUCCESS) {
        *len = l;
    } else {
        *len = 0;
    }
    return ret;
}

double ferrompi_wtime(void) {
    return MPI_Wtime();
}

int ferrompi_abort(int32_t comm_handle, int32_t errorcode) {
    MPI_Comm comm = get_comm(comm_handle);
    return MPI_Abort(comm, errorcode);
}

/* ============================================================
 * Custom Datatype Operations
 * ============================================================ */

int ferrompi_type_contiguous(int32_t count, int32_t basetype_tag,
                              int32_t* newtype_handle) {
    MPI_Datatype base = get_datatype(basetype_tag);
    if (base == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype new_t;
    int ret = MPI_Type_contiguous((int)count, base, &new_t);
    if (ret != MPI_SUCCESS) return ret;
    ret = MPI_Type_commit(&new_t);
    if (ret != MPI_SUCCESS) {
        MPI_Type_free(&new_t);
        return ret;
    }
    *newtype_handle = alloc_datatype(new_t);
    if (*newtype_handle < 0) {
        MPI_Type_free(&new_t);
        return FERROMPI_ERR_DATATYPES_FULL;
    }
    return MPI_SUCCESS;
}

int ferrompi_type_vector(int32_t count, int32_t blocklength,
                         int32_t stride, int32_t basetype_tag,
                         int32_t* newtype_handle) {
    MPI_Datatype base = get_datatype(basetype_tag);
    if (base == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype new_t;
    int ret = MPI_Type_vector((int)count, (int)blocklength, (int)stride, base, &new_t);
    if (ret != MPI_SUCCESS) return ret;
    ret = MPI_Type_commit(&new_t);
    if (ret != MPI_SUCCESS) {
        MPI_Type_free(&new_t);
        return ret;
    }
    *newtype_handle = alloc_datatype(new_t);
    if (*newtype_handle < 0) {
        MPI_Type_free(&new_t);
        return FERROMPI_ERR_DATATYPES_FULL;
    }
    return MPI_SUCCESS;
}

int ferrompi_type_create_struct(int32_t count,
                                const int32_t* blocklengths,
                                const int64_t* displacements,
                                const int32_t* basetype_tags,
                                int32_t* newtype_handle) {
    /* Reject an empty field list before the stack arrays are filled. */
    if (count <= 0) return MPI_ERR_ARG;
    MPI_Aint stack_disp[32];
    MPI_Datatype stack_types[32];
    MPI_Aint* disp = stack_disp;
    MPI_Datatype* types = stack_types;
    int heap_alloc = 0;
    if (count > 32) {
        disp = (MPI_Aint*) malloc(sizeof(MPI_Aint) * (size_t)count);
        types = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * (size_t)count);
        if (!disp || !types) {
            free(disp);
            free(types);
            return MPI_ERR_NO_MEM;
        }
        heap_alloc = 1;
    }
    for (int32_t i = 0; i < count; i++) {
        MPI_Datatype t = get_datatype(basetype_tags[i]);
        if (t == MPI_DATATYPE_NULL) {
            if (heap_alloc) { free(disp); free(types); }
            return MPI_ERR_TYPE;
        }
        disp[i] = (MPI_Aint) displacements[i];
        types[i] = t;
    }
    MPI_Datatype new_t;
    int ret = MPI_Type_create_struct((int)count, (int*)blocklengths, disp, types, &new_t);
    if (heap_alloc) { free(disp); free(types); }
    if (ret != MPI_SUCCESS) return ret;
    ret = MPI_Type_commit(&new_t);
    if (ret != MPI_SUCCESS) {
        MPI_Type_free(&new_t);
        return ret;
    }
    *newtype_handle = alloc_datatype(new_t);
    if (*newtype_handle < 0) {
        MPI_Type_free(&new_t);
        return FERROMPI_ERR_DATATYPES_FULL;
    }
    return MPI_SUCCESS;
}

int ferrompi_type_create_resized(int32_t old_h, int64_t lb,
                                 int64_t extent, int32_t* newtype_handle) {
    MPI_Datatype old_t = get_datatype_committed(old_h);
    if (old_t == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Datatype new_t;
    int ret = MPI_Type_create_resized(old_t, (MPI_Aint)lb,
                                      (MPI_Aint)extent, &new_t);
    if (ret != MPI_SUCCESS) return ret;
    ret = MPI_Type_commit(&new_t);
    if (ret != MPI_SUCCESS) {
        MPI_Type_free(&new_t);
        return ret;
    }
    *newtype_handle = alloc_datatype(new_t);
    if (*newtype_handle < 0) {
        MPI_Type_free(&new_t);
        return FERROMPI_ERR_DATATYPES_FULL;
    }
    return MPI_SUCCESS;
}

int ferrompi_type_get_extents(int32_t type_handle, int64_t* extent,
                              int64_t* true_lb, int64_t* true_extent) {
    MPI_Datatype dt = get_datatype_committed(type_handle);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    MPI_Aint lb, ext;
    int ret = MPI_Type_get_extent(dt, &lb, &ext);
    if (ret != MPI_SUCCESS) return ret;
    MPI_Aint tlb, text;
    ret = MPI_Type_get_true_extent(dt, &tlb, &text);
    if (ret != MPI_SUCCESS) return ret;
    *extent = (int64_t)ext;
    *true_lb = (int64_t)tlb;
    *true_extent = (int64_t)text;
    return MPI_SUCCESS;
}

int ferrompi_type_free(int32_t type_handle) {
    if (type_handle < 0 || type_handle >= MAX_DATATYPES) return MPI_ERR_ARG;
    if (!atomic_load_explicit(&datatype_used[type_handle],
                              memory_order_acquire)) return MPI_SUCCESS;  /* already freed */
    int ret = MPI_Type_free(&datatype_table[type_handle]);
    free_datatype_slot(type_handle);  /* clears slot regardless of MPI_Type_free outcome */
    return ret;
}

/* ============================================================
 * Custom-Datatype Point-to-Point
 * ============================================================ */

int ferrompi_send_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle
) {
    return send_typed(buf, count, get_datatype_committed(datatype_handle), dest, tag, comm_handle);
}

int ferrompi_recv_custom(
    void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int32_t* actual_source,
    int32_t* actual_tag,
    int64_t* actual_count
) {
    return recv_typed(buf, count, get_datatype_committed(datatype_handle), source, tag, comm_handle,
                       actual_source, actual_tag, actual_count);
}

int ferrompi_isend_custom(
    const void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t dest,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    return isend_typed(buf, count, get_datatype_committed(datatype_handle), dest, tag, comm_handle,
                        request_handle);
}

int ferrompi_irecv_custom(
    void* buf,
    int64_t count,
    int32_t datatype_handle,
    int32_t source,
    int32_t tag,
    int32_t comm_handle,
    int64_t* request_handle
) {
    return irecv_typed(buf, count, get_datatype_committed(datatype_handle), source, tag, comm_handle,
                        request_handle);
}

/* ============================================================
 * User-Defined Reduction Op — Trampolines and Shims
 *
 * Strategy: per-op static slot table following the group_table /
 * info_table pattern from ADR-0002.  MAX_OPS distinct trampoline
 * functions are generated by the FERROMPI_DEFINE_OP_TRAMPOLINE macro
 * so that MPI_Op_create receives a unique function pointer per slot,
 * sidestepping the lack of a user-data parameter in the MPI user-
 * function signature.
 *
 * Drop ordering (ADR-0005 Decision 3):
 *   ferrompi_op_free → MPI_Op_free → ferrompi_op_drop_closure → free_op_slot
 * This guarantees the Rust closure is alive for the full lifetime of
 * the MPI_Op handle.
 *
 * The static tables (op_table, op_used) are declared at the top of this
 * file, alongside the other slot tables.  The Rust closure for each slot
 * lives entirely in Rust's own registry (src/op.rs); C only carries the
 * slot number.
 * ============================================================ */

/* Forward declarations of C-invoked Rust callback and helper. */
extern void rust_user_op_invoke(int32_t slot, void* invec, void* inoutvec,
                                int len);
/* Called from ferrompi_op_free to drop the boxed Rust closure. */
extern void ferrompi_op_drop_closure(int32_t slot);

/* ---- slot helpers ---- */

static int32_t alloc_op_slot(void) {
    int hint = atomic_load_explicit(&next_op_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_OPS; i++) {
        int idx = (hint + i) % MAX_OPS;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
                &op_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            atomic_store_explicit(&next_op_hint, (idx + 1) % MAX_OPS,
                                  memory_order_relaxed);
            return (int32_t)idx;
        }
    }
    return -1;  /* table full */
}

static void free_op_slot(int32_t slot) {
    if (slot >= 0 && slot < MAX_OPS) {
        /* Release store: any thread that later acquires op_used == 0 will also
         * observe the NULL op (no dangling handle visible). */
        atomic_store_explicit(&op_table[slot], MPI_OP_NULL, memory_order_release);
        atomic_store_explicit(&op_used[slot], 0, memory_order_release);
    }
}

/* ---- 16 distinct trampoline functions (ADR-0005 Decision 5) ---- */

#define FERROMPI_DEFINE_OP_TRAMPOLINE(N)                              \
static void ferrompi_user_op_trampoline_##N(                          \
    void* invec, void* inoutvec, int* len, MPI_Datatype* dt) {        \
    (void)dt;                                                         \
    rust_user_op_invoke(N, invec, inoutvec, *len);                    \
}

FERROMPI_DEFINE_OP_TRAMPOLINE(0)
FERROMPI_DEFINE_OP_TRAMPOLINE(1)
FERROMPI_DEFINE_OP_TRAMPOLINE(2)
FERROMPI_DEFINE_OP_TRAMPOLINE(3)
FERROMPI_DEFINE_OP_TRAMPOLINE(4)
FERROMPI_DEFINE_OP_TRAMPOLINE(5)
FERROMPI_DEFINE_OP_TRAMPOLINE(6)
FERROMPI_DEFINE_OP_TRAMPOLINE(7)
FERROMPI_DEFINE_OP_TRAMPOLINE(8)
FERROMPI_DEFINE_OP_TRAMPOLINE(9)
FERROMPI_DEFINE_OP_TRAMPOLINE(10)
FERROMPI_DEFINE_OP_TRAMPOLINE(11)
FERROMPI_DEFINE_OP_TRAMPOLINE(12)
FERROMPI_DEFINE_OP_TRAMPOLINE(13)
FERROMPI_DEFINE_OP_TRAMPOLINE(14)
FERROMPI_DEFINE_OP_TRAMPOLINE(15)

/* Static table of trampoline pointers indexed by slot. */
static MPI_User_function* const ferrompi_user_op_trampolines[MAX_OPS] = {
    ferrompi_user_op_trampoline_0,
    ferrompi_user_op_trampoline_1,
    ferrompi_user_op_trampoline_2,
    ferrompi_user_op_trampoline_3,
    ferrompi_user_op_trampoline_4,
    ferrompi_user_op_trampoline_5,
    ferrompi_user_op_trampoline_6,
    ferrompi_user_op_trampoline_7,
    ferrompi_user_op_trampoline_8,
    ferrompi_user_op_trampoline_9,
    ferrompi_user_op_trampoline_10,
    ferrompi_user_op_trampoline_11,
    ferrompi_user_op_trampoline_12,
    ferrompi_user_op_trampoline_13,
    ferrompi_user_op_trampoline_14,
    ferrompi_user_op_trampoline_15,
};

/* ---- Public shims called from Rust ---- */

/* Allocate a free slot; writes slot index to *out_slot.
 * Returns MPI_SUCCESS on success, FERROMPI_ERR_OPS_FULL if table is full. */
int ferrompi_op_alloc_slot(int32_t* out_slot) {
    int32_t slot = alloc_op_slot();
    if (slot < 0) return FERROMPI_ERR_OPS_FULL;
    *out_slot = slot;
    return MPI_SUCCESS;
}

/* Create an MPI_Op for the given slot; commute=1 → commutative.
 * Stores the MPI_Op in op_table[slot] and writes the slot back to
 * *out_handle (callers use the slot as the handle). */
int ferrompi_op_create_user(int32_t slot, int32_t commute, int32_t* out_handle) {
    if (slot < 0 || slot >= MAX_OPS) return MPI_ERR_ARG;
    if (atomic_load_explicit(&op_used[slot], memory_order_acquire) != 1) return MPI_ERR_ARG;
    MPI_Op op;
    int ret = MPI_Op_create(ferrompi_user_op_trampolines[slot],
                            (int)commute, &op);
    if (ret != MPI_SUCCESS) return ret;
    atomic_store_explicit(&op_table[slot], op, memory_order_release);
    *out_handle = slot;
    return MPI_SUCCESS;
}

/* Free the MPI_Op for the given handle, drop the Rust closure, then
 * clear the slot.  Drop ordering: MPI_Op_free first (ADR-0005 Decision 3). */
int ferrompi_op_free(int32_t handle) {
    if (handle < 0 || handle >= MAX_OPS) return MPI_ERR_ARG;
    if (!atomic_load_explicit(&op_used[handle], memory_order_acquire)) return MPI_SUCCESS;  /* already freed */
    /* Step 1: MPI_Op_free — MPI will not invoke the trampoline after this.
     * Stage through a local because MPI_Op_free takes a non-atomic MPI_Op*;
     * after it returns, tmp == MPI_OP_NULL, which we store back atomically. */
    MPI_Op tmp = atomic_load_explicit(&op_table[handle], memory_order_acquire);
    int ret = MPI_Op_free(&tmp);
    atomic_store_explicit(&op_table[handle], tmp, memory_order_release);
    /* If MPI_Op_free failed (rare: MPI_ERR_OP on a predefined op, or an
     * implementation rejecting a free while the op is referenced by an
     * outstanding non-blocking collective), the op is still live and may
     * be invoked.  Dropping the Rust closure now would leave the trampoline
     * with dangling closure-data pointers (UB on next invocation).  Return
     * the error without releasing closure or slot; the caller must retry
     * the free after the offending collective completes. */
    if (ret != MPI_SUCCESS) {
        return ret;
    }
    /* Step 2: drop the Rust-boxed closure via the Rust callback. */
    ferrompi_op_drop_closure(handle);
    /* Step 3: reclaim the slot (also overwrites op_table[handle] with MPI_OP_NULL). */
    free_op_slot(handle);
    return ret;
}

/* Release the op slot WITHOUT calling MPI_Op_free.
 *
 * Used in rollback paths where ferrompi_op_create_user (MPI_Op_create) failed
 * and the slot therefore holds MPI_OP_NULL.  Calling MPI_Op_free on
 * MPI_OP_NULL is implementation-defined; this shim avoids it entirely.
 *
 * The caller must already have called ferrompi_op_drop_closure to drop the
 * Rust closure before calling this function; it is not called here. */
int ferrompi_op_free_slot_only(int32_t handle) {
    if (handle < 0 || handle >= MAX_OPS) return MPI_ERR_ARG;
    free_op_slot(handle);
    return MPI_SUCCESS;
}

/* MPI_Allreduce using a user-defined op identified by op_handle (slot). */
int ferrompi_allreduce_user_op(
    const void* sendbuf,
    void* recvbuf,
    int64_t count,
    int32_t datatype_tag,
    int32_t op_handle,
    int32_t comm_handle
) {
    MPI_Comm comm = get_comm(comm_handle);
    MPI_Datatype dt = get_datatype(datatype_tag);
    if (dt == MPI_DATATYPE_NULL) return MPI_ERR_TYPE;
    if (op_handle < 0 || op_handle >= MAX_OPS) return MPI_ERR_ARG;
    MPI_Op op = atomic_load_explicit(&op_table[op_handle], memory_order_acquire);
    if (op == MPI_OP_NULL) return MPI_ERR_OP;
    /* MPI-4.0 section 6.9.5: when a large count is narrowed to the classic
     * MPI_User_function's int len, MPI calls that function multiple times
     * with a sequence of len values summing to count, so an op created with
     * the classic MPI_Op_create remains valid for a large-count reduction. */
    if (count > INT_MAX) {
#if MPI_VERSION >= 4
        return MPI_Allreduce_c(sendbuf, recvbuf, (MPI_Count)count, dt, op, comm);
#else
        return MPI_ERR_COUNT;
#endif
    }
    return MPI_Allreduce(sendbuf, recvbuf, (int)count, dt, op, comm);
}

