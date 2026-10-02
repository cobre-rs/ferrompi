//! Process-wide FFI lifecycle guard:
//! `Uninit -> Initializing -> Active(level) -> Finalizing -> Finalized`.
//!
//! [`enter`] is called by every guarded extern wrapper in [`crate::ffi`] and
//! rejects the call once state is `Finalizing` or `Finalized`, or once state
//! is `Active(Single)`/`Active(Funneled)` and the caller is not the thread
//! that called `activate`, without touching MPI. It returns an [`InFlight`]
//! token that the wrapper holds for the whole call. At `Active(Serialized)`
//! and `Active(Multiple)` the token counts the call in a per-thread shard of
//! `IN_FLIGHT`; at `Single`/`Funneled` it is empty, and the check stays one
//! relaxed load plus the init-thread test. [`drop_guard`] is the equivalent
//! check for a `Drop` impl, which cannot return `Err`: it skips the MPI call
//! silently once finalizing, and aborts the process on a wrong-thread drop
//! below `Serialized`. `Initializing` is a transient state held only between
//! [`begin_init`] and [`activate`]/[`abandon_init`]; both `enter` and
//! `drop_guard` pass it through like `Uninit` since no `Mpi` handle exists
//! yet to call either.
//!
//! Finalizing is two steps. [`begin_finalize`] stores `Finalizing`, which
//! refuses new calls, and at `Serialized`/`Multiple` waits ([`drain_shards`])
//! until every counted call has returned; [`end_finalize`] then stores
//! `Finalized`, just before `MPI_Finalize` runs.

use std::cell::Cell;
use std::io::Write;
use std::os::raw::c_int;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ThreadLevel;

const UNINIT: u8 = 0;
const ACTIVE_SINGLE: u8 = 1;
const ACTIVE_FUNNELED: u8 = 2;
const ACTIVE_SERIALIZED: u8 = 3;
const ACTIVE_MULTIPLE: u8 = 4;
const FINALIZED: u8 = 5;
const INITIALIZING: u8 = 6;
const FINALIZING: u8 = 7;

static STATE: AtomicU8 = AtomicU8::new(UNINIT);

thread_local! {
    /// Set by `activate` on the thread that calls it. `Mpi` is `!Send`, so
    /// `STATE` and this flag are always written by the same single thread.
    static ON_INIT_THREAD: Cell<bool> = const { Cell::new(false) };

    /// This thread's index into `IN_FLIGHT`; `usize::MAX` until its first
    /// counted use.
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

const SHARDS: usize = 64;

/// One in-flight counter. 128-byte alignment keeps two shards off the same
/// pair of cache lines, so threads counting on different shards do not
/// contend.
#[repr(align(128))]
struct Shard(AtomicUsize);

static IN_FLIGHT: [Shard; SHARDS] = [const { Shard(AtomicUsize::new(0)) }; SHARDS];

static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

/// Held by a guarded call or `Drop` path for as long as it may call MPI. At
/// `Serialized`/`Multiple` it owns one count in a shard of `IN_FLIGHT`,
/// which [`begin_finalize`] waits for; elsewhere it is empty.
#[must_use]
pub(crate) struct InFlight(Option<&'static Shard>);

impl Drop for InFlight {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(shard) = self.0 {
            // Release: pairs with the SeqCst (hence acquiring) loads in
            // `drain_shards`, so every MPI call made under this token
            // happens-before the `MPI_Finalize` that follows the drain.
            shard.0.fetch_sub(1, Ordering::Release);
        }
    }
}

/// This thread's shard, taken round-robin on first use and kept.
#[inline]
fn shard() -> &'static Shard {
    let index = SHARD.with(Cell::get);
    if index < SHARDS {
        &IN_FLIGHT[index]
    } else {
        assign_shard()
    }
}

#[cold]
#[inline(never)]
fn assign_shard() -> &'static Shard {
    // Relaxed: the counter only spreads threads over shards; it orders
    // nothing.
    let index = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
    SHARD.with(|cell| cell.set(index));
    &IN_FLIGHT[index]
}

/// Move `Uninit` to `Initializing`, serialising concurrent
/// `Mpi::init`/`init_thread` calls. Returns `Err(Error::Finalized)` once
/// state is `Finalizing` or `Finalized`, and `Err(Error::AlreadyInitialized)`
/// for any other state (an `Mpi` is alive, or another thread is already
/// initialising).
/// Called once, at the top of `Mpi::init_thread`; every path out of that
/// function afterward reaches either [`activate`] or [`abandon_init`].
pub(crate) fn begin_init() -> Result<()> {
    // Relaxed: a successful compare-exchange only ever transitions out of the
    // initial zero state, which carries no prior writer to synchronize with;
    // a losing thread returns without touching MPI, so it needs no ordering
    // beyond observing the current value through this atomic's modification
    // order, which the compare-exchange itself guarantees regardless of
    // ordering annotation.
    match STATE.compare_exchange(UNINIT, INITIALIZING, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Ok(()),
        Err(FINALIZING | FINALIZED) => Err(Error::Finalized),
        Err(_) => Err(Error::AlreadyInitialized),
    }
}

/// Move `Initializing` back to `Uninit`. Called when `Mpi::init_thread` does
/// not complete after [`begin_init`] succeeded: MPI turned out already
/// finalized, or `MPI_Init_thread` itself failed. Only the thread that won
/// `begin_init`'s compare-exchange calls this, so a plain store is safe.
pub(crate) fn abandon_init() {
    // Relaxed: only the calling thread can observe `Initializing` (it is the
    // sole holder, per this function's contract), so there is no concurrent
    // writer to synchronize with.
    STATE.store(UNINIT, Ordering::Relaxed);
}

/// Move `Initializing` to `Active(level)` and record the calling thread as
/// the init thread. Called once, from `Mpi::init_thread` after
/// `MPI_Init_thread` succeeds.
pub(crate) fn activate(level: ThreadLevel) {
    // Relaxed: `Mpi` is `!Send`/`!Sync`, so every later `enter()` load that
    // must observe this store — whether on this thread or one that receives
    // a `Communicator`/`Request` afterward — is already ordered by the
    // synchronizing operation (thread spawn, channel, mutex, `Arc`) that
    // thread used to obtain that handle; this store needs no ordering of
    // its own to be visible through that chain.
    STATE.store(1 + level as u8, Ordering::Relaxed);
    ON_INIT_THREAD.with(|c| c.set(true));
}

/// Waits until every shard reads zero, that is, until every counted call and
/// `Drop` path has returned. No timeout: a thread blocked forever inside an
/// MPI call blocks this wait, as it would block `MPI_Finalize`.
#[cold]
pub(crate) fn drain_shards() {
    for shard in &IN_FLIGHT {
        // SeqCst: the shard load of the pair with `enter_counted`'s increment,
        // and an acquire of the Release decrement it observes.
        while shard.0.load(Ordering::SeqCst) != 0 {
            for _ in 0..64 {
                std::hint::spin_loop();
            }
            std::thread::yield_now();
        }
    }
}

/// Move an `Active` state to `Finalizing`, which refuses new calls. At
/// `Serialized`/`Multiple`, then waits for every counted call to return.
/// Returns `true`. Returns `false` and changes nothing in any other state, so
/// a stub `Mpi` dropped without init (`Uninit`) — or a second drop after
/// finalize — is a no-op.
pub(crate) fn begin_finalize() -> bool {
    // Relaxed: `Mpi` is `!Send`, so this load only ever races with another
    // thread's `enter()` load, never with a second write on this thread.
    let prev = STATE.load(Ordering::Relaxed);
    if !(ACTIVE_SINGLE..=ACTIVE_MULTIPLE).contains(&prev) {
        return false;
    }
    // SeqCst: the store of the pair with `enter_counted`. A call whose
    // increment precedes this store in the SeqCst order is seen by the drain
    // below; a call whose increment follows it reads `Finalizing` and is
    // refused. So no counted call can begin once `MPI_Finalize` does.
    STATE.store(FINALIZING, Ordering::SeqCst);
    if prev >= ACTIVE_SERIALIZED {
        drain_shards();
    }
    true
}

/// Move `Finalizing` to `Finalized`. Called by `Mpi::drop` after
/// [`begin_finalize`] returned `true`.
pub(crate) fn end_finalize() {
    // SeqCst: totally ordered after the `Finalizing` store and the drain.
    STATE.store(FINALIZED, Ordering::SeqCst);
}

/// Returns `true` once state is `Finalizing` or `Finalized`.
/// `Mpi::is_finalized()` uses this so it reports `true` from the start of
/// `Mpi::drop`, and still does after a skipped `MPI_Finalize` (a live window
/// kept `Mpi::drop` from calling it): `begin_finalize()` and `end_finalize()`
/// already moved `STATE` before that skip decision runs.
pub(crate) fn is_finalized() -> bool {
    // Relaxed: see `enter`'s comment — visibility of the `Finalizing` or
    // `Finalized` state set on another thread is carried by that thread's own
    // handle hand-off, not by this load's ordering.
    matches!(STATE.load(Ordering::Relaxed), FINALIZING | FINALIZED)
}

/// Called at the top of every guarded extern wrapper. Returns
/// `Err(FERROMPI_ERR_FINALIZED)` once state is `Finalizing` or `Finalized`;
/// `Err(FERROMPI_ERR_THREAD_LEVEL)` once state is `Active(Single)` or
/// `Active(Funneled)` and the caller is not the init thread; otherwise the
/// [`InFlight`] token the caller holds until its MPI call returns.
#[inline(always)]
pub(crate) fn enter() -> std::result::Result<InFlight, c_int> {
    // Relaxed: see `activate`'s comment — visibility of an `Active`,
    // `Finalizing` or `Finalized` state set on another thread is carried by
    // that thread's own handle hand-off, not by this load's ordering. The
    // counted path re-checks with SeqCst.
    let state = STATE.load(Ordering::Relaxed);
    if state == ACTIVE_SINGLE || state == ACTIVE_FUNNELED {
        if ON_INIT_THREAD.with(Cell::get) {
            return Ok(InFlight(None));
        }
        return Err(FERROMPI_ERR_THREAD_LEVEL);
    }
    enter_other(state)
}

/// [`enter`] for every state but `Active(Single)`/`Active(Funneled)`, kept out
/// of line so the `Funneled` path stays the single range test and init-thread
/// check it was before calls were counted.
#[inline(never)]
fn enter_other(state: u8) -> std::result::Result<InFlight, c_int> {
    if state == ACTIVE_SERIALIZED || state == ACTIVE_MULTIPLE {
        return enter_counted();
    }
    if state == FINALIZING || state == FINALIZED {
        return Err(FERROMPI_ERR_FINALIZED);
    }
    Ok(InFlight(None))
}

/// The state check of [`enter`] without a token: counts nothing, and refuses
/// the same states. For the two `wait` pre-checks, whose guarded `ffi` call
/// takes the token.
#[inline(always)]
pub(crate) fn check() -> c_int {
    // Relaxed: see `enter`'s comment.
    let state = STATE.load(Ordering::Relaxed);
    if state == FINALIZING || state == FINALIZED {
        return FERROMPI_ERR_FINALIZED;
    }
    if (state == ACTIVE_SINGLE || state == ACTIVE_FUNNELED) && !ON_INIT_THREAD.with(Cell::get) {
        return FERROMPI_ERR_THREAD_LEVEL;
    }
    0
}

/// The counted path, for `Active(Serialized)` and `Active(Multiple)`: take
/// this thread's shard, increment it, then re-check `STATE`.
#[inline]
fn enter_counted() -> std::result::Result<InFlight, c_int> {
    let shard = shard();
    // SeqCst: the increment of the pair with `begin_finalize`'s `Finalizing`
    // store. It is ordered before the re-check below, so either the drain sees
    // this count or the re-check sees `Finalizing`.
    shard.0.fetch_add(1, Ordering::SeqCst);
    let token = InFlight(Some(shard));
    // SeqCst: reads `Finalizing` or `Finalized` if that store precedes the
    // increment in the SeqCst order.
    let state = STATE.load(Ordering::SeqCst);
    if state == FINALIZING || state == FINALIZED {
        return Err(FERROMPI_ERR_FINALIZED);
    }
    Ok(token)
}

/// Set for the duration of one guarded FFI call while `STATE` is
/// `Active(Serialized)`, to detect two such calls overlapping. Debug-only:
/// `Serialized` requires the caller to serialize its own MPI calls, so this
/// is a diagnostic, not a correctness mechanism release builds must pay for.
#[cfg(debug_assertions)]
static SERIALIZED_IN_CALL: AtomicBool = AtomicBool::new(false);

/// Attempts to take [`SERIALIZED_IN_CALL`]. Returns `true` on success.
#[cfg(debug_assertions)]
fn take_in_call_flag() -> bool {
    // Acquire/Relaxed: a successful exchange must synchronize-with the
    // matching `release_in_call_flag`'s Release store, so the guarded FFI
    // call this thread is about to make cannot be reordered ahead of the
    // previous holder's; a failed exchange holds nothing and needs no
    // ordering.
    SERIALIZED_IN_CALL
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// Releases [`SERIALIZED_IN_CALL`].
#[cfg(debug_assertions)]
fn release_in_call_flag() {
    // Release: pairs with `take_in_call_flag`'s Acquire, so this thread's
    // guarded FFI call happens-before the next thread's that takes the flag.
    SERIALIZED_IN_CALL.store(false, Ordering::Release);
}

/// Called at the top of a guarded extern wrapper, after [`enter`] passes. At
/// `Active(Serialized)`, takes [`SERIALIZED_IN_CALL`] and returns `Ok(true)`,
/// or `Err(FERROMPI_ERR_THREAD_LEVEL)` if another call already holds it. In
/// every other state, returns `Ok(false)` without touching the flag.
#[cfg(debug_assertions)]
pub(crate) fn begin_call() -> std::result::Result<bool, c_int> {
    // Relaxed: see `enter`'s comment — visibility of the `Active` state set
    // on another thread is carried by that thread's own handle hand-off,
    // not by this load's ordering.
    if STATE.load(Ordering::Relaxed) == ACTIVE_SERIALIZED {
        if take_in_call_flag() {
            Ok(true)
        } else {
            Err(FERROMPI_ERR_THREAD_LEVEL)
        }
    } else {
        Ok(false)
    }
}

/// Releases the flag taken by [`begin_call`], when `held` is `true`.
#[cfg(debug_assertions)]
pub(crate) fn end_call(held: bool) {
    if held {
        release_in_call_flag();
    }
}

/// Called at the top of a `Drop` impl's MPI-calling path, in place of
/// [`enter`] (a `Drop` impl cannot propagate an `Err`). Returns `None` once
/// state is `Finalizing` or `Finalized`, so the caller skips its MPI call
/// silently. Once state is `Active(Single)`/`Active(Funneled)` and the caller
/// is not the init thread, this never returns: see [`drop_abort`]. Otherwise
/// returns the [`InFlight`] token, which the caller binds for the whole
/// `drop` body so it covers every MPI call there.
pub(crate) fn drop_guard(type_name: &'static str) -> Option<InFlight> {
    // Relaxed: see `enter`'s comment.
    let state = STATE.load(Ordering::Relaxed);
    if state == FINALIZING || state == FINALIZED {
        return None;
    }
    if state == ACTIVE_SINGLE || state == ACTIVE_FUNNELED {
        if !ON_INIT_THREAD.with(Cell::get) {
            drop_abort(type_name);
        }
    } else if state == ACTIVE_SERIALIZED || state == ACTIVE_MULTIPLE {
        return enter_counted().ok();
    }
    Some(InFlight(None))
}

/// Prints `ferrompi: <type_name> dropped on thread <label>` to stderr and
/// aborts the process. Split out of [`drop_guard`] and marked `#[cold]` so
/// the wrong-thread path does not bloat the common-case branch.
#[cold]
fn drop_abort(type_name: &'static str) -> ! {
    let current = std::thread::current();
    let label = match current.name() {
        Some(name) => name.to_string(),
        None => format!("{:?}", current.id()),
    };
    // eprintln! panics on a closed stderr, and a panic inside `Drop` during
    // unwinding aborts without printing this message, so the write result
    // is ignored instead.
    let _ = writeln!(
        std::io::stderr(),
        "ferrompi: {type_name} dropped on thread {label}"
    );
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{drain_shards, enter, shard, InFlight, Shard, IN_FLIGHT, SHARDS};
    #[cfg(debug_assertions)]
    use super::{release_in_call_flag, take_in_call_flag};

    #[cfg(debug_assertions)]
    #[test]
    fn in_call_flag_rejects_a_second_thread() {
        assert!(take_in_call_flag(), "first take must succeed");

        let overlapping_took = std::thread::scope(|s| {
            s.spawn(take_in_call_flag)
                .join()
                .expect("overlap thread panicked")
        });
        assert!(!overlapping_took, "overlapping take must be rejected");

        release_in_call_flag();

        let next_took = std::thread::scope(|s| {
            s.spawn(|| {
                let took = take_in_call_flag();
                if took {
                    release_in_call_flag();
                }
                took
            })
            .join()
            .expect("next thread panicked")
        });
        assert!(next_took, "take after release must succeed");
    }

    #[test]
    fn in_flight_token_drop_decrements_its_shard() {
        static UNDER_TEST: Shard = Shard(AtomicUsize::new(0));

        UNDER_TEST.0.fetch_add(1, Ordering::SeqCst);
        let token = InFlight(Some(&UNDER_TEST));
        assert_eq!(UNDER_TEST.0.load(Ordering::SeqCst), 1);
        drop(token);
        assert_eq!(UNDER_TEST.0.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn shard_is_stable_per_thread() {
        let first = shard();
        assert!(ptr::eq(first, shard()));
        assert!(IN_FLIGHT.iter().any(|s| ptr::eq(s, first)));

        std::thread::scope(|s| {
            s.spawn(|| {
                let mine = shard();
                assert!(ptr::eq(mine, shard()));
                assert!(IN_FLIGHT.iter().any(|s| ptr::eq(s, mine)));
            })
            .join()
            .expect("shard thread panicked");
        });
        assert!(ptr::eq(first, shard()));
        assert_eq!(IN_FLIGHT.len(), SHARDS);
    }

    #[test]
    fn enter_before_init_takes_no_count() {
        let token = enter().expect("an uninitialized process passes through");
        assert!(token.0.is_none());
        assert!(IN_FLIGHT.iter().all(|s| s.0.load(Ordering::SeqCst) == 0));
    }

    #[test]
    fn drain_returns_when_shards_are_zero() {
        drain_shards();
    }
}
