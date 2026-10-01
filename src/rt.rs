//! Process-wide FFI lifecycle guard:
//! `Uninit -> Initializing -> Active(level) -> Finalized`.
//!
//! [`enter`] is called by every guarded extern wrapper in [`crate::ffi`] and
//! rejects the call once state is `Finalized`, or once state is
//! `Active(Single)`/`Active(Funneled)` and the caller is not the thread that
//! called `activate`, without touching MPI. [`drop_guard`] is the equivalent
//! check for a `Drop` impl, which cannot return `Err`: it skips the MPI call
//! silently after finalize, and aborts the process on a wrong-thread drop
//! below `Serialized`. `Initializing` is a transient state held only between
//! [`begin_init`] and [`activate`]/[`abandon_init`]; both `enter` and
//! `drop_guard` pass it through like `Uninit` since no `Mpi` handle exists
//! yet to call either.

use std::cell::Cell;
use std::io::Write;
use std::os::raw::c_int;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ThreadLevel;

const UNINIT: u8 = 0;
const ACTIVE_SINGLE: u8 = 1;
const ACTIVE_FUNNELED: u8 = 2;
const FINALIZED: u8 = 5;
const INITIALIZING: u8 = 6;

static STATE: AtomicU8 = AtomicU8::new(UNINIT);

thread_local! {
    /// Set by `activate` on the thread that calls it. `Mpi` is `!Send`, so
    /// `STATE` and this flag are always written by the same single thread.
    static ON_INIT_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Move `Uninit` to `Initializing`, serialising concurrent
/// `Mpi::init`/`init_thread` calls. Returns `Err(Error::Finalized)` once
/// state is `Finalized`, and `Err(Error::AlreadyInitialized)` for any other
/// state (an `Mpi` is alive, or another thread is already initialising).
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
        Err(FINALIZED) => Err(Error::Finalized),
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

/// Move an `Active` state to `Finalized`, returning `true`. Returns `false`
/// and changes nothing in any other state, so a stub `Mpi` dropped without
/// init (`Uninit`) — or a second drop after finalize — is a no-op.
pub(crate) fn finalize() -> bool {
    // Relaxed: `Mpi` is `!Send`, so this load only ever races with another
    // thread's `enter()` load, never with a second write on this thread.
    let prev = STATE.load(Ordering::Relaxed);
    if (1..=4).contains(&prev) {
        // Relaxed: no ordering on this store closes the window where a
        // concurrent `enter()` on another thread reads `Active` just before
        // this store and then calls MPI just after `ferrompi_finalize` runs
        // below. No ordering on `STATE` can close that gap; until this crate
        // tracks in-flight calls, it is instead a caller contract: `Mpi`
        // must not be dropped while another thread is inside an MPI call
        // through this crate.
        STATE.store(FINALIZED, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Returns `true` once state is `Finalized`. `Mpi::is_finalized()` uses this
/// so it still reports `true` after a skipped `MPI_Finalize` (a live window
/// kept `Mpi::drop` from calling it): `finalize()`
/// already moved `STATE` to `Finalized` before that skip decision runs.
pub(crate) fn is_finalized() -> bool {
    // Relaxed: see `enter`'s comment — visibility of the `Finalized` state
    // set on another thread is carried by that thread's own handle hand-off,
    // not by this load's ordering.
    STATE.load(Ordering::Relaxed) == FINALIZED
}

/// Called at the top of every guarded extern wrapper. Returns
/// `FERROMPI_ERR_FINALIZED` once state is `Finalized`;
/// `FERROMPI_ERR_THREAD_LEVEL` once state is `Active(Single)` or
/// `Active(Funneled)` and the caller is not the init thread; `0` otherwise.
#[inline(always)]
pub(crate) fn enter() -> c_int {
    // Relaxed: see `activate`'s comment — visibility of an `Active`/`Finalized`
    // state set on another thread is carried by that thread's own handle
    // hand-off, not by this load's ordering.
    let state = STATE.load(Ordering::Relaxed);
    if state == FINALIZED {
        return FERROMPI_ERR_FINALIZED;
    }
    if (state == ACTIVE_SINGLE || state == ACTIVE_FUNNELED) && !ON_INIT_THREAD.with(Cell::get) {
        return FERROMPI_ERR_THREAD_LEVEL;
    }
    0
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
    if STATE.load(Ordering::Relaxed) == 1 + ThreadLevel::Serialized as u8 {
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
/// [`enter`] (a `Drop` impl cannot propagate an `Err`). Returns `false` once
/// state is `Finalized`, so the caller skips its MPI call silently. Once
/// state is `Active(Single)`/`Active(Funneled)` and the caller is not the
/// init thread, this never returns: see [`drop_abort`]. Otherwise returns
/// `true`.
pub(crate) fn drop_guard(type_name: &'static str) -> bool {
    // Relaxed: see `enter`'s comment — visibility of an `Active`/`Finalized`
    // state set on another thread is carried by that thread's own handle
    // hand-off, not by this load's ordering.
    let state = STATE.load(Ordering::Relaxed);
    if state == FINALIZED {
        return false;
    }
    if (state == ACTIVE_SINGLE || state == ACTIVE_FUNNELED) && !ON_INIT_THREAD.with(Cell::get) {
        drop_abort(type_name);
    }
    true
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

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::{release_in_call_flag, take_in_call_flag};

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
}
