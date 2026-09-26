//! Process-wide FFI lifecycle guard: `Uninit -> Active(level) -> Finalized`.
//!
//! [`enter`] is called by every guarded extern wrapper in [`crate::ffi`] and
//! rejects the call once state is `Finalized`, or once state is
//! `Active(Single)`/`Active(Funneled)` and the caller is not the thread that
//! called `activate`, without touching MPI. [`drop_guard`] is the equivalent
//! check for a `Drop` impl, which cannot return `Err`: it skips the MPI call
//! silently after finalize, and aborts the process on a wrong-thread drop
//! below `Serialized`.

use std::cell::Cell;
use std::io::Write;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::{FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ThreadLevel;

const UNINIT: u8 = 0;
const ACTIVE_SINGLE: u8 = 1;
const ACTIVE_FUNNELED: u8 = 2;
const FINALIZED: u8 = 5;

static STATE: AtomicU8 = AtomicU8::new(UNINIT);

thread_local! {
    /// Set by `activate` on the thread that calls it. `Mpi` is `!Send`, so
    /// `STATE` and this flag are always written by the same single thread.
    static ON_INIT_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Move `Uninit` to `Active(level)` and record the calling thread as the
/// init thread. Called once, from `Mpi::init_thread` after
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
