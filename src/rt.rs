//! Process-wide FFI lifecycle guard: `Uninit -> Active(level) -> Finalized`.
//!
//! [`enter`] is called by every guarded extern wrapper in [`crate::ffi`] and
//! rejects the call once state is `Finalized`, without touching MPI.

use std::os::raw::c_int;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::FERROMPI_ERR_FINALIZED;
use crate::ThreadLevel;

const UNINIT: u8 = 0;
const FINALIZED: u8 = 5;

static STATE: AtomicU8 = AtomicU8::new(UNINIT);

/// Move `Uninit` to `Active(level)`. Called once, from `Mpi::init_thread`
/// after `MPI_Init_thread` succeeds.
pub(crate) fn activate(level: ThreadLevel) {
    STATE.store(1 + level as u8, Ordering::Relaxed);
}

/// Move an `Active` state to `Finalized`, returning `true`. Returns `false`
/// and changes nothing in any other state, so a stub `Mpi` dropped without
/// init (`Uninit`) — or a second drop after finalize — is a no-op.
pub(crate) fn finalize() -> bool {
    let prev = STATE.load(Ordering::Relaxed);
    if (1..=4).contains(&prev) {
        STATE.store(FINALIZED, Ordering::Relaxed);
        true
    } else {
        false
    }
}

/// Called at the top of every guarded extern wrapper. Returns
/// `FERROMPI_ERR_FINALIZED` once state is `Finalized`, otherwise `0`.
#[inline(always)]
pub(crate) fn enter() -> c_int {
    if STATE.load(Ordering::Relaxed) == FINALIZED {
        FERROMPI_ERR_FINALIZED
    } else {
        0
    }
}
