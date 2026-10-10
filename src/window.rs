//! Shared memory windows for intra-node communication.
//!
//! This module provides [`SharedWindow<T>`], a safe Rust wrapper around
//! `MPI_Win_allocate_shared` with RAII lifecycle management. Shared memory
//! windows allow processes on the same node to directly access each other's
//! memory without explicit message passing, enabling high-performance
//! intra-node communication.
//!
//! It also provides [`Win<T>`], the general RMA window created over a `Vec`
//! it then owns (`Win::create`) or MPI-allocated memory (`Win::allocate`),
//! with fence, PSCW and passive-target synchronization.
//!
//! # Synchronization
//!
//! MPI shared memory windows require explicit synchronization:
//!
//! - **Active target**: Use [`SharedWindow::fence()`] for bulk-synchronous access
//!   patterns where all processes participate in synchronization.
//! - **Passive target**: Use [`SharedWindow::lock()`] or [`SharedWindow::lock_all()`]
//!   for fine-grained, one-sided access patterns. These return RAII guards
//!   ([`LockGuard`] / [`LockAllGuard`]) that automatically unlock on drop.
//!
//! # Feature Gate
//!
//! This module is only available when the `rma` feature is enabled.
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::{Mpi, SharedWindow, LockType};
//!
//! let mpi = Mpi::init().unwrap();
//! let world = mpi.world();
//! let node = world.split_shared().unwrap();
//!
//! // Each process allocates 100 f64s in shared memory
//! let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
//!
//! // Write to local portion
//! {
//!     let local = win.local_slice_mut();
//!     for (i, x) in local.iter_mut().enumerate() {
//!         *x = (node.rank() * 100 + i as i32) as f64;
//!     }
//! }
//!
//! // Fence synchronization
//! win.fence().unwrap();
//!
//! // Read from any rank's memory
//! let remote = win.remote_slice(0).unwrap();
//! println!("Rank 0's first value: {}", remote[0]);
//! ```

use std::cell::Cell;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::{BitOr, BitOrAssign};
use std::os::raw::c_int;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::datatype::{buf, buf_mut};
use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ffi;
use crate::group::Group;
use crate::persistent::vec_buf_mut;
use crate::request::{Request, RequestKind};
use crate::rt;
use crate::scope::Registry;
use crate::Communicator;
use crate::MpiDatatype;
use crate::{AccumulateOp, FetchOp};

// ============================================================================
// WinFenceAssert — bitflags for MPI_Win_fence assert hints
// ============================================================================

/// Assertion bitmask for [`Win::fence`] (active-target synchronization).
///
/// MPI allows callers to pass assertion hints to `MPI_Win_fence` to enable
/// implementation-side optimizations. The integer values of these constants
/// are **implementation-defined** (MPICH and Open MPI differ), so they are
/// queried from the C layer at first use via a `OnceLock<[i32; 4]>` and the
/// `ferrompi_win_fence_mode_values` shim.
///
/// # Composition
///
/// Flags can be combined with `|` and `|=`:
///
/// ```
/// use ferrompi::WinFenceAssert;
///
/// let flags = WinFenceAssert::no_store() | WinFenceAssert::no_put();
/// assert!(flags.bits() != 0);
/// ```
///
/// Use `WinFenceAssert::default()` (equivalent to `WinFenceAssert::none()`)
/// when no assertion is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct WinFenceAssert(i32);

/// Private cache for the four MPI fence mode constants.
///
/// Initialized once on the first call to [`fence_mode_values()`].
/// The array layout is `[NOSTORE, NOPUT, NOPRECEDE, NOSUCCEED]`.
static FENCE_MODE_VALUES: OnceLock<[i32; 4]> = OnceLock::new();

/// Query the four MPI fence mode constants, caching the result.
fn fence_mode_values() -> [i32; 4] {
    *FENCE_MODE_VALUES.get_or_init(|| {
        let mut out = [0i32; 4];
        // SAFETY: `out` is a stack-allocated 4-element array; we pass a valid
        // pointer to its first element. The C shim writes exactly 4 i32 values.
        unsafe { ffi::ferrompi_win_fence_mode_values(out.as_mut_ptr()) };
        out
    })
}

impl WinFenceAssert {
    /// No assertion — always valid; equivalent to passing `0` to
    /// `MPI_Win_fence`.
    #[inline]
    pub fn none() -> Self {
        WinFenceAssert(0)
    }

    /// Asserts that no stores to the window will occur in the following epoch
    /// (`MPI_MODE_NOSTORE`).
    #[inline]
    pub fn no_store() -> Self {
        WinFenceAssert(fence_mode_values()[0])
    }

    /// Asserts that no put operations will occur in the following epoch
    /// (`MPI_MODE_NOPUT`).
    #[inline]
    pub fn no_put() -> Self {
        WinFenceAssert(fence_mode_values()[1])
    }

    /// Asserts that no RMA operations precede this fence (`MPI_MODE_NOPRECEDE`).
    ///
    /// This hints to the implementation that no RMA access epoch was open
    /// before this fence, so no preceding operations need to be completed.
    /// [`Win::fence`] accepts it on the opening fence and rejects it on the
    /// closing one, which must complete the closure's operations.
    #[inline]
    pub fn no_precede() -> Self {
        WinFenceAssert(fence_mode_values()[2])
    }

    /// Asserts that no RMA operations will follow this fence
    /// (`MPI_MODE_NOSUCCEED`).
    ///
    /// This hints that the fence closes the last access epoch, so the
    /// implementation need not prepare for subsequent RMA operations.
    /// [`Win::fence`] accepts it on the closing fence and rejects it on the
    /// opening one, whose epoch holds the closure's operations.
    #[inline]
    pub fn no_succeed() -> Self {
        WinFenceAssert(fence_mode_values()[3])
    }

    /// Returns the raw integer bitmask.
    #[inline]
    pub fn bits(self) -> i32 {
        self.0
    }

    /// Construct from a raw bit value.
    ///
    /// Intended for unit tests only — production code should use the named
    /// constructors so that the correct MPI constant values are used.
    #[cfg(test)]
    pub(crate) fn from_bits_for_test(v: i32) -> Self {
        WinFenceAssert(v)
    }
}

impl BitOr for WinFenceAssert {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        WinFenceAssert(self.0 | rhs.0)
    }
}

impl BitOrAssign for WinFenceAssert {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Rejects the two assertions that would let MPI keep using an origin buffer
/// after the closing fence returned: `no_succeed` on the opening fence and
/// `no_precede` on the closing one (MPI-4.1 §12.5.5). A mode value of 0 sets no
/// bit, so it is never rejected.
fn check_fence_asserts(open: WinFenceAssert, close: WinFenceAssert) -> Result<()> {
    if open.0 & WinFenceAssert::no_succeed().0 != 0 {
        return Err(Error::InvalidArgument {
            arg: "open",
            reason: "no_succeed on an opening fence leaves the epoch's RMA calls outside it",
        });
    }
    if close.0 & WinFenceAssert::no_precede().0 != 0 {
        return Err(Error::InvalidArgument {
            arg: "close",
            reason: "no_precede on a closing fence leaves the epoch's RMA calls uncompleted",
        });
    }
    Ok(())
}

// ============================================================================
// WinPscwAssert — bitflags for MPI_Win_post / MPI_Win_start assert hints
// ============================================================================

/// Assertion bitmask for [`Win::post`] and [`Win::start`] (PSCW active-target
/// synchronization).
///
/// MPI allows callers to pass assertion hints to `MPI_Win_post` and
/// `MPI_Win_start` to enable implementation-side optimizations. The integer
/// values of these constants are **implementation-defined** (MPICH and Open MPI
/// differ), so they are queried from the C layer at first use via a
/// `OnceLock<[i32; 3]>` and the `ferrompi_win_pscw_mode_values` shim.
///
/// # Composition
///
/// Flags can be combined with `|` and `|=`:
///
/// ```
/// use ferrompi::WinPscwAssert;
///
/// let flags = WinPscwAssert::no_store() | WinPscwAssert::no_put();
/// assert!(flags.bits() != 0);
/// ```
///
/// Use `WinPscwAssert::default()` (equivalent to `WinPscwAssert::none()`)
/// when no assertion is needed.
#[derive(Debug, Clone, Copy, Default)]
pub struct WinPscwAssert(i32);

/// Private cache for the three MPI PSCW mode constants.
///
/// Initialized once on the first call to [`pscw_mode_values()`].
/// The array layout is `[NOCHECK, NOSTORE, NOPUT]`.
static PSCW_MODE_VALUES: OnceLock<[i32; 3]> = OnceLock::new();

/// Query the three MPI PSCW mode constants, caching the result.
fn pscw_mode_values() -> [i32; 3] {
    *PSCW_MODE_VALUES.get_or_init(|| {
        let mut out = [0i32; 3];
        // SAFETY: `out` is a stack-allocated 3-element array; we pass a valid
        // pointer to its first element. The C shim writes exactly 3 i32 values.
        unsafe { ffi::ferrompi_win_pscw_mode_values(out.as_mut_ptr()) };
        out
    })
}

impl WinPscwAssert {
    /// No assertion — always valid; equivalent to passing `0` to
    /// `MPI_Win_post` or `MPI_Win_start`.
    #[inline]
    pub fn none() -> Self {
        WinPscwAssert(0)
    }

    /// Asserts that no RMA synchronization is needed before accessing the
    /// window (`MPI_MODE_NOCHECK`).
    ///
    /// This is valid when the caller is certain that the matching post/start
    /// pair will not overlap with another exposure/access epoch.
    #[inline]
    pub fn no_check() -> Self {
        WinPscwAssert(pscw_mode_values()[0])
    }

    /// Asserts that no stores to the window will occur in the following epoch
    /// (`MPI_MODE_NOSTORE`).
    #[inline]
    pub fn no_store() -> Self {
        WinPscwAssert(pscw_mode_values()[1])
    }

    /// Asserts that no put operations will occur in the following epoch
    /// (`MPI_MODE_NOPUT`).
    #[inline]
    pub fn no_put() -> Self {
        WinPscwAssert(pscw_mode_values()[2])
    }

    /// Returns the raw integer bitmask.
    #[inline]
    pub fn bits(self) -> i32 {
        self.0
    }

    /// Construct from a raw bit value.
    ///
    /// Intended for unit tests only — production code should use the named
    /// constructors so that the correct MPI constant values are used.
    #[cfg(test)]
    pub(crate) fn from_bits_for_test(v: i32) -> Self {
        WinPscwAssert(v)
    }
}

impl BitOr for WinPscwAssert {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        WinPscwAssert(self.0 | rhs.0)
    }
}

impl BitOrAssign for WinPscwAssert {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Lock type for window synchronization.
///
/// Controls the level of concurrency allowed when locking an MPI window
/// for passive target access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockType {
    /// Exclusive access (no other locks allowed).
    ///
    /// Only one process may hold an exclusive lock on a given rank's
    /// window at a time. This is suitable for read-write access.
    Exclusive,
    /// Shared access (multiple shared locks allowed).
    ///
    /// Multiple processes may hold shared locks on a given rank's window
    /// simultaneously. This is suitable for read-only access patterns.
    Shared,
}

/// A shared memory window allocated across processes on the same node.
///
/// Created via [`SharedWindow::allocate()`] on a shared-memory communicator
/// (typically obtained from [`Communicator::split_shared()`]).
///
/// Each process in the communicator contributes a local segment of `local_count`
/// elements of type `T` to the shared memory region. All processes can then
/// access any rank's segment via [`remote_slice()`](Self::remote_slice), provided
/// proper synchronization is used.
///
/// # RAII Lifecycle
///
/// The underlying MPI window is freed automatically when the `SharedWindow`
/// is dropped. The shared memory region becomes invalid after the window is freed.
/// If a `SharedWindow` outlives the [`Mpi`](crate::Mpi) handle, `MPI_Finalize`
/// is skipped instead: the shared memory stays valid until the process exits
/// and is never freed.
///
/// # Thread Safety
///
/// `SharedWindow` is intentionally **not** `Send` or `Sync`. MPI windows have
/// specific thread-safety rules that depend on the MPI thread level, and incorrect
/// cross-thread usage can cause data corruption or MPI errors.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, SharedWindow, LockType};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let node = world.split_shared().unwrap();
///
/// // Each process allocates 100 f64s in shared memory
/// let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
///
/// // Write to local portion
/// {
///     let local = win.local_slice_mut();
///     for (i, x) in local.iter_mut().enumerate() {
///         *x = (node.rank() * 100 + i as i32) as f64;
///     }
/// }
///
/// // Fence synchronization
/// win.fence().unwrap();
///
/// // Read from any rank's memory
/// let remote = win.remote_slice(0).unwrap();
/// println!("Rank 0's first value: {}", remote[0]);
/// ```
pub struct SharedWindow<T: MpiDatatype> {
    /// The MPI window handle (index into the C-layer window table).
    win_handle: i32,
    /// Non-null pointer to this process's local shared memory segment.
    local_ptr: NonNull<T>,
    /// Number of `T` elements in the local segment.
    local_len: usize,
    /// Number of processes in the communicator that created this window.
    comm_size: i32,
}

fn win_size_and_disp_unit<T>(count: usize) -> Result<(i64, i32)> {
    let overflow = || Error::InvalidArgument {
        arg: "count",
        reason: "window byte size overflows",
    };
    let byte_size = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(overflow)?;
    let size = i64::try_from(byte_size).map_err(|_| overflow())?;
    let disp_unit = std::mem::size_of::<T>() as i32;
    Ok((size, disp_unit))
}

// Sentinel a creating shim writes to its handle out-parameter when MPI
// created the window but zeroing or registering it failed. This MUST stay
// in sync with `FERROMPI_WIN_LEAKED` in `csrc/ferrompi.h`.
const FERROMPI_WIN_LEAKED: i32 = -2;

// Handle of a `Win` whose table slot `Win::into_buffer` released. The shim
// frees the slot whatever `MPI_Win_free` returned, so `Drop` must not free again.
const WIN_FREED: i32 = -1;

/// Count of live windows of any kind (`Win::create`, `Win::allocate`,
/// `SharedWindow::allocate`). `Mpi::drop` reads this through
/// [`live_windows`] to decide whether `MPI_Finalize` is safe to call: some
/// MPI implementations free MPI-allocated window memory inside
/// `MPI_Finalize` itself, and some tear down internal state that still
/// tracks a caller-supplied window's buffer, aborting the process. Either
/// way, `MPI_Finalize` is unsafe while any window is still alive.
static LIVE_WINDOWS: AtomicUsize = AtomicUsize::new(0);

/// Number of windows of any kind currently alive.
pub(crate) fn live_windows() -> usize {
    // Acquire: pairs with the Release decrement in `Drop for Win`/`Drop for
    // SharedWindow`, in `Win::into_buffer` and in `mark_window_gone`, so a zero
    // read here cannot precede a window's in-flight teardown.
    LIVE_WINDOWS.load(Ordering::Acquire)
}

/// Counts a window being constructed (`SharedWindow::allocate`, `Win::create`
/// or `Win::allocate`) in [`LIVE_WINDOWS`]. Only [`counted_window_call`]
/// calls it, inside the in-flight span that call opens.
fn mark_window_alive() {
    // Relaxed: the increment, and its undo in `mark_window_gone`, happen inside
    // the constructor's own in-flight span. `Mpi::drop`'s drain waits for that
    // span and acquires its release before `live_windows` reads the count, so
    // the increment needs no ordering of its own. A constructor refused before
    // the span opens touches nothing.
    LIVE_WINDOWS.fetch_add(1, Ordering::Relaxed);
}

/// Undoes [`mark_window_alive`] for a constructor whose call created no window.
fn mark_window_gone() {
    // Release: pairs with `live_windows`'s Acquire load, as in `Drop for Win`.
    LIVE_WINDOWS.fetch_sub(1, Ordering::Release);
}

/// Runs a window constructor's creating shim in one in-flight span that also
/// covers its change to [`LIVE_WINDOWS`].
///
/// The token comes first, so a call refused while `Mpi` drops returns before it
/// touches the count. Then the window is counted, `call` runs, and the count is
/// undone if the shim created no window, all before the token is released.
/// `Mpi::drop` reads the count only after its drain has waited for every token,
/// so the count it reads never includes a window that does not exist, and never
/// misses one that does.
///
/// `call` receives `win_handle` for the shim's out-parameter. A shim that
/// created a window it will never free leaves [`FERROMPI_WIN_LEAKED`] there,
/// and that window keeps its count.
fn counted_window_call(
    op: &'static str,
    win_handle: &mut i32,
    call: impl FnOnce(*mut i32) -> c_int,
) -> Result<()> {
    let _call = rt::enter().map_err(|code| Error::from_code_with_op(code, op))?;
    #[cfg(debug_assertions)]
    let held = rt::begin_call().map_err(|code| Error::from_code_with_op(code, op))?;
    mark_window_alive();
    let ret = call(win_handle);
    #[cfg(debug_assertions)]
    rt::end_call(held);
    if ret != 0 && *win_handle != FERROMPI_WIN_LEAKED {
        mark_window_gone();
    }
    Error::check_with_op(ret, op)
}

/// Bit position of the displacement-unit field in a window word (see
/// [`window_word`]). The low 56 bits hold the exposed byte length.
const WINDOW_WORD_DISP_SHIFT: u32 = 56;

/// Mask for the exposed-byte-length field of a window word: the largest
/// length that can be packed into the low 56 bits.
const WINDOW_WORD_LEN_MASK: u64 = (1u64 << WINDOW_WORD_DISP_SHIFT) - 1;

/// Sentinel word for a rank whose exposed length cannot be packed. No real
/// window has a displacement unit of 255 (`MpiDatatype` is sealed to types
/// of at most 8 bytes), so this cannot collide with a real word.
const WINDOW_WORD_REJECT: u64 = u64::MAX;

/// Packs `size_of::<T>()` and `count * size_of::<T>()` into one word:
/// displacement unit in bits 56..63, exposed byte length in bits 0..55.
/// Returns [`WINDOW_WORD_REJECT`] if the byte length does not fit in 56 bits.
fn window_word<T>(count: usize) -> u64 {
    let disp_unit = std::mem::size_of::<T>();
    match count.checked_mul(disp_unit) {
        Some(len) if len as u64 <= WINDOW_WORD_LEN_MASK => {
            ((disp_unit as u64) << WINDOW_WORD_DISP_SHIFT) | len as u64
        }
        _ => WINDOW_WORD_REJECT,
    }
}

/// Exchanges one window word per rank over `comm` via `Communicator::allgather`.
///
/// When the allgather succeeds on every rank, every rank holds the same
/// `words`: if any word is [`WINDOW_WORD_REJECT`], every rank returns
/// `Err(Error::InvalidArgument)` naming `count` before any rank has created a
/// window. An allgather error can reach some ranks only; the other ranks can
/// then block in the window-creation call.
fn exchange_window_words(comm: &Communicator, word: u64) -> Result<Box<[u64]>> {
    let mut words = vec![0u64; comm.size() as usize];
    comm.allgather(&[word], &mut words)?;
    if words.contains(&WINDOW_WORD_REJECT) {
        return Err(Error::InvalidArgument {
            arg: "count",
            reason: "a rank's window length cannot be exchanged",
        });
    }
    Ok(words.into_boxed_slice())
}

/// Rejects a `rank` outside `0..group_size`, reported under the argument name
/// `arg`.
fn check_rank(group_size: usize, arg: &'static str, rank: i32) -> Result<()> {
    if rank < 0 || rank as usize >= group_size {
        return Err(Error::InvalidArgument {
            arg,
            reason: "rank outside the window's communicator",
        });
    }
    Ok(())
}

/// The ranks an epoch may issue transfers to: every rank of the window's
/// communicator, the one rank a `lock` epoch locks, or the members of a `start`
/// epoch's group, indexed by window rank.
enum Access {
    All,
    Rank(i32),
    Ranks(Box<[bool]>),
}

impl Access {
    fn contains(&self, rank: i32) -> bool {
        match self {
            Access::All => true,
            Access::Rank(locked) => *locked == rank,
            Access::Ranks(members) => members.get(rank as usize) == Some(&true),
        }
    }
}

/// Validates an RMA target against `words` (see [`window_word`]) and the
/// epoch's `access` set: rejects a `target_rank` outside `0..words.len()` or
/// outside `access`, a `buffer_lens` entry unequal to `count` (reported under
/// that entry's name), a negative `target_disp`, and an access
/// `target_disp * disp_unit(target_rank) + count * elem_size` that overflows
/// `u64` or exceeds the target's exposed byte length.
fn check_rma_target(
    words: &[u64],
    access: &Access,
    elem_size: usize,
    target_rank: i32,
    target_disp: i64,
    count: usize,
    buffer_lens: &[(&'static str, usize)],
) -> Result<()> {
    check_rank(words.len(), "target_rank", target_rank)?;
    if !access.contains(target_rank) {
        return Err(Error::InvalidArgument {
            arg: "target_rank",
            reason: "rank outside the epoch's access set",
        });
    }
    if let Some(&(arg, actual)) = buffer_lens.iter().find(|&&(_, len)| len != count) {
        return Err(Error::BufferSize {
            arg,
            required: count,
            actual,
        });
    }
    if target_disp < 0 {
        return Err(Error::InvalidArgument {
            arg: "target_disp",
            reason: "negative displacement",
        });
    }
    let past_window = || Error::InvalidArgument {
        arg: "target_disp",
        reason: "access extends past the target's window",
    };
    let word = words[target_rank as usize];
    let disp_unit = word >> WINDOW_WORD_DISP_SHIFT;
    let len = word & WINDOW_WORD_LEN_MASK;
    let disp_bytes = (target_disp as u64)
        .checked_mul(disp_unit)
        .ok_or_else(past_window)?;
    let count_bytes = (count as u64)
        .checked_mul(elem_size as u64)
        .ok_or_else(past_window)?;
    let total = disp_bytes
        .checked_add(count_bytes)
        .ok_or_else(past_window)?;
    if total > len {
        return Err(past_window());
    }
    Ok(())
}

impl<T: MpiDatatype> SharedWindow<T> {
    /// Allocate a shared memory window.
    ///
    /// Each process in `comm` allocates `local_count` elements of type `T`
    /// in a shared memory segment accessible by all processes in the communicator.
    /// The communicator should be a shared-memory communicator (e.g., from
    /// [`Communicator::split_shared()`]).
    ///
    /// The local segment reads as zero when the call returns.
    ///
    /// # Arguments
    ///
    /// * `comm` - A shared-memory communicator (all processes must be on the same node)
    /// * `local_count` - Number of elements of type `T` to allocate locally
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The MPI window allocation fails (e.g., insufficient shared memory);
    ///   this can happen on some ranks only, and the other ranks can then
    ///   block in the barrier after the new segment is zeroed
    /// - The window table is full (`Error::ResourceExhausted`; the window is
    ///   then leaked: never freed, and counted as alive so `Mpi` skips `MPI_Finalize`)
    /// - An error occurs while zeroing the new segment (the window is
    ///   then leaked: never freed, and counted as alive so `Mpi` skips `MPI_Finalize`)
    /// - The MPI implementation returns a null base pointer for a non-zero count
    ///   (the window is then leaked: never freed, and counted as alive so `Mpi` skips `MPI_Finalize`)
    ///
    /// In the last three cases, a peer whose own call succeeded blocks in its
    /// collective `Drop` (`MPI_Win_free`) waiting for this rank.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, SharedWindow};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let node = mpi.world().split_shared().unwrap();
    /// let win = SharedWindow::<f64>::allocate(&node, 1024).unwrap();
    /// ```
    pub fn allocate(comm: &Communicator, local_count: usize) -> Result<Self> {
        let (size, disp_unit) = win_size_and_disp_unit::<T>(local_count)?;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut win_handle: i32 = -1;

        counted_window_call("win_allocate_shared", &mut win_handle, |handle| {
            // SAFETY: We pass valid pointers for out-parameters. The C layer
            // allocates shared memory and returns a window handle + base pointer.
            unsafe {
                ffi::ferrompi_win_allocate_shared(
                    size,
                    disp_unit,
                    -1, // MPI_INFO_NULL
                    comm.handle,
                    &mut baseptr,
                    handle,
                )
            }
        })?;

        let local_ptr = if local_count == 0 {
            // Zero-count: MPI may return a null base pointer for an empty
            // segment (mirrors Win::allocate's zero-count arm).
            NonNull::<T>::dangling()
        } else {
            NonNull::new(baseptr.cast::<T>())
                .ok_or_else(|| Error::Internal("Win_allocate_shared returned null".into()))?
        };

        Ok(SharedWindow {
            win_handle,
            local_ptr,
            local_len: local_count,
            comm_size: comm.size(),
        })
    }

    /// Get a slice of this process's local shared memory segment.
    ///
    /// The returned slice provides read access to the `local_count` elements
    /// that were allocated by this process in [`allocate()`](Self::allocate).
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before reading data that may have been written by other processes.
    ///
    /// # Concurrent writes
    ///
    /// The slice aliases memory that other processes in the window can write
    /// at any time, but a `&[T]` promises the compiler that its contents do
    /// not change while it is borrowed. The compiler may therefore load a
    /// value once and reuse it, so a loop that polls the slice for another
    /// process's write can spin forever; this has been observed in optimised
    /// builds. Polling through this slice is unsupported. Take a new slice
    /// after each synchronising call (fence, lock or unlock) and read data
    /// another process wrote only through that new slice.
    pub fn local_slice(&self) -> &[T] {
        // SAFETY: `local_ptr` was returned by MPI_Win_allocate_shared and is
        // guaranteed valid for `local_len` elements of type T for the lifetime
        // of the window. The pointer is non-null (checked in allocate()).
        unsafe { std::slice::from_raw_parts(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get a mutable slice of this process's local shared memory segment.
    ///
    /// The returned slice provides read-write access to the `local_count` elements
    /// that were allocated by this process in [`allocate()`](Self::allocate).
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before writing data that other processes may read.
    ///
    /// # Concurrent writes
    ///
    /// The slice aliases memory that other processes in the window can read
    /// and write at any time, but a `&mut [T]` promises the compiler
    /// exclusive access while it is borrowed. The compiler may therefore keep
    /// values in registers and skip loads or stores, so a loop that polls the
    /// slice for another process's write can spin forever.
    /// Polling through this slice is unsupported. Take a new slice after each
    /// synchronising call (fence, lock or unlock).
    pub fn local_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `local_ptr` was returned by MPI_Win_allocate_shared and is
        // guaranteed valid for `local_len` elements of type T for the lifetime
        // of the window. We have `&mut self` ensuring exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Query another rank's shared memory region.
    ///
    /// Returns a slice of the shared memory segment allocated by the specified
    /// rank. This allows direct read access to remote memory without any
    /// message passing overhead.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank whose shared memory to query (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The rank is out of range
    /// - The MPI shared query call fails
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure proper MPI synchronization (fence or lock)
    /// before reading data that was written by the remote process.
    ///
    /// # Concurrent writes
    ///
    /// The slice aliases memory that other processes in the window can write
    /// at any time, but a `&[T]` promises the compiler that its contents do
    /// not change while it is borrowed. The compiler may therefore load a
    /// value once and reuse it, so a loop that polls the slice for another
    /// process's write can spin forever; this has been observed in optimised
    /// builds. Polling through this slice is unsupported. Take a new slice
    /// after each synchronising call (fence, lock or unlock) and read data
    /// another process wrote only through that new slice.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// // Read rank 0's shared memory
    /// let remote = win.remote_slice(0).unwrap();
    /// println!("Rank 0 has {} elements, first = {}", remote.len(), remote[0]);
    /// ```
    pub fn remote_slice(&self, rank: i32) -> Result<&[T]> {
        let mut size: i64 = 0;
        let mut disp_unit: i32 = 0;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();

        // SAFETY: We pass valid pointers for out-parameters. The C layer queries
        // the MPI window for the base pointer and size of the specified rank's
        // shared memory segment.
        let ret = unsafe {
            ffi::ferrompi_win_shared_query(
                self.win_handle,
                rank,
                &mut size,
                &mut disp_unit,
                &mut baseptr,
            )
        };
        Error::check_with_op(ret, "win_shared_query")?;

        let count = size as usize / std::mem::size_of::<T>();
        if baseptr.is_null() {
            if count == 0 {
                // Null with zero count: return an empty slice using a dangling pointer
                // SAFETY: NonNull::dangling() is a valid, aligned pointer for zero-length slices
                return Ok(unsafe {
                    std::slice::from_raw_parts(NonNull::<T>::dangling().as_ptr(), 0)
                });
            }
            return Err(Error::Internal(
                "MPI_Win_shared_query returned null for non-zero size".into(),
            ));
        }
        // SAFETY: MPI_Win_shared_query returns a valid pointer to the shared
        // memory segment of the specified rank. The pointer is valid for `count`
        // elements of type T for the lifetime of the window.
        Ok(unsafe { std::slice::from_raw_parts(baseptr.cast::<T>(), count) })
    }

    /// Fence synchronization (active target).
    ///
    /// A collective operation that synchronizes all accesses to the window.
    /// All processes in the window's communicator must call this function.
    /// After `fence()` returns, all preceding local stores are visible to
    /// remote processes, and all remote stores are visible locally.
    ///
    /// This is the simplest synchronization mode, suitable for bulk-synchronous
    /// access patterns (e.g., write phase → fence → read phase → fence).
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI fence operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let mut win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// // Write phase
    /// win.local_slice_mut()[0] = 42.0;
    ///
    /// // Synchronize
    /// win.fence().unwrap();
    ///
    /// // Read phase — all writes from all ranks are now visible
    /// let remote = win.remote_slice(0).unwrap();
    /// ```
    pub fn fence(&self) -> Result<()> {
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_fence(0, self.win_handle) };
        Error::check_with_op(ret, "win_fence")
    }

    /// Lock a specific rank's window (passive target synchronization).
    ///
    /// Acquires a lock on the specified rank's window, allowing one-sided
    /// access to that rank's shared memory. Returns a [`LockGuard`] that
    /// automatically unlocks the window when dropped.
    ///
    /// # Arguments
    ///
    /// * `lock_type` - [`LockType::Exclusive`] for read-write access, or
    ///   [`LockType::Shared`] for read-only access
    /// * `rank` - The rank to lock (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI lock operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow, LockType};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// {
    ///     let guard = win.lock(LockType::Shared, 0).unwrap();
    ///     let remote = win.remote_slice(0).unwrap();
    ///     println!("Value: {}", remote[0]);
    ///     guard.flush().unwrap();
    ///     // Lock is released when `guard` is dropped
    /// }
    /// ```
    pub fn lock(&self, lock_type: LockType, rank: i32) -> Result<LockGuard<'_, T>> {
        let lt = match lock_type {
            LockType::Exclusive => ffi::FERROMPI_LOCK_EXCLUSIVE,
            LockType::Shared => ffi::FERROMPI_LOCK_SHARED,
        };
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_lock(lt, rank, 0, self.win_handle) };
        Error::check_with_op(ret, "win_lock")?;
        Ok(LockGuard { window: self, rank })
    }

    /// Lock all ranks' windows (passive target synchronization).
    ///
    /// Acquires shared locks on all ranks in the window's communicator.
    /// Returns a [`LockAllGuard`] that automatically unlocks all windows
    /// when dropped.
    ///
    /// This is useful for algorithms that need to access multiple ranks'
    /// memory in a single epoch without individual lock/unlock overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI lock_all operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, SharedWindow};
    /// # let mpi = Mpi::init().unwrap();
    /// # let node = mpi.world().split_shared().unwrap();
    /// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
    /// {
    ///     let guard = win.lock_all().unwrap();
    ///     // Access any rank's memory
    ///     for rank in 0..node.size() {
    ///         let remote = win.remote_slice(rank).unwrap();
    ///         println!("Rank {rank} first value: {}", remote[0]);
    ///     }
    ///     guard.flush_all().unwrap();
    ///     // All locks released when `guard` is dropped
    /// }
    /// ```
    pub fn lock_all(&self) -> Result<LockAllGuard<'_, T>> {
        // SAFETY: win_handle is a valid MPI window handle.
        let ret = unsafe { ffi::ferrompi_win_lock_all(0, self.win_handle) };
        Error::check_with_op(ret, "win_lock_all")?;
        Ok(LockAllGuard { window: self })
    }

    /// Get this window's raw handle.
    ///
    /// The value is ferrompi's internal table index for this window. It is
    /// not an MPI handle and cannot be passed to MPI; use it only to tell
    /// objects apart, for example in logs.
    pub fn raw_handle(&self) -> i32 {
        self.win_handle
    }

    /// Get the number of processes in the window's communicator.
    ///
    /// This equals the size of the communicator that was used to create
    /// the window, and determines the valid range of ranks for
    /// [`remote_slice()`](Self::remote_slice) and [`lock()`](Self::lock).
    pub fn comm_size(&self) -> i32 {
        self.comm_size
    }
}

impl<T: MpiDatatype> Drop for SharedWindow<T> {
    fn drop(&mut self) {
        let Some(_call) = rt::drop_guard("SharedWindow") else {
            return;
        };
        // Release: pairs with `live_windows`'s Acquire load, so `Mpi::drop`
        // cannot observe this window as gone before its teardown here has
        // actually run.
        LIVE_WINDOWS.fetch_sub(1, Ordering::Release);
        // SAFETY: win_handle is a valid MPI window handle that was allocated
        // by ferrompi_win_allocate_shared. It has not been freed yet because
        // Drop is only called once, and we don't expose a manual free method.
        unsafe { ffi::ferrompi_win_free(self.win_handle) };
    }
}

// SharedWindow is not Send/Sync by default due to NonNull<T>.
// This is the correct behavior — MPI windows have specific thread-safety
// rules and should not be shared across threads without careful coordination.

// ============================================================================
// Win<T> — general-purpose distributed RMA window
// ============================================================================

/// A general-purpose distributed RMA window.
///
/// `Win<T>` wraps either:
///
/// - `MPI_Win_create` (`Win::create`) — the caller supplies a `Vec<T>` and
///   the window owns it; [`into_buffer`](Win::into_buffer) returns it.
/// - `MPI_Win_allocate` (`Win::allocate`) — MPI allocates the buffer.
///
/// Unlike [`SharedWindow`], which uses `MPI_Win_allocate_shared` and is
/// restricted to intra-node communicators, `Win` works across nodes and is
/// the foundation for all distributed RMA operations (Put, Get, Accumulate,
/// etc.) in this library.
///
/// # Ownership
///
/// `Win::create` owns its `Vec` for as long as MPI may use it, and
/// [`into_buffer`](Win::into_buffer) returns it. Dropping or forgetting the
/// window never frees a buffer a peer may still access: the buffer is released
/// only after `MPI_Win_free` has succeeded, and leaked in every other case.
///
/// # Dropping
///
/// `MPI_Win_free` is collective and is called when the `Win` is dropped. It
/// ends in one of four ways:
///
/// - The free succeeds: the window leaves the live-window count, and the
///   buffer of a `Win::create` window is dropped after the call. For
///   `Win::allocate`, the free also releases the MPI-managed buffer.
/// - [`Mpi`](crate::Mpi) is being dropped, or was dropped and skipped
///   `MPI_Finalize` because this window was alive: no MPI call is made, the
///   window stays counted as alive (so `Mpi` skips `MPI_Finalize`), and the
///   buffer is leaked. Its memory stays valid until the process exits.
/// - The free fails: the buffer is leaked and the window stays counted as
///   alive.
/// - An exposure epoch opened by [`post`](Win::post) is still open: no MPI call
///   is made, the window stays counted as alive (so `Mpi` skips
///   `MPI_Finalize`), and the buffer is leaked, because a peer may still write
///   into the window (MPI-4.1 §12.2.5).
///
/// # Thread Safety
///
/// `Win` is intentionally **not** `Send` or `Sync`. MPI windows have
/// specific thread-safety rules that depend on the MPI thread level.
///
/// # Compile-time guarantees
///
/// [`Win::fence`], [`Win::start`], [`Win::lock`] and [`Win::lock_all`] borrow the
/// window mutably for as long as their closure runs, and give it a handle with the
/// invariant lifetimes of [`Scope`](crate::Scope): `'s` is the epoch, and every
/// buffer an operation names is borrowed for `'s`, which ends only after the closing
/// call that completes the operation. Each program below does not compile, and is
/// followed by the same program made correct. Each is the body of a function that
/// takes the window by `&mut`, or the communicator when the program must own the
/// window.
///
/// ## Buffers
///
/// A buffer declared inside the closure would be freed before the closing call, so
/// the first program is rejected. The rejection also shows that `'s` is invariant:
/// were it covariant, the compiler could shorten `'s` to the buffer's lifetime and
/// accept the program.
///
/// ```compile_fail,E0597
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |e| {
///     let origin = [1u64; 4];
///     e.put(&origin, 1, 0)
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = [1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)
/// })
/// # }
/// ```
///
/// A buffer cannot be dropped while the epoch borrows it:
///
/// ```compile_fail,E0505
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = vec![1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)?;
///     drop(origin);
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = vec![1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)?;
///     Ok(())
/// })?;
/// drop(origin);
/// Ok(())
/// # }
/// ```
///
/// Nor can it be written, because MPI may still be reading it:
///
/// ```compile_fail,E0502
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let mut origin = vec![1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)?;
///     origin[0] = 2;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let mut origin = vec![1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)?;
///     Ok(())
/// })?;
/// origin[0] = 2;
/// Ok(())
/// # }
/// ```
///
/// ## Results MPI writes
///
/// A `get` destination, a `get_accumulate` result and a `fetch_and_op` or
/// `compare_and_swap` result are written when the closing call returns, and stay
/// borrowed until then. Reading one inside the epoch, which would see a value
/// MPI has not written yet, is rejected:
///
/// ```compile_fail,E0503
/// # use ferrompi::{Win, WinFenceAssert, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<u64> {
/// let none = WinFenceAssert::none();
/// let origin = 1u64;
/// let mut old = 0u64;
/// win.fence(none, none, |e| {
///     e.fetch_and_op(&origin, &mut old, 1, 0, ReduceOp::Sum)?;
///     Ok(old)
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<u64> {
/// let none = WinFenceAssert::none();
/// let origin = 1u64;
/// let mut old = 0u64;
/// win.fence(none, none, |e| {
///     e.fetch_and_op(&origin, &mut old, 1, 0, ReduceOp::Sum)
/// })?;
/// Ok(old)
/// # }
/// ```
///
/// The storage of a result cannot be dropped inside the epoch either:
///
/// ```compile_fail,E0505
/// # use ferrompi::{Win, WinFenceAssert, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = 1u64;
/// let mut old = Box::new(0u64);
/// win.fence(none, none, |e| {
///     e.fetch_and_op(&origin, &mut *old, 1, 0, ReduceOp::Sum)?;
///     drop(old);
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = 1u64;
/// let mut old = Box::new(0u64);
/// win.fence(none, none, |e| {
///     e.fetch_and_op(&origin, &mut *old, 1, 0, ReduceOp::Sum)?;
///     Ok(())
/// })?;
/// drop(old);
/// Ok(())
/// # }
/// ```
///
/// ## The handle
///
/// The handle is borrowed for `'s`, so it cannot leave the closure, whether it is
/// stored outside:
///
/// ```compile_fail,E0521
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let mut keep = None;
/// win.fence(none, none, |e| {
///     keep = Some(e);
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |e| {
///     let mut keep = None;
///     keep = Some(e);
///     Ok(())
/// })
/// # }
/// ```
///
/// or returned, which the compiler reports without an error code:
///
/// ```compile_fail
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let _kept = win.fence(none, none, |e| Ok(e))?;
/// Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let _kept = win.fence(none, none, |e| Ok(()))?;
/// Ok(())
/// # }
/// ```
///
/// ## The window itself
///
/// The epoch borrows the window mutably, so epochs cannot nest on one window. An
/// owned window nests the same way, with error E0499 in place of E0500:
///
/// ```compile_fail,E0500
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| {
///     win.fence(none, none, |_| Ok(()))
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| Ok(()))?;
/// win.fence(none, none, |_| Ok(()))
/// # }
/// ```
///
/// The window's memory cannot be read inside the epoch either, while a peer may be
/// writing it:
///
/// ```compile_fail,E0502
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| {
///     let _local = win.local_slice();
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| Ok(()))?;
/// let _local = win.local_slice();
/// Ok(())
/// # }
/// ```
///
/// The handle has no `local_slice` or `local_slice_mut` either, so it offers no
/// other path to the memory. This holds for a fence or `start` epoch:
///
/// ```compile_fail,E0599
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |e| {
///     let _local = e.local_slice();
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| Ok(()))?;
/// let _local = win.local_slice();
/// Ok(())
/// # }
/// ```
///
/// and for a `lock` or `lock_all` epoch:
///
/// ```compile_fail,E0599
/// # use ferrompi::{Win};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// win.lock_all(|e| {
///     let _local = e.local_slice();
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// win.lock_all(|_| Ok(()))?;
/// let _local = win.local_slice();
/// Ok(())
/// # }
/// ```
///
/// ## Freeing a window inside an epoch
///
/// `MPI_Win_free` is only valid once the caller has completed its RMA involvement
/// (MPI-4.1 §12.2.5), so a window cannot be dropped inside its own epoch:
///
/// ```compile_fail,E0505
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let mut win = Win::<u64>::allocate(world, 4)?;
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| {
///     drop(win);
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let mut win = Win::<u64>::allocate(world, 4)?;
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| Ok(()))?;
/// drop(win);
/// Ok(())
/// # }
/// ```
///
/// and so cannot [`into_buffer`](Win::into_buffer), which takes the window by
/// value:
///
/// ```compile_fail,E0505
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let mut win = Win::<u64>::allocate(world, 4)?;
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| {
///     let _buf = win.into_buffer()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let mut win = Win::<u64>::allocate(world, 4)?;
/// let none = WinFenceAssert::none();
/// win.fence(none, none, |_| Ok(()))?;
/// let _buf = win.into_buffer()?;
/// Ok(())
/// # }
/// ```
///
/// `post` opens an exposure epoch with no closure, so no borrow keeps the window
/// alive through it. The window remembers the open epoch instead: until
/// `wait_exposure`, or a `test_exposure` that returns `true`, closes it,
/// [`into_buffer`](Win::into_buffer) returns
/// [`Error::InvalidState`](crate::Error::InvalidState), and dropping the window
/// leaks it, so `Mpi` skips `MPI_Finalize`. The `test_rma_free_exposed_window`
/// example checks both. The epoch's `&mut` borrow also keeps `post` out of a
/// `lock` or `lock_all` closure, where MPI-4.1 §12.5.3 makes it erroneous:
///
/// ```compile_fail,E0502
/// # use ferrompi::{Win, WinPscwAssert, Group};
/// # fn f(win: &mut Win<u64>, group: &Group) -> ferrompi::Result<()> {
/// win.lock_all(|_| {
///     win.post(group, WinPscwAssert::none())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinPscwAssert, Group};
/// # fn f(win: &mut Win<u64>, group: &Group) -> ferrompi::Result<()> {
/// win.lock_all(|_| Ok(()))?;
/// win.post(group, WinPscwAssert::none())
/// # }
/// ```
///
/// ## Creating a window
///
/// A buffer lent to `Win::create` could be dropped while MPI still exposes it, so
/// `create` takes the `Vec` by value and `into_buffer` gives it back:
///
/// ```compile_fail,E0308
/// # use ferrompi::{Win};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let mut buf = vec![0u64; 4];
/// let _win = Win::create(world, &mut buf)?;
/// Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win};
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let win = Win::create(world, vec![0u64; 4])?;
/// let _buf = win.into_buffer()?;
/// Ok(())
/// # }
/// ```
///
/// ## Request-based operations
///
/// `rput`, `rget` and `raccumulate` exist only on the passive-target handle. A
/// request borrows the origin for `'s` and its epoch's registry, which completes
/// every request before the unlock, so the origin cannot be a closure-local:
///
/// ```compile_fail,E0597
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// win.lock(LockType::Shared, 1, |e| {
///     let origin = [1u64; 4];
///     e.rput(&origin, 1, 0)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// win.lock(LockType::Shared, 1, |e| {
///     e.rput(&origin, 1, 0)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```compile_fail,E0597
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// win.lock(LockType::Shared, 1, |e| {
///     let mut origin = [0u64; 4];
///     e.rget(&mut origin, 1, 0)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let mut origin = [0u64; 4];
/// win.lock(LockType::Shared, 1, |e| {
///     e.rget(&mut origin, 1, 0)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```compile_fail,E0597
/// # use ferrompi::{Win, LockType, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// win.lock(LockType::Shared, 1, |e| {
///     let origin = [1u64; 4];
///     e.raccumulate(&origin, 1, 0, ReduceOp::Sum)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, LockType, ReduceOp};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// win.lock(LockType::Shared, 1, |e| {
///     e.raccumulate(&origin, 1, 0, ReduceOp::Sum)?.wait()?;
///     Ok(())
/// })
/// # }
/// ```
///
/// A request cannot leave the closure either, stored outside:
///
/// ```compile_fail,E0521
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// let mut keep = None;
/// win.lock(LockType::Shared, 1, |e| {
///     keep = Some(e.rput(&origin, 1, 0)?);
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// win.lock(LockType::Shared, 1, |e| {
///     let mut keep = None;
///     keep = Some(e.rput(&origin, 1, 0)?);
///     Ok(())
/// })
/// # }
/// ```
///
/// or returned:
///
/// ```compile_fail
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// let _kept = win.lock(LockType::Shared, 1, |e| e.rput(&origin, 1, 0))?;
/// Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, LockType};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let origin = [1u64; 4];
/// let _done = win.lock(LockType::Shared, 1, |e| e.rput(&origin, 1, 0)?.wait())?;
/// Ok(())
/// # }
/// ```
///
/// A fence or `start` epoch has no request-based operations at all:
///
/// ```compile_fail,E0599
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = [1u64; 4];
/// win.fence(none, none, |e| {
///     e.rput(&origin, 1, 0)?;
///     Ok(())
/// })
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Win, WinFenceAssert};
/// # fn f(win: &mut Win<u64>) -> ferrompi::Result<()> {
/// let none = WinFenceAssert::none();
/// let origin = [1u64; 4];
/// win.fence(none, none, |e| {
///     e.put(&origin, 1, 0)?;
///     Ok(())
/// })
/// # }
/// ```
///
/// `compare_and_swap` is rejected on a window of floats, where
/// [`AtomicMpiDatatype`](crate::AtomicMpiDatatype) documents it.
///
/// ## What the compiler does not check
///
/// - A `Win` that is forgotten leaks its buffer instead of freeing it, and so
///   does one dropped after `Mpi` skipped `MPI_Finalize` (the
///   `test_rma_forget_window` example). `Mpi` dropped inside an epoch still lets
///   the epoch complete and close (the `test_rma_epoch_drop_mpi` example).
/// - A peer's collective free waits for every rank, so it can block until the job
///   ends when a rank leaks its window instead of freeing it, whether by
///   forgetting it, through a refused or failed free, or through an open
///   exposure epoch. Likewise an origin's unlock or `complete` can wait for a
///   target that has stopped calling MPI (MPI-4.1 §12.7.3).
/// - The `NOPUT`, `NOSTORE` and `NOCHECK` assertions are the caller's contract
///   (MPI-4.1 §12.5.5). A broken `NOPUT` can let a peer's write land after the
///   target's closing call, possibly in a buffer `into_buffer` has already
///   handed back; a broken `NOCHECK` can let an access land before the epoch
///   opens; a broken `NOSTORE` can let a peer read stale data.
/// - A window locked and exposed at once, by `post` before `lock` or `lock_all`
///   or by a peer's lock during an exposure epoch, is erroneous (MPI-4.1 §12.5.3).
///   The open exposure epoch still keeps the window from being freed.
/// - A panic in a fence or `start` closure aborts the process, because the
///   closing call is collective, and one in a `lock` or `lock_all` closure
///   unwinds after the unlock. Both rely on unwinding: under `panic = "abort"`
///   the process stops at the panic itself, before any closing call, but with
///   Rust's message instead of the crate's, and the `test_rma_epoch_panic`
///   example, which checks that message, needs the default `panic = "unwind"`.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, Win};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Create a window over a buffer it owns until MPI_Win_free
/// let win = Win::create(&world, vec![0i32; 16]).unwrap();
/// assert!(win.raw_handle() >= 0);
/// assert_eq!(win.comm_size(), world.size());
/// drop(win); // MPI_Win_free called here
/// ```
pub struct Win<T: MpiDatatype> {
    /// The MPI window handle (index into the C-layer window table), or
    /// [`WIN_FREED`] once `into_buffer` has called the free.
    win_handle: i32,
    /// Non-null pointer to this process's local window buffer.
    ///
    /// For `Win::create` this is the pointer `Vec::as_mut_ptr` returned for
    /// `buffer`, the same one MPI holds. For `Win::allocate` this points into
    /// the MPI-managed allocation.
    local_ptr: NonNull<T>,
    /// Number of `T` elements in the local buffer.
    local_len: usize,
    /// Every rank's displacement unit and exposed byte length, packed by
    /// [`window_word`] and indexed by rank in the window's communicator.
    words: Box<[u64]>,
    /// The `Win::create` buffer, `None` for `Win::allocate`. Never read or
    /// borrowed while the window lives, so every slice stays a child of
    /// `local_ptr`; only `into_buffer` and `Drop` touch it, after the free.
    buffer: Option<Vec<T>>,
    /// Whether a [`post`](Win::post) exposure epoch is open: set by a
    /// successful `post`, cleared by a successful `wait_exposure` or a
    /// `test_exposure` that returned `true`. While it is set, a peer may still
    /// write into the window, so neither `into_buffer` nor `Drop` frees it.
    exposure_open: Cell<bool>,
}

/// The handle an active-target access epoch gives its closure.
///
/// [`Win::fence`] and [`Win::start`] open an epoch, run a closure with this
/// handle, and close the epoch when the closure returns, whether it returned
/// `Ok` or `Err`, with the epoch's closing call: the closing fence of
/// [`Win::fence`], or `MPI_Win_complete` of [`Win::start`]. The epoch is open
/// exactly while the closure runs, and the window is mutably borrowed for that
/// time.
///
/// # Lifetimes
///
/// The handle carries the invariant lifetimes of [`Scope`](crate::Scope): `'s`
/// is the epoch, and `'env` outlives it. [`put`](Self::put),
/// [`get`](Self::get), [`accumulate`](Self::accumulate),
/// [`get_accumulate`](Self::get_accumulate),
/// [`fetch_and_op`](Self::fetch_and_op) and
/// [`compare_and_swap`](Self::compare_and_swap) borrow their buffers for `'s`,
/// so a buffer outlives the closing call, which is when MPI stops using it, and
/// cannot be dropped or written before then. A buffer MPI writes (a `get`
/// destination or a `result`) cannot be read before then either. These six
/// methods reject a buffer declared inside the closure, because it is freed
/// before then. Declare buffers before the call to `fence` or `start`.
///
/// # Access set
///
/// A transfer's `target_rank` must be a rank the epoch gives access to: every
/// rank of the window's communicator for [`Win::fence`], the members of the
/// `group` for [`Win::start`]. Any other rank returns
/// [`Error::InvalidArgument`] with `arg: "target_rank"` and makes no MPI call,
/// because MPI-4.1 §12.5 makes such a transfer erroneous and its closing call
/// would not complete it.
///
/// # Thread safety
///
/// `ActiveEpoch` is neither `Send` nor `Sync`: it borrows a [`Win`], which is
/// neither.
pub struct ActiveEpoch<'s, 'env: 's, T: MpiDatatype> {
    win: &'s Win<T>,
    access: Access,
    scope: PhantomData<&'s mut &'s ()>,
    env: PhantomData<&'env mut &'env ()>,
}

impl<'s, T: MpiDatatype> ActiveEpoch<'s, '_, T> {
    /// One-sided write: copy `origin` into the remote rank's window memory.
    ///
    /// Wraps `MPI_Put`. The operation is posted at once, but it completes only
    /// when the epoch's closing call returns, and the data is visible at the
    /// target once the target's own closing call returns.
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local buffer whose contents are written to the target.
    /// * `target_rank` — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Put` fails; on MPI before 4.0 this includes
    ///   class `Count` for an `origin` of more than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    #[doc = include_str!("../examples/rma_put.rs")]
    /// ```
    pub fn put(&self, origin: &'s [T], target_rank: i32, target_disp: i64) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the epoch's
        // closing call, and `self.access` is this epoch's access set; see
        // `Win::rma_put`.
        unsafe {
            self.win
                .rma_put(&self.access, origin, target_rank, target_disp)
        }
    }

    /// One-sided read: copy data from a remote rank's window memory into
    /// `origin`.
    ///
    /// Wraps `MPI_Get`. The operation is posted at once, but `origin` holds
    /// the data only after the epoch's closing call returns, and stays borrowed
    /// until then, so it cannot be read earlier.
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local destination buffer filled by the remote read.
    /// * `target_rank` — Source rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Get` fails; on MPI before 4.0 this includes
    ///   class `Count` for an `origin` of more than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    #[doc = include_str!("../examples/rma_get.rs")]
    /// ```
    pub fn get(&self, origin: &'s mut [T], target_rank: i32, target_disp: i64) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the epoch's
        // closing call, and `self.access` is this epoch's access set; see
        // `Win::rma_put`.
        unsafe {
            self.win
                .rma_get(&self.access, origin, target_rank, target_disp)
        }
    }

    /// One-sided reduce-accumulate: combine data from a local origin buffer
    /// with a remote rank's window memory using a reduction operation.
    ///
    /// Wraps `MPI_Accumulate`. The operation is posted at once, but its effect
    /// at the target is visible only after the epoch's closing call returns and
    /// the target's own closing call returns.
    ///
    /// `op` is a [`ReduceOp`](crate::ReduceOp) or [`AccumulateOp::REPLACE`]
    /// (semantically `MPI_Put`); a user op is not accepted (MPI-4.1 §12.3.4).
    /// Under the default `accumulate_ops` window info value, `same_op_no_op`
    /// (MPI-4.1 §12.2.1), concurrent accumulates to one target location must
    /// use the same op or `MPI_NO_OP`, so mixing [`AccumulateOp::REPLACE`] and
    /// [`ReduceOp::Sum`](crate::ReduceOp::Sum) on one location is erroneous,
    /// and no type prevents it.
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local source buffer (read-only).
    /// * `target_rank` — Destination rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `op`          — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] with `arg: "op"` — if `op` is a predefined
    ///   op MPI does not define on `T` (a bitwise or logical op on
    ///   `f32`/`f64`); no MPI call is made.
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Accumulate` fails for any other reason; on
    ///   MPI before 4.0 this includes class `Count` for an `origin` of more
    ///   than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    #[doc = include_str!("../examples/rma_accumulate.rs")]
    /// ```
    pub fn accumulate(
        &self,
        origin: &'s [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<AccumulateOp>,
    ) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the epoch's
        // closing call, and `self.access` is this epoch's access set; see
        // `Win::rma_put`.
        unsafe {
            self.win
                .rma_accumulate(&self.access, origin, target_rank, target_disp, op)
        }
    }

    /// Atomic read-modify-write: fetch the pre-update remote value into
    /// `result`, then apply `op(origin, target_old)` at the target, each element
    /// atomically with respect to other accumulate operations on that location
    /// (MPI-4.1 §12.7.1).
    ///
    /// Wraps `MPI_Get_accumulate`. The operation is posted at once, but `result`
    /// holds the fetched values only after the epoch's closing call returns, and
    /// the update is visible at the target once the target's own closing call
    /// returns. `result` stays borrowed until then, so it cannot be read earlier.
    ///
    /// `op` is a [`ReduceOp`](crate::ReduceOp), [`FetchOp::NO_OP`] or
    /// [`FetchOp::REPLACE`]; a user op is not accepted (MPI-4.1 §12.3.4).
    /// [`FetchOp::NO_OP`] is the canonical "atomic get" pattern: `result`
    /// receives the target's current value and the target memory is left
    /// unchanged; `origin`'s contents are ignored, but its length is the
    /// count. [`FetchOp::REPLACE`] replaces the target value and also
    /// returns the pre-update value in `result`. Concurrent accumulates to one
    /// target location must use the same op or `MPI_NO_OP`, as for
    /// [`accumulate`](Self::accumulate) (MPI-4.1 §12.2.1).
    ///
    /// The count is `origin.len()`, and `result` must be as long as `origin`;
    /// pass sub-slices for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local input buffer (read-only).
    /// * `result`      — Local destination for the pre-update remote value,
    ///   as long as `origin`.
    /// * `target_rank` — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `op`          — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] with `arg: "op"` — if `op` is a predefined
    ///   op MPI does not define on `T` (a bitwise or logical op on
    ///   `f32`/`f64`); no MPI call is made.
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::BufferSize`] with `arg: "result"` — if `result` is not as
    ///   long as `origin`; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Get_accumulate` fails for any other reason;
    ///   on MPI before 4.0 this includes class `Count` for an `origin` of more
    ///   than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    #[doc = include_str!("../examples/rma_get_accumulate.rs")]
    /// ```
    pub fn get_accumulate(
        &self,
        origin: &'s [T],
        result: &'s mut [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        // SAFETY: `origin` and `result` are borrowed for `'s`, which ends only after
        // the epoch's closing call, and `self.access` is this epoch's access set;
        // see `Win::rma_put`.
        unsafe {
            self.win
                .rma_get_accumulate(&self.access, origin, result, target_rank, target_disp, op)
        }
    }

    /// Single-element atomic fetch-and-update: fetch the pre-update remote
    /// value into `result`, then apply `op(origin, target_old)` at the target,
    /// atomically with respect to other accumulate operations on that location
    /// (MPI-4.1 §12.7.1).
    ///
    /// Wraps `MPI_Fetch_and_op`. Restricted to predefined MPI datatypes (no
    /// derived or custom datatypes); the `T: MpiDatatype` bound enforces this at
    /// compile time. The operation is posted at once, but `result` holds the
    /// fetched value only after the epoch's closing call returns, and the update
    /// is visible at the target once the target's own closing call returns.
    /// `result` stays borrowed until then: read it after the closure returns.
    ///
    /// `op` is a [`ReduceOp`](crate::ReduceOp), [`FetchOp::NO_OP`] (an atomic
    /// get, which ignores `origin`'s value) or [`FetchOp::REPLACE`] (a swap); a
    /// user op is not accepted (MPI-4.1 §12.3.4). Concurrent accumulates to one
    /// target location must use the same op or `MPI_NO_OP`, as for
    /// [`accumulate`](Self::accumulate) (MPI-4.1 §12.2.1).
    ///
    /// # Arguments
    ///
    /// * `origin`      — The value to combine with the remote element.
    /// * `result`      — Local destination for the pre-update remote value.
    /// * `target_rank` — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `op`          — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] with `arg: "op"` — if `op` is a predefined
    ///   op MPI does not define on `T` (a bitwise or logical op on
    ///   `f32`/`f64`); no MPI call is made.
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Fetch_and_op` fails for any other reason.
    ///
    /// A call refused before MPI runs writes nothing into `result`, which
    /// stays borrowed until the epoch closes. After [`Error::Mpi`], `result`
    /// holds an unspecified value (MPI-4.1 §9.4).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 1).unwrap();
    ///
    /// // Rank 1 initialises its window slot 0 to 100
    /// if world.rank() == 1 {
    ///     win.local_slice_mut()[0] = 100;
    /// }
    ///
    /// // `origin` and `old` are declared before the epoch: MPI uses them until
    /// // the closing fence returns
    /// let origin = 1i32;
    /// let mut old = 0i32;
    /// win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
    ///     // Rank 0 atomically adds 1 to rank 1's slot 0 and fetches the old value
    ///     if world.rank() == 0 {
    ///         e.fetch_and_op(&origin, &mut old, 1, 0, ReduceOp::Sum)?;
    ///     }
    ///     Ok(())
    /// })
    /// .unwrap();
    ///
    /// // The epoch is closed: old is now valid
    /// if world.rank() == 0 {
    ///     assert_eq!(old, 100); // pre-update value; rank 1's slot 0 is now 101
    /// }
    /// ```
    pub fn fetch_and_op(
        &self,
        origin: &'s T,
        result: &'s mut T,
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        // SAFETY: `origin` and `result` are borrowed for `'s`, which ends only after
        // the epoch's closing call, when MPI has written `result`, and
        // `self.access` is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win
                .rma_fetch_and_op(&self.access, origin, result, target_rank, target_disp, op)
        }
    }
}

impl<'s, T: crate::AtomicMpiDatatype + MpiDatatype> ActiveEpoch<'s, '_, T> {
    /// Single-element atomic compare-and-swap: replace the remote element with
    /// `origin` if it equals `compare`, and fetch the pre-swap remote value into
    /// `result`.
    ///
    /// Wraps `MPI_Compare_and_swap`. `result` is **always** the pre-swap remote
    /// value, whether or not the swap happened; the swap happened if `result`
    /// equals `compare`. The operation is posted at once, but `result` holds
    /// that value only after the epoch's closing call returns, and the update is
    /// visible at the target once the target's own closing call returns.
    /// `result` stays borrowed until then: read it after the closure returns.
    ///
    /// This method exists only for the integer and byte types of
    /// [`AtomicMpiDatatype`], the ones MPI-4.1 §12.3.4 allows, so `f32` and
    /// `f64` are excluded.
    ///
    /// [`AtomicMpiDatatype`]: crate::AtomicMpiDatatype
    ///
    /// # Arguments
    ///
    /// * `origin`      — The new value to swap in if `compare` matches the
    ///   remote element.
    /// * `compare`     — The expected current remote value. The swap occurs
    ///   only if `remote_value == compare`.
    /// * `result`      — Local destination for the pre-swap remote value.
    /// * `target_rank` — Target rank (0-based, within this window's
    ///   communicator).
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::Mpi`] — if `MPI_Compare_and_swap` fails (for example, the MPI
    ///   version is below 3).
    ///
    /// A call refused before MPI runs writes nothing into `result`, which
    /// stays borrowed until the epoch closes. After [`Error::Mpi`], `result`
    /// holds an unspecified value (MPI-4.1 §9.4).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 1).unwrap();
    ///
    /// // Rank 1 initialises its window slot 0 to 100
    /// if world.rank() == 1 {
    ///     win.local_slice_mut()[0] = 100;
    /// }
    ///
    /// // `new_value`, `expected` and `old` are declared before the epoch: MPI
    /// // uses them until the closing fence returns
    /// let new_value = 200i32;
    /// let expected = 100i32;
    /// let mut old = 0i32;
    /// win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |e| {
    ///     // Rank 0: swap in 200 only if rank 1's slot 0 is 100
    ///     if world.rank() == 0 {
    ///         e.compare_and_swap(&new_value, &expected, &mut old, 1, 0)?;
    ///     }
    ///     Ok(())
    /// })
    /// .unwrap();
    ///
    /// // The epoch is closed: old is now valid
    /// if world.rank() == 0 {
    ///     assert_eq!(old, 100); // pre-swap value; rank 1's slot 0 is now 200
    ///     let swapped = old == expected;
    ///     assert!(swapped);
    /// }
    /// ```
    pub fn compare_and_swap(
        &self,
        origin: &'s T,
        compare: &'s T,
        result: &'s mut T,
        target_rank: i32,
        target_disp: i64,
    ) -> Result<()> {
        // SAFETY: `origin`, `compare` and `result` are borrowed for `'s`, which ends
        // only after the epoch's closing call, when MPI has written `result`, and
        // `self.access` is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win.rma_compare_and_swap(
                &self.access,
                origin,
                compare,
                result,
                target_rank,
                target_disp,
            )
        }
    }
}

/// The handle a passive-target epoch gives its closure.
///
/// [`Win::lock`] and [`Win::lock_all`] open an epoch, run a closure with this
/// handle, and end the epoch with `MPI_Win_unlock` or `MPI_Win_unlock_all` when
/// the closure returns, whether it returned `Ok` or `Err`, and also when it
/// panics. The epoch is open exactly while the closure runs, and the window is
/// mutably borrowed for that time.
///
/// The handle carries the calls MPI allows only inside a passive-target epoch:
/// the synchronization calls [`flush`](Self::flush),
/// [`flush_all`](Self::flush_all), [`flush_local`](Self::flush_local),
/// [`flush_local_all`](Self::flush_local_all) and [`sync`](Self::sync), and the
/// request-based transfers [`rput`](Self::rput), [`rget`](Self::rget) and
/// [`raccumulate`](Self::raccumulate) (MPI-4.1 §12.3.5). The epoch owns the
/// requests those return: before it unlocks it completes every one still
/// pending, also when the closure returned an error or panicked.
///
/// # Lifetimes
///
/// The handle carries the invariant lifetimes of [`Scope`](crate::Scope): `'s`
/// is the epoch, and `'env` outlives it. [`put`](Self::put),
/// [`get`](Self::get), [`accumulate`](Self::accumulate),
/// [`get_accumulate`](Self::get_accumulate),
/// [`fetch_and_op`](Self::fetch_and_op),
/// [`compare_and_swap`](Self::compare_and_swap), [`rput`](Self::rput),
/// [`rget`](Self::rget) and [`raccumulate`](Self::raccumulate) borrow their
/// buffers for `'s`, so a buffer outlives the unlock, which is when MPI stops
/// using it, and cannot be dropped or written before then, even after a flush
/// or after a request completed. A buffer MPI writes (a `get` or `rget`
/// destination or a `result`) cannot be read before then either. These nine
/// methods reject a buffer declared inside the closure, because it is freed
/// before then. Declare buffers before the call to `lock` or `lock_all`.
///
/// # Access set
///
/// A transfer's `target_rank` must be a rank the epoch gives access to: the
/// locked rank for [`Win::lock`], any rank of the window's communicator for
/// [`Win::lock_all`]. Any other rank returns [`Error::InvalidArgument`] with
/// `arg: "target_rank"` and makes no MPI call, because MPI-4.1 §12.5 makes such
/// a transfer erroneous and the unlock would not complete it.
/// [`flush`](Self::flush) and [`flush_local`](Self::flush_local) accept any
/// rank of the window's communicator.
///
/// # Completion
///
/// An operation is complete at the origin, where a `get` or `rget` destination
/// or a `result` then holds its values, once a flush of its `target_rank`
/// returns ([`flush`](Self::flush), [`flush_all`](Self::flush_all),
/// [`flush_local`](Self::flush_local) or
/// [`flush_local_all`](Self::flush_local_all)), and at the target once
/// [`flush`](Self::flush) or [`flush_all`](Self::flush_all) returns for it.
/// Every operation is complete at both ends at the latest when the unlock
/// returns.
///
/// # Thread safety
///
/// `PassiveEpoch` is neither `Send` nor `Sync`: it borrows a [`Win`], which is
/// neither.
pub struct PassiveEpoch<'s, 'env: 's, T: MpiDatatype> {
    win: &'s Win<T>,
    access: Access,
    registry: Registry,
    scope: PhantomData<&'s mut &'s ()>,
    env: PhantomData<&'env mut &'env ()>,
}

impl<'s, T: MpiDatatype> PassiveEpoch<'s, '_, T> {
    /// One-sided write, as [`ActiveEpoch::put`], in a passive-target epoch.
    ///
    /// The write completes as the type's `# Completion` section describes, and
    /// `origin` stays borrowed until the unlock returns. A `target_rank` outside
    /// the epoch's access set returns [`Error::InvalidArgument`], as there.
    pub fn put(&self, origin: &'s [T], target_rank: i32, target_disp: i64) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock,
        // and `self.access` is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win
                .rma_put(&self.access, origin, target_rank, target_disp)
        }
    }

    /// One-sided read, as [`ActiveEpoch::get`], in a passive-target epoch.
    ///
    /// `origin` holds the data once the read is complete at the origin (see the
    /// type's `# Completion` section), but stays borrowed until the unlock
    /// returns, so it cannot be read earlier. A `target_rank` outside the
    /// epoch's access set returns [`Error::InvalidArgument`], as there.
    pub fn get(&self, origin: &'s mut [T], target_rank: i32, target_disp: i64) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock,
        // and `self.access` is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win
                .rma_get(&self.access, origin, target_rank, target_disp)
        }
    }

    /// One-sided reduce-accumulate, as [`ActiveEpoch::accumulate`], in a
    /// passive-target epoch.
    ///
    /// The update completes as the type's `# Completion` section describes, and
    /// `origin` stays borrowed until the unlock returns. Concurrent accumulates
    /// to one target location must use the same op or `MPI_NO_OP`, as there. A
    /// `target_rank` outside the epoch's access set returns
    /// [`Error::InvalidArgument`], as there.
    pub fn accumulate(
        &self,
        origin: &'s [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<AccumulateOp>,
    ) -> Result<()> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock,
        // and `self.access` is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win
                .rma_accumulate(&self.access, origin, target_rank, target_disp, op)
        }
    }

    /// Atomic read-modify-write, as [`ActiveEpoch::get_accumulate`], in a
    /// passive-target epoch.
    ///
    /// `result` holds the fetched values once the operation is complete at the
    /// origin, and the target holds the update once it is complete there (see
    /// the type's `# Completion` section). `origin` and `result` stay borrowed
    /// until the unlock returns, so `result` cannot be read earlier. Concurrent
    /// accumulates to one target location must use the same op or `MPI_NO_OP`,
    /// as there. A `target_rank` outside the epoch's access set returns
    /// [`Error::InvalidArgument`], as there.
    pub fn get_accumulate(
        &self,
        origin: &'s [T],
        result: &'s mut [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        // SAFETY: `origin` and `result` are borrowed for `'s`, which ends only after
        // the unlock, and `self.access` is this epoch's access set; see
        // `Win::rma_put`.
        unsafe {
            self.win
                .rma_get_accumulate(&self.access, origin, result, target_rank, target_disp, op)
        }
    }

    /// Single-element atomic fetch-and-update, as [`ActiveEpoch::fetch_and_op`],
    /// in a passive-target epoch.
    ///
    /// `result` holds the fetched value once the operation is complete at the
    /// origin, and the target holds the update once it is complete there (see
    /// the type's `# Completion` section). `origin` and `result` stay borrowed
    /// until the unlock returns, so read `result` after the closure. Concurrent
    /// accumulates to one target location must use the same op or `MPI_NO_OP`,
    /// as there. Errors are as there, including [`Error::InvalidArgument`] for a
    /// `target_rank` outside the epoch's access set; after [`Error::Mpi`],
    /// `result` holds an unspecified value (MPI-4.1 §9.4).
    pub fn fetch_and_op(
        &self,
        origin: &'s T,
        result: &'s mut T,
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        // SAFETY: `origin` and `result` are borrowed for `'s`, which ends only after
        // the unlock, when MPI has written `result`, and `self.access` is this
        // epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win
                .rma_fetch_and_op(&self.access, origin, result, target_rank, target_disp, op)
        }
    }

    /// Request-returning one-sided write, as [`put`](Self::put), with a
    /// [`Request`] for its local completion.
    ///
    /// Wraps `MPI_Rput`. The request completes locally (MPI-4.1 §12.3.5): once
    /// [`Request::wait`], [`Request::test`] or a batch call reports it complete,
    /// MPI no longer reads `origin`. The target holds the data once the write is
    /// complete there (see the type's `# Completion` section). A request that
    /// was not waited is completed by the epoch before it unlocks, so dropping
    /// or forgetting it makes no MPI call. `origin` stays borrowed until the
    /// unlock returns, even after the request completed.
    ///
    /// [`Request::cancel`] on the returned request returns
    /// [`Error::NotSupported`] without calling MPI: `MPI_Cancel` is defined only
    /// for point-to-point requests, and MPICH and Open MPI reject it for RMA
    /// requests. The request stays pending; complete it with
    /// [`wait`](Request::wait) as usual.
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local source buffer (read-only).
    /// * `target_rank` — Destination rank, in the epoch's access set.
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::ResourceExhausted`] with resource
    ///   [`ResourceKind::Request`](crate::ResourceKind::Request) — if the
    ///   internal request table is full.
    /// * [`Error::Mpi`] — if `MPI_Rput` fails; on MPI before 4.0 this includes
    ///   class `Count` for an `origin` of more than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 4).unwrap();
    /// let buf = [10i32, 20, 30, 40];
    ///
    /// if world.rank() == 0 {
    ///     // `buf` is declared before the epoch: it stays borrowed until the
    ///     // unlock, which is when rank 1 holds the data
    ///     win.lock(LockType::Exclusive, 1, |e| {
    ///         let req = e.rput(&buf, 1, 0)?;
    ///         req.wait()?; // MPI no longer reads `buf`
    ///         Ok(())
    ///     })
    ///     .unwrap();
    /// }
    /// ```
    pub fn rput(
        &'s self,
        origin: &'s [T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<Request<'s>> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock;
        // the registry that owns the request completes it before the unlock; and
        // `self.access` is this epoch's access set; see `Win::rma_put`.
        let handle = unsafe {
            self.win
                .rma_rput(&self.access, origin, target_rank, target_disp)?
        };
        Ok(self.registry.request(handle, RequestKind::Rma))
    }

    /// Request-returning one-sided read, as [`get`](Self::get), with a
    /// [`Request`] for its local completion.
    ///
    /// Wraps `MPI_Rget`. When [`Request::wait`], [`Request::test`] or a batch
    /// call reports the request complete, `origin` holds the data read from
    /// `target_rank` (MPI-4.1 §12.3.5). A request that was not waited is
    /// completed by the epoch before it unlocks. `origin` stays mutably
    /// borrowed until the unlock returns, even after the request completed, so
    /// read it after the closure returns.
    ///
    /// [`Request::cancel`] on the returned request returns
    /// [`Error::NotSupported`] without calling MPI, as for
    /// [`rput`](Self::rput).
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local destination buffer filled by the remote read.
    /// * `target_rank` — Source rank, in the epoch's access set.
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::ResourceExhausted`] with resource
    ///   [`ResourceKind::Request`](crate::ResourceKind::Request) — if the
    ///   internal request table is full.
    /// * [`Error::Mpi`] — if `MPI_Rget` fails; on MPI before 4.0 this includes
    ///   class `Count` for an `origin` of more than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 4).unwrap();
    /// let mut buf = [0i32; 4];
    ///
    /// if world.rank() == 0 {
    ///     // `buf` is declared before the epoch and read after it
    ///     win.lock(LockType::Shared, 1, |e| {
    ///         let req = e.rget(&mut buf, 1, 0)?;
    ///         req.wait()?; // `buf` holds the data, and stays borrowed
    ///         Ok(())
    ///     })
    ///     .unwrap();
    ///     println!("{buf:?}");
    /// }
    /// ```
    pub fn rget(
        &'s self,
        origin: &'s mut [T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<Request<'s>> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock;
        // the registry that owns the request completes it before the unlock; and
        // `self.access` is this epoch's access set; see `Win::rma_put`.
        let handle = unsafe {
            self.win
                .rma_rget(&self.access, origin, target_rank, target_disp)?
        };
        Ok(self.registry.request(handle, RequestKind::Rma))
    }

    /// Request-returning one-sided reduce-accumulate, as
    /// [`accumulate`](Self::accumulate), with a [`Request`] for its local
    /// completion.
    ///
    /// Wraps `MPI_Raccumulate`. The request completes locally: once
    /// [`Request::wait`], [`Request::test`] or a batch call reports it complete,
    /// MPI no longer reads `origin`. The update is visible at the target once it
    /// is complete there (see the type's `# Completion` section). A request that
    /// was not waited is completed by the epoch before it unlocks. `origin`
    /// stays borrowed until the unlock returns, even after the request
    /// completed.
    ///
    /// `op` is as for [`ActiveEpoch::accumulate`], with
    /// [`AccumulateOp::REPLACE`] semantically `MPI_Rput`; a user op is not
    /// accepted (MPI-4.1 §12.3.4). Concurrent accumulates to one target location
    /// must use the same op or `MPI_NO_OP`, as there (MPI-4.1 §12.2.1).
    ///
    /// [`Request::cancel`] on the returned request returns
    /// [`Error::NotSupported`] without calling MPI, as for
    /// [`rput`](Self::rput).
    ///
    /// The count is `origin.len()`; pass a sub-slice for a shorter transfer.
    ///
    /// # Arguments
    ///
    /// * `origin`      — Local source buffer (read-only).
    /// * `target_rank` — Destination rank, in the epoch's access set.
    /// * `target_disp` — Displacement at the target, measured in units of
    ///   `disp_unit` (i.e., multiples of `size_of::<T>()`).
    /// * `op`          — Reduction operation to apply.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidArgument`] with `arg: "op"` — if `op` is a predefined
    ///   op MPI does not define on `T` (a bitwise or logical op on
    ///   `f32`/`f64`); no MPI call is made.
    /// * [`Error::InvalidArgument`] — if `target_rank` is not a rank of the
    ///   window's communicator (including -1) or of the epoch's access set
    ///   (`arg: "target_rank"`), `target_disp` is negative, or the access does
    ///   not fit in the target rank's window; no MPI call is made.
    /// * [`Error::Finalized`] — once [`Mpi`](crate::Mpi) is being dropped.
    /// * [`Error::ResourceExhausted`] with resource
    ///   [`ResourceKind::Request`](crate::ResourceKind::Request) — if the
    ///   internal request table is full.
    /// * [`Error::Mpi`] — if `MPI_Raccumulate` fails for any other reason; on
    ///   MPI before 4.0 this includes class `Count` for an `origin` of more
    ///   than `i32::MAX` elements.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, ReduceOp, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<i32>::allocate(&world, 4).unwrap();
    /// let buf = [10i32, 20, 30, 40];
    ///
    /// if world.rank() == 0 {
    ///     // `buf` is declared before the epoch: it stays borrowed until the
    ///     // unlock, which is when the update is complete at rank 1
    ///     win.lock(LockType::Exclusive, 1, |e| {
    ///         let req = e.raccumulate(&buf, 1, 0, ReduceOp::Sum)?;
    ///         req.wait()?; // MPI no longer reads `buf`
    ///         Ok(())
    ///     })
    ///     .unwrap();
    /// }
    /// ```
    pub fn raccumulate(
        &'s self,
        origin: &'s [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<AccumulateOp>,
    ) -> Result<Request<'s>> {
        // SAFETY: `origin` is borrowed for `'s`, which ends only after the unlock;
        // the registry that owns the request completes it before the unlock; and
        // `self.access` is this epoch's access set; see `Win::rma_put`.
        let handle = unsafe {
            self.win
                .rma_raccumulate(&self.access, origin, target_rank, target_disp, op)?
        };
        Ok(self.registry.request(handle, RequestKind::Rma))
    }

    /// Complete the epoch's pending RMA operations to `rank`, at the origin
    /// and at the target.
    ///
    /// Wraps `MPI_Win_flush`. When it returns, every operation this rank
    /// issued to `rank` in the epoch is complete at both ends, and the epoch
    /// stays open.
    ///
    /// # Arguments
    ///
    /// * `rank` - The target rank to flush (0-based, within the window's
    ///   communicator). It need not be in the epoch's access set: in a
    ///   [`Win::lock`] epoch only the locked rank has operations to complete,
    ///   and a flush of another rank completes nothing.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` with `arg: "rank"` if `rank` is outside the
    ///   window's communicator; no MPI call is made.
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped.
    /// - `Error::Mpi` with `operation: Some("win_flush")` if MPI reports an
    ///   error.
    pub fn flush(&self, rank: i32) -> Result<()> {
        check_rank(self.win.words.len(), "rank", rank)?;
        // SAFETY: `win` is a valid MPI window handle, a passive-target epoch is
        // open on this window, which MPI-4.1 §12.5.4 requires, and `rank` was
        // checked against the window's group above.
        let ret = unsafe { ffi::ferrompi_win_flush(rank, self.win.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }

    /// Complete the epoch's pending RMA operations to every rank, at the
    /// origin and at the targets.
    ///
    /// Wraps `MPI_Win_flush_all`. When it returns, every operation this rank
    /// issued in the epoch is complete at both ends, and the epoch stays open.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped.
    /// - `Error::Mpi` with `operation: Some("win_flush_all")` if MPI reports
    ///   an error.
    pub fn flush_all(&self) -> Result<()> {
        // SAFETY: `win` is a valid MPI window handle, and a passive-target
        // epoch is open on this window, which MPI-4.1 §12.5.4 requires.
        let ret = unsafe { ffi::ferrompi_win_flush_all(self.win.win_handle) };
        Error::check_with_op(ret, "win_flush_all")
    }

    /// Complete the epoch's pending RMA operations to `rank` at the origin
    /// only.
    ///
    /// Wraps `MPI_Win_flush_local`. When it returns, the origin buffers of
    /// every operation this rank issued to `rank` in the epoch are safe to
    /// reuse, but the target may not have observed the writes yet.
    ///
    /// # Arguments
    ///
    /// * `rank` - The target rank whose pending operations are locally
    ///   completed (0-based, within the window's communicator). As for
    ///   [`flush`](Self::flush), it need not be in the epoch's access set.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` with `arg: "rank"` if `rank` is outside the
    ///   window's communicator; no MPI call is made.
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped.
    /// - `Error::Mpi` with `operation: Some("win_flush_local")` if MPI reports
    ///   an error.
    pub fn flush_local(&self, rank: i32) -> Result<()> {
        check_rank(self.win.words.len(), "rank", rank)?;
        // SAFETY: `win` is a valid MPI window handle, a passive-target epoch is
        // open on this window, which MPI-4.1 §12.5.4 requires, and `rank` was
        // checked against the window's group above.
        let ret = unsafe { ffi::ferrompi_win_flush_local(rank, self.win.win_handle) };
        Error::check_with_op(ret, "win_flush_local")
    }

    /// Complete the epoch's pending RMA operations to every rank at the origin
    /// only.
    ///
    /// Wraps `MPI_Win_flush_local_all`. When it returns, the origin buffers of
    /// every operation this rank issued in the epoch are safe to reuse, but the
    /// targets may not have observed the writes yet.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped.
    /// - `Error::Mpi` with `operation: Some("win_flush_local_all")` if MPI
    ///   reports an error.
    pub fn flush_local_all(&self) -> Result<()> {
        // SAFETY: `win` is a valid MPI window handle, and a passive-target
        // epoch is open on this window, which MPI-4.1 §12.5.4 requires.
        let ret = unsafe { ffi::ferrompi_win_flush_local_all(self.win.win_handle) };
        Error::check_with_op(ret, "win_flush_local_all")
    }

    /// Synchronize the public and private copies of this window's local memory.
    ///
    /// Wraps `MPI_Win_sync`, a local memory barrier: it does not communicate
    /// and does not complete pending RMA operations. MPI allows it only inside
    /// a passive-target epoch (MPI-4.1 §12.5.4), which this handle
    /// guarantees; MPICH rejects it elsewhere.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped.
    /// - `Error::Mpi` with `operation: Some("win_sync")` if MPI reports an
    ///   error.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: `win` is a valid MPI window handle, and a passive-target
        // epoch is open on this window, the only place `MPI_Win_sync` is
        // allowed (MPI-4.1 §12.5.4).
        let ret = unsafe { ffi::ferrompi_win_sync(self.win.win_handle) };
        Error::check_with_op(ret, "win_sync")
    }
}

impl<'s, T: crate::AtomicMpiDatatype + MpiDatatype> PassiveEpoch<'s, '_, T> {
    /// Single-element atomic compare-and-swap, as
    /// [`ActiveEpoch::compare_and_swap`], in a passive-target epoch.
    ///
    /// `result` holds the pre-swap value once the operation is complete at the
    /// origin, and the target holds the update once it is complete there (see
    /// the type's `# Completion` section). `origin`, `compare` and `result` stay
    /// borrowed until the unlock returns, so read `result` after the closure. It
    /// exists only for the types of
    /// [`AtomicMpiDatatype`](crate::AtomicMpiDatatype), as there. Errors are as
    /// there, including [`Error::InvalidArgument`] for a `target_rank` outside
    /// the epoch's access set; after [`Error::Mpi`], `result` holds an
    /// unspecified value (MPI-4.1 §9.4).
    pub fn compare_and_swap(
        &self,
        origin: &'s T,
        compare: &'s T,
        result: &'s mut T,
        target_rank: i32,
        target_disp: i64,
    ) -> Result<()> {
        // SAFETY: `origin`, `compare` and `result` are borrowed for `'s`, which ends
        // only after the unlock, when MPI has written `result`, and `self.access`
        // is this epoch's access set; see `Win::rma_put`.
        unsafe {
            self.win.rma_compare_and_swap(
                &self.access,
                origin,
                compare,
                result,
                target_rank,
                target_disp,
            )
        }
    }
}

/// Reports why an epoch could not be closed cleanly and aborts the process:
/// its origin buffers are still in MPI's hands, so there is no way to hand
/// them back.
#[cold]
#[inline(never)]
fn epoch_abort(reason: &str) -> ! {
    let msg = format!("ferrompi: {reason}; aborting\n");
    let _ = std::io::stderr().write_all(msg.as_bytes());
    std::process::abort();
}

/// The outcome of an epoch whose closing call returned `ret`: `result`'s error
/// first, then the closing call's. A closing call refused before MPI ran it
/// aborts the process.
fn closing_result<R>(ret: c_int, close_op: &'static str, result: Result<R>) -> Result<R> {
    if ret == FERROMPI_ERR_FINALIZED || ret == FERROMPI_ERR_THREAD_LEVEL {
        epoch_abort("an RMA epoch's closing call was refused");
    }
    let value = result?;
    Error::check_with_op(ret, close_op).map(|()| value)
}

impl<T: MpiDatatype> Win<T> {
    /// Create an RMA window over a buffer the window then owns.
    ///
    /// Wraps `MPI_Win_create`. The window takes `buf` and keeps it, neither
    /// resized nor borrowed, until `MPI_Win_free` has succeeded (see
    /// [`into_buffer`](Self::into_buffer) and the type's `# Dropping`
    /// section), so no safe call can free or reuse memory a peer may still
    /// access. Reach it with [`local_slice`](Self::local_slice) and
    /// [`local_slice_mut`](Self::local_slice_mut). `buf` is consumed on every
    /// return: dropped if MPI created no window, leaked if it did and the
    /// window cannot be returned.
    ///
    /// `disp_unit` is set to `size_of::<T>()` and `size` to
    /// `buf.len() * size_of::<T>()`. Creation first exchanges each rank's
    /// displacement unit and exposed length with one allgather over `comm`.
    ///
    /// # Arguments
    ///
    /// * `comm` - Communicator; all processes must call this collectively.
    /// * `buf`  - The vector that forms this process's local window memory.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any rank's `buf` exposed byte length does not fit in 56 bits
    ///   (`Error::InvalidArgument` on every rank).
    /// - The length exchange fails (`Error::Mpi` with
    ///   `operation: Some("allgather")`); this can happen on some ranks only,
    ///   and the other ranks can then block creating the window.
    /// - The MPI call fails (`Error::Mpi` with `operation: Some("win_create")`).
    /// - The window table is full (`Error::ResourceExhausted`; the window is
    ///   then leaked: never freed, and counted as alive so `Mpi` skips
    ///   `MPI_Finalize`, and the buffer is leaked with it).
    ///
    /// In the last case, a peer whose own call succeeded blocks in its
    /// collective free (`MPI_Win_free`) waiting for this rank.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::create(&world, vec![0f64; 64]).unwrap();
    /// ```
    pub fn create(comm: &Communicator, buf: Vec<T>) -> Result<Self> {
        let words = exchange_window_words(comm, window_word::<T>(buf.len()))?;
        let (size, disp_unit) = win_size_and_disp_unit::<T>(buf.len())?;
        let mut buf = buf;
        let (base, _, _) = vec_buf_mut(&mut buf);
        let mut win_handle: i32 = -1;

        let created = counted_window_call("win_create", &mut win_handle, |handle| {
            // SAFETY: `base` is the pointer `Vec::as_mut_ptr` returned for
            // `buf`, valid for `size` bytes (a dangling pointer with size 0 for
            // an empty `Vec`), and the window owns `buf` from here: it is moved
            // into `buffer` only after this call succeeded, never resized or
            // borrowed while the window lives, and every path that skips
            // `MPI_Win_free` forgets it. The C shim validates the comm handle
            // before calling MPI.
            unsafe {
                ffi::ferrompi_win_create(
                    base,
                    size,
                    disp_unit,
                    -1, // MPI_INFO_NULL
                    comm.handle,
                    handle,
                )
            }
        });
        if let Err(err) = created {
            // A window the shim created but could not register is never
            // freed, so the buffer it exposes must outlive the process.
            if win_handle == FERROMPI_WIN_LEAKED {
                std::mem::forget(buf);
            }
            return Err(err);
        }

        // SAFETY: `Vec::as_mut_ptr` is never null.
        let local_ptr = unsafe { NonNull::new_unchecked(base.cast::<T>()) };
        Ok(Win {
            win_handle,
            local_ptr,
            local_len: buf.len(),
            words,
            buffer: Some(buf),
            exposure_open: Cell::new(false),
        })
    }

    /// Allocate an RMA window with MPI-managed memory.
    ///
    /// Wraps `MPI_Win_allocate`. MPI allocates `local_count` elements of type
    /// `T` for this process. The allocation is freed automatically when the
    /// window is dropped (via `MPI_Win_free`).
    ///
    /// `disp_unit` is set to `size_of::<T>()` and `size` to
    /// `local_count * size_of::<T>()`. Creation first exchanges each rank's
    /// displacement unit and exposed length with one allgather over `comm`.
    ///
    /// The local segment reads as zero when the call returns.
    ///
    /// # Arguments
    ///
    /// * `comm`        - Communicator; all processes must call this collectively.
    /// * `local_count` - Number of `T` elements to allocate for this process.
    ///   Zero is allowed; a valid window with a dangling base pointer is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any rank's `local_count` exposed byte length does not fit in 56 bits
    ///   (`Error::InvalidArgument` on every rank).
    /// - The length exchange fails (`Error::Mpi` with
    ///   `operation: Some("allgather")`); this can happen on some ranks only,
    ///   and the other ranks can then block creating the window.
    /// - The MPI call fails (`Error::Mpi` with `operation: Some("win_allocate")`);
    ///   this can happen on some ranks only, and the other ranks can then
    ///   block in the barrier after the new segment is zeroed.
    /// - The window table is full (`Error::ResourceExhausted`; the window is
    ///   then leaked: never freed, and counted as alive so `Mpi` skips `MPI_Finalize`).
    /// - An error occurs while zeroing the new segment (`Error::Mpi` with
    ///   `operation: Some("win_allocate")`; the window is
    ///   then leaked: never freed, and counted as alive so `Mpi` skips `MPI_Finalize`).
    /// - MPI returns a null pointer for a non-zero count (`Error::Internal`;
    ///   the window is then leaked: never freed, and counted as alive so `Mpi`
    ///   skips `MPI_Finalize`).
    ///
    /// In the last three cases, a peer whose own call succeeded blocks in its
    /// collective `Drop` (`MPI_Win_free`) waiting for this rank.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 32).unwrap();
    /// ```
    pub fn allocate(comm: &Communicator, local_count: usize) -> Result<Self> {
        let words = exchange_window_words(comm, window_word::<T>(local_count))?;
        let (size, disp_unit) = win_size_and_disp_unit::<T>(local_count)?;
        let mut baseptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut win_handle: i32 = -1;

        counted_window_call("win_allocate", &mut win_handle, |handle| {
            // SAFETY: We pass valid out-parameter pointers to the C shim, which
            // calls MPI_Win_allocate. The C shim validates the comm handle. The
            // returned baseptr is either a valid MPI-allocated pointer or null
            // (only valid if local_count == 0).
            unsafe {
                ffi::ferrompi_win_allocate(
                    size,
                    disp_unit,
                    -1, // MPI_INFO_NULL
                    comm.handle,
                    &mut baseptr,
                    handle,
                )
            }
        })?;

        let local_ptr = if local_count == 0 {
            // Zero-count: use a dangling aligned pointer (same trick as
            // SharedWindow::remote_slice for zero-size slices).
            NonNull::<T>::dangling()
        } else {
            NonNull::new(baseptr.cast::<T>()).ok_or_else(|| {
                Error::Internal("Win_allocate returned null base pointer for non-zero count".into())
            })?
        };

        Ok(Win {
            win_handle,
            local_ptr,
            local_len: local_count,
            words,
            buffer: None,
            exposure_open: Cell::new(false),
        })
    }

    /// Free the window and return its buffer.
    ///
    /// Calls `MPI_Win_free` and, once it has succeeded, returns the vector
    /// `Win::create` took, or `None` for a `Win::allocate` window (MPI frees
    /// its memory with the window). Unlike `Drop`, it reports a failed or
    /// refused free.
    ///
    /// The call is collective: it can block until every peer has called
    /// `MPI_Win_free`, because MPI-4.1 §12.2.5 requires it to wait until every
    /// passive-target access to this rank has completed. An error can reach
    /// some ranks only. A rank whose free failed, or was refused, leaks its
    /// buffer and keeps the window counted as alive; a peer whose free
    /// succeeded gets its buffer back; a peer that never calls the free
    /// blocks in its own.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidState` while an exposure epoch opened by
    ///   [`post`](Self::post) is open, because a peer may still write into the
    ///   window (MPI-4.1 §12.2.5): no MPI call is made. The call takes the
    ///   window, so the refusal drops it, which leaks the window and its buffer
    ///   and keeps the window counted as alive. Close the epoch with
    ///   [`wait_exposure`](Self::wait_exposure) or a
    ///   [`test_exposure`](Self::test_exposure) that returns `true` first.
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped: no MPI
    ///   call is made and the buffer is leaked.
    /// - `Error::Mpi` with `operation: Some("win_free")` if the free fails:
    ///   the buffer is leaked and the window stays counted as alive, so `Mpi`
    ///   skips `MPI_Finalize`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::create(&world, vec![0f64; 64]).unwrap();
    /// win.local_slice_mut()[0] = 1.5;
    /// let buf = win.into_buffer().unwrap().expect("Win::create owns its buffer");
    /// assert_eq!(buf[0], 1.5);
    /// ```
    pub fn into_buffer(mut self) -> Result<Option<Vec<T>>> {
        if self.exposure_open.get() {
            return Err(Error::InvalidState {
                reason: "an exposure epoch opened by post is still open",
            });
        }
        let _call = rt::enter().map_err(|code| Error::from_code_with_op(code, "win_free"))?;
        // SAFETY: `win_handle` is the live window's handle. Only this call
        // frees it: `WIN_FREED`, set right below, keeps `Drop` from freeing it
        // again.
        let ret = unsafe { ffi::ferrompi_win_free(self.win_handle) };
        // The shim releases the table slot whatever MPI returned, so `Drop`
        // must not free again.
        self.win_handle = WIN_FREED;
        if ret == 0 {
            // Release: pairs with `live_windows`'s Acquire; this decrement
            // follows the successful free.
            LIVE_WINDOWS.fetch_sub(1, Ordering::Release);
            return Ok(self.buffer.take());
        }
        std::mem::forget(self.buffer.take());
        Err(Error::from_code_with_op(ret, "win_free"))
    }

    /// Get a shared slice of this process's local window buffer.
    ///
    /// For `Win::create`, this returns a view into the buffer the window
    /// owns. For `Win::allocate`, this returns a view into the MPI-managed
    /// buffer.
    ///
    /// The borrow lifetime is tied to `&self`, so the slice cannot outlive the
    /// window. Proper MPI epoch synchronization (fence, lock, or PSCW) must be
    /// used before reading data written by remote processes.
    pub fn local_slice(&self) -> &[T] {
        // SAFETY: `local_ptr` was set in the constructor from either the
        // pointer `Vec::as_mut_ptr` returned for the owned buffer
        // (`Win::create`) or a valid MPI allocation (`Win::allocate`). Both
        // are valid for `local_len` elements of type T until the window is
        // freed, which needs `self` by value or a drop. The pointer is
        // non-null (zero-length windows use NonNull::dangling()). We hold
        // `&self` so no mutable alias exists.
        unsafe { std::slice::from_raw_parts(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get a mutable slice of this process's local window buffer.
    ///
    /// For `Win::create`, this returns a mutable view into the buffer the
    /// window owns. For `Win::allocate`, this returns a mutable view into the
    /// MPI-managed buffer.
    ///
    /// The borrow lifetime is tied to `&mut self`, ensuring exclusive access.
    /// Proper MPI epoch synchronization must be used before writing data that
    /// remote processes may read.
    pub fn local_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: `local_ptr` was set in the constructor from either the
        // pointer `Vec::as_mut_ptr` returned for the owned buffer
        // (`Win::create`) or a valid MPI allocation (`Win::allocate`). Both
        // are valid for `local_len` elements of type T until the window is
        // freed. We hold `&mut self`, ensuring exclusive access and no
        // concurrent shared references to the same memory.
        unsafe { std::slice::from_raw_parts_mut(self.local_ptr.as_ptr(), self.local_len) }
    }

    /// Get this window's raw handle.
    ///
    /// The value is ferrompi's internal table index for this window. It is
    /// not an MPI handle and cannot be passed to MPI; use it only to tell
    /// objects apart, for example in logs.
    pub fn raw_handle(&self) -> i32 {
        self.win_handle
    }

    /// Get the number of processes in the window's communicator.
    ///
    /// Equals the size of the communicator used to create the window and
    /// determines the valid rank range for RMA operations.
    pub fn comm_size(&self) -> i32 {
        self.words.len() as i32
    }

    /// Run a closure in a fence epoch (active-target synchronization).
    ///
    /// Wraps `MPI_Win_fence`, which is collective over this window's
    /// communicator: every rank calls `fence` at the same logical program
    /// point. The call opens the epoch with a fence carrying `open`, runs `f`
    /// with the epoch's [`ActiveEpoch`], and closes it with a fence carrying
    /// `close`, whether `f` returned `Ok` or `Err`, so every rank makes the
    /// closing collective call. When it returns, the RMA operations `f` issued
    /// are complete at this rank, and visible at each target once that target's
    /// closing fence returns.
    ///
    /// # Arguments
    ///
    /// * `open`  — Assertion hint bitmask of the opening fence. Use
    ///   [`WinFenceAssert::none()`] when no hints are needed. Compose multiple
    ///   hints with `|`. [`WinFenceAssert::no_succeed()`] is rejected.
    /// * `close` — Assertion hint bitmask of the closing fence.
    ///   [`WinFenceAssert::no_precede()`] is rejected.
    /// * `f`     — The closure that issues the epoch's RMA operations.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` with `arg: "open"` if `open` carries
    ///   `no_succeed`, and with `arg: "close"` if `close` carries
    ///   `no_precede`; no MPI call is made. Either would let MPI keep using an
    ///   origin buffer after the closing fence returned (MPI-4.1 §12.5.5).
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped: the
    ///   opening fence is refused and `f` does not run.
    /// - `Error::Mpi` with `operation: Some("win_fence")` if the opening fence
    ///   fails (`f` does not run) or the closing fence fails.
    ///
    /// When `f` returns an error, that error is returned in place of a closing
    /// fence's. MPI ran the closing fence, and the crate trusts that it
    /// completed or failed the epoch's operations, as MPICH and Open MPI do;
    /// MPI-4.1 §9.4 (p.460) does not promise it.
    ///
    /// # Aborts
    ///
    /// A panic in `f` aborts the process: the closing fence is collective, and
    /// one rank closing it alone can hang or mismatch the job. A closing fence
    /// refused before MPI ran it also aborts, since the epoch's origin buffers
    /// are still in MPI's hands; in a debug build at
    /// [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized), an overlap
    /// with another thread's MPI call is the one way to reach that.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinFenceAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<f64>::allocate(&world, 16).unwrap();
    ///
    /// win.fence(WinFenceAssert::none(), WinFenceAssert::none(), |_epoch| {
    ///     // ... RMA operations would go here ...
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub fn fence<'env, R>(
        &mut self,
        open: WinFenceAssert,
        close: WinFenceAssert,
        f: impl for<'s> FnOnce(&'s ActiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        check_fence_asserts(open, close)?;
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // `ferrompi_win_create` or `ferrompi_win_allocate`. The assert value
        // is a valid bitmask obtained from `WinFenceAssert`, which is either
        // 0 (none) or a combination of values returned by the MPI implementation.
        let ret = unsafe { ffi::ferrompi_win_fence(open.bits(), self.win_handle) };
        Error::check_with_op(ret, "win_fence")?;
        self.run_epoch(
            "win_fence",
            // SAFETY: as for the opening fence; the closing fence takes the
            // same arguments.
            |win_handle| unsafe { ffi::ferrompi_win_fence_close(close.bits(), win_handle) },
            Access::All,
            f,
        )
    }

    /// Runs `f` in the epoch the caller just opened, with `access` as its access
    /// set, then makes the closing call `close`, whatever `f` returned. A panic
    /// in `f` aborts before the closing call, and so does a closing call refused
    /// before MPI ran it. Otherwise `f`'s error is returned first, then the
    /// closing call's.
    fn run_epoch<'env, R>(
        &mut self,
        close_op: &'static str,
        close: impl FnOnce(i32) -> c_int,
        access: Access,
        f: impl for<'s> FnOnce(&'s ActiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        let epoch = ActiveEpoch {
            win: &*self,
            access,
            scope: PhantomData,
            env: PhantomData,
        };
        let result = match catch_unwind(AssertUnwindSafe(|| f(&epoch))) {
            Ok(result) => result,
            Err(_) => epoch_abort("a fence or PSCW epoch was still open during unwind"),
        };
        closing_result(close(self.win_handle), close_op, result)
    }

    /// Start an exposure epoch (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_post`. This rank exposes its window memory to the
    /// ranks in `group`, which are expected to run [`Win::start`]. Close the
    /// exposure epoch with [`Win::wait_exposure`] or [`Win::test_exposure`].
    ///
    /// The exposure epoch has no handle: this rank issues no RMA call in it.
    /// While it is open, a peer may still write into this window, so
    /// [`into_buffer`](Win::into_buffer) is refused and dropping the window
    /// leaks it instead of freeing it (MPI-4.1 §12.2.5; see the type's
    /// `# Dropping` section).
    ///
    /// # Arguments
    ///
    /// * `group`  — The group of ranks that will issue RMA operations against
    ///   this rank's window.
    /// * `assert` — Assertion hint bitmask. Use [`WinPscwAssert::default()`]
    ///   (or [`WinPscwAssert::none()`]) when no optimization hints are needed.
    ///
    /// # Errors
    ///
    /// * `Error::Finalized` or `Error::ThreadLevelViolation` if the call is
    ///   refused before MPI runs it: no exposure epoch opens.
    /// * `Error::Mpi { operation: Some("win_post"), .. }` if MPI reports an
    ///   error. The exposure epoch then counts as open, because a post that
    ///   failed may still have opened it (MPI-4.1 §9.4, p.460).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinPscwAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// let access_group = world.group().unwrap().include(&[1]).unwrap();
    ///
    /// // Rank 0 exposes its window to rank 1
    /// win.post(&access_group, WinPscwAssert::default()).unwrap();
    /// win.wait_exposure().unwrap();
    /// ```
    pub fn post(&self, group: &Group, assert: WinPscwAssert) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. `group.handle`
        // returns a valid group handle from the C-layer group table.
        // `assert.bits()` is either 0 or a combination of MPI_MODE_* constants
        // returned by the MPI implementation.
        let ret = unsafe { ffi::ferrompi_win_post(group.handle, assert.bits(), self.win_handle) };
        // Any other return means MPI ran the call, and a post that failed may
        // still have opened the epoch (MPI-4.1 §9.4), so the window leaks.
        if ret != FERROMPI_ERR_FINALIZED && ret != FERROMPI_ERR_THREAD_LEVEL {
            self.exposure_open.set(true);
        }
        Error::check_with_op(ret, "win_post")
    }

    /// Run a closure in a PSCW access epoch (active-target synchronization).
    ///
    /// Wraps `MPI_Win_start` and `MPI_Win_complete`. This rank will issue RMA
    /// operations against the ranks in `group`, which must have called (or will
    /// call) [`Win::post`]. The call translates `group` into ranks of the
    /// window's communicator, once, as the epoch's access set: a transfer to any
    /// other rank returns an error (see [`ActiveEpoch`]). It then opens the
    /// epoch, runs `f` with the epoch's [`ActiveEpoch`], and closes it with
    /// `MPI_Win_complete`, whether `f` returned `Ok` or `Err`, so the target
    /// ranks' [`Win::wait_exposure`] never waits on an epoch left open. When it
    /// returns, the RMA operations `f` issued are complete at the origin.
    ///
    /// # Arguments
    ///
    /// * `group`  — The group of ranks that have called or will call
    ///   [`Win::post`].
    /// * `assert` — Assertion hint bitmask. Use [`WinPscwAssert::default()`]
    ///   when no optimization hints are needed.
    /// * `f`      — The closure that issues the epoch's RMA operations.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped: the
    ///   epoch is not opened and `f` does not run.
    /// - `Error::InvalidArgument` with `arg: "group"` if a member of `group` is
    ///   not a rank of the window's communicator; no MPI call opens the epoch
    ///   and `f` does not run.
    /// - `Error::ResourceExhausted` with resource
    ///   [`ResourceKind::Group`](crate::ResourceKind::Group) if the group table
    ///   has no slot for the window's group, which the translation needs
    ///   briefly (`f` does not run).
    /// - `Error::Mpi` with `operation: Some("win_get_group")`,
    ///   `Some("group_size")` or `Some("group_translate_ranks")` if translating
    ///   `group` fails (`f` does not run).
    /// - `Error::Mpi` with `operation: Some("win_start")` if opening the epoch
    ///   fails (`f` does not run).
    /// - `Error::Mpi` with `operation: Some("win_complete")` if closing it
    ///   fails.
    ///
    /// When `f` returns an error, that error is returned in place of
    /// `MPI_Win_complete`'s. MPI ran the call, and the crate trusts that it
    /// completed or failed the epoch's operations, as MPICH and Open MPI do;
    /// MPI-4.1 §9.4 (p.460) does not promise it.
    ///
    /// # Aborts
    ///
    /// A panic in `f` aborts the process: closing the epoch synchronizes with
    /// the target ranks, and one rank closing it alone can hang or mismatch the
    /// job. An `MPI_Win_complete` refused before MPI ran it also aborts, since
    /// the epoch's origin buffers are still in MPI's hands; in a debug build at
    /// [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized), an overlap
    /// with another thread's MPI call is the one way to reach that.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win, WinPscwAssert};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<f64>::allocate(&world, 8).unwrap();
    /// let exposure_group = world.group().unwrap().include(&[0]).unwrap();
    ///
    /// // Rank 1 opens an access epoch to rank 0's window
    /// win.start(&exposure_group, WinPscwAssert::default(), |_epoch| {
    ///     // ... RMA operations would go here ...
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub fn start<'env, R>(
        &mut self,
        group: &Group,
        assert: WinPscwAssert,
        f: impl for<'s> FnOnce(&'s ActiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        let access = self.group_access(group)?;
        // SAFETY: `win_handle` is a valid MPI window handle. `group.handle`
        // returns a valid group handle from the C-layer group table.
        // `assert.bits()` is either 0 or a combination of MPI_MODE_* constants.
        let ret = unsafe { ffi::ferrompi_win_start(group.handle, assert.bits(), self.win_handle) };
        Error::check_with_op(ret, "win_start")?;
        self.run_epoch(
            "win_complete",
            // SAFETY: `win_handle` is a valid MPI window handle, and the
            // `MPI_Win_start` above opened the access epoch this call closes.
            |win_handle| unsafe { ffi::ferrompi_win_complete(win_handle) },
            access,
            f,
        )
    }

    /// Wait for the exposure epoch to close (active-target PSCW synchronization).
    ///
    /// Wraps `MPI_Win_wait`. Blocks until all RMA operations issued by the
    /// access group (those that ran [`Win::start`]) against this rank's window
    /// are complete. Closes the exposure epoch opened by [`Win::post`], which
    /// has no handle: this rank issues no RMA call in it. A successful call
    /// lets [`into_buffer`](Win::into_buffer) free the window again.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_wait"), .. }` if MPI reports
    /// an error; the exposure epoch then counts as still open.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinPscwAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// # let g = world.group().unwrap().include(&[1]).unwrap();
    /// win.post(&g, WinPscwAssert::default()).unwrap();
    /// // ... wait for remote access to complete ...
    /// win.wait_exposure().unwrap();
    /// ```
    pub fn wait_exposure(&self) -> Result<()> {
        // SAFETY: `win_handle` is a valid MPI window handle. An active exposure
        // epoch was opened by a prior call to `ferrompi_win_post`.
        let ret = unsafe { ffi::ferrompi_win_wait(self.win_handle) };
        Error::check_with_op(ret, "win_wait")?;
        self.exposure_open.set(false);
        Ok(())
    }

    /// Nonblocking test for exposure epoch completion (active-target PSCW).
    ///
    /// Wraps `MPI_Win_test`. Returns `true` if the exposure epoch started by
    /// [`Win::post`] has completed (i.e., all access-side ranks have closed
    /// their [`Win::start`] epochs), `false` otherwise. Does not block. That
    /// exposure epoch has no handle: this rank issues no RMA call in it. A call
    /// that returns `true` lets [`into_buffer`](Win::into_buffer) free the
    /// window again.
    ///
    /// # Errors
    ///
    /// Returns `Error::Mpi { operation: Some("win_test"), .. }` if MPI reports
    /// an error; the exposure epoch then counts as still open.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, Win, WinPscwAssert};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// # let win = Win::<f64>::allocate(&world, 8).unwrap();
    /// # let g = world.group().unwrap().include(&[1]).unwrap();
    /// win.post(&g, WinPscwAssert::default()).unwrap();
    /// // Polling loop (in practice, do useful work between tests)
    /// while !win.test_exposure().unwrap() {
    ///     // do other work
    /// }
    /// ```
    pub fn test_exposure(&self) -> Result<bool> {
        let mut flag: i32 = 0;
        // SAFETY: `win_handle` is a valid MPI window handle. `flag` is a valid
        // stack-allocated i32; the C shim writes exactly one i32 to it via the
        // `*flag` output parameter.
        let ret = unsafe { ffi::ferrompi_win_test(self.win_handle, &mut flag) };
        Error::check_with_op(ret, "win_test")?;
        let closed = flag != 0;
        if closed {
            self.exposure_open.set(false);
        }
        Ok(closed)
    }

    /// Run a closure in a passive-target epoch on one rank's window.
    ///
    /// Wraps `MPI_Win_lock` and `MPI_Win_unlock`. The call locks `rank`'s
    /// window, which gives one-sided access to that rank's memory without its
    /// active participation, runs `f` with the epoch's [`PassiveEpoch`], and
    /// unlocks the window, whether `f` returned `Ok`, returned `Err` or
    /// panicked. Before it unlocks, it completes every request `f` left pending.
    /// When it returns, the RMA operations `f` issued are complete at this rank
    /// and at `rank` (MPI-4.1 §12.5.3), so the origin buffers they borrowed stay
    /// borrowed until the unlock has returned. `rank` is the epoch's access set:
    /// a transfer to another rank returns an error (see [`PassiveEpoch`]).
    ///
    /// # Arguments
    ///
    /// * `lock_type` - [`LockType::Exclusive`] for read-write access, or
    ///   [`LockType::Shared`] for concurrent read-only access with other
    ///   shared lockers.
    /// * `rank` - The target rank to lock (0-based, within the window's
    ///   communicator).
    /// * `f` - The closure that issues the epoch's RMA operations.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped: the
    ///   epoch is not opened and `f` does not run.
    /// - `Error::Mpi` with `operation: Some("win_lock")` if the lock fails,
    ///   for example when MPI rejects `rank` (`f` does not run).
    /// - `Error::Mpi` with `operation: Some("lock")` if a request `f` left
    ///   pending fails when the epoch completes it; the unlock still runs.
    /// - `Error::Mpi` with `operation: Some("win_unlock")` if the unlock
    ///   fails.
    ///
    /// When `f` returns an error, that error is returned in place of a
    /// request's and the unlock's. MPI ran the unlock, and the crate trusts that
    /// it completed or failed the epoch's operations, as MPICH and Open MPI do;
    /// MPI-4.1 §9.4 (p.460) does not promise it.
    ///
    /// # Panics
    ///
    /// A passive-target epoch is not collective: `MPI_Win_unlock` completes this
    /// rank's operations and needs no matching call at any other rank, though it
    /// can wait until the target makes MPI progress (MPI-4.1 §12.7.3). A panic in
    /// `f` therefore completes the pending requests, unlocks the window and then
    /// continues unwinding, as [`scope`](crate::scope) does; an open fence or
    /// PSCW epoch aborts instead. An error from completing the requests or from
    /// that unlock is dropped.
    ///
    /// # Aborts
    ///
    /// An `MPI_Win_unlock` refused before MPI ran it aborts the process, since
    /// the epoch's origin buffers are still in MPI's hands; so does a wait for
    /// the pending requests that completes none of them. In a debug build at
    /// [`ThreadLevel::Serialized`](crate::ThreadLevel::Serialized), an overlap
    /// with another thread's MPI call is the one way to reach either.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{LockType, Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// win.lock(LockType::Shared, 0, |e| {
    ///     e.flush(0)?;
    ///     e.flush_local(0)?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub fn lock<'env, R>(
        &mut self,
        lock_type: LockType,
        rank: i32,
        f: impl for<'s> FnOnce(&'s PassiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        let lt = match lock_type {
            LockType::Exclusive => ffi::FERROMPI_LOCK_EXCLUSIVE,
            LockType::Shared => ffi::FERROMPI_LOCK_SHARED,
        };
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // `ferrompi_win_create` or `ferrompi_win_allocate`. `lt` is one of
        // the two valid MPI lock-type constants obtained from the C layer.
        // `rank` is caller-supplied; MPI validates it against the window's
        // communicator.
        let ret = unsafe { ffi::ferrompi_win_lock(lt, rank, 0, self.win_handle) };
        Error::check_with_op(ret, "win_lock")?;
        self.run_passive_epoch(
            "lock",
            "win_unlock",
            // SAFETY: `win_handle` is a valid MPI window handle, and the
            // `MPI_Win_lock` above opened the epoch on `rank` this call ends.
            |win_handle| unsafe { ffi::ferrompi_win_unlock(rank, win_handle) },
            Access::Rank(rank),
            f,
        )
    }

    /// Run a closure in a passive-target epoch on every rank's window.
    ///
    /// Wraps `MPI_Win_lock_all` and `MPI_Win_unlock_all`. The call acquires
    /// shared locks on all ranks in the window's communicator, runs `f` with
    /// the epoch's [`PassiveEpoch`], and releases them, whether `f` returned
    /// `Ok`, returned `Err` or panicked. Before it unlocks, it completes every
    /// request `f` left pending. It gives concurrent access to every rank's
    /// window memory in one epoch, without a `lock` and an unlock per rank, and
    /// every rank is in the epoch's access set. When it returns, the RMA
    /// operations `f` issued are complete at this rank and at their targets
    /// (MPI-4.1 §12.5.3), so the origin buffers they borrowed stay borrowed
    /// until the unlock has returned.
    ///
    /// # Arguments
    ///
    /// * `f` - The closure that issues the epoch's RMA operations.
    ///
    /// # Errors
    ///
    /// - `Error::Finalized` once [`Mpi`](crate::Mpi) is being dropped: the
    ///   epoch is not opened and `f` does not run.
    /// - `Error::Mpi` with `operation: Some("win_lock_all")` if the lock
    ///   fails (`f` does not run).
    /// - `Error::Mpi` with `operation: Some("lock_all")` if a request `f` left
    ///   pending fails when the epoch completes it; the unlock still runs.
    /// - `Error::Mpi` with `operation: Some("win_unlock_all")` if the unlock
    ///   fails.
    ///
    /// When `f` returns an error, that error is returned in place of a
    /// request's and the unlock's, as for [`Win::lock`].
    ///
    /// # Panics
    ///
    /// A panic in `f` unlocks the window and then continues unwinding, as for
    /// [`Win::lock`].
    ///
    /// # Aborts
    ///
    /// An `MPI_Win_unlock_all` refused before MPI ran it, and a wait for the
    /// pending requests that completes none of them, abort the process, as for
    /// [`Win::lock`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, Win};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let mut win = Win::<f64>::allocate(&world, 8).unwrap();
    ///
    /// win.lock_all(|e| {
    ///     e.flush_all()?;
    ///     e.flush_local_all()?;
    ///     e.sync()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    pub fn lock_all<'env, R>(
        &mut self,
        f: impl for<'s> FnOnce(&'s PassiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        // SAFETY: `win_handle` is a valid MPI window handle. Passing `0` for
        // the assert value is always valid (no optimization hints).
        let ret = unsafe { ffi::ferrompi_win_lock_all(0, self.win_handle) };
        Error::check_with_op(ret, "win_lock_all")?;
        self.run_passive_epoch(
            "lock_all",
            "win_unlock_all",
            // SAFETY: `win_handle` is a valid MPI window handle, and the
            // `MPI_Win_lock_all` above opened the epoch this call ends.
            |win_handle| unsafe { ffi::ferrompi_win_unlock_all(win_handle) },
            Access::All,
            f,
        )
    }

    /// Runs `f` in the epoch the caller just opened, with `access` as its access
    /// set, then completes the epoch's pending requests (`wait_op` labels their
    /// errors) and makes the closing call `close`, whatever `f` did. A closing
    /// call refused before MPI ran it aborts. Otherwise a panic in `f` resumes
    /// after the closing call, and `f`'s error is returned before the requests'
    /// and the closing call's.
    fn run_passive_epoch<'env, R>(
        &mut self,
        wait_op: &'static str,
        close_op: &'static str,
        close: impl FnOnce(i32) -> c_int,
        access: Access,
        f: impl for<'s> FnOnce(&'s PassiveEpoch<'s, 'env, T>) -> Result<R>,
    ) -> Result<R> {
        let epoch = PassiveEpoch {
            win: &*self,
            access,
            registry: Registry::new(),
            scope: PhantomData,
            env: PhantomData,
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| f(&epoch)));
        let completed = epoch.registry.complete_all(wait_op);
        let ret = close(self.win_handle);
        match outcome {
            Err(payload) => {
                let _ = closing_result(ret, close_op, Ok(()));
                resume_unwind(payload)
            }
            Ok(result) => closing_result(
                ret,
                close_op,
                result.and_then(|value| completed.map(|()| value)),
            ),
        }
    }

    /// The access set of a `start` epoch to `group`: its members' ranks in this
    /// window's communicator. A member outside that communicator makes the
    /// group erroneous for `MPI_Win_start` (MPI-4.1 §12.5.2), so it is refused.
    fn group_access(&self, group: &Group) -> Result<Access> {
        let mut win_group = -1;
        // SAFETY: `win_handle` is a valid MPI window handle, and `win_group` is
        // a valid out-parameter.
        let ret = unsafe { ffi::ferrompi_win_get_group(self.win_handle, &mut win_group) };
        Error::check_with_op(ret, "win_get_group")?;
        // The window's group owns the slot the call allocated, and frees it.
        let win_group = Group { handle: win_group };
        let members: Vec<i32> = (0..group.size()?).collect();
        let mut access = vec![false; self.words.len()];
        for rank in group.translate_ranks(&members, &win_group)? {
            let Some(rank) = rank else {
                return Err(Error::InvalidArgument {
                    arg: "group",
                    reason: "member outside the window's communicator",
                });
            };
            access[rank as usize] = true;
        }
        Ok(Access::Ranks(access.into()))
    }

    /// Forwards to [`check_rma_target`] with this window's exchanged words
    /// and `size_of::<T>()`.
    fn check_target(
        &self,
        access: &Access,
        target_rank: i32,
        target_disp: i64,
        count: usize,
        buffer_lens: &[(&'static str, usize)],
    ) -> Result<()> {
        check_rma_target(
            &self.words,
            access,
            std::mem::size_of::<T>(),
            target_rank,
            target_disp,
            count,
            buffer_lens,
        )
    }

    /// One-sided write: validates the target against `access` and the window,
    /// and calls `MPI_Put` with a count of `origin.len()`.
    ///
    /// The transfer helpers (`rma_put`, `rma_get`, `rma_accumulate`,
    /// `rma_get_accumulate`, `rma_fetch_and_op`, `rma_compare_and_swap`,
    /// `rma_rput`, `rma_rget` and `rma_raccumulate`) hand MPI a pointer into a
    /// caller buffer that MPI keeps using after they return, until the epoch's
    /// closing call (the closing fence, `MPI_Win_complete` or the unlock)
    /// returns; for the request-returning ones, until the request completes,
    /// which the passive epoch's registry guarantees before the unlock. Only the
    /// epoch handles call them, each with its own access set, and each handle
    /// method takes its buffers as `&'s [T]`, `&'s mut [T]`, `&'s T` or
    /// `&'s mut T`, where `'s` is the handle's invariant lifetime. `'s` ends
    /// only after the closing call, so the buffer cannot be dropped, moved,
    /// written, or for a buffer MPI writes read, before MPI is done with it, and
    /// a handle exists only while its epoch is open.
    ///
    /// # Safety
    ///
    /// `access` is the access set of an epoch that is open on this window, and
    /// `origin` must stay valid and unmodified until that epoch's closing call
    /// returns. MPI requires an access epoch to `target_rank`; `check_target`
    /// refuses a `target_rank` outside `access`, so one exists for every call
    /// that reaches MPI.
    unsafe fn rma_put(
        &self,
        access: &Access,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<()> {
        let (p, n, dt) = buf(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        // SAFETY: the caller upholds this function's contract (`'s`, see above).
        // `target_rank` was checked above against the epoch's access set, so an
        // access epoch to it is open, and `target_disp` against the target
        // rank's exposed window for `n` elements, so MPI accesses only memory
        // the target exposed.
        let ret = unsafe { ffi::ferrompi_put(p, n, dt, target_rank, target_disp, self.win_handle) };
        Error::check_with_op(ret, "put")
    }

    /// Request-returning write: as [`rma_put`](Self::rma_put) with `MPI_Rput`.
    /// Returns the request's handle.
    ///
    /// # Safety
    ///
    /// As [`rma_put`](Self::rma_put), and the caller must register the returned
    /// handle with the registry that completes it before the epoch's unlock:
    /// until then MPI may still read `origin`.
    unsafe fn rma_rput(
        &self,
        access: &Access,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<i64> {
        let (p, n, dt) = buf(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        let mut request_handle: i64 = 0;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, so
        // MPI accesses only memory the target exposed.
        let ret = unsafe {
            ffi::ferrompi_rput(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "rput")?;
        Ok(request_handle)
    }

    /// One-sided read: validates the target against `access` and the window,
    /// and calls `MPI_Get` with a count of `origin.len()`.
    ///
    /// # Safety
    ///
    /// As [`rma_put`](Self::rma_put), and `origin` must also stay unread until
    /// the epoch's closing call returns.
    unsafe fn rma_get(
        &self,
        access: &Access,
        origin: &mut [T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<()> {
        let (p, n, dt) = buf_mut(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, so
        // MPI accesses only memory the target exposed.
        let ret = unsafe { ffi::ferrompi_get(p, n, dt, target_rank, target_disp, self.win_handle) };
        Error::check_with_op(ret, "get")
    }

    /// Request-returning read: as [`rma_get`](Self::rma_get) with `MPI_Rget`.
    /// Returns the request's handle.
    ///
    /// # Safety
    ///
    /// As [`rma_rput`](Self::rma_rput), and `origin` must also stay unread until
    /// the epoch's unlock returns.
    unsafe fn rma_rget(
        &self,
        access: &Access,
        origin: &mut [T],
        target_rank: i32,
        target_disp: i64,
    ) -> Result<i64> {
        let (p, n, dt) = buf_mut(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        let mut request_handle: i64 = 0;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, so
        // MPI accesses only memory the target exposed.
        let ret = unsafe {
            ffi::ferrompi_rget(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "rget")?;
        Ok(request_handle)
    }

    /// One-sided reduce-accumulate: validates the op, then the target against
    /// `access` and the window, and calls `MPI_Accumulate` with a count of
    /// `origin.len()`.
    ///
    /// # Safety
    ///
    /// As [`rma_put`](Self::rma_put).
    unsafe fn rma_accumulate(
        &self,
        access: &Access,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<AccumulateOp>,
    ) -> Result<()> {
        let op = op.into().code(T::TAG)?;
        let (p, n, dt) = buf(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, so
        // MPI accesses only memory the target exposed. `op` is a predefined op
        // code that the shim maps to an `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_accumulate(p, n, dt, target_rank, target_disp, op, self.win_handle)
        };
        Error::check_with_op(ret, "accumulate")
    }

    /// Request-returning reduce-accumulate: as
    /// [`rma_accumulate`](Self::rma_accumulate) with `MPI_Raccumulate`. Returns
    /// the request's handle.
    ///
    /// # Safety
    ///
    /// As [`rma_rput`](Self::rma_rput).
    unsafe fn rma_raccumulate(
        &self,
        access: &Access,
        origin: &[T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<AccumulateOp>,
    ) -> Result<i64> {
        let op = op.into().code(T::TAG)?;
        let (p, n, dt) = buf(origin);
        self.check_target(access, target_rank, target_disp, n as usize, &[])?;
        let mut request_handle: i64 = 0;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, so
        // MPI accesses only memory the target exposed. `op` is a predefined op
        // code that the shim maps to an `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_raccumulate(
                p,
                n,
                dt,
                target_rank,
                target_disp,
                op,
                self.win_handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "raccumulate")?;
        Ok(request_handle)
    }

    /// Atomic read-modify-write: validates the op, the target against `access`
    /// and the window, and `result`'s length, and calls `MPI_Get_accumulate` with
    /// a count of `origin.len()`.
    ///
    /// # Safety
    ///
    /// As [`rma_put`](Self::rma_put), and `result` must also stay unread until
    /// the epoch's closing call returns. `origin` and `result` cannot alias:
    /// they are a `&[T]` and a `&mut [T]`.
    unsafe fn rma_get_accumulate(
        &self,
        access: &Access,
        origin: &[T],
        result: &mut [T],
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        let op = op.into().code(T::TAG)?;
        let (o_ptr, n, dt) = buf(origin);
        let (r_ptr, r_n, _) = buf_mut(result);
        self.check_target(
            access,
            target_rank,
            target_disp,
            n as usize,
            &[("result", r_n as usize)],
        )?;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `target_rank` and `target_disp` were checked above against the epoch's
        // access set and the target rank's exposed window for `n` elements, and
        // `result` holds `n` elements, so MPI accesses only memory the target
        // exposed and writes only `result`. `op` is a predefined op code that the
        // shim maps to an `MPI_Op`.
        let ret = unsafe {
            ffi::ferrompi_get_accumulate(
                o_ptr,
                r_ptr,
                n,
                dt,
                target_rank,
                target_disp,
                op,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "get_accumulate")
    }

    /// Single-element atomic fetch-and-update: validates the op and the target
    /// against `access` and the window, and calls `MPI_Fetch_and_op`.
    ///
    /// # Safety
    ///
    /// As [`rma_put`](Self::rma_put), and `result` must also stay unread and
    /// unwritten until the epoch's closing call returns, when MPI has written
    /// it (MPI-4.1 §12.3 and §12.5). `origin` and `result` cannot alias: they
    /// are a `&T` and a `&mut T`.
    unsafe fn rma_fetch_and_op(
        &self,
        access: &Access,
        origin: &T,
        result: &mut T,
        target_rank: i32,
        target_disp: i64,
        op: impl Into<FetchOp>,
    ) -> Result<()> {
        let op = op.into().code(T::TAG)?;
        self.check_target(access, target_rank, target_disp, 1, &[])?;
        // SAFETY: the caller upholds this function's contract (see `rma_put`).
        // `T::TAG` matches `T`'s memory layout per the `MpiDatatype` invariant, and
        // MPI reads one `T` from `origin` and writes one `T` to `result`, both
        // pointers taken from the references themselves (`result` from the
        // `&mut T`, never from a shared borrow), and only MPI writes `result`
        // while it is borrowed. `op` is a predefined op code that the shim maps
        // to an `MPI_Op`. `target_rank` was checked above against the epoch's
        // access set, so an access epoch to it is open, and `target_disp` (with
        // a count of 1) against the target rank's exposed window, so MPI
        // accesses only memory the target exposed.
        let ret = unsafe {
            ffi::ferrompi_fetch_and_op(
                (origin as *const T).cast(),
                (result as *mut T).cast(),
                T::TAG as i32,
                target_rank,
                target_disp,
                op,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "fetch_and_op")
    }

    /// Single-element atomic compare-and-swap: validates the target against
    /// `access` and the window, and calls `MPI_Compare_and_swap`.
    ///
    /// # Safety
    ///
    /// As [`rma_fetch_and_op`](Self::rma_fetch_and_op), and `compare` must also
    /// stay valid and unmodified until the epoch's closing call returns.
    unsafe fn rma_compare_and_swap(
        &self,
        access: &Access,
        origin: &T,
        compare: &T,
        result: &mut T,
        target_rank: i32,
        target_disp: i64,
    ) -> Result<()>
    where
        T: crate::AtomicMpiDatatype,
    {
        self.check_target(access, target_rank, target_disp, 1, &[])?;
        // SAFETY: as in `rma_fetch_and_op`: the caller's contract, `T::TAG`, the
        // pointers taken from the references (`compare` read like `origin`), and
        // `check_target` above for a count of 1. `AtomicMpiDatatype` restricts
        // `T` to the types MPI-4.1 §12.3.4 allows in compare-and-swap (integer,
        // logical and byte).
        let ret = unsafe {
            ffi::ferrompi_compare_and_swap(
                (origin as *const T).cast(),
                (compare as *const T).cast(),
                (result as *mut T).cast(),
                T::TAG as i32,
                target_rank,
                target_disp,
                self.win_handle,
            )
        };
        Error::check_with_op(ret, "compare_and_swap")
    }
}

/// Frees the window with `MPI_Win_free`, which is collective; the type's
/// `# Dropping` section lists the four outcomes.
impl<T: MpiDatatype> Drop for Win<T> {
    fn drop(&mut self) {
        if self.win_handle == WIN_FREED {
            return;
        }
        // A peer may still write into the window until the exposure epoch
        // closes (MPI-4.1 §12.2.5), so the window is neither freed nor
        // uncounted, and the buffer outlives the process.
        if self.exposure_open.get() {
            std::mem::forget(self.buffer.take());
            return;
        }
        let Some(_call) = rt::drop_guard("Win") else {
            std::mem::forget(self.buffer.take());
            return;
        };
        // SAFETY: `win_handle` is a valid MPI window handle allocated by
        // ferrompi_win_create or ferrompi_win_allocate. It has not been freed
        // yet because Drop is only called once and `into_buffer` sets
        // `WIN_FREED` after its free. MPI_Win_free leaves the user buffer
        // alone for Win::create and frees the MPI-allocated buffer for
        // Win::allocate per the MPI standard — the C layer handles this
        // distinction correctly.
        let ret = unsafe { ffi::ferrompi_win_free(self.win_handle) };
        if ret == 0 {
            // Release: pairs with `live_windows`'s Acquire; this decrement
            // follows the successful free. The buffer drops after this body,
            // once MPI is done with it.
            LIVE_WINDOWS.fetch_sub(1, Ordering::Release);
        } else {
            std::mem::forget(self.buffer.take());
        }
    }
}

/// RAII guard for a single-rank window lock.
///
/// Created by [`SharedWindow::lock()`]. When dropped, the lock on the
/// specified rank is automatically released via `MPI_Win_unlock`.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, SharedWindow, LockType};
/// # let mpi = Mpi::init().unwrap();
/// # let node = mpi.world().split_shared().unwrap();
/// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
/// {
///     let guard = win.lock(LockType::Shared, 0).unwrap();
///     let remote = win.remote_slice(0).unwrap();
///     // Ensure all operations are completed
///     guard.flush().unwrap();
/// } // Lock released here
/// ```
pub struct LockGuard<'a, T: MpiDatatype> {
    window: &'a SharedWindow<T>,
    rank: i32,
}

impl<T: MpiDatatype> LockGuard<'_, T> {
    /// Flush pending RMA operations to the locked rank.
    ///
    /// Ensures that all preceding RMA operations (put, get, accumulate)
    /// issued to the locked rank are completed at the target.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush operation fails.
    pub fn flush(&self) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and the rank was locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush(self.rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for LockGuard<'_, T> {
    fn drop(&mut self) {
        let Some(_call) = rt::drop_guard("LockGuard") else {
            return;
        };
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and the rank was locked in the constructor. This unlock matches
        // the lock call that created this guard. It calls the raw declaration
        // because `drop_guard` above is this call's lifecycle check.
        unsafe { ffi::raw::ferrompi_win_unlock(self.rank, self.window.win_handle) };
    }
}

/// RAII guard for a lock-all window lock.
///
/// Created by [`SharedWindow::lock_all()`]. When dropped, all locks
/// are automatically released via `MPI_Win_unlock_all`.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, SharedWindow};
/// # let mpi = Mpi::init().unwrap();
/// # let node = mpi.world().split_shared().unwrap();
/// # let win = SharedWindow::<f64>::allocate(&node, 100).unwrap();
/// {
///     let guard = win.lock_all().unwrap();
///     // Access shared memory from any rank
///     guard.flush_all().unwrap();
/// } // All locks released here
/// ```
pub struct LockAllGuard<'a, T: MpiDatatype> {
    window: &'a SharedWindow<T>,
}

impl<T: MpiDatatype> LockAllGuard<'_, T> {
    /// Flush all pending RMA operations to all ranks.
    ///
    /// Ensures that all preceding RMA operations issued to any rank
    /// in the window are completed at their respective targets.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush_all operation fails.
    pub fn flush_all(&self) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush_all(self.window.win_handle) };
        Error::check_with_op(ret, "win_flush_all")
    }

    /// Flush pending RMA operations to a specific rank.
    ///
    /// Ensures that all preceding RMA operations issued to the specified
    /// rank are completed at the target.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank to flush (0-based)
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI flush operation fails.
    pub fn flush(&self, rank: i32) -> Result<()> {
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor.
        let ret = unsafe { ffi::ferrompi_win_flush(rank, self.window.win_handle) };
        Error::check_with_op(ret, "win_flush")
    }
}

impl<T: MpiDatatype> Drop for LockAllGuard<'_, T> {
    fn drop(&mut self) {
        let Some(_call) = rt::drop_guard("LockAllGuard") else {
            return;
        };
        // SAFETY: The window handle is valid (borrowed from SharedWindow)
        // and all ranks were locked in the constructor. This unlock_all
        // matches the lock_all call that created this guard. It calls the raw
        // declaration because `drop_guard` above is this call's lifecycle check.
        unsafe { ffi::raw::ferrompi_win_unlock_all(self.window.win_handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        check_fence_asserts, check_rma_target, window_word, Access, Error, Win, WinFenceAssert,
        WinPscwAssert, WINDOW_WORD_LEN_MASK, WINDOW_WORD_REJECT, WIN_FREED,
    };
    use crate::group::Group;
    use crate::persistent::vec_buf_mut;
    use crate::{MpiDatatype, ReduceOp};
    use std::cell::Cell;
    use std::ptr::NonNull;

    // -------------------------------------------------------------------------
    // window_word unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn window_word_packs_disp_unit_and_length() {
        assert_eq!(window_word::<u64>(4), (8u64 << 56) | 32);
        assert_eq!(window_word::<u8>(0), 1u64 << 56);

        let max_len = WINDOW_WORD_LEN_MASK;
        assert_eq!(window_word::<u8>(max_len as usize), (1u64 << 56) | max_len);

        assert_eq!(window_word::<u8>(1usize << 56), WINDOW_WORD_REJECT);
        assert_eq!(window_word::<u64>(usize::MAX / 8 + 1), WINDOW_WORD_REJECT);
    }

    // -------------------------------------------------------------------------
    // check_rma_target unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn check_rma_target_boundaries() {
        let words = [window_word::<u64>(4), window_word::<u8>(3)];
        let run = |elem_size: usize,
                   rank: i32,
                   disp: i64,
                   count: usize,
                   lens: &[(&'static str, usize)]| {
            check_rma_target(&words, &Access::All, elem_size, rank, disp, count, lens)
        };

        assert!(run(8, 0, 0, 4, &[("origin", 4)]).is_ok());
        assert!(run(8, 0, 3, 1, &[("origin", 1)]).is_ok());
        assert!(run(8, 0, 4, 0, &[("origin", 0)]).is_ok());
        assert!(run(1, 1, 0, 3, &[("origin", 3), ("result", 3)]).is_ok());

        let past_window = "access extends past the target's window";
        for (result, arg, reason) in [
            (
                run(8, 0, 2, 4, &[("origin", 4)]),
                "target_disp",
                past_window,
            ),
            (run(8, 0, 4, 1, &[]), "target_disp", past_window),
            (
                run(8, 0, i64::MAX, 1, &[("origin", 1)]),
                "target_disp",
                past_window,
            ),
            (
                run(8, 0, -1, 1, &[("origin", 1)]),
                "target_disp",
                "negative displacement",
            ),
            (
                run(8, 2, 0, 1, &[("origin", 1)]),
                "target_rank",
                "rank outside the window's communicator",
            ),
            (
                run(8, -1, 0, 1, &[("origin", 1)]),
                "target_rank",
                "rank outside the window's communicator",
            ),
        ] {
            assert!(
                matches!(
                    &result,
                    Err(Error::InvalidArgument { arg: a, reason: r }) if *a == arg && *r == reason
                ),
                "expected InvalidArgument {{ arg: {arg}, reason: {reason} }}, got {result:?}"
            );
        }

        for (result, arg, required, actual) in [
            (run(8, 0, 0, 4, &[("origin", 3)]), "origin", 4, 3),
            (
                run(1, 1, 0, 3, &[("origin", 3), ("result", 2)]),
                "result",
                3,
                2,
            ),
        ] {
            assert!(
                matches!(
                    &result,
                    Err(Error::BufferSize { arg: a, required: q, actual: c })
                        if *a == arg && *q == required && *c == actual
                ),
                "expected BufferSize {{ arg: {arg}, required: {required}, actual: {actual} }}, got {result:?}"
            );
        }
    }

    #[test]
    fn check_rma_target_rejects_a_rank_outside_the_access_set() {
        let words = [window_word::<u64>(4), window_word::<u64>(4)];
        let outside = "rank outside the epoch's access set";
        let bad_rank = "rank outside the window's communicator";
        let run = |access: &Access, rank: i32| check_rma_target(&words, access, 8, rank, 0, 1, &[]);

        let locked = Access::Rank(1);
        let group = Access::Ranks(vec![false, true].into());
        let empty = Access::Ranks(Box::new([]));
        for (access, rank, reason) in [
            (&locked, 0, Some(outside)),
            (&locked, 1, None),
            (&group, 0, Some(outside)),
            (&group, 1, None),
            (&empty, 1, Some(outside)),
            (&Access::All, 0, None),
            (&Access::All, 1, None),
            // A rank outside the communicator is a bad rank before it is an
            // access-set fault, whatever the set holds.
            (&Access::Rank(2), 2, Some(bad_rank)),
            (&Access::Rank(-1), -1, Some(bad_rank)),
        ] {
            let result = run(access, rank);
            match reason {
                None => assert!(result.is_ok(), "rank {rank}: {result:?}"),
                Some(reason) => assert!(
                    matches!(
                        &result,
                        Err(Error::InvalidArgument { arg: "target_rank", reason: r }) if *r == reason
                    ),
                    "rank {rank}: expected InvalidArgument {{ arg: target_rank, reason: {reason} }}, got {result:?}"
                ),
            }
        }
    }

    // -------------------------------------------------------------------------
    // Win<T> unit tests — exercise the type without an MPI runtime
    // -------------------------------------------------------------------------

    /// Stands in for a peer's RMA write into the window between the slices
    /// the window hands out: the pointer `vec_buf_mut` returned, which
    /// `Win::create` hands to MPI and stores as `local_ptr`, must stay usable
    /// across them. The window carries the sentinel handle, so its drop makes
    /// no MPI call. Run under Miri, with and without `-Zmiri-tree-borrows`, to
    /// check its provenance.
    #[test]
    fn recorded_window_pointer_survives_local_slices() {
        let mut v = vec![0u64; 4];
        let (p, n, _) = vec_buf_mut(&mut v);
        let p = p.cast::<u64>();
        assert_eq!(n, 4);
        let mut win = Win {
            win_handle: WIN_FREED,
            // SAFETY: `Vec::as_mut_ptr` is never null.
            local_ptr: unsafe { NonNull::new_unchecked(p) },
            local_len: 4,
            words: Box::new([]),
            buffer: Some(v),
            exposure_open: Cell::new(false),
        };
        for round in 1..=3u64 {
            win.local_slice_mut().fill(round);
            for i in 0..4 {
                // SAFETY: `p` addresses the live allocation of four `u64`s that
                // `vec_buf_mut` derived it from, and no slice is live while the
                // write stands in for a peer's.
                unsafe { p.add(i).write(round * 10 + i as u64) };
            }
            let expected = [round * 10, round * 10 + 1, round * 10 + 2, round * 10 + 3];
            assert_eq!(win.local_slice(), expected);
        }
        let buffer = win.buffer.take().expect("a create window owns its buffer");
        assert_eq!(buffer, [30, 31, 32, 33]);
    }

    /// `into_buffer` refuses while an exposure epoch is open, before any MPI
    /// call (none is initialized here), and the window it then drops makes none
    /// either. The window's empty buffer owns no allocation, so the leak is
    /// empty.
    #[test]
    fn into_buffer_refuses_while_an_exposure_epoch_is_open() {
        let win = Win::<u64> {
            win_handle: 0,
            local_ptr: NonNull::dangling(),
            local_len: 0,
            words: Box::new([]),
            buffer: Some(Vec::new()),
            exposure_open: Cell::new(true),
        };
        assert!(matches!(win.into_buffer(), Err(Error::InvalidState { .. })));
    }

    /// A `post` that was not refused before MPI could run it counts as opened,
    /// even when it returns an error (MPI-4.1 §9.4). The sentinel window makes
    /// the C shim return `MPI_ERR_WIN` without calling MPI, which stands in for
    /// a call that failed.
    #[test]
    #[cfg_attr(miri, ignore = "calls the C layer")]
    fn post_that_returned_an_mpi_error_counts_as_open() {
        let win = Win::<u64> {
            win_handle: WIN_FREED,
            local_ptr: NonNull::dangling(),
            local_len: 0,
            words: Box::new([]),
            buffer: Some(Vec::new()),
            exposure_open: Cell::new(false),
        };
        assert!(win
            .post(&Group { handle: -1 }, WinPscwAssert::none())
            .is_err());
        assert!(win.exposure_open.get());
    }

    /// The fetch helpers validate the op before the target and refuse before any
    /// MPI call, so a window with the sentinel handle stands in for a real one,
    /// and a refused call writes nothing into the caller's `result`.
    #[test]
    fn fetch_helpers_refuse_op_then_target_and_write_nothing() {
        fn window<T: MpiDatatype>() -> Win<T> {
            Win {
                win_handle: WIN_FREED,
                local_ptr: NonNull::dangling(),
                local_len: 0,
                words: Box::new([window_word::<T>(1), window_word::<T>(1)]),
                buffer: None,
                exposure_open: Cell::new(false),
            }
        }

        let float_win = window::<f64>();
        let mut float_result = 7.5f64;
        // SAFETY: every call below is refused before any MPI call, so MPI never
        // sees the pointers and no epoch is needed.
        let (bad_op, bad_rank, outside_access) = unsafe {
            (
                float_win.rma_fetch_and_op(
                    &Access::All,
                    &1.0,
                    &mut float_result,
                    2,
                    0,
                    ReduceOp::BitwiseOr,
                ),
                float_win.rma_fetch_and_op(
                    &Access::All,
                    &1.0,
                    &mut float_result,
                    2,
                    0,
                    ReduceOp::Sum,
                ),
                float_win.rma_fetch_and_op(
                    &Access::Rank(1),
                    &1.0,
                    &mut float_result,
                    0,
                    0,
                    ReduceOp::Sum,
                ),
            )
        };
        assert!(
            matches!(bad_op, Err(Error::InvalidArgument { arg: "op", .. })),
            "a bad op is refused before a bad rank: {bad_op:?}"
        );
        assert!(
            matches!(
                bad_rank,
                Err(Error::InvalidArgument {
                    arg: "target_rank",
                    ..
                })
            ),
            "{bad_rank:?}"
        );
        assert!(
            matches!(
                outside_access,
                Err(Error::InvalidArgument {
                    arg: "target_rank",
                    reason: "rank outside the epoch's access set"
                })
            ),
            "{outside_access:?}"
        );

        let int_win = window::<i32>();
        let mut int_result = 7i32;
        // SAFETY: refused before any MPI call, as above.
        let past_window =
            unsafe { int_win.rma_compare_and_swap(&Access::All, &1, &2, &mut int_result, 1, 1) };
        assert!(
            matches!(
                past_window,
                Err(Error::InvalidArgument {
                    arg: "target_disp",
                    ..
                })
            ),
            "{past_window:?}"
        );

        assert_eq!(float_result.to_bits(), 7.5f64.to_bits());
        assert_eq!(int_result, 7);
    }

    // -------------------------------------------------------------------------
    // WinFenceAssert unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn win_fence_assert_or_combines_bits() {
        // Use placeholder constants for the test; the real values come from MPI.
        let a = WinFenceAssert::from_bits_for_test(1);
        let b = WinFenceAssert::from_bits_for_test(2);
        assert_eq!((a | b).bits(), 3);
    }

    #[test]
    fn win_fence_assert_or_assign_combines_bits() {
        let mut a = WinFenceAssert::from_bits_for_test(1);
        a |= WinFenceAssert::from_bits_for_test(4);
        assert_eq!(a.bits(), 5);
    }

    #[test]
    #[cfg_attr(miri, ignore = "calls the C layer")]
    fn check_fence_asserts_rejects_no_succeed_open_and_no_precede_close() {
        let none = WinFenceAssert::none();
        let no_succeed = WinFenceAssert::no_succeed();
        let no_precede = WinFenceAssert::no_precede();
        let fault = |open, close| match check_fence_asserts(open, close) {
            Err(Error::InvalidArgument { arg, reason }) => Some((arg, reason)),
            result => {
                assert!(result.is_ok(), "unexpected {result:?}");
                None
            }
        };

        assert_eq!(
            fault(no_succeed, none),
            Some((
                "open",
                "no_succeed on an opening fence leaves the epoch's RMA calls outside it"
            ))
        );
        assert_eq!(
            fault(none, no_precede),
            Some((
                "close",
                "no_precede on a closing fence leaves the epoch's RMA calls uncompleted"
            ))
        );
        // The bit is found among others, and two faults report the opening fence's.
        assert_eq!(
            fault(no_succeed | WinFenceAssert::no_store(), none).map(|f| f.0),
            Some("open")
        );
        assert_eq!(
            fault(none, no_precede | WinFenceAssert::no_put()).map(|f| f.0),
            Some("close")
        );
        assert_eq!(fault(no_succeed, no_precede).map(|f| f.0), Some("open"));

        assert_eq!(fault(no_precede, no_succeed), None);
        assert_eq!(fault(none, none), None);
        assert_eq!(
            fault(WinFenceAssert::no_store(), WinFenceAssert::no_put()),
            None
        );
    }

    // -------------------------------------------------------------------------
    // WinPscwAssert unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn win_pscw_assert_or_combines_bits() {
        // Use placeholder constants for the test; the real values come from MPI.
        let a = WinPscwAssert::from_bits_for_test(1);
        let b = WinPscwAssert::from_bits_for_test(2);
        assert_eq!((a | b).bits(), 3);
    }

    #[test]
    fn win_pscw_assert_or_assign_combines_bits() {
        let mut a = WinPscwAssert::from_bits_for_test(1);
        a |= WinPscwAssert::from_bits_for_test(4);
        assert_eq!(a.bits(), 5);
    }
}
