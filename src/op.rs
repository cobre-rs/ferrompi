//! User-defined MPI reduction operations via [`MPI_Op_create`].
//!
//! This module provides [`UserOp<T>`], a safe wrapper around a user-supplied
//! Rust closure that MPI invokes during reduction collectives
//! (`MPI_Reduce`, `MPI_Allreduce`, `MPI_Scan`, etc.).
//!
//! ## Safety model
//!
//! The closure must satisfy `Send + Sync + 'static`:
//!
//! * **`Send`** — the closure is moved into a static slot table accessible from
//!   any thread (MPI may call it from an internal thread pool).
//! * **`Sync`** — under `MPI_THREAD_MULTIPLE`, the same op may be invoked
//!   concurrently from several threads.
//! * **`'static`** — the closure is held until `MPI_Op_free` returns, which may
//!   be much later than the call site.
//!
//! ## Panic behaviour
//!
//! **A panic inside the closure aborts the process immediately.**
//!
//! Panicking across a C FFI boundary is undefined behaviour; there is no
//! mechanism for MPI to propagate or observe a Rust panic.  The trampoline
//! wraps every closure call in [`std::panic::catch_unwind`]: on `Err` it calls
//! [`std::process::abort`] before the panic can reach the C frame.  Treat a
//! panic inside a reduction closure as a fatal programming error.
//!
//! ## Slot-table limit
//!
//! The implementation supports at most **16** concurrently live `UserOp`
//! instances per process.  Attempting to create a seventeenth returns
//! [`Error::Mpi`] with class `Other`.
//!
//! ## `compile_fail` doctest — `Send + Sync` bound
//!
//! ```compile_fail
//! use ferrompi::UserOp;
//! let local = std::rc::Rc::new(0i32);
//! let op = UserOp::new(move |_a: &[f64], _b: &mut [f64]| {
//!     let _ = local.clone();
//! });
//! ```

use std::marker::PhantomData;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::datatype::MpiDatatype;
use crate::error::{Error, MpiErrorClass, Result};
use crate::ffi;
use crate::rt;

// ============================================================================
// Slot count — must match MAX_OPS in csrc/ferrompi.c
// ============================================================================
const MAX_OPS: usize = 16;

/// Type alias for the byte-level closure stored in each op slot.
///
/// Using an alias avoids the `clippy::type_complexity` lint at every use site.
type ByteClosure = Box<dyn Fn(&[u8], &mut [u8]) + Send + Sync + 'static>;

/// Per-slot registry of thin pointers to boxed closures.
///
/// Each slot holds `Box::into_raw(Box::new(byte_closure))` — a thin
/// `*mut ByteClosure` — or null.  Protocol:
/// - `UserOp::new_impl` publishes a slot with `Ordering::Release` before
///   calling `MPI_Op_create`.
/// - `rust_user_op_invoke` (the trampoline) loads a slot with
///   `Ordering::Acquire`.
/// - `ferrompi_op_drop_closure` swaps a slot to null only after
///   `MPI_Op_free` returns, so no trampoline call can observe a freed
///   closure.
static REGISTRY: [AtomicPtr<ByteClosure>; MAX_OPS] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; MAX_OPS];

// ============================================================================
// Extern "C" callbacks exposed to the C layer
// ============================================================================

/// Called by each C trampoline `ferrompi_user_op_trampoline_N`.
///
/// The C trampoline passes only its slot number and the raw buffer pointers
/// it received from MPI.  This function loads the slot's closure and invokes
/// it, wrapped in `catch_unwind`.
///
/// # Safety
///
/// * `slot` is in range `0..MAX_OPS` and was published by `UserOp::new_impl`
///   before `MPI_Op_create` was called for it.
/// * `invec` is a valid read-only pointer to `len * byte_size` bytes.
/// * `inoutvec` is a valid read-write pointer to `len * byte_size` bytes.
///
/// Called from C, so the ABI must be exactly `extern "C"`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_user_op_invoke(
    slot: std::ffi::c_int,
    invec: *const c_void,
    inoutvec: *mut c_void,
    len: std::ffi::c_int,
) {
    let ptr = REGISTRY[slot as usize].load(Ordering::Acquire);
    if ptr.is_null() {
        // A null slot here is an invariant violation: the publish-before-
        // create / drop-after-free protocol on REGISTRY guarantees a live
        // closure for every slot MPI can still invoke.  Treat it the same as
        // the panic path below — abort rather than deref a null pointer.
        std::process::abort();
    }
    // SAFETY: ptr is non-null, published by `new_impl` with `Ordering::Release`
    // before `MPI_Op_create`, and this `Ordering::Acquire` load of the same
    // atomic synchronizes-with that store.  `ferrompi_op_drop_closure` nulls
    // the slot only after `MPI_Op_free` returns, so no drop can race with
    // this borrow.
    let closure: &ByteClosure = unsafe { &*ptr };

    // Build byte slices from the raw MPI buffers.
    // len is the number of *elements* (MPI's *len parameter).  The byte-level
    // adapter stored in the registry receives slices whose .len() is the
    // element count — it uses that to reconstruct typed &[T] / &mut [T] slices
    // of the correct length via slice::from_raw_parts.
    //
    // We do NOT multiply by size_of::<T>() here; that knowledge lives entirely
    // inside the adapter closure captured in UserOp::new_impl.  Passing the
    // element count as the u8-slice length avoids any accidental OOB: the
    // adapter must not interpret .len() as a byte count.
    let len_usize = len as usize;
    // SAFETY: Both slices span `len * size_of::<T>()` bytes at the MPI-provided
    // addresses; the adapter casts the pointer and uses len_usize as the element
    // count to reconstruct properly-typed slices.  The adapter must not use
    // .len() as a byte count — it is the MPI element count.
    let invec_bytes: &[u8] = unsafe { std::slice::from_raw_parts(invec.cast::<u8>(), len_usize) };
    // SAFETY: inoutvec spans `len * size_of::<T>()` bytes at the MPI-provided
    // address, aliased with no other live reference for the duration of this
    // call; the adapter casts the pointer and uses len_usize as the element
    // count, not a byte count.
    let inoutvec_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(inoutvec.cast::<u8>(), len_usize) };

    // Wrap the closure call in catch_unwind (ADR-0005 Decision 6).
    // A panic across the FFI boundary is UB; abort on Err.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        closure(invec_bytes, inoutvec_bytes);
    }));
    if result.is_err() {
        // A panic inside a user-defined reduction closure is a fatal
        // programming error.  Abort immediately to prevent silent data
        // corruption in the collective result (ADR-0005 Decision 6).
        std::process::abort();
    }
}

/// Called by `ferrompi_op_free` (in C) after `MPI_Op_free` returns — including
/// when the `ferrompi_finalize` sweep invokes `ferrompi_op_free` on a still-used
/// slot — and by `UserOp::new_impl`'s rollback path when `MPI_Op_create` fails.
///
/// Swaps the slot to null and, if it held a pointer, drops the boxed
/// closure.  Idempotent: a second call on the same slot is a no-op — this is
/// the one drop routine both the C free path and the Rust rollback path
/// share.
///
/// # Safety
///
/// * `slot` must be in range `0..MAX_OPS`.
/// * If `REGISTRY[slot]` is non-null, it must point to a `ByteClosure`
///   produced by `Box::into_raw` in `UserOp::new_impl` that has not yet been
///   dropped, and no trampoline call for this slot may be in flight (i.e.
///   `MPI_Op_create` for this slot never succeeded, or `MPI_Op_free` for it
///   has already returned).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrompi_op_drop_closure(slot: i32) {
    if slot < 0 || slot as usize >= MAX_OPS {
        return;
    }
    let ptr = REGISTRY[slot as usize].swap(std::ptr::null_mut(), Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // SAFETY: ptr was produced by Box::into_raw in new_impl and has not been
    // dropped; the swap above claimed sole ownership by nulling the slot,
    // and the caller guarantees no trampoline call for this slot is in
    // flight.
    drop(unsafe { Box::from_raw(ptr) });
}

// ============================================================================
// UserOp<T>
// ============================================================================

/// A user-defined MPI reduction operation backed by a Rust closure.
///
/// `T` must implement [`MpiDatatype`] — i.e., it must be one of the primitive
/// types recognised by ferrompi (`f32`, `f64`, `i32`, `i64`, `u8`, `u32`,
/// `u64`).
///
/// The closure receives `invec` as a shared slice and `inoutvec` as a mutable
/// slice of the same length.  It must accumulate `invec[i]` into `inoutvec[i]`
/// for each index `i` — the standard MPI semantics for user reduction
/// functions.
///
/// # Thread-safety
///
/// The closure is called from whichever thread MPI uses internally for the
/// reduction.  Under `MPI_THREAD_MULTIPLE` the same op may be invoked
/// concurrently from multiple threads; the closure must be safe for concurrent
/// invocation, enforced by the `Sync` bound.
///
/// # Panic behaviour
///
/// A panic inside the closure **aborts the process**.  See module-level
/// documentation for details.
///
/// # Slot-table limit
///
/// At most 16 `UserOp` instances may be live concurrently per process.
///
/// If a `UserOp` outlives the `Mpi` handle, finalizing MPI frees its op and
/// drops its closure, and dropping the `UserOp` afterwards does nothing.
///
/// # Examples
///
/// ```no_run
/// use ferrompi::{Mpi, UserOp};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// let op: UserOp<f64> = UserOp::new(|invec: &[f64], inoutvec: &mut [f64]| {
///     for (x, y) in invec.iter().zip(inoutvec.iter_mut()) {
///         *y = x.max(*y);
///     }
/// }).unwrap();
///
/// let send = vec![world.rank() as f64 + 1.5_f64];
/// let mut recv = vec![0.0_f64];
/// world.allreduce_with_op(&send, &mut recv, &op).unwrap();
/// ```
pub struct UserOp<T: MpiDatatype> {
    handle: i32,
    _marker: PhantomData<T>,
}

impl<T: MpiDatatype> UserOp<T> {
    /// Create a commutative user-defined reduction op.
    ///
    /// MPI is permitted to reorder operands for optimisation purposes.  Use
    /// this constructor for operations that satisfy `f(a, b) == f(b, a)` —
    /// element-wise sums, maxima, minima, etc.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the op-slot table is full (16 concurrent `UserOp`s)
    /// or if `MPI_Op_create` fails.
    pub fn new<F>(f: F) -> Result<Self>
    where
        F: Fn(&[T], &mut [T]) + Send + Sync + 'static,
    {
        Self::new_impl(f, 1)
    }

    /// Create a non-commutative user-defined reduction op.
    ///
    /// MPI will not reorder operands.  Use this constructor for operations
    /// where order matters — matrix multiplication, string concatenation, etc.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the op-slot table is full or if `MPI_Op_create` fails.
    pub fn new_noncommutative<F>(f: F) -> Result<Self>
    where
        F: Fn(&[T], &mut [T]) + Send + Sync + 'static,
    {
        Self::new_impl(f, 0)
    }

    fn new_impl<F>(f: F, commute: i32) -> Result<Self>
    where
        F: Fn(&[T], &mut [T]) + Send + Sync + 'static,
    {
        // Step 1: allocate a slot in the C-side op table.
        let mut slot: i32 = -1;
        // SAFETY: slot is a local out-parameter written by ferrompi_op_alloc_slot
        // before this function reads it below (guarded by the `ret != 0` check).
        let ret = unsafe { ffi::ferrompi_op_alloc_slot(&mut slot) };
        if ret != 0 {
            return Err(Error::Mpi {
                class: MpiErrorClass::Other,
                code: ret,
                message: "op-slot table is full (MAX_OPS=16 concurrent UserOps)".to_string(),
                operation: Some("op_create"),
            });
        }
        let idx = slot as usize;
        debug_assert!(idx < MAX_OPS, "slot out of range");

        // Step 2: wrap the typed closure in a byte-level adapter.
        //
        // The C trampoline passes byte slices whose .len() field carries the
        // MPI element count (not byte count).  The adapter uses that element
        // count directly to reconstruct typed slices via slice::from_raw_parts.
        //
        // Why byte-level: the Rust callback `rust_user_op_invoke` has a single
        // signature regardless of T; it reconstructs a
        // `dyn Fn(&[u8], &mut [u8])` trait object.  The adapter converts back
        // to `&[T]` / `&mut [T]` via `slice::from_raw_parts`, interpreting
        // .len() as the element count (not a byte count).
        // Wrap in a ByteClosure (byte-level adapter over the typed closure).
        let byte_closure: ByteClosure =
            Box::new(move |invec_bytes: &[u8], inoutvec_bytes: &mut [u8]| {
                // `invec_bytes.len()` and `inoutvec_bytes.len()` are the MPI
                // element count forwarded by rust_user_op_invoke.  The actual
                // byte span is elem_count * size_of::<T>(), which MPI guarantees
                // is valid; we use elem_count here as the slice element count.
                let elem_count = invec_bytes.len();
                // SAFETY:
                //   * invec_bytes.as_ptr() points to a valid MPI-provided buffer
                //     of at least elem_count * size_of::<T>() bytes.
                //   * T: MpiDatatype implies T: Copy with stable layout; MPI
                //     provides properly-aligned buffers for the registered type.
                //   * elem_count comes from MPI's *len — the number of elements
                //     MPI needs reduced.
                //   * .len() is used here as element count, NOT byte count.
                let invec: &[T] = unsafe {
                    std::slice::from_raw_parts(invec_bytes.as_ptr().cast::<T>(), elem_count)
                };
                // SAFETY: inoutvec_bytes.as_mut_ptr() points to a valid MPI-provided
                // buffer of at least elem_count * size_of::<T>() bytes, aliased with
                // no other live reference; T: MpiDatatype implies T: Copy with stable
                // layout, and MPI provides properly-aligned buffers for the
                // registered type. elem_count is MPI's *len, used as element count.
                let inoutvec: &mut [T] = unsafe {
                    std::slice::from_raw_parts_mut(
                        inoutvec_bytes.as_mut_ptr().cast::<T>(),
                        elem_count,
                    )
                };
                f(invec, inoutvec);
            });

        // Step 3: publish the boxed closure to the registry.  A thin
        // `*mut ByteClosure` — no fat-pointer decomposition needed.  This
        // Release store, paired with the trampoline's Acquire load of the
        // same atomic, synchronizes-with that load directly (no external
        // happens-before edge required).
        let ptr: *mut ByteClosure = Box::into_raw(Box::new(byte_closure));
        REGISTRY[idx].store(ptr, Ordering::Release);

        // Step 4: call MPI_Op_create via the C shim.
        let mut handle: i32 = -1;
        // SAFETY: handle is a local out-parameter written by
        // ferrompi_op_create_user before this function reads it below; slot is
        // the value ferrompi_op_alloc_slot returned above.
        let ret = unsafe { ffi::ferrompi_op_create_user(slot, commute, &mut handle) };
        if ret != 0 {
            // Rollback: MPI_Op_create failed so no MPI_Op was registered and
            // no trampoline call for this slot can be in flight.  Use the
            // same drop routine the C free path uses (ferrompi_op_drop_closure
            // is idempotent and bounds-checked), then release the slot
            // without calling MPI_Op_free — the slot never held a live
            // MPI_Op, and MPI_Op_free on MPI_OP_NULL is implementation-defined.
            // SAFETY: slot is in range (checked by ferrompi_op_alloc_slot
            // above); the closure at REGISTRY[idx] was published above and no
            // trampoline call for this slot has occurred, since
            // ferrompi_op_create_user just returned failure.
            unsafe { ferrompi_op_drop_closure(slot) };
            // SAFETY: slot is the value ferrompi_op_alloc_slot returned above;
            // the closure stored in it has just been dropped, so no dangling
            // pointer remains for a later trampoline call to observe.
            unsafe { ffi::ferrompi_op_free_slot_only(slot) };
            return Err(Error::from_code_with_op(ret, "op_create"));
        }
        debug_assert_eq!(handle, slot);

        Ok(UserOp {
            handle,
            _marker: PhantomData,
        })
    }

    /// Return the raw slot handle (for use by `allreduce_with_op`).
    #[inline]
    pub(crate) fn raw_handle(&self) -> i32 {
        self.handle
    }
}

impl<T: MpiDatatype> Drop for UserOp<T> {
    fn drop(&mut self) {
        if !rt::drop_guard("UserOp") {
            return;
        }
        // Drop ordering (ADR-0005 Decision 3):
        //   1. ferrompi_op_free → MPI_Op_free (MPI will not invoke the
        //      trampoline after this returns).
        //   2. ferrompi_op_free → ferrompi_op_drop_closure (Rust callback,
        //      drops the Box).
        //   3. ferrompi_op_free → free_op_slot (reclaims the C-side slot).
        //
        // The handle is valid for the lifetime of this UserOp; it is freed
        // exactly once here.
        let ret = unsafe {
            // SAFETY: self.handle was allocated by UserOp::new_impl and has
            // not been freed.  Drop is called exactly once.
            ffi::ferrompi_op_free(self.handle)
        };
        // Log but do not panic in Drop.
        if ret != 0 {
            eprintln!("ferrompi: UserOp::drop — ferrompi_op_free returned error code {ret}");
        }
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::ffi;

    /// `ferrompi_op_create_user` must reject a slot that was never allocated
    /// via `ferrompi_op_alloc_slot`, before it touches MPI.  Needs no MPI
    /// runtime: slot 0's `op_used` entry is zero-initialised static storage,
    /// and this test never calls `ferrompi_op_alloc_slot`.
    #[test]
    fn create_user_rejects_unallocated_slot() {
        let mut handle: i32 = -1;
        // SAFETY: slot 0 has not been allocated in this process, so the
        // op_used check must reject it before any MPI call; handle is a
        // valid i32 out-parameter that ferrompi_op_create_user only writes
        // on success.
        let ret = unsafe { ffi::ferrompi_op_create_user(0, 1, &mut handle) };
        assert_ne!(ret, 0, "must reject a slot that was never allocated");
        assert_eq!(handle, -1, "handle must be untouched on rejection");
    }

    /// `ferrompi_op_free` on a slot that was never allocated must return
    /// MPI_SUCCESS without calling MPI. Needs no MPI runtime: slot 5's
    /// `op_used` entry is zero-initialised static storage, and this test
    /// never calls `ferrompi_op_alloc_slot`.
    #[test]
    fn op_free_on_unused_slot_skips_mpi() {
        // SAFETY: slot 5 is in range and has not been allocated in this
        // process, so the op_used check must return MPI_SUCCESS before any
        // MPI call is made.
        let ret = unsafe { ffi::ferrompi_op_free(5) };
        assert_eq!(ret, 0, "must skip MPI_Op_free on an unused slot");
    }
}
