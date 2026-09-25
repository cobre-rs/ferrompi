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

use crate::datatype::MpiDatatype;
use crate::error::{Error, MpiErrorClass, Result};
use crate::ffi;

// ============================================================================
// Slot count — must match MAX_OPS in csrc/ferrompi.c
// ============================================================================
const MAX_OPS: usize = 16;

/// Type alias for the byte-level closure stored in each op slot.
///
/// Using an alias avoids the `clippy::type_complexity` lint at every use site.
type ByteClosure = Box<dyn Fn(&[u8], &mut [u8]) + Send + Sync + 'static>;

// ============================================================================
// Static closure registry
//
// Each slot holds a byte-level closure adapter.  The adapter is a
// `Box<dyn Fn(&[u8], &mut [u8]) + Send + Sync + 'static>` constructed in
// `UserOp::new_impl` from the caller's typed `Fn(&[T], &mut [T])` by wrapping
// it in a byte-reinterpreting adapter closure.
//
// OnceLock is used so that each slot can be written exactly once and read many
// times concurrently, without requiring a Mutex.  Dropping the UserOp must
// reconstruct the Box from the raw pointer rather than going through OnceLock
// again (OnceLock does not expose a reset path); see `ferrompi_op_drop_closure`.
// ============================================================================

/// Per-slot registry of raw fat-pointer halves.
///
/// We cannot store `Box<dyn Fn(...)>` in a static array of `OnceLock` because
/// statics require `const`-initializable values and `OnceLock::new()` is
/// `const`-stable only from Rust 1.70+, but more importantly we need to be
/// able to reconstruct and drop the `Box` from `ferrompi_op_drop_closure`,
/// which is called from C and cannot go through the OnceLock API.
///
/// The chosen approach: store the two halves of the fat pointer (`*mut ()` data
/// and `*mut ()` vtable) as atomic raw pointers.  We encode "unset" as null and
/// "set" as non-null.  The fat pointer is stored atomically so that concurrent
/// reads from MPI threads are data-race-free (Relaxed load is sufficient since
/// the store in `new_impl` happens-before any MPI invocation of the trampoline —
/// MPI_Op_create establishes that ordering).
///
/// In practice the registry data and vtable are written once (in `new_impl`)
/// and then only read (in `rust_user_op_invoke`) or reset to null (in
/// `ferrompi_op_drop_closure`). We use `AtomicPtr` with `Relaxed` ordering
/// because:
///   * The store in `new_impl` is sequenced before `MPI_Op_create` which is the
///     happens-before anchor for all subsequent MPI trampoline calls.
///   * `ferrompi_op_drop_closure` is called from `ferrompi_op_free` which is
///     called only after `MPI_Op_free` returns — MPI guarantees no further
///     trampoline calls after that point.
use std::sync::atomic::{AtomicPtr, Ordering};

struct RegistrySlot {
    data: AtomicPtr<()>,
    vtbl: AtomicPtr<()>,
}

impl RegistrySlot {
    const fn new() -> Self {
        Self {
            data: AtomicPtr::new(std::ptr::null_mut()),
            vtbl: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

// SAFETY: AtomicPtr<()> is Send + Sync by design; raw pointers are wrapped in
// atomics which provide the necessary synchronisation.
unsafe impl Send for RegistrySlot {}
// SAFETY: &RegistrySlot exposes only atomic loads/stores of the two pointer
// fields; every access goes through AtomicPtr's own synchronisation, so
// concurrent shared access from multiple threads has no data race.
unsafe impl Sync for RegistrySlot {}

static REGISTRY: [RegistrySlot; MAX_OPS] = [
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
    RegistrySlot::new(),
];

// ============================================================================
// Extern "C" callbacks exposed to the C layer
// ============================================================================

/// Called by each C trampoline `ferrompi_user_op_trampoline_N`.
///
/// The C trampoline passes the slot's fat-pointer halves and the raw buffer
/// pointers it received from MPI.  This function reconstructs the byte-level
/// closure and invokes it, wrapped in `catch_unwind`.
///
/// # Safety
///
/// * `closure_data` and `closure_vtbl` are the two halves of a valid
///   `*mut dyn Fn(&[u8], &mut [u8]) + Send + Sync + 'static` fat pointer
///   previously stored by `UserOp::new_impl`.
/// * `invec` is a valid read-only pointer to `len * byte_size` bytes.
/// * `inoutvec` is a valid read-write pointer to `len * byte_size` bytes.
/// * `dt_tag` is the `FERROMPI_*` tag that matches the type `T` the `UserOp`
///   was parameterised with.
///
/// Called from C, so the ABI must be exactly `extern "C"`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_user_op_invoke(
    closure_data: *mut c_void,
    closure_vtbl: *mut c_void,
    invec: *const c_void,
    inoutvec: *mut c_void,
    len: std::ffi::c_int,
    _dt_tag: std::ffi::c_int,
) {
    // Reconstruct the fat pointer from its two halves.
    // The fat pointer layout is [data_ptr, vtable_ptr] on all Rust targets;
    // we encode it as a 2-element array of *mut () and transmute.
    let fat_ptr_halves: [*mut (); 2] = [closure_data.cast(), closure_vtbl.cast()];
    // SAFETY: fat_ptr_halves holds a valid fat pointer for
    // `*const dyn Fn(&[u8], &mut [u8]) + Send + Sync`.  The memory it points
    // to is alive for the duration of this call: ferrompi_op_free (which frees
    // the closure) calls MPI_Op_free first and only drops the closure after
    // MPI_Op_free returns, so no concurrent drop can occur here.
    // SAFETY: fat_ptr_halves encodes a valid &dyn for the ByteClosure stored
    // at slot registration time.  We borrow it as a shared reference; the
    // Box is still owned (it is dropped only in ferrompi_op_drop_closure, after
    // MPI_Op_free returns).
    let closure: &(dyn Fn(&[u8], &mut [u8]) + Send + Sync) =
        // Transmute [*mut (); 2] → fat-pointer reference.  This is the standard
        // pattern for reconstructing a dyn reference from a stored fat pointer.
        unsafe { std::mem::transmute(fat_ptr_halves) };

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

/// Called by `ferrompi_op_free` (in C) after `MPI_Op_free` returns.
///
/// Reconstructs the `Box<dyn Fn(...)>` from the raw fat pointer halves stored
/// in the slot and drops it.  After this function returns, the slot is cleared
/// by `free_op_slot` in C.
///
/// # Safety
///
/// * `slot` must be in range `0..MAX_OPS`.
/// * The fat pointer stored in `REGISTRY[slot]` must point to a valid
///   `Box<dyn Fn(&[u8], &mut [u8]) + Send + Sync + 'static>` that was
///   previously stored by `UserOp::new_impl` and has not yet been dropped.
/// * This function is called exactly once per `UserOp`, from C, after
///   `MPI_Op_free` has returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ferrompi_op_drop_closure(slot: i32) {
    if slot < 0 || slot as usize >= MAX_OPS {
        return;
    }
    let idx = slot as usize;
    let data = REGISTRY[idx]
        .data
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    let vtbl = REGISTRY[idx]
        .vtbl
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    if data.is_null() || vtbl.is_null() {
        return;
    }
    // Reconstruct the fat pointer and drop the Box.
    let fat_ptr_halves: [*mut (); 2] = [data, vtbl];
    // SAFETY: fat_ptr_halves is the valid fat pointer stored by new_impl.
    // We are reconstructing ownership of the Box; it has not been dropped.
    // MPI_Op_free has returned so no trampoline call can race with this drop.
    let closure: ByteClosure = unsafe { std::mem::transmute(fat_ptr_halves) };
    drop(closure);
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

        // Step 3: store the fat pointer in the static registry.
        //
        // We decompose the Box into its two fat-pointer halves and store them
        // as raw pointers so that the C layer can pass them back via
        // rust_user_op_invoke, and so that ferrompi_op_drop_closure can
        // reconstruct and drop the Box.
        let raw_fat: [*mut (); 2] = unsafe {
            // SAFETY: transmuting Box<dyn Fn(...)> into [*mut (); 2] extracts
            // the fat pointer halves without running the destructor.  We
            // reconstruct the Box in ferrompi_op_drop_closure.
            std::mem::transmute(Box::into_raw(byte_closure))
        };
        // Release store: subsequent loads in the trampoline (Relaxed) are
        // guaranteed to observe these values because the call to
        // ferrompi_op_create_user (MPI_Op_create) establishes the
        // happens-before edge between this store and any trampoline invocation.
        REGISTRY[idx].data.store(raw_fat[0], Ordering::Release);
        REGISTRY[idx].vtbl.store(raw_fat[1], Ordering::Release);

        // Step 4: call MPI_Op_create via the C shim.
        //
        // ferrompi_op_set_closure stores the fat-pointer halves into the C-side
        // op_closure_data/op_closure_vtbl arrays so the C trampolines can pass
        // them to rust_user_op_invoke.
        // SAFETY: slot is in range (checked by ferrompi_op_alloc_slot above);
        // raw_fat[0]/raw_fat[1] are the two halves of a fat pointer this
        // function just published into REGISTRY[idx], stored with Release
        // ordering above so the trampoline observes a consistent pair.
        unsafe {
            ffi::ferrompi_op_set_closure(slot, raw_fat[0].cast(), raw_fat[1].cast());
        }

        let mut handle: i32 = -1;
        // SAFETY: handle is a local out-parameter written by
        // ferrompi_op_create_user before this function reads it below; slot is
        // the value ferrompi_op_alloc_slot returned above.
        let ret = unsafe { ffi::ferrompi_op_create_user(slot, commute, &mut handle) };
        if ret != 0 {
            // Rollback: MPI_Op_create failed so no MPI_Op was registered.
            // We must NOT call ferrompi_op_free here — that would call
            // MPI_Op_free on MPI_OP_NULL (the slot was never populated),
            // which is implementation-defined behaviour.
            //
            // Instead:
            //   1. Reconstruct and drop the Box<dyn Fn(...)> directly from
            //      the registry fat-pointer halves we stored above.
            //   2. Call ferrompi_op_free_slot_only to clear the closure
            //      pointers and release the op_used slot without touching
            //      MPI_Op_free.
            let data = REGISTRY[idx]
                .data
                .swap(std::ptr::null_mut(), Ordering::Relaxed);
            let vtbl = REGISTRY[idx]
                .vtbl
                .swap(std::ptr::null_mut(), Ordering::Relaxed);
            if !data.is_null() && !vtbl.is_null() {
                let fat: [*mut (); 2] = [data, vtbl];
                // SAFETY: we just stored this value above; it has not been
                // dropped yet and this is the only owner.
                let closure: ByteClosure = unsafe { std::mem::transmute(fat) };
                drop(closure);
            }
            // Release the C-side slot without calling MPI_Op_free.
            // SAFETY: slot is the value ferrompi_op_alloc_slot returned above;
            // the closure fat pointer stored in it has just been reconstructed
            // and dropped (or was already null), so no dangling pointer remains
            // for a later trampoline call to observe.
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
    use super::*;
    use crate::datatype::MpiDatatype;

    /// Verify that UserOp<T> compiles for any MpiDatatype T.
    #[test]
    fn user_op_struct_compiles() {
        fn _check<T: MpiDatatype>(_: &UserOp<T>) {}
    }
}
