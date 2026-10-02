//! User-defined MPI reduction operations via [`MPI_Op_create`].
//!
//! This module provides [`UserOp<T>`], a safe wrapper around a user-supplied
//! Rust closure that MPI invokes during reduction collectives
//! (`MPI_Reduce`, `MPI_Allreduce`, `MPI_Scan`, etc.).
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
use std::sync::Arc;

use crate::datatype::{DatatypeTag, MpiDatatype, MpiIndexedDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::rt;

// ============================================================================
// ReduceOp
// ============================================================================

/// Reduction operations
///
/// `MPI_MAXLOC` and `MPI_MINLOC` are [`CollectiveOp::MAX_LOC`] and
/// [`CollectiveOp::MIN_LOC`], for the pair types only. The one-sided
/// accumulate calls add `MPI_REPLACE` ([`AccumulateOp::REPLACE`],
/// [`FetchOp::REPLACE`]) and `MPI_NO_OP` ([`FetchOp::NO_OP`]), which no
/// collective accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ReduceOp {
    /// Sum of values
    Sum = 0,
    /// Maximum value
    Max = 1,
    /// Minimum value
    Min = 2,
    /// Product of values
    Prod = 3,
    /// Bitwise OR (`MPI_BOR`). Valid for integer types; on a floating-point
    /// type the call returns [`Error::InvalidArgument`].
    BitwiseOr = 4,
    /// Bitwise AND (`MPI_BAND`). Valid for integer types; on a floating-point
    /// type the call returns [`Error::InvalidArgument`].
    BitwiseAnd = 5,
    /// Bitwise XOR (`MPI_BXOR`). Valid for integer types; on a floating-point
    /// type the call returns [`Error::InvalidArgument`].
    BitwiseXor = 6,
    /// Logical OR (`MPI_LOR`). Interprets nonzero as `true`. Valid for
    /// integer types; on a floating-point type the call returns
    /// [`Error::InvalidArgument`].
    LogicalOr = 7,
    /// Logical AND (`MPI_LAND`). Interprets nonzero as `true`. Valid for
    /// integer types; on a floating-point type the call returns
    /// [`Error::InvalidArgument`].
    LogicalAnd = 8,
    /// Logical XOR (`MPI_LXOR`). Interprets nonzero as `true`. Valid for
    /// integer types; on a floating-point type the call returns
    /// [`Error::InvalidArgument`].
    LogicalXor = 9,
}

/// A predefined MPI operation, before the shim maps its code to an `MPI_Op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Builtin {
    Reduce(ReduceOp),
    PairMax,
    PairMin,
    Replace,
    NoOp,
}

impl Builtin {
    /// Code that `get_op` in csrc/ferrompi.c maps to an `MPI_Op`; keep in sync.
    pub(crate) const fn code(self) -> i32 {
        match self {
            Self::Reduce(op) => op as i32,
            Self::PairMax => 10,
            Self::PairMin => 11,
            Self::Replace => 12,
            Self::NoOp => 13,
        }
    }
}

#[derive(Clone, Copy)]
enum OpFault {
    FloatBitwise,
    PairNeedsLoc,
    LocNeedsPair,
    ByteNeedsBitwise,
}

#[inline]
fn op_fault(op: Builtin, tag: DatatypeTag) -> Option<OpFault> {
    match tag {
        DatatypeTag::F32 | DatatypeTag::F64 => match op {
            Builtin::Reduce(ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min | ReduceOp::Prod)
            | Builtin::Replace
            | Builtin::NoOp => None,
            Builtin::Reduce(
                ReduceOp::BitwiseOr
                | ReduceOp::BitwiseAnd
                | ReduceOp::BitwiseXor
                | ReduceOp::LogicalOr
                | ReduceOp::LogicalAnd
                | ReduceOp::LogicalXor,
            ) => Some(OpFault::FloatBitwise),
            Builtin::PairMax | Builtin::PairMin => Some(OpFault::LocNeedsPair),
        },
        DatatypeTag::I32
        | DatatypeTag::I64
        | DatatypeTag::U8
        | DatatypeTag::U32
        | DatatypeTag::U64 => match op {
            Builtin::Reduce(_) | Builtin::Replace | Builtin::NoOp => None,
            Builtin::PairMax | Builtin::PairMin => Some(OpFault::LocNeedsPair),
        },
        DatatypeTag::FloatInt
        | DatatypeTag::DoubleInt
        | DatatypeTag::LongInt
        | DatatypeTag::Int2
        | DatatypeTag::ShortInt
        | DatatypeTag::LongDoubleInt => match op {
            Builtin::PairMax | Builtin::PairMin => None,
            Builtin::Reduce(_) | Builtin::Replace | Builtin::NoOp => Some(OpFault::PairNeedsLoc),
        },
        DatatypeTag::Byte => match op {
            Builtin::Reduce(ReduceOp::BitwiseOr | ReduceOp::BitwiseAnd | ReduceOp::BitwiseXor) => {
                None
            }
            Builtin::Reduce(
                ReduceOp::Sum
                | ReduceOp::Max
                | ReduceOp::Min
                | ReduceOp::Prod
                | ReduceOp::LogicalOr
                | ReduceOp::LogicalAnd
                | ReduceOp::LogicalXor,
            )
            | Builtin::PairMax
            | Builtin::PairMin
            | Builtin::Replace
            | Builtin::NoOp => Some(OpFault::ByteNeedsBitwise),
        },
    }
}

#[cold]
#[inline(never)]
fn op_error(fault: OpFault) -> Error {
    Error::InvalidArgument {
        arg: "op",
        reason: match fault {
            OpFault::FloatBitwise => "bitwise and logical ops do not apply to floating-point types",
            OpFault::PairNeedsLoc => "only MAX_LOC and MIN_LOC apply to pair types",
            OpFault::LocNeedsPair => "MAX_LOC and MIN_LOC apply only to pair types",
            OpFault::ByteNeedsBitwise => "only the bitwise ops apply to byte reductions",
        },
    }
}

/// Rejects a predefined op that MPI does not define on the element kind `tag`. On
/// pair and byte tags it also rejects `MPI_REPLACE` and `MPI_NO_OP`, which
/// MPI-4.1 §12.3.4 allows but no window element type reaches.
#[inline]
pub(crate) fn check_op(op: Builtin, tag: DatatypeTag) -> Result<()> {
    match op_fault(op, tag) {
        None => Ok(()),
        Some(fault) => Err(op_error(fault)),
    }
}

#[derive(Debug, Clone, Copy)]
enum CollectiveRepr<'a> {
    Builtin(Builtin),
    User(&'a Arc<OpRegistration>),
}

/// The operation of a reducing collective on elements of type `T`.
///
/// For a primitive `T` ([`MpiDatatype`]) it is built from a [`ReduceOp`], or
/// from a `&`[`UserOp<T>`], which the call borrows. For a pair `T`
/// ([`MpiIndexedDatatype`]) the only operations are
/// [`MAX_LOC`](Self::MAX_LOC) and [`MIN_LOC`](Self::MIN_LOC), so `MAX_LOC` or
/// `MIN_LOC` on a primitive type, or any other predefined op on a pair type,
/// does not compile.
///
/// ```
/// use ferrompi::{CollectiveOp, DoubleInt, ReduceOp};
///
/// let _ = CollectiveOp::<DoubleInt>::MAX_LOC;
/// let _: CollectiveOp<'_, f64> = ReduceOp::Sum.into();
/// ```
///
/// `MAX_LOC` is not defined on a primitive type:
///
/// ```compile_fail,E0599
/// let _ = ferrompi::CollectiveOp::<f64>::MAX_LOC;
/// ```
///
/// A user op applies only to buffers of its own element type, which the
/// user function's slices rely on:
///
/// ```compile_fail,E0277
/// fn f(op: &ferrompi::UserOp<f64>) {
///     let _: ferrompi::CollectiveOp<'_, f32> = op.into();
/// }
/// ```
pub struct CollectiveOp<'a, T> {
    op: CollectiveRepr<'a>,
    // T appears only as a `fn` return, so it adds no auto-trait or drop bound.
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for CollectiveOp<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for CollectiveOp<'_, T> {}

impl<T> std::fmt::Debug for CollectiveOp<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CollectiveOp").field(&self.op).finish()
    }
}

impl<T: MpiDatatype> From<ReduceOp> for CollectiveOp<'_, T> {
    fn from(op: ReduceOp) -> Self {
        Self {
            op: CollectiveRepr::Builtin(Builtin::Reduce(op)),
            _marker: PhantomData,
        }
    }
}

impl<'a, T: MpiDatatype> From<&'a UserOp<T>> for CollectiveOp<'a, T> {
    fn from(op: &'a UserOp<T>) -> Self {
        Self {
            op: CollectiveRepr::User(&op.registration),
            _marker: PhantomData,
        }
    }
}

impl<T: MpiIndexedDatatype> CollectiveOp<'static, T> {
    /// `MPI_MAXLOC`: the maximum value and the index that came with it.
    pub const MAX_LOC: Self = Self {
        op: CollectiveRepr::Builtin(Builtin::PairMax),
        _marker: PhantomData,
    };

    /// `MPI_MINLOC`: the minimum value and the index that came with it.
    pub const MIN_LOC: Self = Self {
        op: CollectiveRepr::Builtin(Builtin::PairMin),
        _marker: PhantomData,
    };
}

impl<T> CollectiveOp<'_, T> {
    pub(crate) fn code(&self, tag: DatatypeTag) -> Result<i32> {
        match self.op {
            CollectiveRepr::Builtin(op) => check_op(op, tag).map(|()| op.code()),
            CollectiveRepr::User(registration) => Ok(registration.code()),
        }
    }
}

/// The operation of the one-sided accumulate calls `accumulate` and
/// `raccumulate`.
///
/// MPI-4.1 §12.3.4 allows a predefined reduction operation or `MPI_REPLACE`
/// there, never a user-defined operation, and allows `MPI_NO_OP` only in the
/// fetching calls (see [`FetchOp`]). So it is built from a [`ReduceOp`], or is
/// [`REPLACE`](Self::REPLACE).
///
/// ```
/// use ferrompi::{AccumulateOp, ReduceOp};
///
/// let _ = AccumulateOp::REPLACE;
/// let _: AccumulateOp = ReduceOp::Sum.into();
/// ```
///
/// A user op is not an accumulate op:
///
/// ```compile_fail,E0277
/// fn f(op: &ferrompi::UserOp<f64>) {
///     let _ = ferrompi::AccumulateOp::from(op);
/// }
/// ```
///
/// `MPI_NO_OP` is not an accumulate op:
///
/// ```compile_fail,E0599
/// let _ = ferrompi::AccumulateOp::NO_OP;
/// ```
///
/// `MPI_REPLACE` is not a collective op:
///
/// ```compile_fail,E0277
/// let mpi = ferrompi::Mpi::init().unwrap();
/// let world = mpi.world();
/// let send = [1.0_f64];
/// let mut recv = [0.0_f64];
/// world
///     .allreduce(&send, &mut recv, ferrompi::AccumulateOp::REPLACE)
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumulateOp(Builtin);

/// The operation of the fetching one-sided calls `get_accumulate` and
/// `fetch_and_op`.
///
/// MPI-4.1 §12.3.4 allows a predefined reduction operation, `MPI_REPLACE` or
/// `MPI_NO_OP` there, never a user-defined operation. So it is built from a
/// [`ReduceOp`], or is [`REPLACE`](Self::REPLACE) or [`NO_OP`](Self::NO_OP).
///
/// ```
/// use ferrompi::{FetchOp, ReduceOp};
///
/// let _ = FetchOp::REPLACE;
/// let _ = FetchOp::NO_OP;
/// let _: FetchOp = ReduceOp::Sum.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchOp(Builtin);

impl From<ReduceOp> for AccumulateOp {
    fn from(op: ReduceOp) -> Self {
        Self(Builtin::Reduce(op))
    }
}

impl From<ReduceOp> for FetchOp {
    fn from(op: ReduceOp) -> Self {
        Self(Builtin::Reduce(op))
    }
}

impl AccumulateOp {
    /// `MPI_REPLACE`: the target takes the origin value, like `MPI_Put`.
    pub const REPLACE: Self = Self(Builtin::Replace);

    #[cfg(feature = "rma")]
    pub(crate) fn code(self, tag: DatatypeTag) -> Result<i32> {
        check_op(self.0, tag).map(|()| self.0.code())
    }
}

impl FetchOp {
    /// `MPI_REPLACE`: the target takes the origin value and the call returns
    /// the value it held before (a swap).
    pub const REPLACE: Self = Self(Builtin::Replace);

    /// `MPI_NO_OP`: the target is unchanged and the call returns its value (an
    /// atomic get).
    pub const NO_OP: Self = Self(Builtin::NoOp);

    #[cfg(feature = "rma")]
    pub(crate) fn code(self, tag: DatatypeTag) -> Result<i32> {
        check_op(self.0, tag).map(|()| self.0.code())
    }
}

// ============================================================================
// Slot count — must match MAX_OPS in csrc/ferrompi.c
// ============================================================================
const MAX_OPS: usize = 16;

/// Type-erased closure stored in each op slot: it receives MPI's raw
/// `invec`/`inoutvec` pointers and element count.
type ByteClosure = Box<dyn Fn(*const c_void, *mut c_void, usize) + Send + Sync + 'static>;

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
/// * `invec` and `inoutvec` are the pointers MPI passed to the user
///   function, each to `len` elements of the datatype the op was applied
///   with.
/// * `len` is non-negative.
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

    // Wrap the closure call in catch_unwind (ADR-0005 Decision 6).
    // A panic across the FFI boundary is UB; abort on Err.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        closure(invec, inoutvec, len as usize);
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
/// invocation, enforced by the `Sync` bound. The closure is moved into a
/// static slot table accessible from any thread (MPI may call it from an
/// internal thread pool), enforced by the `Send` bound. The closure is held
/// until `MPI_Op_free` returns, which may be much later than the call site,
/// enforced by the `'static` bound.
///
/// # Panic behaviour
///
/// A panic inside the closure **aborts the process**.
///
/// Panicking across a C FFI boundary is undefined behaviour; there is no
/// mechanism for MPI to propagate or observe a Rust panic.  The trampoline
/// wraps every closure call in [`std::panic::catch_unwind`]: on `Err` it calls
/// [`std::process::abort`] before the panic can reach the C frame.  Treat a
/// panic inside a reduction closure as a fatal programming error.
///
/// # Lifetime
///
/// MPI never invokes the closure after it is dropped, because the op is freed
/// only when the last reference to its registration goes away.
///
/// # Slot-table limit
///
/// At most 16 `UserOp` instances may be live concurrently per process, and a
/// slot is held until the `UserOp` is dropped. Attempting to create a
/// seventeenth returns [`Error::ResourceExhausted`] with resource
/// [`ResourceKind::Operation`](crate::ResourceKind::Operation).
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
/// world.allreduce(&send, &mut recv, &op).unwrap();
/// ```
pub struct UserOp<T: MpiDatatype> {
    pub(crate) registration: Arc<OpRegistration>,
    _marker: PhantomData<T>,
}

/// One live `MPI_Op` slot and its closure; freed when the last holder drops.
#[derive(Debug)]
pub(crate) struct OpRegistration {
    pub(crate) slot: i32,
}

/// Op codes at or above this name the user op in slot `code - USER_OP_CODE_BASE`;
/// smaller codes are the predefined ops `get_op` switches on. Mirrors
/// `FERROMPI_OP_USER_BASE` in csrc/ferrompi.h; keep in sync.
pub(crate) const USER_OP_CODE_BASE: i32 = 64;

impl OpRegistration {
    pub(crate) fn code(&self) -> i32 {
        USER_OP_CODE_BASE + self.slot
    }
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
            return Err(Error::from_code_with_op(ret, "op_create"));
        }
        let idx = slot as usize;
        debug_assert!(idx < MAX_OPS, "slot out of range");

        // Step 2: wrap the typed closure in the type-erased adapter.
        let byte_closure: ByteClosure = typed_adapter::<T, F>(f);

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
            registration: Arc::new(OpRegistration { slot }),
            _marker: PhantomData,
        })
    }
}

/// Wraps a typed reduction closure in the type-erased form `REGISTRY` stores.
fn typed_adapter<T: MpiDatatype, F>(f: F) -> ByteClosure
where
    F: Fn(&[T], &mut [T]) + Send + Sync + 'static,
{
    Box::new(
        move |invec: *const c_void, inoutvec: *mut c_void, len: usize| {
            if len == 0 {
                // MPI may pass null buffers for an empty reduction; an empty
                // slice needs no pointer.
                f(&[], &mut []);
                return;
            }
            // SAFETY: invec/inoutvec point to `len` elements of the datatype the
            // op was applied with, which every reduction passes as T's own
            // because CollectiveOp<'a, T> ties the op's T to the buffers' T;
            // MPI's buffers are aligned for that datatype; the two buffers
            // are distinct; T: MpiDatatype is Copy with a stable layout.
            let (invec, inoutvec) = unsafe {
                (
                    std::slice::from_raw_parts(invec.cast::<T>(), len),
                    std::slice::from_raw_parts_mut(inoutvec.cast::<T>(), len),
                )
            };
            f(invec, inoutvec);
        },
    )
}

impl Drop for OpRegistration {
    fn drop(&mut self) {
        if !rt::drop_guard("UserOp") {
            return;
        }
        // Drop ordering (ADR-0005 Decision 3):
        //   1. ferrompi_op_free → MPI_Op_free (MPI will not invoke the
        //      trampoline after this returns, because no pending operation
        //      references the op: the last OpRegistration holder is gone).
        //   2. ferrompi_op_free → ferrompi_op_drop_closure (Rust callback,
        //      drops the Box).
        //   3. ferrompi_op_free → free_op_slot (reclaims the C-side slot).
        //
        // The slot is valid for the lifetime of this registration; it is freed
        // exactly once here.
        let ret = unsafe {
            // SAFETY: self.slot was allocated by UserOp::new_impl and has
            // not been freed.  Drop is called exactly once.
            ffi::ferrompi_op_free(self.slot)
        };
        // Log but do not panic in Drop.
        if ret != 0 {
            eprintln!("ferrompi: freeing a user op: ferrompi_op_free returned error code {ret}");
        }
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    use crate::datatype::DatatypeTag;
    use crate::error::Error;
    use crate::ffi;
    use crate::DoubleInt;

    use super::{check_op, ferrompi_op_drop_closure, rust_user_op_invoke, typed_adapter};
    use super::{
        AccumulateOp, Builtin, CollectiveOp, FetchOp, OpRegistration, ReduceOp, UserOp, MAX_OPS,
        REGISTRY, USER_OP_CODE_BASE,
    };

    /// The trampoline must hand the closure typed, full-length buffers built
    /// from MPI's raw pointers. Uses slot `MAX_OPS - 1`, which no other test
    /// touches, so it cannot race a concurrent trampoline call.
    #[test]
    fn trampoline_hands_the_closure_full_typed_buffers() {
        let slot = (MAX_OPS - 1) as i32;
        let adapter = typed_adapter::<f64, _>(|a: &[f64], b: &mut [f64]| {
            for (x, y) in a.iter().zip(b.iter_mut()) {
                *y += *x;
            }
        });
        // SAFETY: publishing a freshly boxed adapter into this slot with
        // Release, matching the protocol REGISTRY's doc comment states; slot
        // MAX_OPS - 1 is used by no other test.
        REGISTRY[slot as usize].store(Box::into_raw(Box::new(adapter)), Ordering::Release);

        let invec = [1.0_f64, 2.0, 3.0, 4.0];
        let mut inout = [10.0_f64, 20.0, 30.0, 40.0];
        // SAFETY: slot was just published above with a live typed_adapter
        // closure for f64; invec/inout are 4-element f64 buffers matching
        // len=4, distinct from each other.
        unsafe {
            rust_user_op_invoke(slot, invec.as_ptr().cast(), inout.as_mut_ptr().cast(), 4);
        }
        // SAFETY: slot is in range and was published above by this test
        // alone; no trampoline call for it is in flight.
        unsafe { ferrompi_op_drop_closure(slot) };

        assert_eq!(inout, [11.0, 22.0, 33.0, 44.0]);
    }

    /// The trampoline must accept null buffers when `len == 0` and still
    /// invoke the closure, with empty slices. Uses slot `MAX_OPS - 2`, which
    /// no other test touches.
    #[test]
    fn trampoline_accepts_null_buffers_of_zero_length() {
        let slot = (MAX_OPS - 2) as i32;
        let seen: Arc<Mutex<Option<(usize, usize)>>> = Arc::new(Mutex::new(None));
        let seen_in_closure = Arc::clone(&seen);
        let adapter = typed_adapter::<f64, _>(move |a: &[f64], b: &mut [f64]| {
            *seen_in_closure.lock().unwrap() = Some((a.len(), b.len()));
        });
        // SAFETY: publishing a freshly boxed adapter into this slot with
        // Release; slot MAX_OPS - 2 is used by no other test.
        REGISTRY[slot as usize].store(Box::into_raw(Box::new(adapter)), Ordering::Release);

        // SAFETY: slot was just published above; MPI may pass null buffers
        // for an empty reduction, and len=0 means no element is read from
        // either pointer.
        unsafe {
            rust_user_op_invoke(slot, std::ptr::null(), std::ptr::null_mut(), 0);
        }
        // SAFETY: slot is in range and was published above by this test
        // alone; no trampoline call for it is in flight.
        unsafe { ferrompi_op_drop_closure(slot) };

        assert_eq!(*seen.lock().unwrap(), Some((0, 0)));
    }

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

    /// A user op is a classic `MPI_User_function` with an `int` length, so every
    /// reduction shim refuses a count above `i32::MAX` for it, before any MPI
    /// call or request allocation. Needs no MPI runtime: the refusal returns
    /// ahead of every MPI function, and the lookups before it are table reads.
    #[test]
    fn user_op_code_refuses_large_counts_in_every_reduction_shim() {
        let count = i64::from(i32::MAX) + 1;
        let dt = DatatypeTag::F64 as i32;
        let op = USER_OP_CODE_BASE;
        let sb = std::ptr::null();
        let rb = std::ptr::null_mut();
        let mut reqs = [-1_i64; 10];
        let [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9] = &mut reqs;

        let rets = [
            // SAFETY: the guard returns before any buffer access.
            ("reduce", unsafe {
                ffi::ferrompi_reduce(sb, rb, count, dt, op, 0, 0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("allreduce", unsafe {
                ffi::ferrompi_allreduce(sb, rb, count, dt, op, 0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("scan", unsafe {
                ffi::ferrompi_scan(sb, rb, count, dt, op, 0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("exscan", unsafe {
                ffi::ferrompi_exscan(sb, rb, count, dt, op, 0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("reduce_scatter_block", unsafe {
                ffi::ferrompi_reduce_scatter_block(sb, rb, count, dt, op, 0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("iallreduce", unsafe {
                ffi::ferrompi_iallreduce(sb, rb, count, dt, op, 0, r0)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("ireduce", unsafe {
                ffi::ferrompi_ireduce(sb, rb, count, dt, op, 0, 0, r1)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("iscan", unsafe {
                ffi::ferrompi_iscan(sb, rb, count, dt, op, 0, r2)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("iexscan", unsafe {
                ffi::ferrompi_iexscan(sb, rb, count, dt, op, 0, r3)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("ireduce_scatter_block", unsafe {
                ffi::ferrompi_ireduce_scatter_block(sb, rb, count, dt, op, 0, r4)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("allreduce_init", unsafe {
                ffi::ferrompi_allreduce_init(sb, rb, count, dt, op, 0, r5)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("reduce_init", unsafe {
                ffi::ferrompi_reduce_init(sb, rb, count, dt, op, 0, 0, r6)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("scan_init", unsafe {
                ffi::ferrompi_scan_init(sb, rb, count, dt, op, 0, r7)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("exscan_init", unsafe {
                ffi::ferrompi_exscan_init(sb, rb, count, dt, op, 0, r8)
            }),
            // SAFETY: the guard returns before any buffer access.
            ("reduce_scatter_block_init", unsafe {
                ffi::ferrompi_reduce_scatter_block_init(sb, rb, count, dt, op, 0, r9)
            }),
        ];

        for (shim, ret) in rets {
            assert_ne!(ret, 0, "{shim} must refuse a user op above i32::MAX");
        }
        assert_eq!(reqs, [-1; 10], "a refusal must leave every request unset");
    }

    /// `Arc<OpRegistration>` keeps the auto traits only while the registration
    /// itself is `Send + Sync`.
    #[test]
    fn user_op_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UserOp<f64>>();
        assert_send_sync::<UserOp<u8>>();
    }

    /// The exhaustive `match` has no wildcard and no `cfg`, so a variant added
    /// under one feature configuration fails to compile in that configuration.
    #[test]
    fn reduce_op_repr_values() {
        fn expected(op: ReduceOp) -> i32 {
            match op {
                ReduceOp::Sum => 0,
                ReduceOp::Max => 1,
                ReduceOp::Min => 2,
                ReduceOp::Prod => 3,
                ReduceOp::BitwiseOr => 4,
                ReduceOp::BitwiseAnd => 5,
                ReduceOp::BitwiseXor => 6,
                ReduceOp::LogicalOr => 7,
                ReduceOp::LogicalAnd => 8,
                ReduceOp::LogicalXor => 9,
            }
        }
        for op in FLOAT_OK.into_iter().chain(FLOAT_BAD) {
            assert_eq!(op as i32, expected(op));
        }
    }

    #[test]
    fn collective_op_codes_match_the_shim_switch() {
        for op in FLOAT_OK.into_iter().chain(FLOAT_BAD) {
            assert_eq!(
                CollectiveOp::<i32>::from(op)
                    .code(DatatypeTag::I32)
                    .unwrap(),
                op as i32
            );
            assert_eq!(AccumulateOp::from(op).0.code(), op as i32);
            assert_eq!(FetchOp::from(op).0.code(), op as i32);
        }
        assert_eq!(
            CollectiveOp::<DoubleInt>::MAX_LOC
                .code(DatatypeTag::DoubleInt)
                .unwrap(),
            10
        );
        assert_eq!(
            CollectiveOp::<DoubleInt>::MIN_LOC
                .code(DatatypeTag::DoubleInt)
                .unwrap(),
            11
        );
        assert_eq!(AccumulateOp::REPLACE.0.code(), 12);
        assert_eq!(FetchOp::REPLACE.0.code(), 12);
        assert_eq!(FetchOp::NO_OP.0.code(), 13);
    }

    /// Slot 9 is never allocated by any test, so dropping this registration
    /// makes no MPI call.
    #[test]
    fn collective_op_user_code_is_the_registration_code() {
        let op = UserOp::<f64> {
            registration: Arc::new(OpRegistration { slot: 9 }),
            _marker: PhantomData,
        };
        assert_eq!(
            CollectiveOp::from(&op).code(DatatypeTag::F64).unwrap(),
            USER_OP_CODE_BASE + 9
        );
    }

    const FLOAT_TAGS: [DatatypeTag; 2] = [DatatypeTag::F32, DatatypeTag::F64];
    const INT_TAGS: [DatatypeTag; 5] = [
        DatatypeTag::I32,
        DatatypeTag::I64,
        DatatypeTag::U8,
        DatatypeTag::U32,
        DatatypeTag::U64,
    ];
    const PAIR_TAGS: [DatatypeTag; 6] = [
        DatatypeTag::FloatInt,
        DatatypeTag::DoubleInt,
        DatatypeTag::LongInt,
        DatatypeTag::Int2,
        DatatypeTag::ShortInt,
        DatatypeTag::LongDoubleInt,
    ];
    const FLOAT_OK: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod];
    const BITWISE: [ReduceOp; 3] = [
        ReduceOp::BitwiseOr,
        ReduceOp::BitwiseAnd,
        ReduceOp::BitwiseXor,
    ];
    const FLOAT_BAD: [ReduceOp; 6] = [
        ReduceOp::BitwiseOr,
        ReduceOp::BitwiseAnd,
        ReduceOp::BitwiseXor,
        ReduceOp::LogicalOr,
        ReduceOp::LogicalAnd,
        ReduceOp::LogicalXor,
    ];

    const FLOAT_BITWISE: &str = "bitwise and logical ops do not apply to floating-point types";
    const PAIR_NEEDS_LOC: &str = "only MAX_LOC and MIN_LOC apply to pair types";
    const LOC_NEEDS_PAIR: &str = "MAX_LOC and MIN_LOC apply only to pair types";
    const BYTE_NEEDS_BITWISE: &str = "only the bitwise ops apply to byte reductions";

    fn reduce_ops() -> impl Iterator<Item = Builtin> {
        FLOAT_OK.into_iter().chain(FLOAT_BAD).map(Builtin::Reduce)
    }

    fn builtins() -> Vec<Builtin> {
        reduce_ops()
            .chain([
                Builtin::PairMax,
                Builtin::PairMin,
                Builtin::Replace,
                Builtin::NoOp,
            ])
            .collect()
    }

    fn assert_rejected(op: Builtin, tag: DatatypeTag, expected: &str) {
        assert!(
            matches!(
                check_op(op, tag),
                Err(Error::InvalidArgument { arg: "op", reason }) if reason == expected
            ),
            "{op:?} on {tag:?}"
        );
    }

    #[test]
    fn check_op_float_bitwise_returns_invalid_argument() {
        for tag in FLOAT_TAGS {
            for op in FLOAT_BAD {
                assert_rejected(Builtin::Reduce(op), tag, FLOAT_BITWISE);
            }
        }
    }

    #[test]
    fn check_op_pair_non_loc_returns_invalid_argument() {
        let non_loc: Vec<Builtin> = builtins()
            .into_iter()
            .filter(|op| !matches!(op, Builtin::PairMax | Builtin::PairMin))
            .collect();
        assert_eq!(non_loc.len(), 12);
        for tag in PAIR_TAGS {
            for &op in &non_loc {
                assert_rejected(op, tag, PAIR_NEEDS_LOC);
            }
        }
    }

    #[test]
    fn check_op_loc_on_primitive_returns_invalid_argument() {
        for tag in FLOAT_TAGS.into_iter().chain(INT_TAGS) {
            for op in [Builtin::PairMax, Builtin::PairMin] {
                assert_rejected(op, tag, LOC_NEEDS_PAIR);
            }
        }
    }

    #[test]
    fn check_op_byte_non_bitwise_returns_invalid_argument() {
        let non_bitwise: Vec<Builtin> = builtins()
            .into_iter()
            .filter(|op| !matches!(op, Builtin::Reduce(r) if BITWISE.contains(r)))
            .collect();
        assert_eq!(non_bitwise.len(), 11);
        for op in non_bitwise {
            assert_rejected(op, DatatypeTag::Byte, BYTE_NEEDS_BITWISE);
        }
    }

    #[test]
    fn check_op_accepts_every_op_mpi_defines() {
        let mut accepted = 0;
        let mut accept = |op: Builtin, tag: DatatypeTag| {
            assert!(check_op(op, tag).is_ok(), "{op:?} on {tag:?}");
            accepted += 1;
        };
        for tag in FLOAT_TAGS {
            for op in FLOAT_OK {
                accept(Builtin::Reduce(op), tag);
            }
            accept(Builtin::Replace, tag);
            accept(Builtin::NoOp, tag);
        }
        for tag in INT_TAGS {
            for op in reduce_ops().chain([Builtin::Replace, Builtin::NoOp]) {
                accept(op, tag);
            }
        }
        for tag in PAIR_TAGS {
            accept(Builtin::PairMax, tag);
            accept(Builtin::PairMin, tag);
        }
        for op in BITWISE {
            accept(Builtin::Reduce(op), DatatypeTag::Byte);
        }
        assert_eq!(accepted, 87);

        // The four rejection tests above cover exactly the rest of the 14 x 14 grid.
        let rejected = FLOAT_TAGS.len() * FLOAT_BAD.len()
            + PAIR_TAGS.len() * 12
            + (FLOAT_TAGS.len() + INT_TAGS.len()) * 2
            + 11;
        let tags: Vec<DatatypeTag> = FLOAT_TAGS
            .into_iter()
            .chain(INT_TAGS)
            .chain(PAIR_TAGS)
            .chain([DatatypeTag::Byte])
            .collect();
        let grid = builtins().len() * tags.len();
        assert_eq!(grid, 196);
        assert_eq!(accepted + rejected, grid);
        let ok_in_grid = builtins()
            .into_iter()
            .flat_map(|op| tags.iter().map(move |&tag| check_op(op, tag)))
            .filter(Result::is_ok)
            .count();
        assert_eq!(ok_in_grid, accepted);
    }
}
