//! MPI datatype trait and type tag mapping.
//!
//! This module provides three sealed traits:
//!
//! - [`MpiDatatype`]: maps primitive Rust types to MPI datatype tags for all
//!   communication operations (broadcast, send, recv, allreduce with scalar ops).
//! - [`MpiIndexedDatatype`]: marks the six MPI paired value+index structs that are
//!   only valid with [`MPI_MAXLOC`/`MPI_MINLOC`](crate::CollectiveOp::MAX_LOC) reductions.
//! - [`BytePermutable`]: marks types safe for byte-level bitwise reductions via
//!   `MPI_BYTE` (`BitwiseOr`, `BitwiseAnd`, `BitwiseXor`).
//!
//! # Primitive Types (`MpiDatatype`)
//!
//! | Rust Type | MPI Equivalent     | Tag Value |
//! |-----------|-------------------|-----------|
//! | `f32`     | `MPI_FLOAT`       | 0         |
//! | `f64`     | `MPI_DOUBLE`      | 1         |
//! | `i32`     | `MPI_INT32_T`     | 2         |
//! | `i64`     | `MPI_INT64_T`     | 3         |
//! | `u8`      | `MPI_UINT8_T`     | 4         |
//! | `u32`     | `MPI_UINT32_T`    | 5         |
//! | `u64`     | `MPI_UINT64_T`    | 6         |
//!
//! # Paired Value+Index Types (`MpiIndexedDatatype`)
//!
//! | Rust Struct      | MPI Equivalent        | Tag Value |
//! |------------------|-----------------------|-----------|
//! | [`FloatInt`]     | `MPI_FLOAT_INT`       | 7         |
//! | [`DoubleInt`]    | `MPI_DOUBLE_INT`      | 8         |
//! | [`LongInt`] (Linux x86_64/aarch64/ppc64le only) | `MPI_LONG_INT` | 9 |
//! | [`Int2`]         | `MPI_2INT`            | 10        |
//! | [`ShortInt`]     | `MPI_SHORT_INT`       | 11        |
//! | [`LongDoubleInt`] (Linux x86_64/aarch64/ppc64le only) | `MPI_LONG_DOUBLE_INT` | 12 |
//!
//! # Byte-Permutable Types (`BytePermutable`)
//!
//! | Rust Type       | MPI Equivalent | Tag Value |
//! |-----------------|----------------|-----------|
//! | u8-like bytes   | `MPI_BYTE`     | 13        |

use std::ffi::c_void;

/// Internal module to seal [`MpiDatatype`] — prevents external implementations.
mod sealed {
    use super::DatatypeTag;

    /// Argument of [`Sealed::tag`]: it cannot be named outside this crate, so
    /// the tag cannot be read through a public bound.
    pub struct Token;

    pub trait Sealed {
        fn tag(_: Token) -> DatatypeTag;
    }
}

/// Internal module to seal [`MpiIndexedDatatype`] — a separate seal so that
/// the two marker traits remain disjoint.  A type that implements
/// `MpiIndexedDatatype` must NOT automatically satisfy `MpiDatatype`, and
/// vice-versa.
mod sealed_indexed {
    use super::{sealed::Token, DatatypeTag};

    pub trait Sealed {
        fn tag(_: Token) -> DatatypeTag;
    }
}

/// Internal module to seal [`BytePermutable`] — a separate seal for types
/// whose byte representation is valid for `MPI_BYTE`-typed bitwise reductions.
mod sealed_byte {
    pub trait Sealed {}
}

/// Internal module to seal [`AtomicMpiDatatype`] — a separate seal for types
/// eligible for `MPI_Compare_and_swap` (integer and byte types only).
#[cfg(feature = "rma")]
mod sealed_atomic {
    pub trait Sealed {}
}

/// Tag values matching C-side `FERROMPI_*` defines.
///
/// These discriminants must stay in sync with the `#define FERROMPI_*` values
/// in `csrc/ferrompi.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DatatypeTag {
    /// 32-bit floating point (`MPI_FLOAT`)
    F32 = 0,
    /// 64-bit floating point (`MPI_DOUBLE`)
    F64 = 1,
    /// 32-bit signed integer (`MPI_INT32_T`)
    I32 = 2,
    /// 64-bit signed integer (`MPI_INT64_T`)
    I64 = 3,
    /// 8-bit unsigned integer (`MPI_UINT8_T`)
    U8 = 4,
    /// 32-bit unsigned integer (`MPI_UINT32_T`)
    U32 = 5,
    /// 64-bit unsigned integer (`MPI_UINT64_T`)
    U64 = 6,
    /// Paired `{ f32 value; i32 index }` for `MPI_FLOAT_INT` (MAXLOC/MINLOC)
    FloatInt = 7,
    /// Paired `{ f64 value; i32 index }` for `MPI_DOUBLE_INT` (MAXLOC/MINLOC)
    DoubleInt = 8,
    /// Paired `{ i64 value; i32 index }` for `MPI_LONG_INT` (MAXLOC/MINLOC).
    /// Note: on most 64-bit platforms `long` is 8 bytes, matching `i64`.
    /// The matching `LongInt` struct exists only on Linux x86_64, aarch64
    /// and little-endian powerpc64.
    LongInt = 9,
    /// Paired `{ i32 value; i32 index }` for `MPI_2INT` (MAXLOC/MINLOC)
    Int2 = 10,
    /// Paired `{ i16 value; i32 index }` for `MPI_SHORT_INT` (MAXLOC/MINLOC)
    ShortInt = 11,
    /// Paired `{ f128-equivalent value; i32 index }` for `MPI_LONG_DOUBLE_INT`
    /// (MAXLOC/MINLOC). Uses `[u8; 16]` on x86_64 where `long double` is 80-bit
    /// extended precision stored in 16 bytes.
    /// The matching `LongDoubleInt` struct exists only on Linux x86_64,
    /// aarch64 and little-endian powerpc64.
    LongDoubleInt = 12,
    /// Opaque 1-byte unit (`MPI_BYTE`) for type-erased bitwise reductions.
    ///
    /// Used exclusively with [`BytePermutable`]-bounded types and
    /// [`Communicator::allreduce_bytes`](crate::Communicator::allreduce_bytes). The count passed to MPI is the
    /// total byte count (`element_count * size_of::<T>()`), so MPI treats
    /// the buffer as a flat array of bytes.
    Byte = 13,
}

/// Trait for types that can be used in MPI communication operations.
///
/// This is a **sealed trait** — it cannot be implemented outside this crate.
/// Supported types: [`f32`], [`f64`], [`i32`], [`i64`], [`u8`], [`u32`], [`u64`].
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, MpiDatatype};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Works with f64
/// let mut data_f64 = vec![1.0f64; 10];
/// world.broadcast(&mut data_f64, 0).unwrap();
///
/// // Works with i32
/// let mut data_i32 = vec![42i32; 10];
/// world.broadcast(&mut data_i32, 0).unwrap();
/// ```
///
/// # The datatype code is private
///
/// The code the C layer dispatches on has no public name:
///
/// ```compile_fail,E0432
/// use ferrompi::DatatypeTag;
/// ```
///
/// ```no_run
/// use ferrompi::MpiDatatype;
/// ```
///
/// and it cannot be read through a bound on this trait:
///
/// ```compile_fail,E0599
/// fn f<T: ferrompi::MpiDatatype>() {
///     let _ = T::TAG;
/// }
/// ```
///
/// ```no_run
/// fn f<T: ferrompi::MpiDatatype>() {
///     let _ = ferrompi::CustomDatatype::contiguous::<T>(1);
/// }
/// ```
pub trait MpiDatatype: sealed::Sealed + Copy + Send + 'static {}

/// The tag the C layer dispatches on for `T`.
#[inline]
pub(crate) fn tag_of<T: MpiDatatype>() -> DatatypeTag {
    T::tag(sealed::Token)
}

/// The tag the C layer dispatches on for the pair type `T`.
#[inline]
pub(crate) fn pair_tag_of<T: MpiIndexedDatatype>() -> DatatypeTag {
    T::tag(sealed::Token)
}

/// Bound for the element type of the custom-datatype point-to-point methods
/// ([`Communicator::send_custom`](crate::Communicator::send_custom),
/// [`recv_custom`](crate::Communicator::recv_custom),
/// [`isend_custom`](crate::Communicator::isend_custom) and
/// [`irecv_custom`](crate::Communicator::irecv_custom)).
///
/// # Safety
///
/// Implementors guarantee that every bit pattern of `size_of::<Self>()`
/// bytes is a valid `Self`. The type must therefore have no references, no
/// raw pointers that MPI's write could invalidate, no `Box`, `Vec`, `bool`,
/// `char`, enum, or `NonZero*` fields. Padding bytes are allowed. The type
/// should be `#[repr(C)]` so its layout matches the datatype built for it.
///
/// This trait is blanket-implemented for every [`MpiDatatype`] and for
/// fixed-size arrays `[T; N]` of any `PlainData` type.
///
/// # Example
///
/// ```compile_fail,E0277
/// # use ferrompi::{CustomDatatype, Mpi};
/// # let _mpi = Mpi::init().unwrap();
/// # let world = _mpi.world();
/// let dt8 = CustomDatatype::contiguous::<u8>(8).unwrap();
/// let mut boxes: [Box<u64>; 1] = [Box::new(5)];
/// world.recv_custom(&mut boxes, &dt8, 0, 2).unwrap();
/// ```
///
/// ```no_run
/// # use ferrompi::{CustomDatatype, Mpi, PlainData};
/// #[repr(C)]
/// #[derive(Clone, Copy)]
/// struct Pair { v: f64, i: i32 }
/// // SAFETY: Pair is #[repr(C)] of an f64 and an i32, so any bit pattern is valid.
/// unsafe impl PlainData for Pair {}
/// # let _mpi = Mpi::init().unwrap();
/// # let world = _mpi.world();
/// let dt8 = CustomDatatype::contiguous::<u8>(8).unwrap();
/// let mut boxes: [Pair; 1] = [Pair { v: 0.0, i: 0 }];
/// world.recv_custom(&mut boxes, &dt8, 0, 2).unwrap();
/// ```
pub unsafe trait PlainData: Copy + 'static {}
// SAFETY: every MpiDatatype is a primitive integer or float, and any bit pattern of its size is a valid value.
unsafe impl<T: MpiDatatype> PlainData for T {}
// SAFETY: an array of plain data is plain data — its bytes are exactly N back-to-back elements, each valid for any bit pattern.
unsafe impl<T: PlainData, const N: usize> PlainData for [T; N] {}

/// Describes a typed slice for MPI FFI as `(pointer, count, tag)`.
///
/// The pointer is valid for `s.len()` elements of `T` for as long as the
/// borrow of `s` lives (for [`buf_mut`], it is also valid for writes and
/// unaliased). The count is exact: a slice length never exceeds
/// `isize::MAX <= i64::MAX`. The tag is the MPI datatype that the sealed
/// `MpiDatatype` impl assigns to `T`, so MPI's element size and layout equal
/// `T`'s. The pointer is never null, so a NULL passed elsewhere as an in-place
/// marker stays unambiguous; see [`buf_ptr`] for an empty slice.
// PERF: with no inline attribute on `buf` and `buf_mut` (and `#[inline]` on
// `buf_ptr`), `isend` and `irecv` are no longer inlined into a caller's scope
// body.
#[inline(always)]
pub(crate) fn buf<T: MpiDatatype>(s: &[T]) -> (*const c_void, i64, i32) {
    (
        buf_ptr(s.as_ptr(), s.len(), false),
        s.len() as i64,
        tag_of::<T>() as i32,
    )
}

/// See [`buf`].
#[inline(always)]
pub(crate) fn buf_mut<T: MpiDatatype>(s: &mut [T]) -> (*mut c_void, i64, i32) {
    (
        buf_ptr(s.as_mut_ptr(), s.len(), true).cast_mut(),
        s.len() as i64,
        tag_of::<T>() as i32,
    )
}

/// The addresses MPI gets for a buffer of an alignment-1 type that holds no
/// bytes: the first for a buffer MPI reads, the second for one it writes.
static EMPTY: [u64; 2] = [0; 2];

/// `ptr`, or the address of a private static when the buffer holds no bytes
/// (`len` is zero or `T` is zero-sized) and `T` has alignment 1. Such a buffer
/// usually points at address 1, which is `MPI_IN_PLACE` in Open MPI and in the
/// MPI-5 ABI; a type with a larger alignment has an even address, and
/// `MPI_IN_PLACE` is an odd address there and in MPICH. `written` selects the
/// second address, so that a send and a receive buffer of one call never share
/// one: MPICH's `MPI_Alltoallv` and Open MPI 5's `MPI_Reduce_init` can refuse
/// two equal buffers whatever the count. MPI reads and writes nothing through
/// a buffer of count zero.
#[inline(always)]
pub(crate) fn buf_ptr<T>(ptr: *const T, len: usize, written: bool) -> *const c_void {
    if align_of::<T>() == 1 && (len == 0 || size_of::<T>() == 0) {
        std::ptr::from_ref(&EMPTY[usize::from(written)]).cast()
    } else {
        ptr.cast()
    }
}

macro_rules! impl_mpi_datatype {
    ($ty:ty, $tag:expr) => {
        impl sealed::Sealed for $ty {
            #[inline]
            fn tag(_: sealed::Token) -> DatatypeTag {
                $tag
            }
        }
        impl MpiDatatype for $ty {}
    };
}

impl_mpi_datatype!(f32, DatatypeTag::F32);
impl_mpi_datatype!(f64, DatatypeTag::F64);
impl_mpi_datatype!(i32, DatatypeTag::I32);
impl_mpi_datatype!(i64, DatatypeTag::I64);
impl_mpi_datatype!(u8, DatatypeTag::U8);
impl_mpi_datatype!(u32, DatatypeTag::U32);
impl_mpi_datatype!(u64, DatatypeTag::U64);

// ============================================================
// Paired value+index types for MPI_MAXLOC / MPI_MINLOC
// ============================================================

/// Trait for types that can be used with `MPI_MAXLOC` / `MPI_MINLOC`.
///
/// This is a **sealed trait** — it cannot be implemented outside this crate.
/// Only the six MPI predefined paired value+index types implement it.
///
/// Use these types exclusively with [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed) and
/// [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) / [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC). They are **not** valid for
/// `broadcast`, `send`, `recv`, or other collectives: they implement
/// [`MpiIndexedDatatype`], not [`MpiDatatype`], which those APIs require.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{CollectiveOp, Mpi, DoubleInt, MpiIndexedDatatype};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let rank = world.rank();
///
/// let send = [DoubleInt { value: rank as f64, index: rank }];
/// let mut recv = [DoubleInt { value: 0.0, index: 0 }];
/// world.allreduce_indexed(&send, &mut recv, CollectiveOp::MAX_LOC).unwrap();
/// // Every rank now holds { value: (size-1) as f64, index: size-1 }
/// ```
///
/// # Sealed: cannot implement for external types
///
/// ```compile_fail,E0277
/// use ferrompi::MpiIndexedDatatype;
///
/// #[derive(Clone, Copy)]
/// struct Mine;
/// // This must not compile — MpiIndexedDatatype is sealed.
/// impl MpiIndexedDatatype for Mine {}
/// ```
pub trait MpiIndexedDatatype: sealed_indexed::Sealed + Copy + Send + 'static {}

/// Paired `{ f32 value; i32 index }` — maps to `MPI_FLOAT_INT`.
///
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct FloatInt {
    /// The floating-point value to reduce.
    pub value: f32,
    /// The index (rank) associated with this value.
    pub index: i32,
}

/// Paired `{ f64 value; i32 index }` — maps to `MPI_DOUBLE_INT`.
///
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
///
/// Layout on 64-bit Linux: `sizeof == 16`, `alignof == 8` (4 bytes of trailing
/// padding after the `i32` index to satisfy the `f64` alignment of the next
/// element in an array).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DoubleInt {
    /// The double-precision value to reduce.
    pub value: f64,
    /// The index (rank) associated with this value.
    pub index: i32,
}

/// Paired `{ i64 value; i32 index }` — maps to `MPI_LONG_INT`.
///
/// On 64-bit Linux, C `long` is 8 bytes, so `value` is `i64`.
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
///
/// This type is available only on Linux x86_64, aarch64 and little-endian
/// powerpc64, where its layout matches the C `{ long; int }` struct. Other
/// targets are not supported in this release.
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct LongInt {
    /// The long integer value to reduce.
    pub value: i64,
    /// The index (rank) associated with this value.
    pub index: i32,
}

/// Paired `{ i32 value; i32 index }` — maps to `MPI_2INT`.
///
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
///
/// Layout: `sizeof == 8`, `alignof == 4`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Int2 {
    /// The integer value to reduce.
    pub value: i32,
    /// The index (rank) associated with this value.
    pub index: i32,
}

/// Paired `{ i16 value; i32 index }` — maps to `MPI_SHORT_INT`.
///
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
///
/// Layout on 64-bit Linux: `sizeof == 8`, `alignof == 4` (2 bytes of padding
/// between `i16` value and `i32` index to satisfy `i32` alignment).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct ShortInt {
    /// The short integer value to reduce.
    pub value: i16,
    /// The index (rank) associated with this value.
    pub index: i32,
}

/// Paired `{ long double value; i32 index }` — maps to `MPI_LONG_DOUBLE_INT`.
///
/// C `long double` on x86_64 Linux is 80-bit extended precision stored in
/// 16 bytes (with 6 bytes of padding). Rust has no native `f80` type, so
/// the value field is stored as `[u8; 16]` to match the C struct layout
/// exactly. Users working with this type should cast to/from the actual
/// `long double` representation via FFI if needed.
///
/// The `index` field conventionally holds the rank of the contributing process.
/// Use with [`CollectiveOp::MAX_LOC`](crate::CollectiveOp::MAX_LOC) or [`CollectiveOp::MIN_LOC`](crate::CollectiveOp::MIN_LOC) via
/// [`Communicator::allreduce_indexed`](crate::Communicator::allreduce_indexed).
///
/// This type is available only on Linux x86_64, aarch64 and little-endian
/// powerpc64, where its layout matches the C `{ long double; int }` struct.
/// Other targets are not supported in this release.
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(16))]
pub struct LongDoubleInt {
    /// Raw bytes of the `long double` value (platform-specific layout).
    pub value: [u8; 16],
    /// The index (rank) associated with this value.
    pub index: i32,
}

macro_rules! impl_mpi_indexed_datatype {
    ($ty:ty, $tag:expr) => {
        impl sealed_indexed::Sealed for $ty {
            #[inline]
            fn tag(_: sealed::Token) -> DatatypeTag {
                $tag
            }
        }
        impl MpiIndexedDatatype for $ty {}
    };
}

impl_mpi_indexed_datatype!(FloatInt, DatatypeTag::FloatInt);
impl_mpi_indexed_datatype!(DoubleInt, DatatypeTag::DoubleInt);
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
impl_mpi_indexed_datatype!(LongInt, DatatypeTag::LongInt);
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
const _: () = assert!(std::mem::size_of::<LongInt>() == 16 && std::mem::align_of::<LongInt>() == 8);
impl_mpi_indexed_datatype!(Int2, DatatypeTag::Int2);
impl_mpi_indexed_datatype!(ShortInt, DatatypeTag::ShortInt);
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
impl_mpi_indexed_datatype!(LongDoubleInt, DatatypeTag::LongDoubleInt);
#[cfg(all(
    target_os = "linux",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little")
    )
))]
const _: () = assert!(
    std::mem::size_of::<LongDoubleInt>() == 32 && std::mem::align_of::<LongDoubleInt>() == 16
);

// ============================================================
// Byte-permutable types for MPI_BYTE bitwise reductions
// ============================================================

/// Trait for types whose byte representation is valid for use with
/// `MPI_BYTE`-typed bitwise reductions.
///
/// This is a **sealed trait** — only types whose memory layout
/// consists entirely of meaningful bytes (no padding, no
/// discriminants, no uninhabited cases) can implement it. ferrompi
/// provides blanket impls for `u8`, `u16`, `u32`, `u64`, `i8`,
/// `i16`, `i32`, `i64`, plus `[T; N]` where `T: BytePermutable`.
///
/// # Safety invariants (enforced by sealing)
///
/// - No padding bytes inside the type (every byte is meaningful).
/// - No niche optimizations: every bit pattern is a valid value.
/// - `Copy + Send + 'static`.
///
/// # Sealed: cannot implement for external types
///
/// ```compile_fail,E0277
/// use ferrompi::BytePermutable;
///
/// #[derive(Clone, Copy)]
/// struct Mine;
/// // This must not compile — BytePermutable is sealed.
/// impl BytePermutable for Mine {}
/// ```
pub trait BytePermutable: sealed_byte::Sealed + Copy + Send + 'static {}

macro_rules! impl_byte_permutable {
    ($ty:ty) => {
        impl sealed_byte::Sealed for $ty {}
        impl BytePermutable for $ty {}
    };
}

impl_byte_permutable!(u8);
impl_byte_permutable!(u16);
impl_byte_permutable!(u32);
impl_byte_permutable!(u64);
impl_byte_permutable!(i8);
impl_byte_permutable!(i16);
impl_byte_permutable!(i32);
impl_byte_permutable!(i64);

impl<T: BytePermutable, const N: usize> sealed_byte::Sealed for [T; N] {}
impl<T: BytePermutable, const N: usize> BytePermutable for [T; N] {}

// ============================================================
// Atomic-eligible types for MPI_Compare_and_swap
// ============================================================

/// Trait for types eligible for use with `MPI_Compare_and_swap`.
///
/// This is a **sealed trait** — it cannot be implemented outside this crate.
/// MPI-4.1 §12.3.4 allows only integer, logical and byte datatypes in
/// `MPI_Compare_and_swap`; of ferrompi's element types these are `i32`, `i64`,
/// `u32`, `u64` and `u8`.
///
/// **Floating-point types (`f32`, `f64`) are intentionally excluded.** The
/// MPI standard does not allow them in compare-and-swap. Accepting them would
/// introduce silent portability bugs across MPI implementations.
///
/// This trait is used as a bound on
/// [`ActiveEpoch::compare_and_swap`](crate::ActiveEpoch::compare_and_swap) and
/// [`PassiveEpoch::compare_and_swap`](crate::PassiveEpoch::compare_and_swap)
/// to ensure the method only exists for windows whose element type is
/// CAS-eligible.
///
/// # Eligible types only
///
/// `compare_and_swap` exists only for integer and byte elements: on an `f64`
/// window the method is not found.
///
/// ```compile_fail,E0599
/// fn _check(e: &ferrompi::ActiveEpoch<'_, '_, f64>, v: &'static f64, r: &'static mut f64) -> ferrompi::Result<()> {
///     e.compare_and_swap(v, v, r, 0, 0)
/// }
/// ```
///
/// The same call on an `i64` window compiles:
///
/// ```no_run
/// fn _check(e: &ferrompi::ActiveEpoch<'_, '_, i64>, v: &'static i64, r: &'static mut i64) -> ferrompi::Result<()> {
///     e.compare_and_swap(v, v, r, 0, 0)
/// }
/// ```
#[cfg(feature = "rma")]
pub trait AtomicMpiDatatype: sealed_atomic::Sealed + Copy + Send + 'static {}

#[cfg(feature = "rma")]
macro_rules! impl_atomic_mpi_datatype {
    ($ty:ty) => {
        impl sealed_atomic::Sealed for $ty {}
        impl AtomicMpiDatatype for $ty {}
    };
}

#[cfg(feature = "rma")]
impl_atomic_mpi_datatype!(i32);
#[cfg(feature = "rma")]
impl_atomic_mpi_datatype!(i64);
#[cfg(feature = "rma")]
impl_atomic_mpi_datatype!(u32);
#[cfg(feature = "rma")]
impl_atomic_mpi_datatype!(u64);
#[cfg(feature = "rma")]
impl_atomic_mpi_datatype!(u8);

#[cfg(test)]
mod tests {
    #[cfg(feature = "rma")]
    use super::AtomicMpiDatatype;
    use super::{
        buf, buf_mut, buf_ptr, pair_tag_of, tag_of, BytePermutable, DatatypeTag, DoubleInt,
        FloatInt, Int2, PlainData, ShortInt,
    };
    #[cfg(all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "powerpc64", target_endian = "little")
        )
    ))]
    use super::{LongDoubleInt, LongInt};

    #[test]
    fn datatype_tag_values_match_c_defines() {
        assert_eq!(DatatypeTag::F32 as i32, 0);
        assert_eq!(DatatypeTag::F64 as i32, 1);
        assert_eq!(DatatypeTag::I32 as i32, 2);
        assert_eq!(DatatypeTag::I64 as i32, 3);
        assert_eq!(DatatypeTag::U8 as i32, 4);
        assert_eq!(DatatypeTag::U32 as i32, 5);
        assert_eq!(DatatypeTag::U64 as i32, 6);
        assert_eq!(DatatypeTag::FloatInt as i32, 7);
        assert_eq!(DatatypeTag::DoubleInt as i32, 8);
        assert_eq!(DatatypeTag::LongInt as i32, 9);
        assert_eq!(DatatypeTag::Int2 as i32, 10);
        assert_eq!(DatatypeTag::ShortInt as i32, 11);
        assert_eq!(DatatypeTag::LongDoubleInt as i32, 12);
        assert_eq!(DatatypeTag::Byte as i32, 13);

        assert_eq!(tag_of::<f32>() as i32, 0);
        assert_eq!(tag_of::<f64>() as i32, 1);
        assert_eq!(tag_of::<i32>() as i32, 2);
        assert_eq!(tag_of::<i64>() as i32, 3);
        assert_eq!(tag_of::<u8>() as i32, 4);
        assert_eq!(tag_of::<u32>() as i32, 5);
        assert_eq!(tag_of::<u64>() as i32, 6);
        assert_eq!(pair_tag_of::<FloatInt>() as i32, 7);
        assert_eq!(pair_tag_of::<DoubleInt>() as i32, 8);
        #[cfg(all(
            target_os = "linux",
            any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                all(target_arch = "powerpc64", target_endian = "little")
            )
        ))]
        assert_eq!(pair_tag_of::<LongInt>() as i32, 9);
        assert_eq!(pair_tag_of::<Int2>() as i32, 10);
        assert_eq!(pair_tag_of::<ShortInt>() as i32, 11);
        #[cfg(all(
            target_os = "linux",
            any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                all(target_arch = "powerpc64", target_endian = "little")
            )
        ))]
        assert_eq!(pair_tag_of::<LongDoubleInt>() as i32, 12);
    }

    #[test]
    fn empty_byte_buffer_is_passed_at_a_fixed_even_address() {
        let empty: Vec<u8> = Vec::new();
        assert_eq!(empty.as_ptr() as usize, 1);
        let (ptr, len, _) = buf(&empty);
        assert_eq!(len, 0);
        assert!(!ptr.is_null() && (ptr as usize) % 2 == 0);
        let mut empty_out: Vec<u8> = Vec::new();
        let out = buf_mut(&mut empty_out).0.cast_const();
        assert!(!out.is_null() && (out as usize) % 2 == 0 && out != ptr);
        let no_bytes = [[0u8; 0]; 3];
        assert_eq!(buf_ptr(no_bytes.as_ptr(), no_bytes.len(), false), ptr);
    }

    #[test]
    fn other_buffers_are_passed_at_their_own_address() {
        let bytes = [7u8; 3];
        assert_eq!(buf(&bytes).0, bytes.as_ptr().cast());
        let no_words: Vec<u64> = Vec::new();
        assert_eq!(buf(&no_words).0, no_words.as_ptr().cast());
    }

    #[test]
    fn byte_permutable_implemented_for_integer_primitives() {
        fn assert_byte_permutable<T: BytePermutable>() {}
        assert_byte_permutable::<u8>();
        assert_byte_permutable::<u16>();
        assert_byte_permutable::<u32>();
        assert_byte_permutable::<u64>();
        assert_byte_permutable::<i8>();
        assert_byte_permutable::<i16>();
        assert_byte_permutable::<i32>();
        assert_byte_permutable::<i64>();
        assert_byte_permutable::<[u64; 4]>();
    }

    #[test]
    fn plain_data_implemented_for_mpi_datatypes() {
        fn assert_plain<T: PlainData>() {}
        assert_plain::<f32>();
        assert_plain::<f64>();
        assert_plain::<i32>();
        assert_plain::<i64>();
        assert_plain::<u8>();
        assert_plain::<u32>();
        assert_plain::<u64>();
        assert_plain::<[f64; 3]>();
        assert_plain::<[[i32; 2]; 4]>();
    }

    #[test]
    fn indexed_datatype_struct_layouts() {
        use std::mem::{align_of, size_of};

        // FloatInt: { f32, i32 } — both 4-byte aligned, no padding
        // sizeof == 8, alignof == 4
        assert_eq!(size_of::<FloatInt>(), 8, "FloatInt size");
        assert_eq!(align_of::<FloatInt>(), 4, "FloatInt align");

        // DoubleInt: { f64, i32 } — f64 is 8-byte aligned, i32 is 4-byte;
        // 4 bytes trailing padding to align the next array element at 8 bytes.
        // sizeof == 16, alignof == 8 on x86_64/aarch64 Linux.
        assert!(
            size_of::<DoubleInt>() >= 12,
            "DoubleInt must hold at least f64 + i32"
        );
        assert!(
            align_of::<DoubleInt>() >= 8,
            "DoubleInt alignment must be at least f64 alignment"
        );
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert_eq!(
                size_of::<DoubleInt>(),
                16,
                "DoubleInt size on x86_64/aarch64"
            );
            assert_eq!(
                align_of::<DoubleInt>(),
                8,
                "DoubleInt align on x86_64/aarch64"
            );
        }

        // Int2: { i32, i32 } — no padding
        // sizeof == 8, alignof == 4
        assert_eq!(size_of::<Int2>(), 8, "Int2 size");
        assert_eq!(align_of::<Int2>(), 4, "Int2 align");

        // ShortInt: { i16, i32 } — 2 bytes padding between fields for i32 alignment
        // sizeof == 8, alignof == 4
        assert!(
            size_of::<ShortInt>() >= 6,
            "ShortInt must hold at least i16 + i32"
        );
        assert!(
            align_of::<ShortInt>() >= 4,
            "ShortInt alignment must be at least i32 alignment"
        );
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert_eq!(size_of::<ShortInt>(), 8, "ShortInt size on x86_64/aarch64");
            assert_eq!(
                align_of::<ShortInt>(),
                4,
                "ShortInt align on x86_64/aarch64"
            );
        }
    }

    #[cfg(feature = "rma")]
    #[test]
    fn atomic_mpi_datatype_implemented_for_integer_types() {
        fn assert_atomic<T: AtomicMpiDatatype>() {}
        assert_atomic::<i32>();
        assert_atomic::<i64>();
        assert_atomic::<u32>();
        assert_atomic::<u64>();
        assert_atomic::<u8>();
    }
}
