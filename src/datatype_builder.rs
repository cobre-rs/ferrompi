//! Derived (custom) MPI datatype constructors.
//!
//! This module provides [`CustomDatatype`], a RAII handle to an `MPI_Datatype`
//! that was created by one of the derived-type constructors and is always
//! committed on return. The handle is freed automatically when the
//! [`CustomDatatype`] is dropped.
//!
//! # Supported constructors
//!
//! | Constructor | MPI function |
//! |-------------|-------------|
//! | [`CustomDatatype::contiguous`] | `MPI_Type_contiguous` + `MPI_Type_commit` |
//! | [`CustomDatatype::vector`] | `MPI_Type_vector` + `MPI_Type_commit` |
//! | [`CustomDatatype::create_struct`] | `MPI_Type_create_struct` + `MPI_Type_commit` |
//! | [`CustomDatatype::resized`] | `MPI_Type_create_resized` + `MPI_Type_commit` |
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::{CustomDatatype, DatatypeTag, Mpi};
//!
//! let _mpi = Mpi::init().unwrap();
//!
//! // Create a contiguous block of 10 f64 values.
//! let dt = CustomDatatype::contiguous(10, DatatypeTag::F64).unwrap();
//! assert!(dt.raw_handle() >= 0);
//! // dt is freed when it goes out of scope.
//! ```

use crate::datatype::DatatypeTag;
use crate::error::{Error, Result};
use crate::ffi;
use crate::rt;

/// One field of a struct-type passed to [`CustomDatatype::create_struct`].
///
/// Each field describes a single block of contiguous base elements within
/// a heterogeneous struct layout. The displacement is a byte offset from
/// the start of the struct (suitable for values computed with
/// `std::mem::offset_of!` or `memoffset::offset_of!`).
///
/// # Example
///
/// ```no_run
/// use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
///
/// let _mpi = Mpi::init().unwrap();
///
/// // Model a `{ f64, i32 }` C struct (8-byte f64 at offset 0, i32 at offset 8).
/// let dt = CustomDatatype::create_struct(&[
///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
/// ]).unwrap();
/// assert!(dt.raw_handle() >= 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructField {
    /// Number of elements in this field's block.
    pub blocklength: i32,
    /// Byte displacement from the start of the struct.
    pub displacement: i64,
    /// Base type for this field.
    pub basetype: DatatypeTag,
}

/// A committed derived MPI datatype backed by the C-side `datatype_table`.
///
/// `CustomDatatype` wraps an `MPI_Datatype` that has been created and committed
/// via one of the constructor methods. The underlying MPI handle is freed
/// automatically when the `CustomDatatype` is dropped.
///
/// # Extent and True Bounds
///
/// Each constructor records the datatype's extent and true lower bound/extent
/// at commit time (`MPI_Type_get_extent` / `MPI_Type_get_true_extent`). The
/// custom point-to-point methods in [`Communicator`](crate::Communicator)
/// (`send_custom`, `recv_custom`, `isend_custom`, `irecv_custom`) check these
/// cached values against the element type before issuing any MPI call.
///
/// # No `Clone`
///
/// Cloning is intentionally not supported: each `CustomDatatype` owns its
/// handle slot, and cloning would risk a double-free when both copies are
/// dropped. If you need to share a datatype, keep a single owned instance and
/// pass its handle by reference.
///
/// # Thread Safety
///
/// `CustomDatatype` is `Send + Sync` — the handle is an integer index into a
/// C-side table that is not mutated concurrently (the same threading contract
/// as [`Communicator`](crate::Communicator)).
#[derive(Debug)]
pub struct CustomDatatype {
    /// Index into the C-side `datatype_table`.
    pub(crate) handle: i32,
    /// `MPI_Type_get_extent` extent, in bytes, recorded at commit.
    extent: i64,
    /// `MPI_Type_get_true_extent` true lower bound, in bytes, recorded at commit.
    true_lb: i64,
    /// `MPI_Type_get_true_extent` true extent, in bytes, recorded at commit.
    true_extent: i64,
}

// SAFETY: CustomDatatype holds an integer handle into a C-side table.
// The MPI library manages its own thread safety based on the thread level
// requested via MPI_Init_thread. Sending a handle to another thread is safe
// for the same reasons as Communicator: the handle itself is an immutable
// index after construction, and MPI operations on it are safe at the
// appropriate thread level. The table slot is only mutated (freed) in Drop,
// which consumes the value — so there is no concurrent mutation risk.
unsafe impl Send for CustomDatatype {}
// SAFETY: &CustomDatatype exposes only reads of the immutable handle field and
// FFI calls whose concurrent use MPI governs by the initialized thread level;
// this type does not itself check that level.
unsafe impl Sync for CustomDatatype {}

impl CustomDatatype {
    /// Validate that `basetype` is a primitive type accepted by derived-type
    /// constructors in this family.
    ///
    /// Returns `Ok(())` for the seven numeric primitives (`F32`, `F64`, `I32`,
    /// `I64`, `U8`, `U32`, `U64`) and `Byte`. Returns [`Error::InvalidOp`] for
    /// the indexed paired types (`FloatInt`, `DoubleInt`, etc.), which are
    /// outside the v1 scope of the CustomDatatype builder family.
    fn validate_primitive_basetype(basetype: DatatypeTag) -> Result<()> {
        match basetype {
            DatatypeTag::FloatInt
            | DatatypeTag::DoubleInt
            | DatatypeTag::LongInt
            | DatatypeTag::Int2
            | DatatypeTag::ShortInt
            | DatatypeTag::LongDoubleInt => Err(Error::InvalidOp),
            DatatypeTag::F32
            | DatatypeTag::F64
            | DatatypeTag::I32
            | DatatypeTag::I64
            | DatatypeTag::U8
            | DatatypeTag::U32
            | DatatypeTag::U64
            | DatatypeTag::Byte => Ok(()),
        }
    }

    /// Create a contiguous block of `count` elements of the given `basetype`.
    ///
    /// Wraps `MPI_Type_contiguous` followed by `MPI_Type_commit`. The returned
    /// `CustomDatatype` is always committed and ready for use.
    ///
    /// # Arguments
    ///
    /// * `count` — number of elements; must be positive. MPI returns
    ///   `MPI_ERR_COUNT` (or `MPI_ERR_ARG` on some implementations) for
    ///   non-positive values.
    /// * `basetype` — the predefined primitive datatype to replicate. Must be
    ///   one of `F32`, `F64`, `I32`, `I64`, `U8`, `U32`, `U64`, or `Byte`.
    ///   Passing an indexed type (`FloatInt`, `DoubleInt`, etc.) returns
    ///   [`Error::InvalidOp`] without invoking MPI.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOp`] — `basetype` is an indexed paired type.
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count)
    ///   or [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg) — `count` is non-positive.
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, DatatypeTag, Mpi};
    ///
    /// let _mpi = Mpi::init().unwrap();
    /// let dt = CustomDatatype::contiguous(10, DatatypeTag::F64).unwrap();
    /// assert!(dt.raw_handle() >= 0);
    /// ```
    pub fn contiguous(count: i32, basetype: DatatypeTag) -> Result<Self> {
        Self::validate_primitive_basetype(basetype)?;

        let mut handle: i32 = -1;
        // SAFETY: all arguments are scalar integers; no pointer or lifetime invariant at stake.
        let ret = unsafe { ffi::ferrompi_type_contiguous(count, basetype as i32, &mut handle) };
        Error::check_with_op(ret, "type_contiguous")?;
        Self::from_committed(handle)
    }

    /// Build a strided datatype with `count` blocks of `blocklength` base
    /// elements, separated by `stride` base elements between the start of
    /// consecutive blocks.
    ///
    /// Wraps `MPI_Type_vector` followed by `MPI_Type_commit`. The returned
    /// `CustomDatatype` is always committed and ready for use.
    ///
    /// # Arguments
    ///
    /// * `count` — number of blocks. MPI returns `MPI_ERR_COUNT` (or
    ///   `MPI_ERR_ARG` on some implementations) for zero or negative values.
    /// * `blocklength` — number of base elements per block. Zero is accepted by
    ///   MPI and produces an empty-block type.
    /// * `stride` — number of base elements between the start of consecutive
    ///   blocks. May be negative (MPI allows reverse-direction strided types).
    /// * `basetype` — the predefined primitive datatype. Must be one of `F32`,
    ///   `F64`, `I32`, `I64`, `U8`, `U32`, `U64`, or `Byte`. Passing an indexed
    ///   type (`FloatInt`, `DoubleInt`, etc.) returns [`Error::InvalidOp`]
    ///   without invoking MPI.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOp`] — `basetype` is an indexed paired type.
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count)
    ///   or [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg) — `count` is
    ///   zero or negative.
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, DatatypeTag, Mpi};
    ///
    /// let _mpi = Mpi::init().unwrap();
    /// // 3 blocks of 2 f64 elements, with a stride of 5 elements between
    /// // the start of consecutive blocks (useful for column extraction in a
    /// // 5-column row-major matrix).
    /// let dt = CustomDatatype::vector(3, 2, 5, DatatypeTag::F64).unwrap();
    /// assert!(dt.raw_handle() >= 0);
    /// ```
    pub fn vector(
        count: i32,
        blocklength: i32,
        stride: i32,
        basetype: DatatypeTag,
    ) -> Result<Self> {
        Self::validate_primitive_basetype(basetype)?;

        let mut handle: i32 = -1;
        // SAFETY: all arguments are scalar integers; no pointer or lifetime invariant at stake.
        let ret = unsafe {
            ffi::ferrompi_type_vector(count, blocklength, stride, basetype as i32, &mut handle)
        };
        Error::check_with_op(ret, "type_vector")?;
        Self::from_committed(handle)
    }

    /// Build a heterogeneous struct derived datatype from a slice of field descriptors.
    ///
    /// Each [`StructField`] specifies the block length, byte displacement, and
    /// base type for one field. This wraps `MPI_Type_create_struct` followed by
    /// `MPI_Type_commit`. The returned `CustomDatatype` is always committed and
    /// ready for use.
    ///
    /// # Arguments
    ///
    /// * `fields` — slice of [`StructField`] descriptors. Must be non-empty for
    ///   success (MPI requires `count >= 1`). Each field's `basetype` must be one
    ///   of the primitive types (`F32`, `F64`, `I32`, `I64`, `U8`, `U32`, `U64`,
    ///   `Byte`). Passing an indexed type (`FloatInt`, `DoubleInt`, etc.) returns
    ///   [`Error::InvalidOp`] before invoking MPI.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidOp`] — any field has an indexed paired basetype.
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg)
    ///   or [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) — `fields` is
    ///   empty (MPI requires at least one field).
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    ///
    /// let _mpi = Mpi::init().unwrap();
    ///
    /// // Model a `{ f64, i32 }` C struct.
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// assert!(dt.raw_handle() >= 0);
    /// ```
    pub fn create_struct(fields: &[StructField]) -> Result<Self> {
        let mut blocklengths: Vec<i32> = Vec::with_capacity(fields.len());
        let mut displacements: Vec<i64> = Vec::with_capacity(fields.len());
        let mut basetype_tags: Vec<i32> = Vec::with_capacity(fields.len());
        for f in fields {
            Self::validate_primitive_basetype(f.basetype)?;
            blocklengths.push(f.blocklength);
            displacements.push(f.displacement);
            basetype_tags.push(f.basetype as i32);
        }
        let mut h: i32 = -1;
        // SAFETY: the three Vec pointers are valid for fields.len() elements each and live
        // for the duration of this call; there is no aliasing between the output pointer and
        // the input slices.
        let ret = unsafe {
            ffi::ferrompi_type_create_struct(
                fields.len() as i32,
                blocklengths.as_ptr(),
                displacements.as_ptr(),
                basetype_tags.as_ptr(),
                &mut h,
            )
        };
        Error::check_with_op(ret, "type_create_struct")?;
        Self::from_committed(h)
    }

    /// Build a new datatype with the same payload as `self` but with the
    /// specified lower bound and extent (in bytes).
    ///
    /// Wraps `MPI_Type_create_resized` followed by `MPI_Type_commit`. The
    /// returned `CustomDatatype` is always committed and ready for use. `self`
    /// is not consumed; both the original and the resized type remain valid
    /// and independently owned.
    ///
    /// # Arguments
    ///
    /// * `lb` — new lower bound in bytes. Typically `0`.
    /// * `extent` — total byte extent between consecutive elements of an array.
    ///   MPI returns `MPI_ERR_ARG` for negative values.
    ///
    /// # Common Use Case
    ///
    /// Fix an extent mismatch when an array of `#[repr(C)]` structs has
    /// natural padding-to-alignment that MPI's auto-computed extent does not
    /// match (common when the struct has interior padding but no trailing
    /// padding-to-alignment).
    ///
    /// # Errors
    ///
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg)
    ///   — `extent` is negative (implementation-defined; some MPI stacks may
    ///   return a different class).
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, DatatypeTag, Mpi, StructField};
    ///
    /// let _mpi = Mpi::init().unwrap();
    ///
    /// // Build a `{ f64, i32 }` struct type (natural extent = 12 bytes) and
    /// // resize it to a 16-byte stride for alignment-padded arrays.
    /// let s = CustomDatatype::create_struct(&[
    ///     StructField { blocklength: 1, displacement: 0, basetype: DatatypeTag::F64 },
    ///     StructField { blocklength: 1, displacement: 8, basetype: DatatypeTag::I32 },
    /// ]).unwrap();
    /// let r = s.resized(0, 16).unwrap();
    /// assert!(r.raw_handle() >= 0);
    /// assert_ne!(r.raw_handle(), s.raw_handle());
    /// ```
    pub fn resized(&self, lb: i64, extent: i64) -> Result<Self> {
        let mut h: i32 = -1;
        // SAFETY: self.handle is owned and committed; lb and extent are scalar integers.
        let ret = unsafe { ffi::ferrompi_type_create_resized(self.handle, lb, extent, &mut h) };
        Error::check_with_op(ret, "type_create_resized")?;
        Self::from_committed(h)
    }

    /// Return the raw integer handle for this datatype.
    ///
    /// The handle is an index into the C-side `datatype_table`. It is always
    /// `>= 0` for a successfully constructed `CustomDatatype`.
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }

    /// Build a `CustomDatatype` from a just-committed handle, caching its
    /// extent and true bounds via `ferrompi_type_get_extents`.
    ///
    /// On failure, the zero-extent value built first is dropped, freeing
    /// `handle` through [`Drop`].
    fn from_committed(handle: i32) -> Result<Self> {
        let mut dt = CustomDatatype {
            handle,
            extent: 0,
            true_lb: 0,
            true_extent: 0,
        };
        // SAFETY: handle was just committed by the caller; extent/true_lb/true_extent
        // are exclusive stack locations valid for the duration of this call.
        let ret = unsafe {
            ffi::ferrompi_type_get_extents(
                handle,
                &mut dt.extent,
                &mut dt.true_lb,
                &mut dt.true_extent,
            )
        };
        Error::check_with_op(ret, "type_get_extents")?;
        Ok(dt)
    }

    /// Check that this datatype's extent equals `size_of::<T>()` and that its
    /// true bounds lie within one `T`.
    ///
    /// Returns [`Error::InvalidBuffer`] otherwise.
    pub(crate) fn check_layout<T>(&self) -> Result<()> {
        let size = std::mem::size_of::<T>() as i64;
        if self.extent != size || self.true_lb < 0 {
            return Err(Error::InvalidBuffer);
        }
        let end = self
            .true_lb
            .checked_add(self.true_extent)
            .ok_or(Error::InvalidBuffer)?;
        if end > size {
            return Err(Error::InvalidBuffer);
        }
        Ok(())
    }
}

impl Drop for CustomDatatype {
    fn drop(&mut self) {
        if self.handle >= 0 {
            if !rt::drop_guard("CustomDatatype") {
                return;
            }
            // SAFETY: handle is a valid index allocated by ferrompi_type_contiguous
            // (or a future constructor). We only free non-negative handles and do
            // not use the handle after this point. The return value is intentionally
            // ignored: Drop must not panic, and an MPI error during type free is
            // non-recoverable at this point.
            unsafe {
                ffi::ferrompi_type_free(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CustomDatatype, StructField};
    use crate::datatype::DatatypeTag;
    use crate::error::Error;

    /// Calling `contiguous` with an indexed basetype returns `Error::InvalidOp`
    /// without invoking any FFI. This test does not require an MPI runtime.
    #[test]
    fn contiguous_rejects_indexed_basetype() {
        let result = CustomDatatype::contiguous(10, DatatypeTag::FloatInt);
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "expected Err(Error::InvalidOp), got: {:?}",
            result
        );
    }

    /// Calling `vector` with an indexed basetype returns `Error::InvalidOp`
    /// without invoking any FFI. This test does not require an MPI runtime.
    #[test]
    fn vector_rejects_indexed_basetype() {
        let result = CustomDatatype::vector(3, 2, 5, DatatypeTag::FloatInt);
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "expected Err(Error::InvalidOp), got: {:?}",
            result
        );
    }

    /// Calling `create_struct` with a field whose basetype is an indexed paired
    /// type returns `Error::InvalidOp` without invoking any FFI. This test does
    /// not require an MPI runtime.
    #[test]
    fn create_struct_rejects_indexed_basetype() {
        let result = CustomDatatype::create_struct(&[StructField {
            blocklength: 1,
            displacement: 0,
            basetype: DatatypeTag::FloatInt,
        }]);
        assert!(
            matches!(result, Err(Error::InvalidOp)),
            "expected Err(Error::InvalidOp), got: {:?}",
            result
        );
    }

    /// `check_layout` boundaries against a 16-byte `#[repr(C)]` type, using
    /// inert (`handle: -1`) `CustomDatatype` values that need no MPI runtime.
    #[test]
    fn check_layout_boundaries() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Sample {
            v: f64,
            i: i32,
        }
        assert_eq!(std::mem::size_of::<Sample>(), 16);

        let dt = |extent: i64, true_lb: i64, true_extent: i64| CustomDatatype {
            handle: -1,
            extent,
            true_lb,
            true_extent,
        };

        assert!(dt(16, 0, 12).check_layout::<Sample>().is_ok());
        assert!(dt(16, 0, 16).check_layout::<Sample>().is_ok());
        assert!(dt(12, 0, 12).check_layout::<Sample>().is_err());
        assert!(dt(32, 0, 12).check_layout::<Sample>().is_err());
        assert!(dt(16, -1, 12).check_layout::<Sample>().is_err());
        assert!(dt(16, 8, 12).check_layout::<Sample>().is_err());
        assert!(dt(16, i64::MAX, 1).check_layout::<Sample>().is_err());
    }
}
