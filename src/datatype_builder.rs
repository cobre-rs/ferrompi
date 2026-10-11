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
//! use ferrompi::{CustomDatatype, Mpi};
//!
//! let _mpi = Mpi::init().unwrap();
//!
//! // Create a contiguous block of 10 f64 values.
//! let dt = CustomDatatype::contiguous::<f64>(10).unwrap();
//! // dt is freed when it goes out of scope.
//! ```

use crate::datatype::{tag_of, DatatypeTag, MpiDatatype};
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
/// use ferrompi::{CustomDatatype, Mpi, StructField};
///
/// let _mpi = Mpi::init().unwrap();
///
/// // Model a `{ f64, i32 }` C struct (8-byte f64 at offset 0, i32 at offset 8).
/// let dt = CustomDatatype::create_struct(&[
///     StructField::new::<f64>(1, 0),
///     StructField::new::<i32>(1, 8),
/// ]).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructField {
    /// Number of elements in this field's block.
    blocklength: i32,
    /// Byte displacement from the start of the struct.
    displacement: i64,
    basetype: DatatypeTag,
}

impl StructField {
    /// A block of `blocklength` elements of `T` at byte `displacement` from
    /// the start of the struct.
    pub fn new<T: MpiDatatype>(blocklength: i32, displacement: i64) -> Self {
        StructField {
            blocklength,
            displacement,
            basetype: tag_of::<T>(),
        }
    }
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
/// # Base types
///
/// The constructors take the base type as a type parameter, so only an
/// [`MpiDatatype`] can be one. A pair type is not:
///
/// ```compile_fail,E0277
/// let _ = ferrompi::CustomDatatype::contiguous::<ferrompi::FloatInt>(5);
/// ```
///
/// ```no_run
/// let _ = ferrompi::CustomDatatype::contiguous::<f32>(5);
/// ```
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

// SAFETY: CustomDatatype handles are integer indices into a C-side table.
// Every MPI call on a CustomDatatype goes through the lifecycle guard
// (`rt::enter`, and `rt::drop_guard` in Drop), which enforces the requested
// ThreadLevel as described on Communicator's Send impl in src/comm/mod.rs.
// The table slot is only mutated (freed) in Drop, which consumes the value.
unsafe impl Send for CustomDatatype {}
// SAFETY: &CustomDatatype exposes only reads of the immutable handle field
// and FFI calls gated by the same lifecycle guard, so its thread safety
// follows the requested ThreadLevel as Communicator's does.
unsafe impl Sync for CustomDatatype {}

impl CustomDatatype {
    /// Create a contiguous block of `count` elements of the base type `T`.
    ///
    /// Wraps `MPI_Type_contiguous` followed by `MPI_Type_commit`. The returned
    /// `CustomDatatype` is always committed and ready for use.
    ///
    /// # Arguments
    ///
    /// * `T` — the element type to replicate; see
    ///   [Base types](CustomDatatype#base-types).
    /// * `count` — number of elements; must be positive. MPI returns
    ///   `MPI_ERR_COUNT` (or `MPI_ERR_ARG` on some implementations) for
    ///   non-positive values.
    ///
    /// # Errors
    ///
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count)
    ///   or [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg) — `count` is non-positive.
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, Mpi};
    ///
    /// let _mpi = Mpi::init().unwrap();
    /// let dt = CustomDatatype::contiguous::<f64>(10).unwrap();
    /// ```
    pub fn contiguous<T: MpiDatatype>(count: i32) -> Result<Self> {
        let mut handle: i32 = -1;
        // SAFETY: all arguments are scalar integers; no pointer or lifetime invariant at stake.
        let ret =
            unsafe { ffi::ferrompi_type_contiguous(count, tag_of::<T>() as i32, &mut handle) };
        Error::check_with_op(ret, "type_contiguous")?;
        Self::from_committed(handle)
    }

    /// Build a strided datatype with `count` blocks of `blocklength` elements
    /// of the base type `T`, separated by `stride` elements between the start
    /// of consecutive blocks.
    ///
    /// Wraps `MPI_Type_vector` followed by `MPI_Type_commit`. The returned
    /// `CustomDatatype` is always committed and ready for use.
    ///
    /// # Arguments
    ///
    /// * `T` — the element type of each block; see
    ///   [Base types](CustomDatatype#base-types).
    /// * `count` — number of blocks. MPI returns `MPI_ERR_COUNT` (or
    ///   `MPI_ERR_ARG` on some implementations) for zero or negative values.
    /// * `blocklength` — number of base elements per block. Zero is accepted by
    ///   MPI and produces an empty-block type.
    /// * `stride` — number of base elements between the start of consecutive
    ///   blocks. May be negative (MPI allows reverse-direction strided types).
    ///
    /// # Errors
    ///
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count)
    ///   or [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg) — `count` is
    ///   zero or negative.
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, Mpi};
    ///
    /// let _mpi = Mpi::init().unwrap();
    /// // 3 blocks of 2 f64 elements, with a stride of 5 elements between
    /// // the start of consecutive blocks (useful for column extraction in a
    /// // 5-column row-major matrix).
    /// let dt = CustomDatatype::vector::<f64>(3, 2, 5).unwrap();
    /// ```
    pub fn vector<T: MpiDatatype>(count: i32, blocklength: i32, stride: i32) -> Result<Self> {
        let mut handle: i32 = -1;
        // SAFETY: all arguments are scalar integers; no pointer or lifetime invariant at stake.
        let ret = unsafe {
            ffi::ferrompi_type_vector(
                count,
                blocklength,
                stride,
                tag_of::<T>() as i32,
                &mut handle,
            )
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
    /// * `fields` — slice of [`StructField`] descriptors. Must be non-empty:
    ///   an empty slice returns [`Error::Mpi`] with class
    ///   [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg) without calling MPI.
    ///
    /// # Errors
    ///
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Arg`](crate::MpiErrorClass::Arg)
    ///   — `fields` is empty.
    /// - [`Error::ResourceExhausted`] with `resource: ResourceKind::Datatype`
    ///   — the internal datatype table is full (max 64 concurrent custom datatypes).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{CustomDatatype, Mpi, StructField};
    ///
    /// let _mpi = Mpi::init().unwrap();
    ///
    /// // Model a `{ f64, i32 }` C struct.
    /// let dt = CustomDatatype::create_struct(&[
    ///     StructField::new::<f64>(1, 0),
    ///     StructField::new::<i32>(1, 8),
    /// ]).unwrap();
    /// ```
    pub fn create_struct(fields: &[StructField]) -> Result<Self> {
        let mut blocklengths: Vec<i32> = Vec::with_capacity(fields.len());
        let mut displacements: Vec<i64> = Vec::with_capacity(fields.len());
        let mut basetype_tags: Vec<i32> = Vec::with_capacity(fields.len());
        for f in fields {
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
    /// use ferrompi::{CustomDatatype, Mpi, StructField};
    ///
    /// let _mpi = Mpi::init().unwrap();
    ///
    /// // Build a `{ f64, i32 }` struct type (natural extent = 12 bytes) and
    /// // resize it to a 16-byte stride for alignment-padded arrays.
    /// let s = CustomDatatype::create_struct(&[
    ///     StructField::new::<f64>(1, 0),
    ///     StructField::new::<i32>(1, 8),
    /// ]).unwrap();
    /// let r = s.resized(0, 16).unwrap();
    /// ```
    pub fn resized(&self, lb: i64, extent: i64) -> Result<Self> {
        let mut h: i32 = -1;
        // SAFETY: self.handle is owned and committed; lb and extent are scalar integers.
        let ret = unsafe { ffi::ferrompi_type_create_resized(self.handle, lb, extent, &mut h) };
        Error::check_with_op(ret, "type_create_resized")?;
        Self::from_committed(h)
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
    /// Returns [`Error::InvalidArgument`] naming `datatype` otherwise.
    pub(crate) fn check_layout<T>(&self) -> Result<()> {
        let size = std::mem::size_of::<T>() as i64;
        if self.extent != size {
            return Err(Error::InvalidArgument {
                arg: "datatype",
                reason: "extent differs from the element size",
            });
        }
        match self.true_lb.checked_add(self.true_extent) {
            Some(end) if self.true_lb >= 0 && end <= size => Ok(()),
            _ => Err(Error::InvalidArgument {
                arg: "datatype",
                reason: "data lies outside one element",
            }),
        }
    }
}

impl Drop for CustomDatatype {
    fn drop(&mut self) {
        if self.handle >= 0 {
            let Some(_call) = rt::drop_guard("CustomDatatype") else {
                return;
            };
            // SAFETY: handle is a valid index allocated by one of the
            // CustomDatatype constructors. We only free non-negative handles and do
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
    use super::CustomDatatype;
    use crate::error::Error;

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

        for (extent, true_lb, true_extent) in [(12, 0, 12), (32, 0, 12)] {
            let result = dt(extent, true_lb, true_extent).check_layout::<Sample>();
            assert!(
                matches!(
                    result,
                    Err(Error::InvalidArgument {
                        arg: "datatype",
                        reason: "extent differs from the element size"
                    })
                ),
                "extent {extent}: got {result:?}"
            );
        }

        for (true_lb, true_extent) in [(-1, 12), (8, 12), (i64::MAX, 1)] {
            let result = dt(16, true_lb, true_extent).check_layout::<Sample>();
            assert!(
                matches!(
                    result,
                    Err(Error::InvalidArgument {
                        arg: "datatype",
                        reason: "data lies outside one element"
                    })
                ),
                "true_lb {true_lb}, true_extent {true_extent}: got {result:?}"
            );
        }
    }
}
