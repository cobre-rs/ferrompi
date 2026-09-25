//! MPI Group handle for rank-set operations.
//!
//! [`Group`] wraps an `MPI_Group` handle via the C-side group table and
//! provides methods for querying group size and rank. Use
//! [`Communicator::group`](crate::Communicator::group) to obtain a `Group`
//! from an existing communicator, then call [`include`](Group::include) or
//! [`exclude`](Group::exclude) to derive sub-groups.
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::Mpi;
//!
//! let mpi = Mpi::init().unwrap();
//! let world = mpi.world();
//!
//! let world_group = world.group().unwrap();
//! assert_eq!(world_group.size().unwrap(), world.size());
//!
//! // Include only ranks 0 and 2
//! let sub = world_group.include(&[0, 2]).unwrap();
//! assert_eq!(sub.size().unwrap(), 2);
//! ```

use crate::error::{Error, Result};
use crate::ffi;
use crate::Communicator;

/// Outcome of [`Group::compare`].
///
/// Mirrors `MPI_Group_compare`'s three possible result codes:
/// `MPI_IDENT`, `MPI_SIMILAR`, `MPI_UNEQUAL`.
///
/// The discriminant values (0, 1, 2) are ferrompi-stable and do **not**
/// correspond directly to the MPI implementation's integer constants, which
/// differ between MPICH and Open MPI. Normalisation is performed in the C shim
/// so that this enum can use `#[repr(i32)]` with fixed discriminants.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{GroupComparison, Mpi};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let gw = world.group().unwrap();
///
/// // Same object → Identical
/// assert_eq!(gw.compare(&gw).unwrap(), GroupComparison::Identical);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GroupComparison {
    /// `MPI_IDENT` — same group object (identity, not just rank equality).
    Identical = 0,
    /// `MPI_SIMILAR` — same rank set, possibly different ordering.
    Similar = 1,
    /// `MPI_UNEQUAL` — different rank sets.
    Unequal = 2,
}

/// Inclusive rank range `[first, last]` with positive stride.
///
/// Used as input to [`Group::range_include`] and
/// [`Group::range_exclude`] for compact specification of arithmetic
/// progressions over rank ids.
///
/// Maps to a single row of the `ranges[3][N]` array argument of
/// `MPI_Group_range_incl` / `MPI_Group_range_excl`.
///
/// # Example
///
/// ```no_run
/// use ferrompi::{Mpi, RankRange};
///
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
/// let gw = world.group().unwrap();
///
/// // Include ranks 0, 2 (every other rank starting from 0, up to 3)
/// let sub = gw.range_include(&[RankRange { first: 0, last: 3, stride: 2 }]).unwrap();
/// assert_eq!(sub.size().unwrap(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankRange {
    /// First rank in the range (inclusive).
    pub first: i32,
    /// Last rank in the range (inclusive).
    pub last: i32,
    /// Step between consecutive ranks (must be positive).
    pub stride: i32,
}

/// An MPI group handle.
///
/// This type wraps an `MPI_Group` handle with RAII semantics: the underlying
/// MPI group object is freed automatically when the `Group` is dropped,
/// unless it is slot 0 (reserved for `MPI_GROUP_EMPTY`).
///
/// Obtain a `Group` from a communicator via
/// [`Communicator::group`](crate::Communicator::group), then use
/// [`include`](Self::include) or [`exclude`](Self::exclude) to derive
/// sub-groups.
///
/// # MPI_UNDEFINED
///
/// [`rank`](Self::rank) returns `-1` when the calling process is not a
/// member of the group (`MPI_UNDEFINED` in the MPI standard). Consumers
/// must handle this sentinel explicitly.
pub struct Group {
    pub(crate) handle: i32,
}

// SAFETY: Group handles are integer indices into a C-side table, identical
// in nature to Communicator handles. The C MPI library manages its own
// thread safety based on the thread level requested via MPI_Init_thread.
// Sending a Group to another thread is safe under the same conditions as
// Communicator (see src/comm/mod.rs); users must ensure adequate thread
// support and serialize access when using ThreadLevel::Serialized.
unsafe impl Send for Group {}
// SAFETY: &Group exposes only reads of the immutable handle field and FFI
// calls whose concurrent use MPI governs by the initialized thread level;
// this type does not itself check that level.
unsafe impl Sync for Group {}

impl Group {
    /// Get the number of processes in this group.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let g = world.group().unwrap();
    /// assert_eq!(g.size().unwrap(), world.size());
    /// ```
    pub fn size(&self) -> Result<i32> {
        let mut size: i32 = 0;
        // SAFETY: self.handle is owned by this Group and was allocated by the C-side group table.
        let ret = unsafe { ffi::ferrompi_group_size(self.handle, &mut size) };
        Error::check_with_op(ret, "group_size")?;
        Ok(size)
    }

    /// Get the rank of the calling process in this group.
    ///
    /// Returns `MPI_UNDEFINED` (`-1`) when the calling process is not a member
    /// of the group. Consumers must handle this sentinel explicitly.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying MPI call fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let g = world.group().unwrap();
    /// assert_eq!(g.rank().unwrap(), world.rank());
    /// ```
    pub fn rank(&self) -> Result<i32> {
        let mut rank: i32 = 0;
        // SAFETY: self.handle is owned by this Group and was allocated by the C-side group table.
        let ret = unsafe { ffi::ferrompi_group_rank(self.handle, &mut rank) };
        Error::check_with_op(ret, "group_rank")?;
        Ok(rank)
    }

    /// Get the raw group handle for internal use.
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }

    /// Return the normalised `MPI_UNDEFINED` sentinel value (`-1`).
    ///
    /// [`rank`](Self::rank) returns `-1` when the calling process is not a
    /// member of the group.
    ///
    /// This function does not require an active MPI session.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Group, Mpi};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let sub = world.group().unwrap().include(&[0, 2]).unwrap();
    /// let r = sub.rank().unwrap();
    /// if r == Group::undefined() {
    ///     println!("not a member of this sub-group");
    /// }
    /// ```
    pub const fn undefined() -> i32 {
        Communicator::UNDEFINED
    }

    /// Create a new group containing only the specified ranks from this group.
    ///
    /// The resulting group contains the processes from this group whose ranks
    /// appear in the `ranks` slice, in the order given.
    ///
    /// # Arguments
    ///
    /// * `ranks` - Slice of rank indices from this group to include.
    ///
    /// # Errors
    ///
    /// Returns an error if any rank in `ranks` is out of range for this group,
    /// or if the C-side group table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let sub = world.group().unwrap().include(&[0, 2]).unwrap();
    /// assert_eq!(sub.size().unwrap(), 2);
    /// ```
    pub fn include(&self, ranks: &[i32]) -> Result<Group> {
        let mut new_handle: i32 = 0;
        // SAFETY: self.handle is owned; ranks.as_ptr() is valid for ranks.len() elements
        // for the duration of this call.
        let ret = unsafe {
            ffi::ferrompi_group_incl(
                self.handle,
                ranks.len() as i32,
                ranks.as_ptr(),
                &mut new_handle,
            )
        };
        Error::check_with_op(ret, "group_incl")?;
        Ok(Group { handle: new_handle })
    }

    /// Create a new group that is the set-theoretic union of this group and `other`.
    ///
    /// The resulting group contains all ranks from this group (in their order),
    /// followed by ranks from `other` that are not already in this group.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying MPI call fails or the C-side group
    /// table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    /// let g1 = gw.include(&[0, 1]).unwrap();
    /// let g2 = gw.include(&[1, 2]).unwrap();
    /// assert_eq!(g1.union(&g2).unwrap().size().unwrap(), 3);
    /// ```
    pub fn union(&self, other: &Group) -> Result<Group> {
        let mut h: i32 = -1;
        // SAFETY: both self.handle and other.handle are owned by their respective Groups.
        let ret = unsafe { ffi::ferrompi_group_union(self.handle, other.handle, &mut h) };
        Error::check_with_op(ret, "group_union")?;
        Ok(Group { handle: h })
    }

    /// Create a new group that is the set-theoretic intersection of this group and `other`.
    ///
    /// The resulting group contains ranks present in both groups, ordered as in this group.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying MPI call fails or the C-side group
    /// table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    /// let g1 = gw.include(&[0, 1]).unwrap();
    /// let g2 = gw.include(&[1, 2]).unwrap();
    /// assert_eq!(g1.intersection(&g2).unwrap().size().unwrap(), 1);
    /// ```
    pub fn intersection(&self, other: &Group) -> Result<Group> {
        let mut h: i32 = -1;
        // SAFETY: both self.handle and other.handle are owned by their respective Groups.
        let ret = unsafe { ffi::ferrompi_group_intersection(self.handle, other.handle, &mut h) };
        Error::check_with_op(ret, "group_intersection")?;
        Ok(Group { handle: h })
    }

    /// Create a new group that is the set-theoretic difference of this group minus `other`.
    ///
    /// The resulting group contains ranks in this group that are not in `other`,
    /// ordered as in this group.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying MPI call fails or the C-side group
    /// table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    /// let g1 = gw.include(&[0, 1]).unwrap();
    /// let g2 = gw.include(&[1, 2]).unwrap();
    /// assert_eq!(g1.difference(&g2).unwrap().size().unwrap(), 1);
    /// ```
    pub fn difference(&self, other: &Group) -> Result<Group> {
        let mut h: i32 = -1;
        // SAFETY: both self.handle and other.handle are owned by their respective Groups.
        let ret = unsafe { ffi::ferrompi_group_difference(self.handle, other.handle, &mut h) };
        Error::check_with_op(ret, "group_difference")?;
        Ok(Group { handle: h })
    }

    /// Compare this group against another.
    ///
    /// Returns:
    /// - [`GroupComparison::Identical`] if `self` and `other` are the same MPI
    ///   group object (object identity, not just rank equality).
    /// - [`GroupComparison::Similar`] if both groups contain the same set of
    ///   processes but in a different order.
    /// - [`GroupComparison::Unequal`] if the rank sets differ.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `MPI_Group_compare` call fails.
    /// An unexpected return value from MPI (which should never happen on a
    /// conforming implementation) is surfaced via `MPI_ERR_INTERN` from the
    /// C shim and maps to [`Error::Mpi`] with class `Intern`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{GroupComparison, Mpi};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    /// let g1 = gw.include(&[0, 1, 2]).unwrap();
    /// let g2 = gw.include(&[0, 1, 2]).unwrap();
    ///
    /// // Same rank set, different objects → Similar
    /// assert_eq!(g1.compare(&g2).unwrap(), GroupComparison::Similar);
    ///
    /// // Same object → Identical
    /// assert_eq!(gw.compare(&gw).unwrap(), GroupComparison::Identical);
    /// ```
    pub fn compare(&self, other: &Group) -> Result<GroupComparison> {
        let mut result: i32 = -1;
        // SAFETY: both self.handle and other.handle are owned by their respective Groups.
        let ret = unsafe { ffi::ferrompi_group_compare(self.handle, other.handle, &mut result) };
        Error::check_with_op(ret, "group_compare")?;
        match result {
            0 => Ok(GroupComparison::Identical),
            1 => Ok(GroupComparison::Similar),
            2 => Ok(GroupComparison::Unequal),
            other => Err(Error::Internal(format!(
                "ferrompi_group_compare returned unexpected result {other}"
            ))),
        }
    }

    /// Create a new group containing all ranks from this group except those
    /// in the `ranks` slice.
    ///
    /// # Arguments
    ///
    /// * `ranks` - Slice of rank indices from this group to exclude.
    ///
    /// # Errors
    ///
    /// Returns an error if any rank in `ranks` is out of range for this group,
    /// or if the C-side group table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let sub = world.group().unwrap().exclude(&[1, 3]).unwrap();
    /// assert_eq!(sub.size().unwrap(), 2); // 4-rank world minus ranks 1 and 3
    /// ```
    pub fn exclude(&self, ranks: &[i32]) -> Result<Group> {
        let mut new_handle: i32 = 0;
        // SAFETY: self.handle is owned; ranks.as_ptr() is valid for ranks.len() elements
        // for the duration of this call.
        let ret = unsafe {
            ffi::ferrompi_group_excl(
                self.handle,
                ranks.len() as i32,
                ranks.as_ptr(),
                &mut new_handle,
            )
        };
        Error::check_with_op(ret, "group_excl")?;
        Ok(Group { handle: new_handle })
    }

    /// Create a new group containing the union of all rank progressions
    /// described by the `ranges` triples.
    ///
    /// Each [`RankRange`] specifies an arithmetic progression
    /// `first, first+stride, ..., last` of ranks to include from this group.
    /// The result group contains those ranks in the order they appear across
    /// the triples.
    ///
    /// An empty `ranges` slice produces an empty group (`size() == 0`).
    ///
    /// # Errors
    ///
    /// Returns an error if MPI validation rejects any triple (e.g., negative
    /// or zero stride, `first > last` with positive stride), or if the
    /// C-side group table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, RankRange};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    ///
    /// // Include ranks 0, 1, 2
    /// let sub = gw.range_include(&[RankRange { first: 0, last: 2, stride: 1 }]).unwrap();
    /// assert_eq!(sub.size().unwrap(), 3);
    /// ```
    pub fn range_include(&self, ranges: &[RankRange]) -> Result<Group> {
        let mut flat: Vec<i32> = Vec::with_capacity(3 * ranges.len());
        for r in ranges {
            flat.push(r.first);
            flat.push(r.last);
            flat.push(r.stride);
        }
        let mut h: i32 = -1;
        // SAFETY: self.handle is owned; flat is a contiguous Vec<i32> with 3*ranges.len() elements
        // and outlives this call.
        let ret = unsafe {
            ffi::ferrompi_group_range_incl(self.handle, ranges.len() as i32, flat.as_ptr(), &mut h)
        };
        Error::check_with_op(ret, "group_range_incl")?;
        Ok(Group { handle: h })
    }

    /// Create a new group that is this group minus the union of all rank
    /// progressions described by the `ranges` triples.
    ///
    /// Each [`RankRange`] specifies an arithmetic progression of ranks to
    /// remove from this group. The result contains the remaining ranks in
    /// their original order.
    ///
    /// An empty `ranges` slice produces a copy of the parent group.
    ///
    /// # Errors
    ///
    /// Returns an error if MPI validation rejects any triple (e.g., negative
    /// or zero stride, `first > last` with positive stride), or if the
    /// C-side group table is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::{Mpi, RankRange};
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    ///
    /// // Exclude ranks 1 and 2 from a 4-rank world → {0, 3}
    /// let sub = gw.range_exclude(&[RankRange { first: 1, last: 2, stride: 1 }]).unwrap();
    /// assert_eq!(sub.size().unwrap(), 2);
    /// ```
    pub fn range_exclude(&self, ranges: &[RankRange]) -> Result<Group> {
        let mut flat: Vec<i32> = Vec::with_capacity(3 * ranges.len());
        for r in ranges {
            flat.push(r.first);
            flat.push(r.last);
            flat.push(r.stride);
        }
        let mut h: i32 = -1;
        // SAFETY: self.handle is owned; flat is a contiguous Vec<i32> with 3*ranges.len() elements
        // and outlives this call.
        let ret = unsafe {
            ffi::ferrompi_group_range_excl(self.handle, ranges.len() as i32, flat.as_ptr(), &mut h)
        };
        Error::check_with_op(ret, "group_range_excl")?;
        Ok(Group { handle: h })
    }

    /// Translate `ranks` from `self`'s rank space into `other`'s rank space.
    ///
    /// Returns a `Vec<Option<i32>>` of the same length as `ranks`:
    /// `Some(rank_in_other)` for ranks present in `other`, `None` for ranks
    /// not present in `other` (`MPI_UNDEFINED` entries normalised by the C shim
    /// to `-1` before Rust sees them).
    ///
    /// # Arguments
    ///
    /// * `ranks` — Input ranks, all referring to `self`'s rank space. Values may
    ///   be any non-negative `i32`; out-of-range values surface as `MPI_ERR_RANK`
    ///   from MPI.
    /// * `other` — Target rank space.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying `MPI_Group_translate_ranks` call fails
    /// (e.g. `MPI_ERR_RANK` for an out-of-range input rank).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Mpi;
    ///
    /// let mpi = Mpi::init().unwrap();
    /// let world = mpi.world();
    /// let gw = world.group().unwrap();
    /// let gsub = gw.include(&[1, 3]).unwrap();
    ///
    /// // World ranks 0 and 2 are not in gsub → None; ranks 1 and 3 map to 0 and 1.
    /// let translated = gw.translate_ranks(&[0, 1, 2, 3], &gsub).unwrap();
    /// assert_eq!(translated, vec![None, Some(0), None, Some(1)]);
    /// ```
    pub fn translate_ranks(&self, ranks: &[i32], other: &Group) -> Result<Vec<Option<i32>>> {
        if ranks.is_empty() {
            return Ok(vec![]);
        }
        let mut out: Vec<i32> = vec![-1; ranks.len()];
        // SAFETY: self.handle and other.handle are owned; ranks.as_ptr() and out.as_mut_ptr() are
        // valid for ranks.len() elements each with no aliasing between them.
        let ret = unsafe {
            ffi::ferrompi_group_translate_ranks(
                self.handle,
                ranks.len() as i32,
                ranks.as_ptr(),
                other.handle,
                out.as_mut_ptr(),
            )
        };
        Error::check_with_op(ret, "group_translate_ranks")?;
        Ok(out
            .into_iter()
            .map(|r| if r == -1 { None } else { Some(r) })
            .collect())
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        // Slot 0 is reserved for MPI_GROUP_EMPTY and must not be freed.
        if self.handle > 0 {
            // SAFETY: self.handle is owned and valid; Drop runs exactly once, so no double-free.
            unsafe { ffi::ferrompi_group_free(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Group, GroupComparison, RankRange};

    #[test]
    fn group_drop_with_zero_handle_is_no_op() {
        // Handle 0 corresponds to MPI_GROUP_EMPTY (reserved slot).
        // The Drop impl guards against handle == 0, so this must not
        // call ferrompi_group_free and must not panic or segfault.
        let g = Group { handle: 0 };
        drop(g); // Must complete without panicking
    }

    /// Empty-input fast path: translate_ranks with an empty slice must return
    /// Ok(vec![]) without calling FFI (which would require an active MPI session).
    ///
    /// This test is safe to run without `mpiexec` because the wrapper
    /// short-circuits on `ranks.is_empty()` before touching any MPI handle.
    #[test]
    fn translate_ranks_empty_input_returns_empty_vec() {
        let g = Group { handle: 0 };
        let result = g
            .translate_ranks(&[], &Group { handle: 0 })
            .expect("translate_ranks with empty slice must succeed");
        assert!(result.is_empty(), "expected empty Vec, got {result:?}");
        // Prevent Drop from calling ferrompi_group_free on handle 0 twice.
        // The guard in Drop already skips handle 0, but forget is explicit
        // about our intent here.
        std::mem::forget(g);
    }

    // ── GroupComparison unit tests ────────────────────────────────────────

    #[test]
    fn group_comparison_repr_values() {
        assert_eq!(GroupComparison::Identical as i32, 0);
        assert_eq!(GroupComparison::Similar as i32, 1);
        assert_eq!(GroupComparison::Unequal as i32, 2);
    }

    // ── RankRange unit tests ──────────────────────────────────────────────

    #[test]
    fn rank_range_repr_and_size() {
        // RankRange has 3 × i32 fields with natural alignment.
        // On all current targets (x86_64, aarch64, riscv64) the natural layout
        // is 12 bytes with no tail padding.
        assert_eq!(std::mem::size_of::<RankRange>(), 12);
    }
}
