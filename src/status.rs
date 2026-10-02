//! MPI message status information.
//!
//! This module provides the [`Status`] struct returned by probe operations,
//! containing metadata about a message without actually receiving it, and the
//! [`Source`] and [`Tag`] types that select which messages a receive or probe
//! matches.

use crate::error::{Error, Result};

// Private codes carried across the FFI for the wildcard sources and tags; the C
// layer translates them to the linked MPI's constants. These MUST match the
// `FERROMPI_ANY_SOURCE`, `FERROMPI_PROC_NULL` and `FERROMPI_ANY_TAG` defines in
// `csrc/ferrompi.h`.
pub(crate) const ANY_SOURCE_CODE: i32 = -1;
pub(crate) const PROC_NULL_CODE: i32 = -2;
pub(crate) const ANY_TAG_CODE: i32 = -1;

/// Which rank a receive or probe matches messages from.
///
/// An `i32` converts to [`Source::Rank`], so `world.recv(&mut buf, 0, 7)` still
/// compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Match a message from any rank (`MPI_ANY_SOURCE`).
    Any,
    /// Match no rank (`MPI_PROC_NULL`): the call completes at once with no data.
    ProcNull,
    /// Match messages from this rank. A negative rank is rejected with
    /// [`Error::InvalidArgument`] before any MPI call; use [`Source::Any`] or
    /// [`Source::ProcNull`] for the wildcards.
    Rank(i32),
}

impl From<i32> for Source {
    fn from(rank: i32) -> Self {
        Source::Rank(rank)
    }
}

impl Source {
    pub(crate) fn source_code(self, arg: &'static str) -> Result<i32> {
        match self {
            Source::Any => Ok(ANY_SOURCE_CODE),
            Source::ProcNull => Ok(PROC_NULL_CODE),
            Source::Rank(r) if r >= 0 => Ok(r),
            Source::Rank(_) => Err(Error::InvalidArgument {
                arg,
                reason: "negative rank: use Source::Any or Source::ProcNull",
            }),
        }
    }
}

/// Which tag a receive or probe matches messages with.
///
/// An `i32` converts to [`Tag::Value`], so `world.recv(&mut buf, 0, 7)` still
/// compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tag {
    /// Match a message with any tag (`MPI_ANY_TAG`).
    Any,
    /// Match messages with this tag. A negative tag is rejected with
    /// [`Error::InvalidArgument`] before any MPI call; use [`Tag::Any`] for the
    /// wildcard.
    Value(i32),
}

impl From<i32> for Tag {
    fn from(tag: i32) -> Self {
        Tag::Value(tag)
    }
}

impl Tag {
    pub(crate) fn tag_code(self, arg: &'static str) -> Result<i32> {
        match self {
            Tag::Any => Ok(ANY_TAG_CODE),
            Tag::Value(t) if t >= 0 => Ok(t),
            Tag::Value(_) => Err(Error::InvalidArgument {
                arg,
                reason: "negative tag: use Tag::Any",
            }),
        }
    }
}

/// Information about a probed or received MPI message.
///
/// Returned by [`Communicator::probe`](crate::Communicator::probe) and
/// [`Communicator::iprobe`](crate::Communicator::iprobe) to describe an
/// incoming message without consuming it.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, Source, Tag};
/// let mpi = Mpi::init().unwrap();
/// let world = mpi.world();
///
/// // Blocking probe for any f64 message
/// let status = world.probe::<f64>(Source::Any, Tag::Any).unwrap();
/// println!("Message from rank {} with tag {}, {} elements",
///          status.source, status.tag, status.count);
/// ```
#[derive(Debug, Clone)]
pub struct Status {
    /// Source rank of the message.
    pub source: i32,
    /// Tag of the message.
    pub tag: i32,
    /// Number of elements of the probed type in the message, from `MPI_Get_count`;
    /// `-1` when the message is not a whole number of elements (MPI reports
    /// `MPI_UNDEFINED`).
    pub count: i64,
}

#[cfg(test)]
mod tests {
    use super::{Source, Tag, ANY_SOURCE_CODE, ANY_TAG_CODE, PROC_NULL_CODE};
    use crate::error::Error;

    #[test]
    fn source_code_any() {
        assert_eq!(Source::Any.source_code("source").unwrap(), ANY_SOURCE_CODE);
    }

    #[test]
    fn source_code_proc_null() {
        assert_eq!(
            Source::ProcNull.source_code("source").unwrap(),
            PROC_NULL_CODE
        );
    }

    #[test]
    fn source_code_non_negative_rank() {
        assert_eq!(Source::Rank(0).source_code("source").unwrap(), 0);
        assert_eq!(Source::Rank(7).source_code("source").unwrap(), 7);
    }

    #[test]
    fn source_code_negative_rank() {
        for r in [-1, -2, i32::MIN] {
            let err = Source::Rank(r).source_code("source").unwrap_err();
            assert!(matches!(
                err,
                Error::InvalidArgument {
                    arg: "source",
                    reason: "negative rank: use Source::Any or Source::ProcNull",
                }
            ));
        }
    }

    #[test]
    fn tag_code_any() {
        assert_eq!(Tag::Any.tag_code("tag").unwrap(), ANY_TAG_CODE);
    }

    #[test]
    fn tag_code_non_negative_value() {
        assert_eq!(Tag::Value(0).tag_code("tag").unwrap(), 0);
        assert_eq!(Tag::Value(7).tag_code("tag").unwrap(), 7);
    }

    #[test]
    fn tag_code_negative_value() {
        for t in [-1, -5, i32::MIN] {
            let err = Tag::Value(t).tag_code("tag").unwrap_err();
            assert!(matches!(
                err,
                Error::InvalidArgument {
                    arg: "tag",
                    reason: "negative tag: use Tag::Any",
                }
            ));
        }
        let err = Tag::Value(-1).tag_code("recvtag").unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "recvtag", .. }));
    }

    #[test]
    fn from_i32_maps_to_rank_and_value() {
        assert_eq!(Source::from(3), Source::Rank(3));
        assert_eq!(Source::from(-1), Source::Rank(-1));
        assert_eq!(Tag::from(3), Tag::Value(3));
        assert_eq!(Tag::from(-1), Tag::Value(-1));
    }
}
