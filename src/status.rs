//! MPI message status information.
//!
//! This module provides the [`Status`] struct returned by probe operations,
//! containing metadata about a message without actually receiving it, and the
//! [`Source`] and [`Tag`] types that select which messages a receive or probe
//! matches.

use crate::error::{Error, MpiErrorClass, Result};
use crate::ffi;

// Private codes carried across the FFI for the wildcard sources and tags; the C
// layer translates them to the linked MPI's constants. These MUST match the
// `FERROMPI_ANY_SOURCE`, `FERROMPI_PROC_NULL` and `FERROMPI_ANY_TAG` defines in
// `csrc/ferrompi.h`.
pub(crate) const ANY_SOURCE_CODE: i32 = -1;
pub(crate) const PROC_NULL_CODE: i32 = -2;
pub(crate) const ANY_TAG_CODE: i32 = -1;

/// Which rank a receive or probe matches messages from, or a send's destination.
///
/// An `i32` converts to [`Source::Rank`], so `world.recv(&mut buf, 0, 7)` and
/// `world.send(&buf, 1, 7)` still compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Match a message from any rank (`MPI_ANY_SOURCE`). As a destination it is
    /// rejected with [`Error::InvalidArgument`] before any MPI call: MPI has no
    /// "any destination".
    Any,
    /// Match no rank (`MPI_PROC_NULL`): a receive or probe completes at once with
    /// no data. As a destination, a send sends nothing and completes at once.
    ProcNull,
    /// Match messages from this rank, or send to it. A negative rank is rejected
    /// with [`Error::InvalidArgument`] before any MPI call; use [`Source::Any`]
    /// or [`Source::ProcNull`] for the wildcards of a receive, and
    /// [`Source::ProcNull`] for a destination.
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

    pub(crate) fn dest_code(self) -> Result<i32> {
        match self {
            Source::Any => Err(Error::InvalidArgument {
                arg: "dest",
                reason: "a destination cannot be Any",
            }),
            Source::ProcNull => Ok(PROC_NULL_CODE),
            Source::Rank(r) if r >= 0 => Ok(r),
            Source::Rank(_) => Err(Error::InvalidArgument {
                arg: "dest",
                reason: "negative rank: use Source::ProcNull",
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
/// Returned by [`Communicator::recv`](crate::Communicator::recv),
/// [`Communicator::sendrecv`](crate::Communicator::sendrecv) and
/// [`Communicator::recv_custom`](crate::Communicator::recv_custom) to describe
/// the message just received, by
/// [`Communicator::probe`](crate::Communicator::probe) and
/// [`Communicator::iprobe`](crate::Communicator::iprobe) to describe an
/// incoming message without consuming it, and by
/// [`Request::wait`](crate::Request::wait),
/// [`test`](crate::Request::test),
/// [`wait_any`](crate::Request::wait_any) and
/// [`test_any`](crate::Request::test_any) to describe the receive they
/// completed.
///
/// # The empty status
///
/// A send, nonblocking-collective or RMA request, a cancelled receive, and a
/// request that was already completed have no message to describe (MPI leaves
/// the source, tag and count of the first four undefined). They report the
/// empty status instead: [`Source::Any`], [`Tag::Any`] and `count: Some(0)`.
/// For a send request only `error` has meaning.
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
/// println!("Message from {:?} with tag {:?}, {:?} elements",
///          status.source, status.tag, status.count);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Status {
    /// The matched source: [`Source::Rank`] for a received or probed message,
    /// [`Source::ProcNull`] for a receive or probe from `PROC_NULL`,
    /// [`Source::Any`] in the empty status.
    pub source: Source,
    /// The matched tag: [`Tag::Value`] for a message; [`Tag::Any`] for a
    /// receive or probe from `PROC_NULL` and in the empty status.
    pub tag: Tag,
    /// Number of elements of the call's datatype in the message, from
    /// `MPI_Get_count`; `None` when the message is not a whole number of
    /// elements (MPI reports `MPI_UNDEFINED`). `Some(0)` in the empty status.
    pub count: Option<usize>,
    /// `Some` only for a request whose own error MPI reported in a
    /// multi-request completion; `None` for a single-request call.
    pub error: Option<MpiErrorClass>,
}

impl Status {
    pub(crate) const EMPTY: Status = Status {
        source: Source::Any,
        tag: Tag::Any,
        count: Some(0),
        error: None,
    };

    #[inline]
    pub(crate) fn from_ffi(raw: ffi::FerrompiStatus) -> Status {
        Status {
            source: match raw.source {
                PROC_NULL_CODE => Source::ProcNull,
                ANY_SOURCE_CODE => Source::Any,
                rank => Source::Rank(rank),
            },
            tag: match raw.tag {
                ANY_TAG_CODE => Tag::Any,
                value => Tag::Value(value),
            },
            count: usize::try_from(raw.count).ok(),
            // `raw.error` is already an error class (see `ferrompi_status`), never an error code.
            error: (raw.error != 0).then(|| MpiErrorClass::from_raw(raw.error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Source, Status, Tag, ANY_SOURCE_CODE, ANY_TAG_CODE, PROC_NULL_CODE};
    use crate::error::{Error, MpiErrorClass};
    use crate::ffi::FerrompiStatus;

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
    fn dest_code_any() {
        let err = Source::Any.dest_code().unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "dest",
                reason: "a destination cannot be Any",
            }
        ));
    }

    #[test]
    fn dest_code_proc_null() {
        assert_eq!(Source::ProcNull.dest_code().unwrap(), PROC_NULL_CODE);
    }

    #[test]
    fn dest_code_non_negative_rank() {
        assert_eq!(Source::Rank(0).dest_code().unwrap(), 0);
        assert_eq!(Source::Rank(7).dest_code().unwrap(), 7);
    }

    #[test]
    fn dest_code_negative_rank() {
        for r in [-1, -2, i32::MIN] {
            let err = Source::Rank(r).dest_code().unwrap_err();
            assert!(matches!(
                err,
                Error::InvalidArgument {
                    arg: "dest",
                    reason: "negative rank: use Source::ProcNull",
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

    fn raw(source: i32, tag: i32, count: i64, error: i32) -> FerrompiStatus {
        FerrompiStatus {
            source,
            tag,
            count,
            error,
        }
    }

    #[test]
    fn status_from_ffi_rank_and_value() {
        let status = Status::from_ffi(raw(3, 7, 5, 0));
        assert_eq!(status.source, Source::Rank(3));
        assert_eq!(status.tag, Tag::Value(7));
        assert_eq!(status.count, Some(5));
        assert_eq!(status.error, None);
    }

    #[test]
    fn status_from_ffi_proc_null() {
        let status = Status::from_ffi(raw(PROC_NULL_CODE, ANY_TAG_CODE, 0, 0));
        assert_eq!(status.source, Source::ProcNull);
        assert_eq!(status.tag, Tag::Any);
        assert_eq!(status.count, Some(0));
        assert_eq!(status.error, None);
    }

    #[test]
    fn status_from_ffi_any_source() {
        let status = Status::from_ffi(raw(ANY_SOURCE_CODE, 0, 0, 0));
        assert_eq!(status.source, Source::Any);
    }

    #[test]
    fn empty_status_decodes_from_private_codes() {
        assert_eq!(
            Status::from_ffi(raw(ANY_SOURCE_CODE, ANY_TAG_CODE, 0, 0)),
            Status::EMPTY
        );
    }

    #[test]
    fn status_from_ffi_partial_count_is_none() {
        let status = Status::from_ffi(raw(0, 0, -1, 0));
        assert_eq!(status.count, None);
    }

    #[test]
    fn status_from_ffi_error_zero_is_none() {
        assert_eq!(Status::from_ffi(raw(0, 0, 0, 0)).error, None);
    }

    #[test]
    fn status_from_ffi_nonzero_error_class() {
        assert_eq!(
            Status::from_ffi(raw(0, 0, 0, 999)).error,
            Some(MpiErrorClass::Raw(999))
        );
    }
}
