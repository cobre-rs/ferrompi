//! Error types for ferrompi.
//!
//! This module provides structured MPI error handling with error class
//! categorization and human-readable messages obtained from the MPI runtime.

use std::sync::OnceLock;

use crate::ffi;

/// Cached implementation-specific MPI error class values.
/// Returns (MPI_ERR_FILE, MPI_ERR_INFO, MPI_ERR_WIN) from the C layer.
fn impl_error_classes() -> (i32, i32, i32) {
    static CLASSES: OnceLock<(i32, i32, i32)> = OnceLock::new();
    *CLASSES.get_or_init(|| unsafe {
        (
            ffi::ferrompi_err_file(),
            ffi::ferrompi_err_info(),
            ffi::ferrompi_err_win(),
        )
    })
}

/// Result type for MPI operations.
pub type Result<T> = std::result::Result<T, Error>;

// Internal resource-exhaustion sentinels returned by the C layer when a
// fixed-size handle table is full. Negative so they never collide with MPI
// return codes (which are non-negative); intercepted in [`Error::from_code`]
// and mapped to [`Error::ResourceExhausted`]. These MUST match the
// `FERROMPI_ERR_*_FULL` defines in `csrc/ferrompi.c`.
const FERROMPI_ERR_REQUESTS_FULL: i32 = -7001;
const FERROMPI_ERR_COMMS_FULL: i32 = -7002;
const FERROMPI_ERR_DATATYPES_FULL: i32 = -7003;
const FERROMPI_ERR_OPS_FULL: i32 = -7004;
const FERROMPI_ERR_WINDOWS_FULL: i32 = -7005;
const FERROMPI_ERR_GROUPS_FULL: i32 = -7006;
const FERROMPI_ERR_INFOS_FULL: i32 = -7007;

/// Identifies which internal ferrompi handle table was exhausted in an
/// [`Error::ResourceExhausted`].
///
/// Each variant corresponds to a fixed-size table in the C layer (e.g.
/// `MAX_COMMS`, `MAX_DATATYPES`, `MAX_OPS`, `MAX_REQUESTS`). The `Display`
/// form is the lowercase resource noun used in the error message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum ResourceKind {
    /// Nonblocking / persistent request table (`MAX_REQUESTS`).
    #[error("request")]
    Request,
    /// Communicator table (`MAX_COMMS`).
    #[error("communicator")]
    Communicator,
    /// Custom datatype table (`MAX_DATATYPES`).
    #[error("datatype")]
    Datatype,
    /// Custom reduction-operation table (`MAX_OPS`).
    #[error("operation")]
    Operation,
    /// RMA window table.
    #[error("window")]
    Window,
    /// Group table.
    #[error("group")]
    Group,
    /// Info-object table.
    #[error("info")]
    Info,
}

impl ResourceKind {
    /// Map a C-layer resource-exhaustion sentinel to its `ResourceKind`, or
    /// `None` if `code` is not one of the `FERROMPI_ERR_*_FULL` sentinels.
    fn from_sentinel(code: i32) -> Option<Self> {
        match code {
            FERROMPI_ERR_REQUESTS_FULL => Some(ResourceKind::Request),
            FERROMPI_ERR_COMMS_FULL => Some(ResourceKind::Communicator),
            FERROMPI_ERR_DATATYPES_FULL => Some(ResourceKind::Datatype),
            FERROMPI_ERR_OPS_FULL => Some(ResourceKind::Operation),
            FERROMPI_ERR_WINDOWS_FULL => Some(ResourceKind::Window),
            FERROMPI_ERR_GROUPS_FULL => Some(ResourceKind::Group),
            FERROMPI_ERR_INFOS_FULL => Some(ResourceKind::Info),
            _ => None,
        }
    }
}

/// MPI error class, categorizing the type of MPI error.
///
/// These correspond to the standard MPI error classes defined by the MPI specification.
/// The C layer calls `MPI_Error_class` to map an error code to one of these classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum MpiErrorClass {
    /// `MPI_SUCCESS` — no error
    #[error("SUCCESS")]
    Success,
    /// `MPI_ERR_BUFFER` — invalid buffer pointer
    #[error("ERR_BUFFER")]
    Buffer,
    /// `MPI_ERR_COUNT` — invalid count argument
    #[error("ERR_COUNT")]
    Count,
    /// `MPI_ERR_TYPE` — invalid datatype argument
    #[error("ERR_TYPE")]
    Type,
    /// `MPI_ERR_TAG` — invalid tag argument
    #[error("ERR_TAG")]
    Tag,
    /// `MPI_ERR_COMM` — invalid communicator
    #[error("ERR_COMM")]
    Comm,
    /// `MPI_ERR_RANK` — invalid rank
    #[error("ERR_RANK")]
    Rank,
    /// `MPI_ERR_REQUEST` — invalid request handle
    #[error("ERR_REQUEST")]
    Request,
    /// `MPI_ERR_ROOT` — invalid root
    #[error("ERR_ROOT")]
    Root,
    /// `MPI_ERR_GROUP` — invalid group
    #[error("ERR_GROUP")]
    Group,
    /// `MPI_ERR_OP` — invalid operation
    #[error("ERR_OP")]
    Op,
    /// `MPI_ERR_TOPOLOGY` — invalid topology
    #[error("ERR_TOPOLOGY")]
    Topology,
    /// `MPI_ERR_DIMS` — invalid dimension argument
    #[error("ERR_DIMS")]
    Dims,
    /// `MPI_ERR_ARG` — invalid argument
    #[error("ERR_ARG")]
    Arg,
    /// `MPI_ERR_UNKNOWN` — unknown error
    #[error("ERR_UNKNOWN")]
    Unknown,
    /// `MPI_ERR_TRUNCATE` — message truncated
    #[error("ERR_TRUNCATE")]
    Truncate,
    /// `MPI_ERR_OTHER` — other error
    #[error("ERR_OTHER")]
    Other,
    /// `MPI_ERR_INTERN` — internal MPI error
    #[error("ERR_INTERN")]
    Intern,
    /// `MPI_ERR_IN_STATUS` — error code is in status
    #[error("ERR_IN_STATUS")]
    InStatus,
    /// `MPI_ERR_PENDING` — pending request
    #[error("ERR_PENDING")]
    Pending,
    /// `MPI_ERR_WIN` — invalid window
    #[error("ERR_WIN")]
    Win,
    /// `MPI_ERR_INFO` — invalid info object
    #[error("ERR_INFO")]
    Info,
    /// `MPI_ERR_FILE` — invalid file handle
    #[error("ERR_FILE")]
    File,
    /// Unrecognized error class from the MPI implementation
    #[error("ERR_CLASS({0})")]
    Raw(i32),
}

impl MpiErrorClass {
    /// Map a raw MPI error class integer to the enum variant.
    ///
    /// Standard MPI error class values (MPI-3.1 Table 9.4):
    /// 0=SUCCESS, 1=BUFFER, 2=COUNT, 3=TYPE, 4=TAG, 5=COMM,
    /// 6=RANK, 7=REQUEST, 8=ROOT, 9=GROUP, 10=OP, 11=TOPOLOGY,
    /// 12=DIMS, 13=ARG, 14=UNKNOWN, 15=TRUNCATE, 16=OTHER,
    /// 17=INTERN, 18=IN_STATUS, 19=PENDING, plus implementation-
    /// specific classes for WIN (45), INFO (28), FILE (27).
    pub fn from_raw(class: i32) -> Self {
        // Standard MPI error classes (0-19) have fixed values per the MPI spec.
        // Implementation-specific classes (File, Info, Win) are queried from
        // the C layer to support both MPICH and Open MPI.
        match class {
            0 => MpiErrorClass::Success,
            1 => MpiErrorClass::Buffer,
            2 => MpiErrorClass::Count,
            3 => MpiErrorClass::Type,
            4 => MpiErrorClass::Tag,
            5 => MpiErrorClass::Comm,
            6 => MpiErrorClass::Rank,
            7 => MpiErrorClass::Request,
            8 => MpiErrorClass::Root,
            9 => MpiErrorClass::Group,
            10 => MpiErrorClass::Op,
            11 => MpiErrorClass::Topology,
            12 => MpiErrorClass::Dims,
            13 => MpiErrorClass::Arg,
            14 => MpiErrorClass::Unknown,
            15 => MpiErrorClass::Truncate,
            16 => MpiErrorClass::Other,
            17 => MpiErrorClass::Intern,
            18 => MpiErrorClass::InStatus,
            19 => MpiErrorClass::Pending,
            other => {
                // Query implementation-specific error class values from C layer
                let (err_file, err_info, err_win) = impl_error_classes();
                if other == err_file {
                    MpiErrorClass::File
                } else if other == err_info {
                    MpiErrorClass::Info
                } else if other == err_win {
                    MpiErrorClass::Win
                } else {
                    MpiErrorClass::Raw(other)
                }
            }
        }
    }
}

fn fmt_mpi(
    class: &MpiErrorClass,
    code: &i32,
    message: &str,
    operation: &Option<&'static str>,
) -> String {
    match operation {
        Some(op) => format!("MPI error in {op}: {message} (class={class}, code={code})"),
        None => format!("MPI error: {message} (class={class}, code={code})"),
    }
}

/// Error types for MPI operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// MPI has already been initialized.
    #[error("MPI has already been initialized")]
    AlreadyInitialized,

    /// MPI error with class, code, descriptive message, and optional operation name.
    #[error("{}", fmt_mpi(.class, .code, .message, .operation))]
    Mpi {
        /// The error class (category of error).
        class: MpiErrorClass,
        /// The raw MPI error code.
        code: i32,
        /// Human-readable error message from `MPI_Error_string`.
        message: String,
        /// The ferrompi operation that produced the error, if known.
        operation: Option<&'static str>,
    },

    /// Invalid buffer provided (e.g., send/recv buffer size mismatch).
    #[error("Invalid buffer")]
    InvalidBuffer,

    /// Invalid reduction operation for the method being called
    /// (e.g., passing a non-MAXLOC/MINLOC op to `allreduce_indexed`).
    #[error("Invalid reduction operation for this method")]
    InvalidOp,

    /// Operation not supported (e.g., MPI 4.0 persistent collectives on older MPI).
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// An internal ferrompi handle table is full (too many concurrently live
    /// requests / communicators / datatypes / etc.).
    ///
    /// Distinct from a genuine MPI fault ([`Error::Mpi`] with class
    /// `ERR_OTHER`) so a long-running caller can recognize an internal cap,
    /// release handles, back off, or retry rather than treating it as fatal.
    #[error("ferrompi {resource} table is full")]
    ResourceExhausted {
        /// Which internal handle table overflowed.
        resource: ResourceKind,
    },

    /// Internal ferrompi error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Create a structured error from an MPI error code.
    ///
    /// Calls `ferrompi_error_info` to obtain the error class and human-readable
    /// message from the MPI runtime.
    ///
    /// # Returns
    ///
    /// If called with `MPI_SUCCESS` (code 0), returns
    /// `Error::Internal("from_code called with success code 0")`.
    /// Callers should use [`Error::check`] or [`Error::check_with_op`]
    /// for the standard "Ok on success, Err on failure" idiom.
    ///
    /// Marked `#[cold]` + `#[inline(never)]`: this is only reached on the error
    /// path of [`Error::check`]/[`check_with_op`], and its body (a 512-byte
    /// stack buffer, an FFI call to `ferrompi_error_info`, and a `format!`
    /// fallback) must stay out of line so the inlined success-path check folds
    /// to a single return-code branch in the caller.
    #[cold]
    #[inline(never)]
    pub fn from_code(code: i32) -> Self {
        if code == 0 {
            return Error::Internal("from_code called with success code 0".into());
        }

        // Resource-exhaustion sentinels from the C layer are negative; MPI
        // codes are non-negative. Map them before touching the MPI runtime —
        // MPI_Error_class on a synthetic negative code is undefined behavior.
        if let Some(resource) = ResourceKind::from_sentinel(code) {
            return Error::ResourceExhausted { resource };
        }

        let mut class: i32 = 0;
        let mut msg_buf = [0u8; 512];
        let mut msg_len: i32 = 0;

        let ret = unsafe {
            ffi::ferrompi_error_info(
                code,
                &mut class,
                msg_buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                &mut msg_len,
            )
        };

        if ret == 0 {
            let len = msg_len.max(0) as usize;
            let message = std::str::from_utf8(&msg_buf[..len])
                .unwrap_or("unknown error")
                .to_string();
            Error::Mpi {
                class: MpiErrorClass::from_raw(class),
                code,
                message,
                operation: None,
            }
        } else {
            // ferrompi_error_info itself failed — provide a fallback
            Error::Mpi {
                class: MpiErrorClass::Raw(code),
                code,
                message: format!("MPI error code {code}"),
                operation: None,
            }
        }
    }

    /// Create a structured error from an MPI error code with an operation name.
    ///
    /// Behaves identically to [`Error::from_code`] but populates `operation:
    /// Some(operation)` on the resulting [`Error::Mpi`] variant, so callers can
    /// record which ferrompi wrapper produced the error.
    ///
    /// # Contract
    ///
    /// Callers must not pass `code = 0`. The canonical success-vs-error
    /// idiom is [`Error::check_with_op`], which short-circuits to `Ok(())`
    /// when `code == 0`. Passing 0 here returns `Error::Internal("from_code
    /// called with success code 0")` as a defensive fallback — this signals
    /// a misuse of the API, not an MPI error condition.
    ///
    /// # Returns
    ///
    /// - `Error::Mpi { .. }` for any non-zero MPI error code, with the
    ///   `operation` field populated.
    /// - `Error::Internal` if called with `code = 0` (delegated from
    ///   [`Error::from_code`]; treat this as a programming error in the
    ///   caller, not a runtime MPI failure).
    ///
    /// Marked `#[cold]` + `#[inline(never)]` for the same reason as
    /// [`Error::from_code`]: it is only invoked on the error path, so keeping
    /// it out of line lets [`Error::check_with_op`] inline to a bare branch.
    #[cold]
    #[inline(never)]
    pub fn from_code_with_op(code: i32, operation: &'static str) -> Self {
        match Error::from_code(code) {
            Error::Mpi {
                class,
                code,
                message,
                operation: _,
            } => Error::Mpi {
                class,
                code,
                message,
                operation: Some(operation),
            },
            // from_code returns Error::Mpi for non-zero codes and Error::Internal
            // for code 0. Both are preserved verbatim here.
            other => other,
        }
    }

    /// Check an MPI return code, returning `Ok(())` for success.
    ///
    /// Returns `Err(Error::Mpi { .. })` for non-zero codes.
    ///
    /// `#[inline]` so the success-path check (`code == 0`) folds into the
    /// caller across codegen-unit/crate boundaries; the cold error
    /// construction is delegated to the `#[cold]` [`Error::from_code`].
    #[inline]
    pub fn check(code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error::from_code(code))
        }
    }

    /// Check an MPI return code with an operation name, returning `Ok(())` for
    /// success.
    ///
    /// Returns `Err(Error::Mpi { operation: Some(operation), .. })` for
    /// non-zero codes.
    ///
    /// `#[inline]` so the success-path check folds into each of the ~150 call
    /// sites; the cold error construction stays out of line in the `#[cold]`
    /// [`Error::from_code_with_op`].
    #[inline]
    pub fn check_with_op(code: i32, operation: &'static str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error::from_code_with_op(code, operation))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_success_returns_ok() {
        assert!(Error::check(0).is_ok());
    }

    #[test]
    fn error_class_from_known_values() {
        assert_eq!(MpiErrorClass::from_raw(0), MpiErrorClass::Success);
        assert_eq!(MpiErrorClass::from_raw(1), MpiErrorClass::Buffer);
        assert_eq!(MpiErrorClass::from_raw(2), MpiErrorClass::Count);
        assert_eq!(MpiErrorClass::from_raw(3), MpiErrorClass::Type);
        assert_eq!(MpiErrorClass::from_raw(4), MpiErrorClass::Tag);
        assert_eq!(MpiErrorClass::from_raw(5), MpiErrorClass::Comm);
        assert_eq!(MpiErrorClass::from_raw(6), MpiErrorClass::Rank);
        assert_eq!(MpiErrorClass::from_raw(7), MpiErrorClass::Request);
        assert_eq!(MpiErrorClass::from_raw(8), MpiErrorClass::Root);
        assert_eq!(MpiErrorClass::from_raw(9), MpiErrorClass::Group);
        assert_eq!(MpiErrorClass::from_raw(10), MpiErrorClass::Op);
        assert_eq!(MpiErrorClass::from_raw(11), MpiErrorClass::Topology);
        assert_eq!(MpiErrorClass::from_raw(12), MpiErrorClass::Dims);
        assert_eq!(MpiErrorClass::from_raw(13), MpiErrorClass::Arg);
        assert_eq!(MpiErrorClass::from_raw(14), MpiErrorClass::Unknown);
        assert_eq!(MpiErrorClass::from_raw(15), MpiErrorClass::Truncate);
        assert_eq!(MpiErrorClass::from_raw(16), MpiErrorClass::Other);
        assert_eq!(MpiErrorClass::from_raw(17), MpiErrorClass::Intern);
        assert_eq!(MpiErrorClass::from_raw(18), MpiErrorClass::InStatus);
        assert_eq!(MpiErrorClass::from_raw(19), MpiErrorClass::Pending);
        // File, Info, Win are implementation-specific — cannot test without MPI runtime
    }

    #[test]
    fn error_class_unknown_raw_value() {
        assert_eq!(MpiErrorClass::from_raw(999), MpiErrorClass::Raw(999));
        assert_eq!(MpiErrorClass::from_raw(-1), MpiErrorClass::Raw(-1));
    }

    #[test]
    fn error_class_display_formats() {
        assert_eq!(format!("{}", MpiErrorClass::Success), "SUCCESS");
        assert_eq!(format!("{}", MpiErrorClass::Buffer), "ERR_BUFFER");
        assert_eq!(format!("{}", MpiErrorClass::Comm), "ERR_COMM");
        assert_eq!(format!("{}", MpiErrorClass::Rank), "ERR_RANK");
        assert_eq!(format!("{}", MpiErrorClass::Raw(42)), "ERR_CLASS(42)");
    }

    #[test]
    fn error_display_formats_correctly() {
        let err = Error::InvalidBuffer;
        assert_eq!(format!("{err}"), "Invalid buffer");

        let err = Error::AlreadyInitialized;
        assert_eq!(format!("{err}"), "MPI has already been initialized");

        let err = Error::NotSupported("persistent collectives".to_string());
        assert_eq!(
            format!("{err}"),
            "Operation not supported: persistent collectives"
        );

        let err = Error::Internal("test failure".to_string());
        assert_eq!(format!("{err}"), "Internal error: test failure");

        let err = Error::Mpi {
            class: MpiErrorClass::Rank,
            code: 6,
            message: "invalid rank".to_string(),
            operation: None,
        };
        assert_eq!(
            format!("{err}"),
            "MPI error: invalid rank (class=ERR_RANK, code=6)"
        );
    }

    #[test]
    #[allow(clippy::clone_on_copy)] // Intentionally exercising Clone derive
    fn error_class_hash_and_clone() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(MpiErrorClass::Success);
        set.insert(MpiErrorClass::Buffer);
        set.insert(MpiErrorClass::Raw(42));
        set.insert(MpiErrorClass::Raw(42)); // duplicate — should not increase len
        assert_eq!(set.len(), 3);

        // Verify membership
        assert!(set.contains(&MpiErrorClass::Success));
        assert!(set.contains(&MpiErrorClass::Buffer));
        assert!(set.contains(&MpiErrorClass::Raw(42)));
        assert!(!set.contains(&MpiErrorClass::Comm));

        // Exercise Clone
        let original = MpiErrorClass::Comm;
        let cloned = original.clone();
        assert_eq!(cloned, MpiErrorClass::Comm);
        assert_eq!(original, cloned);

        // Clone of Raw variant
        let raw_original = MpiErrorClass::Raw(77);
        let raw_cloned = raw_original.clone();
        assert_eq!(raw_cloned, MpiErrorClass::Raw(77));
    }

    #[test]
    fn error_class_display_all_variants() {
        // Comprehensive test of ALL Display implementations.
        // The existing `error_class_display_formats` test covers Success,
        // Buffer, Comm, Rank, and Raw. This test covers every variant
        // exhaustively for completeness.
        let cases = [
            (MpiErrorClass::Success, "SUCCESS"),
            (MpiErrorClass::Buffer, "ERR_BUFFER"),
            (MpiErrorClass::Count, "ERR_COUNT"),
            (MpiErrorClass::Type, "ERR_TYPE"),
            (MpiErrorClass::Tag, "ERR_TAG"),
            (MpiErrorClass::Comm, "ERR_COMM"),
            (MpiErrorClass::Rank, "ERR_RANK"),
            (MpiErrorClass::Request, "ERR_REQUEST"),
            (MpiErrorClass::Root, "ERR_ROOT"),
            (MpiErrorClass::Group, "ERR_GROUP"),
            (MpiErrorClass::Op, "ERR_OP"),
            (MpiErrorClass::Topology, "ERR_TOPOLOGY"),
            (MpiErrorClass::Dims, "ERR_DIMS"),
            (MpiErrorClass::Arg, "ERR_ARG"),
            (MpiErrorClass::Unknown, "ERR_UNKNOWN"),
            (MpiErrorClass::Truncate, "ERR_TRUNCATE"),
            (MpiErrorClass::Other, "ERR_OTHER"),
            (MpiErrorClass::Intern, "ERR_INTERN"),
            (MpiErrorClass::InStatus, "ERR_IN_STATUS"),
            (MpiErrorClass::Pending, "ERR_PENDING"),
            (MpiErrorClass::Win, "ERR_WIN"),
            (MpiErrorClass::Info, "ERR_INFO"),
            (MpiErrorClass::File, "ERR_FILE"),
            (MpiErrorClass::Raw(100), "ERR_CLASS(100)"),
        ];
        for (class, expected) in &cases {
            assert_eq!(
                format!("{class}"),
                *expected,
                "Display mismatch for {class:?}"
            );
        }
    }

    #[test]
    fn error_debug_format() {
        // Exercise Debug derive on Error::InvalidBuffer
        let err = Error::InvalidBuffer;
        let debug = format!("{err:?}");
        assert!(
            debug.contains("InvalidBuffer"),
            "Debug output should contain 'InvalidBuffer', got: {debug}"
        );

        // Exercise Debug on Error::Mpi variant
        let mpi_err = Error::Mpi {
            class: MpiErrorClass::Arg,
            code: 13,
            message: "invalid argument".to_string(),
            operation: None,
        };
        let debug = format!("{mpi_err:?}");
        assert!(
            debug.contains("Mpi"),
            "Debug output should contain 'Mpi', got: {debug}"
        );
        assert!(
            debug.contains("Arg"),
            "Debug output should contain 'Arg', got: {debug}"
        );

        // Exercise Debug on other Error variants
        let err = Error::AlreadyInitialized;
        let debug = format!("{err:?}");
        assert!(debug.contains("AlreadyInitialized"));

        let err = Error::NotSupported("test op".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("NotSupported"));

        let err = Error::Internal("internal msg".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Internal"));
    }

    #[test]
    fn error_mpi_fields_accessible() {
        // Verify Error::Mpi struct fields are accessible and correct
        let err = Error::Mpi {
            class: MpiErrorClass::Topology,
            code: 11,
            message: "invalid topology".to_string(),
            operation: None,
        };

        // Pattern-match to access fields
        if let Error::Mpi {
            class,
            code,
            message,
            operation,
        } = &err
        {
            assert_eq!(*class, MpiErrorClass::Topology);
            assert_eq!(*code, 11);
            assert_eq!(message, "invalid topology");
            assert_eq!(*operation, None);
        } else {
            panic!("Expected Error::Mpi variant");
        }

        // Verify Display uses all three fields
        let display = format!("{err}");
        assert!(display.contains("invalid topology"));
        assert!(display.contains("ERR_TOPOLOGY"));
        assert!(display.contains("11"));
    }

    #[test]
    fn error_mpi_display_with_operation_some() {
        let err = Error::Mpi {
            class: MpiErrorClass::Rank,
            code: 6,
            message: "invalid rank".to_string(),
            operation: Some("allreduce"),
        };
        assert_eq!(
            format!("{err}"),
            "MPI error in allreduce: invalid rank (class=ERR_RANK, code=6)"
        );
    }

    #[test]
    fn error_mpi_display_with_operation_none() {
        let err = Error::Mpi {
            class: MpiErrorClass::Rank,
            code: 6,
            message: "invalid rank".to_string(),
            operation: None,
        };
        assert_eq!(
            format!("{err}"),
            "MPI error: invalid rank (class=ERR_RANK, code=6)"
        );
    }

    #[test]
    fn from_code_with_zero_returns_internal_error() {
        let err = Error::from_code(0);
        match err {
            Error::Internal(msg) => {
                assert_eq!(msg, "from_code called with success code 0");
            }
            other => panic!("expected Error::Internal, got {other:?}"),
        }
    }

    #[test]
    fn from_code_with_op_sets_operation_field() {
        // Construct Error::Mpi directly (cannot call from_code_with_op in unit tests
        // without an MPI runtime) and verify that the operation field is honoured by
        // Display.  The delegation path in from_code_with_op is verified by inspection.
        let err = Error::Mpi {
            class: MpiErrorClass::Comm,
            code: 5,
            message: "invalid communicator".to_string(),
            operation: Some("broadcast"),
        };
        if let Error::Mpi { operation, .. } = &err {
            assert_eq!(*operation, Some("broadcast"));
        } else {
            panic!("Expected Error::Mpi variant");
        }
        let display = format!("{err}");
        assert_eq!(
            display,
            "MPI error in broadcast: invalid communicator (class=ERR_COMM, code=5)"
        );
    }

    #[test]
    fn from_code_maps_resource_exhaustion_sentinels() {
        // Sentinels are intercepted before any MPI FFI call, so this is safe
        // to run without an initialized MPI runtime.
        let cases = [
            (-7001, ResourceKind::Request),
            (-7002, ResourceKind::Communicator),
            (-7003, ResourceKind::Datatype),
            (-7004, ResourceKind::Operation),
            (-7005, ResourceKind::Window),
            (-7006, ResourceKind::Group),
            (-7007, ResourceKind::Info),
        ];
        for (code, expected) in cases {
            match Error::from_code(code) {
                Error::ResourceExhausted { resource } => assert_eq!(resource, expected),
                other => panic!("expected ResourceExhausted for code {code}, got {other:?}"),
            }
        }
    }

    #[test]
    fn from_code_with_op_preserves_resource_exhaustion() {
        // The operation name is dropped (ResourceExhausted carries no operation
        // field) but the resource must survive the with_op delegation.
        match Error::from_code_with_op(-7002, "comm_split") {
            Error::ResourceExhausted { resource } => {
                assert_eq!(resource, ResourceKind::Communicator);
            }
            other => panic!("expected ResourceExhausted, got {other:?}"),
        }
    }

    #[test]
    fn resource_exhausted_display() {
        let err = Error::ResourceExhausted {
            resource: ResourceKind::Request,
        };
        assert_eq!(format!("{err}"), "ferrompi request table is full");
        let err = Error::ResourceExhausted {
            resource: ResourceKind::Window,
        };
        assert_eq!(format!("{err}"), "ferrompi window table is full");
    }
}
