//! Error types for ferrompi.
//!
//! This module provides structured MPI error handling with error class
//! categorization and human-readable messages obtained from the MPI runtime.

use std::sync::OnceLock;

use crate::ffi;
use thiserror::Error;

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

/// MPI error class, categorizing the type of MPI error.
///
/// These correspond to the standard MPI error classes defined by the MPI specification.
/// The C layer calls `MPI_Error_class` to map an error code to one of these classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpiErrorClass {
    /// `MPI_SUCCESS` — no error
    Success,
    /// `MPI_ERR_BUFFER` — invalid buffer pointer
    Buffer,
    /// `MPI_ERR_COUNT` — invalid count argument
    Count,
    /// `MPI_ERR_TYPE` — invalid datatype argument
    Type,
    /// `MPI_ERR_TAG` — invalid tag argument
    Tag,
    /// `MPI_ERR_COMM` — invalid communicator
    Comm,
    /// `MPI_ERR_RANK` — invalid rank
    Rank,
    /// `MPI_ERR_REQUEST` — invalid request handle
    Request,
    /// `MPI_ERR_ROOT` — invalid root
    Root,
    /// `MPI_ERR_GROUP` — invalid group
    Group,
    /// `MPI_ERR_OP` — invalid operation
    Op,
    /// `MPI_ERR_TOPOLOGY` — invalid topology
    Topology,
    /// `MPI_ERR_DIMS` — invalid dimension argument
    Dims,
    /// `MPI_ERR_ARG` — invalid argument
    Arg,
    /// `MPI_ERR_UNKNOWN` — unknown error
    Unknown,
    /// `MPI_ERR_TRUNCATE` — message truncated
    Truncate,
    /// `MPI_ERR_OTHER` — other error
    Other,
    /// `MPI_ERR_INTERN` — internal MPI error
    Intern,
    /// `MPI_ERR_IN_STATUS` — error code is in status
    InStatus,
    /// `MPI_ERR_PENDING` — pending request
    Pending,
    /// `MPI_ERR_WIN` — invalid window
    Win,
    /// `MPI_ERR_INFO` — invalid info object
    Info,
    /// `MPI_ERR_FILE` — invalid file handle
    File,
    /// Unrecognized error class from the MPI implementation
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

impl std::fmt::Display for MpiErrorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MpiErrorClass::Success => write!(f, "SUCCESS"),
            MpiErrorClass::Buffer => write!(f, "ERR_BUFFER"),
            MpiErrorClass::Count => write!(f, "ERR_COUNT"),
            MpiErrorClass::Type => write!(f, "ERR_TYPE"),
            MpiErrorClass::Tag => write!(f, "ERR_TAG"),
            MpiErrorClass::Comm => write!(f, "ERR_COMM"),
            MpiErrorClass::Rank => write!(f, "ERR_RANK"),
            MpiErrorClass::Request => write!(f, "ERR_REQUEST"),
            MpiErrorClass::Root => write!(f, "ERR_ROOT"),
            MpiErrorClass::Group => write!(f, "ERR_GROUP"),
            MpiErrorClass::Op => write!(f, "ERR_OP"),
            MpiErrorClass::Topology => write!(f, "ERR_TOPOLOGY"),
            MpiErrorClass::Dims => write!(f, "ERR_DIMS"),
            MpiErrorClass::Arg => write!(f, "ERR_ARG"),
            MpiErrorClass::Unknown => write!(f, "ERR_UNKNOWN"),
            MpiErrorClass::Truncate => write!(f, "ERR_TRUNCATE"),
            MpiErrorClass::Other => write!(f, "ERR_OTHER"),
            MpiErrorClass::Intern => write!(f, "ERR_INTERN"),
            MpiErrorClass::InStatus => write!(f, "ERR_IN_STATUS"),
            MpiErrorClass::Pending => write!(f, "ERR_PENDING"),
            MpiErrorClass::Win => write!(f, "ERR_WIN"),
            MpiErrorClass::Info => write!(f, "ERR_INFO"),
            MpiErrorClass::File => write!(f, "ERR_FILE"),
            MpiErrorClass::Raw(c) => write!(f, "ERR_CLASS({c})"),
        }
    }
}

/// Error types for MPI operations.
#[derive(Error, Debug)]
pub enum Error {
    /// MPI has already been initialized.
    #[error("MPI has already been initialized")]
    AlreadyInitialized,

    /// MPI error with class, code, and descriptive message from the MPI runtime.
    #[error("MPI error: {message} (class={class}, code={code})")]
    Mpi {
        /// The error class (category of error).
        class: MpiErrorClass,
        /// The raw MPI error code.
        code: i32,
        /// Human-readable error message from `MPI_Error_string`.
        message: String,
    },

    /// Invalid buffer provided (e.g., send/recv buffer size mismatch).
    #[error("Invalid buffer")]
    InvalidBuffer,

    /// Operation not supported (e.g., MPI 4.0 persistent collectives on older MPI).
    #[error("Operation not supported: {0}")]
    NotSupported(String),

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
    /// Calls `ferrompi_error_info` to obtain the error class and human-readable
    /// message from the MPI runtime.
    ///
    /// # Panics
    ///
    /// Panics if called with `MPI_SUCCESS` (code 0).
    pub fn from_code(code: i32) -> Self {
        assert!(code != 0, "from_code called with success code 0");

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
            }
        } else {
            // ferrompi_error_info itself failed — provide a fallback
            Error::Mpi {
                class: MpiErrorClass::Raw(code),
                code,
                message: format!("MPI error code {code}"),
            }
        }
    }

    /// Check an MPI return code, returning `Ok(())` for success.
    ///
    /// Returns `Err(Error::Mpi { .. })` for non-zero codes.
    pub fn check(code: i32) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error::from_code(code))
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
        };

        // Pattern-match to access fields
        if let Error::Mpi {
            class,
            code,
            message,
        } = &err
        {
            assert_eq!(*class, MpiErrorClass::Topology);
            assert_eq!(*code, 11);
            assert_eq!(message, "invalid topology");
        } else {
            panic!("Expected Error::Mpi variant");
        }

        // Verify Display uses all three fields
        let display = format!("{err}");
        assert!(display.contains("invalid topology"));
        assert!(display.contains("ERR_TOPOLOGY"));
        assert!(display.contains("11"));
    }
}
