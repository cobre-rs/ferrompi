//! MPI Info object for passing hints to MPI operations.
//!
//! [`Info`] wraps an `MPI_Info` handle and provides methods for setting and
//! querying key-value pairs used as hints by various MPI operations (e.g.,
//! window allocation, file I/O).
//!
//! # Example
//!
//! ```no_run
//! use ferrompi::Info;
//!
//! let mut info = Info::new().unwrap();
//! info.set("alloc_shared_noncontig", "true").unwrap();
//! assert_eq!(info.get("alloc_shared_noncontig").unwrap(), Some("true".to_string()));
//! ```

use crate::error::{Error, Result};
use crate::ffi;
use std::ffi::{CStr, CString};

/// Maximum buffer size for retrieving info values from MPI.
const INFO_VALUE_MAX_LEN: i32 = 1024;

/// An MPI info object for passing hints to MPI operations.
///
/// This type wraps an `MPI_Info` handle with RAII semantics: the underlying
/// MPI info object is freed automatically when the `Info` is dropped.
///
/// Use [`Info::null()`] to represent `MPI_INFO_NULL` (no hints), or
/// [`Info::new()`] to create a mutable info object that can hold key-value
/// pairs.
///
/// # Example
///
/// ```no_run
/// use ferrompi::Info;
///
/// // Create an info object with hints
/// let mut info = Info::new().unwrap();
/// info.set("alloc_shared_noncontig", "true").unwrap();
///
/// // Use Info::null() when no hints are needed
/// let null_info = Info::null();
/// assert_eq!(null_info.raw_handle(), -1);
/// ```
pub struct Info {
    handle: i32,
    is_null: bool,
}

impl Info {
    /// Create a new empty MPI info object.
    ///
    /// The info object is initially empty (no key-value pairs). Use
    /// [`set()`](Self::set) to add hints.
    ///
    /// # Errors
    ///
    /// Returns an error if the MPI info object could not be created (e.g.,
    /// the internal info handle table is full).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Info;
    ///
    /// let mut info = Info::new().unwrap();
    /// info.set("key", "value").unwrap();
    /// ```
    pub fn new() -> Result<Self> {
        let mut handle: i32 = 0;
        let ret = unsafe { ffi::ferrompi_info_create(&mut handle) };
        Error::check(ret)?;
        Ok(Info {
            handle,
            is_null: false,
        })
    }

    /// Get a handle representing `MPI_INFO_NULL`.
    ///
    /// This is used when an MPI operation does not require any hints.
    /// The returned `Info` does not own any MPI resource and will not
    /// call `MPI_Info_free` on drop.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Info;
    ///
    /// let info = Info::null();
    /// assert_eq!(info.raw_handle(), -1);
    /// ```
    pub fn null() -> Self {
        Info {
            handle: -1,
            is_null: true,
        }
    }

    /// Set a key-value pair on this info object.
    ///
    /// If the key already exists, its value is replaced.
    ///
    /// # Arguments
    ///
    /// * `key` - The hint key (must not contain null bytes)
    /// * `value` - The hint value (must not contain null bytes)
    ///
    /// # Errors
    ///
    /// Returns an error if this is a null info object, if the key or value
    /// contains a null byte, or if the MPI operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Info;
    ///
    /// let mut info = Info::new().unwrap();
    /// info.set("alloc_shared_noncontig", "true").unwrap();
    /// ```
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        if self.is_null {
            return Err(Error::Internal(
                "cannot set key-value on MPI_INFO_NULL".into(),
            ));
        }
        let c_key =
            CString::new(key).map_err(|_| Error::Internal("info key contains null byte".into()))?;
        let c_value = CString::new(value)
            .map_err(|_| Error::Internal("info value contains null byte".into()))?;
        let ret = unsafe { ffi::ferrompi_info_set(self.handle, c_key.as_ptr(), c_value.as_ptr()) };
        Error::check(ret)
    }

    /// Get the value associated with a key.
    ///
    /// Returns `Ok(Some(value))` if the key exists, or `Ok(None)` if the key
    /// was not found.
    ///
    /// # Arguments
    ///
    /// * `key` - The hint key to look up (must not contain null bytes)
    ///
    /// # Errors
    ///
    /// Returns an error if this is a null info object, if the key contains
    /// a null byte, or if the MPI operation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ferrompi::Info;
    ///
    /// let mut info = Info::new().unwrap();
    /// info.set("my_key", "my_value").unwrap();
    /// assert_eq!(info.get("my_key").unwrap(), Some("my_value".to_string()));
    /// assert_eq!(info.get("nonexistent").unwrap(), None);
    /// ```
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        if self.is_null {
            return Err(Error::Internal(
                "cannot get key-value from MPI_INFO_NULL".into(),
            ));
        }
        let c_key =
            CString::new(key).map_err(|_| Error::Internal("info key contains null byte".into()))?;
        let mut buf = vec![0u8; INFO_VALUE_MAX_LEN as usize];
        let mut valuelen: i32 = INFO_VALUE_MAX_LEN;
        let mut flag: i32 = 0;
        let ret = unsafe {
            ffi::ferrompi_info_get(
                self.handle,
                c_key.as_ptr(),
                buf.as_mut_ptr().cast::<std::ffi::c_char>(),
                &mut valuelen,
                &mut flag,
            )
        };
        Error::check(ret)?;
        if flag == 0 {
            return Ok(None);
        }
        // SAFETY: The C layer writes a null-terminated string into buf.
        // CStr::from_ptr finds the first null terminator.
        let c_str = unsafe { CStr::from_ptr(buf.as_ptr().cast::<std::ffi::c_char>()) };
        let value = c_str
            .to_str()
            .map_err(|_| Error::Internal("info value is not valid UTF-8".into()))?;
        Ok(Some(value.to_string()))
    }

    /// Get the raw info handle for passing to C functions.
    ///
    /// Returns `-1` for `MPI_INFO_NULL`.
    pub fn raw_handle(&self) -> i32 {
        self.handle
    }
}

impl Drop for Info {
    fn drop(&mut self) {
        if !self.is_null && self.handle >= 0 {
            // SAFETY: handle is valid — it was allocated by ferrompi_info_create
            // and has not been freed yet. We only free non-null info objects.
            unsafe { ffi::ferrompi_info_free(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Null info object tests ---

    #[test]
    fn null_info_raw_handle() {
        let info = Info::null();
        assert_eq!(info.raw_handle(), -1);
        // Drop is a no-op for null info (is_null == true)
    }

    #[test]
    fn null_info_set_returns_error() {
        let mut info = Info::null();
        let result = info.set("key", "value");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("MPI_INFO_NULL"), "got: {err_msg}");
    }

    #[test]
    fn null_info_get_returns_error() {
        let info = Info::null();
        let result = info.get("key");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("MPI_INFO_NULL"), "got: {err_msg}");
    }

    // --- CString null-byte rejection tests ---
    //
    // We construct Info { handle: -1, is_null: false } to reach the CString path.
    // Drop check: !is_null (true) && handle >= 0 (-1 is not >= 0) → no FFI call.

    #[test]
    fn set_key_with_null_byte_returns_error() {
        let mut info = Info {
            handle: -1,
            is_null: false,
        };
        let result = info.set("key\0bad", "value");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("null byte"), "got: {err_msg}");
    }

    #[test]
    fn set_value_with_null_byte_returns_error() {
        let mut info = Info {
            handle: -1,
            is_null: false,
        };
        let result = info.set("key", "value\0bad");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("null byte"), "got: {err_msg}");
    }

    #[test]
    fn get_key_with_null_byte_returns_error() {
        let info = Info {
            handle: -1,
            is_null: false,
        };
        let result = info.get("key\0bad");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("null byte"), "got: {err_msg}");
    }
}
