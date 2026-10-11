//! The `Message` handle returned by the matched probes.

use crate::datatype::{buf_mut, buf_ptr, DatatypeTag, MpiDatatype};
use crate::error::{Error, Result, FERROMPI_ERR_FINALIZED, FERROMPI_ERR_THREAD_LEVEL};
use crate::ffi;
use crate::rt;
use crate::status::Status;

/// A message that a matched probe took out of the receive queue and that has
/// not been received yet.
///
/// [`Communicator::mprobe`](crate::Communicator::mprobe) and
/// [`Communicator::improbe`](crate::Communicator::improbe) return it with the
/// message's [`Status`]. Only [`mrecv`](Self::mrecv) on this handle receives
/// the message: no other probe or receive, on any thread, can take it. That
/// makes probe-then-receive correct when several threads receive on one
/// communicator, where [`probe`](crate::Communicator::probe) followed by
/// [`recv`](crate::Communicator::recv) can receive a message other than the
/// one that was probed.
///
/// # Example
///
/// ```no_run
/// # use ferrompi::{Mpi, Source, Tag};
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let (message, status) = world.mprobe::<f64>(Source::Any, Tag::Any)?;
/// let mut buf = vec![0.0f64; status.count.expect("whole number of f64")];
/// message.mrecv(&mut buf)?;
/// # Ok(())
/// # }
/// ```
///
/// # Dropping
///
/// MPI requires every matched message to be received (MPI-4.1 §3.8.2, p.98),
/// so dropping a `Message` receives it into a scratch buffer and throws the
/// data away. The scratch receive uses `MPI_BYTE` whatever type the message
/// was sent with. That strictly breaks MPI's type-matching rule
/// (MPI-4.1 §3.3.1, p.46), under which a send and its receive name the same
/// type, but MPICH and Open MPI accept the receive.
///
/// - The drop is an MPI call. On a thread that may not call MPI at the provided
///   thread level, it makes no MPI call and does not abort: the message stays
///   matched and unreceived for the rest of the process.
/// - A drop whose receive fails, whose scratch buffer cannot be allocated, or
///   whose message has a length MPI could not report (for example more than
///   `i32::MAX` bytes on a library without the large-count calls) is treated
///   the same way: the message counts as unreceived for the rest of the
///   process.
/// - While a `Message` is unreceived, whatever thread holds it, dropping
///   [`Mpi`](crate::Mpi) skips `MPI_Finalize` (see [`Mpi`](crate::Mpi)):
///   finalizing with a matched message pending is erroneous (MPI-4.1 §11.2.2,
///   p.489). The message can still be received or dropped afterwards.
///
/// # Compile-time guarantees
///
/// A message is received at most once. `mrecv` takes the handle by value:
///
/// ```compile_fail,E0382
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let (message, _) = world.mprobe::<u64>(1, 9)?;
/// let mut first = [0u64; 4];
/// let mut second = [0u64; 4];
/// message.mrecv(&mut first)?;
/// message.mrecv(&mut second)?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let (message, _) = world.mprobe::<u64>(1, 9)?;
/// let mut first = [0u64; 4];
/// message.mrecv(&mut first)?;
/// # Ok(())
/// # }
/// ```
///
/// and the handle cannot be copied:
///
/// ```compile_fail,E0599
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let (message, _) = world.mprobe::<u64>(1, 9)?;
/// let copy = message.clone();
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # fn f(world: &ferrompi::Communicator) -> ferrompi::Result<()> {
/// let (message, _) = world.mprobe::<u64>(1, 9)?;
/// let moved = message;
/// # Ok(())
/// # }
/// ```
#[must_use = "a matched message that is not received is discarded when dropped"]
pub struct Message {
    /// The `MPI_Message`, as the C layer carries it.
    handle: i64,
    /// The message's length in bytes, for the discard; negative if MPI could
    /// not report it.
    bytes: i64,
    /// Held from the match until MPI received the message. `Mpi::drop` skips
    /// `MPI_Finalize` while it is counted.
    token: Option<rt::HandleToken>,
}

impl Message {
    pub(crate) fn new(handle: i64, bytes: i64, token: Option<rt::HandleToken>) -> Self {
        Message {
            handle,
            bytes,
            token,
        }
    }

    /// Receive the matched message into `buf`.
    ///
    /// Returns the receive's [`Status`], as [`recv`](crate::Communicator::recv)
    /// does: the matched source and tag, and the number of `T` elements
    /// received. A message matched from [`Source::ProcNull`](crate::Source::ProcNull)
    /// returns at once with `Source::ProcNull`, `Tag::Any` and `Some(0)`, and
    /// leaves `buf` untouched.
    ///
    /// `buf` may be longer than the message. With an MPI library that has no
    /// large-count calls, MPI is told of at most `i32::MAX` elements of room,
    /// so a longer `buf` is accepted here, where every other call returns
    /// class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) for a count
    /// above `i32::MAX`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Mpi`] with class
    /// [`MpiErrorClass::Truncate`](crate::MpiErrorClass::Truncate) if the
    /// message does not fit in `buf`. A receive that MPI ran consumes the
    /// message whatever it returned, so the message is then gone.
    ///
    /// A call rejected before reaching MPI ([`Error::ThreadLevelViolation`] on a
    /// thread that may not call MPI, or, in a debug build at `Serialized`,
    /// while another thread's call is in progress) receives nothing; the
    /// message is then dropped as described under [Dropping](Self#dropping).
    pub fn mrecv<T: MpiDatatype>(mut self, buf: &mut [T]) -> Result<Status> {
        let (p, n, dt) = buf_mut(buf);
        let mut status = ffi::FerrompiStatus::default();
        // SAFETY: this blocking call returns only after MPI is done with the buffer.
        // self.handle is a message that a matched probe returned and that no receive
        // was issued for: the token is still held, and it is given up below as soon as
        // MPI ran one.
        let ret = unsafe { ffi::ferrompi_mrecv(p, n, dt, self.handle, &mut status) };
        // A call the guard rejects before MPI sees it leaves the message matched,
        // token included, and the drop of `self` deals with it. Otherwise MPI
        // consumed the message whatever it returned, and the handle must never
        // reach MPI again: MPICH leaves it set after a failed receive.
        if ret != FERROMPI_ERR_THREAD_LEVEL && ret != FERROMPI_ERR_FINALIZED {
            self.token = None;
        }
        Error::check_with_op(ret, "mrecv")?;
        Ok(Status::from_ffi(status))
    }
}

impl Drop for Message {
    /// Receives an unreceived message into a scratch buffer, as `MPI_BYTE`.
    ///
    /// The token is released only once that receive succeeded. On every other
    /// path it is forgotten, so the message keeps counting and `Mpi`'s drop
    /// skips `MPI_Finalize`: a refused call (wrong thread, or `Mpi` already
    /// finalized), a failed receive, an unknown length, or a scratch buffer
    /// that could not be allocated.
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let mut scratch: Vec<u8> = Vec::new();
        let discarded = match usize::try_from(self.bytes) {
            Ok(bytes) if scratch.try_reserve_exact(bytes).is_ok() => {
                let p = buf_ptr(scratch.as_mut_ptr(), bytes, true).cast_mut();
                // SAFETY: `p` is valid for writes of `bytes` bytes: it is the
                // reserved capacity of `scratch`, or a fixed address when `bytes`
                // is zero, where MPI writes nothing. self.handle is a matched,
                // unreceived message, because the token was still held. A null
                // status asks for none.
                let ret = unsafe {
                    ffi::ferrompi_mrecv(
                        p,
                        self.bytes,
                        DatatypeTag::Byte as i32,
                        self.handle,
                        std::ptr::null_mut(),
                    )
                };
                ret == 0
            }
            _ => false,
        };
        if !discarded {
            std::mem::forget(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Message;

    #[test]
    fn message_is_send_and_sync() {
        fn is_send_and_sync<X: Send + Sync>() {}
        is_send_and_sync::<Message>();
    }

    /// A message that MPI already received holds no token, and its drop must
    /// not hand the stale handle to MPI again. Any call here would also run
    /// before `MPI_Init_thread`.
    #[test]
    fn drop_of_a_received_message_makes_no_mpi_call() {
        drop(Message::new(0, 4, None));
    }
}
