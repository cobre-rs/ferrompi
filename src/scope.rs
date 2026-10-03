//! The nonblocking scope: [`scope`] and [`Scope`].

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use crate::error::{Error, Result};
use crate::ffi;
use crate::request::{Request, RequestKind};
use crate::rt;

const INLINE: usize = 64;

/// A scope for nonblocking requests, created only by [`scope`].
///
/// A constructor that takes a `&Scope` registers the request it returns with
/// the scope. The scope completes every request still pending when its closure
/// returns, or while it unwinds, so a buffer borrowed for the scope is never
/// left to a request MPI still owns.
pub struct Scope<'s, 'env: 's> {
    registry: Registry,
    scope: PhantomData<&'s mut &'s ()>,
    env: PhantomData<&'env mut &'env ()>,
}

/// Runs `f` with a [`Scope`], and returns only when every request created
/// through that scope has completed.
///
/// Requests are created with the scope as their first argument, for example
/// [`Communicator::ibarrier`](crate::Communicator::ibarrier). A request needs
/// no completion call of its own: whatever is still pending when `f` returns
/// is completed by the scope, and `wait`, `test` and the batch completions are
/// for the caller who needs the result earlier. A request belongs to the thread
/// that created it.
///
/// # Examples
///
/// ```no_run
/// use ferrompi::Mpi;
///
/// # fn main() -> ferrompi::Result<()> {
/// let mpi = Mpi::init()?;
/// let world = mpi.world();
///
/// ferrompi::scope(|s| {
///     let req = world.ibarrier(s)?;
///     req.wait()?;
///
///     // Completed when the scope ends.
///     world.ibarrier(s)?;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// # Compile-time guarantees
///
/// A request borrows its buffers, and a user op, for the scope's lifetime `'s`.
/// `'s` is invariant, so the compiler can neither shorten it to end before the
/// scope completes the request nor stretch it past the scope. Each program
/// below does not compile, and is followed by the same program made correct.
///
/// A buffer declared inside the closure would be freed before the scope's final
/// wait, so the first program is rejected; the second declares the buffer
/// outside. The rejection also shows that `'s` is invariant: were `'s`
/// covariant, the compiler could shorten it to the buffer's lifetime and accept
/// the first program.
///
/// ```compile_fail,E0597
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// ferrompi::scope(|s| {
///     let mut buf = vec![0u8; 4];
///     world.irecv(s, &mut buf, 1, 0)?;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     world.irecv(s, &mut buf, 1, 0)?;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// A buffer cannot be dropped while a request borrows it:
///
/// ```compile_fail,E0505
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     world.irecv(s, &mut buf, 1, 0)?;
///     drop(buf);
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     world.irecv(s, &mut buf, 1, 0)?;
///     Ok(())
/// })?;
/// drop(buf);
/// # Ok(())
/// # }
/// ```
///
/// Nor can a receive buffer be written while a request borrows it:
///
/// ```compile_fail,E0499
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     world.irecv(s, &mut buf, 1, 0)?;
///     buf[0] = 1;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     world.irecv(s, &mut buf, 1, 0)?;
///     Ok(())
/// })?;
/// buf[0] = 1;
/// # Ok(())
/// # }
/// ```
///
/// Or read, even after the request was waited, because the borrow lasts until
/// the scope returns:
///
/// ```compile_fail,E0502
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     let req = world.irecv(s, &mut buf, 1, 0)?;
///     req.wait()?;
///     let x = buf[0];
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     let req = world.irecv(s, &mut buf, 1, 0)?;
///     req.wait()?;
///     Ok(())
/// })?;
/// let x = buf[0];
/// # Ok(())
/// # }
/// ```
///
/// A request cannot leave the closure, because it borrows the scope:
///
/// ```compile_fail,E0521
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// let mut keep = None;
/// ferrompi::scope(|s| {
///     keep = Some(world.irecv(s, &mut buf, 1, 0)?);
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut buf = vec![0u8; 4];
/// ferrompi::scope(|s| {
///     let mut keep = None;
///     keep = Some(world.irecv(s, &mut buf, 1, 0)?);
///     if let Some(req) = keep {
///         req.wait()?;
///     }
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// Neither can the scope itself:
///
/// ```compile_fail,E0521
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let mut keep = None;
/// ferrompi::scope(|s| {
///     keep = Some(s);
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::Mpi;
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// ferrompi::scope(|s| {
///     let mut keep = None;
///     keep = Some(s);
///     if let Some(s) = keep {
///         world.ibarrier(s)?;
///     }
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// A [`UserOp`](crate::UserOp) is borrowed by a nonblocking reduction in the
/// same way, so the op cannot be dropped while the reduction is pending:
///
/// ```compile_fail,E0505
/// # use ferrompi::{Mpi, UserOp};
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let op: UserOp<f64> = UserOp::new(|a: &[f64], b: &mut [f64]| {
///     for (x, y) in a.iter().zip(b.iter_mut()) {
///         *y += x;
///     }
/// })?;
/// let send = [1.0f64; 4];
/// let mut recv = [0.0f64; 4];
/// ferrompi::scope(|s| {
///     world.iallreduce(s, &send, &mut recv, &op)?;
///     drop(op);
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// ```no_run
/// # use ferrompi::{Mpi, UserOp};
/// # fn main() -> ferrompi::Result<()> {
/// # let mpi = Mpi::init()?;
/// # let world = mpi.world();
/// let op: UserOp<f64> = UserOp::new(|a: &[f64], b: &mut [f64]| {
///     for (x, y) in a.iter().zip(b.iter_mut()) {
///         *y += x;
///     }
/// })?;
/// let send = [1.0f64; 4];
/// let mut recv = [0.0f64; 4];
/// ferrompi::scope(|s| {
///     world.iallreduce(s, &send, &mut recv, &op)?;
///     Ok(())
/// })?;
/// drop(op);
/// # Ok(())
/// # }
/// ```
///
/// A [`Request`](crate::Request) cannot be sent to another thread; the
/// documentation of `Request` has that fence.
///
/// # Panics and errors
///
/// - If `f` panics, the scope waits for every request, then resumes the panic.
/// - If `f` returns an error, the scope waits for every request and returns
///   that error. Otherwise an error from completing the requests at the end is
///   returned in place of the value.
/// - Opening a scope after [`Mpi`](crate::Mpi) began to drop fails with
///   [`Error::Finalized`] without running `f`.
/// - If the scope cannot establish that its requests completed, it prints a
///   message and aborts the process: the buffers it lent cannot be leaked.
///   That is the case when MPI refuses the final wait, and when an error
///   leaves the state of every request unknown.
///
/// # Finalize
///
/// [`Mpi`](crate::Mpi)'s drop waits for a scope open on another thread, with
/// no timeout, so a scope that never ends (for example one blocked on a peer
/// that is itself finalizing) blocks the drop. A scope open on the thread that
/// drops `Mpi` makes the drop skip `MPI_Finalize`, with a warning on stderr.
/// MPI then stays initialized, and the scope still completes its requests when
/// it ends.
pub fn scope<'env, R>(f: impl for<'s> FnOnce(&'s Scope<'s, 'env>) -> Result<R>) -> Result<R> {
    let token = rt::open_scope().map_err(|code| Error::from_code_with_op(code, "scope"))?;
    let scope = Scope {
        registry: Registry::new(),
        scope: PhantomData,
        env: PhantomData,
    };
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));
    let end = scope.registry.complete_all();
    drop(token);
    match result {
        Err(payload) => resume_unwind(payload),
        Ok(Err(error)) => Err(error),
        Ok(Ok(value)) => end.map(|()| value),
    }
}

impl<'s> Scope<'s, '_> {
    #[inline]
    pub(crate) fn request(&'s self, handle: i64, kind: RequestKind) -> Request<'s> {
        let slot = self.registry.register(handle);
        Request::scoped(handle, kind, &self.registry, slot)
    }
}

/// The handles of a scope's pending requests, one slot each. A slot is a
/// position: handles are never searched or compared. The `MPI_Request` values
/// behind two handles can be equal (a send to `MPI_PROC_NULL` returns the same
/// builtin request every time), which needs no special case.
pub(crate) struct Registry {
    inline: [Cell<i64>; INLINE],
    /// Bit `i` set: `inline[i]` is live.
    used: Cell<u64>,
    /// `-1` marks a free entry.
    spill: RefCell<Vec<i64>>,
}

impl Registry {
    pub(crate) fn new() -> Self {
        Registry {
            inline: [const { Cell::new(0) }; INLINE],
            used: Cell::new(0),
            spill: RefCell::new(Vec::new()),
        }
    }

    fn register(&self, handle: i64) -> u32 {
        let used = self.used.get();
        if used != u64::MAX {
            let slot = (!used).trailing_zeros();
            self.inline[slot as usize].set(handle);
            self.used.set(used | (1 << slot));
            return slot;
        }
        let mut spill = self.spill.borrow_mut();
        let index = match spill.iter().position(|&entry| entry == -1) {
            Some(index) => {
                spill[index] = handle;
                index
            }
            None => {
                spill.push(handle);
                spill.len() - 1
            }
        };
        (INLINE + index) as u32
    }

    pub(crate) fn release(&self, slot: u32) {
        if (slot as usize) < INLINE {
            self.used.set(self.used.get() & !(1 << slot));
        } else {
            self.spill.borrow_mut()[slot as usize - INLINE] = -1;
        }
    }

    fn live(&self) -> usize {
        let spilled = self.spill.borrow().iter().filter(|&&e| e != -1).count();
        self.used.get().count_ones() as usize + spilled
    }

    /// Copies the live slots, inline first, and their handles into `slots` and
    /// `handles`, which are `live()` long.
    fn snapshot(&self, slots: &mut [u32], handles: &mut [i64]) {
        let mut n = 0;
        let mut bits = self.used.get();
        while bits != 0 {
            let slot = bits.trailing_zeros();
            bits &= bits - 1;
            slots[n] = slot;
            handles[n] = self.inline[slot as usize].get();
            n += 1;
        }
        for (index, &entry) in self.spill.borrow().iter().enumerate() {
            if entry != -1 {
                slots[n] = (INLINE + index) as u32;
                handles[n] = entry;
                n += 1;
            }
        }
    }

    /// One `MPI_Waitall` over the live slots, releasing those MPI completed.
    /// Returns its code and whether any slot was released.
    fn wait_pass(&self, slots: &mut [u32], handles: &mut [i64], done: &mut [u8]) -> (c_int, bool) {
        self.snapshot(slots, handles);
        let mut failed: i64 = -1;
        // SAFETY: handles holds one registered request handle per live slot
        // and done is zeroed and as long, which is the count passed; failed is
        // a valid output parameter.
        let ret = unsafe {
            ffi::ferrompi_waitall(
                handles.len() as i64,
                handles.as_ptr(),
                done.as_mut_ptr(),
                &mut failed,
            )
        };
        let mut progress = false;
        for (&slot, &complete) in slots.iter().zip(done.iter()) {
            if complete != 0 {
                self.release(slot);
                progress = true;
            }
        }
        (ret, progress)
    }

    /// Waits until no slot is pending, and returns the first error.
    ///
    /// A failed pass that completed some slots is repeated for the rest, which
    /// MPI reports as `MPI_ERR_PENDING` (Open MPI returns at the first failure
    /// with them still incomplete; MPICH returns once every receive arrived,
    /// with them complete but unprocessed). A pass that completed no slot was
    /// refused or left the state of every request unknown, so the process
    /// aborts.
    pub(crate) fn complete_all(&self) -> Result<()> {
        let mut first = Ok(());
        loop {
            let live = self.live();
            if live == 0 {
                return first;
            }
            let (ret, progress) = if live <= INLINE {
                self.wait_pass(
                    &mut [0; INLINE][..live],
                    &mut [0; INLINE][..live],
                    &mut [0; INLINE][..live],
                )
            } else {
                self.wait_pass(&mut vec![0; live], &mut vec![0; live], &mut vec![0; live])
            };
            if !progress {
                scope_abort(ret);
            }
            if ret != 0 && first.is_ok() {
                first = Error::check_with_op(ret, "scope");
            }
        }
    }
}

#[cold]
#[inline(never)]
fn scope_abort(ret: c_int) -> ! {
    let msg = format!(
        "ferrompi: a nonblocking scope could not complete its requests (MPI_Waitall returned {ret}); aborting\n"
    );
    let _ = std::io::stderr().write_all(msg.as_bytes());
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::{scope, Registry, INLINE};
    use crate::error::Error;
    use crate::rt;

    #[test]
    fn registry_reuses_released_inline_slots() {
        let registry = Registry::new();
        assert_eq!(registry.register(10), 0);
        assert_eq!(registry.register(11), 1);
        assert_eq!(registry.register(12), 2);
        registry.release(1);
        assert_eq!(registry.register(13), 1);
        assert_eq!(registry.register(14), 3);
        assert_eq!(registry.live(), 4);
    }

    #[test]
    fn registry_spills_past_64_and_reuses_spill_slots() {
        let registry = Registry::new();
        for expected in 0..INLINE as u32 {
            assert_eq!(registry.register(100 + i64::from(expected)), expected);
        }
        assert_eq!(registry.register(200), INLINE as u32);
        assert_eq!(registry.register(201), INLINE as u32 + 1);
        assert_eq!(registry.live(), INLINE + 2);

        registry.release(INLINE as u32);
        assert_eq!(registry.register(202), INLINE as u32);
        registry.release(5);
        assert_eq!(registry.register(203), 5);
        assert_eq!(registry.register(204), INLINE as u32 + 2);

        let live = registry.live();
        let mut slots = vec![0; live];
        let mut handles = vec![0; live];
        registry.snapshot(&mut slots, &mut handles);
        assert_eq!(slots.last(), Some(&(INLINE as u32 + 2)));
        assert_eq!(handles[5], 203);
        assert_eq!(handles[INLINE], 202);
        assert_eq!(handles[INLINE + 1], 201);
    }

    #[test]
    fn registry_never_dedupes_equal_handles() {
        let registry = Registry::new();
        let first = registry.register(7);
        let second = registry.register(7);
        assert_ne!(first, second);
        assert_eq!(registry.live(), 2);

        registry.release(first);
        assert_eq!(registry.live(), 1);
        let mut slots = [0];
        let mut handles = [0];
        registry.snapshot(&mut slots, &mut handles);
        assert_eq!((slots[0], handles[0]), (second, 7));
    }

    #[test]
    fn scope_without_requests_makes_no_mpi_call() {
        assert_eq!(Registry::new().live(), 0);
        assert!(Registry::new().complete_all().is_ok());
        assert_eq!(scope(|_| Ok(41)).expect("an empty scope returns Ok"), 41);
    }

    #[test]
    fn scope_returns_the_closure_error() {
        let result = scope::<()>(|_| Err(Error::InvalidState { reason: "closure" }));
        assert!(matches!(
            result,
            Err(Error::InvalidState { reason: "closure" })
        ));
    }

    #[test]
    fn scope_resumes_a_panic_and_releases_its_token() {
        let outcome = std::panic::catch_unwind(|| {
            let _ = scope::<()>(|_| std::panic::resume_unwind(Box::new("scope payload")));
        });
        let payload = outcome.expect_err("the panic must propagate");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"scope payload"));
        assert_eq!(rt::scopes_on_this_thread(), 0);
    }

    #[test]
    fn scope_token_counts_this_thread() {
        assert_eq!(rt::scopes_on_this_thread(), 0);
        scope(|_| {
            assert_eq!(rt::scopes_on_this_thread(), 1);
            scope(|_| {
                assert_eq!(rt::scopes_on_this_thread(), 2);
                Ok(())
            })
        })
        .expect("nested empty scopes return Ok");
        assert_eq!(rt::scopes_on_this_thread(), 0);
    }
}
