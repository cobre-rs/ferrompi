//! Nonblocking collective operations: ibroadcast, iallreduce, ireduce, igather, etc.

use crate::comm::{
    check_rank_slots, check_same_len, rank_block, scatter_inplace_args, Communicator,
};
use crate::datatype::{buf, buf_mut, MpiDatatype};
use crate::error::{Error, Result};
use crate::ffi;
use crate::op::CollectiveOp;
use crate::request::{Request, RequestKind};
use crate::scope::Scope;

impl Communicator {
    // ========================================================================
    // Generic Nonblocking Collectives
    // ========================================================================

    /// Nonblocking broadcast.
    ///
    /// Initiates the broadcast and returns immediately with a [`Request`] that
    /// belongs to the scope `s`. `data` is borrowed mutably for the scope: it
    /// cannot be read or changed until [`scope`](crate::scope) returns, even
    /// after the request was waited, and the scope completes the broadcast at
    /// the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `data` - The payload at the root, and the buffer that receives it at every
    ///   other rank, borrowed for the scope
    /// * `root` - Rank of the root process
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = vec![0.0f64; 100];
    /// ferrompi::scope(|s| {
    ///     let req = world.ibroadcast(s, &mut data, 0)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // data is readable again here.
    /// ```
    #[inline]
    pub fn ibroadcast<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        data: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        let mut request_handle: i64 = 0;
        let (p, n, dt) = buf_mut(data);
        // SAFETY: data is borrowed mutably for 's. The scope completes every request it holds
        // before 's ends, so the buffer outlives the span in which MPI may use it, and nothing
        // touches it meanwhile.
        let ret = unsafe { ffi::ferrompi_ibcast(p, n, dt, root, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibcast")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-reduce.
    ///
    /// Initiates the reduction and returns immediately with a [`Request`] that
    /// belongs to the scope `s`. `send` and a user `op` are borrowed for the
    /// scope, and `recv` is borrowed mutably for it: `recv` cannot be read until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the reduction at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for the result, borrowed for the scope
    /// * `op` - a [`ReduceOp`](crate::ReduceOp), a `&`[`UserOp<T>`](crate::UserOp) (borrowed
    ///   until the scope returns), or a [`CollectiveOp`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] with `arg: "op"` if `op` is a predefined op MPI does not
    ///   define on `T` (a bitwise or logical op on `f32`/`f64`)
    /// - [`Error::BufferSize`] if `send.len() != recv.len()`
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) if
    ///   `op` is a user op and `send.len()` exceeds `i32::MAX`, on every MPI version
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// ferrompi::scope(|s| {
    ///     let req = world.iallreduce(s, &send, &mut recv, ReduceOp::Sum)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn iallreduce<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        op: impl Into<CollectiveOp<'s, T>>,
    ) -> Result<Request<'s>> {
        let op = op.into().code(T::TAG)?;
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). send and recv are borrowed for 's (recv mutably), and a user op
        // is borrowed for 's through `op`. The scope completes every request it holds before 's
        // ends, so the buffers and the op outlive the span in which MPI may use them, and
        // nothing touches the buffers meanwhile.
        let ret = unsafe {
            ffi::ferrompi_iallreduce(sp, rp, n, dt, op, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iallreduce")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking reduce to root.
    ///
    /// Initiates a reduction operation and returns immediately with a [`Request`]
    /// that belongs to the scope `s`. `send` and a user `op` are borrowed for the
    /// scope, and `recv` is borrowed mutably for it: `recv` cannot be read until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the reduction at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for the result at the root, which must have
    ///   `send.len()` elements there; ignored at other ranks, where it may be
    ///   empty. Borrowed for the scope on every rank
    /// * `op` - a [`ReduceOp`](crate::ReduceOp), a `&`[`UserOp<T>`](crate::UserOp) (borrowed
    ///   until the scope returns), or a [`CollectiveOp`]
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] with `arg: "op"` if `op` is a predefined op MPI does not
    ///   define on `T` (a bitwise or logical op on `f32`/`f64`)
    /// - [`Error::BufferSize`] if this rank is `root` and `recv.len() != send.len()`
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) if
    ///   `op` is a user op and `send.len()` exceeds `i32::MAX`, on every MPI version
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// ferrompi::scope(|s| {
    ///     let req = world.ireduce(s, &send, &mut recv, ReduceOp::Sum, 0)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn ireduce<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        op: impl Into<CollectiveOp<'s, T>>,
        root: i32,
    ) -> Result<Request<'s>> {
        let op = op.into().code(T::TAG)?;
        if self.rank == root {
            check_same_len("recv", send.len(), recv.len())?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above at the root; MPI ignores recvbuf at
        // every other rank (MPI-4.1 section 6.9.1). The two slices cannot alias (&[T] vs
        // &mut [T]). send and recv are borrowed for 's (recv mutably), and a user op is borrowed
        // for 's through `op`. The scope completes every request it holds before 's ends, so the
        // buffers and the op outlive the span in which MPI may use them, and nothing touches the
        // buffers meanwhile.
        let ret = unsafe {
            ffi::ferrompi_ireduce(sp, rp, n, dt, op, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "ireduce")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking gather to root.
    ///
    /// Initiates a gather operation and returns immediately with a [`Request`]
    /// that belongs to the scope `s`. Each process sends `send.len()` elements.
    /// Root receives `send.len() * size` elements total. `send` is borrowed for
    /// the scope, and `recv` is borrowed mutably for it: `recv` cannot be read
    /// until [`scope`](crate::scope) returns, even after the request was
    /// waited, and the scope completes the gather at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for received data (only significant at root), borrowed for
    ///   the scope on every rank
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if this rank is `root` and `recv.len() < send.len() *
    /// size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as f64; 5];
    /// let mut recv = vec![0.0f64; 5 * world.size() as usize];
    /// ferrompi::scope(|s| {
    ///     let req = world.igather(s, &send, &mut recv, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn igather<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        if self.rank == root {
            check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]); at non-root, recv is
        // ignored by MPI. The root-side receive-length relation (recv.len() >= send.len()
        // * size) is checked above. send and recv are borrowed for 's (recv mutably). The
        // scope completes every request it holds before 's ends, so the buffers outlive the
        // span in which MPI may use them, and nothing touches them meanwhile.
        let ret = unsafe {
            ffi::ferrompi_igather(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "igather")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-gather.
    ///
    /// Initiates an all-gather operation and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. Each process sends
    /// `send.len()` elements and receives from all. `send` is borrowed for the
    /// scope, and `recv` is borrowed mutably for it: `recv` cannot be read until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the all-gather at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope
    /// * `recv` - Buffer for the data of every rank, borrowed for the scope
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if `recv.len() < send.len() * size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![world.rank() as i32; 3];
    /// let mut recv = vec![0i32; 3 * world.size() as usize];
    /// ferrompi::scope(|s| {
    ///     let req = world.iallgather(s, &send, &mut recv)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn iallgather<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
    ) -> Result<Request<'s>> {
        check_rank_slots("recv", recv.len(), send.len(), self.size)?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). The every-rank
        // receive-length relation (recv.len() >= send.len() * size) is checked above. send
        // and recv are borrowed for 's (recv mutably). The scope completes every request it
        // holds before 's ends, so the buffers outlive the span in which MPI may use them,
        // and nothing touches them meanwhile.
        let ret =
            unsafe { ffi::ferrompi_iallgather(sp, n, rp, n, dt, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "iallgather")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking scatter from root.
    ///
    /// Initiates a scatter operation and returns immediately with a [`Request`]
    /// that belongs to the scope `s`. Root sends `recv.len() * size` elements
    /// total, each process receives `recv.len()` elements. `send` is borrowed
    /// for the scope, and `recv` is borrowed mutably for it: `recv` cannot be
    /// read until [`scope`](crate::scope) returns, even after the request was
    /// waited, and the scope completes the scatter at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to scatter (only significant at root), borrowed for the scope
    ///   on every rank
    /// * `recv` - Buffer for this rank's block, borrowed for the scope
    /// * `root` - Rank of the root process
    ///
    /// # Errors
    ///
    /// [`Error::BufferSize`] if this rank is `root` and `send.len() < recv.len() *
    /// size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![0.0f64; 5 * world.size() as usize];
    /// let mut recv = vec![0.0f64; 5];
    /// ferrompi::scope(|s| {
    ///     let req = world.iscatter(s, &send, &mut recv, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn iscatter<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        if self.rank == root {
            check_rank_slots("send", send.len(), recv.len(), self.size)?;
        }
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send is ignored by MPI at non-root; send and recv cannot alias (&[T] vs
        // &mut [T]). The root-side send-length relation (send.len() >= recv.len() * size)
        // is checked above. send and recv are borrowed for 's (recv mutably). The scope
        // completes every request it holds before 's ends, so the buffers outlive the span
        // in which MPI may use them, and nothing touches them meanwhile.
        let ret = unsafe {
            ffi::ferrompi_iscatter(sp, n, rp, n, dt, root, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "iscatter")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking barrier.
    ///
    /// Initiates a barrier synchronization and returns immediately with a
    /// [`Request`] handle that belongs to the scope `s`. The barrier is complete
    /// when the request is waited on, or when the scope ends.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// ferrompi::scope(|s| {
    ///     let req = world.ibarrier(s)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    #[inline]
    pub fn ibarrier<'s>(&self, s: &'s Scope<'s, '_>) -> Result<Request<'s>> {
        let mut request_handle: i64 = 0;
        // SAFETY: this call takes only the communicator handle and a request out-pointer;
        // no buffer is involved, so the request has no buffer-lifetime obligation, and
        // the scope completes it.
        let ret = unsafe { ffi::ferrompi_ibarrier(self.handle, &mut request_handle) };
        Error::check_with_op(ret, "ibarrier")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking inclusive prefix reduction (scan).
    ///
    /// Initiates an inclusive scan and returns immediately with a [`Request`]
    /// that belongs to the scope `s`. On rank `i`, `recv` will contain the
    /// reduction of `send` values from ranks `0..=i` once the request completes.
    /// `send` and a user `op` are borrowed for the scope, and `recv` is borrowed
    /// mutably for it: `recv` cannot be read until [`scope`](crate::scope)
    /// returns, even after the request was waited, and the scope completes the
    /// scan at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to contribute from this process, borrowed for the scope
    /// * `recv` - Buffer for the prefix-reduced result (must be same length as
    ///   `send`), borrowed for the scope
    /// * `op` - a [`ReduceOp`](crate::ReduceOp), a `&`[`UserOp<T>`](crate::UserOp) (borrowed
    ///   until the scope returns), or a [`CollectiveOp`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] with `arg: "op"` if `op` is a predefined op MPI does not
    ///   define on `T` (a bitwise or logical op on `f32`/`f64`)
    /// - [`Error::BufferSize`] if `send.len() != recv.len()`
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) if
    ///   `op` is a user op and `send.len()` exceeds `i32::MAX`, on every MPI version
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// ferrompi::scope(|s| {
    ///     let req = world.iscan(s, &send, &mut recv, ReduceOp::Sum)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // On rank i, recv[j] == (i + 1) * send[j]
    /// ```
    #[inline]
    pub fn iscan<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        op: impl Into<CollectiveOp<'s, T>>,
    ) -> Result<Request<'s>> {
        let op = op.into().code(T::TAG)?;
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). send and recv are borrowed for 's (recv mutably), and a user op
        // is borrowed for 's through `op`. The scope completes every request it holds before 's
        // ends, so the buffers and the op outlive the span in which MPI may use them, and
        // nothing touches the buffers meanwhile.
        let ret =
            unsafe { ffi::ferrompi_iscan(sp, rp, n, dt, op, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "iscan")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking exclusive prefix reduction (exscan).
    ///
    /// Initiates an exclusive scan and returns immediately with a [`Request`]
    /// that belongs to the scope `s`. On rank `i`, `recv` will contain the
    /// reduction of `send` values from ranks `0..i` once the request completes.
    /// `send` and a user `op` are borrowed for the scope, and `recv` is borrowed
    /// mutably for it: `recv` cannot be read until [`scope`](crate::scope)
    /// returns, even after the request was waited, and the scope completes the
    /// scan at the latest then.
    ///
    /// # Rank 0 Behavior
    ///
    /// **Per the MPI standard, the contents of `recv` on rank 0 are undefined.**
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to contribute from this process, borrowed for the scope
    /// * `recv` - Buffer for the prefix-reduced result (must be same length as
    ///   `send`; **undefined on rank 0**), borrowed for the scope
    /// * `op` - a [`ReduceOp`](crate::ReduceOp), a `&`[`UserOp<T>`](crate::UserOp) (borrowed
    ///   until the scope returns), or a [`CollectiveOp`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] with `arg: "op"` if `op` is a predefined op MPI does not
    ///   define on `T` (a bitwise or logical op on `f32`/`f64`)
    /// - [`Error::BufferSize`] if `send.len() != recv.len()`
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) if
    ///   `op` is a user op and `send.len()` exceeds `i32::MAX`, on every MPI version
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let send = vec![1.0f64; 10];
    /// let mut recv = vec![0.0f64; 10];
    /// ferrompi::scope(|s| {
    ///     let req = world.iexscan(s, &send, &mut recv, ReduceOp::Sum)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // On rank i > 0, recv[j] == i * send[j]
    /// // On rank 0, recv is undefined per the MPI standard.
    /// ```
    #[inline]
    pub fn iexscan<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        op: impl Into<CollectiveOp<'s, T>>,
    ) -> Result<Request<'s>> {
        let op = op.into().code(T::TAG)?;
        check_same_len("recv", send.len(), recv.len())?;
        let mut request_handle: i64 = 0;
        let (sp, n, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). send and recv are borrowed for 's (recv mutably), and a user op
        // is borrowed for 's through `op`. The scope completes every request it holds before 's
        // ends, so the buffers and the op outlive the span in which MPI may use them, and
        // nothing touches the buffers meanwhile. MPI leaves recv undefined on rank 0,
        // documented above.
        let ret =
            unsafe { ffi::ferrompi_iexscan(sp, rp, n, dt, op, self.handle, &mut request_handle) };
        Error::check_with_op(ret, "iexscan")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking all-to-all personalized communication.
    ///
    /// Initiates an all-to-all operation and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. `send` is borrowed for the
    /// scope, and `recv` is borrowed mutably for it: `recv` cannot be read until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the all-to-all at the latest then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send, one block per rank, borrowed for the scope
    /// * `recv` - Buffer for the blocks received from every rank (same length as
    ///   `send`), borrowed for the scope
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferSize`] if `send.len() != recv.len()`, or
    /// [`Error::InvalidArgument`] if `send.len()` is not evenly divisible by the
    /// communicator size.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![world.rank() as f64; size * 3];
    /// let mut recv = vec![0.0f64; size * 3];
    /// ferrompi::scope(|s| {
    ///     let req = world.ialltoall(s, &send, &mut recv)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn ialltoall<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
    ) -> Result<Request<'s>> {
        check_same_len("recv", send.len(), recv.len())?;
        let count = rank_block("send", send.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (sp, _, dt) = buf(send);
        let (rp, _, _) = buf_mut(recv);
        // SAFETY: send and recv cannot alias (&[T] vs &mut [T]). send.len() == recv.len()
        // and divisibility by size are both verified above. send and recv are borrowed for
        // 's (recv mutably). The scope completes every request it holds before 's ends, so
        // the buffers outlive the span in which MPI may use them, and nothing touches them
        // meanwhile.
        let ret = unsafe {
            ffi::ferrompi_ialltoall(sp, count, rp, count, dt, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "ialltoall")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking reduce-scatter with uniform block size.
    ///
    /// Initiates a reduce-scatter operation and returns immediately with a
    /// [`Request`] that belongs to the scope `s`. Performs an element-wise
    /// reduction across all processes, then scatters the result so that each
    /// process receives `recv.len()` elements. `send` and a user `op` are
    /// borrowed for the scope, and `recv` is borrowed mutably for it: `recv`
    /// cannot be read until [`scope`](crate::scope) returns, even after the
    /// request was waited, and the scope completes the reduction at the latest
    /// then.
    ///
    /// # Arguments
    ///
    /// * `s` - The scope that owns the request
    /// * `send` - Data to send from this process, borrowed for the scope; it must
    ///   have exactly `recv.len() * size` elements
    /// * `recv` - Buffer for this rank's block of the result, borrowed for the scope
    /// * `op` - a [`ReduceOp`](crate::ReduceOp), a `&`[`UserOp<T>`](crate::UserOp) (borrowed
    ///   until the scope returns), or a [`CollectiveOp`]
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] with `arg: "op"` if `op` is a predefined op MPI does not
    ///   define on `T` (a bitwise or logical op on `f32`/`f64`)
    /// - [`Error::InvalidArgument`] if `send.len()` is not evenly divisible by the
    ///   communicator size
    /// - [`Error::BufferSize`] if `send.len() != recv.len() * size`
    /// - [`Error::Mpi`] with class [`MpiErrorClass::Count`](crate::MpiErrorClass::Count) if
    ///   `op` is a user op and `send.len()` exceeds `i32::MAX`, on every MPI version
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::{Mpi, ReduceOp};
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let size = world.size() as usize;
    /// let send = vec![1.0f64; size * 5];
    /// let mut recv = vec![0.0f64; 5];
    /// ferrompi::scope(|s| {
    ///     let req = world.ireduce_scatter_block(s, &send, &mut recv, ReduceOp::Sum)?;
    ///     // ... do other work ...
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// // recv is readable again here.
    /// ```
    #[inline]
    pub fn ireduce_scatter_block<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        send: &'s [T],
        recv: &'s mut [T],
        op: impl Into<CollectiveOp<'s, T>>,
    ) -> Result<Request<'s>> {
        let op = op.into().code(T::TAG)?;
        check_same_len(
            "recv",
            rank_block("send", send.len(), self.size)?,
            recv.len(),
        )?;
        let mut request_handle: i64 = 0;
        let (sp, _, _) = buf(send);
        let (rp, n, dt) = buf_mut(recv);
        // SAFETY: send.len() == recv.len() * size is verified above; the two slices cannot alias
        // (&[T] vs &mut [T]). send and recv are borrowed for 's (recv mutably), and a user op
        // is borrowed for 's through `op`. The scope completes every request it holds before 's
        // ends, so the buffers and the op outlive the span in which MPI may use them, and
        // nothing touches the buffers meanwhile.
        let ret = unsafe {
            ffi::ferrompi_ireduce_scatter_block(sp, rp, n, dt, op, self.handle, &mut request_handle)
        };
        Error::check_with_op(ret, "ireduce_scatter_block")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place gather. At root, `data` is both the send contribution
    /// and the receive buffer. At non-root, `data` is that rank's own block (here its
    /// send block), as for [`iscatter_inplace`](Self::iscatter_inplace).
    ///
    /// The returned [`Request`] belongs to the scope `s`. `data` is borrowed
    /// mutably for the scope: it cannot be read or changed until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the gather at the latest then.
    ///
    /// # Buffer Layout
    ///
    /// At root, `data` must have length `recvcount * size()` where `recvcount` is
    /// the per-rank count. Rank `r`'s contribution lives at offset
    /// `r * recvcount`. Root's own contribution must be pre-written into
    /// `data[rank() * recvcount .. (rank()+1) * recvcount]` before the call.
    ///
    /// At non-root, `data` is the rank's block of `recvcount` elements. Only root
    /// knows `recvcount`, so a block of another length is not checked locally; MPI
    /// reports the mismatch.
    ///
    /// # Errors
    ///
    /// `Error::InvalidArgument` at root if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = if world.rank() == 0 {
    ///     vec![0i32; 4 * world.size() as usize]
    /// } else {
    ///     vec![world.rank(); 4]
    /// };
    /// ferrompi::scope(|s| {
    ///     let req = world.igather_inplace(s, &mut data, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    #[inline]
    pub fn igather_inplace<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        data: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        let mut request_handle: i64 = 0;
        let ret = if self.rank() == root {
            let recvcount = rank_block("data", data.len(), self.size)? as i64;
            let (p, _, dt) = buf_mut(data);
            // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
            // this NULL is unambiguous); ferrompi_igather maps it to MPI_IN_PLACE, so data
            // serves as both root's send contribution and the receive buffer. recvcount is
            // checked to evenly divide data.len() above, and this branch runs only at root,
            // the only rank MPI_IN_PLACE is valid for in MPI_Igather. data is borrowed mutably
            // for 's. The scope completes every request it holds before 's ends, so the buffer
            // outlives the span in which MPI may use it, and nothing touches it meanwhile.
            unsafe {
                ffi::ferrompi_igather(
                    std::ptr::null(),
                    0,
                    p,
                    recvcount,
                    dt,
                    root,
                    self.handle,
                    &mut request_handle,
                )
            }
        } else {
            let (p, n, dt) = buf_mut(data);
            // SAFETY: p is non-null, so ferrompi_igather does not map it to MPI_IN_PLACE; data is
            // this rank's send block. It is also passed as recvbuf because strict MPI builds
            // reject a NULL recvbuf at non-root; MPI ignores recvbuf and recvcount there. data
            // is borrowed mutably for 's. The scope completes every request it holds before 's
            // ends, so the buffer outlives the span in which MPI may use it, and nothing
            // touches it meanwhile.
            unsafe { ffi::ferrompi_igather(p, n, p, n, dt, root, self.handle, &mut request_handle) }
        };
        Error::check_with_op(ret, "igather_inplace")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place all-gather. Every rank's `data` is both send
    /// contribution and receive buffer.
    ///
    /// The returned [`Request`] belongs to the scope `s`. `data` is borrowed
    /// mutably for the scope: it cannot be read or changed until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the all-gather at the latest then.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `recvcount * size()`. Rank `r`'s contribution
    /// lives at offset `r * recvcount` and must be pre-written before the call.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let rank = world.rank() as usize;
    /// let size = world.size() as usize;
    /// let mut data = vec![0i32; size];
    /// data[rank] = rank as i32 * 10;
    /// ferrompi::scope(|s| {
    ///     let req = world.iallgather_inplace(s, &mut data)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    #[inline]
    pub fn iallgather_inplace<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        data: &'s mut [T],
    ) -> Result<Request<'s>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_iallgather maps it to MPI_IN_PLACE. recvcount
        // is checked to evenly divide data.len() above; each rank's slot must be pre-written
        // by the caller before this call. data is borrowed mutably for 's. The scope completes
        // every request it holds before 's ends, so the buffer outlives the span in which MPI
        // may use it, and nothing touches it meanwhile.
        let ret = unsafe {
            ffi::ferrompi_iallgather(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iallgather_inplace")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place scatter. At root, `data` is the `sendcount * size()`
    /// send buffer; root's own slot is retained in place. At non-root, `data` is
    /// the `recvcount`-element receive buffer.
    ///
    /// The returned [`Request`] belongs to the scope `s`. `data` is borrowed
    /// mutably for the scope: it cannot be read or changed until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the scatter at the latest then.
    ///
    /// # Buffer Layout (root)
    ///
    /// `data` must have length `sendcount * size()`. Rank `r`'s slot is
    /// `data[r*sendcount .. (r+1)*sendcount]`. After the wait, only root's own
    /// slot is guaranteed to remain intact.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` at root if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let mut data = if world.rank() == 0 {
    ///     vec![0i32, 10, 20, 30]
    /// } else {
    ///     vec![0i32; 1]
    /// };
    /// ferrompi::scope(|s| {
    ///     let req = world.iscatter_inplace(s, &mut data, 0)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    #[inline]
    pub fn iscatter_inplace<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        data: &'s mut [T],
        root: i32,
    ) -> Result<Request<'s>> {
        let (sendbuf, sendcount, recvbuf, recvcount, dt) =
            scatter_inplace_args(data, self.rank() == root, self.size)?;
        let mut request_handle: i64 = 0;
        // SAFETY: at root, recvbuf is NULL, the in-place marker (buf's pointer is never
        // null, so this NULL is unambiguous); ferrompi_iscatter maps it to MPI_IN_PLACE so
        // root's own slot is retained. scatter_inplace_args checks that the block size
        // evenly divides data.len(). At non-root, sendbuf is null, which the MPI standard
        // ignores on non-root scatter. data is borrowed mutably for 's. The scope completes
        // every request it holds before 's ends, so the buffer outlives the span in which
        // MPI may use it, and nothing touches it meanwhile.
        let ret = unsafe {
            ffi::ferrompi_iscatter(
                sendbuf,
                sendcount,
                recvbuf,
                recvcount,
                dt,
                root,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "iscatter_inplace")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }

    /// Nonblocking in-place all-to-all personalized communication. `data` is
    /// both send and receive buffer on every rank.
    ///
    /// The returned [`Request`] belongs to the scope `s`. `data` is borrowed
    /// mutably for the scope: it cannot be read or changed until
    /// [`scope`](crate::scope) returns, even after the request was waited, and
    /// the scope completes the all-to-all at the latest then.
    ///
    /// Before the call, rank `r` must pre-write into slot `s` (at offset
    /// `s * count`) the payload it wishes to send to rank `s`. After the wait,
    /// the same slot contains the data received FROM rank `s`.
    ///
    /// # Buffer Layout
    ///
    /// `data` must have length `count * size()`. Slot `s` at
    /// `data[s*count..(s+1)*count]` holds data sent to (and later received from)
    /// rank `s`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` if `data.len()` is not divisible by `size()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use ferrompi::Mpi;
    /// # let mpi = Mpi::init().unwrap();
    /// # let world = mpi.world();
    /// let r = world.rank() as i32;
    /// let size = world.size() as usize;
    /// let mut data: Vec<i32> = (0..size as i32).map(|s| r * 10 + s).collect();
    /// ferrompi::scope(|s| {
    ///     let req = world.ialltoall_inplace(s, &mut data)?;
    ///     req.wait()?;
    ///     Ok(())
    /// })
    /// .unwrap();
    /// ```
    #[inline]
    pub fn ialltoall_inplace<'s, T: MpiDatatype>(
        &self,
        s: &'s Scope<'s, '_>,
        data: &'s mut [T],
    ) -> Result<Request<'s>> {
        let recvcount = rank_block("data", data.len(), self.size)? as i64;
        let mut request_handle: i64 = 0;
        let (p, _, dt) = buf_mut(data);
        // SAFETY: NULL sendbuf is the in-place marker (buf_mut's pointer is never null, so
        // this NULL is unambiguous); ferrompi_ialltoall maps it to MPI_IN_PLACE. recvcount
        // is checked to evenly divide data.len() above; the caller must pre-write each slot
        // before calling this method. data is borrowed mutably for 's. The scope completes
        // every request it holds before 's ends, so the buffer outlives the span in which MPI
        // may use it, and nothing touches it meanwhile.
        let ret = unsafe {
            ffi::ferrompi_ialltoall(
                std::ptr::null(),
                0,
                p,
                recvcount,
                dt,
                self.handle,
                &mut request_handle,
            )
        };
        Error::check_with_op(ret, "ialltoall_inplace")?;
        Ok(s.request(request_handle, RequestKind::Collective))
    }
}

#[cfg(test)]
mod tests {
    use crate::comm::test_comm;
    use crate::error::Error;
    use crate::ReduceOp;

    #[test]
    fn iallreduce_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = crate::scope(|s| {
            comm.iallreduce(s, &send, &mut recv, ReduceOp::Sum)
                .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn ireduce_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = crate::scope(|s| {
            comm.ireduce(s, &send, &mut recv, ReduceOp::Sum, 0)
                .map(|_| ())
        });
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn iscan_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = crate::scope(|s| comm.iscan(s, &send, &mut recv, ReduceOp::Sum).map(|_| ()));
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn iexscan_mismatched_buffers_returns_buffer_size() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let result = crate::scope(|s| comm.iexscan(s, &send, &mut recv, ReduceOp::Sum).map(|_| ()));
        assert!(matches!(
            result,
            Err(Error::BufferSize {
                arg: "recv",
                required: 10,
                actual: 5
            })
        ));
    }

    #[test]
    fn nonblocking_reductions_report_a_bad_op_before_the_buffers() {
        let comm = test_comm(0, 1);
        let send = vec![1.0f64; 10];
        let mut recv = vec![0.0f64; 5];
        let op = ReduceOp::BitwiseOr;
        let results = [
            crate::scope(|s| comm.iallreduce(s, &send, &mut recv, op).map(|_| ())),
            crate::scope(|s| comm.ireduce(s, &send, &mut recv, op, 0).map(|_| ())),
            crate::scope(|s| comm.iscan(s, &send, &mut recv, op).map(|_| ())),
            crate::scope(|s| comm.iexscan(s, &send, &mut recv, op).map(|_| ())),
            crate::scope(|s| {
                comm.ireduce_scatter_block(s, &send, &mut recv, op)
                    .map(|_| ())
            }),
        ];
        for result in results {
            assert!(matches!(
                result,
                Err(Error::InvalidArgument {
                    arg: "op",
                    reason: "bitwise and logical ops do not apply to floating-point types"
                })
            ));
        }
    }

    #[test]
    fn iallgather_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let mut data = vec![0u32; 7];
        let result = crate::scope(|s| comm.iallgather_inplace(s, &mut data).map(|_| ()));
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }

    #[test]
    fn ialltoall_inplace_mismatched_len_returns_invalid_argument() {
        let comm = test_comm(0, 4);
        let mut data = vec![0u32; 7];
        let result = crate::scope(|s| comm.ialltoall_inplace(s, &mut data).map(|_| ()));
        assert!(matches!(
            result,
            Err(Error::InvalidArgument {
                arg: "data",
                reason: "length is not a multiple of the communicator size"
            })
        ));
    }
}
