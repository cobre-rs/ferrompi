//! A persistent request owns its buffers, and nothing safe reaches them while
//! MPI may use them.
//!
//! Every part runs on rank 0 against itself, at `ThreadLevel::Funneled`, with
//! one tag per part:
//! 1. The accessors return `Error::InvalidState` while a request is active
//!    (the send buffer of a persistent allreduce too, when the library has
//!    persistent collectives). `into_buffer` on an active receive waits for
//!    it and returns the sent data.
//! 2. A receive dropped while active, on a self pair: the send copies its data
//!    when it matches, before the drop runs, so this part checks that the drop
//!    path runs cleanly under valgrind and that a victim allocation of the same
//!    size, made right after the drop, is not written. That `Drop` waits before
//!    it frees the buffer rests on its code order (MPI-4.1 §3.9), and
//!    `test_persistent_drop_order` observes it with a delayed send.
//! 3. A loop that rewrites the buffers between iterations allocates nothing
//!    per iteration: a counting global allocator wraps `System` and counts the
//!    allocations Rust code makes (MPI's own `malloc`s are not counted). After
//!    10 warm-up iterations, 1000 iterations each of `start`/`wait` over a self
//!    pair, `start_all`/`wait_all` over the same pair, and a persistent
//!    allreduce rewritten through `send_buffer_mut`, leave the counter
//!    unchanged. A plain `vec!` raises it, which shows the counter works.
//! 4. A receive forgotten while active leaks its buffer, so the send that
//!    matches it writes into memory nothing else owns: a victim allocation of
//!    the same size is unchanged, and valgrind sees no invalid write. This part
//!    runs last, because the forgotten request stays active: `Mpi`'s drop then
//!    skips `MPI_Finalize` (MPI-4.1 §11.2.2).
//!
//! No check after the drop can use `common::check`, whose allreduce is refused
//! by then. A failed check prints `FAIL:` and exits with status 2, which the
//! runner reports as a failure; status 1 would be accepted under
//! `expect=unfinalized`.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_persistent_ownership
// mpi-test: np=1 expect=unfinalized valgrind
// mpi-test-stderr: persistent request(s) still active

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrompi::{Communicator, Error, Mpi, PersistentRequest, ReduceOp, ThreadLevel};

mod common;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract, and only adds a counter increment.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's layout is forwarded as received.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from this allocator, which forwards
        // both to `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` and `layout` came from this allocator, which forwards
        // both to `System`; `new_size` is forwarded as received.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const VICTIM: u64 = 0xAAAA_AAAA_AAAA_AAAA;
const SENT: u64 = 0x5151_5151_5151_5151;

fn check(ok: bool, name: &str) {
    if !ok {
        eprintln!("FAIL: {name}");
        std::process::exit(2);
    }
}

fn is_active_error<T>(result: ferrompi::Result<T>) -> bool {
    matches!(
        result,
        Err(Error::InvalidState {
            reason: "request is active",
            ..
        })
    )
}

fn part1_access_while_active(world: &Communicator, collectives: bool) {
    let mut recv = world
        .recv_init(vec![0u64; 4], 0, 1)
        .expect("part 1: recv_init");
    let mut send = world
        .send_init(vec![1u64, 2, 3, 4], 0, 1)
        .expect("part 1: send_init");
    recv.start().expect("part 1: start (recv)");
    send.start().expect("part 1: start (send)");
    let refused = is_active_error(recv.buffer()) && is_active_error(recv.buffer_mut());

    let mut sum_refused = true;
    let mut sum_result = true;
    if collectives {
        let mut sum = world
            .allreduce_init(vec![1.5f64; 8], vec![0.0f64; 8], ReduceOp::Sum)
            .expect("part 1: allreduce_init");
        sum.start().expect("part 1: start (allreduce)");
        sum_refused = is_active_error(sum.send_buffer_mut());
        sum.wait().expect("part 1: wait (allreduce)");
        sum_result = sum.buffer().is_ok_and(|b| b == [1.5; 8]);
    }
    common::check(
        world,
        refused,
        "part 1: buffer and buffer_mut are refused while the receive is active",
    );
    common::check(
        world,
        sum_refused,
        "part 1: send_buffer_mut is refused while the allreduce is active",
    );
    common::check(
        world,
        sum_result,
        "part 1: the allreduce delivered its result",
    );

    let received = recv.into_buffer().expect("part 1: into_buffer");
    send.wait().expect("part 1: wait (send)");
    common::check(
        world,
        received == [1, 2, 3, 4],
        "part 1: into_buffer on an active receive waits and returns the sent data",
    );
    common::check(
        world,
        send.buffer().is_ok_and(|b| b == [1, 2, 3, 4]),
        "part 1: the completed send's buffer is accessible",
    );
}

fn part2_drop_while_active(world: &Communicator) {
    let mut recv = world
        .recv_init(vec![0u64; 4], 0, 2)
        .expect("part 2: recv_init");
    let mut send = world
        .send_init(vec![SENT; 4], 0, 2)
        .expect("part 2: send_init");
    recv.start().expect("part 2: start (recv)");
    send.start().expect("part 2: start (send)");
    drop(recv);
    let victim = black_box(vec![VICTIM; 4]);
    send.wait().expect("part 2: wait (send)");
    common::check(
        world,
        victim == [VICTIM; 4],
        "part 2: the drop of an active receive leaves a later allocation alone",
    );
}

/// One self exchange: `iter` goes through the send's buffer, both requests
/// start and complete, and the receive's buffer must hold `iter`.
fn exchange(
    recv: &mut PersistentRequest<u64>,
    send: &mut PersistentRequest<u64>,
    iter: u64,
) -> bool {
    send.buffer_mut().expect("part 3: send buffer_mut")[0] = iter;
    recv.start().expect("part 3: start (recv)");
    send.start().expect("part 3: start (send)");
    send.wait().expect("part 3: wait (send)");
    recv.wait().expect("part 3: wait (recv)");
    recv.buffer().expect("part 3: recv buffer")[0] == iter
}

/// The same exchange through `start_all` and `wait_all` over `[recv, send]`.
fn exchange_all(pair: &mut [PersistentRequest<u64>; 2], iter: u64) -> bool {
    pair[1].buffer_mut().expect("part 3: send buffer_mut")[0] = iter;
    PersistentRequest::start_all(pair).expect("part 3: start_all");
    PersistentRequest::wait_all(pair).expect("part 3: wait_all");
    pair[0].buffer().expect("part 3: recv buffer")[0] == iter
}

/// One persistent allreduce on one rank: the send buffer is rewritten, and the
/// result must equal it.
fn reduce(sum: &mut PersistentRequest<f64>, iter: u64) -> bool {
    let mut want = [0.0f64; 8];
    for (i, x) in want.iter_mut().enumerate() {
        *x = (iter + i as u64) as f64;
    }
    sum.send_buffer_mut()
        .expect("part 3: allreduce send_buffer_mut")
        .copy_from_slice(&want);
    sum.start().expect("part 3: start (allreduce)");
    sum.wait().expect("part 3: wait (allreduce)");
    sum.buffer().expect("part 3: allreduce buffer") == want
}

fn part3_zero_allocations(world: &Communicator, collectives: bool) {
    let before = ALLOCS.load(Ordering::Relaxed);
    drop(black_box(vec![0u64; 4]));
    let after = ALLOCS.load(Ordering::Relaxed);
    common::check(world, after > before, "part 3: a vec! raises the counter");

    let recv = world
        .recv_init(vec![0u64; 8], 0, 3)
        .expect("part 3: recv_init");
    let send = world
        .send_init(vec![0u64; 8], 0, 3)
        .expect("part 3: send_init");
    let mut pair = [recv, send];

    let [recv, send] = &mut pair;
    for iter in 0..10 {
        exchange(recv, send, iter);
    }
    let before = ALLOCS.load(Ordering::Relaxed);
    let mut delivered = true;
    for iter in 10..1010 {
        delivered &= exchange(recv, send, iter);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    common::check(
        world,
        after == before,
        &format!(
            "part 3: 1000 start/wait iterations allocated {} times",
            after - before
        ),
    );
    common::check(
        world,
        delivered,
        "part 3: start/wait delivered every iteration's data",
    );

    for iter in 0..10 {
        exchange_all(&mut pair, iter);
    }
    let before = ALLOCS.load(Ordering::Relaxed);
    let mut delivered = true;
    for iter in 10..1010 {
        delivered &= exchange_all(&mut pair, iter);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    common::check(
        world,
        after == before,
        &format!(
            "part 3: 1000 start_all/wait_all iterations allocated {} times",
            after - before
        ),
    );
    common::check(
        world,
        delivered,
        "part 3: start_all/wait_all delivered every iteration's data",
    );

    if collectives {
        let mut sum = world
            .allreduce_init(vec![0.0f64; 8], vec![0.0f64; 8], ReduceOp::Sum)
            .expect("part 3: allreduce_init");
        for iter in 0..10 {
            reduce(&mut sum, iter);
        }
        let before = ALLOCS.load(Ordering::Relaxed);
        let mut delivered = true;
        for iter in 10..1010 {
            delivered &= reduce(&mut sum, iter);
        }
        let after = ALLOCS.load(Ordering::Relaxed);
        common::check(
            world,
            after == before,
            &format!(
                "part 3: 1000 allreduce iterations allocated {} times",
                after - before
            ),
        );
        common::check(
            world,
            delivered,
            "part 3: the allreduce delivered every iteration's result",
        );
    }
}

fn main() {
    let mpi = Mpi::init_thread(ThreadLevel::Funneled).expect("MPI init failed");
    check(
        mpi.thread_level() == ThreadLevel::Funneled,
        "MPI provides the level asked for",
    );
    let world = mpi.world();
    let collectives = common::has_mpi4_collectives();
    if !collectives {
        println!("test_persistent_ownership: no persistent collectives, skipping the allreduce");
    }

    part1_access_while_active(&world, collectives);
    part2_drop_while_active(&world);
    part3_zero_allocations(&world, collectives);

    let mut forgotten = world
        .recv_init(vec![0u64; 4], 0, 4)
        .expect("part 4: recv_init");
    forgotten.start().expect("part 4: start");
    std::mem::forget(forgotten);
    let victim = black_box(vec![VICTIM; 4]);
    world
        .send(&[SENT; 4], 0, 4)
        .expect("part 4: send to the forgotten receive");
    common::check(
        &world,
        victim == [VICTIM; 4],
        "part 4: a receive forgotten while active leaves a later allocation alone",
    );

    drop(mpi);
    check(
        Mpi::is_finalized(),
        "the skipped finalize reports finalized",
    );
    println!("test_persistent_ownership: PASS");
}
