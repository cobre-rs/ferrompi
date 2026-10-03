//! A nonblocking scope holding 64 requests or fewer allocates nothing.
//!
//! A counting global allocator wraps `System` and counts the allocations Rust
//! code makes; MPI's own `malloc`s are not counted. After 10 warm-up
//! iterations, 1000 iterations each run one scope holding 64 requests (32
//! self receives into a stack array, then 32 self sends), complete them with
//! `Request::wait_all`, and leave the counter unchanged. One scope holding 65
//! requests then spills its registry to the heap and raises the counter, which
//! marks the 64-slot boundary.
//!
//! Run with: mpiexec -n 1 ./target/debug/examples/test_scope_alloc
// mpi-test: np=1

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrompi::{Communicator, Mpi, Request};

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

/// One scope with `N` requests: 32 self receives, 32 self sends, then an
/// `ibarrier` for each request beyond 64.
fn round<const N: usize>(world: &Communicator, recv: &mut [f64; 32], send: &[f64; 32]) {
    ferrompi::scope(|s| {
        let mut recv_chunks = recv.chunks_mut(1);
        let mut send_chunks = send.chunks(1);
        let mut reqs: [Request<'_>; N] = std::array::from_fn(|i| match i {
            0..32 => world
                .irecv(s, recv_chunks.next().unwrap(), 0, i as i32)
                .unwrap(),
            32..64 => world
                .isend(s, send_chunks.next().unwrap(), 0, (i - 32) as i32)
                .unwrap(),
            _ => world.ibarrier(s).unwrap(),
        });
        Request::wait_all(&mut reqs).unwrap();
        Ok(())
    })
    .unwrap();
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let mut recv = [0.0f64; 32];
    let send = [1.0f64; 32];

    for _ in 0..10 {
        round::<64>(&world, &mut recv, &send);
    }

    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..1000 {
        round::<64>(&world, &mut recv, &send);
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    common::check(
        &world,
        after == before,
        &format!(
            "1000 scopes of 64 requests allocated {} times",
            after - before
        ),
    );

    recv.fill(0.0);
    let before = ALLOCS.load(Ordering::Relaxed);
    round::<65>(&world, &mut recv, &send);
    let after = ALLOCS.load(Ordering::Relaxed);
    common::check(
        &world,
        after > before,
        "a scope of 65 requests spills to the heap",
    );
    common::check(&world, recv == send, "the 65 requests completed their data");

    println!("test_scope_alloc: PASS");
}
