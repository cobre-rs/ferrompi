use ferrompi::{Mpi, Request};

fn main() -> ferrompi::Result<()> {
    let mpi = Mpi::init()?;
    let world = mpi.world();
    let me = world.rank();

    // Phase 1: two self-requests, completed individually via test().
    let sbuf = [1i32];
    let mut rbuf = [0i32];
    let mut old = vec![world.irecv(&mut rbuf, me, 0)?, world.isend(&sbuf, me, 0)?];
    println!("old handles: {} {}", old[0].raw_handle(), old[1].raw_handle());
    while !old[0].test()? {}
    while !old[1].test()? {}
    println!("old completed: {} {}", old[0].is_completed(), old[1].is_completed());

    // Phase 2: new, unrelated requests. They reuse the freed slots.
    let sbuf2 = [7i32];
    let mut rbuf2 = [0i32];
    let new_recv = world.irecv(&mut rbuf2, me, 5)?;
    let new_send = world.isend(&sbuf2, me, 5)?;
    println!("new handles: {} {}", new_recv.raw_handle(), new_send.raw_handle());

    // Phase 3: user waits on the OLD (already-completed) slice.
    let r = Request::wait_all(&mut old);
    println!("wait_all(old) -> {:?}", r);

    // Phase 4: the NEW requests have been completed & freed behind their owners' back.
    println!("new_recv.wait() -> {:?}", new_recv.wait());
    println!("new_send.wait() -> {:?}", new_send.wait());

    // wait_any loop idiom (without removing completed entries)
    let mut rb3 = [0i32];
    let sb3 = [3i32];
    let mut v = vec![world.irecv(&mut rb3, me, 9)?, world.isend(&sb3, me, 9)?];
    let a = Request::wait_any(&mut v);
    println!("wait_any #1 -> {:?}", a);
    let b = Request::wait_any(&mut v);
    println!("wait_any #2 (completed entry left in slice) -> {:?}", b);
    Ok(())
}
