//! Typed receive sources and tags: a receive or probe from `Source::ProcNull`
//! completes at once with no data on every MPI; a negative rank or tag is
//! rejected before any MPI call; a wildcard receive delivers the message of a
//! ring (a send to self at np 1).
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_proc_null
// mpi-test: np=1.. valgrind

use ferrompi::{Error, Mpi, Source, Tag};

mod common;

const UNTOUCHED: [i32; 4] = [-7; 4];
const RING_TAG: i32 = 5;

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();

    let mut buf = UNTOUCHED;

    let r = world.recv(&mut buf, Source::ProcNull, Tag::Any);
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "recv from ProcNull completes and leaves the buffer untouched",
    );

    let r = world
        .irecv(&mut buf, Source::ProcNull, Tag::Any)
        .and_then(|req| req.wait());
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "irecv from ProcNull completes and leaves the buffer untouched",
    );

    let r = world.probe::<i32>(Source::ProcNull, Tag::Any);
    common::check(&world, r.is_ok(), "probe from ProcNull completes");

    let r = world.recv(&mut buf, Source::Rank(-1), Tag::Any);
    common::check(
        &world,
        matches!(r, Err(Error::InvalidArgument { arg: "source", .. })),
        "recv from a negative rank is an invalid source",
    );

    let r = world.recv(&mut buf, Source::Any, Tag::Value(-5));
    common::check(
        &world,
        matches!(r, Err(Error::InvalidArgument { arg: "tag", .. })),
        "recv with a negative tag is an invalid tag",
    );

    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;
    let send = [rank; 4];
    let sent = world.isend(&send, next, RING_TAG);
    let received = world.recv(&mut buf, Source::Any, Tag::Any);
    let waited = sent.and_then(|req| req.wait());
    common::check(
        &world,
        waited.is_ok()
            && matches!(received, Ok((source, RING_TAG, 4)) if source == prev)
            && buf == [prev; 4],
        "wildcard recv delivers the ring message",
    );

    if rank == 0 {
        println!("PASS: test_proc_null");
    }
}
