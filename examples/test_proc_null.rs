//! Typed point-to-point endpoints: a send, receive, probe or persistent request
//! with `Source::ProcNull` completes at once with no data on every MPI, and a
//! receive, probe or `sendrecv` from it reports `Source::ProcNull`, `Tag::Any`
//! and a count of 0; a negative rank or tag, or `Source::Any` as a destination,
//! is rejected before any MPI call; a wildcard receive delivers the message of
//! a ring (a send to self at np 1).
//!
//! Run with: mpiexec -n 4 ./target/debug/examples/test_proc_null
// mpi-test: np=1.. valgrind

use ferrompi::{CustomDatatype, DatatypeTag, Error, Mpi, Source, Status, StructField, Tag};

mod common;

const UNTOUCHED: [i32; 4] = [-7; 4];
const RING_TAG: i32 = 5;
const SENDRECV_TAG: i32 = 6;

fn is_proc_null_status(st: &Status) -> bool {
    st.source == Source::ProcNull && st.tag == Tag::Any && st.count == Some(0) && st.error.is_none()
}

fn is_empty_status(st: &Status) -> bool {
    st.source == Source::Any && st.tag == Tag::Any && st.count == Some(0) && st.error.is_none()
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    let mut buf = UNTOUCHED;

    let r = world.recv(&mut buf, Source::ProcNull, Tag::Any);
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "recv from ProcNull completes and leaves the buffer untouched",
    );
    common::check(
        &world,
        r.as_ref().is_ok_and(is_proc_null_status),
        "recv from ProcNull reports the ProcNull status",
    );

    let r = ferrompi::scope(|s| {
        Ok(world
            .irecv(s, &mut buf, Source::ProcNull, Tag::Any)
            .and_then(|req| req.wait()))
    })
    .expect("scope failed");
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "irecv from ProcNull completes and leaves the buffer untouched",
    );
    common::check(
        &world,
        r.as_ref().is_ok_and(is_proc_null_status),
        "irecv from ProcNull waits to the ProcNull status",
    );

    let r = world.probe::<i32>(Source::ProcNull, Tag::Any);
    common::check(&world, r.is_ok(), "probe from ProcNull completes");
    common::check(
        &world,
        r.as_ref().is_ok_and(is_proc_null_status),
        "probe from ProcNull reports the ProcNull status",
    );

    let r = world.iprobe::<i32>(Source::ProcNull, Tag::Any);
    common::check(
        &world,
        matches!(&r, Ok(Some(st)) if is_proc_null_status(st)),
        "iprobe from ProcNull reports a match with the ProcNull status",
    );

    let mut ring = [0i32; 4];
    let (r, delivered) = ferrompi::scope(|s| {
        let pending = world.irecv(s, &mut ring, prev, SENDRECV_TAG);
        let r = world.sendrecv(
            &[rank; 4],
            next,
            SENDRECV_TAG,
            &mut buf,
            Source::ProcNull,
            Tag::Any,
        );
        let delivered = pending.and_then(|req| req.wait());
        Ok((r, delivered))
    })
    .expect("scope failed");
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED && delivered.is_ok() && ring == [prev; 4],
        "sendrecv from ProcNull leaves the receive buffer untouched and still sends",
    );
    common::check(
        &world,
        r.as_ref().is_ok_and(is_proc_null_status),
        "sendrecv from ProcNull reports the ProcNull receive status",
    );

    let mut persistent = world
        .recv_init(&mut buf, Source::ProcNull, Tag::Any)
        .expect("recv_init failed");
    let mut rounds_ok = true;
    for _ in 0..3 {
        rounds_ok &= persistent.start().and_then(|()| persistent.wait()).is_ok();
    }
    common::check(
        &world,
        rounds_ok && buf == UNTOUCHED,
        "recv_init from ProcNull completes on every start and leaves the buffer untouched",
    );

    let one_i32 = CustomDatatype::create_struct(&[StructField {
        blocklength: 1,
        displacement: 0,
        basetype: DatatypeTag::I32,
    }])
    .expect("create_struct failed");
    let r = world.recv_custom(&mut buf, &one_i32, Source::ProcNull, Tag::Any);
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "recv_custom from ProcNull completes and leaves the buffer untouched",
    );

    let send = [rank; 4];

    let r = world.send(&send, Source::ProcNull, RING_TAG);
    common::check(&world, r.is_ok(), "send to ProcNull completes");

    let r = ferrompi::scope(|s| {
        Ok(world
            .isend(s, &send, Source::ProcNull, RING_TAG)
            .and_then(|req| req.wait()))
    })
    .expect("scope failed");
    common::check(&world, r.is_ok(), "isend to ProcNull completes");
    common::check(
        &world,
        r.as_ref().is_ok_and(is_empty_status),
        "isend to ProcNull waits to the empty status",
    );

    mpi.buffer_attach(vec![0u8; 64 * 1024].into_boxed_slice())
        .expect("buffer_attach failed");
    let requests = [
        world.send_init(&send, Source::ProcNull, RING_TAG),
        world.bsend_init(&send, Source::ProcNull, RING_TAG),
        world.rsend_init(&send, Source::ProcNull, RING_TAG),
        world.ssend_init(&send, Source::ProcNull, RING_TAG),
    ];
    let mut sends_ok = true;
    for request in requests {
        let mut request = request.expect("persistent send init failed");
        for _ in 0..3 {
            sends_ok &= request.start().and_then(|()| request.wait()).is_ok();
        }
    }
    common::check(
        &world,
        sends_ok,
        "persistent sends to ProcNull complete on every start",
    );
    mpi.buffer_detach().expect("buffer_detach failed");

    let r = world.sendrecv(
        &send,
        Source::ProcNull,
        SENDRECV_TAG,
        &mut buf,
        Source::ProcNull,
        Tag::Any,
    );
    common::check(
        &world,
        r.is_ok() && buf == UNTOUCHED,
        "sendrecv with both ends ProcNull completes and leaves the buffer untouched",
    );

    ring = [0; 4];
    let (r, waited) = ferrompi::scope(|s| {
        let sent = world.isend(s, &send, next, RING_TAG);
        let r = world.sendrecv(
            &send,
            Source::ProcNull,
            SENDRECV_TAG,
            &mut ring,
            prev,
            RING_TAG,
        );
        let waited = sent.and_then(|req| req.wait());
        Ok((r, waited))
    })
    .expect("scope failed");
    common::check(
        &world,
        waited.is_ok()
            && matches!(
                r,
                Ok(st) if st.source == Source::Rank(prev)
                    && st.tag == Tag::Value(RING_TAG)
                    && st.count == Some(4)
            )
            && ring == [prev; 4],
        "sendrecv to ProcNull still receives the ring message",
    );

    let r = world.send(&send, -1, RING_TAG);
    common::check(
        &world,
        matches!(r, Err(Error::InvalidArgument { arg: "dest", .. })),
        "send to a negative rank is an invalid destination",
    );

    let r = world.send(&send, Source::Any, RING_TAG);
    common::check(
        &world,
        matches!(r, Err(Error::InvalidArgument { arg: "dest", .. })),
        "send to Any is an invalid destination",
    );

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

    let (received, waited) = ferrompi::scope(|s| {
        let sent = world.isend(s, &send, next, RING_TAG);
        let received = world.recv(&mut buf, Source::Any, Tag::Any);
        let waited = sent.and_then(|req| req.wait());
        Ok((received, waited))
    })
    .expect("scope failed");
    common::check(
        &world,
        waited.is_ok()
            && matches!(
                received,
                Ok(st) if st.source == Source::Rank(prev)
                    && st.tag == Tag::Value(RING_TAG)
                    && st.count == Some(4)
            )
            && buf == [prev; 4],
        "wildcard recv delivers the ring message",
    );

    if rank == 0 {
        println!("PASS: test_proc_null");
    }
}
