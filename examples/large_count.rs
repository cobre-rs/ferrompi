//! Opt-in large-count example: moves 2^31 + 16 bytes through send/recv,
//! broadcast, allreduce and one RMA put, and checks the transferred bytes.
//!
//! On MPI >= 4, each call takes the large-count path and the transfer is
//! verified byte-for-byte. Below MPI 4, the same calls have no large-count
//! path, so each one is expected to fail with `Err(Mpi { class: Count, .. })`
//! instead.
//!
//! Needs about 4.3 GB resident on MPI >= 4 (two ~2 GiB buffers) and about
//! 6.4 GB below MPI 4, where the put check adds a ~2 GiB `Win::allocate`
//! window. It takes a few seconds in a release build, so this file carries
//! no runner directive and the default test runner never picks it up. Run it
//! manually:
//!
//! ```text
//! cargo build --release --features rma --example large_count
//! mpiexec -n 1 ./target/release/examples/large_count
//! ```

use ferrompi::{Mpi, ReduceOp, Win, WinFenceAssert};

mod common;

const N: usize = (1 << 31) + 16;

/// Prints `PASS: {name}` or `FAIL: {name}` and returns `ok`, so callers can
/// accumulate the overall result while still reporting every check.
fn report(name: &str, ok: bool) -> bool {
    if ok {
        println!("PASS: {name}");
    } else {
        println!("FAIL: {name}");
    }
    ok
}

fn main() {
    let mpi = Mpi::init().expect("MPI init failed");
    let world = mpi.world();
    assert_eq!(world.size(), 1, "large_count requires exactly 1 process");

    let mut a = vec![0u8; N];
    let mut b = vec![0u8; N];
    let mut all_ok = true;

    if common::mpi_major() >= 4 {
        for (i, x) in a.iter_mut().enumerate() {
            *x = (i % 251) as u8;
        }

        ferrompi::scope(|s| {
            let req = world.isend(s, &a, 0, 1).expect("isend failed");
            world.recv(&mut b, 0, 1).expect("recv failed");
            req.wait().expect("wait failed");
            Ok(())
        })
        .expect("scope failed");
        all_ok &= report("isend/recv of 2^31+16 bytes", b == a);

        world.broadcast(&mut b, 0).expect("broadcast failed");
        all_ok &= report("broadcast of 2^31+16 bytes", b == a);

        b.fill(0);
        world
            .allreduce(&a, &mut b, ReduceOp::Max)
            .expect("allreduce failed");
        all_ok &= report("allreduce of 2^31+16 bytes", b == a);

        b.fill(0);
        {
            let win = Win::create(&world, &mut b).expect("Win::create failed");
            win.fence(WinFenceAssert::default())
                .expect("opening fence failed");
            win.put(&a, 0, 0, N as i64).expect("put failed");
            win.fence(WinFenceAssert::default())
                .expect("closing fence failed");
            drop(win);
        }
        all_ok &= report("put of 2^31+16 bytes", b == a);
    } else {
        all_ok &= report(
            "isend of 2^31+16 bytes below MPI 4 returns Count",
            common::is_count(&ferrompi::scope(|s| world.isend(s, &a, 0, 1).map(|_| ()))),
        );

        all_ok &= report(
            "broadcast of 2^31+16 bytes below MPI 4 returns Count",
            common::is_count(&world.broadcast(&mut b, 0)),
        );

        all_ok &= report(
            "allreduce of 2^31+16 bytes below MPI 4 returns Count",
            common::is_count(&world.allreduce(&a, &mut b, ReduceOp::Max)),
        );

        let win = Win::<u8>::allocate(&world, N).expect("Win::allocate failed");
        win.fence(WinFenceAssert::default())
            .expect("opening fence failed");
        let put_ok = common::is_count(&win.put(&a, 0, 0, N as i64));
        win.fence(WinFenceAssert::default())
            .expect("closing fence failed");
        all_ok &= report("put of 2^31+16 bytes below MPI 4 returns Count", put_ok);
    }

    if !all_ok {
        std::process::exit(1);
    }
}
