use std::hint::black_box;
use std::mem::MaybeUninit;
struct PR { handle: i64, _active: bool }
const CAP: usize = 64;
#[inline(never)] fn zeroed(reqs: &[PR]) -> i64 {
    let len = reqs.len(); let mut buf = [0i64; CAP];
    for (s, r) in buf[..len].iter_mut().zip(reqs) { *s = r.handle; }
    black_box(&mut buf[..len]); buf[0]
}
#[inline(never)] fn uninit(reqs: &[PR]) -> i64 {
    let len = reqs.len(); let mut buf = [MaybeUninit::<i64>::uninit(); CAP];
    for (s, r) in buf[..len].iter_mut().zip(reqs) { s.write(r.handle); }
    let sl = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<i64>(), len) };
    black_box(sl); unsafe { buf[0].assume_init() }
}
fn main() {
    let reqs: Vec<PR> = (0..2).map(|i| PR { handle: i, _active: false }).collect();
    for _ in 0..3 {
        let n = 50_000_000;
        let t = std::time::Instant::now(); for _ in 0..n { black_box(zeroed(black_box(&reqs))); } let a = t.elapsed().as_nanos() as f64 / n as f64;
        let t = std::time::Instant::now(); for _ in 0..n { black_box(uninit(black_box(&reqs))); } let b = t.elapsed().as_nanos() as f64 / n as f64;
        println!("with_handles(2 reqs): zeroed [0i64;64] {a:.2} ns | MaybeUninit {b:.2} ns | saving {:.2} ns per call", a - b);
    }
}
