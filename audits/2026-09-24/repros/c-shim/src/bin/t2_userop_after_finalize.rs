use ferrompi::{Mpi, UserOp};
struct App { _mpi: Mpi, _op: UserOp<f64> }
fn main() {
    let mpi = Mpi::init().unwrap();
    let op: UserOp<f64> = UserOp::new(|a: &[f64], b: &mut [f64]| { for (x,y) in a.iter().zip(b.iter_mut()) { *y += *x; } }).unwrap();
    // Struct fields drop in declaration order: Mpi (finalize) first, then UserOp.
    let app = App { _mpi: mpi, _op: op };
    eprintln!("dropping app (Mpi field first)");
    drop(app);
    eprintln!("SURVIVED");
}
