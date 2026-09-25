use ferrompi::Mpi;
fn main() {
    { let _m = Mpi::init().unwrap(); }
    eprintln!("finalized; re-init");
    let r = Mpi::init();
    eprintln!("re-init is_err={}", r.is_err());
    if let Err(e) = r { eprintln!("{e}"); }
    eprintln!("SURVIVED");
}
