//! Integration test for MPI lifecycle functions.
//!
//! Exercises Mpi::init, version, library_version, wtime, is_initialized,
//! is_finalized, thread_level, and world, and the refusal of the constructors
//! that take no handle before Mpi::init.
//!
//! Run with: mpiexec -n 2 ./target/debug/examples/test_lifecycle
// mpi-test: np=2

use ferrompi::{CustomDatatype, Error, Mpi, Result, StructField, ThreadLevel, UserOp};

/// Panics unless `result` is the refusal of a call made before `Mpi::init`.
fn refused_before_init<T>(what: &str, result: Result<T>) {
    match result {
        Err(Error::InvalidState {
            reason: "MPI is not initialized",
            ..
        }) => {}
        Err(e) => panic!("{what} before init returned {e:?}"),
        Ok(_) => panic!("{what} before init returned Ok"),
    }
}

fn main() {
    // Before Mpi::init, the constructors that take no handle return an error
    // without calling MPI. The op table holds 16 slots, so 20 refusals also
    // show that a refused constructor leaves no slot taken.
    assert!(
        !Mpi::is_initialized(),
        "is_initialized should be false before init"
    );
    for _ in 0..20 {
        refused_before_init("UserOp::new", UserOp::<f64>::new(|_, _| {}));
    }
    refused_before_init(
        "UserOp::new_noncommutative",
        UserOp::<f64>::new_noncommutative(|_, _| {}),
    );
    refused_before_init(
        "CustomDatatype::contiguous",
        CustomDatatype::contiguous::<f64>(2),
    );
    refused_before_init(
        "CustomDatatype::vector",
        CustomDatatype::vector::<f64>(2, 1, 2),
    );
    refused_before_init(
        "CustomDatatype::create_struct",
        CustomDatatype::create_struct(&[StructField::new::<f64>(1, 0)]),
    );
    assert!(
        !Mpi::is_initialized(),
        "a refused constructor should not initialize MPI"
    );
    println!("PASS: constructors are refused before init");

    // Test init_thread (which init() delegates to)
    let mpi = Mpi::init().expect("MPI init failed");

    // Test is_initialized after init
    assert!(
        Mpi::is_initialized(),
        "is_initialized should be true after init"
    );
    println!("PASS: is_initialized");

    // Test is_finalized before drop
    assert!(
        !Mpi::is_finalized(),
        "is_finalized should be false before drop"
    );
    println!("PASS: is_finalized (false before drop)");

    // Test thread_level
    let level = mpi.thread_level();
    assert!(
        level >= ThreadLevel::Single,
        "thread_level should be >= Single"
    );
    println!("PASS: thread_level = {:?}", level);

    // Test version
    let version = Mpi::version().expect("version() failed");
    assert!(!version.is_empty(), "version string should not be empty");
    println!("PASS: version = {}", version);

    // Test library_version
    let library = Mpi::library_version().expect("library_version failed");
    assert!(!library.is_empty(), "library_version should not be empty");
    assert!(
        !library.contains('\0'),
        "library_version contains a NUL: {library:?}"
    );
    println!("PASS: library_version has no NUL");

    // Test wtime
    let t1 = mpi.wtime();
    assert!(t1 >= 0.0, "wtime should return non-negative value");
    let t2 = mpi.wtime();
    assert!(t2 >= t1, "wtime should be monotonic");
    println!("PASS: wtime = {}", t1);

    // Test world
    let world = mpi.world();
    let rank = world.rank();
    let size = world.size();
    assert!(rank >= 0 && rank < size, "rank should be in [0, size)");
    assert!(size >= 1, "size should be >= 1");
    println!("PASS: world rank={} size={}", rank, size);

    // Test processor_name
    let name = world.processor_name().expect("processor_name failed");
    assert!(!name.is_empty(), "processor_name should not be empty");
    assert!(
        !name.contains('\0'),
        "processor_name contains a NUL: {name:?}"
    );
    println!("PASS: processor_name = {}", name);

    // Mpi drops here, which calls MPI_Finalize
    drop(mpi);

    if rank == 0 {
        println!("\n========================================");
        println!("All lifecycle tests passed!");
        println!("========================================");
    }
}
