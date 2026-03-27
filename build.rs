//! Build script for ferrompi
//!
//! This script:
//! 1. Finds the MPICH installation via pkg-config or mpicc
//! 2. Compiles the C wrapper (ferrompi.c)
//! 3. Links against the MPI library

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/ferrompi.c");
    println!("cargo:rerun-if-changed=csrc/ferrompi.h");

    // Try to find MPI configuration
    let mpi_config = find_mpi_config();

    // Build the C wrapper
    let mut build = cc::Build::new();
    build
        .file("csrc/ferrompi.c")
        .include("csrc")
        .warnings(true)
        .extra_warnings(true);

    // Add MPI include paths
    for path in &mpi_config.include_paths {
        build.include(path);
    }

    // Set optimization level
    if env::var("PROFILE").unwrap_or_default() == "release" {
        build.opt_level(3);
    }

    // Compile
    build.compile("ferrompi");

    // Link MPI library
    for path in &mpi_config.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
        // RPATH is intentionally NOT embedded. Pre-built release binaries should
        // not bake in the build machine's library paths — they are almost never
        // correct on the target machine (HPC clusters, containers, etc.). Users
        // must ensure libmpi is discoverable at runtime via LD_LIBRARY_PATH,
        // ldconfig, or their cluster's module system.
    }

    // Only link the main MPI library and essential system libraries
    // The MPI library (e.g., mpich, mpi, ompi) will handle its own dependencies (hwloc, pmix, etc.)
    // Explicitly linking transitive dependencies can cause linker errors
    // when those libraries are not in standard search paths

    // Common transitive dependencies that should be handled by the main MPI library
    // This primarily affects pkg-config detection on Ubuntu/Debian systems
    // where MPICH's pkg-config file includes all dependencies
    // MPICH 3.x/4.x and OpenMPI typically include: hwloc, pmix, ucx/ucp/ucs
    // Note: For non-standard MPI implementations, set MPI_SKIP_LIBS environment variable
    const SKIP_LIBS: &[&str] = &["hwloc", "pmix", "ucp", "ucs", "ucx", "slurm", "amdhip64"];

    for lib in &mpi_config.libs {
        if !SKIP_LIBS.contains(&lib.as_str()) {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    // Export MPI version info for Rust code
    if let Some(version) = mpi_config.version {
        println!("cargo:rustc-env=MPI_VERSION={version}");
    }
}

struct MpiConfig {
    include_paths: Vec<PathBuf>,
    link_paths: Vec<PathBuf>,
    libs: Vec<String>,
    version: Option<String>,
}

fn find_mpi_config() -> MpiConfig {
    // Strategy 1: Use MPI_PKG_CONFIG environment variable
    if let Ok(pkg_name) = env::var("MPI_PKG_CONFIG") {
        if let Ok(config) = try_pkg_config(&pkg_name) {
            eprintln!("Found MPI via MPI_PKG_CONFIG={pkg_name}");
            return config;
        }
    }

    // Strategy 2: Try common pkg-config names
    for pkg_name in &["mpich", "ompi", "mpi"] {
        if let Ok(config) = try_pkg_config(pkg_name) {
            eprintln!("Found MPI via pkg-config: {pkg_name}");
            return config;
        }
    }

    // Strategy 3: Use mpicc -show
    if let Ok(config) = try_mpicc() {
        eprintln!("Found MPI via mpicc");
        return config;
    }

    // Strategy 4: Check for Cray environment
    if let Ok(mpich_dir) = env::var("CRAY_MPICH_DIR") {
        eprintln!("Found Cray MPI at {mpich_dir}");
        return MpiConfig {
            include_paths: vec![PathBuf::from(format!("{mpich_dir}/include"))],
            link_paths: vec![PathBuf::from(format!("{mpich_dir}/lib"))],
            libs: vec!["mpi".to_string()],
            version: None,
        };
    }

    // Strategy 5: Try common installation paths
    for prefix in &["/usr", "/usr/local", "/opt/mpich", "/opt/openmpi"] {
        let include = PathBuf::from(format!("{prefix}/include"));
        let lib = PathBuf::from(format!("{prefix}/lib"));
        if include.join("mpi.h").exists() {
            eprintln!("Found MPI at {prefix}");
            return MpiConfig {
                include_paths: vec![include],
                link_paths: vec![lib],
                libs: vec!["mpi".to_string()],
                version: None,
            };
        }
    }

    panic!(
        "Could not find MPI installation. Please ensure MPICH or OpenMPI is installed and either:\n\
         - Set MPI_PKG_CONFIG to the pkg-config name (e.g., 'mpich')\n\
         - Ensure 'mpicc' is in PATH\n\
         - Set CRAY_MPICH_DIR for Cray systems"
    );
}

fn try_pkg_config(name: &str) -> Result<MpiConfig, pkg_config::Error> {
    let lib = pkg_config::Config::new()
        .cargo_metadata(false) // We'll handle linking ourselves
        .probe(name)?;

    Ok(MpiConfig {
        include_paths: lib.include_paths,
        link_paths: lib.link_paths,
        libs: lib.libs,
        version: Some(lib.version),
    })
}

fn try_mpicc() -> Result<MpiConfig, String> {
    let mpicc = env::var("MPICC").unwrap_or_else(|_| "mpicc".to_string());

    let output = Command::new(&mpicc)
        .arg("-show")
        .output()
        .map_err(|e| format!("Failed to run '{mpicc}': {e}"))?;

    if !output.status.success() {
        return Err("mpicc -show failed".to_string());
    }

    let show_output = String::from_utf8_lossy(&output.stdout);
    parse_mpicc_show(&show_output)
}

#[allow(clippy::unnecessary_wraps)]
fn parse_mpicc_show(output: &str) -> Result<MpiConfig, String> {
    let mut include_paths = Vec::new();
    let mut link_paths = Vec::new();
    let mut libs = Vec::new();

    for part in output.split_whitespace() {
        if let Some(path) = part.strip_prefix("-I") {
            include_paths.push(PathBuf::from(path));
        } else if let Some(path) = part.strip_prefix("-L") {
            link_paths.push(PathBuf::from(path));
        } else if let Some(lib) = part.strip_prefix("-l") {
            libs.push(lib.to_string());
        }
    }

    // Ensure we have at least the basic MPI library
    if libs.is_empty() {
        libs.push("mpi".to_string());
    }

    Ok(MpiConfig {
        include_paths,
        link_paths,
        libs,
        version: None,
    })
}
