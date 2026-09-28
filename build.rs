//! Build script for ferrompi
//!
//! This script:
//! 1. Finds the MPICH installation via pkg-config or mpicc
//! 2. Compiles the C wrapper (ferrompi.c)
//! 3. Links against the MPI library

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/ferrompi.c");
    println!("cargo:rerun-if-changed=csrc/ferrompi.h");
    for var in ["MPI_PKG_CONFIG", "MPICC", "CRAY_MPICH_DIR", "PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

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
    for (name, value) in &mpi_config.defines {
        build.define(name, value.as_deref());
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
    defines: Vec<(String, Option<String>)>,
    version: Option<String>,
}

fn find_mpi_config() -> MpiConfig {
    // An explicitly set variable wins and never falls through to auto-detection.
    if let Some(name) = explicit_var("MPI_PKG_CONFIG") {
        let config = try_pkg_config(&name)
            .unwrap_or_else(|e| panic!("MPI_PKG_CONFIG={name}: pkg-config probe failed: {e}"));
        eprintln!("Found MPI via MPI_PKG_CONFIG={name}");
        return config;
    }
    if let Some(wrapper) = explicit_var("MPICC") {
        let config = try_mpicc(&wrapper).unwrap_or_else(|e| panic!("MPICC={wrapper}: {e}"));
        eprintln!("Found MPI via MPICC={wrapper}");
        return config;
    }
    if let Some(dir) = explicit_var("CRAY_MPICH_DIR") {
        let config =
            try_cray(Path::new(&dir)).unwrap_or_else(|e| panic!("CRAY_MPICH_DIR={dir}: {e}"));
        eprintln!("Found MPI via CRAY_MPICH_DIR={dir}");
        return config;
    }

    for pkg_name in &["mpich", "ompi", "mpi"] {
        if let Ok(config) = try_pkg_config(pkg_name) {
            eprintln!("Found MPI via pkg-config: {pkg_name}");
            return config;
        }
    }

    if let Ok(config) = try_mpicc("mpicc") {
        eprintln!("Found MPI via mpicc");
        return config;
    }

    let multiarch = multiarch_lib_dir();
    for prefix in &["/usr", "/usr/local", "/opt/mpich", "/opt/openmpi"] {
        if let Some(config) = try_prefix(Path::new(prefix), &["lib", "lib64", &multiarch], &["mpi"])
        {
            eprintln!("Found MPI at {prefix}");
            return config;
        }
    }

    panic!(
        "Could not find MPI. Install MPICH or Open MPI so that pkg-config or mpicc finds it, \
         or set one of:\n \
         - MPI_PKG_CONFIG to the pkg-config package name (e.g., 'mpich')\n \
         - MPICC to the MPI compiler wrapper (e.g., '/opt/mpich/bin/mpicc')\n \
         - CRAY_MPICH_DIR to the Cray MPICH installation directory"
    );
}

/// The value of `name` when it is set and non-empty.
fn explicit_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

/// Resolves a Cray MPICH installation: its `lib/pkgconfig/mpich.pc` when present,
/// else `include/mpi.h` with `libmpich` or `libmpi` in `lib` or `lib64`.
fn try_cray(dir: &Path) -> Result<MpiConfig, String> {
    let pc = dir.join("lib/pkgconfig/mpich.pc");
    if pc.exists() {
        let pc = pc
            .to_str()
            .ok_or_else(|| format!("{} is not valid UTF-8", pc.display()))?;
        return try_pkg_config(pc).map_err(|e| format!("pkg-config probe of {pc} failed: {e}"));
    }
    try_prefix(dir, &["lib", "lib64"], &["mpich", "mpi"])
        .ok_or_else(|| "found no include/mpi.h with libmpich or libmpi in lib or lib64".to_string())
}

/// `prefix/include` with the first of `libs` found as `lib<name>.so` or `lib<name>.a`
/// in one of `lib_dirs`, trying every directory for a name before the next name.
/// `None` without `prefix/include/mpi.h`.
fn try_prefix(prefix: &Path, lib_dirs: &[&str], libs: &[&str]) -> Option<MpiConfig> {
    let include = prefix.join("include");
    if !include.join("mpi.h").exists() {
        return None;
    }
    for lib in libs {
        for lib_dir in lib_dirs {
            let dir = prefix.join(lib_dir);
            if ["so", "a"]
                .iter()
                .any(|ext| dir.join(format!("lib{lib}.{ext}")).exists())
            {
                return Some(MpiConfig {
                    include_paths: vec![include],
                    link_paths: vec![dir],
                    libs: vec![(*lib).to_string()],
                    defines: Vec::new(),
                    version: None,
                });
            }
        }
    }
    None
}

/// `lib/<arch>-<os>-<env>`, the Debian multiarch directory of the target triple.
fn multiarch_lib_dir() -> String {
    let target = env::var("TARGET").expect("cargo sets TARGET for build scripts");
    match target.split('-').collect::<Vec<_>>().as_slice() {
        [arch, _vendor, os, abi] => format!("lib/{arch}-{os}-{abi}"),
        _ => format!("lib/{target}"),
    }
}

fn try_pkg_config(name: &str) -> Result<MpiConfig, pkg_config::Error> {
    let lib = pkg_config::Config::new()
        .cargo_metadata(false) // We'll handle linking ourselves
        .probe(name)?;

    Ok(MpiConfig {
        include_paths: lib.include_paths,
        link_paths: lib.link_paths,
        libs: lib.libs,
        defines: lib.defines.into_iter().collect(),
        version: Some(lib.version),
    })
}

/// Runs the compiler wrapper with `-show` (MPICH, Intel MPI) and, if that exits
/// non-zero, with `--showme` (Open MPI), and parses the first success.
fn try_mpicc(mpicc: &str) -> Result<MpiConfig, String> {
    for option in ["-show", "--showme"] {
        let output = Command::new(mpicc)
            .arg(option)
            .output()
            .map_err(|e| format!("failed to run '{mpicc}': {e}"))?;
        if output.status.success() {
            return Ok(parse_wrapper_flags(&String::from_utf8_lossy(
                &output.stdout,
            )));
        }
    }
    Err(format!(
        "both '{mpicc} -show' and '{mpicc} --showme' failed"
    ))
}

/// The joined `-I`, `-L`, `-l` and `-DNAME[=VAL]` flags of a wrapper's output,
/// with surrounding quotes stripped from each value. Other tokens are ignored.
/// Without a `-l`, links `mpi`.
fn parse_wrapper_flags(output: &str) -> MpiConfig {
    let mut config = MpiConfig {
        include_paths: Vec::new(),
        link_paths: Vec::new(),
        libs: Vec::new(),
        defines: Vec::new(),
        version: None,
    };

    for token in output.split_whitespace() {
        let (Some(flag), Some(value)) = (token.get(..2), token.get(2..)) else {
            continue;
        };
        let value = value.trim_matches(|c| c == '"' || c == '\'');
        match flag {
            "-I" => config.include_paths.push(PathBuf::from(value)),
            "-L" => config.link_paths.push(PathBuf::from(value)),
            "-l" => config.libs.push(value.to_string()),
            "-D" => config.defines.push(match value.split_once('=') {
                Some((name, val)) => (name.to_string(), Some(val.to_string())),
                None => (value.to_string(), None),
            }),
            _ => {}
        }
    }

    if config.libs.is_empty() {
        config.libs.push("mpi".to_string());
    }
    config
}
