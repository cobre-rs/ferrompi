# Contributing to FerroMPI

Thank you for your interest in contributing to FerroMPI! This guide will help you get started.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ferrompi.git`
3. Create a branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Run tests locally (see below)
6. Push and create a Pull Request

## Development Setup

### Requirements

- **Rust 1.74+** (MSRV - Minimum Supported Rust Version)
- **MPI 4.0+** (MPICH 4.0+ or OpenMPI 5.0+)
- **pkg-config**

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install mpich libmpich-dev pkg-config
```

#### macOS

```bash
brew install mpich pkg-config
```

### Building

```bash
# Build library
cargo build

# Build examples
cargo build --examples --release
```

## Before Submitting a PR

### 1. Format Code

```bash
cargo fmt --all
```

### 2. Run Clippy

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

**Note:** The library code passes `clippy::pedantic` checks. Running with pedantic lints is encouraged but not required:

```bash
# Optional: stricter checks
cargo clippy --lib -- -D warnings -W clippy::pedantic
```

### 3. Run Tests

```bash
# Unit tests
cargo test --lib

# Doc tests
cargo test --doc

# All tests
cargo test --all-features
```

### 4. Test Examples with MPI

```bash
cargo build --examples --release
for ex in hello_world ring allreduce nonblocking persistent_bcast pi_monte_carlo; do
  mpiexec -n 4 ./target/release/examples/$ex
done
```

### 5. Run Benchmarks (if applicable)

```bash
cargo bench
```

## CI/CD

All pull requests automatically run:

- ✅ **Tests** on Ubuntu with stable Rust (MPICH and OpenMPI matrix)
- ✅ **Clippy** lints
- ✅ **Format checking** with rustfmt
- ✅ **Security audit** for vulnerabilities

Your PR will be automatically labeled based on changed files.

See the workflow files under `.github/workflows/` for details.

## Code Style

### General Guidelines

- Follow Rust's [API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Write idiomatic Rust code
- Prefer zero-cost abstractions
- Document all public APIs
- Add examples to documentation

### Documentation

All public items must be documented:

````rust
/// Computes the sum of two numbers.
///
/// # Arguments
///
/// * `a` - First number
/// * `b` - Second number
///
/// # Returns
///
/// The sum of `a` and `b`.
///
/// # Examples
///
/// ```
/// use ferrompi::add;
/// assert_eq!(add(2, 3), 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
````

### Error Handling

Use `Result` with descriptive errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrompiError {
    #[error("Operation failed: {0}")]
    OperationFailed(String),
}

pub fn my_function() -> Result<(), FerrompiError> {
    // implementation
    Ok(())
}
```

### Testing

Write comprehensive tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }

    #[test]
    fn test_edge_cases() {
        // Test edge cases
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_error_handling() {
        // Test error conditions
    }
}
```

## Performance Considerations

FerroMPI is an HPC library - performance matters!

### Do's ✅

- Pre-allocate buffers when size is known
- Use iterators over indexing
- Minimize allocations in hot paths
- Use `#[inline]` for small, frequently-called functions
- Profile before optimizing

### Don'ts ❌

- Don't use `clone()` unnecessarily
- Don't allocate in loops
- Don't use `unwrap()` or `expect()` in library code
- Don't ignore performance implications

## Adding New Features

### 1. Discuss First

For significant changes:

- Open an issue first to discuss the approach
- Get feedback before implementing
- Ensure it aligns with project goals

### 2. Write Tests

All new features need:

- Unit tests
- Integration tests (if applicable)
- Examples demonstrating usage
- Benchmarks (for performance-critical code)

### 3. Update Documentation

- Add documentation to new public APIs
- Update README.md if user-facing
- Add examples
- Update CHANGELOG.md

### 4. Maintain Compatibility

- Don't break existing APIs without discussion
- Follow semantic versioning
- Test MSRV compatibility

## Reporting Issues

### Bug Reports

Include:

- Rust version: `rustc --version`
- MPI version: `mpiexec --version`
- Operating system
- Minimal reproduction case
- Error messages and backtraces

### Feature Requests

Include:

- Use case description
- Example API (if applicable)
- Why existing APIs don't suffice
- Performance implications

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

## Questions?

- Open an issue for questions
- Check existing issues and PRs

## Recognition

All contributors will be recognized in release notes and the project README.

Thank you for contributing to FerroMPI! 🚀
