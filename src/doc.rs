//! Long-form documentation for ferrompi.
//!
//! This module embeds the Markdown files from the `docs/` directory directly
//! into rustdoc via `#[doc = include_str!(...)]`. Each sub-module corresponds
//! to one documentation artifact. The same content is available as plain
//! Markdown on GitHub at `docs/` and `docs/adr/`; see [`docs/README.md`] for
//! a navigable index with relative-link support.
//!
//! [`docs/README.md`]: https://github.com/cobre-rs/ferrompi/blob/main/docs/README.md

/// Architecture overview for ferrompi contributors.
///
/// Describes the six-layer stack, handle tables, thread-safety model, C layer
/// scope, FFI/ABI invariants, the generic `MpiDatatype` trait family, and the
/// error handling model.
#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}

/// Migration guide from rsmpi to ferrompi.
///
/// Covers the quick-comparison table, a function-for-function API mapping,
/// migration cookbook examples, unsupported features, and API ergonomic
/// differences.
#[doc = include_str!("../docs/migrating-from-rsmpi.md")]
pub mod migrating_from_rsmpi {}

/// MPI implementation compatibility matrix.
///
/// Documents which features are available on MPICH 3.x/4.x, Open MPI 4/5,
/// Intel MPI, and Cray MPI, including known issues and how to report new
/// compatibility data.
#[doc = include_str!("../docs/mpi-compatibility.md")]
pub mod mpi_compatibility {}

#[doc = include_str!("../docs/adr/0001-why-c-wrapper.md")]
pub mod adr_0001_why_c_wrapper {}

#[doc = include_str!("../docs/adr/0002-handle-tables.md")]
pub mod adr_0002_handle_tables {}

#[doc = include_str!("../docs/adr/0003-generic-mpi-datatype.md")]
pub mod adr_0003_generic_mpi_datatype {}

#[doc = include_str!("../docs/adr/0004-persistent-collective-approach.md")]
pub mod adr_0004_persistent_collective_approach {}

#[doc = include_str!("../docs/adr/0005-mpi-op-create.md")]
pub mod adr_0005_mpi_op_create {}

#[doc = include_str!("../docs/adr/0006-mpi5-abi-direction.md")]
pub mod adr_0006_mpi5_abi_direction {}
