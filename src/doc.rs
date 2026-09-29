//! Long-form documentation for ferrompi.
//!
//! This module embeds the Markdown files from the `docs/` directory directly
//! into rustdoc via `#[doc = include_str!(...)]`. Each sub-module corresponds
//! to one documentation artifact. The same content is available as plain
//! Markdown on GitHub at `docs/` and `docs/adr/`; see [`docs/README.md`] for
//! a navigable index with relative-link support.
//!
//! [`docs/README.md`]: https://github.com/cobre-rs/ferrompi/blob/main/docs/README.md

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}

// A leading empty `#[doc]` attribute keeps this module's rendered doc
// comment from starting at byte 0 of the included file. Without it,
// clippy's `doc_overindented_list_items` misparses the six-space-aligned
// continuation lines under the "Migration Checklist" task list in
// docs/migrating-from-rsmpi.md as over-indented and fails `-D warnings`
// (verified: removing this line alone reproduces all 14 false positives;
// the architecture and mpi-compatibility includes need no such attribute).
#[doc = ""]
#[doc = include_str!("../docs/migrating-from-rsmpi.md")]
pub mod migrating_from_rsmpi {}

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
