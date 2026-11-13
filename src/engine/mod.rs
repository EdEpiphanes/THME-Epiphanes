// SPDX-License-Identifier: AGPL-3.0-only
//! Math engine (research dossier v3.1, Appendix M).
//!
//! Validated numerical primitives live here, each tested against an oracle outside its own
//! code (an analytic answer, a brute-force search, or a conservation law). Every function
//! returns [`EngineResult`]; none panics on bad input and none falls back to a silent
//! default for a scientific quantity.
//!
//! Only the error module exists so far (work package WP1). Later packages (allocation,
//! number formats, transport, bases, ODE steppers, the SVG visual engine) are proposed in
//! BACKLOG.md and wait for the owner to place them on the roadmap.

pub mod alloc;
pub mod basis;
pub mod error;
pub mod minifloat;
pub mod numfmt;

pub use error::{ensure_finite, EngineError, EngineResult};
