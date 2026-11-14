// SPDX-License-Identifier: AGPL-3.0-only
//! Math engine (research dossier v3.1, Appendix M).
//!
//! Validated numerical primitives live here: the error type, the exact allocator,
//! number formats, orthonormal bases, ODE steppers, isometric figures and the SVG visual
//! engine, each tested against an oracle outside its own code (an analytic answer, a
//! brute-force search, or a conservation law). Every function returns [`EngineResult`];
//! none panics on bad input and none falls back to a silent default for a scientific
//! quantity.
//!
//! Later packages proposed in BACKLOG.md and waiting for the owner to place them on the
//! roadmap: transport.

pub mod alloc;
pub mod basis;
pub mod error;
pub mod iso;
pub mod minifloat;
pub mod numfmt;
pub mod ode;
pub mod transport;
pub mod viz;

pub use error::{ensure_finite, EngineError, EngineResult};
