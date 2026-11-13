# Licensing

THME is open source under the **GNU Affero General Public License v3.0** (AGPL-3.0-only).
Epiphanes plans to offer a **commercial license** that removes the AGPL's obligations in a later
release; it is **not available yet** (see [Commercial license](#commercial-license)).

## Which license do I need?

| You want to | License |
| :--- | :--- |
| Use THME for research, study, personal projects or internal work | AGPL-3.0, free |
| Modify THME and share the modified source under the AGPL | AGPL-3.0, free |
| Run a modified THME as a network service **and publish your changes** under the AGPL | AGPL-3.0, free |
| Ship THME, or a product built on it, **without releasing your source code** | Commercial (planned) |
| Offer a hosted or managed THME service **without publishing your modifications** | Commercial (planned) |
| Embed THME in proprietary software or hardware | Commercial (planned) |

The AGPL's key condition (section 13) is that anyone who lets users interact with a modified
version over a network must offer those users the complete corresponding source code. If that
does not work for your business, the planned commercial license is meant for you.

## Commercial license

**Status: not yet offered.** THME is being built as open-source software and tools first. A
commercial release is planned once that work is complete; its version and timing are still to
be decided. Until then no commercial right, warranty or support is offered beyond the
AGPL-3.0, and the terms below describe the intended license, which will be reviewed by a lawyer
before it is offered.

A commercial license grants the right to use, modify, embed and distribute THME, and to offer
it as a service, without the AGPL's source-disclosure and copyleft obligations. Terms (seats,
deployment scope, support and updates) are agreed per customer.

To register interest in a future commercial license, hosted service or trial, contact Epiphanes through
[github.com/EdEpiphanes](https://github.com/EdEpiphanes).

## Earlier versions

THME was published under the Apache License 2.0 until 2026-11-13. Copies of the source obtained
before that date, up to and including commit `56ae520` on `main`, remain available under
Apache-2.0 to whoever received them. All later changes are AGPL-3.0-only or commercial.

## Contributions

To keep dual licensing possible, contributors agree to the contributor terms in
[CONTRIBUTING.md](CONTRIBUTING.md#licensing-of-contributions) when they open a pull request.

## Third-party components

THME depends on third-party crates under their own permissive licences (MIT, Apache-2.0, BSD,
MPL-2.0, Unicode and font licences). `cargo deny check licenses` enforces the allow-list in
[deny.toml](deny.toml). Those licences continue to apply to those components under either THME
license. Every release archive includes their full notices in `THIRD_PARTY_LICENSES.html`.

Some source in this repository is derived from third-party projects and keeps their notices:

| Files | Derived from | Licence |
| :--- | :--- | :--- |
| `src/engine/iso.rs`, `assets/iso.css` | anatomy figure kit (`kit/iso-kit.ts`, `kit/iso.css`), Copyright (c) 2026 Ryan | MIT, full text in [LICENSES/anatomy-MIT.txt](LICENSES/anatomy-MIT.txt) |

## Trademarks

"THME", "Epiphanes", the mascot and the logo are not licensed under the AGPL or the commercial
license unless stated otherwise in writing. You may refer to THME by name to describe it
accurately.

_This page explains the licensing model. It is not legal advice, and the license texts govern._
